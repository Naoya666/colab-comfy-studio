"""Model source resolution. Credentials never appear in persisted URLs.

Network requests pin a public DNS result while retaining TLS SNI/Host, including
on redirects. No environment proxy, automatic redirect, or cross-host auth.
"""
from __future__ import annotations

import hashlib
import io
import ipaddress
import json
import logging
import os
import pickletools
import re
import socket
import struct
import zipfile
from contextlib import closing, contextmanager
from contextvars import ContextVar
from pathlib import Path, PurePosixPath
from urllib.parse import parse_qs, quote, unquote, urlencode, urljoin, urlsplit, urlunsplit

import httpx

SHA = re.compile(r"^[a-fA-F0-9]{64}$")
SAFE_EXTENSIONS = {".safetensors", ".gguf", ".onnx"}
CONTROLLED_UPSCALE_EXTENSIONS = {".pt", ".pth"}
PYTORCH_LEGACY_MAGIC = 0x1950A86A20F9469CFC6C
MAX_PICKLE_ENTRY_BYTES = 64 * 1024 * 1024
TRUSTED_UPSCALE_GITHUB_OWNERS = frozenset({
    "xinntao",
    "jingyunliang",
    "phhofm",
    "helaman",
    "chainner-org",
    "cszn",
    "k4yt3x",
    "umzi2",
    "sniklaus",
})
SAFE_TORCH_PICKLE_GLOBALS = frozenset({
    ("collections", "OrderedDict"),
    ("torch", "Size"),
    ("torch._utils", "_rebuild_tensor"),
    ("torch._utils", "_rebuild_tensor_v2"),
    ("torch._utils", "_rebuild_parameter"),
    ("torch._utils", "_rebuild_device_tensor_from_numpy"),
    ("torch", "FloatStorage"),
    ("torch", "DoubleStorage"),
    ("torch", "HalfStorage"),
    ("torch", "BFloat16Storage"),
    ("torch", "IntStorage"),
    ("torch", "LongStorage"),
    ("torch", "ShortStorage"),
    ("torch", "CharStorage"),
    ("torch", "ByteStorage"),
    ("torch", "BoolStorage"),
    ("torch", "ComplexFloatStorage"),
    ("torch", "ComplexDoubleStorage"),
    ("torch", "UntypedStorage"),
    ("torch", "QUInt8Storage"),
    ("torch", "QInt8Storage"),
    ("torch", "QInt32Storage"),
    ("_codecs", "encode"),
    ("builtins", "set"),
    ("builtins", "frozenset"),
    ("builtins", "slice"),
    ("builtins", "bytearray"),
})
DISALLOWED_PICKLE_OPCODES = frozenset({"INST", "OBJ", "EXT1", "EXT2", "EXT4", "PERSID"})
HF_REPO_TYPE_ALIASES = {
    "model": "models",
    "models": "models",
    "dataset": "datasets",
    "datasets": "datasets",
    "space": "spaces",
    "spaces": "spaces",
}


_SOURCE_REQUEST = ContextVar("model_source_request", default=False)


def _omit_source_request(record):
    return not _SOURCE_REQUEST.get()


# HTTP client request logs can contain signed redirect URLs. Keep other
# clients' request logging and suppress transport tracing only for this stream.
for _logger_name in ("httpx", "httpcore.connection", "httpcore.http11", "httpcore.http2", "httpcore.proxy"):
    logging.getLogger(_logger_name).addFilter(_omit_source_request)


class ModelError(ValueError):
    pass


class SourceNotFound(ModelError):
    pass


def relative_path(value: str) -> str:
    if (not isinstance(value, str) or not value or len(value) > 512 or "\\" in value
            or any(ord(c) < 32 for c in value) or value.startswith("/")
            or any(p in {"", ".", ".."} for p in value.split("/")) or ":" in value):
        raise ModelError("模型文件路径无效，不能包含绝对路径或目录穿越")
    return value


def filename(value: str, *, model_type: str | None = None) -> str:
    value = relative_path(value)
    if "/" in value:
        raise ModelError("模型文件路径无效，不能包含绝对路径或目录穿越")
    ext = PurePosixPath(value).suffix.lower()
    if ext in SAFE_EXTENSIONS:
        return value
    if ext in CONTROLLED_UPSCALE_EXTENSIONS:
        if model_type == "upscale_models":
            return value
        raise ModelError("只允许 safetensors、gguf、onnx 文件；.pt/.pth 仅允许在 upscale_models 分类下通过可信来源受控导入")
    raise ModelError("只允许 safetensors、gguf、onnx 文件；可执行或 pickle 权重需先在可信环境转换")


def is_controlled_upscale_extension(value: str) -> bool:
    return isinstance(value, str) and PurePosixPath(value).suffix.lower() in CONTROLLED_UPSCALE_EXTENSIONS


def trusted_upscale_github_owners() -> frozenset[str]:
    extra = {
        item.strip().lower()
        for item in os.getenv("MODEL_LIBRARY_TRUSTED_UPSCALE_GITHUB_OWNERS", "").split(",")
        if item.strip()
    }
    return TRUSTED_UPSCALE_GITHUB_OWNERS | extra


def trusted_upscale_hosts() -> frozenset[str]:
    return frozenset(
        item.strip().lower()
        for item in os.getenv("MODEL_LIBRARY_TRUSTED_UPSCALE_HOSTS", "").split(",")
        if item.strip()
    )


def parse_github_release_url(url: str) -> tuple[str, str, str, str] | None:
    p = urlsplit(normalize_source_url(url))
    if p.hostname != "github.com":
        return None
    parts = [unquote(seg) for seg in p.path.split("/") if seg != ""]
    if len(parts) >= 6 and parts[2] == "releases" and parts[3] == "download":
        owner, repo, tag, asset = parts[0], parts[1], parts[4], "/".join(parts[5:])
        if (re.fullmatch(r"[\w.-]+", owner) and re.fullmatch(r"[\w.-]+", repo)
                and tag and "/" not in asset and asset not in {".", ".."}):
            return owner, repo, tag, asset
    return None


def validate_controlled_upscale_source(url: str | None, *, source: str | None = None) -> None:
    """Enforce trusted release origins for controlled .pt/.pth upscale imports."""
    if not url:
        raise ModelError("受控 .pt/.pth 超分模型必须提供可信 HTTPS 下载来源，不支持无来源本地或任意导入")
    normalized = normalize_source_url(url)
    p = urlsplit(normalized)
    host = (p.hostname or "").lower()
    if source == "huggingface" or host == "huggingface.co":
        if host != "huggingface.co":
            raise ModelError("Hugging Face 来源地址域名无效")
        parse_hf_url(normalized)
        return
    if host == "github.com":
        release = parse_github_release_url(normalized)
        if release and release[0].lower() in trusted_upscale_github_owners():
            return
        raise ModelError("未授权的 GitHub .pt/.pth 来源；upscale_models 仅允许受信任上游项目的 Release 发布文件")
    if source == "civitai" and host == "civitai.com":
        return
    if host in trusted_upscale_hosts():
        return
    raise ModelError("未授权的 .pt/.pth 来源；upscale_models 受控导入仅允许 Hugging Face、受信任 GitHub Release 或管理员授权来源")


def _inspect_pickle_stream(stream, *, expect_magic: int | None = None) -> tuple[set[tuple[str, str]], bool]:
    """Statically inspect one pickle stream using pickletools without executing any code."""
    globals_seen: set[tuple[str, str]] = set()
    stack: list[str | int | None | object] = []
    memo: dict[int, str | int | None] = {}
    mark = object()
    ops_count = 0
    saw_stop = False
    magic_matched = False

    def pop_value():
        return stack.pop() if stack else None

    def pop_to_mark():
        while stack:
            if stack.pop() is mark:
                return

    try:
        for op, arg, _pos in pickletools.genops(stream):
            ops_count += 1
            if ops_count > 2_000_000:
                raise ModelError("权重 pickle 指令数量超出安全上限，已拒绝导入")
            name = op.name
            if name in DISALLOWED_PICKLE_OPCODES:
                raise ModelError(f"检测到不安全的 pickle 指令（{name}），已拒绝导入")
            if name == "STOP":
                saw_stop = True
                break
            if name == "GLOBAL":
                mod, sep, attr = str(arg).partition(" ")
                if not sep or (mod, attr) not in SAFE_TORCH_PICKLE_GLOBALS:
                    raise ModelError(f"检测到未授权的 pickle 引用（{arg}），已拒绝导入")
                globals_seen.add((mod, attr))
                stack.append(None)
            elif name == "STACK_GLOBAL":
                attr = pop_value()
                mod = pop_value()
                if not isinstance(mod, str) or not isinstance(attr, str) or (mod, attr) not in SAFE_TORCH_PICKLE_GLOBALS:
                    raise ModelError("检测到未授权或不可静态解析的 STACK_GLOBAL 引用，已拒绝导入")
                globals_seen.add((mod, attr))
                stack.append(None)
            elif name in {"SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE", "SHORT_BINSTRING", "BINSTRING", "STRING"}:
                stack.append(arg if isinstance(arg, str) else None)
            elif name in {"LONG", "LONG1", "LONG4", "INT", "BININT", "BININT1", "BININT2"}:
                if expect_magic is not None and arg == expect_magic:
                    magic_matched = True
                stack.append(arg if isinstance(arg, int) else None)
            elif name == "MEMOIZE":
                top = stack[-1] if stack else None
                memo[len(memo)] = top if isinstance(top, (str, int)) else None
            elif name in {"PUT", "BINPUT", "LONG_BINPUT"}:
                top = stack[-1] if stack else None
                memo[int(arg)] = top if isinstance(top, (str, int)) else None
            elif name in {"GET", "BINGET", "LONG_BINGET"}:
                stack.append(memo.get(int(arg)))
            elif name == "MARK":
                stack.append(mark)
            elif name == "POP":
                pop_value()
            elif name == "POP_MARK":
                pop_to_mark()
            elif name == "DUP":
                stack.append(stack[-1] if stack else None)
            elif name in {"TUPLE", "LIST", "DICT", "FROZENSET"}:
                pop_to_mark()
                stack.append(None)
            elif name in {"SETITEMS", "APPENDS", "ADDITEMS"}:
                pop_to_mark()
            elif name == "TUPLE1":
                pop_value()
                stack.append(None)
            elif name in {"TUPLE2", "REDUCE", "BUILD", "NEWOBJ"}:
                pop_value()
                pop_value()
                stack.append(None)
            elif name in {"TUPLE3", "NEWOBJ_EX"}:
                pop_value()
                pop_value()
                pop_value()
                stack.append(None)
            elif name == "SETITEM":
                pop_value()
                pop_value()
            elif name == "APPEND":
                pop_value()
            elif name == "BINPERSID":
                pop_value()
                stack.append(None)
            elif name in {
                "NONE", "NEWTRUE", "NEWFALSE", "FLOAT", "BINFLOAT",
                "EMPTY_DICT", "EMPTY_LIST", "EMPTY_TUPLE", "EMPTY_SET",
                "BINBYTES", "SHORT_BINBYTES", "BINBYTES8", "BYTEARRAY8",
            }:
                stack.append(None)
            elif name in {"PROTO", "FRAME"}:
                continue
            else:
                raise ModelError(f"检测到不受支持的 pickle 指令（{name}），已拒绝导入")
    except ModelError:
        raise
    except (ValueError, EOFError, struct.error) as exc:
        raise ModelError("权重文件损坏或 pickle 结构异常，已拒绝导入") from exc
    if not saw_stop:
        raise ModelError("权重文件不完整或已损坏，已拒绝导入")
    if expect_magic is not None and not magic_matched:
        raise ModelError("无效的 PyTorch 权重文件头，已拒绝导入")
    return globals_seen, any(mod in {"torch", "torch._utils"} for mod, _ in globals_seen)


def validate_weight_file(path: Path, filename_value: str, size: int, *, model_type: str | None = None) -> None:
    """Byte-only static validation preventing spoofing and unsafe pickle payloads."""
    if size <= 0:
        raise ModelError("模型文件为空，已拒绝导入")
    ext = PurePosixPath(filename_value).suffix.lower()
    with path.open("rb") as stream:
        prefix = stream.read(64)
    lower_prefix = prefix.lstrip().lower()

    if ext in SAFE_EXTENSIONS:
        if (prefix.startswith((b"\x80\x02", b"\x80\x03", b"\x80\x04", b"\x80\x05", b"PK\x03\x04", b"\x7fELF", b"MZ", b"#!"))
                or lower_prefix.startswith((b"<!doctype", b"<html", b"<?xml"))):
            raise ModelError("文件内容与扩展名不匹配或包含不受支持的格式（禁止通过修改扩展名绕过安全检查）")
        if ext == ".gguf" and not prefix.startswith(b"GGUF"):
            raise ModelError("无效的 GGUF 文件头，文件内容与扩展名不匹配")
        return

    if ext in CONTROLLED_UPSCALE_EXTENSIONS:
        if model_type != "upscale_models":
            raise ModelError("仅 upscale_models 分类允许受控导入 .pt/.pth 文件")
        if size < 16:
            raise ModelError("PyTorch 权重文件过小或已损坏，已拒绝导入")
        if prefix.startswith(b"PK\x03\x04"):
            try:
                with zipfile.ZipFile(path, "r") as archive:
                    names = archive.namelist()
                    if any(n.startswith("/") or ".." in PurePosixPath(n).parts for n in names):
                        raise ModelError("PyTorch ZIP 权重包含非法路径，已拒绝导入")
                    pkl_entries = [n for n in names if n == "data.pkl" or n.endswith("/data.pkl")]
                    if not pkl_entries:
                        raise ModelError("ZIP 文件缺少 PyTorch data.pkl 权重元数据，禁止伪造扩展名")
                    saw_tensor = False
                    for entry_name in pkl_entries:
                        info = archive.getinfo(entry_name)
                        if info.file_size <= 0 or info.file_size > MAX_PICKLE_ENTRY_BYTES:
                            raise ModelError("PyTorch 权重元数据大小异常，已拒绝导入")
                        with archive.open(info, "r") as entry_stream:
                            _, has_tensor = _inspect_pickle_stream(io.BytesIO(entry_stream.read()))
                            saw_tensor = saw_tensor or has_tensor
                    if not saw_tensor:
                        raise ModelError("权重文件不包含有效的 PyTorch 张量重建结构，已拒绝导入")
                    return
            except zipfile.BadZipFile as exc:
                raise ModelError("PyTorch ZIP 权重文件损坏，已拒绝导入") from exc
        if prefix.startswith(b"\x80\x02"):
            with path.open("rb") as stream:
                _inspect_pickle_stream(stream, expect_magic=PYTORCH_LEGACY_MAGIC)
                _inspect_pickle_stream(stream)
                _inspect_pickle_stream(stream)
                _, has_tensor = _inspect_pickle_stream(stream)
                if not has_tensor:
                    raise ModelError("权重文件不包含有效的 PyTorch 张量重建结构，已拒绝导入")
                return
        raise ModelError("无效或不受支持的 PyTorch 权重文件格式（文件内容与 .pt/.pth 不匹配）")

    raise ModelError("不支持的模型文件扩展名")


def https_url(value: str, *, persisted=True) -> str:
    if not isinstance(value, str):
        raise ModelError("下载链接必须是 HTTPS 字符串")
    try:
        p = urlsplit(value)
        if (len(value) > 4096 or p.scheme != "https" or not p.hostname or p.username or p.password
                or p.port not in {None, 443} or p.fragment or any(ord(c) < 33 for c in value)):
            raise ValueError
        if persisted and any(
            k.lower() == "sig" or any(s in k.lower() for s in ("token", "key", "auth", "signature", "credential", "password", "secret"))
            for k in parse_qs(p.query)
        ):
            raise ModelError("链接含凭据或签名，请使用无密钥来源链接并在设置中配置 Token")
    except (ValueError, TypeError) as exc:
        raise ModelError("只允许无凭据的 HTTPS 公网链接") from exc
    return value


def normalize_source_url(value: str) -> str:
    """Percent-encode literal spaces in the URL path before strict HTTPS validation."""
    if isinstance(value, str) and " " in value and not any(ord(c) < 32 for c in value):
        p = urlsplit(value)
        if p.scheme == "https" and p.hostname and " " not in p.netloc and " " not in p.scheme:
            value = urlunsplit((p.scheme, p.netloc, p.path.replace(" ", "%20"), p.query, p.fragment))
    return https_url(value)


def normalize_hf_repo_type(value: str | None = None) -> str:
    if value is None or value == "":
        return "models"
    if isinstance(value, str):
        normalized = HF_REPO_TYPE_ALIASES.get(value.strip().lower())
        if normalized:
            return normalized
    raise ModelError("不支持的 Hugging Face 仓库类型")


def _validate_hf_repo_and_revision(repo: str, revision: str) -> None:
    if (not isinstance(repo, str) or not re.fullmatch(r"[\w.-]+/[\w.-]+", repo)
            or any(seg in {".", ".."} for seg in repo.split("/"))
            or not isinstance(revision, str) or not revision
            or any(ord(c) < 33 for c in revision)):
        raise ModelError("Hugging Face 仓库或 revision 无效")


def hf_metadata_url(repo: str, revision: str, repo_type: str = "models") -> str:
    repo_type = normalize_hf_repo_type(repo_type)
    _validate_hf_repo_and_revision(repo, revision)
    return f"https://huggingface.co/api/{repo_type}/{repo}/revision/{quote(revision, safe='')}?blobs=true"


def hf_download_url(repo: str, revision: str, path: str, repo_type: str = "models") -> str:
    repo_type = normalize_hf_repo_type(repo_type)
    _validate_hf_repo_and_revision(repo, revision)
    path = relative_path(path)
    prefix = "" if repo_type == "models" else f"{repo_type}/"
    return f"https://huggingface.co/{prefix}{repo}/resolve/{quote(revision, safe='')}/{quote(path, safe='/')}"


def parse_hf_url(url: str) -> tuple[str, str, str, str | None, bool]:
    """Parse a Hugging Face URL into (repo_type, repo_id, revision, path, is_tree)."""
    p = urlsplit(normalize_source_url(url))
    if p.hostname != "huggingface.co":
        raise ModelError("不是有效的 Hugging Face 链接")
    parts = [unquote(seg) for seg in p.path.split("/") if seg != ""]
    if parts and parts[0] in {"datasets", "spaces"}:
        repo_type = parts[0]
        parts = parts[1:]
    else:
        repo_type = "models"
    if len(parts) < 2:
        raise ModelError("Hugging Face 链接缺少仓库")
    repo = f"{parts[0]}/{parts[1]}"
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo) or any(seg in {".", ".."} for seg in parts[:2]):
        raise ModelError("Hugging Face 仓库或 revision 无效")
    rest = parts[2:]
    if not rest:
        return repo_type, repo, "main", None, False
    action = rest[0]
    if action not in {"resolve", "blob", "tree"}:
        raise ModelError("不支持的 Hugging Face 链接路径，请使用仓库、resolve、blob 或 tree 链接")
    if len(rest) < 2 or not rest[1]:
        raise ModelError("Hugging Face 链接缺少 revision")
    revision = rest[1]
    _validate_hf_repo_and_revision(repo, revision)
    subpath = "/".join(rest[2:]) if len(rest) > 2 else None
    if action in {"resolve", "blob"} and not subpath:
        raise ModelError("Hugging Face 文件链接缺少文件路径")
    if subpath is not None:
        subpath = relative_path(subpath)
    return repo_type, repo, revision, subpath, action == "tree"


def public_addresses(host: str) -> list[str]:
    try:
        addresses = list(dict.fromkeys(x[4][0] for x in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)))
        if not addresses or any(not ipaddress.ip_address(ip).is_global or ipaddress.ip_address(ip).is_multicast or ipaddress.ip_address(ip).is_reserved for ip in addresses):
            raise ModelError("下载地址指向内部、保留或本地网络，已拒绝")
        return addresses
    except OSError as exc:
        raise ModelError("下载域名解析失败，请检查来源链接") from exc


class SourceHTTP:
    def __init__(self, tokens=None, client=None, resolver=public_addresses):
        self.tokens = tokens or {}
        self.client = client or httpx.Client(timeout=60, follow_redirects=False, trust_env=False)
        self.resolver = resolver

    @contextmanager
    def stream(self, url, *, offset=0, validator=None, drive_token=None):
        token = _SOURCE_REQUEST.set(True)
        try:
            with self._stream(url, offset, validator, drive_token) as response:
                yield response
        finally:
            _SOURCE_REQUEST.reset(token)

    @contextmanager
    def _stream(self, url, offset, validator, drive_token):
        initial_host = urlsplit(https_url(url, persisted=False)).hostname
        for _ in range(6):
            parsed = urlsplit(https_url(url, persisted=False))
            host = parsed.hostname
            address = self.resolver(host)[0]
            original = httpx.URL(url)
            pinned = original.copy_with(host=address)
            headers = {"Host": host, "Accept-Encoding": "identity", "Connection": "close"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
                if validator:
                    headers["If-Range"] = validator
            source = {"huggingface.co": "huggingface", "civitai.com": "civitai"}.get(host)
            if host == initial_host and source and self.tokens.get(source):
                headers["Authorization"] = "Bearer " + self.tokens[source]
            if host == initial_host == "www.googleapis.com" and drive_token:
                headers["Authorization"] = "Bearer " + drive_token
            try:
                request = self.client.build_request("GET", pinned, headers=headers,
                                                    extensions={"sni_hostname": host.encode()})
                request.headers.pop("Cookie", None)
                with closing(self.client.send(request, stream=True)) as response:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location")
                        if not location:
                            raise ModelError("下载来源重定向缺少目标")
                        url = urljoin(url, location)
                        continue
                    if not (offset and response.status_code in {206, 416}):
                        self.check(response)
                    response.extensions["studio_source_resource"] = hashlib.sha256(
                        f"{host}{parsed.path}".encode()
                    ).hexdigest()
                    yield response
                    return
            except httpx.HTTPError as exc:
                raise ModelError("来源网络连接中断，请重试；不会记录可能含凭据的响应内容") from exc
        raise ModelError("下载来源重定向次数过多")

    @staticmethod
    def check(response):
        code = response.status_code
        if code >= 400:
            message = {401: "鉴权失败，请更新下载源凭据", 403: "无权限或许可未接受，请到来源页面授权/接受许可",
                       404: "模型或版本不存在，或链接已失效", 429: "来源限流，请稍后重试"}.get(code, "来源服务错误，请稍后重试")
            error = SourceNotFound if code == 404 else ModelError
            raise error(f"{message}（HTTP {code}）")
        if code != 200:
            raise ModelError(f"来源返回非完整响应（HTTP {code}）")

    def json(self, url):
        with self.stream(url) as response:
            raw = bytearray()
            for chunk in response.iter_bytes(65536):
                raw.extend(chunk)
                if len(raw) > 16 * 1024 * 1024:
                    raise ModelError("来源元数据超过大小限制，请指定具体版本/文件")
        try:
            return json.loads(raw)
        except ValueError as exc:
            raise ModelError("来源未返回有效 JSON") from exc

    def close(self):
        self.client.close()


CIVITAI_TYPES = {"Checkpoint": "checkpoints", "LORA": "loras", "LoCon": "loras", "VAE": "vae",
                 "TextualInversion": "embeddings", "Controlnet": "controlnet", "Upscaler": "upscale_models"}


class Sources:
    def __init__(self, http):
        self.http = http

    def civitai_version(self, version):
        model = version.get("model") or {}
        model_type = CIVITAI_TYPES.get(model.get("type"), "unknown")
        allowed = SAFE_EXTENSIONS | CONTROLLED_UPSCALE_EXTENSIONS if model_type == "upscale_models" else SAFE_EXTENSIONS
        result = []
        for f in version.get("files") or []:
            if PurePosixPath(f["name"]).suffix.lower() not in allowed:
                continue
            digest = (f.get("hashes") or {}).get("SHA256")
            result.append({"display_name": model.get("name") or version.get("name"), "filename": f["name"],
                           "model_type": model_type,
                           "architecture": version.get("baseModel") or "unknown", "source": "civitai",
                           "source_url": f"https://civitai.com/models/{version.get('modelId')}?modelVersionId={version['id']}",
                           "download_url": https_url(f["downloadUrl"]),
                           "expected_sha256": digest.lower() if isinstance(digest, str) and SHA.fullmatch(digest) else None,
                           "expected_size": None,  # sizeKB is rounded, not an exact byte count
                           "reported_size": round(float(f.get("sizeKB") or 0) * 1024),
                           "source_identity": {"model_id": version.get("modelId"), "version_id": version["id"], "file_id": f["id"]},
                           "version_name": version.get("name"), "trigger_words": version.get("trainedWords") or []})
        return result

    def _hf_resolve_url(self, url: str, *, model_type: str | None = None):
        repo_type, repo, revision, path, is_tree = parse_hf_url(url)
        try:
            return self.huggingface(repo, revision, path, repo_type=repo_type, is_tree=is_tree, model_type=model_type)
        except SourceNotFound as initial_error:
            if "/" in revision:  # An encoded slash already identifies the exact revision.
                raise
            # Disambiguate branch/tag names containing '/' (e.g. release/v1.0 or refs/pr/1).
            p = urlsplit(normalize_source_url(url))
            parts = [unquote(seg) for seg in p.path.split("/") if seg != ""]
            if parts and parts[0] in {"datasets", "spaces"}:
                parts = parts[1:]
            rest = parts[2:]
            if len(rest) < 3 or rest[0] not in {"resolve", "blob", "tree"}:
                raise
            action = rest[0]
            tail = rest[1:]
            candidates: list[tuple[str, str | None]] = []
            try:
                refs_data = self.http.json(f"https://huggingface.co/api/{repo_type}/{repo}/refs")
                ref_names = [
                    item.get("name")
                    for group in ("branches", "tags", "converts")
                    for item in (refs_data.get(group) or [])
                    if isinstance(item, dict) and isinstance(item.get("name"), str) and "/" in item["name"]
                ]
                for name in sorted(set(ref_names), key=len, reverse=True):
                    segs = name.split("/")
                    if tail[:len(segs)] == segs:
                        sub = "/".join(tail[len(segs):]) or None
                        if action in {"resolve", "blob"} and not sub:
                            continue
                        candidates.append((name, sub))
            except ModelError:
                pass
            max_split = min(len(tail) if action == "tree" else len(tail) - 1, 4)
            for idx in range(2, max_split + 1):
                cand_rev = "/".join(tail[:idx])
                cand_sub = "/".join(tail[idx:]) or None
                pair = (cand_rev, cand_sub)
                if pair not in candidates:
                    candidates.append(pair)
            for cand_rev, cand_sub in candidates:
                try:
                    return self.huggingface(repo, cand_rev, cand_sub, repo_type=repo_type, is_tree=(action == "tree"), model_type=model_type)
                except SourceNotFound:
                    continue
            raise initial_error

    def _github_release_metadata(self, owner: str, repo: str, tag: str, asset_name: str) -> tuple[int | None, str | None]:
        try:
            release = self.http.json(f"https://api.github.com/repos/{owner}/{repo}/releases/tags/{quote(tag, safe='')}")
        except ModelError as exc:
            if "HTTP 404" in str(exc):
                raise
            return None, None
        for asset in release.get("assets") or []:
            if asset.get("name") == asset_name:
                size = asset.get("size") if type(asset.get("size")) is int and asset["size"] > 0 else None
                raw_digest = asset.get("digest") or ""
                digest = None
                if isinstance(raw_digest, str) and raw_digest.lower().startswith("sha256:"):
                    candidate = raw_digest.split(":", 1)[1].strip()
                    if SHA.fullmatch(candidate):
                        digest = candidate.lower()
                return size, digest
        return None, None

    def resolve(self, body):
        requested_type = body.get("model_type")
        if body.get("sha256"):
            value = body["sha256"]
            if not isinstance(value, str) or not SHA.fullmatch(value):
                raise ModelError("Hash 来源查询需要完整 SHA-256，不能把短 Hash 当作 SHA-256")
            version = self.http.json("https://civitai.com/api/v1/model-versions/by-hash/" + value)
            return [f for f in self.civitai_version(version) if f["expected_sha256"] == value.lower()]
        if body.get("repo_id"):
            repo = body["repo_id"]
            return self.huggingface(
                repo,
                body.get("revision") or "main",
                body.get("filename"),
                repo_type=body.get("repo_type") or "models",
                model_type=requested_type,
            )
        url = normalize_source_url(body.get("url") or "")
        p = urlsplit(url)
        parts = [unquote(seg) for seg in p.path.split("/") if seg != ""]
        if p.hostname == "civitai.com":
            query = parse_qs(p.query)
            version_id = (query.get("modelVersionId") or [None])[0]
            if parts[:3] == ["api", "download", "models"] and len(parts) == 4:
                version_id = parts[3]
            if version_id:
                if not str(version_id).isdigit():
                    raise ModelError("Civitai 版本 ID 无效")
                return self.civitai_version(self.http.json("https://civitai.com/api/v1/model-versions/" + str(version_id)))
            if len(parts) >= 2 and parts[0] == "models" and parts[1].isdigit():
                model = self.http.json("https://civitai.com/api/v1/models/" + parts[1])
                return [f for v in model.get("modelVersions") or []
                        for f in self.civitai_version({**v, "model": model, "modelId": model["id"]})]
            raise ModelError("请提供 Civitai 模型页面或指定版本链接")
        if p.hostname == "huggingface.co":
            return self._hf_resolve_url(url, model_type=requested_type)
        resolved_name = body.get("filename") or unquote(PurePosixPath(p.path).name)
        effective_type = requested_type or "unknown"
        expected_size = body.get("expected_size")
        expected_sha256 = body.get("expected_sha256")
        if resolved_name and is_controlled_upscale_extension(resolved_name):
            if requested_type not in {None, "", "unknown", "upscale_models"}:
                raise ModelError("只允许 safetensors、gguf、onnx 文件；.pt/.pth 仅允许在 upscale_models 分类下通过可信来源受控导入")
            validate_controlled_upscale_source(url, source="https")
            effective_type = "upscale_models"
        gh = parse_github_release_url(url)
        if gh and (expected_size is None or expected_sha256 is None):
            gh_size, gh_sha = self._github_release_metadata(*gh)
            expected_size = expected_size if expected_size is not None else gh_size
            expected_sha256 = expected_sha256 if expected_sha256 is not None else gh_sha
        return [{"source": "https", "source_url": url, "download_url": url,
                 "filename": resolved_name,
                 "display_name": body.get("display_name") or resolved_name or None,
                 "model_type": effective_type,
                 "architecture": body.get("architecture") or "unknown", "expected_sha256": expected_sha256,
                 "expected_size": expected_size, "source_identity": {"url": url}}]

    def huggingface(self, repo, revision, path=None, *, repo_type="models", is_tree=False, model_type=None):
        repo_type = normalize_hf_repo_type(repo_type)
        _validate_hf_repo_and_revision(repo, revision)
        if path is not None:
            path = relative_path(path)
        meta = self.http.json(hf_metadata_url(repo, revision, repo_type))
        commit = meta.get("sha") or ""
        if not re.fullmatch(r"[a-fA-F0-9]{40}", commit):
            raise ModelError("无法固定 Hugging Face commit，请稍后重试")
        siblings = meta.get("siblings") or []
        if path and not is_tree:
            exact = [f for f in siblings if (f.get("rfilename") or "") == path]
            if not exact:
                raise SourceNotFound("仓库中未找到指定文件，请检查路径或版本（HTTP 404）")
            leaf = PurePosixPath(path).name
            if is_controlled_upscale_extension(leaf):
                if model_type not in {None, "", "unknown", "upscale_models"}:
                    raise ModelError("只允许 safetensors、gguf、onnx 文件；.pt/.pth 仅允许在 upscale_models 分类下通过可信来源受控导入")
                allowed = SAFE_EXTENSIONS | CONTROLLED_UPSCALE_EXTENSIONS
            else:
                filename(leaf, model_type=model_type)
                allowed = SAFE_EXTENSIONS
        else:
            allowed = SAFE_EXTENSIONS | CONTROLLED_UPSCALE_EXTENSIONS if model_type == "upscale_models" else SAFE_EXTENSIONS
        files = []
        for f in siblings:
            name = f.get("rfilename") or ""
            ext = PurePosixPath(name).suffix.lower()
            if ext not in allowed:
                continue
            if path:
                if is_tree:
                    if name != path and not name.startswith(path + "/"):
                        continue
                elif name != path:
                    continue
            relative_path(name)
            lfs = f.get("lfs") or {}
            digest = lfs.get("sha256")
            download = hf_download_url(repo, commit, name, repo_type)
            identity = {"repo_id": repo, "revision": commit, "filename": name}
            if repo_type != "models":
                identity["repo_type"] = repo_type
            item_type = "upscale_models" if ext in CONTROLLED_UPSCALE_EXTENSIONS else "unknown"
            files.append({"display_name": name, "filename": PurePosixPath(name).name,
                          "source": "huggingface", "source_url": download, "download_url": download,
                          "architecture": "unknown", "model_type": item_type,
                          "source_identity": identity,
                          "expected_sha256": digest.lower() if isinstance(digest, str) and SHA.fullmatch(digest) else None,
                          "expected_size": lfs.get("size") or f.get("size")})
        if is_tree and not files:
            if path and not any((f.get("rfilename") or "").startswith(path + "/") for f in siblings):
                raise SourceNotFound("仓库中未找到指定目录，请检查路径或版本（HTTP 404）")
            raise ModelError("该目录下没有符合当前分类安全策略的受支持模型文件（目录链接需选择具体权重文件）")
        return files

    def search(self, query):
        if not isinstance(query, str) or not 1 <= len(query) <= 200:
            raise ModelError("搜索名称需为 1–200 字符")
        data = self.http.json("https://civitai.com/api/v1/models?" + urlencode({"query": query, "limit": 20}))
        civitai = [{"name": m["name"], "source": "civitai", "url": f"https://civitai.com/models/{m['id']}"}
                   for m in data.get("items") or []]
        hf = self.http.json("https://huggingface.co/api/models?" + urlencode({"search": query, "limit": 20}))
        return civitai + [{"name": m["id"], "source": "huggingface", "url": "https://huggingface.co/" + m["id"]} for m in hf]
