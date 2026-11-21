"""Model source resolution. Credentials never appear in persisted URLs.

Network requests pin a public DNS result while retaining TLS SNI/Host, including
on redirects. No environment proxy, automatic redirect, or cross-host auth.
"""
from __future__ import annotations

import hashlib
import ipaddress
import json
import logging
import re
import socket
from contextlib import closing, contextmanager
from contextvars import ContextVar
from pathlib import PurePosixPath
from urllib.parse import parse_qs, quote, unquote, urlencode, urljoin, urlsplit, urlunsplit

import httpx

SHA = re.compile(r"^[a-fA-F0-9]{64}$")
SAFE_EXTENSIONS = {".safetensors", ".gguf", ".onnx"}
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


def relative_path(value: str) -> str:
    if (not isinstance(value, str) or not value or len(value) > 512 or "\\" in value
            or any(ord(c) < 32 for c in value) or value.startswith("/")
            or any(p in {"", ".", ".."} for p in value.split("/")) or ":" in value):
        raise ModelError("模型文件路径无效，不能包含绝对路径或目录穿越")
    return value


def filename(value: str) -> str:
    value = relative_path(value)
    if "/" in value or PurePosixPath(value).suffix.lower() not in SAFE_EXTENSIONS:
        raise ModelError("只允许 safetensors、gguf、onnx 文件；可执行或 pickle 权重需先在可信环境转换")
    return value


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
            raise ModelError(f"{message}（HTTP {code}）")
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
        result = []
        for f in version.get("files") or []:
            if PurePosixPath(f["name"]).suffix.lower() not in SAFE_EXTENSIONS:
                continue
            digest = (f.get("hashes") or {}).get("SHA256")
            result.append({"display_name": model.get("name") or version.get("name"), "filename": f["name"],
                           "model_type": CIVITAI_TYPES.get(model.get("type"), "unknown"),
                           "architecture": version.get("baseModel") or "unknown", "source": "civitai",
                           "source_url": f"https://civitai.com/models/{version.get('modelId')}?modelVersionId={version['id']}",
                           "download_url": https_url(f["downloadUrl"]),
                           "expected_sha256": digest.lower() if isinstance(digest, str) and SHA.fullmatch(digest) else None,
                           "expected_size": None,  # sizeKB is rounded, not an exact byte count
                           "reported_size": round(float(f.get("sizeKB") or 0) * 1024),
                           "source_identity": {"model_id": version.get("modelId"), "version_id": version["id"], "file_id": f["id"]},
                           "version_name": version.get("name"), "trigger_words": version.get("trainedWords") or []})
        return result

    def resolve(self, body):
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
            repo_type, repo, revision, path, is_tree = parse_hf_url(url)
            return self.huggingface(repo, revision, path, repo_type=repo_type, is_tree=is_tree)
        return [{"source": "https", "source_url": url, "download_url": url,
                 "filename": body.get("filename") or unquote(PurePosixPath(p.path).name),
                 "display_name": body.get("display_name"), "model_type": body.get("model_type", "unknown"),
                 "architecture": body.get("architecture") or "unknown", "expected_sha256": body.get("expected_sha256"),
                 "expected_size": body.get("expected_size"), "source_identity": {"url": url}}]

    def huggingface(self, repo, revision, path=None, *, repo_type="models", is_tree=False):
        repo_type = normalize_hf_repo_type(repo_type)
        _validate_hf_repo_and_revision(repo, revision)
        if path is not None:
            path = relative_path(path)
        meta = self.http.json(hf_metadata_url(repo, revision, repo_type))
        commit = meta.get("sha") or ""
        if not re.fullmatch(r"[a-fA-F0-9]{40}", commit):
            raise ModelError("无法固定 Hugging Face commit，请稍后重试")
        files = []
        for f in meta.get("siblings") or []:
            name = f.get("rfilename") or ""
            if PurePosixPath(name).suffix.lower() not in SAFE_EXTENSIONS:
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
            files.append({"display_name": name, "filename": PurePosixPath(name).name,
                          "source": "huggingface", "source_url": download, "download_url": download,
                          "architecture": "unknown", "model_type": "unknown",
                          "source_identity": identity,
                          "expected_sha256": digest.lower() if isinstance(digest, str) and SHA.fullmatch(digest) else None,
                          "expected_size": lfs.get("size") or f.get("size")})
        return files

    def search(self, query):
        if not isinstance(query, str) or not 1 <= len(query) <= 200:
            raise ModelError("搜索名称需为 1–200 字符")
        data = self.http.json("https://civitai.com/api/v1/models?" + urlencode({"query": query, "limit": 20}))
        civitai = [{"name": m["name"], "source": "civitai", "url": f"https://civitai.com/models/{m['id']}"}
                   for m in data.get("items") or []]
        hf = self.http.json("https://huggingface.co/api/models?" + urlencode({"search": query, "limit": 20}))
        return civitai + [{"name": m["id"], "source": "huggingface", "url": "https://huggingface.co/" + m["id"]} for m in hf]
