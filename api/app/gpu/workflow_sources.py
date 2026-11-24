"""Bounded, SSRF-hardened fetching and parsing for remote workflow JSON and PNG files.

Streams remote HTTPS payloads into memory with strict size and structure limits,
never writes fetched images to disk, never decodes image pixels, and never persists
or logs source URLs (which may contain temporary query signatures).
"""
from __future__ import annotations

import json
import re
import struct
import zlib
from pathlib import PurePosixPath
from urllib.parse import unquote, urlsplit, urlunsplit

from .model_dependencies import _is_api_prompt_graph, _is_visual_workflow
from .model_sources import ModelError, SourceHTTP, SourceNotFound, https_url

MAX_PNG_BYTES = 32 * 1024 * 1024
MAX_JSON_BYTES = 5 * 1024 * 1024
MAX_TEXT_BYTES = 4 * 1024 * 1024
MAX_PNG_CHUNKS = 10_000
MAX_PNG_PIXELS = 100_000_000
MAX_WORKFLOW_NODES = 10_000
MAX_WALK_DEPTH = 64

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_WEIGHT_FILE_RE = re.compile(r"\.(safetensors|ckpt|pt|pth|bin|gguf|onnx)$", re.IGNORECASE)
_REPO_SEGMENT_RE = re.compile(r"^[\w.-]+$")
_CIVITAI_OPTION_RE = re.compile(r"^[A-Za-z0-9_-]+=[^,/]+$")
_DISPOSITION_STAR_RE = re.compile(r"filename\*\s*=\s*(?:UTF-8|utf-8)''([^;\r\n]+)")
_DISPOSITION_PLAIN_RE = re.compile(r'filename\s*=\s*"?([^";\r\n]+)"?')

PLAIN_COMFYUI_WORKFLOW_ERROR = (
    "检测到普通 ComfyUI 工作流 JSON（不含 Studio manifest / object_info）。"
    "如需检测并下载所需模型，请前往「模型库 › 检测工作流模型」；"
    "创作模式工作流需导入包含 manifest、api_graph、object_info 的完整工作流包，或含元数据的 PNG 原图。"
)


def _clean_source_url(raw_url: str) -> str:
    if not isinstance(raw_url, str) or not raw_url.strip():
        raise ModelError("请输入有效的 HTTPS 链接")
    cleaned = raw_url.strip()
    if any(ord(c) < 32 for c in cleaned):
        raise ModelError("链接包含非法控制字符")
    parsed = urlsplit(cleaned)
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        raise ModelError("仅支持 HTTPS 公网链接")
    # Strip trailing fragments (common when copying GitHub/browser links) and encode literal spaces in path.
    path = parsed.path.replace(" ", "%20") if " " in parsed.path else parsed.path
    normalized = urlunsplit((parsed.scheme.lower(), parsed.netloc, path, parsed.query, ""))
    return https_url(normalized, persisted=False)


def normalize_workflow_url(raw_url: str) -> tuple[str, str | None, str | None]:
    """Validate and lightly normalize common community file URLs.

    Returns ``(primary_url, fallback_url, note)``.
    """
    url = _clean_source_url(raw_url)
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower()
    raw_segments = [seg for seg in parsed.path.split("/") if seg != ""]
    decoded_segments = [unquote(seg) for seg in raw_segments]

    # GitHub blob page -> raw.githubusercontent.com direct file URL
    if host == "github.com" and len(raw_segments) >= 5 and raw_segments[2] == "blob":
        owner, repo = decoded_segments[0], decoded_segments[1]
        if (
            _REPO_SEGMENT_RE.fullmatch(owner)
            and _REPO_SEGMENT_RE.fullmatch(repo)
            and not any(seg in {".", ".."} for seg in decoded_segments)
        ):
            raw_path = "/" + "/".join([raw_segments[0], raw_segments[1], *raw_segments[3:]])
            converted = https_url(urlunsplit(("https", "raw.githubusercontent.com", raw_path, "", "")), persisted=False)
            return converted, None, "已自动将 GitHub 网页链接转换为 Raw 原始文件链接"

    # Hugging Face blob page -> resolve direct file URL
    if host == "huggingface.co" and len(raw_segments) >= 5:
        blob_idx = 3 if raw_segments[0] in {"datasets", "spaces"} else 2
        if len(raw_segments) > blob_idx + 2 and raw_segments[blob_idx] == "blob":
            if not any(seg in {".", ".."} for seg in decoded_segments):
                updated = [*raw_segments[:blob_idx], "resolve", *raw_segments[blob_idx + 1 :]]
                converted = https_url(
                    urlunsplit(("https", parsed.netloc, "/" + "/".join(updated), parsed.query, "")),
                    persisted=False,
                )
                return converted, None, "已自动将 Hugging Face 网页链接转换为原始文件链接"

    # Civitai image CDN resized URL -> attempt original=true to preserve PNG workflow metadata
    if host == "image.civitai.com" and len(raw_segments) >= 4:
        transform = raw_segments[2]
        tokens = [t for t in transform.split(",") if t]
        if tokens and all(_CIVITAI_OPTION_RE.fullmatch(t) for t in tokens) and transform != "original=true":
            original_segments = [raw_segments[0], raw_segments[1], "original=true", *raw_segments[3:]]
            original_url = https_url(
                urlunsplit(("https", parsed.netloc, "/" + "/".join(original_segments), parsed.query, "")),
                persisted=False,
            )
            return original_url, url, "已自动将 Civitai 缩略图链接切换为原图地址以尝试保留工作流元数据"

    return url, None, None


def _inflate_bounded(payload: bytes, limit: int) -> bytes:
    decompressor = zlib.decompressobj()
    try:
        out = decompressor.decompress(payload, limit + 1)
        if len(out) > limit or decompressor.unconsumed_tail:
            raise ModelError("元数据解压后超过 4 MiB 限制")
        tail = decompressor.flush(max(1, limit - len(out) + 1))
        out += tail
        if len(out) > limit or not decompressor.eof:
            raise ModelError("元数据解压后超过 4 MiB 限制")
        return out
    except zlib.error as exc:
        raise ModelError("PNG 压缩元数据损坏，无法解压") from exc


def extract_png_workflow(data: bytes) -> dict:
    """Bounded, pure-Python PNG metadata extraction matching web/src/lib/png-workflow.ts."""
    if len(data) > MAX_PNG_BYTES:
        raise ModelError("图片不能超过 32 MiB")
    if len(data) < 8 or data[:8] != PNG_SIGNATURE:
        raise ModelError("请选择 PNG 原图，不支持此文件格式")

    texts: dict[str, str] = {}
    metadata: list[dict[str, str]] = []
    offset = 8
    chunks = 0
    total_text = 0
    compressed_text = 0
    ended = False
    image_data = False
    data_len = len(data)

    while offset < data_len:
        chunks += 1
        if chunks > MAX_PNG_CHUNKS or offset + 12 > data_len:
            raise ModelError("PNG 损坏或数据块过多")
        length = struct.unpack_from(">I", data, offset)[0]
        if length > data_len - offset - 12:
            raise ModelError("PNG 数据块被截断")
        kind_bytes = data[offset + 4 : offset + 8]
        chunk_data = data[offset + 8 : offset + 8 + length]
        expected_crc = struct.unpack_from(">I", data, offset + 8 + length)[0]
        if (zlib.crc32(data[offset + 4 : offset + 8 + length]) & 0xFFFFFFFF) != expected_crc:
            raise ModelError("PNG 校验失败，文件可能损坏")
        kind = kind_bytes.decode("latin-1")

        if chunks == 1 and (kind != "IHDR" or length != 13):
            raise ModelError("PNG 缺少有效文件头")
        if kind == "IHDR":
            width, height = struct.unpack_from(">II", data, offset + 8)
            if chunks != 1 or not width or not height:
                raise ModelError("PNG 文件头无效")
            if width * height > MAX_PNG_PIXELS:
                raise ModelError("图片像素超过限制")
        if kind == "IDAT":
            image_data = True
        if kind in {"tEXt", "iTXt", "zTXt"}:
            compressed_text += length
            if compressed_text > MAX_TEXT_BYTES:
                raise ModelError("元数据超过 4 MiB 限制")
            split = chunk_data.find(b"\x00")
            if split < 1 or split > 79:
                raise ModelError("PNG 文本块无效")
            key = chunk_data[:split].decode("latin-1")
            payload = chunk_data[split + 1 :]
            packed = False
            if kind == "zTXt":
                if not payload or payload[0] != 0:
                    raise ModelError("PNG 文本压缩方式不支持")
                packed = True
                payload = payload[1:]
            elif kind == "iTXt":
                if len(payload) < 2 or payload[0] not in (0, 1) or payload[1] != 0:
                    raise ModelError("PNG 国际文本块无效")
                packed = payload[0] == 1
                payload = payload[2:]
                for _ in range(2):
                    end = payload.find(b"\x00")
                    if end < 0:
                        raise ModelError("PNG 国际文本块被截断")
                    payload = payload[end + 1 :]
            if packed:
                payload = _inflate_bounded(payload, MAX_TEXT_BYTES - total_text)
            total_text += len(payload)
            if total_text > MAX_TEXT_BYTES:
                raise ModelError("元数据解压后超过 4 MiB 限制")
            try:
                original_text = payload.decode("utf-8")
            except UnicodeDecodeError:
                original_text = payload.decode("cp1252")
            metadata.append({"keyword": key, "text": original_text})
            if key in {"workflow", "prompt", "parameters"}:
                if key in texts:
                    raise ModelError(f"重复的 {key} 元数据，无法确定应使用哪份")
                try:
                    texts[key] = payload.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ModelError(f"{key} 元数据不是有效 UTF-8 文本") from exc

        offset += length + 12
        if kind == "IEND":
            if length != 0 or offset != data_len or not image_data:
                raise ModelError("PNG 结束块无效")
            ended = True
            break

    if not ended:
        raise ModelError("PNG 文件不完整")

    result: dict = {
        "metadata": metadata,
        "warnings": [],
        "node_count": 0,
        "node_types": [],
        "model_files": [],
    }
    for key in ("workflow", "prompt"):
        if key not in texts:
            continue
        try:
            value = json.loads(texts[key])
            if not isinstance(value, dict):
                raise ValueError("必须为 JSON 对象")
            if key == "workflow":
                nodes = value.get("nodes")
                links = value.get("links")
                if (
                    not isinstance(nodes, list)
                    or not all(
                        isinstance(n, dict)
                        and isinstance(n.get("id"), (int, str))
                        and not isinstance(n.get("id"), bool)
                        and isinstance(n.get("type"), str)
                        for n in nodes
                    )
                    or (links is not None and not isinstance(links, list))
                ):
                    raise ValueError("不是可视化节点工作流")
            elif not value or not all(
                isinstance(n, dict) and isinstance(n.get("class_type"), str) and isinstance(n.get("inputs"), dict)
                for n in value.values()
            ):
                raise ValueError("不是 API prompt 工作流")
            result[key] = value
        except ValueError as exc:
            result["warnings"].append(f"{key} 无效：{exc}")

    if "workflow" not in result and "prompt" not in result:
        if result["warnings"]:
            raise ModelError("；".join(result["warnings"]))
        if "parameters" in texts:
            raise ModelError("检测到 A1111 / Forge parameters，不包含完整 ComfyUI 节点工作流")
        raise ModelError("没有找到 ComfyUI 元数据。图片可能被社交平台压缩或转码，请提供 PNG 原图链接")

    if "prompt" not in result:
        result["warnings"].append("仅有可视化 workflow，无法可靠自动转换为 API prompt；运行前需要提供 API 格式。")

    nodes = list(result["prompt"].values()) if "prompt" in result else result["workflow"]["nodes"]
    if len(nodes) > MAX_WORKFLOW_NODES:
        raise ModelError("节点数量超过 10000 限制")
    result["node_count"] = len(nodes)
    result["node_types"] = sorted({str(n.get("class_type") or n.get("type")) for n in nodes if isinstance(n, dict)})

    files: set[str] = set()

    def walk(value, depth: int = 0) -> None:
        if depth > MAX_WALK_DEPTH:
            raise ModelError("元数据嵌套过深")
        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned.lower().startswith(("http://", "https://")) and _WEIGHT_FILE_RE.search(cleaned):
                files.add(cleaned)
        elif isinstance(value, list):
            for item in value:
                walk(item, depth + 1)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item, depth + 1)

    if "workflow" in result:
        walk(result["workflow"])
    if "prompt" in result:
        walk(result["prompt"])
    result["model_files"] = sorted(files)
    return result


def _sanitize_label(raw: str, fallback: str) -> str:
    cleaned = raw.replace("\\", "/").strip()
    name = PurePosixPath(cleaned).name if cleaned else ""
    name = "".join(ch for ch in name if ord(ch) >= 32 and ch not in {'"', "<", ">"})
    name = name.strip()
    if not name or name in {".", ".."}:
        return fallback
    return name[:120]


def _extract_source_name(disposition: str | None, url_path: str, fallback_host: str) -> str:
    if disposition:
        match = _DISPOSITION_STAR_RE.search(disposition)
        if match:
            candidate = _sanitize_label(unquote(match.group(1)), "")
            if candidate:
                return candidate
        match = _DISPOSITION_PLAIN_RE.search(disposition)
        if match:
            candidate = _sanitize_label(match.group(1), "")
            if candidate:
                return candidate
    if url_path:
        candidate = _sanitize_label(unquote(url_path), "")
        if candidate:
            return candidate
    return fallback_host or "remote-workflow"


def _detect_payload_kind(head: bytes, content_type: str) -> tuple[str, int]:
    if head.startswith(PNG_SIGNATURE):
        return "png", MAX_PNG_BYTES
    if (
        head.startswith(b"\xff\xd8\xff")
        or (len(head) >= 12 and head[:4] == b"RIFF" and head[8:12] == b"WEBP")
        or head.startswith((b"GIF87a", b"GIF89a", b"BM"))
        or (len(head) >= 12 and head[4:8] == b"ftyp")
    ):
        raise ModelError("仅支持 PNG 原图或工作流 JSON；该链接返回的是其他图片格式（可能已被社区转码压缩，丢失了工作流元数据）")

    stripped = head.lstrip(b"\xef\xbb\xbf \t\r\n")
    media_type = content_type.split(";", 1)[0].strip().lower()
    if media_type in {"text/html", "application/xhtml+xml"} or stripped[:16].lower().startswith(
        (b"<!doctype", b"<html", b"<head", b"<svg", b"<?xml")
    ):
        raise ModelError("链接返回的是网页（HTML），而非工作流 JSON 或 PNG 原图；请复制原始文件或 PNG 原图直链")

    if not stripped:
        raise ModelError("链接返回内容为空")
    if stripped.startswith(b"["):
        raise ModelError("JSON 根节点必须是对象，不支持数组")
    if not stripped.startswith(b"{"):
        raise ModelError("该链接返回的内容既不是 PNG 原图，也不是有效的工作流 JSON 对象")
    return "json", MAX_JSON_BYTES


def _translate_http_error(exc: Exception) -> ModelError:
    if isinstance(exc, SourceNotFound):
        return ModelError("链接目标不存在或已失效（HTTP 404），请确认地址可公开访问")
    msg = str(exc)
    if "HTTP 401" in msg or "HTTP 403" in msg:
        code = "401" if "HTTP 401" in msg else "403"
        return ModelError(f"无法访问该链接（HTTP {code}），可能需要登录或受防盗链/人机验证保护，请手动下载原图或 JSON 后拖入")
    if "HTTP 429" in msg:
        return ModelError("目标站点限流（HTTP 429），请稍后重试或手动下载后拖入")
    return exc if isinstance(exc, ModelError) else ModelError(msg)


def _download_bounded(http: SourceHTTP, url: str) -> tuple[str, bytes, str, str]:
    try:
        with http.stream(url) as response:
            content_type = response.headers.get("content-type", "")
            content_length_raw = response.headers.get("content-length")
            if content_length_raw and content_length_raw.isdigit() and int(content_length_raw) > MAX_PNG_BYTES:
                raise ModelError("目标文件超过 32 MiB 大小限制")

            source_host = (
                (response.request.headers.get("host") if response.request is not None else None)
                or urlsplit(url).hostname
                or ""
            ).lower()
            source_name = _extract_source_name(
                response.headers.get("content-disposition"),
                str(response.url.path) if response.url is not None else urlsplit(url).path,
                source_host,
            )

            buf = bytearray()
            kind: str | None = None
            limit = MAX_PNG_BYTES
            for chunk in response.iter_bytes(65536):
                if not chunk:
                    continue
                buf.extend(chunk)
                if kind is None and len(buf) >= 32:
                    kind, limit = _detect_payload_kind(bytes(buf[:512]), content_type)
                if len(buf) > limit:
                    if kind == "json":
                        raise ModelError("工作流 JSON 超过 5 MiB 大小限制")
                    raise ModelError("图片不能超过 32 MiB")

            if not buf:
                raise ModelError("链接返回内容为空")
            if kind is None:
                kind, limit = _detect_payload_kind(bytes(buf[:512]), content_type)
                if len(buf) > limit:
                    if kind == "json":
                        raise ModelError("工作流 JSON 超过 5 MiB 大小限制")
                    raise ModelError("图片不能超过 32 MiB")

            return kind, bytes(buf), source_name, source_host
    except (SourceNotFound, ModelError) as exc:
        raise _translate_http_error(exc) from None


def fetch_workflow_source(http: SourceHTTP, raw_url: str) -> dict:
    """Fetch and parse a remote HTTPS workflow JSON or PNG file into structured data."""
    primary_url, fallback_url, note = normalize_workflow_url(raw_url)
    try:
        kind, data, source_name, source_host = _download_bounded(http, primary_url)
    except ModelError as primary_exc:
        if fallback_url is not None and any(tag in str(primary_exc) for tag in ("HTTP ", "网络连接中断", "重定向")):
            kind, data, source_name, source_host = _download_bounded(http, fallback_url)
            note = None
        else:
            raise

    if kind == "png":
        extraction = extract_png_workflow(data)
        return {
            "kind": "png",
            "source_name": source_name,
            "source_host": source_host,
            **({"note": note} if note else {}),
            "image_extraction": extraction,
        }

    try:
        parsed = json.loads(data.decode("utf-8-sig"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ModelError("链接返回的 JSON 格式无效") from exc
    if not isinstance(parsed, dict):
        raise ModelError("JSON 根节点必须是对象")

    return {
        "kind": "json",
        "source_name": source_name,
        "source_host": source_host,
        **({"note": note} if note else {}),
        "payload": parsed,
    }


def is_studio_workflow_bundle(payload: dict) -> bool:
    """True if payload satisfies /api/gpu/workflows/import bundle requirements."""
    if not isinstance(payload, dict):
        return False
    source = payload.get("workflow") if isinstance(payload.get("workflow"), dict) else payload
    if isinstance(source.get("image_extraction"), dict):
        return True
    return all(isinstance(source.get(k), dict) for k in ("manifest", "api_graph", "object_info"))


def is_plain_comfyui_workflow(payload: dict) -> bool:
    """True if payload is a standard ComfyUI visual/API workflow JSON rather than a Studio bundle."""
    if not isinstance(payload, dict):
        return False
    if _is_visual_workflow(payload) or _is_api_prompt_graph(payload):
        return True
    if "workflow" in payload or "prompt" in payload:
        return (
            (payload.get("workflow") is not None and _is_visual_workflow(payload.get("workflow")))
            or (payload.get("prompt") is not None and _is_api_prompt_graph(payload.get("prompt")))
        )
    return False
