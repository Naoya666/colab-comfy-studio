"""Shared bounded-memory batch ZIP packaging and short-lived download tickets."""

from __future__ import annotations

import re
import secrets
import tempfile
import time
import zipfile
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any
from zoneinfo import ZoneInfo

from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from .common import APIError
from .config import settings
from .image_metadata import clean_download_filename, content_disposition_header

STUDIO_BATCH_ZIP_MAX_ITEMS = 200
STUDIO_BATCH_ZIP_MAX_BYTES = 1024 * 1024 * 1024  # 1 GiB
NATIVE_BATCH_ZIP_MAX_ITEMS = 50
NATIVE_BATCH_ZIP_MAX_BYTES = 512 * 1024 * 1024  # 512 MiB
DOWNLOAD_TICKET_TTL_SECONDS = 180
ZIP_SPOOL_MEMORY_LIMIT = 16 * 1024 * 1024  # 16 MiB before spilling to temp disk
STREAM_CHUNK_BYTES = 64 * 1024

TICKET_KEY_PREFIX = "download-ticket:"
_TICKET_ID_RE = re.compile(r"^[A-Za-z0-9_-]{16,96}$")
_UNSAFE_SEGMENT_RE = re.compile(r"[\x00-\x1f\x7f<>:\"/\\|?*]+")


def _tz() -> ZoneInfo:
    try:
        return ZoneInfo(settings.studio_timezone)
    except Exception:
        return ZoneInfo("UTC")


def build_zip_archive_filename(
    prefix: str,
    *,
    strip_metadata: bool = False,
    now: float | None = None,
) -> str:
    safe_prefix = re.sub(r"[^A-Za-z0-9_-]+", "-", prefix).strip("-") or "gallery"
    stamp = datetime.fromtimestamp(time.time() if now is None else now, tz=_tz()).strftime("%Y%m%d-%H%M%S")
    suffix = "_clean" if strip_metadata else ""
    return f"{safe_prefix}-{stamp}{suffix}.zip"


def sanitize_zip_segment(value: str | None, fallback: str = "file") -> str:
    raw = str(value or "").replace("\\", "/")
    base = PurePosixPath(raw).name if raw else ""
    cleaned = _UNSAFE_SEGMENT_RE.sub("_", base).strip(" .")
    if not cleaned or cleaned in {".", ".."}:
        return fallback
    if len(cleaned) > 120:
        dot = cleaned.rfind(".")
        if 0 < dot < len(cleaned) - 1 and len(cleaned) - dot <= 16:
            ext = cleaned[dot:]
            cleaned = cleaned[: 120 - len(ext)].rstrip(" .") + ext
        else:
            cleaned = cleaned[:120].rstrip(" .")
    return cleaned or fallback


def sanitize_zip_subfolder(subfolder: str | None) -> str:
    if not subfolder:
        return ""
    parts: list[str] = []
    for raw_part in str(subfolder).replace("\\", "/").split("/"):
        seg = raw_part.strip()
        if not seg or seg in {".", ".."}:
            continue
        cleaned = _UNSAFE_SEGMENT_RE.sub("_", seg).strip(" .")
        if cleaned and cleaned not in {".", ".."}:
            parts.append(cleaned[:80])
        if len(parts) >= 6:
            break
    return "/".join(parts)


def unique_zip_entry_name(
    raw_name: str | None,
    used_lower: set[str],
    *,
    strip_metadata: bool = False,
    strippable: bool = False,
    subfolder: str = "",
    fallback: str = "file",
) -> str:
    base_input = clean_download_filename(raw_name) if (strip_metadata and strippable) else (raw_name or fallback)
    filename = sanitize_zip_segment(base_input, fallback=fallback)
    folder = sanitize_zip_subfolder(subfolder)

    dot = filename.rfind(".")
    if 0 < dot < len(filename) - 1:
        stem, ext = filename[:dot], filename[dot:]
    else:
        stem, ext = filename, ""

    index = 1
    while True:
        candidate_file = filename if index == 1 else f"{stem} ({index}){ext}"
        candidate = f"{folder}/{candidate_file}" if folder else candidate_file
        lowered = candidate.lower()
        if lowered not in used_lower:
            used_lower.add(lowered)
            return candidate
        index += 1


def _ticket_record_id(ticket_id: str) -> str:
    return f"{TICKET_KEY_PREFIX}{ticket_id}"


def prune_expired_tickets(store, owner_id: str, *, now: float | None = None) -> int:
    current = time.time() if now is None else now
    pruned = 0
    try:
        rows = store.list("settings", owner_id)
    except Exception:
        return 0
    for row in rows:
        rid = str(row.get("id") or "")
        if not rid.startswith(TICKET_KEY_PREFIX):
            continue
        exp = row.get("expires_at")
        if not isinstance(exp, (int, float)) or exp <= current:
            try:
                store.delete("settings", rid, owner_id)
                pruned += 1
            except Exception:
                pass
    return pruned


def create_download_ticket(
    store,
    owner_id: str,
    kind: str,
    payload: dict[str, Any],
    *,
    ttl_seconds: int = DOWNLOAD_TICKET_TTL_SECONDS,
    now: float | None = None,
) -> dict[str, Any]:
    current = time.time() if now is None else now
    prune_expired_tickets(store, owner_id, now=current)
    ticket_id = secrets.token_urlsafe(24)
    record = {
        "kind": "download_ticket",
        "ticket_kind": kind,
        "ticket_id": ticket_id,
        "payload": payload,
        "created_at_epoch": current,
        "expires_at": current + max(30, int(ttl_seconds)),
    }
    store.create("settings", owner_id, record, record_id=_ticket_record_id(ticket_id))
    return {
        "ticket_id": ticket_id,
        "expires_at": record["expires_at"],
    }


def load_download_ticket(
    store,
    owner_id: str,
    ticket_id: str,
    expected_kind: str,
    *,
    now: float | None = None,
) -> dict[str, Any]:
    if not isinstance(ticket_id, str) or not _TICKET_ID_RE.fullmatch(ticket_id):
        raise APIError("DOWNLOAD_TICKET_INVALID", "下载票据无效", 404)
    record = store.get("settings", _ticket_record_id(ticket_id), owner_id)
    if not record or record.get("kind") != "download_ticket" or record.get("ticket_kind") != expected_kind:
        raise APIError("DOWNLOAD_TICKET_EXPIRED", "下载链接已失效或不存在，请重新点击打包下载", 404)
    current = time.time() if now is None else now
    exp = record.get("expires_at")
    if not isinstance(exp, (int, float)) or exp <= current:
        try:
            store.delete("settings", _ticket_record_id(ticket_id), owner_id)
        except Exception:
            pass
        raise APIError("DOWNLOAD_TICKET_EXPIRED", "下载链接已过期，请重新点击打包下载", 410)
    payload = record.get("payload")
    if not isinstance(payload, dict):
        raise APIError("DOWNLOAD_TICKET_INVALID", "下载票据数据损坏", 404)
    return payload


def open_zip_spool() -> tempfile.SpooledTemporaryFile[bytes]:
    return tempfile.SpooledTemporaryFile(max_size=ZIP_SPOOL_MEMORY_LIMIT, mode="w+b")


def write_zip_file_entry(
    zf: zipfile.ZipFile,
    arcname: str,
    file_path,
) -> None:
    """Write a local file into the ZIP archive in bounded chunks using ZIP_STORED."""
    info = zipfile.ZipInfo(filename=arcname)
    info.compress_type = zipfile.ZIP_STORED
    try:
        st = file_path.stat()
        dt = datetime.fromtimestamp(st.st_mtime)
        if dt.year >= 1980:
            info.date_time = (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
    except OSError:
        pass
    with zf.open(info, "w") as dst, open(file_path, "rb") as src:
        while True:
            chunk = src.read(STREAM_CHUNK_BYTES)
            if not chunk:
                break
            dst.write(chunk)


def write_zip_bytes_entry(
    zf: zipfile.ZipFile,
    arcname: str,
    data: bytes,
) -> None:
    info = zipfile.ZipInfo(filename=arcname)
    info.compress_type = zipfile.ZIP_STORED
    zf.writestr(info, data)


def spooled_file_streaming_response(
    spool: tempfile.SpooledTemporaryFile[bytes],
    *,
    filename: str,
    media_type: str,
) -> StreamingResponse:
    spool.seek(0, 2)
    size = spool.tell()
    spool.seek(0)

    closed = False

    def close_spool() -> None:
        nonlocal closed
        if closed:
            return
        closed = True
        try:
            spool.close()
        except Exception:
            pass

    def body_stream():
        try:
            while True:
                chunk = spool.read(STREAM_CHUNK_BYTES)
                if not chunk:
                    break
                yield chunk
        finally:
            close_spool()

    headers = {
        "Content-Disposition": content_disposition_header(filename, disposition="attachment"),
        "Content-Length": str(size),
        "Cache-Control": "private, no-store",
        "Vary": "Cookie, Authorization",
    }
    return StreamingResponse(
        body_stream(),
        media_type=media_type,
        headers=headers,
        background=BackgroundTask(close_spool),
    )
