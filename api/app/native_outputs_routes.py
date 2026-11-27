"""Authenticated routes for browsing and acting on Native ComfyUI Drive outputs."""

from __future__ import annotations

import asyncio
import hashlib
import re
import zipfile
from urllib.parse import quote

import httpx
from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response, StreamingResponse
from starlette.background import BackgroundTask

from .batch_zip import (
    NATIVE_BATCH_ZIP_MAX_BYTES,
    NATIVE_BATCH_ZIP_MAX_ITEMS,
    STREAM_CHUNK_BYTES,
    build_zip_archive_filename,
    create_download_ticket,
    load_download_ticket,
    open_zip_spool,
    spooled_file_streaming_response,
    unique_zip_entry_name,
    write_zip_bytes_entry,
)
from .common import APIError, current_user, get_store, ok
from .config import settings
from .gpu import native_outputs
from .image_metadata import (
    ImageMetadataStripError,
    clean_download_filename,
    content_disposition_header,
    is_strippable_image_mime,
    strip_image_metadata,
)

router = APIRouter(prefix="/api/gpu/native/outputs")
_ORIGINAL_ASYNC_CLIENT = httpx.AsyncClient


def _get_stream_client(app) -> tuple[httpx.AsyncClient, bool]:
    """Reuse a persistent Drive streaming client in production; honor monkeypatches in tests."""
    if app is not None and httpx.AsyncClient is _ORIGINAL_ASYNC_CLIENT:
        client = getattr(app.state, "_native_outputs_http_client", None)
        if client is None or client.is_closed:
            client = httpx.AsyncClient(
                timeout=httpx.Timeout(connect=15.0, read=120.0, write=30.0, pool=15.0),
                limits=httpx.Limits(max_connections=40, max_keepalive_connections=20),
            )
            app.state._native_outputs_http_client = client
        return client, False
    return httpx.AsyncClient(timeout=httpx.Timeout(connect=15.0, read=120.0, write=30.0, pool=15.0)), True


def _etag_matches(header: str | None, etag: str) -> bool:
    if not header:
        return False
    target = etag.removeprefix("W/").strip()
    for part in header.split(","):
        token = part.strip()
        if token == "*" or token == etag or token.removeprefix("W/").strip() == target:
            return True
    return False


def _content_disposition(filename: str) -> str:
    return content_disposition_header(filename, disposition="inline")


@router.get("")
async def list_outputs(
    request: Request,
    refresh: bool = False,
    limit: int = 48,
    offset: int = 0,
    media_type: str = "all",
    favorite: str = "all",
    q: str = "",
    subfolder: str = "",
    user=Depends(current_user),
):
    if media_type not in {"all", "image", "video"}:
        raise APIError("VALIDATION_ERROR", "无效的媒体类型筛选", 422)
    if favorite not in {"all", "only", "unfavorited"}:
        raise APIError("VALIDATION_ERROR", "无效的收藏筛选", 422)
    result = await asyncio.to_thread(
        native_outputs.list_native_outputs,
        request.app.state.repo,
        user["id"],
        refresh=refresh,
        limit=limit,
        offset=offset,
        media_type=media_type,
        favorite=favorite,
        query=q,
        subfolder=subfolder,
    )
    return ok(result, request)


def _validate_native_batch_items(items: list[dict], *, strip_metadata: bool) -> tuple[int, int]:
    total_bytes = 0
    strippable_count = 0
    for item in items:
        mime = str(item.get("mime") or "")
        size = int(item.get("size") or 0)
        strippable = is_strippable_image_mime(mime)
        if strippable:
            strippable_count += 1
            if strip_metadata and size > settings.max_upload_bytes:
                raise APIError("PAYLOAD_TOO_LARGE", "存在超过大小限制的图片，无法去除元数据后打包", 413)
        total_bytes += max(0, size)
        if total_bytes > NATIVE_BATCH_ZIP_MAX_BYTES:
            raise APIError(
                "PAYLOAD_TOO_LARGE",
                f"所选原生作品总大小超过单次打包上限（{NATIVE_BATCH_ZIP_MAX_BYTES // (1024 * 1024)} MB），请分批下载",
                413,
            )
    return total_bytes, strippable_count


@router.post("/batch-download")
async def create_native_batch_download_ticket(
    body: dict,
    request: Request,
    user=Depends(current_user),
    st=Depends(get_store),
):
    if not isinstance(body, dict):
        raise APIError("VALIDATION_ERROR", "请求体格式无效")
    raw_ids = body.get("file_ids")
    if (
        not isinstance(raw_ids, list)
        or not raw_ids
        or len(raw_ids) > NATIVE_BATCH_ZIP_MAX_ITEMS
        or any(not isinstance(fid, str) or not fid.strip() for fid in raw_ids)
    ):
        raise APIError("VALIDATION_ERROR", f"请提供 1~{NATIVE_BATCH_ZIP_MAX_ITEMS} 个待下载的原生作品编号")
    raw_strip = body.get("strip_metadata", False)
    if not isinstance(raw_strip, bool):
        raise APIError("VALIDATION_ERROR", "strip_metadata 必须为布尔值")
    strip_metadata = raw_strip
    unique_ids = list(dict.fromkeys(fid.strip() for fid in raw_ids))

    def _resolve():
        items, storage, owns = native_outputs.resolve_output_items(request.app.state.repo, user["id"], unique_ids)
        try:
            return items
        finally:
            if owns and storage._owns_client:
                storage.client.close()

    items = await asyncio.to_thread(_resolve)
    total_bytes, strippable_count = _validate_native_batch_items(items, strip_metadata=strip_metadata)
    effective_strip = strip_metadata and strippable_count > 0
    filename = build_zip_archive_filename("native-gallery", strip_metadata=effective_strip)
    ticket = await asyncio.to_thread(
        create_download_ticket,
        st,
        user["id"],
        "native_batch_zip",
        {
            "file_ids": [item["id"] for item in items],
            "strip_metadata": strip_metadata,
            "filename": filename,
        },
    )
    ticket_id = ticket["ticket_id"]
    return ok(
        {
            "ticket_id": ticket_id,
            "download_url": f"/api/gpu/native/outputs/batch-download/{ticket_id}",
            "filename": filename,
            "item_count": len(items),
            "total_bytes": total_bytes,
            "strippable_count": strippable_count,
            "skipped_strip_count": len(items) - strippable_count if strip_metadata else 0,
            "strip_metadata": strip_metadata,
            "expires_at": ticket["expires_at"],
        },
        request,
    )


@router.get("/batch-download/{ticket_id}")
async def download_native_batch_zip(
    ticket_id: str,
    request: Request,
    user=Depends(current_user),
    st=Depends(get_store),
):
    payload = await asyncio.to_thread(load_download_ticket, st, user["id"], ticket_id, "native_batch_zip")
    strip_metadata = bool(payload.get("strip_metadata", False))
    file_ids = payload.get("file_ids")
    if not isinstance(file_ids, list) or not file_ids or len(file_ids) > NATIVE_BATCH_ZIP_MAX_ITEMS:
        raise APIError("DOWNLOAD_TICKET_INVALID", "下载票据内容无效", 404)
    filename = str(payload.get("filename") or build_zip_archive_filename("native-gallery", strip_metadata=strip_metadata))

    def _resolve_and_token():
        items, storage, owns = native_outputs.resolve_output_items(request.app.state.repo, user["id"], file_ids)
        try:
            token = storage.google.access_token("drive")
            base_url = storage.base_url
            return items, token, base_url
        finally:
            if owns and storage._owns_client:
                storage.client.close()

    items, token, base_url = await asyncio.to_thread(_resolve_and_token)
    _validate_native_batch_items(items, strip_metadata=strip_metadata)

    client, owns_client = _get_stream_client(request.app)
    spool = open_zip_spool()
    running_bytes = 0
    used_lower: set[str] = set()
    try:
        with zipfile.ZipFile(spool, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
            for item in items:
                fid = item["id"]
                mime = str(item.get("mime") or "")
                strippable = is_strippable_image_mime(mime)
                arcname = unique_zip_entry_name(
                    item.get("filename"),
                    used_lower,
                    strip_metadata=strip_metadata,
                    strippable=strippable,
                    subfolder=str(item.get("subfolder") or ""),
                    fallback=fid,
                )
                url = f"{base_url}/files/{quote(fid, safe='')}?alt=media"
                req = client.build_request(
                    "GET",
                    url,
                    headers={"Authorization": "Bearer " + token, "Accept-Encoding": "identity"},
                )
                try:
                    upstream = await client.send(req, stream=True)
                except httpx.HTTPError as exc:
                    raise APIError("DRIVE_STREAM_FAILED", "连接 Google Drive 批量读取失败", 502, True) from exc
                try:
                    if upstream.status_code != 200:
                        status = upstream.status_code
                        raise APIError(
                            "DRIVE_STREAM_FAILED",
                            f"读取 Drive 原生作品失败（HTTP {status}）",
                            502 if status >= 500 else status,
                        )
                    if strip_metadata and strippable:
                        buf = bytearray()
                        async for chunk in upstream.aiter_bytes(STREAM_CHUNK_BYTES):
                            buf.extend(chunk)
                            running_bytes += len(chunk)
                            if len(buf) > settings.max_upload_bytes:
                                raise APIError("PAYLOAD_TOO_LARGE", "图片超过大小限制，无法去除元数据", 413)
                            if running_bytes > NATIVE_BATCH_ZIP_MAX_BYTES:
                                raise APIError("PAYLOAD_TOO_LARGE", "所选原生作品总大小超过单次打包上限，请分批下载", 413)
                        try:
                            cleaned = await asyncio.to_thread(strip_image_metadata, bytes(buf), mime)
                        except ImageMetadataStripError as exc:
                            raise APIError("IMAGE_METADATA_STRIP_FAILED", f"去除图片元数据失败：{exc}", 422) from exc
                        write_zip_bytes_entry(zf, arcname, cleaned)
                    else:
                        info = zipfile.ZipInfo(filename=arcname)
                        info.compress_type = zipfile.ZIP_STORED
                        with zf.open(info, "w") as dst:
                            async for chunk in upstream.aiter_bytes(STREAM_CHUNK_BYTES):
                                running_bytes += len(chunk)
                                if running_bytes > NATIVE_BATCH_ZIP_MAX_BYTES:
                                    raise APIError("PAYLOAD_TOO_LARGE", "所选原生作品总大小超过单次打包上限，请分批下载", 413)
                                dst.write(chunk)
                except httpx.HTTPError as exc:
                    raise APIError("DRIVE_STREAM_FAILED", "读取 Google Drive 文件流失败", 502, True) from exc
                finally:
                    await upstream.aclose()
    except BaseException:
        spool.close()
        raise
    finally:
        if owns_client:
            await client.aclose()

    return spooled_file_streaming_response(spool, filename=filename, media_type="application/zip")


@router.patch("/{file_id}")
async def favorite_output(
    file_id: str,
    body: dict,
    request: Request,
    user=Depends(current_user),
):
    if not isinstance(body, dict) or not isinstance(body.get("favorite"), bool):
        raise APIError("VALIDATION_ERROR", "favorite 必须为布尔值", 422)
    updated = await asyncio.to_thread(
        native_outputs.set_native_output_favorite,
        request.app.state.repo,
        user["id"],
        file_id,
        body["favorite"],
    )
    return ok(updated, request)


@router.get("/{file_id}/thumbnail")
async def output_thumbnail(
    file_id: str,
    request: Request,
    user=Depends(current_user),
):
    data, media_type = await asyncio.to_thread(
        native_outputs.get_native_output_thumbnail,
        request.app.state.repo,
        user["id"],
        file_id,
    )
    digest = hashlib.sha256(data).hexdigest()[:16]
    etag = f'W/"native-thumb-{file_id[:32]}-{digest}"'
    headers = {
        "Cache-Control": "private, no-cache",
        "ETag": etag,
        "Vary": "Cookie, Authorization",
    }
    if _etag_matches(request.headers.get("if-none-match"), etag):
        return Response(status_code=304, headers=headers)
    return Response(
        data,
        media_type=media_type,
        headers=headers,
    )


@router.get("/{file_id}/content")
async def output_content(
    file_id: str,
    request: Request,
    download: bool = False,
    strip_metadata: bool = False,
    user=Depends(current_user),
):
    repo = request.app.state.repo
    item, storage, owns = await asyncio.to_thread(
        native_outputs.resolve_output_item, repo, user["id"], file_id,
    )
    if strip_metadata:
        mime = str(item.get("mime") or "")
        if not is_strippable_image_mime(mime):
            if owns and storage._owns_client:
                storage.client.close()
            raise APIError("STRIP_METADATA_UNSUPPORTED", "仅支持对图片去除元数据", 422)
        declared_size = int(item.get("size") or 0)
        if declared_size > settings.max_upload_bytes:
            if owns and storage._owns_client:
                storage.client.close()
            raise APIError("PAYLOAD_TOO_LARGE", "图片超过大小限制，无法去除元数据", 413)
        safe_rev = re.sub(r"[^A-Za-z0-9_-]+", "", str(item.get("revision") or "v0"))[:32] or "v0"
        clean_etag = f'W/"native-clean-{file_id[:32]}-{safe_rev}"'
        clean_headers: dict[str, str] = {
            "Cache-Control": "private, max-age=3600",
            "ETag": clean_etag,
            "Vary": "Cookie, Authorization",
        }
        if _etag_matches(request.headers.get("if-none-match"), clean_etag):
            if owns and storage._owns_client:
                storage.client.close()
            return Response(status_code=304, headers=clean_headers)

    try:
        token = await asyncio.to_thread(storage.google.access_token, "drive")
        base_url = storage.base_url
    finally:
        if owns and storage._owns_client:
            storage.client.close()

    headers = {"Authorization": "Bearer " + token, "Accept-Encoding": "identity"}
    range_header = None if strip_metadata else request.headers.get("range")
    if range_header:
        headers["Range"] = range_header
    if_none_match = None if strip_metadata else request.headers.get("if-none-match")
    if if_none_match and not range_header:
        headers["If-None-Match"] = if_none_match
    url = f"{base_url}/files/{quote(file_id, safe='')}?alt=media"

    client, owns_client = _get_stream_client(request.app)
    req = client.build_request("GET", url, headers=headers)
    try:
        upstream = await client.send(req, stream=True)
    except httpx.HTTPError as exc:
        if owns_client:
            await client.aclose()
        raise APIError("DRIVE_STREAM_FAILED", "连接 Google Drive 流式读取失败", 502, True) from exc

    if strip_metadata:
        try:
            if upstream.status_code != 200:
                status = upstream.status_code
                raise APIError(
                    "DRIVE_STREAM_FAILED",
                    f"读取 Drive 原生作品失败（HTTP {status}）",
                    502 if status >= 500 else status,
                )
            buf = bytearray()
            async for chunk in upstream.aiter_bytes(65536):
                buf.extend(chunk)
                if len(buf) > settings.max_upload_bytes:
                    raise APIError("PAYLOAD_TOO_LARGE", "图片超过大小限制，无法去除元数据", 413)
        except httpx.HTTPError as exc:
            raise APIError("DRIVE_STREAM_FAILED", "读取 Google Drive 图片流失败", 502, True) from exc
        finally:
            await upstream.aclose()
            if owns_client:
                await client.aclose()

        try:
            cleaned = await asyncio.to_thread(strip_image_metadata, bytes(buf), item["mime"])
        except ImageMetadataStripError as exc:
            raise APIError("IMAGE_METADATA_STRIP_FAILED", f"去除图片元数据失败：{exc}", 422) from exc

        clean_name = clean_download_filename(item["filename"])
        clean_headers["Content-Disposition"] = content_disposition_header(
            clean_name,
            disposition="attachment" if download else "inline",
        )
        return Response(cleaned, media_type=item["mime"], headers=clean_headers)

    if upstream.status_code == 304:
        not_modified_headers: dict[str, str] = {
            "Cache-Control": "private, max-age=3600",
            "Vary": "Cookie, Authorization",
        }
        etag_val = upstream.headers.get("etag")
        if etag_val:
            not_modified_headers["ETag"] = etag_val
        await upstream.aclose()
        if owns_client:
            await client.aclose()
        return Response(status_code=304, headers=not_modified_headers)

    if upstream.status_code not in {200, 206}:
        status = upstream.status_code
        await upstream.aclose()
        if owns_client:
            await client.aclose()
        raise APIError("DRIVE_STREAM_FAILED", f"读取 Drive 原生作品失败（HTTP {status}）", 502 if status >= 500 else status)

    resp_headers: dict[str, str] = {
        "Accept-Ranges": "bytes",
        "Cache-Control": "private, max-age=3600",
        "Vary": "Cookie, Authorization",
        "Content-Disposition": (
            _content_disposition(item["filename"]).replace("inline;", "attachment;", 1)
            if download
            else _content_disposition(item["filename"])
        ),
    }
    has_upstream_encoding = bool(upstream.headers.get("content-encoding"))
    for key in ("content-length", "content-range", "etag"):
        if key == "content-length" and has_upstream_encoding:
            continue
        val = upstream.headers.get(key)
        if val:
            resp_headers[key.title()] = val

    closed = False

    async def close_upstream():
        nonlocal closed
        if closed:
            return
        closed = True
        try:
            await upstream.aclose()
        finally:
            if owns_client:
                await client.aclose()

    async def body_stream():
        try:
            async for chunk in upstream.aiter_bytes(65536):
                yield chunk
        finally:
            await close_upstream()

    return StreamingResponse(
        body_stream(),
        status_code=upstream.status_code,
        media_type=item["mime"],
        headers=resp_headers,
        background=BackgroundTask(close_upstream),
    )


@router.post("/{file_id}/import-reference", status_code=201)
async def import_reference(
    file_id: str,
    body: dict,
    request: Request,
    user=Depends(current_user),
):
    if not isinstance(body, dict) or not isinstance(body.get("project_id"), str) or not body["project_id"].strip():
        raise APIError("VALIDATION_ERROR", "请提供目标项目 project_id", 422)
    asset = await asyncio.to_thread(
        native_outputs.import_native_output_as_reference,
        request.app.state.repo,
        user["id"],
        file_id,
        body["project_id"].strip(),
    )
    return ok(asset, request, 201)


@router.delete("/{file_id}")
async def delete_output(
    file_id: str,
    request: Request,
    user=Depends(current_user),
):
    result = await asyncio.to_thread(
        native_outputs.trash_native_output,
        request.app.state.repo,
        user["id"],
        file_id,
    )
    return ok(result, request)
