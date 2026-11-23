"""Authenticated routes for browsing and acting on Native ComfyUI Drive outputs."""

from __future__ import annotations

import asyncio
import hashlib
from urllib.parse import quote

import httpx
from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response, StreamingResponse

from .common import APIError, current_user, ok
from .gpu import native_outputs

router = APIRouter(prefix="/api/gpu/native/outputs")


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
    ascii_name = filename.encode("ascii", "replace").decode("ascii").replace('"', "_")
    utf8_name = quote(filename, safe="")
    return f'inline; filename="{ascii_name}"; filename*=UTF-8\'\'{utf8_name}'


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
        "Cache-Control": "private, max-age=86400",
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
    user=Depends(current_user),
):
    repo = request.app.state.repo
    item, storage, owns = await asyncio.to_thread(
        native_outputs.resolve_output_item, repo, user["id"], file_id,
    )
    try:
        token = await asyncio.to_thread(storage.google.access_token, "drive")
        base_url = storage.base_url
    finally:
        if owns and storage._owns_client:
            storage.client.close()

    headers = {"Authorization": "Bearer " + token}
    range_header = request.headers.get("range")
    if range_header:
        headers["Range"] = range_header
    if_none_match = request.headers.get("if-none-match")
    if if_none_match and not range_header:
        headers["If-None-Match"] = if_none_match
    url = f"{base_url}/files/{quote(file_id, safe='')}?alt=media"

    client = httpx.AsyncClient(timeout=httpx.Timeout(connect=15.0, read=120.0, write=30.0, pool=15.0))
    req = client.build_request("GET", url, headers=headers)
    try:
        upstream = await client.send(req, stream=True)
    except httpx.HTTPError as exc:
        await client.aclose()
        raise APIError("DRIVE_STREAM_FAILED", "连接 Google Drive 流式读取失败", 502, True) from exc

    if upstream.status_code == 304:
        not_modified_headers: dict[str, str] = {
            "Cache-Control": "private, max-age=3600",
            "Vary": "Cookie, Authorization",
        }
        etag_val = upstream.headers.get("etag")
        if etag_val:
            not_modified_headers["ETag"] = etag_val
        await upstream.aclose()
        await client.aclose()
        return Response(status_code=304, headers=not_modified_headers)

    if upstream.status_code not in {200, 206}:
        status = upstream.status_code
        await upstream.aclose()
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
    for key in ("content-length", "content-range", "etag"):
        val = upstream.headers.get(key)
        if val:
            resp_headers[key.title()] = val

    async def body_stream():
        try:
            async for chunk in upstream.aiter_bytes(65536):
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    return StreamingResponse(
        body_stream(),
        status_code=upstream.status_code,
        media_type=item["mime"],
        headers=resp_headers,
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
