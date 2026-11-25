"""Authenticated Studio gallery and asset CRUD/media routes extracted from main.py."""

from __future__ import annotations

import asyncio
import io
import re
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, Response
from PIL import Image

from .common import APIError, current_user, get_store, ok, own
from .config import settings
from .gpu.repository import Repository
from .image_metadata import (
    ImageMetadataStripError,
    clean_download_filename,
    content_disposition_header,
    is_strippable_image_mime,
    strip_image_metadata,
)
from .presenters import public_asset
from .storage import asset_path, delete_asset_file, save_asset

gallery_router = APIRouter()
assets_router = APIRouter()

_SAFE_TOKEN_RE = re.compile(r"[^A-Za-z0-9_-]+")


def _etag_matches(header: str | None, etag: str) -> bool:
    if not header:
        return False
    target = etag.removeprefix("W/").strip()
    for part in header.split(","):
        token = part.strip()
        if token == "*" or token == etag or token.removeprefix("W/").strip() == target:
            return True
    return False


def _asset_thumb_dir() -> Path:
    root = (settings.data_dir / "asset-thumb-cache").resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _asset_thumb_path(asset: dict) -> tuple[Path, str]:
    safe_id = _SAFE_TOKEN_RE.sub("", str(asset.get("id") or ""))[:64] or "asset"
    raw_rev = str(asset.get("sha256") or asset.get("updated_at") or "v0")
    safe_rev = _SAFE_TOKEN_RE.sub("", raw_rev)[:16] or "v0"
    root = _asset_thumb_dir()
    target = (root / f"{safe_id}-{safe_rev}.webp").resolve()
    if not target.is_relative_to(root):
        raise ValueError("非法缩略图路径")
    etag = f'W/"asset-thumb-{safe_id}-{safe_rev}"'
    return target, etag


def _delete_asset_thumbs(asset: dict) -> None:
    safe_id = _SAFE_TOKEN_RE.sub("", str(asset.get("id") or ""))[:64]
    if not safe_id:
        return
    root = (settings.data_dir / "asset-thumb-cache").resolve()
    if not root.is_dir():
        return
    for item in root.glob(f"{safe_id}-*.webp"):
        item.unlink(missing_ok=True)


@gallery_router.get("/gallery")
def gallery(
    request: Request, project_id: str | None = None, user=Depends(current_user), st=Depends(get_store)
):
    if project_id:
        own(st, "projects", project_id, user)
    raw_assets = st.list("assets", user["id"], project_id)
    sourced = [
        (a, source_job_id)
        for a in raw_assets
        if (source_job_id := (a.get("source_job_id") or a.get("gpu_job_id")))
    ]
    # Load only jobs actually referenced by gallery assets instead of scanning all historical jobs.
    job_map = Repository(st).list_jobs_by_ids(user["id"], [jid for _, jid in sourced]) if sourced else {}
    result = []
    for a, source_job_id in sourced:
        item = public_asset(a)
        job = job_map.get(source_job_id)
        if job:
            if job.get("batch_id") and not item.get("batch_id"):
                item["batch_id"] = job["batch_id"]
            if job.get("workflow_key") and not item.get("workflow_key"):
                item["workflow_key"] = job["workflow_key"]
            if job.get("workflow_version") is not None and item.get("workflow_version") is None:
                item["workflow_version"] = job["workflow_version"]
            if job.get("gpu_reported") and not item.get("gpu_reported"):
                item["gpu_reported"] = job["gpu_reported"]
            if job.get("provider") == "mock" and item.get("mock") is None:
                item["mock"] = True
        result.append(item)
    return ok(result, request)


@assets_router.get("/api/assets")
def reference_assets(
    request: Request, project_id: str | None = None, user=Depends(current_user), st=Depends(get_store)
):
    if project_id:
        own(st, "projects", project_id, user)
    rows = [
        public_asset(a)
        for a in st.list("assets", user["id"], project_id)
        if not a.get("mock")
        and not (a.get("metadata") or {}).get("mock")
        and str(a.get("mime", "")).startswith(("image/", "video/", "audio/"))
    ]
    return ok(rows, request)


@assets_router.post("/api/assets")
async def upload_asset(
    request: Request,
    project_id: str = Form(...),
    file: UploadFile = File(...),
    user=Depends(current_user),
    st=Depends(get_store),
):
    await asyncio.to_thread(own, st, "projects", project_id, user)
    content = await file.read(settings.max_upload_bytes + 1)
    saved = await asyncio.to_thread(save_asset, st, user["id"], project_id, content, file.filename or "upload.bin")
    return ok(public_asset(saved), request, 201)


@assets_router.get("/api/assets/{asset_id}/download")
@assets_router.get("/api/assets/{asset_id}/preview")
def download_asset(
    asset_id: str,
    request: Request,
    strip_metadata: bool = False,
    user=Depends(current_user),
    st=Depends(get_store),
):
    asset = own(st, "assets", asset_id, user)
    is_preview = request.url.path.endswith("/preview")
    headers = {
        "Cache-Control": "private, max-age=86400",
        "Vary": "Cookie, Authorization",
    }
    sha = asset.get("sha256")
    if strip_metadata:
        mime = str(asset.get("mime") or "")
        if not is_strippable_image_mime(mime):
            raise APIError("STRIP_METADATA_UNSUPPORTED", "仅支持对图片去除元数据", 422)
        if isinstance(sha, str) and sha:
            etag = f'"{sha[:32]}-clean"'
            headers["ETag"] = etag
            if _etag_matches(request.headers.get("if-none-match"), etag):
                return Response(status_code=304, headers=headers)
        path = asset_path(asset)
        try:
            file_size = path.stat().st_size
        except FileNotFoundError as exc:
            raise APIError("NOT_FOUND", "素材文件不存在", 404) from exc
        if file_size > settings.max_upload_bytes:
            raise APIError("PAYLOAD_TOO_LARGE", "图片超过大小限制，无法去除元数据", 413)
        try:
            cleaned = strip_image_metadata(path.read_bytes(), mime)
        except ImageMetadataStripError as exc:
            raise APIError("IMAGE_METADATA_STRIP_FAILED", f"去除图片元数据失败：{exc}", 422) from exc
        clean_name = clean_download_filename(asset.get("filename") or asset_id)
        headers["Content-Disposition"] = content_disposition_header(
            clean_name,
            disposition="inline" if is_preview else "attachment",
        )
        return Response(cleaned, media_type=mime, headers=headers)

    if isinstance(sha, str) and sha:
        etag = f'"{sha[:32]}"'
        headers["ETag"] = etag
        if _etag_matches(request.headers.get("if-none-match"), etag):
            return Response(status_code=304, headers=headers)
    return FileResponse(
        asset_path(asset),
        media_type=asset["mime"],
        filename=asset["filename"],
        content_disposition_type="inline" if is_preview else "attachment",
        headers=headers,
    )


@assets_router.get("/api/assets/{asset_id}/thumbnail")
def thumbnail(asset_id: str, request: Request, user=Depends(current_user), st=Depends(get_store)):
    asset = own(st, "assets", asset_id, user)
    if not asset["mime"].startswith("image/"):
        raise APIError("THUMBNAIL_UNAVAILABLE", "非图片素材不提供图片缩略图", 404)
    cache_path, etag = _asset_thumb_path(asset)
    headers = {
        "Cache-Control": "private, max-age=31536000, immutable",
        "ETag": etag,
        "Vary": "Cookie, Authorization",
    }
    if _etag_matches(request.headers.get("if-none-match"), etag):
        return Response(status_code=304, headers=headers)
    if cache_path.is_file():
        return Response(cache_path.read_bytes(), media_type="image/webp", headers=headers)
    with Image.open(asset_path(asset)) as image:
        image.thumbnail((480, 480))
        target = io.BytesIO()
        mode = "RGBA" if image.mode in {"RGBA", "LA"} else "RGB"
        image.convert(mode).save(target, "WEBP", quality=80)
        webp = target.getvalue()
    tmp = cache_path.with_name(f"{cache_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_bytes(webp)
        tmp.replace(cache_path)
    except OSError:
        tmp.unlink(missing_ok=True)
    return Response(webp, media_type="image/webp", headers=headers)


def _require_target_project(st, project_id: str, user: dict) -> dict:
    if not isinstance(project_id, str) or not project_id.strip():
        raise APIError("VALIDATION_ERROR", "请指定目标项目")
    project = own(st, "projects", project_id.strip(), user)
    if project.get("archived_at") is not None:
        raise APIError("PROJECT_ARCHIVED", "目标项目已归档，无法移入作品", 409)
    if project.get("system") is True:
        raise APIError("PROJECT_SYSTEM_PROTECTED", "不能将作品移入系统内部项目", 409)
    return project


@assets_router.patch("/api/assets/{asset_id}")
def favorite_asset(
    asset_id: str, body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)
):
    own(st, "assets", asset_id, user)
    if not isinstance(body, dict):
        raise APIError("VALIDATION_ERROR", "请求体格式无效")
    patch: dict = {}
    if "favorite" in body:
        if not isinstance(body.get("favorite"), bool):
            raise APIError("VALIDATION_ERROR", "favorite必须为布尔值")
        patch["favorite"] = body["favorite"]
    if "project_id" in body:
        target = _require_target_project(st, body.get("project_id"), user)
        patch["project_id"] = target["id"]
    if not patch:
        raise APIError("VALIDATION_ERROR", "favorite必须为布尔值")
    return ok(public_asset(st.update("assets", asset_id, patch, user["id"])), request)


@assets_router.post("/api/assets/batch-move")
def batch_move_assets(body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)):
    raw_ids = body.get("asset_ids") if isinstance(body, dict) else None
    target_project_id = body.get("target_project_id") if isinstance(body, dict) else None
    if (
        not isinstance(raw_ids, list)
        or not raw_ids
        or len(raw_ids) > 200
        or any(not isinstance(item, str) for item in raw_ids)
    ):
        raise APIError("VALIDATION_ERROR", "请提供 1~200 个待移动的素材编号")
    target = _require_target_project(st, target_project_id, user)
    unique_ids = list(dict.fromkeys(raw_ids))
    asset_map = st.get_many("assets", unique_ids, user["id"]) if hasattr(st, "get_many") else {}
    assets = [asset_map[aid] if aid in asset_map else own(st, "assets", aid, user) for aid in unique_ids]
    moved_ids = []
    for asset in assets:
        if asset.get("project_id") != target["id"]:
            st.update("assets", asset["id"], {"project_id": target["id"]}, user["id"])
        moved_ids.append(asset["id"])
    return ok({"moved": len(moved_ids), "ids": moved_ids, "target_project_id": target["id"]}, request)


@assets_router.delete("/api/assets/{asset_id}")
def delete_asset(asset_id: str, request: Request, user=Depends(current_user), st=Depends(get_store)):
    asset = own(st, "assets", asset_id, user)
    delete_asset_file(asset)
    _delete_asset_thumbs(asset)
    st.delete("assets", asset_id, user["id"])
    return ok({"deleted": True}, request)


@assets_router.post("/api/assets/batch-delete")
def batch_delete_assets(body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)):
    raw_ids = body.get("asset_ids") if isinstance(body, dict) else None
    if not isinstance(raw_ids, list) or not raw_ids or len(raw_ids) > 200 or any(not isinstance(item, str) for item in raw_ids):
        raise APIError("VALIDATION_ERROR", "请提供 1~200 个待删除的素材编号")
    unique_ids = list(dict.fromkeys(raw_ids))
    asset_map = st.get_many("assets", unique_ids, user["id"]) if hasattr(st, "get_many") else {}
    assets = [asset_map[aid] if aid in asset_map else own(st, "assets", aid, user) for aid in unique_ids]
    deleted_ids = []
    for asset in assets:
        delete_asset_file(asset)
        _delete_asset_thumbs(asset)
        st.delete("assets", asset["id"], user["id"])
        deleted_ids.append(asset["id"])
    return ok({"deleted": len(deleted_ids), "ids": deleted_ids}, request)
