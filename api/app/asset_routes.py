"""Authenticated Studio gallery and asset CRUD/media routes extracted from main.py."""

from __future__ import annotations

import io

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, Response
from PIL import Image

from .common import APIError, current_user, get_store, ok, own
from .config import settings
from .gpu.repository import Repository
from .presenters import public_asset
from .storage import asset_path, delete_asset_file, save_asset

gallery_router = APIRouter()
assets_router = APIRouter()


@gallery_router.get("/gallery")
def gallery(
    request: Request, project_id: str | None = None, user=Depends(current_user), st=Depends(get_store)
):
    if project_id:
        own(st, "projects", project_id, user)
    job_map = {job["id"]: job for job in Repository(st).list("jobs", user["id"], project_id)}
    result = []
    for a in st.list("assets", user["id"], project_id):
        source_job_id = a.get("source_job_id") or a.get("gpu_job_id")
        if not source_job_id:
            continue
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
def reference_assets(request: Request, project_id: str, user=Depends(current_user), st=Depends(get_store)):
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
    own(st, "projects", project_id, user)
    content = await file.read(settings.max_upload_bytes + 1)
    return ok(
        public_asset(save_asset(st, user["id"], project_id, content, file.filename or "upload.bin")),
        request,
        201,
    )


@assets_router.get("/api/assets/{asset_id}/download")
@assets_router.get("/api/assets/{asset_id}/preview")
def download_asset(asset_id: str, user=Depends(current_user), st=Depends(get_store)):
    asset = own(st, "assets", asset_id, user)
    return FileResponse(asset_path(asset), media_type=asset["mime"], filename=asset["filename"])


@assets_router.get("/api/assets/{asset_id}/thumbnail")
def thumbnail(asset_id: str, user=Depends(current_user), st=Depends(get_store)):
    asset = own(st, "assets", asset_id, user)
    if not asset["mime"].startswith("image/"):
        raise APIError("THUMBNAIL_UNAVAILABLE", "非图片素材不提供图片缩略图", 404)
    with Image.open(asset_path(asset)) as image:
        image.thumbnail((480, 480))
        target = io.BytesIO()
        image.convert("RGB").save(target, "WEBP", quality=80)
    return Response(target.getvalue(), media_type="image/webp")


@assets_router.patch("/api/assets/{asset_id}")
def favorite_asset(
    asset_id: str, body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)
):
    own(st, "assets", asset_id, user)
    if not isinstance(body.get("favorite"), bool):
        raise APIError("VALIDATION_ERROR", "favorite必须为布尔值")
    return ok(
        public_asset(st.update("assets", asset_id, {"favorite": body["favorite"]}, user["id"])), request
    )


@assets_router.delete("/api/assets/{asset_id}")
def delete_asset(asset_id: str, request: Request, user=Depends(current_user), st=Depends(get_store)):
    asset = own(st, "assets", asset_id, user)
    delete_asset_file(asset)
    st.delete("assets", asset_id, user["id"])
    return ok({"deleted": True}, request)


@assets_router.post("/api/assets/batch-delete")
def batch_delete_assets(body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)):
    raw_ids = body.get("asset_ids") if isinstance(body, dict) else None
    if not isinstance(raw_ids, list) or not raw_ids or len(raw_ids) > 200 or any(not isinstance(item, str) for item in raw_ids):
        raise APIError("VALIDATION_ERROR", "请提供 1~200 个待删除的素材编号")
    unique_ids = list(dict.fromkeys(raw_ids))
    assets = [own(st, "assets", aid, user) for aid in unique_ids]
    deleted_ids = []
    for asset in assets:
        delete_asset_file(asset)
        st.delete("assets", asset["id"], user["id"])
        deleted_ids.append(asset["id"])
    return ok({"deleted": len(deleted_ids), "ids": deleted_ids}, request)
