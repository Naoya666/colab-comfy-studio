"""Authenticated Studio gallery and asset CRUD/media routes extracted from main.py."""

from __future__ import annotations

import asyncio
import io
import re
import uuid
import zipfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, Response
from PIL import Image

from .batch_zip import (
    STUDIO_BATCH_ZIP_MAX_BYTES,
    STUDIO_BATCH_ZIP_MAX_ITEMS,
    build_zip_archive_filename,
    create_download_ticket,
    load_download_ticket,
    open_zip_spool,
    spooled_file_streaming_response,
    unique_zip_entry_name,
    write_zip_bytes_entry,
    write_zip_file_entry,
)
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
_PROMPT_KEY_RE = re.compile(r"prompt|caption", re.IGNORECASE)
_NEGATIVE_KEY_RE = re.compile(r"(?:^|_)negative(?:_prompt)?$", re.IGNORECASE)
_MAX_GALLERY_PROMPT_LEN = 2000
_MAX_GALLERY_MAIN_PROMPT_LEN = 1000
_MAX_GALLERY_NEGATIVE_LEN = 1000


def _clean_prompt_str(value, max_len: int) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip()
    if not text:
        return ""
    return text[:max_len]


def _pick_prompt_from_params(params: dict) -> tuple[str, str]:
    positive = ""
    negative = ""
    if not isinstance(params, dict):
        return positive, negative
    if isinstance(params.get("prompt"), str) and params["prompt"].strip():
        positive = params["prompt"].strip()
    if isinstance(params.get("negative_prompt"), str) and params["negative_prompt"].strip():
        negative = params["negative_prompt"].strip()
    for key, val in params.items():
        if not isinstance(val, str) or not val.strip():
            continue
        k = str(key)
        if not negative and _NEGATIVE_KEY_RE.search(k):
            negative = val.strip()
        elif not positive and _PROMPT_KEY_RE.search(k) and "negative" not in k.lower():
            positive = val.strip()
    return positive, negative


def _extract_job_search_fields(job: dict) -> dict:
    """Extract compact prompt/model/seed metadata for gallery search without exposing internal handles."""
    snapshot = job.get("snapshot") if isinstance(job.get("snapshot"), dict) else {}
    resolved_params = snapshot.get("resolved_params") if isinstance(snapshot.get("resolved_params"), dict) else {}
    raw_params = job.get("params") if isinstance(job.get("params"), dict) else {}
    comp = job.get("prompt_composition") if isinstance(job.get("prompt_composition"), dict) else {}
    comp_pos = comp.get("positive") if isinstance(comp.get("positive"), dict) else {}
    comp_neg = comp.get("negative") if isinstance(comp.get("negative"), dict) else {}

    pos, neg = _pick_prompt_from_params(resolved_params)
    raw_pos, raw_neg = _pick_prompt_from_params(raw_params)
    if not pos:
        pos = raw_pos
    if not neg:
        neg = raw_neg

    main_prompt = _clean_prompt_str(comp_pos.get("main"), _MAX_GALLERY_MAIN_PROMPT_LEN)
    if not pos and main_prompt:
        pos = main_prompt
    if not neg:
        neg = _clean_prompt_str(comp_neg.get("main"), _MAX_GALLERY_NEGATIVE_LEN)

    out: dict = {}
    clean_pos = _clean_prompt_str(pos, _MAX_GALLERY_PROMPT_LEN)
    clean_neg = _clean_prompt_str(neg, _MAX_GALLERY_NEGATIVE_LEN)
    if clean_pos:
        out["prompt"] = clean_pos
    if main_prompt:
        out["main_prompt"] = main_prompt
    if clean_neg:
        out["negative_prompt"] = clean_neg

    model_id = job.get("model_id")
    if isinstance(model_id, str) and model_id.strip():
        out["model_id"] = model_id.strip()

    seed = job.get("resolved_seed")
    if seed is None:
        seed = resolved_params.get("seed")
    if seed is None:
        seed = raw_params.get("seed")
    if isinstance(seed, (int, float)) and not isinstance(seed, bool):
        out["resolved_seed"] = int(seed)
    elif isinstance(seed, str) and seed.strip().lstrip("-").isdigit():
        out["resolved_seed"] = int(seed.strip())
    return out


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
            for field_key, field_val in _extract_job_search_fields(job).items():
                if item.get(field_key) is None:
                    item[field_key] = field_val
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


def _resolve_studio_batch_assets(
    st,
    user: dict,
    raw_ids,
    *,
    strip_metadata: bool = False,
) -> tuple[list[tuple[dict, Path, int]], int, int]:
    if (
        not isinstance(raw_ids, list)
        or not raw_ids
        or len(raw_ids) > STUDIO_BATCH_ZIP_MAX_ITEMS
        or any(not isinstance(item, str) or not item.strip() for item in raw_ids)
    ):
        raise APIError("VALIDATION_ERROR", f"请提供 1~{STUDIO_BATCH_ZIP_MAX_ITEMS} 个待下载的素材编号")
    unique_ids = list(dict.fromkeys(item.strip() for item in raw_ids))
    asset_map = st.get_many("assets", unique_ids, user["id"]) if hasattr(st, "get_many") else {}
    assets = [asset_map[aid] if aid in asset_map else own(st, "assets", aid, user) for aid in unique_ids]
    resolved: list[tuple[dict, Path, int]] = []
    total_bytes = 0
    strippable_count = 0
    for asset in assets:
        path = asset_path(asset)
        try:
            file_size = path.stat().st_size
        except FileNotFoundError as exc:
            raise APIError("NOT_FOUND", f"素材文件不存在：{asset.get('filename') or asset['id']}", 404) from exc
        mime = str(asset.get("mime") or "")
        strippable = is_strippable_image_mime(mime)
        if strippable:
            strippable_count += 1
            if strip_metadata and file_size > settings.max_upload_bytes:
                raise APIError("PAYLOAD_TOO_LARGE", "存在超过大小限制的图片，无法去除元数据后打包", 413)
        total_bytes += file_size
        if total_bytes > STUDIO_BATCH_ZIP_MAX_BYTES:
            raise APIError(
                "PAYLOAD_TOO_LARGE",
                f"所选文件总大小超过单次打包上限（{STUDIO_BATCH_ZIP_MAX_BYTES // (1024 * 1024)} MB），请分批下载",
                413,
            )
        resolved.append((asset, path, file_size))
    return resolved, total_bytes, strippable_count


def _build_studio_batch_zip(
    resolved: list[tuple[dict, Path, int]],
    *,
    strip_metadata: bool,
):
    spool = open_zip_spool()
    try:
        used_lower: set[str] = set()
        with zipfile.ZipFile(spool, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
            for asset, path, _ in resolved:
                mime = str(asset.get("mime") or "")
                strippable = is_strippable_image_mime(mime)
                arcname = unique_zip_entry_name(
                    asset.get("filename"),
                    used_lower,
                    strip_metadata=strip_metadata,
                    strippable=strippable,
                    fallback=str(asset.get("id") or "asset"),
                )
                if strip_metadata and strippable:
                    try:
                        cleaned = strip_image_metadata(path.read_bytes(), mime)
                    except ImageMetadataStripError as exc:
                        raise APIError("IMAGE_METADATA_STRIP_FAILED", f"去除图片元数据失败：{exc}", 422) from exc
                    write_zip_bytes_entry(zf, arcname, cleaned)
                else:
                    write_zip_file_entry(zf, arcname, path)
        return spool
    except BaseException:
        spool.close()
        raise


@assets_router.post("/api/assets/batch-download")
def create_batch_download_ticket(body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)):
    if not isinstance(body, dict):
        raise APIError("VALIDATION_ERROR", "请求体格式无效")
    raw_strip = body.get("strip_metadata", False)
    if not isinstance(raw_strip, bool):
        raise APIError("VALIDATION_ERROR", "strip_metadata 必须为布尔值")
    strip_metadata = raw_strip
    resolved, total_bytes, strippable_count = _resolve_studio_batch_assets(
        st,
        user,
        body.get("asset_ids"),
        strip_metadata=strip_metadata,
    )
    effective_strip = strip_metadata and strippable_count > 0
    filename = build_zip_archive_filename("studio-gallery", strip_metadata=effective_strip)
    ticket = create_download_ticket(
        st,
        user["id"],
        "studio_batch_zip",
        {
            "asset_ids": [asset["id"] for asset, _, _ in resolved],
            "strip_metadata": strip_metadata,
            "filename": filename,
        },
    )
    ticket_id = ticket["ticket_id"]
    return ok(
        {
            "ticket_id": ticket_id,
            "download_url": f"/api/assets/batch-download/{ticket_id}",
            "filename": filename,
            "item_count": len(resolved),
            "total_bytes": total_bytes,
            "strippable_count": strippable_count,
            "skipped_strip_count": len(resolved) - strippable_count if strip_metadata else 0,
            "strip_metadata": strip_metadata,
            "expires_at": ticket["expires_at"],
        },
        request,
    )


@assets_router.get("/api/assets/batch-download/{ticket_id}")
async def download_batch_zip(ticket_id: str, user=Depends(current_user), st=Depends(get_store)):
    payload = await asyncio.to_thread(load_download_ticket, st, user["id"], ticket_id, "studio_batch_zip")
    strip_metadata = bool(payload.get("strip_metadata", False))
    filename = str(payload.get("filename") or build_zip_archive_filename("studio-gallery", strip_metadata=strip_metadata))
    resolved, _, _ = await asyncio.to_thread(
        _resolve_studio_batch_assets,
        st,
        user,
        payload.get("asset_ids"),
        strip_metadata=strip_metadata,
    )
    spool = await asyncio.to_thread(_build_studio_batch_zip, resolved, strip_metadata=strip_metadata)
    return spooled_file_streaming_response(spool, filename=filename, media_type="application/zip")


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
