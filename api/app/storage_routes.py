"""Authenticated routes for local VPS/host storage breakdown, safe cleanup, VACUUM, and DB snapshot."""

from __future__ import annotations

import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, Request

from .batch_zip import (
    create_download_ticket,
    load_download_ticket,
    spooled_file_streaming_response,
)
from .common import APIError, admin, current_user, get_store, ok
from .config import settings
from .local_storage import (
    _sqlite_storage_info,
    build_sqlite_snapshot_spool,
    execute_local_storage_cleanup,
    get_local_storage_overview,
    preview_local_storage_cleanup,
    vacuum_local_sqlite,
)

router = APIRouter(prefix="/api/gpu/storage/local", tags=["local-storage"])


def _snapshot_filename() -> str:
    try:
        tz = ZoneInfo(settings.studio_timezone)
    except Exception:
        tz = ZoneInfo("UTC")
    stamp = datetime.now(tz).strftime("%Y%m%d-%H%M%S")
    return f"studio-db-{stamp}.db"


@router.get("")
async def local_storage_overview(
    request: Request,
    user=Depends(current_user),
    st=Depends(get_store),
):
    data = await asyncio.to_thread(get_local_storage_overview, st, user["id"])
    return ok(data, request)


@router.post("/cleanup/preview")
async def local_storage_cleanup_preview(
    body: dict,
    request: Request,
    user=Depends(current_user),
    st=Depends(get_store),
):
    admin(user)
    if not isinstance(body, dict):
        raise APIError("VALIDATION_ERROR", "请求体格式无效")
    result = await asyncio.to_thread(
        preview_local_storage_cleanup,
        st,
        user["id"],
        action=str(body.get("action") or ""),
        older_than_days=body.get("older_than_days"),
        project_id=body.get("project_id"),
    )
    return ok(result, request)


@router.post("/cleanup")
async def local_storage_cleanup(
    body: dict,
    request: Request,
    user=Depends(current_user),
    st=Depends(get_store),
):
    admin(user)
    if not isinstance(body, dict) or body.get("confirmed") is not True:
        raise APIError("CONFIRMATION_REQUIRED", "清理操作必须明确确认 confirmed=true", 422)
    result = await asyncio.to_thread(
        execute_local_storage_cleanup,
        st,
        user["id"],
        action=str(body.get("action") or ""),
        older_than_days=body.get("older_than_days"),
        project_id=body.get("project_id"),
    )
    return ok(result, request)


@router.post("/vacuum")
async def local_storage_vacuum(
    body: dict,
    request: Request,
    user=Depends(current_user),
    st=Depends(get_store),
):
    admin(user)
    if not isinstance(body, dict) or body.get("confirmed") is not True:
        raise APIError("CONFIRMATION_REQUIRED", "数据库空间回收必须明确确认 confirmed=true", 422)
    result = await asyncio.to_thread(vacuum_local_sqlite, st, user["id"])
    return ok(result, request)


@router.post("/snapshot")
async def create_sqlite_snapshot_ticket(
    request: Request,
    user=Depends(current_user),
    st=Depends(get_store),
):
    admin(user)
    db_info = await asyncio.to_thread(_sqlite_storage_info, st)
    if not db_info.get("supported"):
        raise APIError(
            "SQLITE_ONLY",
            str(db_info.get("message") or "当前数据库不支持网页快照下载"),
            409,
        )
    filename = _snapshot_filename()
    ticket = await asyncio.to_thread(
        create_download_ticket,
        st,
        user["id"],
        "sqlite_snapshot",
        {"filename": filename},
        ttl_seconds=120,
    )
    ticket_id = ticket["ticket_id"]
    return ok(
        {
            "ticket_id": ticket_id,
            "download_url": f"/api/gpu/storage/local/snapshot/{ticket_id}",
            "filename": filename,
            "estimated_bytes": db_info.get("db_bytes", 0),
            "expires_at": ticket["expires_at"],
            "notice": "仅包含 SQLite 一致性快照（其中凭据已加密）；不包含 .env、ENCRYPTION_KEY 与本地图片/视频文件，请另行安全保管主密钥。",
        },
        request,
    )


@router.get("/snapshot/{ticket_id}")
async def download_sqlite_snapshot(
    ticket_id: str,
    user=Depends(current_user),
    st=Depends(get_store),
):
    admin(user)
    payload = await asyncio.to_thread(load_download_ticket, st, user["id"], ticket_id, "sqlite_snapshot")
    filename = str(payload.get("filename") or _snapshot_filename())
    spool = await asyncio.to_thread(build_sqlite_snapshot_spool, st)
    return spooled_file_streaming_response(
        spool,
        filename=filename,
        media_type="application/x-sqlite3",
    )
