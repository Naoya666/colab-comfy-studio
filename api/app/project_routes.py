"""Authenticated project CRUD routes extracted from main.py."""

from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, Depends, Request
from sqlalchemy import select

from .common import APIError, current_user, get_store, ok, own
from .db import ProjectRow, utcnow
from .gpu.repository import Repository

router = APIRouter()

_TERMINAL_JOB_STATES = {
    "completed",
    "succeeded",
    "failed",
    "failed_final",
    "cancelled",
    "cancelled_local",
    "blocked",
}


def _is_visible_project(row: dict, *, include_archived: bool = False, include_system: bool = False) -> bool:
    if not include_archived and row.get("archived_at") is not None:
        return False
    if not include_system and row.get("system") is True:
        return False
    return True


def _sort_projects(rows: list[dict]) -> list[dict]:
    # Default project always appears first; others preserve created_at desc order.
    return sorted(rows, key=lambda row: 0 if row.get("is_default") is True else 1)


def _save_default_project(st, owner_id: str, target_id: str | None = None, *, patch: dict | None = None) -> dict:
    with st.admission_lock() as acquired:
        if not acquired:
            raise APIError("SCHEDULER_BUSY", "项目设置繁忙，请重试", 503, True)
        with st.db_lock(), st.Session.begin() as session:
            st.begin_immediate(session)
            rows = session.scalars(select(ProjectRow).where(ProjectRow.owner_id == owner_id).with_for_update()).all()
            changes = {**(patch or {}), "is_default": True}
            if target_id is None:
                target = ProjectRow(id=uuid.uuid4().hex, owner_id=owner_id, data=changes)
                session.add(target)
            else:
                target = next((row for row in rows if row.id == target_id), None)
                if target is None or target.data.get("archived_at") is not None:
                    raise APIError("NOT_FOUND", "项目不存在或已归档", 404)
                if target.data.get("system") is True:
                    raise APIError("PROJECT_SYSTEM_PROTECTED", "系统内部项目不能设为默认项目", 409)
                target.data = {**target.data, **changes}
                target.version += 1
                target.updated_at = utcnow()
            for row in rows:
                if row is not target and (row.data or {}).get("is_default") is True:
                    row.data = {**row.data, "is_default": False}
                    row.version += 1
                    row.updated_at = utcnow()
            session.flush()
            return st._row(target)


def _ensure_initial_project(st, owner_id: str) -> list[dict]:
    rows = [row for row in st.list("projects", owner_id) if _is_visible_project(row)]
    if rows:
        return rows
    with st.admission_lock() as acquired:
        if not acquired:
            return []
        rows = [row for row in st.list("projects", owner_id) if _is_visible_project(row)]
        if rows:
            return rows
        created = st.create("projects", owner_id, {"title": "默认项目", "atelier_project_id": "", "is_default": True})
        return [created]


@router.get("/projects")
def projects(
    request: Request,
    include_archived: bool = False,
    include_system: bool = False,
    user=Depends(current_user),
    st=Depends(get_store),
):
    rows = [
        row
        for row in st.list("projects", user["id"])
        if _is_visible_project(row, include_archived=include_archived, include_system=include_system)
    ]
    if not include_archived and not include_system and not rows:
        rows = _ensure_initial_project(st, user["id"])
    return ok(_sort_projects(rows), request)


@router.post("/projects")
def create_project(body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)):
    title = str(body.get("title", "")).strip()[:200]
    if not title:
        raise APIError("VALIDATION_ERROR", "项目名称不能为空")
    data = {"title": title, "atelier_project_id": str(body.get("atelier_project_id", ""))[:128]}
    if body.get("is_default") is True:
        created = _save_default_project(st, user["id"], patch=data)
    else:
        created = st.create("projects", user["id"], data)
    return ok(created, request, 201)


@router.patch("/projects/{project_id}")
def update_project(project_id: str, body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)):
    project = own(st, "projects", project_id, user)
    if project.get("archived_at") is not None:
        raise APIError("NOT_FOUND", "项目不存在或已归档", 404)
    has_title = "title" in body
    set_default = body.get("is_default") is True
    if not has_title and not set_default:
        raise APIError("PROJECT_INVALID", "仅支持修改本地项目名称或设为默认项目")
    patch = {}
    if has_title:
        title = str(body["title"]).strip()[:200]
        if not title:
            raise APIError("PROJECT_INVALID", "项目名不能为空")
        patch = {"title": title}
        if "atelier_project_id" in body:
            patch["atelier_project_id"] = str(body["atelier_project_id"])[:128]
    if set_default:
        updated = _save_default_project(st, user["id"], project_id, patch=patch)
    else:
        updated = st.update("projects", project_id, patch, user["id"])
    return ok(updated, request)


@router.post("/projects/{project_id}/default")
def set_default_project(project_id: str, request: Request, user=Depends(current_user), st=Depends(get_store)):
    own(st, "projects", project_id, user)
    return ok(_save_default_project(st, user["id"], project_id), request)


@router.delete("/projects/{project_id}")
def delete_project(project_id: str, request: Request, user=Depends(current_user), st=Depends(get_store)):
    project = own(st, "projects", project_id, user)
    if project.get("system") is True:
        raise APIError("PROJECT_SYSTEM_PROTECTED", "系统内部项目不能手动归档删除", 409)
    from . import creation_api

    if creation_api.config(st, user["id"]).get("project_id") == project_id:
        raise APIError(
            "PROJECT_BOUND_BY_CREATION_API",
            "该项目正被创作 API 绑定使用，请先到「创作 → API 设置」修改保存项目后再移除",
            409,
        )
    if project.get("is_default") is True:
        others = [
            row
            for row in st.list("projects", user["id"])
            if row["id"] != project_id and _is_visible_project(row)
        ]
        if others:
            raise APIError("PROJECT_IS_DEFAULT", "当前项目为默认项目，请先将其他项目设为默认后再移除", 409)
    active = [
        job
        for job in Repository(st).list("jobs", user["id"], project_id)
        if str(job.get("state") or "").lower() not in _TERMINAL_JOB_STATES
    ]
    if active:
        raise APIError("PROJECT_HAS_ACTIVE_JOBS", "该项目还有正在排队或运行的任务，请先取消后再归档删除", 409)
    st.update("projects", project_id, {"archived_at": time.time(), "is_default": False}, user["id"])
    return ok({"deleted": True, "id": project_id}, request)
