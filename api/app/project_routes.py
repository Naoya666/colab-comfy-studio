"""Authenticated project CRUD routes extracted from main.py."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Request

from .common import APIError, current_user, get_store, ok, own
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


@router.get("/projects")
def projects(
    request: Request,
    include_archived: bool = False,
    user=Depends(current_user),
    st=Depends(get_store),
):
    rows = st.list("projects", user["id"])
    if not include_archived:
        rows = [row for row in rows if row.get("archived_at") is None]
    return ok(rows, request)


@router.post("/projects")
def create_project(body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)):
    title = str(body.get("title", "")).strip()[:200]
    if not title:
        raise APIError("VALIDATION_ERROR", "项目名称不能为空")
    return ok(
        st.create(
            "projects",
            user["id"],
            {"title": title, "atelier_project_id": str(body.get("atelier_project_id", ""))[:128]},
        ),
        request,
        201,
    )


@router.patch("/projects/{project_id}")
def update_project(project_id: str, body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)):
    own(st, "projects", project_id, user)
    if "title" not in body:
        raise APIError("PROJECT_INVALID", "仅支持修改本地项目名称")
    title = str(body["title"]).strip()[:200]
    if not title:
        raise APIError("PROJECT_INVALID", "项目名不能为空")
    patch = {"title": title}
    if "atelier_project_id" in body:
        patch["atelier_project_id"] = str(body["atelier_project_id"])[:128]
    return ok(st.update("projects", project_id, patch, user["id"]), request)


@router.delete("/projects/{project_id}")
def delete_project(project_id: str, request: Request, user=Depends(current_user), st=Depends(get_store)):
    own(st, "projects", project_id, user)
    active = [
        job
        for job in Repository(st).list("jobs", user["id"], project_id)
        if str(job.get("state") or "").lower() not in _TERMINAL_JOB_STATES
    ]
    if active:
        raise APIError("PROJECT_HAS_ACTIVE_JOBS", "该项目还有正在排队或运行的任务，请先取消后再归档删除", 409)
    st.update("projects", project_id, {"archived_at": time.time()}, user["id"])
    return ok({"deleted": True, "id": project_id}, request)
