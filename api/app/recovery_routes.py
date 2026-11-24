"""HTTP API routes for GPU auto-recovery visibility and safe cancellation."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from .common import current_user, ok
from .gpu import recovery
from .gpu.scheduler import Scheduler

router = APIRouter(prefix="/api/gpu/recovery")


@router.get("/status")
def recovery_status(request: Request, user=Depends(current_user)):
    repo = request.app.state.repo
    response = ok(recovery.status_summary(repo, user["id"]), request)
    response.headers["Cache-Control"] = "no-store"
    return response


@router.post("/{plan_id}/cancel")
def cancel_recovery(plan_id: str, request: Request, user=Depends(current_user)):
    repo = request.app.state.repo
    result = recovery.cancel_plan(
        repo,
        user["id"],
        plan_id,
        scheduler=Scheduler(repo),
    )
    response = ok(result, request)
    response.headers["Cache-Control"] = "no-store"
    return response
