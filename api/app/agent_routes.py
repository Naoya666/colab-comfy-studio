"""White-listed Agent API routes extracted from main.py."""

from __future__ import annotations

import os
import secrets

from fastapi import APIRouter, Depends, Request

from .admission import admit_job, repo_registry
from .common import APIError, current_user, get_store, ok, own
from .gpu.ccu import usage_summary
from .gpu.scheduler import Scheduler
from .job_schemas import JobInput
from .presenters import public_asset, public_job

router = APIRouter()

AGENT_TOOLS = {
    "workflow.list",
    "workflow.import_draft",
    "workflow.validate",
    "model.list",
    "model.inspect",
    "model.register_draft",
    "job.create",
    "job.get",
    "job.cancel",
    "job.list_logs_redacted",
    "storage.list_project_files",
    "usage.inspect",
}


def agent_user(request: Request):
    supplied = request.headers.get("Authorization", "").removeprefix("Bearer ")
    expected = os.getenv("GPU_AGENT_TOKEN", "")
    owner = os.getenv("GPU_AGENT_OWNER_ID", "")
    if expected and owner and supplied and secrets.compare_digest(supplied, expected):
        return {"id": owner, "role": "agent"}
    return current_user(request)


@router.get("/agent/tools")
def agent_tools(request: Request):
    agent_user(request)
    return ok(sorted(AGENT_TOOLS), request)


@router.post("/agent/invoke")
def agent_invoke(body: dict, request: Request, st=Depends(get_store)):
    user = agent_user(request)
    tool, args = body.get("tool"), body.get("arguments", {})
    if tool not in AGENT_TOOLS or not isinstance(args, dict):
        raise APIError("TOOL_FORBIDDEN", "Agent 不具备该工具权限；预算、发布与直接开机不可绕过", 403)
    repo, registry = repo_registry(st, user)
    if tool == "usage.inspect":
        result = usage_summary(repo, user["id"])
    elif tool in {"workflow.list", "model.list"}:
        result = registry.workflows(user["id"]) if tool == "workflow.list" else registry.models(user["id"])
    elif tool == "model.inspect":
        result = next((m for m in registry.models(user["id"]) if m["model_id"] == args.get("model_id")), None)
    elif tool == "model.register_draft":
        if not args.get("model_id"):
            raise APIError("MODEL_INVALID", "缺少模型 ID")
        result = repo.create(
            "models",
            user["id"],
            {
                k: v
                for k, v in args.items()
                if k in {"model_id", "display_name", "source", "revision", "license", "resource_profile"}
            }
            | {"enabled": False, "is_mock": False, "disabled_reason": "待管理员审查"},
        )
    elif tool == "workflow.import_draft":
        result = registry.import_workflow(user["id"], args)
    elif tool == "workflow.validate":
        result = registry.validate(user["id"], args.get("id"))
    elif tool == "job.create":
        key = args.pop("idempotency_key", None)
        result = public_job(admit_job(st, user, JobInput(**args).model_dump(), key))
    elif tool in {"job.get", "job.cancel", "job.list_logs_redacted"}:
        job = own(repo, "jobs", args.get("id"), user)
        if tool == "job.cancel":
            job = Scheduler(st).cancel(user["id"], job["id"])
        result = (
            [e for e in repo.list("events", user["id"]) if e.get("job_id") == job["id"]]
            if tool.endswith("redacted")
            else public_job(job)
        )
    elif tool == "storage.list_project_files":
        own(st, "projects", args.get("project_id"), user)
        result = [public_asset(a) for a in st.list("assets", user["id"], args["project_id"])]
    else:
        raise APIError("TOOL_FORBIDDEN", "工具不可用", 403)
    return ok(result, request)
