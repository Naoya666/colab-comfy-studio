"""Independent personal AI Studio API."""

from __future__ import annotations

import copy
import io
import json
import os
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, File, Form, Header, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from . import __version__
from .account_routes import router as account_router
from .admission import _drive, admit_job, repo_registry
from .atelier_routes import router as atelier_router
from .batch_routes import build_batch_router
from .common import APIError, admin, current_user, envelope, get_store, ok, own
from .compute_routes import _require_account
from .compute_routes import router as compute_router
from .config import settings
from .creation_api import CreationHostMiddleware
from .creation_api import admin_router as creation_api_router
from .db import Store
from .gpu import loras as lora_registry
from .gpu.compute import GpuConfig, MockColabProvider
from .gpu.google import GoogleError
from .gpu.manifest import bind_parameters, digest, resolve_parameters, validate_manifest
from .gpu.rates import estimate as estimate_units
from .gpu.rates import mode_param
from .gpu.registry import Registry
from .gpu.repository import Repository
from .gpu.scheduler import Scheduler
from .history_routes import router as history_router
from .instance_routes import router as instance_router
from .job_history import job_timing_details, job_timings
from .job_schemas import JobInput
from .model_library_routes import router as model_library_router
from .native_proxy import NativeHostMiddleware
from .native_routes import router as native_router
from .oauth_config_routes import router as oauth_config_router
from .presenters import public_asset, public_job
from .review_routes import router as review_router
from .security import sign_session
from .storage import asset_path, delete_asset_file, save_asset


def usage(st, owner):
    from .gpu.ccu import usage_summary

    return usage_summary(Repository(st), owner)


@asynccontextmanager
async def lifespan(app):
    if not getattr(app.state, "store", None):
        app.state.store = Store(settings.database_url)
    from .gpu.instances import InstanceService
    from .gpu.workbench import WorkbenchService

    app.state.repo = Repository(app.state.store)
    app.state.instance_service = InstanceService(app.state.repo, config=GpuConfig.from_env())

    def admit_workbench(owner, body, key, *, instance=None):
        parsed = JobInput.model_validate(body)
        return admit_job(app.state.store, {"id": owner}, parsed.model_dump(), key)

    app.state.workbench_service = WorkbenchService(
        repo=app.state.repo, registry=Registry(app.state.repo), admit_job=admit_workbench
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="Colab GPU Studio", version=__version__, lifespan=lifespan)
router = APIRouter()
_login_limits = {}


@app.middleware("http")
async def security_context(request, call_next):
    request.state.request_id = str(uuid.uuid4())
    try:
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("Origin")
            if origin and origin.rstrip("/") != settings.site_url.rstrip("/"):
                from urllib.parse import urlsplit

                if urlsplit(origin).netloc != request.url.netloc:
                    raise APIError("CSRF_ORIGIN", "跨站写入被拒绝", 403)
        if int(request.headers.get("Content-Length", "0")) > settings.max_upload_bytes + 1048576:
            raise APIError("PAYLOAD_TOO_LARGE", "请求超过上传限制", 413)
        response = await call_next(request)
    except APIError as exc:
        response = error_response(request, exc)
    response.headers["X-Request-ID"] = request.state.request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    # A same-origin referrer policy turns cross-origin form POST Origin into
    # "null" in browsers, breaking the signed native-host login handoff.
    # HTML documents may send only their origin cross-site; API responses keep
    # the stricter policy. NativeHostMiddleware still requires exact Origin.
    response.headers["Referrer-Policy"] = (
        "strict-origin-when-cross-origin"
        if response.headers.get("content-type", "").startswith("text/html")
        else "same-origin"
    )
    response.headers["X-Frame-Options"] = "DENY"
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "private, no-store"
    return response


def error_response(request, exc):
    error = {"code": exc.code, "message": exc.message, "retryable": exc.retryable}
    result = envelope(error=error, request_id=getattr(request.state, "request_id", None))
    result.update(code=exc.code, message=exc.message)
    return JSONResponse(result, status_code=exc.status)


@app.exception_handler(APIError)
async def api_error(request, exc):
    return error_response(request, exc)


@app.exception_handler(RequestValidationError)
async def validation_error(request, exc):
    # Do not return rejected input values (they may contain secrets).
    return error_response(
        request, APIError("VALIDATION_ERROR", "请求格式错误，请检查参数类型与必填字段", 422)
    )


@app.exception_handler(ValueError)
async def value_error(request, exc):
    return error_response(request, APIError("VALIDATION_ERROR", str(exc)[:300], 422))


@app.exception_handler(KeyError)
async def key_error(request, exc):
    return error_response(request, APIError("NOT_FOUND", "资源不存在", 404))


@app.exception_handler(Exception)
async def internal_error(request, exc):
    return error_response(request, APIError("INTERNAL_ERROR", "后台操作失败，请查看已脱敏的任务事件", 500))


@app.get("/api/health")
def health(st=Depends(get_store)):
    from sqlalchemy import text

    with st.engine.connect() as connection:
        connection.execute(text("SELECT 1"))
    return {"status": "ok", "version": __version__, "application": "colab-gpu-studio", "independent": True}


class Login(BaseModel):
    password: str = Field(min_length=1, max_length=1024)


@app.post("/api/auth/login")
def login(body: Login, request: Request):
    key = request.client.host if request.client else "unknown"
    previous = [t for t in _login_limits.get(key, []) if t > time.monotonic() - 60]
    if len(previous) >= 10:
        raise APIError("RATE_LIMITED", "登录尝试过于频繁，请稍后重试", 429)
    previous.append(time.monotonic())
    _login_limits[key] = previous
    if not settings.studio_password or not secrets.compare_digest(body.password, settings.studio_password):
        raise APIError("INVALID_CREDENTIALS", "密码不正确或尚未初始化", 401)
    token = sign_session(
        {"uid": "owner", "role": "admin", "iat": int(time.time()), "exp": int(time.time()) + 43200}
    )
    response = JSONResponse({"access_token": token, "token_type": "bearer", "expires_in": 43200})
    response.set_cookie(
        "studio_session",
        token,
        httponly=True,
        secure=settings.site_url.startswith("https://"),
        samesite="lax",
        max_age=43200,
    )
    return response


@app.get("/api/me")
def me(user=Depends(current_user)):
    return user


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


_TERMINAL_JOB_STATES = {
    "completed",
    "succeeded",
    "failed",
    "failed_final",
    "cancelled",
    "cancelled_local",
    "blocked",
}
_FAILED_JOB_STATES = {"failed", "failed_final", "cancelled", "cancelled_local", "blocked"}


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


@router.get("/models")
def models(request: Request, user=Depends(current_user), st=Depends(get_store)):
    _, registry = repo_registry(st, user)
    return ok(
        [
            {
                **r,
                "display_name": r.get("display_name", r.get("name", r["model_id"])),
                "reason": r.get("disabled_reason"),
            }
            for r in registry.models(user["id"])
        ],
        request,
    )


@router.post("/models/register")
def register_model(body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)):
    admin(user)
    repo, _ = repo_registry(st, user)
    model_id = str(body.get("model_id", "")).strip()
    if (
        not model_id
        or len(model_id) > 100
        or any(x.get("model_id") == model_id for x in repo.list("models", user["id"]))
    ):
        raise APIError("MODEL_INVALID", "模型 ID 无效或已存在")
    allowed = {
        "model_id",
        "display_name",
        "revision",
        "source",
        "license",
        "modality",
        "resource_profile",
        "variants",
    }
    payload = {k: v for k, v in body.items() if k in allowed}
    payload.update(
        enabled=False, is_mock=False, disabled_reason="待管理员来源审查、许可验证和真实 GPU 冒烟测试"
    )
    return ok(repo.create("models", user["id"], payload), request, 201)


@router.get("/workflows")
def workflows(
    request: Request, model_id: str | None = None, user=Depends(current_user), st=Depends(get_store)
):
    _, registry = repo_registry(st, user)
    return ok(registry.workflows(user["id"], model_id), request)


@router.get("/workflows/{workflow_id}/schema")
def workflow_schema(workflow_id: str, request: Request, compute_spec: str | None = None,
                    user=Depends(current_user), st=Depends(get_store)):
    repo = Repository(st)
    row = own(repo, "workflows", workflow_id, user)
    manifest = row["manifest"]
    cap = manifest.get("max_runtime_minutes", 30)
    spec = manifest.get("compute_spec")
    if compute_spec is not None:
        from .gpu.specs import spec_key

        try:
            spec = spec_key(json.loads(compute_spec))
        except ValueError as exc:
            raise APIError("COMPUTE_SPEC_INVALID", "无效的 GPU 规格", 422) from exc
    picked = mode_param(manifest)
    if picked:
        estimates = {m: estimate_units(repo, user["id"], row["workflow_key"], manifest, spec, cap, mode=m)[0] for m in picked[1]["variants"]}
        return ok({**manifest, "estimated_units": estimates[picked[1]["default"]], "estimates": estimates}, request)
    units, _ = estimate_units(repo, user["id"], row["workflow_key"], manifest, spec, cap)
    return ok({**manifest, "estimated_units": units}, request)




def cloud_sync_workflow(st, owner, row):
    if os.getenv("GPU_STORAGE", "local") != "drive":
        return row
    from .gpu.accounts import AccountService
    from .gpu.drive import DriveStorage
    from .gpu.google import GoogleService

    repo = Repository(st)
    storage_job = {"owner_id": owner, "id": row["id"]}
    if not GoogleService(repo, owner).status()["drive"]["connected"]:
        service = AccountService(repo, owner)
        accounts = [
            account for account in service.list()
            if account.get("enabled", True)
            and account.get("authorization", {}).get("drive", {}).get("connected")
        ]
        drive_ids = {service.resolve_drive_account_id(account["account_id"]) for account in accounts}
        if len(drive_ids) != 1:
            raise APIError("DRIVE_ACCOUNT_REQUIRED", "工作流同步需要唯一的已连接 Drive 账号", 409)
        account_id = next(iter(drive_ids))
        storage_job.update(account_id=account_id, drive_account_id=account_id)
    storage = DriveStorage(storage_job, repo)
    folder = f"workflows/{row['workflow_key']}/{row['workflow_version']}"
    records = {}
    for name, field in [
        ("manifest.json", "manifest"),
        ("workflow.api.json", "api_graph"),
        ("workflow.editable.json", "editable_graph"),
        ("object_info.json", "object_info"),
    ]:
        content = json.dumps(row.get(field, {}), ensure_ascii=False, sort_keys=True).encode()
        record = storage.put(folder + "/" + name, content, "application/json")
        if not storage.verify(record, content):
            raise APIError("DRIVE_VERIFY_FAILED", "工作流保存后校验失败")
        records[field] = record
    return Repository(st).update("workflows", row["id"], {"drive_files": records}, owner)


@router.post("/workflows/import")
def import_workflow(body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)):
    admin(user)
    if len(json.dumps(body)) > 5 * 1024 * 1024:
        raise APIError("WORKFLOW_TOO_LARGE", "工作流 JSON 过大", 413)
    repo, registry = repo_registry(st, user)
    manifest = body.get("manifest") or {}
    body = {
        **body,
        "workflow_key": body.get("workflow_key", manifest.get("workflow_key")),
        "model_id": body.get("model_id", manifest.get("model_id")),
    }
    row = registry.import_workflow(user["id"], body)
    from .gpu.model_library import ModelLibrary
    lib = ModelLibrary(repo, user["id"])
    try:
        lib.dependency_record(row["id"], rescan=True)
        # Exact manifest sources can enter the durable queue without allocating GPU.
        if any(d.get("weight") for d in row.get("model_dependency_scan") or []):
            from .gpu.model_sources import ModelError
            try:
                lib.prepare_dependencies(row["id"])
            except ModelError as exc:
                row = repo.update("workflows", row["id"], {"model_preparation_error": str(exc)}, user["id"])
    finally:
        lib.close()
    row = cloud_sync_workflow(st, user["id"], row)
    return ok(row, request, 201)


@router.put("/workflows/{workflow_id}/configuration")
def configure_image_draft(workflow_id: str, body: dict, request: Request,
                          user=Depends(current_user), st=Depends(get_store)):
    admin(user)
    if len(json.dumps(body)) > 5 * 1024 * 1024:
        raise APIError("WORKFLOW_TOO_LARGE", "工作流 JSON 过大", 413)
    repo, _ = repo_registry(st, user)
    row = own(repo, "workflows", workflow_id, user)
    if row.get("state") != "draft" or not row.get("image_extraction"):
        raise APIError("WORKFLOW_IMMUTABLE", "只能配置图片导入的未审核草稿", 409)
    for name in ("manifest", "api_graph", "object_info"):
        if not isinstance(body.get(name), dict):
            raise ValueError(f"缺少 {name} 对象")
    if not isinstance(body.get("model_id"), str) or not body["model_id"].strip():
        raise ValueError("请提供 model_id（未关联时填写 unconfigured）")
    from .gpu.image_import import missing_configuration
    patch = {name: body[name] for name in ("manifest", "api_graph", "object_info", "model_id")}
    pending = missing_configuration(patch)
    if not pending:
        from .gpu.specs import resolve_compute
        resolve_compute(patch["manifest"], None)
    patch.update(pending_configuration=pending, review_approved=False, smoke_test_passed=False,
                 smoke_test_kind=None, sha256=digest({k: patch[k] for k in ("manifest", "api_graph", "object_info")}))
    row = repo.update("workflows", workflow_id, patch, user["id"])
    from .gpu.model_library import ModelLibrary
    ModelLibrary(repo, user["id"]).dependency_record(row["id"], rescan=True)
    return ok(row, request)


@router.post("/workflows/{workflow_id}/validate")
def validate_workflow(workflow_id: str, request: Request, user=Depends(current_user), st=Depends(get_store)):
    _, registry = repo_registry(st, user)
    return ok(registry.validate(user["id"], workflow_id), request)


@router.post("/workflows/{workflow_id}/smoke-test")
def smoke_workflow(workflow_id: str, request: Request, user=Depends(current_user), st=Depends(get_store)):
    admin(user)
    repo, registry = repo_registry(st, user)
    row = own(repo, "workflows", workflow_id, user)
    model = next((m for m in registry.models(user["id"]) if m["model_id"] == row["model_id"]), None)
    if not model or not model.get("is_mock"):
        raise APIError(
            "REAL_SMOKE_REQUIRES_REVIEW",
            "真实工作流须先通过来源审查并完成受预算保护的 GPU 冒烟；不能提交客户端伪造证据",
        )
    validate_manifest(row["manifest"], row["api_graph"], row["object_info"])
    params = resolve_parameters(row["manifest"], {})
    graph = bind_parameters(row["manifest"], row["api_graph"], params)
    from .gpu.compute import MockGPUExecutor

    test_job = {
        "id": str(uuid.uuid4()),
        "provider": "mock",
        "snapshot": {"resolved_params": params, "bound_graph": graph},
        "execution_id": str(uuid.uuid4()),
    }
    executor = MockGPUExecutor()
    executor.start({"runtime_name": "mock-static-smoke"}, test_job, test_job["execution_id"])
    result = executor.poll({"runtime_name": "mock-static-smoke"}, test_job)
    if result.get("state") != "completed" or not result.get("outputs"):
        raise APIError("SMOKE_FAILED", "模拟执行冒烟未通过")
    return ok(
        repo.update(
            "workflows", workflow_id, {"smoke_test_passed": True, "smoke_test_kind": "mock"}, user["id"]
        ),
        request,
    )


@router.post("/workflows/{workflow_id}/publish")
def publish_workflow(
    workflow_id: str, body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)
):
    admin(user)
    _, registry = repo_registry(st, user)
    return ok(registry.publish(user["id"], workflow_id, confirmed=body.get("confirmed") is True), request)


@router.post("/workflows/rollback")
def rollback_workflow(body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)):
    admin(user)
    _, registry = repo_registry(st, user)
    return ok(registry.rollback(user["id"], body.get("workflow_key"), body.get("version")), request)


@router.post("/workflows/{workflow_id}/deactivate")
def deactivate_workflow(
    workflow_id: str, body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)
):
    admin(user)
    if not isinstance(body, dict) or body.get("confirmed") is not True:
        raise APIError("CONFIRMATION_REQUIRED", "停用必须明确confirmed")
    _, registry = repo_registry(st, user)
    return ok(registry.deactivate(user["id"], workflow_id), request)


@router.delete("/workflows/{workflow_id}")
def delete_workflow(workflow_id: str, request: Request, user=Depends(current_user), st=Depends(get_store)):
    admin(user)
    with st.admission_lock() as acquired:
        if not acquired:
            raise APIError("SCHEDULER_BUSY", "工作流管理繁忙，请重试", 503, True)
        try:
            Repository(st).delete_workflow(user["id"], workflow_id)
        except KeyError as exc:
            raise APIError("NOT_FOUND", "工作流不存在或不属于当前用户", 404) from exc
    return ok({"id": workflow_id}, request)


@router.get("/presets")
def presets(
    request: Request,
    workflow_version_id: str | None = None,
    user=Depends(current_user),
    st=Depends(get_store),
):
    rows = Repository(st).list("presets", user["id"])
    return ok(
        [r for r in rows if not workflow_version_id or r.get("workflow_version_id") == workflow_version_id],
        request,
    )


@router.post("/presets")
def save_preset(body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)):
    repo = Repository(st)
    workflow = own(repo, "workflows", body.get("workflow_version_id"), user)
    overrides = body.get("overrides", {})
    resolve_parameters(workflow["manifest"], overrides)
    row = repo.create(
        "presets",
        user["id"],
        {
            "name": str(body.get("name", "未命名预设"))[:100],
            "workflow_version_id": workflow["id"],
            "overrides": overrides,
        },
    )
    return ok(row, request, 201)


@router.patch("/presets/{preset_id}")
def update_preset(preset_id: str, body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)):
    repo = Repository(st)
    preset = own(repo, "presets", preset_id, user)
    patch: dict = {}
    if "name" in body:
        name = str(body.get("name") or "").strip()[:100]
        if not name:
            raise APIError("VALIDATION_ERROR", "预设名称不能为空")
        patch["name"] = name
    if "overrides" in body:
        workflow = own(repo, "workflows", preset.get("workflow_version_id"), user)
        overrides = body.get("overrides", {})
        resolve_parameters(workflow["manifest"], overrides)
        patch["overrides"] = overrides
    if not patch:
        raise APIError("VALIDATION_ERROR", "未提供需要更新的字段")
    return ok(repo.update("presets", preset_id, patch, user["id"]), request)


@router.delete("/presets/{preset_id}")
def delete_preset(preset_id: str, request: Request, user=Depends(current_user), st=Depends(get_store)):
    repo = Repository(st)
    own(repo, "presets", preset_id, user)
    repo.delete("presets", preset_id, user["id"])
    return ok({"deleted": True, "id": preset_id}, request)








@router.get("/loras")
def list_loras(request: Request, workflow_key: str | None = None, user=Depends(current_user), st=Depends(get_store)):
    # LoRAs always live in Drive (like the model cache), whatever GPU_STORAGE says about results.
    files, error = [], None
    try:
        files = _drive(Repository(st), user).list_loras()
    except Exception:  # noqa: BLE001 - Drive down must not hide the curated list
        error = "暂时读不到 Drive 里的 LoRA 文件夹"
    return ok({"curated": lora_registry.curated_for(workflow_key), "files": files, "folder": "AI-Studio/我的文件夹/loras/h3",
               "max_bytes": lora_registry.MAX_BYTES, "error": error}, request)


@router.post("/loras/upload")
def lora_upload(body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)):
    name, size = str(body.get("name") or "").strip(), body.get("size")
    if "/" in name or not lora_registry.is_lora_file(name, size if isinstance(size, int) else None):
        raise APIError("VALIDATION_ERROR", "只接受 1.5GB 以内的 .safetensors 文件")
    origin = request.headers.get("origin") or f"{request.url.scheme}://{request.url.netloc}"
    return ok({"upload_url": _drive(Repository(st), user).lora_upload_session(name, size, origin)}, request)


@router.post("/jobs", status_code=202)
def create_job(
    body: JobInput,
    request: Request,
    idempotency_key: str | None = Header(default=None),
    user=Depends(current_user),
    st=Depends(get_store),
):
    return ok(public_job(admit_job(st, user, body.model_dump(), idempotency_key)), request, 202)


@router.get("/jobs")
def jobs(
    request: Request,
    project_id: str | None = None,
    include_cleared: bool = False,
    user=Depends(current_user),
    st=Depends(get_store),
):
    if project_id:
        own(st, "projects", project_id, user)
    rows = Repository(st).list("jobs", user["id"], project_id)
    if not include_cleared:
        rows = [
            j
            for j in rows
            if j.get("history_cleared_at") is None or str(j.get("state") or "").lower() not in _TERMINAL_JOB_STATES
        ]
    return ok([public_job(j) for j in rows], request)


@router.post("/jobs/history/clear")
def clear_job_history(
    request: Request,
    body: dict | None = None,
    user=Depends(current_user),
    st=Depends(get_store),
):
    payload = body if isinstance(body, dict) else {}
    project_id = payload.get("project_id")
    if project_id:
        own(st, "projects", project_id, user)
    scope = str(payload.get("scope") or "all_terminal")
    if scope not in {"all_terminal", "failed_cancelled", "mock"}:
        raise APIError("VALIDATION_ERROR", "无效的任务清理范围")
    raw_ids = payload.get("job_ids")
    target_ids: set[str] | None = None
    if raw_ids is not None:
        if not isinstance(raw_ids, list) or len(raw_ids) > 500 or any(not isinstance(item, str) for item in raw_ids):
            raise APIError("VALIDATION_ERROR", "job_ids 格式无效")
        target_ids = set(raw_ids)
    repo = Repository(st)
    rows = repo.list("jobs", user["id"], project_id)
    now = time.time()
    cleared = 0
    for row in rows:
        state = str(row.get("state") or "").lower()
        if state not in _TERMINAL_JOB_STATES or row.get("history_cleared_at") is not None:
            continue
        if target_ids is not None:
            if row["id"] not in target_ids:
                continue
        elif scope == "failed_cancelled" and state not in _FAILED_JOB_STATES:
            continue
        elif scope == "mock" and row.get("provider") != "mock":
            continue
        repo.update("jobs", row["id"], {"history_cleared_at": now}, user["id"])
        cleared += 1
    return ok({"cleared": cleared}, request)


@router.get("/jobs/{job_id}")
def get_job(job_id: str, request: Request, user=Depends(current_user), st=Depends(get_store)):
    repo = Repository(st)
    job = own(repo, "jobs", job_id, user)
    result = public_job(job)
    result["events"] = [e for e in repo.list("events", user["id"]) if e.get("job_id") == job_id][::-1]
    result["logs"] = result["events"]
    now = time.time()
    result["timings"] = job_timings(job, result["events"], now=now)
    result["timing_details"] = job_timing_details(job, result["events"], now=now)
    allowed={'id','state','attempt','started_at','finished_at','error_code'}
    result['attempts']=sorted([{k:v for k,v in row.items() if k in allowed}
        for row in repo.list('executions',user['id']) if row.get('job_id')==job_id],key=lambda row:row.get('attempt',0))
    return ok(result, request)


@router.get("/jobs/{job_id}/events")
def job_events(job_id: str, request: Request, user=Depends(current_user), st=Depends(get_store)):
    repo = Repository(st)
    own(repo, "jobs", job_id, user)
    return ok([e for e in repo.list("events", user["id"]) if e.get("job_id") == job_id][::-1], request)


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str, request: Request, user=Depends(current_user), st=Depends(get_store)):
    own(Repository(st), "jobs", job_id, user)
    return ok(public_job(Scheduler(st).cancel(user["id"], job_id)), request)


@router.post("/jobs/{job_id}/retry")
def retry_job(job_id: str, request: Request, user=Depends(current_user), st=Depends(get_store)):
    original = own(Repository(st), "jobs", job_id, user)
    if original["state"] not in {"failed", "cancelled", "completed"}:
        raise APIError("JOB_ACTIVE", "原任务尚未结束，不能重复提交")
    key = request.headers.get("Idempotency-Key")
    if not key:
        raise APIError("IDEMPOTENCY_REQUIRED", "再次生成需提供新的幂等键")
    raw=copy.deepcopy(original.get('request_body') or {})
    if raw.get('kind') or any((original.get('snapshot') or {}).get(flag) or original.get(flag) for flag in ('system_smoke','workflow_smoke','prep','creation_warmup')):
        raise APIError('SPECIAL_JOB_RETRY_REQUIRED','诊断或预热任务需回到对应入口重新确认',409)
    if original.get('resolved_seed') is not None:
        raw.setdefault('params',{})['seed']=original['resolved_seed']
    raw['retry_of']=original['id']
    return ok(public_job(admit_job(st, user, raw, key)), request, 202)


@router.post('/jobs/{job_id}/attempts')
def retry_attempt(job_id: str, request: Request, user=Depends(current_user), st=Depends(get_store)):
    repo=Repository(st)
    own(repo,'jobs',job_id,user)
    try:
        job=repo.retry_attempt(job_id,user['id'],request.headers.get('Idempotency-Key'))
    except ValueError as exc:
        raise APIError('ATTEMPT_RETRY_REJECTED',str(exc),409) from exc
    return ok(public_job(job),request,202)


@router.get('/jobs/{job_id}/attempts')
def job_attempts(job_id: str, request: Request, user=Depends(current_user), st=Depends(get_store)):
    repo=Repository(st)
    own(repo,'jobs',job_id,user)
    allowed={'id','job_id','state','attempt','started_at','finished_at','error_code','created_at'}
    rows=[{k:v for k,v in row.items() if k in allowed} for row in repo.list('executions',user['id']) if row.get('job_id')==job_id]
    return ok(sorted(rows,key=lambda row:row.get('attempt',0)),request)


@router.get("/jobs/{job_id}/outputs")
def outputs(job_id: str, request: Request, user=Depends(current_user), st=Depends(get_store)):
    job = own(Repository(st), "jobs", job_id, user)
    rows = [own(st, "assets", aid, user) for aid in job.get("output_asset_ids", [])]
    return ok([public_asset(a) for a in rows], request)


@router.get("/gallery")
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
            if job.get("provider") == "mock" and item.get("mock") is None:
                item["mock"] = True
        result.append(item)
    return ok(result, request)


@app.get("/api/assets")
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


@app.post("/api/assets")
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


@app.get("/api/assets/{asset_id}/download")
@app.get("/api/assets/{asset_id}/preview")
def download_asset(asset_id: str, user=Depends(current_user), st=Depends(get_store)):
    asset = own(st, "assets", asset_id, user)
    return FileResponse(asset_path(asset), media_type=asset["mime"], filename=asset["filename"])


@app.get("/api/assets/{asset_id}/thumbnail")
def thumbnail(asset_id: str, user=Depends(current_user), st=Depends(get_store)):
    asset = own(st, "assets", asset_id, user)
    if not asset["mime"].startswith("image/"):
        raise APIError("THUMBNAIL_UNAVAILABLE", "非图片素材不提供图片缩略图", 404)
    with Image.open(asset_path(asset)) as image:
        image.thumbnail((480, 480))
        target = io.BytesIO()
        image.convert("RGB").save(target, "WEBP", quality=80)
    return Response(target.getvalue(), media_type="image/webp")


@app.patch("/api/assets/{asset_id}")
def favorite_asset(
    asset_id: str, body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)
):
    own(st, "assets", asset_id, user)
    if not isinstance(body.get("favorite"), bool):
        raise APIError("VALIDATION_ERROR", "favorite必须为布尔值")
    return ok(
        public_asset(st.update("assets", asset_id, {"favorite": body["favorite"]}, user["id"])), request
    )


@app.delete("/api/assets/{asset_id}")
def delete_asset(asset_id: str, request: Request, user=Depends(current_user), st=Depends(get_store)):
    asset = own(st, "assets", asset_id, user)
    delete_asset_file(asset)
    st.delete("assets", asset_id, user["id"])
    return ok({"deleted": True}, request)


@app.post("/api/assets/batch-delete")
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


@router.get("/usage")
def get_usage(request: Request, user=Depends(current_user), st=Depends(get_store)):
    return ok(usage(st, user["id"]), request)


@router.get("/settings")
def get_settings(request: Request, user=Depends(current_user), st=Depends(get_store)):
    from .gpu.google import GoogleService

    cfg = GpuConfig.from_env()
    return ok(
        {
            "offline_mode": os.getenv("STUDIO_OFFLINE_TEST") == "1",
            "provider_default": "mock",
            "real_enabled": cfg.real_enabled,
            "google": GoogleService(Repository(st), user["id"]).status(),
            "budget": usage(st, user["id"]),
            "storage": os.getenv("GPU_STORAGE", "local"),
            "independent": True,
        },
        request,
    )


@router.get("/compute/specs")
def compute_specs(
    request: Request,
    provider: str = "mock",
    account_id: str | None = None,
    user=Depends(current_user),
    st=Depends(get_store),
):
    if provider == "mock":
        return ok(MockColabProvider().list_specs(), request)
    if provider != "colab":
        raise APIError("PROVIDER_INVALID", "不支持该 Provider")
    if os.getenv("STUDIO_OFFLINE_TEST") == "1":
        raise APIError("OFFLINE_MODE", "站点处于离线测试模式，不能连接 Colab。可改查模拟 GPU。", status=409)
    if not GpuConfig.from_env().real_enabled:
        raise APIError("REAL_GPU_DISABLED", "真实 GPU 未启用，不能连接 Colab。可改查模拟 GPU。", status=409)
    repo = Repository(st)
    _require_account(repo, user["id"], account_id, require_allocation=False)
    from .gpu.colab import GoogleColabProvider
    from .gpu.rates import minute_rate

    try:
        specs = GoogleColabProvider(repo, user["id"], account_id=account_id or None).list_specs()
        for spec in specs:
            rate = minute_rate(spec)
            spec["reference_rate_hourly"] = round(rate * 60, 6) if rate is not None else None
        return ok(specs, request)
    except APIError:
        raise
    except Exception as exc:
        raise APIError("COLAB_CHECK_FAILED", f"检查 Colab GPU 失败：{str(exc)[:160]}", status=502, retryable=True) from exc


@router.get("/compute/subscription")
def subscription(
    request: Request,
    account_id: str | None = None,
    user=Depends(current_user),
    st=Depends(get_store),
):
    from .gpu.colab import GoogleColabProvider

    repo = Repository(st)
    _require_account(repo, user["id"], account_id, require_allocation=False)
    return ok(GoogleColabProvider(repo, user["id"], account_id=account_id or None).subscription(), request)


@router.post("/admin/compute/stop")
def emergency_stop(request: Request, user=Depends(current_user), st=Depends(get_store)):
    admin(user)
    return ok(Scheduler(st).stop(user["id"]), request)


@router.post("/auth/google/start")
def google_start(body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)):
    from .gpu.google import GoogleService

    result = GoogleService(Repository(st), user["id"]).start(body.get("purpose"))
    return ok({"authorization_url": result["authorization_url"]}, request)


@router.get("/auth/google/callback")
def google_callback(code: str, state: str, st=Depends(get_store)):
    from .gpu.google import GoogleService

    GoogleService(Repository(st), None).callback(code, state)
    return RedirectResponse("/?google=connected", status_code=303)


@router.post("/auth/google/disconnect")
def google_disconnect(body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)):
    from .gpu.google import GoogleService

    return ok(GoogleService(Repository(st), user["id"]).disconnect(body.get("purpose")), request)


# White-listed Agent API: it is not a shell and has no budget or compute-create tool.
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


def agent_user(request):
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
        result = usage(st, user["id"])
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


if os.getenv("STUDIO_ATELIER_ENABLED", "0").strip().lower() in {"1", "true"}:
    app.include_router(atelier_router)
app.include_router(account_router)
app.include_router(instance_router)
app.include_router(native_router)
app.include_router(creation_api_router)
app.add_middleware(CreationHostMiddleware, studio_app=app)
app.add_middleware(NativeHostMiddleware, studio_app=app)
app.include_router(history_router)
app.include_router(compute_router)
app.include_router(review_router)
app.include_router(oauth_config_router)
app.include_router(model_library_router)


@app.exception_handler(GoogleError)
async def google_error(request, exc):
    return error_response(request, APIError("GOOGLE_REQUEST_FAILED", str(exc)[:250], 422))


router.include_router(build_batch_router(JobInput, admit_job, public_job))
app.include_router(router, prefix="/api/gpu")
# Canonical task-book endpoints are available too; provider switches keep /api/jobs unchanged.
app.include_router(router, prefix="/api", include_in_schema=False)

# Same-origin production PWA. The API always registers before the SPA fallback.

_static_root = Path(__file__).resolve().parents[2] / "web" / "dist"
if not _static_root.exists():
    _static_root = Path(__file__).resolve().parents[3] / "web" / "dist"
if _static_root.exists():
    if (_static_root / "assets").exists():
        app.mount("/assets", StaticFiles(directory=_static_root / "assets"), name="web-assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path.startswith("api/"):
            raise APIError("NOT_FOUND", "接口不存在", 404)
        target = (_static_root / path).resolve()
        if path and target.is_relative_to(_static_root.resolve()) and target.is_file():
            return FileResponse(target)
        return FileResponse(_static_root / "index.html")
