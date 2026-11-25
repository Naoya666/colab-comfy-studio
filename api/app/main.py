"""Independent personal AI Studio API."""

from __future__ import annotations

import copy
import gzip
import io
import json
import os
import secrets
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.datastructures import Headers, MutableHeaders

from . import __version__
from .account_routes import router as account_router
from .admission import _drive, admit_job, preview_job_reuse, repo_registry
from .agent_routes import AGENT_TOOLS
from .agent_routes import router as agent_router
from .asset_routes import assets_router, gallery_router
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
from .gpu.fixed_prompts import get_fixed_prompts, normalize_fixed_prompts, save_fixed_prompts
from .gpu.google import GoogleError
from .gpu.manifest import (
    bind_parameters,
    digest,
    resolve_parameters,
    validate_manifest,
    validate_partial_parameters,
)
from .gpu.rates import estimate as estimate_units
from .gpu.rates import mode_param
from .gpu.registry import Registry
from .gpu.repository import Repository
from .gpu.scheduler import Scheduler
from .history_routes import router as history_router
from .instance_routes import router as instance_router
from .job_history import job_timing_details
from .job_schemas import JobInput, ReusePreviewInput
from .model_library_routes import router as model_library_router
from .native_outputs_routes import router as native_outputs_router
from .native_proxy import NativeHostMiddleware
from .native_routes import router as native_router
from .oauth_config_routes import router as oauth_config_router
from .presenters import public_asset, public_job
from .project_routes import router as project_router
from .recovery_routes import router as recovery_router
from .review_routes import router as review_router
from .security import sign_session


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
    from .port_listeners import PortListenerManager

    app.state.port_listeners = PortListenerManager(app)
    await app.state.port_listeners.start_from_store(app.state.store)
    try:
        yield
    finally:
        await app.state.port_listeners.shutdown()
        outputs_client = getattr(app.state, "_native_outputs_http_client", None)
        if outputs_client is not None:
            await outputs_client.aclose()
            app.state._native_outputs_http_client = None


app = FastAPI(title="Colab GPU Studio", version=__version__, lifespan=lifespan)
router = APIRouter()
_login_limits: dict[str, list[float]] = {}
_login_limits_lock = threading.Lock()
_MAX_LOGIN_LIMIT_KEYS = 4096


@app.middleware("http")
async def security_context(request, call_next):
    request.state.request_id = str(uuid.uuid4())
    try:
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("Origin")
            if origin and origin.rstrip("/") != settings.site_url.rstrip("/"):
                from urllib.parse import urlsplit

                parsed_origin = urlsplit(origin)
                if (parsed_origin.scheme.lower(), parsed_origin.netloc.lower()) != (
                    request.url.scheme.lower(),
                    request.url.netloc.lower(),
                ):
                    raise APIError("CSRF_ORIGIN", "跨站写入被拒绝", 403)
        raw_cl = request.headers.get("Content-Length")
        if raw_cl is not None:
            try:
                content_length = int(raw_cl.strip())
            except ValueError as exc:
                raise APIError("VALIDATION_ERROR", "无效的 Content-Length 请求头", 400) from exc
            if content_length < 0 or content_length > settings.max_upload_bytes + 1048576:
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
    if request.url.path.startswith("/api/") and "cache-control" not in response.headers:
        response.headers["Cache-Control"] = "private, no-store"
    return response


_NON_COMPRESSIBLE_PREFIXES = ("image/", "video/", "audio/")
_NON_COMPRESSIBLE_TYPES = frozenset({"application/octet-stream", "text/event-stream"})


def _accepts_gzip(header_value: str) -> bool:
    if not header_value:
        return False
    for part in header_value.split(","):
        token, *params = [seg.strip() for seg in part.split(";")]
        if token.lower() != "gzip":
            continue
        q = 1.0
        for param in params:
            if param.lower().startswith("q="):
                try:
                    q = float(param[2:].strip())
                except ValueError:
                    q = 0.0
        if q > 0.0:
            return True
    return False


def _prepare_gzip_headers(headers: MutableHeaders) -> None:
    headers["Content-Encoding"] = "gzip"
    if "accept-ranges" in headers:
        del headers["accept-ranges"]
    etag = headers.get("etag")
    if etag and not etag.startswith("W/"):
        headers["ETag"] = f"W/{etag}"
    headers.add_vary_header("Accept-Encoding")


class SelectiveGZipMiddleware:
    """Compress JSON/text/JS/CSS responses >= minimum_size while leaving media/range streams untouched."""

    def __init__(self, app, *, minimum_size: int = 1024, compresslevel: int = 5):
        self.app = app
        self.minimum_size = minimum_size
        self.compresslevel = compresslevel

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        method = str(scope.get("method", "GET")).upper()
        accepts_gzip = method != "HEAD" and _accepts_gzip(Headers(scope=scope).get("accept-encoding", ""))

        initial_message = None
        passthrough = False
        gzip_buffer: io.BytesIO | None = None
        gzip_file: gzip.GzipFile | None = None

        async def send_wrapper(message):
            nonlocal initial_message, passthrough, gzip_buffer, gzip_file
            msg_type = message["type"]
            if msg_type == "http.response.start":
                status = message.get("status", 200)
                headers = Headers(raw=message.get("headers", []))
                ctype = headers.get("content-type", "").split(";", 1)[0].strip().lower()
                compressible = not (
                    status < 200
                    or status in {204, 206, 304}
                    or "content-encoding" in headers
                    or ctype.startswith(_NON_COMPRESSIBLE_PREFIXES)
                    or ctype in _NON_COMPRESSIBLE_TYPES
                )
                if not compressible or not accepts_gzip:
                    if compressible:
                        MutableHeaders(raw=message["headers"]).add_vary_header("Accept-Encoding")
                    passthrough = True
                    await send(message)
                    return
                initial_message = message
                return
            if msg_type == "http.response.body":
                if passthrough or initial_message is None:
                    await send(message)
                    return
                body = message.get("body", b"")
                more_body = message.get("more_body", False)
                if gzip_file is None and not more_body:
                    # Single-chunk response (JSONResponse, small text, FileResponse single body):
                    if len(body) < self.minimum_size:
                        passthrough = True
                        MutableHeaders(raw=initial_message["headers"]).add_vary_header("Accept-Encoding")
                        await send(initial_message)
                        await send(message)
                        return
                    compressed = gzip.compress(body, compresslevel=self.compresslevel)
                    headers = MutableHeaders(raw=initial_message["headers"])
                    _prepare_gzip_headers(headers)
                    headers["Content-Length"] = str(len(compressed))
                    await send(initial_message)
                    await send({"type": "http.response.body", "body": compressed, "more_body": False})
                    return
                if gzip_file is None:
                    gzip_buffer = io.BytesIO()
                    gzip_file = gzip.GzipFile(mode="wb", fileobj=gzip_buffer, compresslevel=self.compresslevel)
                    headers = MutableHeaders(raw=initial_message["headers"])
                    _prepare_gzip_headers(headers)
                    if "content-length" in headers:
                        del headers["content-length"]
                    await send(initial_message)
                if body:
                    gzip_file.write(body)
                    if more_body:
                        gzip_file.flush()
                if not more_body:
                    gzip_file.close()
                chunk = gzip_buffer.getvalue()
                gzip_buffer.seek(0)
                gzip_buffer.truncate()
                await send({"type": "http.response.body", "body": chunk, "more_body": more_body})
                return
            await send(message)

        await self.app(scope, receive, send_wrapper)


class ImmutableStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        if response.status_code in {200, 304}:
            response.headers.setdefault("Cache-Control", "public, max-age=31536000, immutable")
        return response


def error_response(request, exc):
    error = {"code": exc.code, "message": exc.message, "retryable": exc.retryable}
    if getattr(exc, "detail", None) is not None:
        error["detail"] = exc.detail
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
    now_mono = time.monotonic()
    cutoff = now_mono - 60
    with _login_limits_lock:
        if len(_login_limits) >= _MAX_LOGIN_LIMIT_KEYS:
            stale_keys = [k for k, ts_list in _login_limits.items() if not ts_list or ts_list[-1] <= cutoff]
            for stale in stale_keys:
                _login_limits.pop(stale, None)
            while len(_login_limits) >= _MAX_LOGIN_LIMIT_KEYS:
                _login_limits.pop(next(iter(_login_limits)), None)
        previous = [t for t in _login_limits.get(key, []) if t > cutoff]
        if len(previous) >= 10:
            _login_limits[key] = previous
            raise APIError("RATE_LIMITED", "登录尝试过于频繁，请稍后重试", 429)
        previous.append(now_mono)
        _login_limits[key] = previous
    if not settings.studio_password or not secrets.compare_digest(body.password, settings.studio_password):
        raise APIError("INVALID_CREDENTIALS", "密码不正确或尚未初始化", 401)
    token = sign_session(
        {
            "uid": "owner",
            "role": "admin",
            "purpose": "studio-session",
            "iat": int(time.time()),
            "exp": int(time.time()) + 43200,
        }
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
        if not (isinstance(record, dict) and record.get("verified") is True):
            if not storage.verify(record, content):
                raise APIError("DRIVE_VERIFY_FAILED", "工作流保存后校验失败")
        records[field] = record
    return Repository(st).update("workflows", row["id"], {"drive_files": records}, owner)


@router.post("/workflows/fetch-url")
def fetch_workflow_from_url(body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)):
    admin(user)
    raw_url = body.get("url") if isinstance(body, dict) else None
    from .gpu.model_library import ModelLibrary
    from .gpu.model_sources import ModelError
    from .gpu.workflow_sources import (
        PLAIN_COMFYUI_WORKFLOW_ERROR,
        fetch_workflow_source,
        is_plain_comfyui_workflow,
        is_studio_workflow_bundle,
    )

    repo = Repository(st)
    lib = ModelLibrary(repo, user["id"])
    try:
        fetched = fetch_workflow_source(lib.http, raw_url)
    except ModelError as exc:
        raise APIError("WORKFLOW_URL_ERROR", str(exc), 422) from exc
    finally:
        lib.close()

    if fetched["kind"] == "png":
        return ok({
            "kind": "png",
            "source_name": fetched["source_name"],
            "source_host": fetched["source_host"],
            **({"note": fetched["note"]} if fetched.get("note") else {}),
            "extraction": fetched["image_extraction"],
        }, request)

    payload = fetched["payload"]
    if is_studio_workflow_bundle(payload):
        return ok({
            "kind": "bundle",
            "source_name": fetched["source_name"],
            "source_host": fetched["source_host"],
            **({"note": fetched["note"]} if fetched.get("note") else {}),
            "bundle": payload,
        }, request)
    if is_plain_comfyui_workflow(payload):
        raise APIError("UNSUPPORTED_WORKFLOW_FORMAT", PLAIN_COMFYUI_WORKFLOW_ERROR, 422)
    raise APIError("INVALID_WORKFLOW_BUNDLE", "JSON 必须包含 manifest、api_graph、object_info", 422)


@router.get("/workflows/{workflow_id}/export")
def export_workflow(workflow_id: str, request: Request, user=Depends(current_user), st=Depends(get_store)):
    admin(user)
    repo, _ = repo_registry(st, user)
    row = own(repo, "workflows", workflow_id, user)
    if row.get("state") != "published":
        raise APIError("WORKFLOW_NOT_PUBLISHED", "仅支持导出已发布的工作流版本", 409)
    from .gpu.workflow_export import build_workflow_export

    return ok(build_workflow_export(repo, user["id"], row), request)


@router.post("/workflows/import")
def import_workflow(body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)):
    admin(user)
    if len(json.dumps(body)) > 5 * 1024 * 1024:
        raise APIError("WORKFLOW_TOO_LARGE", "工作流 JSON 过大", 413)
    repo, registry = repo_registry(st, user)
    from .gpu.workflow_export import (
        find_workflow_version_conflict,
        sanitize_import_model,
        summarize_portable_weights,
    )

    body_copy = copy.deepcopy(body)
    is_wrapped = isinstance(body_copy.get("workflow"), dict)
    source = body_copy["workflow"] if is_wrapped else body_copy
    manifest = source.get("manifest") if isinstance(source.get("manifest"), dict) else {}
    resolved_key = source.get("workflow_key") or source.get("key") or manifest.get("workflow_key")
    resolved_model_id = source.get("model_id") or manifest.get("model_id")
    if resolved_key is not None and not source.get("workflow_key"):
        source["workflow_key"] = resolved_key
    if resolved_model_id is not None and not source.get("model_id"):
        source["model_id"] = resolved_model_id

    manifest_version = manifest.get("version")
    if isinstance(resolved_key, str) and resolved_key and manifest_version is not None:
        conflict = find_workflow_version_conflict(repo, user["id"], resolved_key, manifest_version)
        if conflict is not None:
            raise APIError(
                "WORKFLOW_VERSION_CONFLICT",
                f"工作流「{resolved_key}」版本 v{manifest_version} 已存在，请修改版本号或工作流标识后导入",
                status=409,
                detail=conflict,
            )

    row = registry.import_workflow(user["id"], body_copy)

    model_auto_registered = False
    raw_model = source.get("model") if isinstance(source.get("model"), dict) else body_copy.get("model")
    target_model_id = row.get("model_id")
    if isinstance(raw_model, dict) and isinstance(target_model_id, str) and target_model_id and target_model_id != "unconfigured":
        existing_models = repo.list("models", user["id"])
        if not any(m.get("model_id") == target_model_id for m in existing_models):
            sanitized_model = sanitize_import_model(raw_model, target_model_id)
            if sanitized_model is not None:
                repo.create("models", user["id"], sanitized_model)
                model_auto_registered = True

    from .gpu.model_library import ModelLibrary
    lib = ModelLibrary(repo, user["id"])
    try:
        lib.dependency_record(row["id"], rescan=True)
        # Only exact pinned HF / local Drive weights enter the background queue immediately on import.
        # Civitai / manual portable weights stay visible in Model Dependencies for explicit user preparation.
        has_auto_queue_weight = any(
            isinstance(d.get("weight"), dict)
            and (
                (d["weight"].get("source") == "drive" and d["weight"].get("drive_file_id"))
                or (
                    d["weight"].get("source") in {None, "huggingface"}
                    and all(d["weight"].get(k) for k in ("repo_id", "revision", "filename"))
                )
            )
            for d in (row.get("model_dependency_scan") or [])
        )
        if has_auto_queue_weight:
            from .gpu.model_sources import ModelError
            try:
                lib.prepare_dependencies(row["id"], auto_import=True)
            except ModelError as exc:
                row = repo.update("workflows", row["id"], {"model_preparation_error": str(exc)}, user["id"])
    finally:
        lib.close()
    row = cloud_sync_workflow(st, user["id"], row)
    row_manifest = row.get("manifest") or {}
    raw_weights = (row_manifest.get("dependencies") or {}).get("weight_files") or row_manifest.get("weight_files") or []
    weight_summary = summarize_portable_weights(raw_weights)
    response_payload = {
        **row,
        "model_auto_registered": model_auto_registered,
        "import_weight_summary": weight_summary,
    }
    return ok(response_payload, request, 201)


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


@router.get("/projects/{project_id}/workflows/{workflow_id}/fixed-prompts")
def read_fixed_prompts(
    project_id: str,
    workflow_id: str,
    request: Request,
    user=Depends(current_user),
    st=Depends(get_store),
):
    own(st, "projects", project_id, user)
    own(Repository(st), "workflows", workflow_id, user)
    return ok(get_fixed_prompts(st, user["id"], project_id, workflow_id), request)


@router.put("/projects/{project_id}/workflows/{workflow_id}/fixed-prompts")
def write_fixed_prompts(
    project_id: str,
    workflow_id: str,
    body: dict,
    request: Request,
    user=Depends(current_user),
    st=Depends(get_store),
):
    own(st, "projects", project_id, user)
    own(Repository(st), "workflows", workflow_id, user)
    try:
        saved = save_fixed_prompts(st, user["id"], project_id, workflow_id, body)
    except ValueError as exc:
        raise APIError("VALIDATION_ERROR", str(exc)) from exc
    return ok(saved, request)


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
    raw_overrides = body.get("overrides", {})
    try:
        overrides = validate_partial_parameters(workflow["manifest"], raw_overrides)
        fixed_prompts = (
            normalize_fixed_prompts(body["fixed_prompts"])
            if "fixed_prompts" in body and body["fixed_prompts"] is not None
            else None
        )
    except ValueError as exc:
        raise APIError("VALIDATION_ERROR", str(exc)) from exc
    payload = {
        "name": str(body.get("name", "未命名预设"))[:100],
        "workflow_version_id": workflow["id"],
        "overrides": overrides,
    }
    if fixed_prompts is not None:
        payload["fixed_prompts"] = fixed_prompts
    row = repo.create("presets", user["id"], payload)
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
        try:
            patch["overrides"] = validate_partial_parameters(workflow["manifest"], body.get("overrides", {}))
        except ValueError as exc:
            raise APIError("VALIDATION_ERROR", str(exc)) from exc
    if "fixed_prompts" in body:
        if body["fixed_prompts"] is None:
            patch["fixed_prompts"] = None
        else:
            try:
                patch["fixed_prompts"] = normalize_fixed_prompts(body["fixed_prompts"])
            except ValueError as exc:
                raise APIError("VALIDATION_ERROR", str(exc)) from exc
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


@router.post("/jobs/reuse-preview")
def reuse_preview(
    body: ReusePreviewInput,
    request: Request,
    user=Depends(current_user),
    st=Depends(get_store),
):
    return ok(preview_job_reuse(st, user, body.model_dump()), request)


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
    include_system = bool(payload.get("include_system"))
    repo = Repository(st)
    rows = repo.list_by_states("jobs", _TERMINAL_JOB_STATES, user["id"], project_id=project_id)
    now = time.time()
    cleared = 0
    for row in rows:
        state = str(row.get("state") or "").lower()
        if state not in _TERMINAL_JOB_STATES or row.get("history_cleared_at") is not None:
            continue
        if target_ids is not None:
            if row["id"] not in target_ids:
                continue
        else:
            if not project_id and not include_system:
                snap = row.get("snapshot") or {}
                wf_key = str(row.get("workflow_key") or "")
                if snap.get("native_comfy") is True or snap.get("system_smoke") is True or wf_key.startswith("system/"):
                    continue
            if scope == "failed_cancelled" and state not in _FAILED_JOB_STATES:
                continue
            if scope == "mock" and row.get("provider") != "mock":
                continue
        repo.update("jobs", row["id"], {"history_cleared_at": now}, user["id"])
        cleared += 1
    return ok({"cleared": cleared}, request)


@router.get("/jobs/{job_id}")
def get_job(job_id: str, request: Request, user=Depends(current_user), st=Depends(get_store)):
    repo = Repository(st)
    job = own(repo, "jobs", job_id, user)
    result = public_job(job)
    result["events"] = repo.list_events_for_job(user["id"], job_id, asc=True)
    result["logs"] = result["events"]
    now = time.time()
    timing_details = job_timing_details(job, result["events"], now=now)
    result["timing_details"] = timing_details
    result["timings"] = {
        name: value["seconds"]
        for name, value in timing_details.items()
        if value.get("status") == "completed" and value.get("seconds") is not None
    }
    allowed = {"id", "state", "attempt", "started_at", "finished_at", "error_code"}
    result["attempts"] = sorted(
        [{k: v for k, v in row.items() if k in allowed} for row in repo.list_executions_for_job(user["id"], job_id)],
        key=lambda row: row.get("attempt", 0),
    )
    return ok(result, request)


@router.get("/jobs/{job_id}/events")
def job_events(job_id: str, request: Request, user=Depends(current_user), st=Depends(get_store)):
    repo = Repository(st)
    own(repo, "jobs", job_id, user)
    return ok(repo.list_events_for_job(user["id"], job_id, asc=True), request)


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
    repo = Repository(st)
    own(repo, "jobs", job_id, user)
    allowed = {"id", "job_id", "state", "attempt", "started_at", "finished_at", "error_code", "created_at"}
    rows = [{k: v for k, v in row.items() if k in allowed} for row in repo.list_executions_for_job(user["id"], job_id)]
    return ok(sorted(rows, key=lambda row: row.get("attempt", 0)), request)


@router.get("/jobs/{job_id}/outputs")
def outputs(job_id: str, request: Request, user=Depends(current_user), st=Depends(get_store)):
    job = own(Repository(st), "jobs", job_id, user)
    output_ids = job.get("output_asset_ids", [])
    asset_map = st.get_many("assets", output_ids, user["id"]) if hasattr(st, "get_many") else {}
    rows = [asset_map[aid] if aid in asset_map else own(st, "assets", aid, user) for aid in output_ids]
    return ok([public_asset(a) for a in rows], request)


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
    from .gpu.specs import record_discovered_gpu_specs

    try:
        specs = GoogleColabProvider(repo, user["id"], account_id=account_id or None).list_specs()
        for spec in specs:
            rate = minute_rate(spec)
            spec["reference_rate_hourly"] = round(rate * 60, 6) if rate is not None else None
        try:
            record_discovered_gpu_specs(st, user["id"], specs)
        except Exception:
            pass
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


if os.getenv("STUDIO_ATELIER_ENABLED", "0").strip().lower() in {"1", "true"}:
    app.include_router(atelier_router)
app.include_router(account_router)
app.include_router(instance_router)
app.include_router(recovery_router)
app.include_router(native_outputs_router)
app.include_router(native_router)
app.include_router(creation_api_router)
app.add_middleware(SelectiveGZipMiddleware, minimum_size=1024)
app.add_middleware(CreationHostMiddleware, studio_app=app)
app.add_middleware(NativeHostMiddleware, studio_app=app)
app.include_router(history_router)
app.include_router(compute_router)
app.include_router(review_router)
app.include_router(oauth_config_router)
app.include_router(model_library_router)
app.include_router(assets_router)


@app.exception_handler(GoogleError)
async def google_error(request, exc):
    return error_response(request, APIError("GOOGLE_REQUEST_FAILED", str(exc)[:250], 422))


router.include_router(project_router)
router.include_router(gallery_router)
router.include_router(agent_router)
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
        app.mount("/assets", ImmutableStaticFiles(directory=_static_root / "assets"), name="web-assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path.startswith("api/"):
            raise APIError("NOT_FOUND", "接口不存在", 404)
        target = (_static_root / path).resolve()
        if path and target.is_relative_to(_static_root.resolve()) and target.is_file():
            if target.suffix == ".webmanifest":
                return FileResponse(target, media_type="application/manifest+json")
            return FileResponse(target)
        return FileResponse(_static_root / "index.html", headers={"Cache-Control": "no-cache"})

__all__ = ["AGENT_TOOLS", "app", "security_context"]
