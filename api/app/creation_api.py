"""ComfyUI-compatible submission backed by the durable Studio job queue."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import secrets
import uuid
from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .admission import _account_is_usable, _google_status, admit_job
from .api_access import (
    DEFAULT_WEAK_KEY_ID,
    build_access_urls,
    check_weak_origin,
    cors_headers_for,
    get_or_create_path_code,
    is_trusted_local_client,
    match_gateway_route,
    normalize_origins,
    peek_path_code,
    reset_path_code,
    validate_listen_port,
)
from .common import APIError, admin, current_user, get_store, ok
from .config import settings
from .db import utc_isoformat, utcnow
from .gpu.instances import POOL_CREATION, AllocationError, InstanceService, instance_pool_of
from .gpu.manifest import (
    _binding,
    _canonical_parameters,
    _check,
    _optional_image,
    _type,
    digest,
    resolve_parameters,
)
from .gpu.model_dependencies import rules as model_rules
from .gpu.repository import Repository
from .gpu.scheduler import ACTIVE_STATES, Scheduler
from .gpu.specs import (
    check_requirements,
    gpu_spec_order_settings,
    is_below_workflow_default,
    spec_id,
    spec_key,
)
from .job_schemas import JobInput, Policy
from .native_api import MAX_OUTSTANDING, MAX_WORKFLOW_BYTES, RATE_PER_MINUTE, token_digest
from .port_listeners import probe_port_available
from .storage import asset_path

admin_router = APIRouter(prefix="/api/gpu/creation-api")
compat = APIRouter()
KEYS = "creation_api_keys"
DEFAULT_GPU = {"variant": "VARIANT_GPU", "accelerator": "L4", "shape": "SHAPE_STANDARD"}

_COMPAT_API_PREFIXES = (
    "/api/system_stats",
    "/api/object_info",
    "/api/models",
    "/api/prompt",
    "/api/history",
    "/api/view",
    "/api/interrupt",
    "/api/queue",
)


async def _read_bounded_body(request: Request, limit: int, error_message: str) -> bytes:
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            raise APIError("PAYLOAD_TOO_LARGE", error_message, 413)
        chunks.append(chunk)
    return b"".join(chunks)


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    auto_start: bool = False
    account_id: str | None = None
    project_id: str | None = None
    gpu: dict = DEFAULT_GPU
    lifecycle: str = "save"
    retain_idle_minutes: int = 15
    max_runtime_minutes: int = 30
    auth_mode: Literal["strict", "url_path", "local_open"] = "strict"
    allow_lan: bool = False
    listen_port: int | None = None
    workflow_scope: Literal["active", "selected"] = "active"
    workflow_ids: list[str] = Field(default_factory=list)
    allowed_origins: list[str] = Field(default_factory=list)


def config(store, owner):
    defaults = Config().model_dump()
    saved = store.get("settings", f"creation-api:{owner}", owner)
    if saved and isinstance(saved.get("policy"), dict):
        return {**defaults, **saved["policy"]}
    return defaults


def execution_policy(cfg):
    return Policy.model_validate({
        "provider": "colab", "account_id": cfg["account_id"], "compute_spec": cfg["gpu"],
        "lifecycle": cfg["lifecycle"], "retain_idle_minutes": cfg["retain_idle_minutes"],
        "max_runtime_minutes": cfg["max_runtime_minutes"],
    }).model_dump()


def eligible(repo, owner):
    models = {item["model_id"] for item in repo.list("models", owner)
              if item.get("enabled") and not item.get("is_mock")}
    return [item for item in repo.list("workflows", owner)
            if item["state"] == "published" and item.get("review_approved")
            and item.get("smoke_test_kind") == "real" and item["model_id"] in models]


def public_workflow(row):
    return {
        **{field: row[field] for field in ("id", "workflow_key", "workflow_version", "model_id")},
        "active": bool(row.get("active")),
    }


def public_key(row):
    return {field: row.get(field) for field in ("id", "name", "workflow_ids", "created_at", "revoked_at")}


def _weak_virtual_key(store, owner: str = "owner", cfg: dict | None = None) -> dict:
    cfg = cfg or config(store, owner)
    rows = eligible(Repository(store), owner)
    if cfg.get("workflow_scope") == "selected":
        allowed = set(cfg.get("workflow_ids") or [])
        wf_ids = [r["id"] for r in rows if r["id"] in allowed]
    else:
        active_ids = [r["id"] for r in rows if r.get("active")]
        wf_ids = active_ids if active_ids else [r["id"] for r in rows]
    return {
        "id": DEFAULT_WEAK_KEY_ID,
        "owner_id": owner,
        "name": "默认弱鉴权通道",
        "workflow_ids": wf_ids,
        "virtual": True,
    }


def require_key(store, token):
    prefix, _, secret = token.partition(".")
    ident = prefix.removeprefix("cca_")
    key = store.get(KEYS, ident) if prefix.startswith("cca_") and len(ident) == 32 and len(secret) == 43 else None
    if not key or key.get("revoked_at") or not secrets.compare_digest(key["token_digest"], token_digest(token)):
        raise APIError("UNAUTHENTICATED", "创作 API Key 无效或已撤销", 401)
    return key


def authorize(request: Request, store=Depends(get_store)):
    auth_header = request.headers.get("Authorization", "")
    if auth_header:
        scheme, _, token = auth_header.partition(" ")
        key = require_key(store, token if scheme.lower() == "bearer" else "")
        request.state.cors_headers = cors_headers_for(request.headers.get("origin"), wildcard_fallback=True)
        return key

    owner = "owner"
    cfg = config(store, owner)
    route_token = getattr(request.state, "route_token", None)

    if route_token:
        if cfg["auth_mode"] not in {"url_path", "local_open"}:
            raise APIError("UNAUTHENTICATED", "当前未启用 URL 路径短码模式", 401)
        expected = peek_path_code(store, f"creation-api:{owner}", owner)
        if not expected or not secrets.compare_digest(route_token, expected):
            raise APIError("UNAUTHENTICATED", "创作 API 路径短码无效或已重置", 401)
        request.state.cors_headers = check_weak_origin(request.headers, via_token=True)
        return _weak_virtual_key(store, owner, cfg)

    if cfg["auth_mode"] != "local_open":
        raise APIError("UNAUTHENTICATED", "创作 API Key 无效或已撤销", 401)
    if not is_trusted_local_client(request.scope, request.headers, allow_lan=bool(cfg.get("allow_lan"))):
        raise APIError(
            "UNAUTHENTICATED",
            "当前仅允许本机免密访问；局域网/容器互通请开启「允许局域网/容器私网访问」或改用 URL 路径短 Token 模式",
            401,
        )
    request.state.cors_headers = check_weak_origin(
        request.headers,
        via_token=False,
        allowed_origins=cfg.get("allowed_origins"),
    )
    return _weak_virtual_key(store, owner, cfg)


async def json_body(request: Request):
    data = bytearray()
    async for chunk in request.stream():
        data.extend(chunk)
        if len(data) > MAX_WORKFLOW_BYTES:
            raise APIError("PAYLOAD_TOO_LARGE", "工作流 JSON 超过 1 MiB", 413)
    try:
        result = json.loads(data)
    except (ValueError, RecursionError) as exc:
        raise APIError("INVALID_JSON", "工作流 JSON 格式无效") from exc
    if not isinstance(result, dict):
        raise APIError("INVALID_JSON", "请求必须是 JSON 对象")
    return result


def difference(actual, expected, path):
    if isinstance(expected, dict) and isinstance(actual, dict):
        missing, extra = sorted(expected.keys() - actual.keys()), sorted(actual.keys() - expected.keys())
        if missing or extra:
            return f"{path} 缺少 {missing[:10]}，多出 {extra[:10]}"
        for key in sorted(expected):
            mismatch = difference(actual[key], expected[key], f"{path}.{key}")
            if mismatch:
                return mismatch
        return None
    if isinstance(expected, list) and isinstance(actual, list) and len(actual) == len(expected):
        for index, (value, reference) in enumerate(zip(actual, expected)):
            mismatch = difference(value, reference, f"{path}[{index}]")
            if mismatch:
                return mismatch
        return None
    if actual == expected and (type(actual) is type(expected)
                               or (type(actual) in (int, float) and type(expected) in (int, float))):
        return None
    return f"{path} 与已发布版本不同"


def match_version(graph, workflow):
    reference = workflow["api_graph"]
    if set(graph) != set(reference):
        return None, difference({k: None for k in graph}, {k: None for k in reference}, "节点")
    definitions, _ = _canonical_parameters(workflow["manifest"])
    bindings = {}
    params = {}
    weights = workflow["manifest"].get("weight_files", workflow["manifest"].get("weights")) or []
    locked_files = {name for item in weights for target in [item["target"]]
                    for name in (target, target.split("/", 1)[-1], target.rsplit("/", 1)[-1])}
    locked_fields = model_rules()
    for name, spec in definitions.items():
        if "variants" in spec:
            continue
        node, field = _binding(name, spec.get("binding"))
        node_bindings = bindings.setdefault(node, {})
        if field in node_bindings:
            return None, f"节点 {node}.inputs.{field} 有多个参数绑定"
        node_bindings[field] = (name, spec)
    for ident, node in reference.items():
        given = graph[ident]
        where = f"节点 {ident}"
        if not isinstance(given, dict):
            return None, f"{where} 必须是对象"
        extra, expected = ({k: v for k, v in value.items() if k not in {"_meta", "inputs"}}
                           for value in (given, node))
        mismatch = difference(extra, expected, where)
        if mismatch:
            return None, mismatch
        inputs = given.get("inputs")
        if not isinstance(inputs, dict):
            return None, f"{where}.inputs 必须是对象"
        fixed = node["inputs"]
        if set(inputs) != set(fixed):
            return None, difference({k: None for k in inputs}, {k: None for k in fixed}, where + ".inputs")
        for field, expected_value in fixed.items():
            value = inputs[field]
            locked = ((isinstance(expected_value, str) and expected_value in locked_files)
                      or field in locked_fields.get(node["class_type"], {}))
            binding = bindings.get(ident, {}).get(field)
            if binding is None or locked:
                mismatch = difference(value, expected_value, f"{where}.inputs.{field}")
                if mismatch:
                    return None, mismatch
                if binding is None:
                    continue
            name, spec = binding
            kind = _type(spec.get("type"))
            try:
                if kind == "image" and isinstance(value, str) and value.startswith("studio-asset:"):
                    value = str(uuid.UUID(value.removeprefix("studio-asset:")))
                elif kind in {"image", "video", "asset"} and not (_optional_image(spec) and value in (None, "")):
                    return None, f"{where}.inputs.{field}（{name}）需要 Studio 项目素材，酒馆文件名暂无法转译"
                if not (_optional_image(spec) and value in (None, "")):
                    _check(name, spec, value)
            except ValueError as exc:
                return None, f"{where}.inputs.{field}：{exc}"
            params[name] = value
    try:
        resolve_parameters(workflow["manifest"], params)
    except ValueError as exc:
        return None, str(exc)
    return params, None


def check(graph, workflows):
    matches, differences = [], []
    for row in workflows:
        params, reason = match_version(graph, row)
        if reason:
            differences.append({"workflow": public_workflow(row), "reason": reason})
        else:
            matches.append({"workflow": public_workflow(row), "params": params})
    return {"status": "matched" if len(matches) == 1 else "ambiguous" if matches else "unmatched",
            "matches": matches, "differences": differences}


def is_active(row):
    return row["state"] in ACTIVE_STATES


def warm_status(repo, row):
    instance = repo.get("instances", row.get("instance_id"), row["owner_id"]) if row.get("instance_id") else None
    runtime = (instance or {}).get("creation_runtime") or {}
    deadlines = [(instance or {}).get(k) for k in ("hard_deadline_at", "idle_deadline_at")]
    expires = min((float(d) for d in deadlines if d is not None), default=None)
    ready = bool(instance and instance["state"] == "retained_idle" and expires and expires > utcnow().timestamp()
                 and runtime.get("instance_id") == instance["id"]
                 and runtime.get("lease_id") == instance.get("lease_id")
                 and runtime.get("fingerprint") == row.get("creation_fingerprint")
                 and runtime.get("process") == (row.get("creation_runtime") or {}).get("process"))
    if ready and (binding_digest := row.get("creation_binding_digest")):
        workflow_id = row["creation_workflow_id"]
        valid = any(item["id"] == workflow_id for item in eligible(repo, row["owner_id"]))
        binding = next((item for item in repo.list("model_dependencies", row["owner_id"])
                        if item.get("workflow_id") == workflow_id), None)
        ready = valid and binding_digest == digest({"dependencies": binding["dependencies"] if binding else None})
    from .gpu.allocation_retry import public_allocation_retry

    return {**{field: row.get(field) for field in (
        "id", "state", "stage", "error_code", "error_message", "worker_timings", "instance_id", "workflow_key",
        "workflow_version", "creation_runtime", "compute_warnings", "gpu_reported",
        "auto_select_account", "retry_not_before_epoch",
    )}, "allocation_retry": public_allocation_retry(row.get("allocation_retry")), "ready": ready, "expires_at": expires if ready else None}


def default_project_id(owner: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"studio:creation-api:{owner}"))


def get_project(store, owner):
    cfg = config(store, owner)
    configured = cfg.get("project_id")
    if isinstance(configured, str) and configured:
        target = store.get("projects", configured, owner)
        if target and target.get("archived_at") is None and target.get("system") is not True:
            return configured
    ident = default_project_id(owner)
    existing = store.get("projects", ident, owner)
    if not existing:
        with store.admission_lock() as locked:
            if not locked:
                raise APIError("SCHEDULER_BUSY", "创作 API 项目初始化中，请重试", 503, True)
            existing = store.get("projects", ident, owner)
            if not existing:
                store.create("projects", owner, {"title": "创作 API"}, record_id=ident)
    elif existing.get("archived_at") is not None:
        with store.admission_lock() as locked:
            if locked:
                store.update("projects", ident, {"archived_at": None}, owner)
    return ident


def submit(store, owner, workflow, params, *, key=None, idem=None, graph_hash=None):
    cfg = config(store, owner)
    warmup = key is None
    key_id = key["id"] if key else None
    is_virtual_key = bool(key and key.get("virtual"))
    existing_only = not warmup and not cfg["auto_start"]
    key_value = (
        "creation:" + (key_id or "warmup") + ":" + hashlib.sha256(idem.encode()).hexdigest()
        if idem
        else "creation:" + uuid.uuid4().hex
    )

    raw = JobInput(project_id=get_project(store, owner), model_id=workflow["model_id"],
                   workflow_key=workflow["workflow_key"], workflow_version=workflow["workflow_version"],
                   params=params, execution_policy=Policy.model_validate(execution_policy(cfg))).model_dump()

    def _check_boot_spec():
        order_table = gpu_spec_order_settings(store, owner)["ordered_specs"]
        manifest = workflow.get("manifest") or {}
        try:
            check_requirements(manifest, {"key": cfg["gpu"]})
        except ValueError as exc:
            raise APIError("GPU_SPEC_BELOW_WORKFLOW_DEFAULT", str(exc), 422) from exc
        if is_below_workflow_default(manifest, cfg["gpu"], order_table):
            wf_default = manifest.get("compute_spec")
            raise APIError(
                "GPU_SPEC_BELOW_WORKFLOW_DEFAULT",
                f"API 默认 GPU 规格（{spec_id(cfg['gpu'])}）低于工作流默认最低规格（{spec_id(wf_default)}），不支持降配启动",
                422,
            )

    def before_create(repo, job):
        rows = repo.list("jobs", owner)
        versions = {row["id"] for row in eligible(repo, owner)}
        if workflow["id"] not in versions:
            raise APIError("WORKFLOW_UNPUBLISHED", "发布版本已撤销或不可执行", 409)
        if warmup:
            target_acct = job["execution_policy"].get("account_id")
            for row in rows:
                same_acct = (
                    (not target_acct and bool(row.get("auto_select_account")))
                    or row.get("requested_account_id") == target_acct
                )
                row_initial_spec = row.get("initial_requested_spec") or row.get("requested_spec")
                if (row.get("warmup") and row.get("creation_fingerprint") == job["creation_fingerprint"]
                        and same_acct
                        and all(row.get("execution_policy", {}).get(field) == job["execution_policy"].get(field)
                                for field in ("lifecycle", "retain_idle_minutes", "max_runtime_minutes"))
                        and row_initial_spec == job.get("requested_spec")
                        and (is_active(row) or warm_status(repo, row)["ready"])):
                    return row
            try:
                pool_state = InstanceService(repo).inspect_creation_api_admission({**raw, **job, "owner_id": owner}, rows)
            except AllocationError as exc:
                raise APIError("ACCOUNT_UNAVAILABLE", str(exc), 409) from exc
            if pool_state["status"] == "spec_too_low":
                raise APIError(
                    "CREATION_INSTANCE_SPEC_TOO_LOW",
                    "当前创作实例规格低于该工作流要求，API 不自动多开显卡，请先在算力页停止低规格实例或由前端手动启动高规格实例",
                    409,
                )
            if pool_state["status"] not in {"compatible_instance", "inflight_compatible"}:
                _check_boot_spec()
            return None
        if is_virtual_key:
            fresh_cfg = config(store, owner)
            if fresh_cfg["auth_mode"] not in {"url_path", "local_open"}:
                raise APIError("UNAUTHENTICATED", "弱鉴权接入已关闭", 401)
            virtual = _weak_virtual_key(store, owner, fresh_cfg)
            if workflow["id"] not in virtual["workflow_ids"]:
                raise APIError("WORKFLOW_FORBIDDEN", "该工作流版本未在弱鉴权通道开放", 403)
        else:
            current = store.get(KEYS, key_id, owner)
            if not current or current.get("revoked_at") or workflow["id"] not in current["workflow_ids"]:
                raise APIError("WORKFLOW_FORBIDDEN", "Key 的工作流授权已撤销", 403)
        relevant = [row for row in rows if row.get("creation_key_id") == key_id]
        for row in relevant:
            if row.get("idempotency_key") == key_value:
                if row.get("creation_body_hash") != graph_hash:
                    raise APIError("IDEMPOTENCY_CONFLICT", "相同幂等键不能用于不同工作流参数", 409)
                return row
        now = utcnow().timestamp()
        if sum(datetime.fromisoformat(row["created_at"]).timestamp() > now - 60
               for row in relevant) >= RATE_PER_MINUTE:
            raise APIError("API_RATE_LIMIT", "每把 Key 每分钟最多提交 20 次", 429, True)
        if sum(is_active(row) for row in relevant) >= MAX_OUTSTANDING:
            raise APIError("API_CONCURRENCY_LIMIT", "每把 Key 最多同时有 4 个未完成任务", 429, True)
        try:
            pool_state = InstanceService(repo).inspect_creation_api_admission({**raw, **job, "owner_id": owner}, rows)
        except AllocationError as exc:
            raise APIError("ACCOUNT_UNAVAILABLE", str(exc), 409) from exc
        if pool_state["status"] == "spec_too_low":
            raise APIError(
                "CREATION_INSTANCE_SPEC_TOO_LOW",
                "当前创作实例规格低于该工作流要求，API 不自动多开显卡，请先在算力页停止低规格实例或由前端手动启动高规格实例",
                409,
            )
        if pool_state["status"] not in {"compatible_instance", "inflight_compatible"}:
            if existing_only:
                raise APIError("CREATION_START_REQUIRED", "请先在创作 → API 设置启动工作流", 409, True)
            _check_boot_spec()
        return None

    try:
        return admit_job(store, {"id": owner}, raw, key_value, creation={
            "creation_key_id": key_id, "creation_workflow_id": workflow["id"],
            "creation_existing_only": existing_only, "creation_body_hash": graph_hash, "warmup": warmup,
        }, before_create=before_create)
    except ValueError as exc:
        raise APIError("INVALID_WORKFLOW_PARAMETERS", str(exc)) from exc


def _listener_status(app):
    mgr = getattr(app.state, "port_listeners", None)
    return mgr.status("creation") if mgr else {"active": False, "host": None, "port": None, "error": None}


def _access_metadata(store, owner: str, cfg: dict) -> tuple[str | None, dict]:
    code = (
        get_or_create_path_code(store, f"creation-api:{owner}", owner)
        if cfg.get("auth_mode") in {"url_path", "local_open"}
        else peek_path_code(store, f"creation-api:{owner}", owner)
    )
    urls = build_access_urls(
        kind="creation",
        auth_mode=cfg.get("auth_mode", "strict"),
        path_code=code,
        listen_port=cfg.get("listen_port"),
        dedicated_url=settings.creation_comfy_url,
        allow_lan=bool(cfg.get("allow_lan")),
    )
    return code, urls


def _pool_instances(repo, owner: str) -> list[dict]:
    now = utcnow().timestamp()
    rows = []
    for inst in repo.list("instances", owner):
        if inst.get("provider") != "colab" or inst.get("runtime_deleted"):
            continue
        if instance_pool_of(inst) != POOL_CREATION:
            continue
        if inst.get("state") not in {"allocating", "ready", "busy", "retained_idle"}:
            continue
        if inst.get("manual_stop_requested"):
            continue
        deadlines = [inst.get(k) for k in ("hard_deadline_at", "idle_deadline_at") if inst.get(k) is not None]
        expires = min((float(d) for d in deadlines), default=None)
        if inst.get("state") == "retained_idle" and expires is not None and expires <= now:
            continue
        rows.append({
            "id": inst["id"],
            "instance_label": inst.get("instance_label"),
            "account_id": inst.get("account_id"),
            "account_alias_snapshot": inst.get("account_alias_snapshot"),
            "state": inst["state"],
            "retention_mode": inst.get("retention_mode"),
            "retain_idle_minutes": inst.get("retain_idle_minutes"),
            "idle_deadline_at": inst.get("idle_deadline_at"),
            "hard_deadline_at": inst.get("hard_deadline_at"),
            "expires_at": expires,
            "runtime_spec_requested": inst.get("runtime_spec_requested"),
            "runtime_spec_reported": inst.get("runtime_spec_reported"),
            "gpu_reported": inst.get("gpu_reported"),
            "creation_runtime": inst.get("creation_runtime"),
            "current_job_id": inst.get("current_job_id"),
            "instance_pool": POOL_CREATION,
        })
    return rows


@admin_router.get("")
def admin_status(request: Request, user=Depends(current_user), store=Depends(get_store)):
    admin(user)
    owner = user["id"]
    repo = Repository(store)
    cfg = config(store, owner)
    path_code, urls = _access_metadata(store, owner, cfg)
    warmups = []
    for row in repo.list("jobs", owner):
        if not row.get("warmup"):
            continue
        status = warm_status(repo, row)
        if row.get("history_cleared_at") is not None and not is_active(row) and not status["ready"]:
            continue
        warmups.append(status)
        if len(warmups) >= 20:
            break
    return ok({
        "url": settings.creation_comfy_url,
        "path_code": path_code,
        "urls": urls,
        "listener": _listener_status(request.app),
        "config": cfg,
        "default_project_id": default_project_id(owner),
        "keys": [public_key(row) for row in store.list(KEYS, owner)],
        "workflows": [public_workflow(row) for row in eligible(repo, owner)],
        "warmups": warmups,
        "pool_instances": _pool_instances(repo, owner),
    }, request)


@admin_router.post("/reset-code")
def rotate_path_code(request: Request, user=Depends(current_user), store=Depends(get_store)):
    admin(user)
    owner = user["id"]
    cfg = config(store, owner)
    code = reset_path_code(store, f"creation-api:{owner}", owner)
    urls = build_access_urls(
        kind="creation",
        auth_mode=cfg.get("auth_mode", "strict"),
        path_code=code,
        listen_port=cfg.get("listen_port"),
        dedicated_url=settings.creation_comfy_url,
        allow_lan=bool(cfg.get("allow_lan")),
    )
    response = ok({"path_code": code, "urls": urls}, request)
    response.headers["Cache-Control"] = "no-store"
    return response


@admin_router.post("/warmups/clear")
def clear_warmups(request: Request, user=Depends(current_user), store=Depends(get_store)):
    admin(user)
    owner = user["id"]
    repo = Repository(store)
    now = utcnow().timestamp()
    cleared = 0
    for row in repo.list("jobs", owner):
        if not row.get("warmup") or row.get("history_cleared_at") is not None or is_active(row):
            continue
        if warm_status(repo, row)["ready"]:
            continue
        repo.update("jobs", row["id"], {"history_cleared_at": now}, owner)
        cleared += 1
    return ok({"cleared": cleared}, request)


@admin_router.put("/config")
async def save_config(body: Config, request: Request, user=Depends(current_user), store=Depends(get_store)):
    admin(user)
    owner = user["id"]
    def _validate_cfg():
        cfg_local = body.model_dump()
        repo = Repository(store)
        try:
            cfg_local["gpu"] = spec_key(cfg_local["gpu"])
            if cfg_local["gpu"]["variant"] != "VARIANT_GPU":
                raise ValueError("请选择 GPU 计算规格")
            execution_policy(cfg_local)
        except (ValidationError, ValueError) as exc:
            raise APIError("CREATION_CONFIG_INVALID", str(exc)) from exc
        cfg_local["listen_port"] = validate_listen_port(cfg_local["listen_port"], kind="creation", store=store, owner=owner)
        cfg_local["allowed_origins"] = normalize_origins(cfg_local["allowed_origins"])
        cfg_local["workflow_ids"] = list(dict.fromkeys(cfg_local["workflow_ids"]))
        if len(cfg_local["workflow_ids"]) > 100:
            raise APIError("CREATION_CONFIG_INVALID", "选择的发布版本数量过多")
        eligible_ids = {row["id"] for row in eligible(repo, owner)}
        if set(cfg_local["workflow_ids"]) - eligible_ids:
            raise APIError("WORKFLOW_FORBIDDEN", "只能授权仍可执行且通过真实冒烟的发布版本", 409)
        if cfg_local["account_id"]:
            account = repo.get("accounts", cfg_local["account_id"], owner)
            if not account or not _account_is_usable(account) or not _google_status(repo, owner, cfg_local["account_id"])["colab"]["connected"]:
                raise APIError("ACCOUNT_UNAVAILABLE", "请选择已经授权且允许分配的 Colab 账号", 409)
        raw_project_id = (cfg_local.get("project_id") or "").strip()
        if not raw_project_id or raw_project_id == default_project_id(owner):
            cfg_local["project_id"] = None
        else:
            target_project = store.get("projects", raw_project_id, owner)
            if not target_project or target_project.get("archived_at") is not None or target_project.get("system") is True:
                raise APIError("CREATION_PROJECT_UNAVAILABLE", "请选择有效且未归档的普通项目", 409)
            cfg_local["project_id"] = raw_project_id
        return cfg_local

    prev_cfg = await asyncio.to_thread(config, store, owner)
    cfg = await asyncio.to_thread(_validate_cfg)

    mgr = getattr(request.app.state, "port_listeners", None)
    if mgr is not None:
        await mgr.apply("creation", cfg["listen_port"], allow_lan=cfg["allow_lan"], strict=True)
    elif cfg["listen_port"] is not None:
        await asyncio.to_thread(probe_port_available, cfg["listen_port"], allow_lan=cfg["allow_lan"])

    def _persist_cfg():
        ident = f"creation-api:{owner}"
        with store.admission_lock() as acquired:
            if not acquired:
                raise APIError("SCHEDULER_BUSY", "正在保存运行设置，请重试", 503)
            existing = store.get("settings", ident, owner)
            patch = {"policy": cfg}
            if cfg["auth_mode"] in {"url_path", "local_open"} and not (existing and existing.get("path_code")):
                patch["path_code"] = secrets.token_urlsafe(18)
            if existing:
                store.update("settings", ident, patch, owner)
            else:
                store.create("settings", owner, patch, record_id=ident)

    try:
        await asyncio.to_thread(_persist_cfg)
    except Exception:
        if mgr is not None:
            await mgr.apply(
                "creation",
                prev_cfg.get("listen_port"),
                allow_lan=bool(prev_cfg.get("allow_lan")),
                strict=False,
            )
        raise
    return ok(cfg, request)


@admin_router.post("/keys")
@admin_router.put("/keys/{key_id}")
def save_key(body: dict, request: Request, key_id: str | None = None,
             user=Depends(current_user), store=Depends(get_store)):
    admin(user)
    owner = user["id"]
    name, version_ids = body.get("name"), body.get("workflow_ids")
    if (not isinstance(name, str) or not 1 <= len(name.strip()) <= 80
            or not isinstance(version_ids, list) or len(version_ids) > 100
            or any(not isinstance(item, str) for item in version_ids)):
        raise APIError("CREATION_KEY_INVALID", "请填写 Key 名称并选择获准的发布版本")
    data = {"name": name.strip(), "workflow_ids": list(dict.fromkeys(version_ids))}
    repo = Repository(store)
    with store.admission_lock() as acquired:
        if not acquired:
            raise APIError("SCHEDULER_BUSY", "正在修改接入权限，请重试", 503)
        if set(version_ids) - {row["id"] for row in eligible(repo, owner)}:
            raise APIError("WORKFLOW_FORBIDDEN", "只能授权仍可执行且通过真实冒烟的发布版本", 409)
        if key_id:
            row = store.get(KEYS, key_id, owner)
            if not row or row.get("revoked_at"):
                raise APIError("NOT_FOUND", "接入 Key 不存在或已撤销", 404)
            return ok(public_key(store.update(KEYS, key_id, data, owner)), request)
        ident = uuid.uuid4().hex
        token = f"cca_{ident}.{secrets.token_urlsafe(32)}"
        row = store.create(KEYS, owner, {**data, "token_digest": token_digest(token)}, record_id=ident)
        return ok({**public_key(row), "token": token}, request, 201)


@admin_router.delete("/keys/{key_id}")
def revoke_key(key_id: str, request: Request, user=Depends(current_user), store=Depends(get_store)):
    admin(user)
    with store.admission_lock() as acquired:
        if not acquired:
            raise APIError("SCHEDULER_BUSY", "正在修改接入权限，请重试", 503)
        row = store.get(KEYS, key_id, user["id"])
        if not row:
            raise APIError("NOT_FOUND", "接入 Key 不存在", 404)
        if not row.get("revoked_at"):
            store.update(KEYS, key_id, {"revoked_at": utc_isoformat(utcnow())}, user["id"])
    return ok({"id": key_id}, request)


@admin_router.post("/check")
def inspect_workflow(request: Request, user=Depends(current_user), store=Depends(get_store), body=Depends(json_body)):
    admin(user)
    owner = user["id"]
    rows = eligible(Repository(store), owner)
    target_key_id = body.get("key_id")
    if target_key_id == DEFAULT_WEAK_KEY_ID:
        allowed_ids = set(_weak_virtual_key(store, owner)["workflow_ids"])
        rows = [row for row in rows if row["id"] in allowed_ids]
    elif target_key_id:
        key = store.get(KEYS, target_key_id, owner)
        if not key or key.get("revoked_at"):
            raise APIError("NOT_FOUND", "接入 Key 不存在或已撤销", 404)
        rows = [row for row in rows if row["id"] in key["workflow_ids"]]
    if not isinstance(body.get("workflow"), dict):
        raise APIError("INVALID_GRAPH", "请提供完整 ComfyUI API JSON")
    return ok(check(body["workflow"], rows), request)


@admin_router.post("/warmups")
def warm_workflow(body: dict, request: Request, user=Depends(current_user), store=Depends(get_store)):
    admin(user)
    owner = user["id"]
    repo = Repository(store)
    row = next((row for row in eligible(repo, owner) if row["id"] == body.get("workflow_id")), None)
    if not row:
        raise APIError("WORKFLOW_UNPUBLISHED", "请选择仍可执行且通过真实冒烟的发布版本", 404)
    return ok(warm_status(repo, submit(store, owner, row, {})), request, 202)


@admin_router.get("/warmups/{job_id}")
def warm_progress(job_id: str, request: Request, user=Depends(current_user), store=Depends(get_store)):
    admin(user)
    repo = Repository(store)
    row = repo.get("jobs", job_id, user["id"])
    if not row or not row.get("warmup"):
        raise APIError("NOT_FOUND", "启动任务不存在", 404)
    return ok(warm_status(repo, row), request)


def authorized_workflows(store, key):
    return [row for row in eligible(Repository(store), key["owner_id"]) if row["id"] in key["workflow_ids"]]


@compat.get("/system_stats")
def system_stats(store=Depends(get_store), key=Depends(authorize)):
    cfg = config(store, key["owner_id"])
    return {"system": {"comfyui_version": "Studio", "argv": []},
            "devices": [{"name": cfg["gpu"]["accelerator"], "type": "cuda", "index": 0}]}


def node_metadata(rows):
    result = {}
    for row in rows:
        for name, value in row["object_info"].items():
            target = result.setdefault(name, copy.deepcopy(value))
            for group in ("required", "optional"):
                inputs = target.setdefault("input", {}).setdefault(group, {})
                for field, definition in value.get("input", {}).get(group, {}).items():
                    current = inputs.setdefault(field, copy.deepcopy(definition))
                    if current and definition and isinstance(current[0], list) and isinstance(definition[0], list):
                        current[0].extend(item for item in definition[0] if item not in current[0])
    # Tavern 1.19 reads these fields unconditionally, even for workflows using other loaders/samplers.
    for node, field in (("CheckpointLoaderSimple", "ckpt_name"), ("UNETLoader", "unet_name"),
                        ("KSampler", "sampler_name"), ("KSampler", "scheduler"), ("VAELoader", "vae_name")):
        result.setdefault(node, {}).setdefault("input", {}).setdefault("required", {}).setdefault(field, [[]])
    return result


@compat.get("/object_info")
@compat.get("/object_info/{node}")
def object_info(node: str | None = None, store=Depends(get_store), key=Depends(authorize)):
    data = node_metadata(authorized_workflows(store, key))
    return {node: data[node]} if node in data else {} if node else data


def model_inventory(rows):
    from .gpu.model_dependencies import split_target_path

    models = {}
    for row in rows:
        manifest = row["manifest"]
        for item in manifest.get("weight_files", manifest.get("weights")) or []:
            target = item.get("target", "")
            folder, sep, name = target.partition("/")
            if sep:
                models.setdefault(folder, set()).add(name)
            split = split_target_path(target)
            if split and split[0] != folder:
                models.setdefault(split[0], set()).add(split[1])
    return models


@compat.get("/models")
@compat.get("/models/{folder}")
def models(folder: str | None = None, store=Depends(get_store), key=Depends(authorize)):
    inventory = model_inventory(authorized_workflows(store, key))
    return sorted(inventory.get(folder, [])) if folder else sorted(inventory)


@compat.get("/queue")
def queue_status(store=Depends(get_store), key=Depends(authorize)):
    repo = Repository(store)
    allowed_wf = set(key["workflow_ids"])
    active_rows = (
        repo.list_by_states("jobs", ACTIVE_STATES, key["owner_id"])
        if hasattr(repo, "list_by_states")
        else repo.list("jobs", key["owner_id"])
    )
    rows = [
        row for row in active_rows
        if row.get("creation_key_id") == key["id"] and row.get("creation_workflow_id") in allowed_wf and is_active(row)
    ]
    rows.sort(key=lambda r: r.get("created_at") or "")
    running, pending = [], []
    for index, row in enumerate(rows):
        entry = [index, row["id"]]
        if row["state"] == "running":
            running.append(entry)
        else:
            pending.append(entry)
    return {"queue_running": running, "queue_pending": pending}


@compat.post("/queue")
async def queue_manage(request: Request, store=Depends(get_store), key=Depends(authorize)):
    raw = await _read_bounded_body(request, 65536, "队列请求过大")
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        raise APIError("INVALID_JSON", "队列请求必须为有效 JSON") from exc
    deletes = payload.get("delete") if isinstance(payload, dict) else None
    if isinstance(deletes, list):
        def _cancel_many():
            repo = Repository(store)
            sched = Scheduler(store)
            allowed_wf = set(key["workflow_ids"])
            for pid in deletes[:20]:
                if isinstance(pid, str):
                    job = repo.get("jobs", pid, key["owner_id"])
                    if (job and job.get("creation_key_id") == key["id"]
                            and job.get("creation_workflow_id") in allowed_wf and is_active(job)):
                        sched.cancel(key["owner_id"], job["id"])

        await asyncio.to_thread(_cancel_many)
    return {}


@compat.post("/prompt")
def prompt(request: Request, store=Depends(get_store), key=Depends(authorize), body=Depends(json_body)):
    graph = body.get("prompt")
    if not isinstance(graph, dict) or not graph:
        raise APIError("INVALID_GRAPH", "需要标准的 ComfyUI prompt API 图")
    idem = request.headers.get("Idempotency-Key")
    if idem is not None and not 1 <= len(idem) <= 200:
        raise APIError("IDEMPOTENCY_INVALID", "幂等键长度须为 1 至 200")
    rows = authorized_workflows(store, key)
    try:
        result = check(graph, rows)
        graph_hash = digest({node: {key: value for key, value in item.items() if key != "_meta"}
                             for node, item in graph.items() if isinstance(item, dict)})
    except (ValueError, RecursionError) as exc:
        raise APIError("INVALID_GRAPH", "工作流包含无效的 JSON 值") from exc
    if result["status"] != "matched":
        raise APIError("WORKFLOW_MATCH_" + result["status"].upper(),
                       "工作流匹配存在歧义，请检查授权版本范围" if result["matches"]
                       else "工作流与已开放的发布版本不匹配", 422)
    match = result["matches"][0]
    row = next(row for row in rows if row["id"] == match["workflow"]["id"])
    job = submit(store, key["owner_id"], row, match["params"], key=key, idem=idem, graph_hash=graph_hash)
    return {"prompt_id": job["id"], "number": 0, "node_errors": {}}


def own_job(repo, key, prompt_id):
    job = repo.get("jobs", prompt_id, key["owner_id"])
    if (not job or job.get("creation_key_id") != key["id"]
            or job.get("creation_workflow_id") not in key["workflow_ids"]):
        raise APIError("NOT_FOUND", "任务不存在", 404)
    return job


@compat.get("/history/{prompt_id}")
def history(prompt_id: str, store=Depends(get_store), key=Depends(authorize)):
    job = own_job(Repository(store), key, prompt_id)
    if is_active(job):
        return {}
    success = job["state"] == "completed"
    outputs = {}
    if success:
        for ident in job.get("output_asset_ids", []):
            asset = store.get("assets", ident, key["owner_id"])
            if not asset or asset.get("source_job_id") != job["id"] or not asset.get("comfy_node"):
                continue
            outputs.setdefault(asset["comfy_node"], {}).setdefault(asset["comfy_category"], []).append({
                "filename": asset["filename"], "subfolder": ident, "type": "output",
            })
    message = {"prompt_id": prompt_id, "node_id": "", "node_type": "Studio",
               "exception_type": job.get("error_code") or job["state"],
               "exception_message": job.get("error_message") or "任务已取消"}
    status = {"status_str": "success" if success else "error", "completed": success,
              "messages": [] if success else [["execution_error", message]]}
    return {prompt_id: {"status": status, "outputs": outputs}}


@compat.get("/view")
def view(filename: str, subfolder: str = "", type: str = "output", store=Depends(get_store), key=Depends(authorize)):
    asset = store.get("assets", subfolder, key["owner_id"]) if subfolder else None
    if not asset or type != "output" or not asset.get("comfy_node") or asset["filename"] != filename:
        raise APIError("NOT_FOUND", "产物不存在", 404)
    job = own_job(Repository(store), key, asset["source_job_id"])
    if job["state"] != "completed" or subfolder not in job.get("output_asset_ids", []):
        raise APIError("NOT_FOUND", "产物不存在", 404)
    path = asset_path(asset)
    if not path.is_file():
        raise APIError("NOT_FOUND", "产物文件不存在", 404)
    return FileResponse(path, media_type=asset["mime"])


@compat.post("/interrupt")
async def interrupt(request: Request, prompt_id: str | None = None, store=Depends(get_store), key=Depends(authorize)):
    if not prompt_id:
        raw = await _read_bounded_body(request, 8192, "取消请求过大")
        if raw:
            try:
                payload = json.loads(raw)
                if isinstance(payload, dict) and isinstance(payload.get("prompt_id"), str):
                    prompt_id = payload["prompt_id"].strip()
            except ValueError:
                pass
    if not prompt_id:
        # Tavern 1.19 legacy disconnect sends no prompt ID; BaiBai sends {"prompt_id": "..."}.
        raise APIError("PROMPT_ID_REQUIRED", "取消任务需要明确的 prompt_id", 409)

    def _cancel_one():
        repo = Repository(store)
        job = own_job(repo, key, prompt_id)
        Scheduler(store).cancel(key["owner_id"], job["id"])

    await asyncio.to_thread(_cancel_one)
    return {}


def _strip_optional_compat_api_prefix(scope: dict) -> dict:
    path = scope.get("path", "")
    if any(path == p or path.startswith(p + "/") for p in _COMPAT_API_PREFIXES):
        updated = dict(scope)
        updated["path"] = path.removeprefix("/api")
        raw_path = scope.get("raw_path")
        if isinstance(raw_path, bytes) and raw_path.startswith(b"/api/"):
            updated["raw_path"] = raw_path[4:]
        return updated
    return scope


def _preflight_cors_headers(request: Request, store) -> dict[str, str]:
    origin = request.headers.get("origin")
    req_headers = request.headers.get("access-control-request-headers", "").lower()
    if "authorization" in req_headers:
        return cors_headers_for(origin, wildcard_fallback=True)
    cfg = config(store, "owner")
    route_token = getattr(request.state, "route_token", None)
    if route_token and cfg["auth_mode"] in {"url_path", "local_open"}:
        expected = peek_path_code(store, "creation-api:owner", "owner")
        if expected and secrets.compare_digest(route_token, expected):
            return check_weak_origin(request.headers, via_token=True)
    if cfg["auth_mode"] == "local_open" and is_trusted_local_client(
        request.scope, request.headers, allow_lan=bool(cfg.get("allow_lan"))
    ):
        return check_weak_origin(request.headers, via_token=False, allowed_origins=cfg.get("allowed_origins"))
    return {}


def _get_compat_app(studio_app):
    cached = getattr(studio_app.state, "_creation_compat_app", None)
    if cached is not None:
        return cached
    from fastapi import FastAPI

    sub = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    sub.state = studio_app.state

    @sub.exception_handler(APIError)
    async def error(request: Request, exc: APIError):
        headers = {"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"}
        cors = getattr(request.state, "cors_headers", None)
        if cors:
            headers.update(cors)
        return JSONResponse(
            {"error": {"type": exc.code, "message": exc.message}, "node_errors": {}},
            status_code=exc.status,
            headers=headers,
        )

    @sub.middleware("http")
    async def private(request: Request, call_next):
        if request.method == "OPTIONS":
            try:
                cors = await asyncio.to_thread(_preflight_cors_headers, request, studio_app.state.store)
            except APIError as exc:
                return JSONResponse(
                    {"error": {"type": exc.code, "message": exc.message}, "node_errors": {}},
                    status_code=exc.status,
                    headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"},
                )
            headers = {"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff", **cors}
            return Response(status_code=204 if cors else 403, headers=headers)
        response = await call_next(request)
        response.headers.update({"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"})
        cors = getattr(request.state, "cors_headers", None)
        if cors:
            response.headers.update(cors)
        return response

    sub.include_router(compat)
    studio_app.state._creation_compat_app = sub
    return sub


async def dispatch_creation_scope(studio_app, scope, receive, send, *, dedicated_port: bool = False) -> bool:
    matched = match_gateway_route(
        scope,
        kind="creation",
        dedicated_host_url=settings.creation_comfy_url,
        dedicated_port=dedicated_port,
    )
    if matched is None:
        return False
    if scope["type"] == "websocket":
        await send({"type": "websocket.close", "code": 1008})
        return True
    if scope["type"] != "http":
        return False
    rewritten = _strip_optional_compat_api_prefix(matched["scope"])
    state = dict(rewritten.get("state") or {})
    state["route_token"] = matched["route_token"]
    rewritten = {**rewritten, "state": state}
    compat_app = _get_compat_app(studio_app)
    await compat_app(rewritten, receive, send)
    return True


class CreationHostMiddleware:
    def __init__(self, app, studio_app):
        self.app = app
        self.studio_app = studio_app

    async def __call__(self, scope, receive, send):
        if scope["type"] in {"http", "websocket"}:
            handled = await dispatch_creation_scope(self.studio_app, scope, receive, send, dedicated_port=False)
            if handled:
                return
        await self.app(scope, receive, send)
