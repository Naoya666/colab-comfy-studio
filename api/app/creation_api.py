"""ComfyUI-compatible submission backed by the durable Studio job queue."""

import copy
import hashlib
import json
import secrets
import uuid
from datetime import datetime
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, ValidationError

from .admission import _account_is_usable, _google_status, admit_job
from .common import APIError, admin, current_user, get_store, ok
from .config import settings
from .db import utc_isoformat, utcnow
from .gpu.compute import GpuConfig
from .gpu.instances import InstanceService
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
from .gpu.specs import spec_key
from .job_schemas import JobInput, Policy
from .native_api import MAX_OUTSTANDING, MAX_WORKFLOW_BYTES, RATE_PER_MINUTE, token_digest
from .storage import asset_path

admin_router = APIRouter(prefix="/api/gpu/creation-api")
compat = APIRouter()
KEYS = "creation_api_keys"
DEFAULT_GPU = {"variant": "VARIANT_GPU", "accelerator": "L4", "shape": "SHAPE_STANDARD"}


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    auto_start: bool = False
    account_id: str | None = None
    gpu: dict = DEFAULT_GPU
    lifecycle: str = "save"
    retain_idle_minutes: int = 15
    max_runtime_minutes: int = 30


def config(store, owner):
    saved = store.get("settings", f"creation-api:{owner}", owner)
    if saved:
        return saved["policy"]
    return Config(max_runtime_minutes=min(30, GpuConfig.from_env().max_runtime_minutes)).model_dump()


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
    return {field: row[field] for field in ("id", "workflow_key", "workflow_version", "model_id")}


def public_key(row):
    return {field: row.get(field) for field in ("id", "name", "workflow_ids", "created_at", "revoked_at")}


def require_key(store, token):
    prefix, _, secret = token.partition(".")
    ident = prefix.removeprefix("cca_")
    key = store.get(KEYS, ident) if prefix.startswith("cca_") and len(ident) == 32 and len(secret) == 43 else None
    if not key or key.get("revoked_at") or not secrets.compare_digest(key["token_digest"], token_digest(token)):
        raise APIError("UNAUTHENTICATED", "创作 API Key 无效或已撤销", 401)
    return key


def authorize(request: Request, store=Depends(get_store)):
    scheme, _, token = request.headers.get("Authorization", "").partition(" ")
    return require_key(store, token if scheme.lower() == "bearer" else "")


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
    return {**{field: row.get(field) for field in (
        "id", "state", "stage", "error_code", "error_message", "worker_timings", "instance_id", "workflow_key",
        "workflow_version", "creation_runtime", "compute_warnings", "gpu_reported",
    )}, "ready": ready, "expires_at": expires if ready else None}


def get_project(store, owner):
    ident = str(uuid.uuid5(uuid.NAMESPACE_URL, f"studio:creation-api:{owner}"))
    if not store.get("projects", ident, owner):
        with store.admission_lock() as locked:
            if not locked:
                raise APIError("SCHEDULER_BUSY", "创作 API 项目初始化中，请重试", 503, True)
            if not store.get("projects", ident, owner):
                store.create("projects", owner, {"title": "创作 API"}, record_id=ident)
    return ident


def submit(store, owner, workflow, params, *, key=None, idem=None, graph_hash=None):
    repo = Repository(store)
    cfg = config(store, owner)
    warmup = key is None
    key_id = key["id"] if key else None
    existing_only = not warmup and not cfg["auto_start"]
    key_value = "creation:" + key_id + ":" + hashlib.sha256(idem.encode()).hexdigest() if idem else (
        "creation:" + uuid.uuid4().hex
    )
    if idem:
        prior = next((row for row in repo.list("jobs", owner) if row.get("idempotency_key") == key_value), None)
        if prior:
            if prior.get("creation_body_hash") != graph_hash:
                raise APIError("IDEMPOTENCY_CONFLICT", "相同幂等键不能用于不同工作流参数", 409)
            return prior

    raw = JobInput(project_id=get_project(store, owner), model_id=workflow["model_id"],
                   workflow_key=workflow["workflow_key"], workflow_version=workflow["workflow_version"],
                   params=params, execution_policy=Policy.model_validate(execution_policy(cfg))).model_dump()

    def before_create(repo, job):
        rows = repo.list("jobs", owner)
        versions = {row["id"] for row in eligible(repo, owner)}
        if workflow["id"] not in versions:
            raise APIError("WORKFLOW_UNPUBLISHED", "发布版本已撤销或不可执行", 409)
        if warmup:
            for row in rows:
                if (row.get("warmup") and row.get("creation_fingerprint") == job["creation_fingerprint"]
                        and row.get("requested_account_id") == job["execution_policy"].get("account_id")
                        and all(row.get("execution_policy", {}).get(field) == job["execution_policy"].get(field)
                                for field in ("lifecycle", "retain_idle_minutes", "max_runtime_minutes"))
                        and row.get("requested_spec") == job.get("requested_spec")
                        and (is_active(row) or warm_status(repo, row)["ready"])):
                    return row
            return None
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
        if existing_only:
            from .gpu.instances import AllocationError

            try:
                reusable = InstanceService(repo).creation_capacity({**raw, **job, "owner_id": owner}, rows)
            except AllocationError as exc:
                raise APIError("ACCOUNT_UNAVAILABLE", str(exc), 409) from exc
            if not reusable:
                raise APIError("CREATION_START_REQUIRED", "请先在创作 → API 设置启动工作流", 409, True)
        return None

    try:
        return admit_job(store, {"id": owner}, raw, key_value, creation={
            "creation_key_id": key_id, "creation_workflow_id": workflow["id"],
            "creation_existing_only": existing_only, "creation_body_hash": graph_hash, "warmup": warmup,
        }, before_create=before_create)
    except ValueError as exc:
        raise APIError("INVALID_WORKFLOW_PARAMETERS", str(exc)) from exc


@admin_router.get("")
def admin_status(request: Request, user=Depends(current_user), store=Depends(get_store)):
    admin(user)
    owner = user["id"]
    repo = Repository(store)
    return ok({"url": settings.creation_comfy_url, "config": config(store, owner),
               "max_runtime_limit": GpuConfig.from_env().max_runtime_minutes,
               "keys": [public_key(row) for row in store.list(KEYS, owner)],
               "workflows": [public_workflow(row) for row in eligible(repo, owner)],
               "warmups": [warm_status(repo, row) for row in repo.list("jobs", owner) if row.get("warmup")][:20]}, request)


@admin_router.put("/config")
def save_config(body: Config, request: Request, user=Depends(current_user), store=Depends(get_store)):
    admin(user)
    owner = user["id"]
    cfg = body.model_dump()
    repo = Repository(store)
    try:
        cfg["gpu"] = spec_key(cfg["gpu"])
        if cfg["gpu"]["variant"] != "VARIANT_GPU":
            raise ValueError("请选择 GPU 计算规格")
        execution_policy(cfg)
        if cfg["max_runtime_minutes"] > GpuConfig.from_env().max_runtime_minutes:
            raise ValueError("单个任务最长运行时间超过服务端上限")
    except (ValidationError, ValueError) as exc:
        raise APIError("CREATION_CONFIG_INVALID", str(exc)) from exc
    if cfg["account_id"]:
        account = repo.get("accounts", cfg["account_id"], owner)
        if not account or not _account_is_usable(account) or not _google_status(repo, owner, cfg["account_id"])["colab"]["connected"]:
            raise APIError("ACCOUNT_UNAVAILABLE", "请选择已经授权且允许分配的 Colab 账号", 409)
    ident = f"creation-api:{owner}"
    with store.admission_lock() as acquired:
        if not acquired:
            raise APIError("SCHEDULER_BUSY", "正在保存运行设置，请重试", 503)
        if store.get("settings", ident, owner):
            store.update("settings", ident, {"policy": cfg}, owner)
        else:
            store.create("settings", owner, {"policy": cfg}, record_id=ident)
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
    if body.get("key_id"):
        key = store.get(KEYS, body["key_id"], owner)
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
    models = {}
    for row in rows:
        manifest = row["manifest"]
        for item in manifest.get("weight_files", manifest.get("weights")) or []:
            folder, sep, name = item.get("target", "").partition("/")
            if sep:
                models.setdefault(folder, set()).add(name)
    return models


@compat.get("/models")
@compat.get("/models/{folder}")
def models(folder: str | None = None, store=Depends(get_store), key=Depends(authorize)):
    inventory = model_inventory(authorized_workflows(store, key))
    return sorted(inventory.get(folder, [])) if folder else sorted(inventory)


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
                       "工作流匹配存在歧义，请检查 Key 的授权版本" if result["matches"]
                       else "工作流与该 Key 授权的发布版本不匹配", 422)
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
def interrupt(prompt_id: str | None = None, store=Depends(get_store), key=Depends(authorize)):
    repo = Repository(store)
    if not prompt_id:
        # Tavern sends no prompt ID. A shared Key cannot safely identify the disconnected request.
        raise APIError("PROMPT_ID_REQUIRED", "取消任务需要明确的 prompt_id", 409)
    job = own_job(repo, key, prompt_id)
    Scheduler(store).cancel(key["owner_id"], job["id"])
    return {}


class CreationHostMiddleware:
    def __init__(self, app, studio_app):
        self.app = app
        self.studio_app = studio_app
        self.compat_app = None

    async def __call__(self, scope, receive, send):
        host = next((value.decode("latin1").lower() for name, value in scope.get("headers", [])
                     if name == b"host"), "")
        target = urlsplit(settings.creation_comfy_url).netloc.lower()
        if not target or host != target:
            return await self.app(scope, receive, send)
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        if self.compat_app is None:
            from fastapi import FastAPI

            app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
            app.state = self.studio_app.state

            @app.exception_handler(APIError)
            async def error(request, exc):
                return JSONResponse({"error": {"type": exc.code, "message": exc.message}, "node_errors": {}},
                                    status_code=exc.status)

            @app.middleware("http")
            async def private(request, call_next):
                response = await call_next(request)
                response.headers.update({"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"})
                return response

            app.include_router(compat)
            self.compat_app = app
        await self.compat_app(scope, receive, send)
