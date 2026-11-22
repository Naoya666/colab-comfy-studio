"""Studio control plane for native ComfyUI. Data plane lives on its own host, subpath, or port."""
import asyncio
import re
import threading
import time
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
from fastapi import APIRouter, Depends, Request

from . import native_api, native_model_api
from .api_access import build_access_urls, get_or_create_path_code, peek_path_code, reset_path_code
from .common import APIError, admin, current_user, ok
from .gpu import native, native_config, native_restart
from .gpu.colab import RUNTIME_MISSING, RUNTIME_MISSING_MESSAGE
from .port_listeners import probe_port_available
from .security import sign_session

router = APIRouter(prefix="/api/gpu/native")

_TELEMETRY_CACHE_TTL = 4.0
_TELEMETRY_LOCK = threading.Lock()
_TELEMETRY_CACHE: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}


def _int_or_none(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
        return parsed if parsed >= 0 else None
    except (TypeError, ValueError):
        return None


def _parse_telemetry(
    lease: dict,
    storage_cfg: dict,
    stats_payload: dict | None,
    queue_payload: dict | None,
    state_payload: dict | None,
) -> dict[str, Any] | None:
    if not isinstance(stats_payload, dict) or not isinstance(stats_payload.get("system"), dict):
        return None
    system = stats_payload["system"]
    devices = stats_payload.get("devices")
    device = None
    if isinstance(devices, list) and devices:
        device = next(
            (d for d in devices if isinstance(d, dict) and str(d.get("type") or "").lower() in {"cuda", "mps", "xpu"}),
            devices[0] if isinstance(devices[0], dict) else None,
        )

    vram_total = _int_or_none(device.get("vram_total")) if device else None
    vram_free = _int_or_none(device.get("vram_free")) if device else None
    vram_used = max(0, vram_total - vram_free) if vram_total is not None and vram_free is not None else None
    torch_vram_total = _int_or_none(device.get("torch_vram_total")) if device else None
    torch_vram_free = _int_or_none(device.get("torch_vram_free")) if device else None
    torch_vram_used = (
        max(0, torch_vram_total - torch_vram_free)
        if torch_vram_total is not None and torch_vram_free is not None
        else None
    )

    ram_total = _int_or_none(system.get("ram_total"))
    ram_free = _int_or_none(system.get("ram_free"))
    ram_used = max(0, ram_total - ram_free) if ram_total is not None and ram_free is not None else None

    queue_running = None
    queue_pending = None
    if isinstance(queue_payload, dict):
        running = queue_payload.get("queue_running")
        pending = queue_payload.get("queue_pending")
        if isinstance(running, list):
            queue_running = len(running)
        if isinstance(pending, list):
            queue_pending = len(pending)

    disk_free = _int_or_none(state_payload.get("disk_free_bytes")) if isinstance(state_payload, dict) else None
    vfs_cache = _int_or_none(state_payload.get("vfs_cache_bytes")) if isinstance(state_payload, dict) else None
    state_busy = state_payload.get("busy") if isinstance(state_payload, dict) else None
    busy_reason = state_payload.get("busy_reason") if isinstance(state_payload, dict) else None
    if not busy_reason and ((queue_running or 0) > 0 or (queue_pending or 0) > 0):
        busy_reason = "queue"
        state_busy = True

    return {
        "sampled_at": time.time(),
        "instance_id": lease.get("instance_id"),
        "account_id": lease.get("account_id"),
        "gpu_name": str(device.get("name")) if device and device.get("name") else None,
        "vram_total_bytes": vram_total,
        "vram_free_bytes": vram_free,
        "vram_used_bytes": vram_used,
        "torch_vram_total_bytes": torch_vram_total,
        "torch_vram_free_bytes": torch_vram_free,
        "torch_vram_used_bytes": torch_vram_used,
        "ram_total_bytes": ram_total,
        "ram_free_bytes": ram_free,
        "ram_used_bytes": ram_used,
        "queue_running": queue_running,
        "queue_pending": queue_pending,
        "busy": bool(state_busy) if state_busy is not None else bool((queue_running or 0) or (queue_pending or 0)),
        "busy_reason": busy_reason,
        "disk_free_bytes": disk_free,
        "vfs_cache_bytes": vfs_cache,
        "model_storage_mode": storage_cfg.get("model_storage_mode"),
        "cache_gb": storage_cfg.get("cache_gb"),
        "comfyui_version": system.get("comfyui_version"),
        "pytorch_version": system.get("pytorch_version"),
    }


async def _fetch_native_telemetry(repo, owner_id: str, lease: dict, *, force_refresh: bool = False) -> dict[str, Any] | None:
    cache_key = (owner_id, str(lease.get("instance_id") or lease.get("id") or ""))
    now = time.monotonic()
    if not force_refresh:
        with _TELEMETRY_LOCK:
            cached = _TELEMETRY_CACHE.get(cache_key)
            if cached and now - cached[0] < _TELEMETRY_CACHE_TTL:
                return cached[1]

    try:
        url, headers = native.backend_for(repo, owner_id)
    except APIError:
        return None

    async def _get_json(client: httpx.AsyncClient, path: str) -> dict | None:
        try:
            resp = await client.get(url + path, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                return data if isinstance(data, dict) else None
        except (httpx.HTTPError, ValueError):
            pass
        return None

    try:
        async with httpx.AsyncClient(timeout=8) as client:
            stats_data, queue_data, state_data = await asyncio.gather(
                _get_json(client, "/system_stats"),
                _get_json(client, "/queue"),
                _get_json(client, "/studio_native/state"),
            )
    except httpx.HTTPError:
        return None

    storage_cfg = native.lease_storage(repo, lease)
    telemetry = _parse_telemetry(lease, storage_cfg, stats_data, queue_data, state_data)
    if telemetry is not None:
        with _TELEMETRY_LOCK:
            _TELEMETRY_CACHE[cache_key] = (time.monotonic(), telemetry)
    return telemetry


def _listener_status(app):
    mgr = getattr(app.state, "port_listeners", None)
    return mgr.status("native") if mgr else {"active": False, "host": None, "port": None, "error": None}


def _access_metadata(store, owner: str, pol: dict) -> tuple[str | None, dict]:
    code = (
        get_or_create_path_code(store, f"native-api:{owner}", owner)
        if pol.get("auth_mode") in {"url_path", "local_open"}
        else peek_path_code(store, f"native-api:{owner}", owner)
    )
    urls = build_access_urls(
        kind="native",
        auth_mode=pol.get("auth_mode", "strict"),
        path_code=code,
        listen_port=pol.get("listen_port"),
        dedicated_url=native_config.public_url(store, owner),
        allow_lan=bool(pol.get("allow_lan")),
    )
    return code, urls


@router.get("/setup")
def get_setup(request: Request, user=Depends(current_user)):
    admin(user)
    response = ok(native_config.public_setup(request.app.state.repo, user["id"]), request)
    response.headers["Cache-Control"] = "no-store"
    return response


@router.put("/setup")
def save_setup(body: dict, request: Request, user=Depends(current_user)):
    admin(user)
    response = ok(native_config.save_setup(request.app.state.repo, user["id"], body), request)
    response.headers["Cache-Control"] = "no-store"
    return response


@router.delete("/setup")
def clear_setup(request: Request, user=Depends(current_user)):
    admin(user)
    response = ok(native_config.clear_setup(request.app.state.repo, user["id"]), request)
    response.headers["Cache-Control"] = "no-store"
    return response


@router.post("/setup/diagnose")
async def diagnose_setup(request: Request, user=Depends(current_user)):
    admin(user)
    result = await asyncio.to_thread(native_config.diagnose_setup, request.app.state.repo, user["id"])
    response = ok(result, request)
    response.headers["Cache-Control"] = "no-store"
    return response


@router.get("/api-settings")
def api_settings(request: Request, user=Depends(current_user)):
    admin(user)
    owner = user["id"]
    repo = request.app.state.repo
    pol = native_api.policy(repo.store, owner)
    path_code, urls = _access_metadata(repo.store, owner, pol)
    keys = repo.store.list(native_api.KEYS, owner)
    starts = [job for job in repo.list("jobs", owner) if job.get("native_api_key_id")]
    return ok({
        "policy": pol,
        "url": native_config.public_url(repo.store, owner),
        "path_code": path_code,
        "urls": urls,
        "listener": _listener_status(request.app),
        "keys": [native_api.public_key(row) for row in keys],
        "recent_starts": [{"job_id": job["id"], "key_id": job["native_api_key_id"],
                           "state": job["state"], "created_at": job["created_at"]} for job in starts[:10]],
    }, request)


@router.put("/api-settings")
async def update_api_settings(body: dict, request: Request, user=Depends(current_user)):
    admin(user)
    owner = user["id"]
    repo = request.app.state.repo
    values = await asyncio.to_thread(native_api.validate_policy, repo, owner, body)
    mgr = getattr(request.app.state, "port_listeners", None)
    if mgr is not None:
        await mgr.apply("native", values["listen_port"], allow_lan=values["allow_lan"], strict=True)
    elif values["listen_port"] is not None:
        probe_port_available(values["listen_port"], allow_lan=values["allow_lan"])
    saved = await asyncio.to_thread(native_api.persist_policy, repo.store, owner, values)
    return ok(saved, request)


@router.post("/api-settings/reset-code")
def rotate_native_path_code(request: Request, user=Depends(current_user)):
    admin(user)
    owner = user["id"]
    store = request.app.state.repo.store
    pol = native_api.policy(store, owner)
    code = reset_path_code(store, f"native-api:{owner}", owner)
    urls = build_access_urls(
        kind="native",
        auth_mode=pol.get("auth_mode", "strict"),
        path_code=code,
        listen_port=pol.get("listen_port"),
        dedicated_url=native_config.public_url(store, owner),
        allow_lan=bool(pol.get("allow_lan")),
    )
    response = ok({"path_code": code, "urls": urls}, request)
    response.headers["Cache-Control"] = "no-store"
    return response


@router.post("/api-keys", status_code=201)
def create_api_key(body: dict, request: Request, user=Depends(current_user)):
    admin(user)
    response = ok(native_api.create_key(request.app.state.repo.store, user["id"], body), request, 201)
    response.headers["Cache-Control"] = "no-store"
    return response


@router.patch("/api-keys/{key_id}")
def update_api_key(key_id: str, body: dict, request: Request, user=Depends(current_user)):
    admin(user)
    store, owner = request.app.state.repo.store, user["id"]
    if set(body) != {"auto_start"} or type(body["auto_start"]) is not bool:
        raise APIError("NATIVE_API_INPUT", "开机权限无效")
    with store.admission_lock() as acquired:
        if not acquired:
            raise APIError("NATIVE_API_BUSY", "暂无法更新接入 Key，请重试", 503)
        row = store.get(native_api.KEYS, key_id, owner)
        if not row or row.get("revoked_at"):
            raise APIError("NATIVE_API_NOT_FOUND", "有效接入 Key 不存在", 404)
        row = store.update(native_api.KEYS, key_id, body, owner)
    return ok(native_api.public_key(row), request)


@router.delete("/api-keys/{key_id}")
def revoke_api_key(key_id: str, request: Request, user=Depends(current_user)):
    admin(user)
    return ok(native_api.revoke_key(request.app.state.repo.store, user["id"], key_id), request)


@router.get("/telemetry")
async def telemetry(request: Request, refresh: bool = False, user=Depends(current_user)):
    repo = request.app.state.repo
    lease = native.active_lease(repo, user["id"])
    if not lease:
        return ok({"active": False, "ready": False, "restarting": False, "instance_id": None, "telemetry": None}, request)
    restarting = native.restart_active(lease)
    if not lease.get("native_ready") or restarting:
        return ok(
            {
                "active": True,
                "ready": False,
                "restarting": restarting,
                "instance_id": lease.get("instance_id"),
                "telemetry": None,
            },
            request,
        )
    data = await _fetch_native_telemetry(repo, user["id"], lease, force_refresh=refresh)
    return ok(
        {
            "active": True,
            "ready": data is not None,
            "restarting": False,
            "instance_id": lease.get("instance_id"),
            "telemetry": data,
        },
        request,
    )


@router.get("/status")
async def status(request: Request, user=Depends(current_user)):
    repo = request.app.state.repo
    lease = native.active_lease(repo, user["id"])
    restart_lease = lease or max(
        (row for row in repo.list("leases", user["id"]) if row.get("native_restart")),
        key=lambda row: row["native_restart"]["created_at"], default=None,
    )
    restart = native_restart.public_restart(restart_lease)
    native_jobs = [j for j in repo.list("jobs", user["id"]) if (j.get("snapshot") or {}).get("native_comfy")]
    job = max(native_jobs, key=lambda j: j.get("created_at") or "", default=None)
    last_instance_id = (job or {}).get("instance_id")
    instance = repo.get("instances", last_instance_id, user["id"]) if not lease and last_instance_id else None
    reclaimed = bool(instance and instance.get("state") == "deleted"
                     and instance.get("remote_delete_source") == RUNTIME_MISSING)
    ready = False
    disk = {}
    if lease and lease.get("native_ready") and not native.restart_active(lease):
        try:
            url, headers = native.backend_for(repo, user["id"])
            async with httpx.AsyncClient(timeout=8) as client:
                response = await client.get(url + "/system_stats", headers=headers)
                ready = response.status_code == 200 and isinstance(response.json().get("system"), dict)
                if ready and lease.get("native_model_storage"):
                    metrics = await client.get(url + "/studio_native/state", headers=headers)
                    if metrics.is_success:
                        state = metrics.json()
                        disk = {k: state[k] for k in ("disk_free_bytes", "vfs_cache_bytes") if type(state.get(k)) is int}
        except (httpx.HTTPError, ValueError, AttributeError, APIError):
            pass
    events = repo.list("events", user["id"]) if job else []
    can_restart = ready and native_restart.can_restart(repo, lease)
    ncfg = native_config.resolve(repo, user["id"], use_cache=False, file_reader=native.secret_file, strict_decrypt=False)
    return ok({"configured": bool(ncfg.tunnel_configured or (not ncfg.has_db_override and ncfg.any_configured)),
               "tunnel_configured": ncfg.tunnel_configured,
               "web_ui_configured": ncfg.web_ui_configured,
               "url": ncfg.native_comfy_url,
               "timeouts": native.timeout_options(), "model_storage_defaults": native.model_storage_defaults(),
               "model_storage_known": not lease or bool(lease.get("native_model_storage")),
               "model_storage": native.lease_storage(repo, lease) if lease else (
                   native.model_storage(job.get("execution_policy")) if job and job["state"] not in native.TERMINAL else None
               ), "disk": disk,
               "ready": ready, "instance_id": lease.get("instance_id") if lease else None,
               "job_id": job["id"] if job else None, "state": job["state"] if job else "stopped",
               "restart": restart, "can_restart": can_restart,
               "reason": RUNTIME_MISSING if reclaimed else None,
               "error": RUNTIME_MISSING_MESSAGE if reclaimed else (job or {}).get("error_message"),
               "events": [{"message": e.get("message"), "state": e.get("state")}
                          for e in events if e.get("job_id") == (job or {}).get("id")][:30]}, request)


@router.post("/restart", status_code=202)
def restart(body: dict, request: Request, user=Depends(current_user)):
    if set(body) != {"instance_id"}:
        raise APIError("NATIVE_INPUT", "必须提供当前原生实例 ID", 422)
    restart_id = native_restart.submit(request.app.state.repo, user["id"], body["instance_id"],
                                       request.headers.get("Idempotency-Key"))
    return ok({"restart_id": restart_id}, request, 202)


@router.post("/start", status_code=202)
def start(body: dict, request: Request, user=Depends(current_user)):
    unknown = set(body) - {"account_id", "spec", "instance_id", "plugin_generation",
                           "retain_idle_minutes", "max_runtime_minutes", *native.model_storage()}
    if unknown:
        raise APIError("NATIVE_INPUT", "不支持的启动参数", 422)
    job = native.admit(request.app.state.repo, user["id"], body, request.headers.get("Idempotency-Key"))
    return ok({"job_id": job["id"], "state": job["state"]}, request, 202)


def _local_lease(request, owner):
    repo = request.app.state.repo
    lease = native.active_lease(repo, owner)
    if not lease or not lease.get("native_ready") or native.lease_storage(repo, lease)["model_storage_mode"] != "local":
        raise APIError("NATIVE_NOT_LOCAL", "请先启动本地模型模式的原生 ComfyUI", 409)
    return repo, lease


@router.get("/workflows")
async def list_workflows(request: Request, refresh: bool = False, user=Depends(current_user)):
    repo, lease = await asyncio.to_thread(_local_lease, request, user["id"])
    base, auth = native.backend_for(repo, user["id"])
    suffix = "/workflows?refresh=1" if refresh else "/workflows"
    async with httpx.AsyncClient() as client:
        result = await native_model_api.rpc(client, "GET", base, native_model_api.headers(base, auth, lease, repo=repo), suffix)
    return ok(result, request)


@router.post("/workflows", status_code=201)
async def save_workflow(body: dict, request: Request, user=Depends(current_user)):
    repo, lease = await asyncio.to_thread(_local_lease, request, user["id"])
    if (set(body) - {"name", "workflow", "overwrite"}
            or not isinstance(body.get("name"), str) or not body["name"].strip()
            or not isinstance(body.get("workflow"), dict)
            or not isinstance(body.get("overwrite", False), bool)):
        raise APIError("NATIVE_MODEL_INPUT", "请提供有效的工作流名称和 JSON 对象", 422)
    write_id = uuid4().hex
    await asyncio.to_thread(native.begin_write, repo, lease, write_id=write_id)
    try:
        base, auth = native.backend_for(repo, user["id"])
        async with httpx.AsyncClient() as client:
            result = await native_model_api.rpc(
                client, "POST", base, native_model_api.headers(base, auth, lease, repo=repo), "/workflows",
                body={"name": body["name"], "workflow": body["workflow"], "overwrite": body.get("overwrite", False)},
            )
        return ok(result, request, 201)
    finally:
        await asyncio.to_thread(native.update_write, repo, lease, write_id, finished=True)


@router.delete("/workflows")
async def delete_workflow(source: str, path: str, request: Request, user=Depends(current_user)):
    repo, lease = await asyncio.to_thread(_local_lease, request, user["id"])
    if source != "warmup" or not isinstance(path, str) or not path.strip():
        raise APIError("NATIVE_MODEL_INPUT", "仅允许移除已保存的 API 预热工作流", 422)
    write_id = uuid4().hex
    await asyncio.to_thread(native.begin_write, repo, lease, write_id=write_id)
    try:
        base, auth = native.backend_for(repo, user["id"])
        async with httpx.AsyncClient() as client:
            result = await native_model_api.rpc(
                client, "POST", base, native_model_api.headers(base, auth, lease, repo=repo), "/workflows/delete",
                body={"source": source, "path": path},
            )
        return ok(result, request)
    finally:
        await asyncio.to_thread(native.update_write, repo, lease, write_id, finished=True)


def _valid_workflow_ref(ref):
    return (isinstance(ref, dict) and set(ref) == {"source", "path"}
            and ref.get("source") in {"comfy", "warmup"}
            and isinstance(ref.get("path"), str) and bool(ref["path"].strip()))


@router.post("/model-preparation", status_code=202)
async def prepare_models(body: dict, request: Request, user=Depends(current_user)):
    repo, lease = await asyncio.to_thread(_local_lease, request, user["id"])
    has_workflow = "workflow" in body
    has_ref = "workflow_ref" in body
    if (set(body) - {"workflow", "workflow_ref", "dependencies"}
            or has_workflow == has_ref
            or (has_workflow and not isinstance(body.get("workflow"), dict))
            or (has_ref and not _valid_workflow_ref(body.get("workflow_ref")))):
        raise APIError("NATIVE_MODEL_INPUT", "请提供工作流 JSON 或选择 Drive 工作流，并可附带显式模型依赖", 422)
    workflow = None
    workflow_ref = body.get("workflow_ref")
    if has_workflow:
        workflow = body["workflow"]
        if "prompt" not in workflow and "workflow" not in workflow:
            workflow = {"workflow": workflow} if "nodes" in workflow else {"prompt": workflow}
    write_id = uuid4().hex
    await asyncio.to_thread(native.begin_write, repo, lease, write_id=write_id)
    try:
        base, auth = native.backend_for(repo, user["id"])
        async with httpx.AsyncClient() as client:
            result = await native_model_api.start(
                client, base, native_model_api.headers(base, auth, lease, repo=repo), repo, lease,
                workflow, body.get("dependencies", []), visual=True, workflow_ref=workflow_ref,
            )
        return ok(result, request, 202)
    finally:
        await asyncio.to_thread(native.update_write, repo, lease, write_id, finished=True)


@router.get("/model-preparation/{ident}")
async def model_preparation(ident: str, request: Request, user=Depends(current_user)):
    repo, lease = await asyncio.to_thread(_local_lease, request, user["id"])
    if ident != "latest" and not re.fullmatch(r"[0-9a-f]{32}", ident):
        raise APIError("NATIVE_MODEL_PREPARATION_MISSING", "准备记录不存在", 404)
    base, auth = native.backend_for(repo, user["id"])
    async with httpx.AsyncClient() as client:
        result = await native_model_api.rpc(client, "GET", base, native_model_api.headers(base, auth, lease, repo=repo),
                                            "/prepare/" + ident)
    return ok(result, request)


@router.post("/ticket")
def ticket(request: Request, user=Depends(current_user)):
    url = native_config.public_url(request.app.state.repo, user["id"])
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower()
    is_local_http = parsed.scheme == "http" and (host in {"localhost", "127.0.0.1", "::1"} or host.endswith(".localhost"))
    if (parsed.scheme != "https" and not is_local_http) or not parsed.netloc or parsed.path or parsed.query or parsed.fragment:
        raise APIError("NATIVE_CONFIG", "NATIVE_COMFY_URL 必须为独立 HTTPS 源地址（或本机 http://*.localhost 地址）", 503)
    now = int(time.time())
    return ok({"url": url + "/_studio/session",
               "ticket": sign_session({"uid": user["id"], "aud": url, "purpose": "native-handoff",
                                       "iat": now, "exp": now + 60})}, request)


@router.get("/plugins")
def plugins(request: Request, user=Depends(current_user)):
    from .gpu.node_persistence import digest
    repo = request.app.state.repo
    lease = native.active_lease(repo, user["id"])
    online = False
    if lease and lease.get("native_ready"):
        try:
            native.sync_plugins(repo, lease)
            online = True
        except (httpx.HTTPError, ValueError, APIError):
            pass
    saved = repo.store.get("settings", "native-plugins-" + digest(user["id"])[:32], user["id"]) or {}
    return ok({**saved.get("status", {}), "manifest": saved.get("manifest"), "history": saved.get("history", []),
               "offline": not online}, request)
