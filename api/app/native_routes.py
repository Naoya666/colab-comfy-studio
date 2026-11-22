"""Studio control plane for native ComfyUI. Data plane lives on its own host."""
import asyncio
import re
import time
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
from fastapi import APIRouter, Depends, Request

from . import native_api, native_model_api
from .common import APIError, admin, current_user, ok
from .config import settings
from .gpu import native, native_restart
from .gpu.colab import RUNTIME_MISSING, RUNTIME_MISSING_MESSAGE
from .security import sign_session

router = APIRouter(prefix="/api/gpu/native")


@router.get("/api-settings")
def api_settings(request: Request, user=Depends(current_user)):
    admin(user)
    owner = user["id"]
    repo = request.app.state.repo
    keys = repo.store.list(native_api.KEYS, owner)
    starts = [job for job in repo.list("jobs", owner) if job.get("native_api_key_id")]
    return ok({
        "policy": native_api.policy(repo.store, owner),
        "url": settings.native_comfy_url.rstrip("/"),
        "keys": [native_api.public_key(row) for row in keys],
        "recent_starts": [{"job_id": job["id"], "key_id": job["native_api_key_id"],
                           "state": job["state"], "created_at": job["created_at"]} for job in starts[:10]],
    }, request)


@router.put("/api-settings")
def update_api_settings(body: dict, request: Request, user=Depends(current_user)):
    admin(user)
    return ok(native_api.save_policy(request.app.state.repo, user["id"], body), request)


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
            url, headers = native.backend()
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
    return ok({"configured": bool(settings.native_comfy_url), "url": settings.native_comfy_url,
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


@router.post("/model-preparation", status_code=202)
async def prepare_models(body: dict, request: Request, user=Depends(current_user)):
    repo, lease = await asyncio.to_thread(_local_lease, request, user["id"])
    if set(body) - {"workflow", "dependencies"} or not isinstance(body.get("workflow"), dict):
        raise APIError("NATIVE_MODEL_INPUT", "请提供工作流 JSON 和可选的显式模型依赖", 422)
    workflow = body["workflow"]
    if "prompt" not in workflow and "workflow" not in workflow:
        workflow = {"workflow": workflow} if "nodes" in workflow else {"prompt": workflow}
    write_id = uuid4().hex
    await asyncio.to_thread(native.begin_write, repo, lease, write_id=write_id)
    try:
        base, auth = native.backend()
        async with httpx.AsyncClient() as client:
            result = await native_model_api.start(
                client, base, native_model_api.headers(base, auth, lease), repo, lease,
                workflow, body.get("dependencies", []), visual=True,
            )
        return ok(result, request, 202)
    finally:
        await asyncio.to_thread(native.update_write, repo, lease, write_id, finished=True)


@router.get("/model-preparation/{ident}")
async def model_preparation(ident: str, request: Request, user=Depends(current_user)):
    _, lease = await asyncio.to_thread(_local_lease, request, user["id"])
    if ident != "latest" and not re.fullmatch(r"[0-9a-f]{32}", ident):
        raise APIError("NATIVE_MODEL_PREPARATION_MISSING", "准备记录不存在", 404)
    base, auth = native.backend()
    async with httpx.AsyncClient() as client:
        result = await native_model_api.rpc(client, "GET", base, native_model_api.headers(base, auth, lease),
                                            "/prepare/" + ident)
    return ok(result, request)


@router.post("/ticket")
def ticket(request: Request, user=Depends(current_user)):
    url = settings.native_comfy_url.rstrip("/")
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.netloc or parsed.path or parsed.query or parsed.fragment:
        raise APIError("NATIVE_CONFIG", "NATIVE_COMFY_URL 必须为独立 HTTPS 源地址", 503)
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
