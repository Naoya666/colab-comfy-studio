"""Studio control plane for native ComfyUI. Data plane lives on its own host."""
import time
from urllib.parse import urlsplit

import httpx
from fastapi import APIRouter, Depends, Request

from .common import APIError, current_user, ok
from .config import settings
from .gpu import native
from .gpu.colab import RUNTIME_MISSING, RUNTIME_MISSING_MESSAGE
from .security import sign_session

router = APIRouter(prefix="/api/gpu/native")


@router.get("/status")
async def status(request: Request, user=Depends(current_user)):
    repo = request.app.state.repo
    lease = native.active_lease(repo, user["id"])
    native_jobs = [j for j in repo.list("jobs", user["id"]) if (j.get("snapshot") or {}).get("native_comfy")]
    job = max(native_jobs, key=lambda j: j.get("created_at") or "", default=None)
    last_instance_id = (job or {}).get("instance_id")
    instance = repo.get("instances", last_instance_id, user["id"]) if not lease and last_instance_id else None
    reclaimed = bool(instance and instance.get("state") == "deleted"
                     and instance.get("remote_delete_source") == RUNTIME_MISSING)
    ready = False
    if lease and lease.get("native_ready"):
        try:
            url, headers = native.backend()
            async with httpx.AsyncClient(timeout=8) as client:
                response = await client.get(url + "/system_stats", headers=headers)
                ready = response.status_code == 200 and isinstance(response.json().get("system"), dict)
        except (httpx.HTTPError, ValueError, APIError):
            pass
    events = repo.list("events", user["id"]) if job else []
    return ok({"configured": bool(settings.native_comfy_url), "url": settings.native_comfy_url,
               "timeouts": native.timeout_options(),
               "ready": ready, "instance_id": lease.get("instance_id") if lease else None,
               "job_id": job["id"] if job else None, "state": job["state"] if job else "stopped",
               "reason": RUNTIME_MISSING if reclaimed else None,
               "error": RUNTIME_MISSING_MESSAGE if reclaimed else (job or {}).get("error_message"),
               "events": [{"message": e.get("message"), "state": e.get("state")}
                          for e in events if e.get("job_id") == (job or {}).get("id")][-30:]}, request)


@router.post("/start", status_code=202)
def start(body: dict, request: Request, user=Depends(current_user)):
    unknown = set(body) - {"account_id", "spec", "instance_id", "plugin_generation",
                           "retain_idle_minutes", "max_runtime_minutes"}
    if unknown:
        raise APIError("NATIVE_INPUT", "不支持的启动参数", 422)
    job = native.admit(request.app.state.repo, user["id"], body, request.headers.get("Idempotency-Key"))
    return ok({"job_id": job["id"], "state": job["state"]}, request, 202)


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
