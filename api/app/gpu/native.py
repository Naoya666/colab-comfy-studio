"""Native ComfyUI admission and conservative activity checks on existing leases."""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from app.common import APIError
from app.config import settings
from app.security import decrypt_secret

from .compute import GpuConfig
from .manifest import digest

TERMINAL = {"completed", "failed", "cancelled", "rejected"}


def timeout_options():
    return {
        "defaults": {"retain_idle_minutes": settings.native_comfy_idle_minutes,
                     "max_runtime_minutes": settings.native_comfy_max_runtime_minutes},
        "limits": {"retain_idle_minutes": None, "max_runtime_minutes": None},
    }


def secret_file(name):
    path = os.getenv(name, "")
    if not path:
        raise APIError("NATIVE_CONFIG", f"必须配置 {name}", 503)
    try:
        value = Path(path).read_text().strip()
    except OSError as exc:
        raise APIError("NATIVE_CONFIG", f"无法读取 {name} 指定的凭据文件", 503) from exc
    if not value:
        raise APIError("NATIVE_CONFIG", f"{name} 指定的凭据文件为空", 503)
    return value


def backend():
    url = settings.native_comfy_backend_url.rstrip("/")
    p = urlsplit(url)
    if p.scheme != "https" or not p.hostname or p.username or p.password or p.path or p.query or p.fragment:
        raise APIError("NATIVE_CONFIG", "NATIVE_COMFY_BACKEND_URL 必须为独立 HTTPS 源地址", 503)
    if p.netloc in {urlsplit(settings.site_url).netloc, urlsplit(settings.native_comfy_url).netloc}:
        raise APIError("NATIVE_CONFIG", "内部 tunnel 域名必须与 Studio 和原生公网域名不同", 503)
    try:
        access = json.loads(secret_file("NATIVE_COMFY_ACCESS_FILE"))
        if not isinstance(access, dict) or any(not isinstance(access.get(k), str) or not access[k] for k in ("client_id", "client_secret")):
            raise ValueError
    except (ValueError, TypeError) as exc:
        raise APIError("NATIVE_CONFIG", "Access 凭据文件必须包含 client_id 和 client_secret", 503) from exc
    return url, {"CF-Access-Client-Id": access["client_id"],
                 "CF-Access-Client-Secret": access["client_secret"]}


def active_lease(repo, owner="owner"):
    return next((x for x in repo.list("leases", owner)
                 if x.get("native_comfy") and x.get("state") not in {"deleted", "quarantined", "releasing", "deleting"}), None)


def busy(client=None):
    """Unknown is busy: network errors must never stop a video or an upload."""
    try:
        url, headers = backend()
        with httpx.Client(timeout=8) if client is None else _borrow(client) as c:
            q = c.get(url + "/queue", headers=headers)
            q.raise_for_status()
            data = q.json()
            if not isinstance(data.get("queue_running"), list) or not isinstance(data.get("queue_pending"), list):
                return True
            queued = bool(data["queue_running"] or data["queue_pending"])
            state = c.get(url + "/studio_native/state", headers=headers)
            state.raise_for_status()
            return queued or state.json().get("busy") is not False
    except Exception:
        return True


class _borrow:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self.value

    def __exit__(self, *args):
        pass


def admit(repo, owner, body, key):
    cfg = GpuConfig.from_env()
    timeouts = {"retain_idle_minutes": settings.native_comfy_idle_minutes,
                "max_runtime_minutes": settings.native_comfy_max_runtime_minutes}
    for field in timeouts:
        value = body.get(field, timeouts[field])
        if type(value) is not int or value < 1:
            raise APIError("NATIVE_INPUT", f"{field} 必须为正整数（分钟）", 422)
        timeouts[field] = value
    if not settings.native_comfy_url or not cfg.real_enabled:
        raise APIError("NATIVE_DISABLED", "请先配置原生 ComfyUI 域名并启用真实 GPU", 503)
    backend()
    secret_file("NATIVE_COMFY_TUNNEL_TOKEN_FILE")
    commit = os.getenv("NATIVE_COMFY_COMMIT", "")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise APIError("NATIVE_CONFIG", "NATIVE_COMFY_COMMIT 必须锁定 40 位官方提交", 503)
    if not key or len(key) > 200:
        raise APIError("IDEMPOTENCY_REQUIRED", "必须提供固定幂等键", 422)
    generation = body.get("plugin_generation")
    if generation is not None and (not isinstance(generation, str) or not re.fullmatch(r"[0-9a-f]{64}", generation)):
        raise APIError("NATIVE_INPUT", "plugin_generation 必须为历史清单的 64 位 SHA256", 422)
    account = body.get("account_id")
    if not account or not isinstance(body.get("spec"), dict):
        raise APIError("NATIVE_INPUT", "请选择账号和 GPU 规格", 422)
    # A durable dispatch lock serializes admission with both API and worker processes.
    with repo.store.dispatch_lock() as acquired:
        if not acquired:
            raise APIError("SCHEDULER_BUSY", "调度正在同步，请使用相同幂等键重试", 503)
        previous = next((j for j in repo.list("jobs", owner) if j.get("idempotency_key") == key), None)
        if previous:
            if previous.get("native_request_hash") != digest(body):
                raise APIError("IDEMPOTENCY_CONFLICT", "幂等键参数冲突", 409)
            return previous
        running = next((j for j in repo.list("jobs", owner)
                        if (j.get("snapshot") or {}).get("native_comfy") and j["state"] not in TERMINAL), None)
        live = active_lease(repo, owner)
        if running or live:
            source = running or repo.get("jobs", live["owner_job_id"], owner)
            from .specs import spec_key
            try:
                same_spec = source and spec_key(source.get("requested_spec")) == spec_key(body["spec"])
            except ValueError:
                same_spec = False
            if not source or source.get("account_id") != account or not same_spec:
                raise APIError("NATIVE_ACTIVE", "已有原生会话；请等待任务和同步完成后停止旧实例，再切换 GPU", 409)
            policy = source.get("execution_policy") or {}
            if any(field in body and timeouts[field] != policy.get(field, default)
                   for field, default in (("retain_idle_minutes", settings.native_comfy_idle_minutes),
                                          ("max_runtime_minutes", settings.native_comfy_max_runtime_minutes))):
                raise APIError("NATIVE_ACTIVE", "已有原生会话；请停止旧实例后重新启动以调整时间参数", 409)
            return source
        # Never create a second GPU while another live instance is already present.
        instances = [i for i in repo.list("instances", owner) if i.get("state") not in {"deleted"} and i.get("provider") == "colab"]
        requested_instance = body.get("instance_id")
        if requested_instance and not instances:
            raise APIError("INSTANCE_CONFLICT", "所选实例已释放或不能复用，请重新选择 GPU", 409)
        if instances:
            if len(instances) != 1 or (requested_instance and instances[0]["id"] != requested_instance):
                raise APIError("INSTANCE_CONFLICT", "请明确选择唯一可复用实例，或先释放其他实例", 409)
            instance = instances[0]
            if instance.get("state") not in {"ready", "retained_idle"} or instance.get("current_job_id"):
                raise APIError("INSTANCE_BUSY", "现有实例仍在执行任务或初始化，请等待完成", 409)
            if instance.get("account_id") != account:
                raise APIError("INSTANCE_CONFLICT", "已有实例属于其他账号，请先停止旧实例", 409)
            requested_instance = instance["id"]
        from app.compute_routes import _require_account
        _require_account(repo, owner, account)
        if not cfg.unit_rate or cfg.unit_rate <= 0:
            raise APIError("RATE_UNKNOWN", "请先配置 GPU_UNIT_RATE 与预算", 422)
        project = "native-comfy-" + owner
        if not repo.store.get("projects", project, owner):
            repo.store.create("projects", owner, {"title": "原生 ComfyUI", "system": True}, record_id=project)
        raw = {"kind": "native-comfy", **body}
        job, _ = repo.create_job(owner, raw, key)
        policy = {"provider": "colab", "lifecycle": "retain", "account_id": account,
                  **timeouts,
                  **({"instance_id": requested_instance, "allocation": "existing"} if requested_instance else {})}
        return repo.update("jobs", job["id"], {
            "native_request_hash": digest(body), "project_id": project, "provider": "colab",
            "account_id": account, "requested_account_id": account,
            "requested_instance_id": requested_instance, "requested_spec": body["spec"],
            "lifecycle": "retain", "execution_policy": policy,
            "model_id": "native-comfy", "workflow_key": "system/native-comfy", "workflow_version": 1,
            "max_runtime_minutes": timeouts["max_runtime_minutes"], "unit_rate": cfg.unit_rate,
            "native_session_deadline_epoch": time.time() + timeouts["max_runtime_minutes"] * 60,
            "estimated_units": timeouts["max_runtime_minutes"] * cfg.unit_rate,
            "snapshot": {"native_comfy": True, "native_plugin_generation": body.get("plugin_generation"), "bound_graph": {},
                         "manifest": {"reviewed": True, "state": "published", "comfy_commit": commit,
                                      "compute_spec": body["spec"].get("key", body["spec"])}}}, owner)


def runtime_secrets(repo, job):
    """Only sent over the trusted Colab execution channel; never saved in a DB row."""
    from .drive import DriveStorage
    drive = DriveStorage(job, repo)
    try:
        cache = drive.model_cache()
        google = drive.google
        cred = google._credential("drive", account_id=google.account_id)
        client_id = cred.get("client_id")
        if client_id and cred.get("encrypted_client_secret"):
            client_secret = decrypt_secret(cred["encrypted_client_secret"])
        else:
            client_id, client_secret, _ = google._config()
        return {"owner_folder_id": drive._owner_folder_id, "models_folder_id": cache["folder_id"],
                "client_id": client_id, "client_secret": client_secret,
                "refresh_token": decrypt_secret(cred["encrypted_refresh_token"]),
                "access_token": cache["access_token"],
                "tunnel_token": secret_file("NATIVE_COMFY_TUNNEL_TOKEN_FILE"),
                "cache_gb": settings.native_comfy_cache_gb, "max_upload_mb": settings.native_comfy_max_upload_mb,
                "models": [dict(filename=m.get("filename"), category=m.get("model_type"), file_id=m["drive_file_id"])
                           for m in repo.list("model_library", job["owner_id"])
                           if m.get("state") == "ready" and m.get("drive_file_id")
                           and m.get("drive_account_id") in {None, drive.account_id}]}
    finally:
        drive.close()


def _runtime_deleted(repo, lease):
    """Only provider confirmation can distinguish a gone VM from a broken tunnel."""
    handle = lease.get("handle")
    if lease.get("provider") != "colab" or not lease.get("account_id") or not isinstance(handle, dict):
        return False
    from .colab import GoogleColabProvider, runtime_status

    try:
        handle = {**handle, "job_id": lease.get("owner_job_id"), "owner_job_id": lease.get("owner_job_id")}
        with httpx.Client(timeout=10) as client:
            provider = GoogleColabProvider(repo, lease["owner_id"], account_id=lease["account_id"], client=client)
            return runtime_status(provider.inspect(handle)) == "missing"
    except Exception:
        return False  # Provider outages are not deletion evidence.


def _lease_identity(lease):
    return tuple(lease.get(k) for k in (
        "id", "provider", "owner_id", "owner_job_id", "current_job_id", "instance_id", "account_id",
        "runtime_id", "request_id", "handle", "tunnel_endpoint", "tunnel_nbh",
    ))


def keepalive_lease(repo, lease, clock=time.time, *, provider_factory=None):
    """Send bounded Colab TFE + proxy/kernel keep-alive only while retention gates hold."""
    if (
        not isinstance(lease, dict)
        or lease.get("provider") != "colab"
        or lease.get("state") != "idle"
        or lease.get("current_job_id")
        or lease.get("native_draining")
        or not lease.get("account_id")
    ):
        return None
    if provider_factory is None and not GpuConfig.from_env().real_enabled:
        return None
    if provider_factory is None and not (lease.get("tunnel_endpoint") or lease.get("proxy_url") or isinstance(lease.get("handle"), dict)):
        return None
    now = float(clock())
    deadline = lease.get("deadline_epoch")
    if deadline is not None and now >= float(deadline):
        return None
    from .colab import (
        KEEPALIVE_INTERVAL_SECONDS,
        KEEPALIVE_RETRY_SECONDS,
        KERNEL_EXEC_INTERVAL_SECONDS,
        GoogleColabProvider,
    )

    prev_status = lease.get("native_keepalive_status")
    interval = (
        KEEPALIVE_RETRY_SECONDS
        if isinstance(prev_status, dict) and prev_status.get("ok") is False
        else KEEPALIVE_INTERVAL_SECONDS
    )
    last_keepalive = float(lease.get("native_last_keepalive_at") or 0)
    if last_keepalive > 0 and now - last_keepalive < interval:
        return None
    last_kernel_exec = float(lease.get("native_last_kernel_exec_at") or 0)
    run_kernel_exec = last_kernel_exec <= 0 or (now - last_kernel_exec) >= KERNEL_EXEC_INTERVAL_SECONDS
    identity = _lease_identity(lease)
    job_stub = {
        "id": lease.get("owner_job_id"),
        "owner_id": lease["owner_id"],
        "provider": "colab",
        "account_id": lease["account_id"],
    }
    provider = None
    owns_provider = False
    try:
        if provider_factory is not None:
            provider = provider_factory(job_stub)
            if not hasattr(provider, "keep_alive"):
                return None
        else:
            provider = GoogleColabProvider(repo, lease["owner_id"], account_id=lease["account_id"])
            owns_provider = True
        result = provider.keep_alive(lease, run_kernel_exec=run_kernel_exec)
    except Exception as exc:
        result = {
            "ok": False,
            "tfe": f"err:{type(exc).__name__}",
            "proxy": "skipped",
            "kernel_exec": "skipped",
            "lease_patch": {},
        }
    finally:
        if owns_provider and provider is not None:
            provider.close()

    if not hasattr(repo, "store") or not hasattr(repo.store, "dispatch_lock"):
        return result
    commit_now = float(clock())
    with repo.store.dispatch_lock() as acquired:
        if not acquired:
            return result
        fresh = repo.get("leases", lease["id"], lease["owner_id"])
        if (
            not fresh
            or fresh.get("state") != "idle"
            or fresh.get("current_job_id")
            or fresh.get("native_draining")
            or _lease_identity(fresh) != identity
        ):
            return None
        fresh_deadline = fresh.get("deadline_epoch")
        if fresh_deadline is not None and commit_now >= float(fresh_deadline):
            return None
        patch = dict(result.get("lease_patch") or {})
        patch["native_last_keepalive_at"] = commit_now
        if result.get("kernel_exec") == "ok":
            patch["native_last_kernel_exec_at"] = commit_now
        patch["native_keepalive_status"] = {
            "at": commit_now,
            "ok": bool(result.get("ok")),
            "tfe": result.get("tfe"),
            "proxy": result.get("proxy"),
            "kernel_exec": result.get("kernel_exec"),
        }
        repo.update("leases", fresh["id"], patch, fresh["owner_id"])
    return result


def protect_lease(repo, lease, clock=time.time, *, check_remote=True, provider_factory=None):
    """Fence idle shutdown against new uploads/prompts under the existing dispatch lock."""
    if not lease.get("native_comfy"):
        return False
    if GpuConfig.from_env().real_enabled:
        try:
            sync_plugins(repo, lease)
        except Exception as exc:
            repo.update("leases", lease["id"], {"native_plugin_sync_error": type(exc).__name__}, lease["owner_id"])
            # Busy probe fails closed if persistence has an error or dirty files.
            # A VPS mirror outage must not replace the last committed manifest.
            pass
    identity = _lease_identity(lease)
    upstream_busy = busy()
    # An offline tunnel also reports "busy". It cannot veto reconciliation of
    # a VM already confirmed absent by Colab's independent control plane.
    remote_deleted = check_remote and upstream_busy and _runtime_deleted(repo, lease)
    retained = False
    with repo.store.dispatch_lock() as acquired:
        if not acquired:
            return True
        lease = repo.get("leases", lease["id"], lease["owner_id"])
        if not lease or lease.get("state") == "deleted" or _lease_identity(lease) != identity:
            return True
        if remote_deleted:
            repo.update("leases", lease["id"], {
                "native_ready": False, "native_draining": True,
                "native_remote_deleted_verified_at": clock(),
            }, lease["owner_id"])
            return False  # Existing scheduler cleanup still confirms deletion.
        instance = repo.get("instances", lease.get("instance_id"), lease["owner_id"]) or {}
        manual = bool(instance.get("manual_stop_requested"))
        if not manual:
            if upstream_busy:
                lease = repo.update("leases", lease["id"], {"native_last_busy": clock()}, lease["owner_id"])
                retained = True
            elif clock() - float(lease.get("native_http_heartbeat") or 0) < 30:
                retained = True
            else:
                last = max(float(lease.get("native_last_busy") or 0), float(lease.get("native_last_write") or 0))
                idle_minutes = lease.get("native_idle_minutes")
                if idle_minutes is None:
                    job = repo.get("jobs", lease.get("owner_job_id"), lease["owner_id"]) if lease.get("owner_job_id") else None
                    idle_minutes = ((job or {}).get("execution_policy") or {}).get("retain_idle_minutes", settings.native_comfy_idle_minutes)
                deadline = lease.get("deadline_epoch")
                expired = deadline is not None and clock() >= float(deadline)
                if not expired and clock() - last < idle_minutes * 60:
                    retained = True
        if not retained:
            repo.update("leases", lease["id"], {"native_draining": True}, lease["owner_id"])
            if lease.get("instance_id") and instance:
                repo.update("instances", lease["instance_id"], {"native_draining": True}, lease["owner_id"])
            return False
    keepalive_lease(repo, lease, clock=clock, provider_factory=provider_factory)
    return True


def begin_write(repo, lease, *, allow_stopping=False, timeout=2.0):
    """Wait only for local admission; never replay an upstream write.

    Keep the scheduler's nonblocking lock semantics. Sleep after releasing each
    failed lock context, and revalidate the original session under the fence.
    """
    deadline = time.monotonic() + timeout
    while True:
        with repo.store.dispatch_lock() as acquired:
            if acquired:
                fresh = repo.get("leases", lease["id"], lease["owner_id"])
                if not fresh or _lease_identity(fresh) != _lease_identity(lease):
                    raise APIError("NATIVE_SESSION_CHANGED", "会话已结束或切换，请回到 Studio 重新打开", 409)
                if (not fresh.get("native_ready") or fresh.get("native_draining")
                        or fresh.get("state") in {"deleted", "releasing", "deleting"}):
                    raise APIError("NATIVE_DRAINING", "实例正在释放，请回到 Studio 重启", 409)
                instance = repo.get("instances", fresh.get("instance_id"), fresh["owner_id"]) or {}
                if (instance.get("native_draining")
                        or instance.get("state") in {"deleted", "releasing", "deleting"}):
                    raise APIError("NATIVE_DRAINING", "实例正在释放，请回到 Studio 重启", 409)
                now = time.time()
                if not allow_stopping and (instance.get("manual_stop_requested")
                        or now >= float(fresh.get("deadline_epoch") or float("inf"))):
                    raise APIError("NATIVE_DRAINING", "会话已到运行时限或正在停止；等待已有任务及同步结束后重启", 409)
                repo.update("leases", fresh["id"], {
                    "native_last_write": now, "native_http_heartbeat": now,
                }, fresh["owner_id"])
                return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise APIError("SCHEDULER_BUSY", "会话状态繁忙，请稍后重试", 503)
        time.sleep(min(0.02, remaining))


def ready(client=None):
    try:
        url, headers = backend()
        with httpx.Client(timeout=8, follow_redirects=False) if client is None else _borrow(client) as c:
            # Service-token protection of the internal tunnel is mandatory.
            # Probe installation paths with GET only: never submit an installation.
            # Access must protect Manager too, including deployments with path-based rules.
            for path in ("/system_stats", "/v2/manager/queue/status", "/manager/queue/status",
                         "/v2/manager/queue/install_model", "/customnode/install"):
                unauthenticated = c.get(url + path)
                if unauthenticated.status_code not in {401, 403, 302}:
                    return False
            response = c.get(url + "/system_stats", headers=headers)
            response.raise_for_status()
            state = c.get(url + "/studio_native/state", headers=headers)
            state.raise_for_status()
            return isinstance(response.json().get("system"), dict) and isinstance(state.json().get("busy"), bool)
    except Exception:
        return False


def sync_plugins(repo, lease, client=None):
    """Pull committed metadata over the existing authenticated tunnel; no GPU token on VPS."""
    url, headers = backend()
    with httpx.Client(timeout=15) if client is None else _borrow(client) as c:
        response = c.get(url + '/studio_native/state', headers=headers)
        response.raise_for_status()
        status = response.json().get('plugins') or {}
    manifest = status.get('manifest')
    from .node_persistence import digest
    if manifest and (manifest.get('schema') != 1 or manifest.get('generation') != digest(manifest.get('plugins'))):
        raise ValueError('Invalid runtime plugin manifest')
    # Existing settings table: each update commits the entire manifest in one transaction.
    owner = lease['owner_id']
    key = 'native-plugins-' + digest(owner)[:32]
    with repo.store.dispatch_lock() as acquired:
        if not acquired:
            return status
        saved = repo.store.get('settings', key, owner) or {}
        history = saved.get('history', [])
        if manifest and (not history or history[-1]['generation'] != manifest['generation']):
            history = [*history, manifest][-20:]
        data = {'manifest': manifest or saved.get('manifest'), 'status': status, 'history': history}
        if saved:
            if any(saved.get(k) != value for k, value in data.items()):
                repo.store.update('settings', key, data, owner)
        else:
            repo.store.create('settings', owner, data, record_id=key)
    return status
