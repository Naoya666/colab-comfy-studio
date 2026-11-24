"""Native ComfyUI admission and conservative activity checks on existing leases."""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

import httpx

from app.common import APIError
from app.config import settings
from app.security import decrypt_secret

from . import native_config
from .compute import GpuConfig
from .manifest import digest
from .node_persistence import NATIVE_FLAG_NAMES, env_flag

log = logging.getLogger("studio.native")

TERMINAL = {"completed", "failed", "cancelled", "rejected"}
UNAUTHENTICATED_TUNNEL_PROBES = (
    "/system_stats",
    "/v2/manager/queue/status",
    "/manager/queue/status",
    "/v2/manager/queue/install_model",
    "/customnode/install",
)


def native_optimization_flags() -> dict[str, str]:
    """Snapshot independent startup optimization switches from server environment."""
    return {
        name: ("1" if env_flag(name, settings.gpu_in_vm_self_heal if name == "GPU_IN_VM_SELF_HEAL" else True) else "0")
        for name in NATIVE_FLAG_NAMES
    }


def timeout_options():
    return {
        "defaults": {"retain_idle_minutes": settings.native_comfy_idle_minutes,
                     "max_runtime_minutes": settings.native_comfy_max_runtime_minutes},
        "limits": {"retain_idle_minutes": None, "max_runtime_minutes": None},
    }


def model_storage_defaults():
    """Default model storage configuration for newly started native sessions."""
    return {
        "model_storage_mode": "local",
        "cache_gb": settings.native_comfy_cache_gb,
        "cache_max_age_minutes": settings.native_comfy_cache_max_age_minutes,
    }


def model_storage(values=None, *, validate=False):
    defaults = {"model_storage_mode": "drive_mount", "cache_gb": settings.native_comfy_cache_gb,
                "cache_max_age_minutes": settings.native_comfy_cache_max_age_minutes}
    result = {field: (values or {}).get(field, value) for field, value in defaults.items()}
    if validate:
        if result["model_storage_mode"] not in ("drive_mount", "local"):
            raise APIError("NATIVE_INPUT", "模型存储模式无效", 422)
        for field, limit in (("cache_gb", 2**31 - 1), ("cache_max_age_minutes", 153722867)):
            if type(result[field]) is not int or not 1 <= result[field] <= limit:
                raise APIError("NATIVE_INPUT", f"{field} 必须为正整数，且不超过 {limit}", 422)
    return result


def lease_storage(repo, lease):
    if lease.get("native_model_storage"):
        return model_storage(lease["native_model_storage"])
    job = repo.get("jobs", lease.get("native_bootstrap_job_id") or lease.get("owner_job_id"), lease["owner_id"])
    return model_storage((job or {}).get("execution_policy"))


def model_channel(lease_id, tunnel_token=None, *, repo=None, owner="owner"):
    token = tunnel_token
    if not token:
        reader = None if secret_file is _default_secret_file else secret_file
        cfg = native_config.resolve(repo, owner, use_cache=True, file_reader=reader)
        token = cfg.tunnel_token or secret_file("NATIVE_COMFY_TUNNEL_TOKEN_FILE")
    return hmac.new(token.encode(), ("native-models:" + lease_id).encode(), hashlib.sha256).hexdigest()


_default_model_channel = model_channel


def model_channel_for(lease_id, tunnel_token=None, *, repo=None, owner="owner"):
    if model_channel is not _default_model_channel:
        return model_channel(lease_id) if tunnel_token is None else model_channel(lease_id, tunnel_token)
    return model_channel(lease_id, tunnel_token, repo=repo, owner=owner)


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


_default_secret_file = secret_file


def backend(repo=None, owner="owner"):
    reader = None if secret_file is _default_secret_file else secret_file
    cfg = native_config.resolve(repo, owner, use_cache=True, file_reader=reader)
    url = cfg.native_comfy_backend_url.rstrip("/")
    p = urlsplit(url)
    if p.scheme != "https" or not p.hostname or p.username or p.password or p.path or p.query or p.fragment:
        raise APIError("NATIVE_CONFIG", "NATIVE_COMFY_BACKEND_URL 必须为独立 HTTPS 源地址", 503)
    if p.netloc in {urlsplit(settings.site_url).netloc, urlsplit(cfg.native_comfy_url).netloc}:
        raise APIError("NATIVE_CONFIG", "内部 tunnel 域名必须与 Studio 和原生公网域名不同", 503)
    if cfg.access_client_id and cfg.access_client_secret:
        return url, {"CF-Access-Client-Id": cfg.access_client_id,
                     "CF-Access-Client-Secret": cfg.access_client_secret}
    try:
        access = json.loads(secret_file("NATIVE_COMFY_ACCESS_FILE"))
        if not isinstance(access, dict) or any(not isinstance(access.get(k), str) or not access[k] for k in ("client_id", "client_secret")):
            raise ValueError
    except (ValueError, TypeError) as exc:
        raise APIError("NATIVE_CONFIG", "Access 凭据文件必须包含 client_id 和 client_secret", 503) from exc
    return url, {"CF-Access-Client-Id": access["client_id"],
                 "CF-Access-Client-Secret": access["client_secret"]}


_default_backend = backend


def backend_for(repo=None, owner="owner"):
    if backend is not _default_backend:
        return backend()
    return backend(repo, owner)


_INACTIVE_LEASE_STATES = ("deleted", "quarantined", "releasing", "deleting")
WORKSPACE_WRITE_LOCK_TTL_SECONDS = 90.0


def _workspace_lock_key(owner: str) -> str:
    return "native-workspace-lock-" + digest(owner)[:32]


def plugin_settings_key(owner_id: str, drive_account_id: str | None = None) -> str:
    """Return VPS settings record ID for native plugin state, isolated per resolved Drive account."""
    account = drive_account_id.strip() if isinstance(drive_account_id, str) and drive_account_id.strip() else None
    if not account:
        return "native-plugins-" + digest(owner_id)[:32]
    return "native-plugins-" + digest(f"{owner_id}:{account}")[:32]


def resolve_lease_drive_account_id(
    repo,
    owner_id: str,
    lease: dict | None = None,
    job: dict | None = None,
) -> str | None:
    """Resolve the actual Google Drive storage account mounted by a native lease or job.

    Prefers explicit Drive accounts pinned on the lease or its bootstrap/owner job over
    mutable compute-account delegations so rebinding a compute account mid-session does
    not redirect plugin sync away from the currently mounted Drive.
    """
    from .accounts import AccountService

    def _non_empty(row: dict | None, *fields: str) -> str | None:
        if not isinstance(row, dict):
            return None
        for field in fields:
            val = row.get(field)
            if isinstance(val, str) and val.strip():
                return val.strip()
        return None

    # 1. Explicit Drive account already pinned on lease or passed job.
    for row in (lease, job):
        pinned = _non_empty(row, "native_drive_account_id", "drive_account_id")
        if pinned:
            return pinned

    # 2. Explicit Drive account pinned on the lease's bootstrap / owner job in DB.
    linked_jobs: list[dict] = []
    if isinstance(lease, dict) and repo is not None:
        seen_ids: set[str] = set()
        for jfield in ("native_bootstrap_job_id", "owner_job_id"):
            jid = lease.get(jfield)
            if isinstance(jid, str) and jid and jid not in seen_ids:
                seen_ids.add(jid)
                try:
                    jrow = repo.get("jobs", jid, owner_id)
                except Exception:
                    jrow = None
                if isinstance(jrow, dict):
                    linked_jobs.append(jrow)
                    pinned = _non_empty(jrow, "native_drive_account_id", "drive_account_id")
                    if pinned:
                        return pinned

    # 3. Fallback to compute account_id only when no pinned Drive account exists anywhere.
    candidate = None
    for row in (lease, job, *linked_jobs):
        candidate = _non_empty(row, "account_id")
        if candidate:
            break
    if not candidate:
        return None
    if repo is None:
        return candidate
    try:
        return AccountService(repo, owner_id).resolve_drive_account_id(candidate)
    except Exception:
        return candidate


def active_workspace_write_lock(repo, owner: str = "owner", now: float | None = None) -> dict | None:
    """Return active offline workspace write lock row if held and unexpired."""
    if not hasattr(repo, "store") or not hasattr(repo.store, "get"):
        return None
    row = repo.store.get("settings", _workspace_lock_key(owner), owner)
    if not isinstance(row, dict) or not row.get("lock_id"):
        return None
    current = time.time() if now is None else float(now)
    if float(row.get("expires_at") or 0) <= current:
        return None
    return row


def workspace_offline_status(repo, owner: str = "owner", now: float | None = None) -> tuple[bool, str | None]:
    """Return (offline_writable, lock_reason) for native offline workspace operations."""
    lease = active_lease(repo, owner)
    if lease:
        if lease.get("native_draining"):
            return False, "当前原生 ComfyUI 正在排空停止，请等待实例完全释放后再执行修改"
        if restart_active(lease):
            return False, "当前原生 ComfyUI 正在重启，请等待实例完全停止后再执行修改"
        return False, "当前原生 ComfyUI 实例正在运行，仅支持只读查看；停止实例后可进行停用、删除与缓存清理"
    all_leases = (
        repo.list_by_states("leases", ("deleted", "quarantined"), owner, exclude=True)
        if hasattr(repo, "list_by_states")
        else repo.list("leases", owner)
    )
    lingering = next((x for x in all_leases if x.get("native_comfy") and x.get("state") not in {"deleted", "quarantined"}), None)
    if lingering:
        return False, "当前原生 ComfyUI 实例正在释放或核对中，请等待实例完全停止后再执行修改"
    active_jobs = (
        repo.list_by_states("jobs", TERMINAL, owner, exclude=True)
        if hasattr(repo, "list_by_states")
        else repo.list("jobs", owner)
    )
    running = next(
        (j for j in active_jobs if (j.get("snapshot") or {}).get("native_comfy") and j.get("state") not in TERMINAL),
        None,
    )
    if running:
        return False, "当前原生 ComfyUI 正在启动或收尾中，请等待任务结束并停止实例后再执行修改"
    from . import recovery

    plan = recovery.active_plan(repo, owner, pool="native")
    if plan:
        return False, "当前原生 ComfyUI 正在自动恢复中（等待重新拉起实例），请等待恢复完成或先取消自动恢复后再执行离线修改"
    return True, None


def acquire_workspace_write_lock(
    repo,
    owner: str = "owner",
    *,
    reason: str = "workspace_write",
    ttl_seconds: float = WORKSPACE_WRITE_LOCK_TTL_SECONDS,
    now: float | None = None,
) -> str:
    """Verify native ComfyUI is offline and acquire short-lived write lock mutually exclusive with admit()."""
    with repo.store.dispatch_lock() as acquired:
        if not acquired:
            raise APIError("SCHEDULER_BUSY", "调度正在同步，请稍后重试", 503)
        writable, lock_reason = workspace_offline_status(repo, owner, now=now)
        if not writable:
            raise APIError("NATIVE_INSTANCE_ACTIVE", lock_reason or "当前原生 ComfyUI 处于活动状态，禁止离线修改", 409)
        current = time.time() if now is None else float(now)
        existing = active_workspace_write_lock(repo, owner, now=current)
        if existing:
            raise APIError("NATIVE_WORKSPACE_BUSY", "另一项离线管理写入操作正在进行，请数秒后重试", 409)
        lock_id = uuid4().hex
        key = _workspace_lock_key(owner)
        payload = {
            "lock_id": lock_id,
            "reason": reason,
            "started_at": current,
            "expires_at": current + float(ttl_seconds),
        }
        if repo.store.get("settings", key, owner):
            repo.store.update("settings", key, payload, owner)
        else:
            repo.store.create("settings", owner, payload, record_id=key)
        return lock_id


def _verify_and_renew_workspace_write_lock_locked(
    repo,
    owner: str,
    lock_id: str,
    *,
    ttl_seconds: float = WORKSPACE_WRITE_LOCK_TTL_SECONDS,
    now: float | None = None,
) -> tuple[str, float | None]:
    """Verify and extend an active workspace write lock while caller already holds dispatch_lock()."""
    if not lock_id or not hasattr(repo, "store") or not hasattr(repo.store, "get"):
        return "lost", None
    key = _workspace_lock_key(owner)
    row = repo.store.get("settings", key, owner)
    current = time.time() if now is None else float(now)
    if (
        isinstance(row, dict)
        and row.get("lock_id") == lock_id
        and float(row.get("expires_at") or 0.0) > current
    ):
        writable, _ = workspace_offline_status(repo, owner, now=current)
        if not writable:
            return "lost", None
        existing_exp = float(row.get("expires_at") or 0.0)
        new_expires = max(existing_exp, current + float(ttl_seconds))
        repo.store.update("settings", key, {"expires_at": new_expires}, owner)
        return "renewed", new_expires
    return "lost", None


def renew_workspace_write_lock_status(
    repo,
    owner: str,
    lock_id: str,
    *,
    ttl_seconds: float = WORKSPACE_WRITE_LOCK_TTL_SECONDS,
    timeout: float = 0.5,
    now: float | None = None,
) -> tuple[str, float | None]:
    """Attempt to extend an active, unexpired workspace write lock.

    Returns ``("renewed", new_expires_at)``, ``("lost", None)`` when DB confirms
    the lock expired/changed or native ComfyUI is no longer offline, or
    ``("busy", None)`` when ``dispatch_lock`` could not be acquired within ``timeout``.
    """
    if not lock_id or not hasattr(repo, "store") or not hasattr(repo.store, "dispatch_lock"):
        return "lost", None
    deadline = time.monotonic() + max(0.0, float(timeout))
    while True:
        with repo.store.dispatch_lock() as acquired:
            if acquired:
                return _verify_and_renew_workspace_write_lock_locked(
                    repo, owner, lock_id, ttl_seconds=ttl_seconds, now=now
                )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return "busy", None
        time.sleep(min(0.02, remaining))


def renew_workspace_write_lock(
    repo,
    owner: str,
    lock_id: str,
    *,
    ttl_seconds: float = WORKSPACE_WRITE_LOCK_TTL_SECONDS,
    timeout: float = 0.5,
    now: float | None = None,
) -> bool:
    """Extend an active, unexpired workspace write lock TTL while still offline and owned by lock_id."""
    status, _ = renew_workspace_write_lock_status(
        repo, owner, lock_id, ttl_seconds=ttl_seconds, timeout=timeout, now=now
    )
    return status == "renewed"


def release_workspace_write_lock(repo, owner: str, lock_id: str) -> None:
    """Release workspace write lock if still owned by lock_id (retrying brief lock contention)."""
    if not lock_id or not hasattr(repo, "store") or not hasattr(repo.store, "dispatch_lock"):
        return
    deadline = time.monotonic() + 2.0
    while True:
        with repo.store.dispatch_lock() as acquired:
            if acquired:
                key = _workspace_lock_key(owner)
                row = repo.store.get("settings", key, owner)
                if isinstance(row, dict) and row.get("lock_id") == lock_id:
                    repo.store.update("settings", key, {"lock_id": None, "expires_at": 0.0, "reason": None}, owner)
                return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(0.02, remaining))


def active_lease(repo, owner="owner"):
    if hasattr(repo, "list_by_states"):
        rows = repo.list_by_states("leases", _INACTIVE_LEASE_STATES, owner, exclude=True)
    else:
        rows = repo.list("leases", owner)
    return next((x for x in rows if x.get("native_comfy") and x.get("state") not in _INACTIVE_LEASE_STATES), None)


def restart_active(lease):
    return (lease.get("native_restart") or {}).get("phase") in {"checking", "syncing", "rebooting", "recovering"}


def guard_generation(repo, owner, policy):
    if policy.get("provider") == "colab":
        from .cu_guard import CuGuardService
        CuGuardService(repo).check(owner, "colab", policy.get("account_id"), policy.get("instance_id"))


def busy_state(client=None, *, repo=None, owner="owner", state_payload=None) -> tuple[str, str | None]:
    """Distinguish confirmed busy, confirmed idle, and unreachable/unknown upstream state.

    Returns ``(status, reason)`` where ``status`` is ``"busy"``, ``"idle"``, or ``"unknown"``.
    """
    try:
        url, headers = backend_for(repo, owner)
        with httpx.Client(timeout=8) if client is None else _borrow(client) as c:
            q = c.get(url + "/queue", headers=headers)
            q.raise_for_status()
            data = q.json()
            if not isinstance(data.get("queue_running"), list) or not isinstance(data.get("queue_pending"), list):
                return "unknown", "queue_malformed"
            if data["queue_running"] or data["queue_pending"]:
                return "busy", "queue"
            if state_payload is None:
                state = c.get(url + "/studio_native/state", headers=headers)
                state.raise_for_status()
                state_payload = state.json()
            if not isinstance(state_payload, dict):
                return "unknown", "state_malformed"
            flag = state_payload.get("busy")
            if flag is False:
                return "idle", None
            reason = state_payload.get("busy_reason")
            if flag is True and reason != "unknown":
                return "busy", str(reason) if reason is not None else None
            return "unknown", str(reason or "unknown")
    except Exception:
        return "unknown", "unreachable"


def busy(client=None, *, repo=None, owner="owner", state_payload=None):
    """Unknown is busy: network errors must never stop a video or an upload."""
    status, _ = busy_state(client=client, repo=repo, owner=owner, state_payload=state_payload)
    return status != "idle"


_default_busy = busy


def busy_state_for(repo=None, owner="owner", client=None, *, state_payload=None) -> tuple[str, str | None]:
    if busy is not _default_busy:
        val = busy() if client is None else busy(client)
        return ("busy", None) if val else ("idle", None)
    return busy_state(client=client, repo=repo, owner=owner, state_payload=state_payload)


def busy_for(repo=None, owner="owner", client=None, *, state_payload=None):
    status, _ = busy_state_for(repo=repo, owner=owner, client=client, state_payload=state_payload)
    return status != "idle"


class _borrow:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self.value

    def __exit__(self, *args):
        pass


def _validate_admit_request(repo, owner, body, key):
    cfg = GpuConfig.from_env()
    timeouts = {"retain_idle_minutes": settings.native_comfy_idle_minutes,
                "max_runtime_minutes": settings.native_comfy_max_runtime_minutes}
    for field in timeouts:
        value = body.get(field, timeouts[field])
        if type(value) is not int or not 1 <= value <= 153722867:
            raise APIError("NATIVE_INPUT", f"{field} 必须为正整数（分钟）", 422)
        timeouts[field] = value
    storage = model_storage(body, validate=True)
    ncfg = native_config.resolve(repo, owner, use_cache=False, file_reader=secret_file)
    if (not ncfg.native_comfy_url and not ncfg.native_comfy_backend_url) or not cfg.real_enabled:
        raise APIError("NATIVE_DISABLED", "请先配置原生 ComfyUI 隧道并启用真实 GPU", 503)
    backend_for(repo, owner)
    if not ncfg.tunnel_token:
        secret_file("NATIVE_COMFY_TUNNEL_TOKEN_FILE")
    commit = ncfg.native_comfy_commit
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise APIError("NATIVE_CONFIG", "NATIVE_COMFY_COMMIT 必须锁定 40 位官方提交", 503)
    if not key or len(key) > 200:
        raise APIError("IDEMPOTENCY_REQUIRED", "必须提供固定幂等键", 422)
    generation = body.get("plugin_generation")
    if generation is not None and (not isinstance(generation, str) or not re.fullmatch(r"[0-9a-f]{64}", generation)):
        raise APIError("NATIVE_INPUT", "plugin_generation 必须为历史清单的 64 位 SHA256", 422)
    account = body.get("account_id")
    if not isinstance(account, str) or not account.strip() or not isinstance(body.get("spec"), dict):
        raise APIError("NATIVE_INPUT", "请选择账号和 GPU 规格", 422)
    return cfg, timeouts, storage, commit, account


def _admit_locked(
    repo,
    owner,
    body,
    key,
    *,
    api_key_id=None,
    extra_job_fields: dict | None = None,
    cancel_recovery_plan: bool = False,
    _validated=None,
):
    """Admit a native ComfyUI start job while caller already holds dispatch_lock (non-reentrant)."""
    cfg, timeouts, storage, commit, account = _validated or _validate_admit_request(repo, owner, body, key)
    if hasattr(repo, "get_by_idempotency_key"):
        previous = repo.get_by_idempotency_key("jobs", owner, key)
    else:
        previous = next((j for j in repo.list("jobs", owner) if j.get("idempotency_key") == key), None)
    if previous:
        if previous.get("native_request_hash") != digest(body):
            raise APIError("IDEMPOTENCY_CONFLICT", "幂等键参数冲突", 409)
        return previous
    if active_workspace_write_lock(repo, owner):
        raise APIError("NATIVE_WORKSPACE_LOCKED", "离线工作流或插件管理正在写入 Drive，请数秒后重试启动", 409)
    from .cu_guard import CuGuardService
    CuGuardService(repo).check(owner, "colab", account, body.get("instance_id"))
    active_jobs = (
        repo.list_by_states("jobs", TERMINAL, owner, exclude=True)
        if hasattr(repo, "list_by_states")
        else repo.list("jobs", owner)
    )
    running = next((j for j in active_jobs
                    if (j.get("snapshot") or {}).get("native_comfy") and j["state"] not in TERMINAL), None)
    live = active_lease(repo, owner)
    if running or live:
        source = running or repo.get("jobs", live.get("native_bootstrap_job_id") or live.get("owner_job_id"), owner)
        if api_key_id and source:
            return source
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
        if any(field in body and storage[field] != model_storage(policy)[field] for field in storage):
            raise APIError("NATIVE_ACTIVE", "已有原生会话；请停止实例并重新启动以修改模型存储配置", 409)
        return source
    if body.get("instance_id"):
        raise APIError("INSTANCE_CONFLICT", "原生 ComfyUI 仅使用独立原生实例，不复用其他域实例", 409)
    # Prevent double native VM allocation while an old native lease is still releasing/deleting
    # or while an L3 native recovery plan is actively dispatching/cancelling.
    all_leases = (
        repo.list_by_states("leases", ("deleted", "quarantined"), owner, exclude=True)
        if hasattr(repo, "list_by_states")
        else repo.list("leases", owner)
    )
    lingering_lease = next((x for x in all_leases if x.get("native_comfy") and x.get("state") not in {"deleted", "quarantined"}), None)
    if lingering_lease:
        raise APIError("NATIVE_ACTIVE", "旧原生实例正在释放或核对中，请等待释放完成后再启动", 409)
    from . import recovery
    inflight_plan = recovery.active_plan(repo, owner, pool="native")
    if inflight_plan and inflight_plan.get("state") in {"dispatching", "cancelling"}:
        raise APIError("NATIVE_ACTIVE", "原生 ComfyUI 自动恢复正在推进或取消中，请稍后重试", 409)
    from app.compute_routes import _require_account
    _require_account(repo, owner, account)
    if cancel_recovery_plan:
        with repo.store.db_lock(), repo.store.Session.begin() as session:
            repo.store.begin_immediate(session)
            recovery.cancel_scheduled_native_plans_in_tx(session, owner, time.time())
    from .rates import minute_rate
    rate = minute_rate(body["spec"], fallback_rate=cfg.unit_rate)
    project = "native-comfy-" + owner
    if not repo.store.get("projects", project, owner):
        repo.store.create("projects", owner, {"title": "原生 ComfyUI", "system": True}, record_id=project)
    raw = {"kind": "native-comfy", **body}
    job, _ = repo.create_job(owner, raw, key)
    policy = {"provider": "colab", "lifecycle": "retain", "account_id": account,
              **timeouts, **storage}
    patch = {
        **({"native_api_key_id": api_key_id} if api_key_id else {}),
        "instance_pool": "native",
        "native_request_hash": digest(body), "project_id": project, "provider": "colab",
        "account_id": account, "requested_account_id": account,
        "requested_instance_id": None, "requested_spec": body["spec"],
        "lifecycle": "retain", "execution_policy": policy,
        "model_id": "native-comfy", "workflow_key": "system/native-comfy", "workflow_version": 1,
        "max_runtime_minutes": timeouts["max_runtime_minutes"], "unit_rate": rate,
        "native_session_deadline_epoch": time.time() + timeouts["max_runtime_minutes"] * 60,
        "estimated_units": None,
        "snapshot": {"native_comfy": True, "native_plugin_generation": body.get("plugin_generation"), "bound_graph": {},
                     "manifest": {"reviewed": True, "state": "published", "comfy_commit": commit,
                                  "compute_spec": body["spec"].get("key", body["spec"])}},
    }
    if isinstance(extra_job_fields, dict):
        patch.update(extra_job_fields)
    return repo.update("jobs", job["id"], patch, owner)


def admit(repo, owner, body, key, *, api_key_id=None):
    validated = _validate_admit_request(repo, owner, body, key)
    # A durable dispatch lock serializes admission with both API and worker processes.
    with repo.store.dispatch_lock() as acquired:
        if not acquired:
            raise APIError("SCHEDULER_BUSY", "调度正在同步，请使用相同幂等键重试", 503)
        return _admit_locked(
            repo,
            owner,
            body,
            key,
            api_key_id=api_key_id,
            cancel_recovery_plan=True,
            _validated=validated,
        )


def runtime_secrets(repo, job):
    """Only sent over the trusted Colab execution channel; never saved in a DB row."""
    from .accounts import AccountService
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
        storage = model_storage(job.get("execution_policy"))
        ncfg = native_config.resolve(repo, job["owner_id"], use_cache=False, file_reader=secret_file)
        tunnel = ncfg.tunnel_token or secret_file("NATIVE_COMFY_TUNNEL_TOKEN_FILE")
        local = storage["model_storage_mode"] == "local"
        default = AccountService.default_account_id(job["owner_id"])
        drive_account = drive.account_id or default
        # Pin the exact resolved Drive storage account mounted by this runtime onto both
        # the job and its lease so mid-launch or mid-session rebindings cannot misroute sync_plugins.
        mounted_drive = drive.account_id.strip() if isinstance(drive.account_id, str) and drive.account_id.strip() else None
        if mounted_drive and repo is not None and hasattr(repo, "get") and hasattr(repo, "update"):
            owner_id = job["owner_id"]
            job_id = job.get("id")
            if isinstance(job_id, str) and job_id:
                try:
                    existing_job = repo.get("jobs", job_id, owner_id)
                    if isinstance(existing_job, dict) and (
                        existing_job.get("native_drive_account_id") != mounted_drive
                        or existing_job.get("drive_account_id") != mounted_drive
                    ):
                        repo.update(
                            "jobs",
                            job_id,
                            {"native_drive_account_id": mounted_drive, "drive_account_id": mounted_drive},
                            owner_id,
                        )
                except Exception:
                    pass
            lease_id = job.get("lease_id")
            if isinstance(lease_id, str) and lease_id:
                try:
                    existing_lease = repo.get("leases", lease_id, owner_id)
                    if isinstance(existing_lease, dict) and (
                        existing_lease.get("native_drive_account_id") != mounted_drive
                        or existing_lease.get("drive_account_id") != mounted_drive
                    ):
                        repo.update(
                            "leases",
                            lease_id,
                            {"native_drive_account_id": mounted_drive, "drive_account_id": mounted_drive},
                            owner_id,
                        )
                except Exception:
                    pass
        refresh_token = decrypt_secret(cred["encrypted_refresh_token"])
        extra = {}
        if local:
            from app.native_model_api import runtime_context

            from . import model_dependencies

            context = runtime_context(repo, {"owner_id": job["owner_id"], "native_drive_account_id": drive_account})
            context["drive_auth"][drive_account] = {"client_id": client_id, "client_secret": client_secret,
                                                    "refresh_token": refresh_token}
            extra = {"model_channel": model_channel(job["lease_id"], tunnel), "drive_account": drive_account,
                     "source_tokens": context["tokens"], "drive_auth": context["drive_auth"],
                     "catalog": context["catalog"], "directories": model_dependencies.directories(),
                     "rules": model_dependencies.rules()}
        return {"owner_folder_id": drive._owner_folder_id, "models_folder_id": cache["folder_id"],
                "lease_id": job.get("lease_id"),
                "client_id": client_id, "client_secret": client_secret,
                "refresh_token": refresh_token,
                "access_token": cache["access_token"],
                "tunnel_token": tunnel, **storage, **extra, "max_upload_mb": settings.native_comfy_max_upload_mb,
                "models": [] if local else ready_models(repo, job["owner_id"], drive.account_id)}
    finally:
        drive.close()


def local_catalog(repo, owner):
    from .accounts import AccountService
    from .model_library import external_source

    default = AccountService.default_account_id(owner)
    result = []
    for row in repo.list("model_library", owner):
        binding = external_source(row)
        if row.get("state") != "ready" and not binding:
            continue
        if binding and urlsplit(binding["download_url"]).hostname == "civitai.com" and (
            row.get("source") == "civitai"
            or ("external_source" in row and urlsplit(binding.get("url") or "").hostname == "civitai.com")
        ):
            # Only the server-resolved catalog can grant Civitai provenance.
            binding = {**binding, "kind": "civitai"}
        result.append({"model_type": row["model_type"], "filename": row["filename"],
                       "relative_path": row.get("relative_path"), "legacy_target": row.get("legacy_target"),
                       "sha256": row.get("sha256") or row.get("expected_sha256"),
                       "size": row.get("size") or row.get("expected_size"), "md5": row.get("drive_md5"),
                       "drive_file_id": row.get("drive_file_id") if row.get("state") == "ready" else None,
                       "account": row.get("drive_account_id") or default, "source": binding})
    return result


def ready_models(repo, owner, account):
    from .accounts import AccountService

    default = AccountService.default_account_id(owner)
    return [dict(filename=m.get("filename"), category=m.get("model_type"), file_id=m["drive_file_id"])
            for m in repo.list("model_library", owner)
            if m.get("state") == "ready" and m.get("drive_file_id")
            and (m.get("drive_account_id") or default) == (account or default)]


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


def _smart_keepalive_allowed(repo, lease, now):
    from .instances import POOL_CREATION, instance_pool_of

    instance_id = lease.get("instance_id")
    instance = repo.get("instances", instance_id, lease["owner_id"]) if instance_id else None
    if (
        not instance
        or lease.get("native_comfy")
        or lease.get("native_ready")
        or lease.get("native_draining")
        or lease.get("cleanup_claim_token")
        or instance.get("native_comfy")
        or instance_pool_of(instance) != POOL_CREATION
        or instance.get("runtime_deleted")
        or instance.get("state") != "retained_idle"
        or instance.get("retention_mode") not in {"smart", "retain"}
        or instance.get("lease_id") != lease["id"]
        or instance.get("account_id") != lease.get("account_id")
        or instance.get("current_job_id")
        or instance.get("manual_stop_requested")
        or instance.get("cleanup_claim_token")
    ):
        return False
    hard_deadline = instance.get("hard_deadline_at")
    if hard_deadline is None and lease.get("generation_timeouts"):
        hard_deadline = lease.get("deadline_epoch")
    deadlines = [
        float(lease.get("idle_deadline_at") or 0),
        float(lease.get("deadline_epoch") or 0),
        float(instance.get("idle_deadline_at") or 0),
    ]
    if hard_deadline is not None:
        deadlines.append(float(hard_deadline))
    return now < min(deadlines)


def keepalive_lease(repo, lease, clock=time.time, *, provider_factory=None, smart=False):
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
    identity = _lease_identity(lease)
    if smart:
        # Refresh before sending so a pending watchdog tick observes policy
        # changes, stop requests, or a handoff that followed its snapshot.
        with repo.store.admission_lock() as acquired:
            if not acquired:
                return None
            fresh = repo.get("leases", lease["id"], lease["owner_id"])
            if (
                not fresh
                or fresh.get("state") != "idle"
                or _lease_identity(fresh) != identity
                or not _smart_keepalive_allowed(repo, fresh, clock())
            ):
                return None
            lease = fresh
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
        if (owns_provider or isinstance(provider, GoogleColabProvider)) and hasattr(provider, "close"):
            provider.close()

    if not hasattr(repo, "store") or not hasattr(repo.store, "dispatch_lock"):
        return result
    commit_now = float(clock())
    lock = repo.store.admission_lock if smart else repo.store.dispatch_lock
    with lock() as acquired:
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
        if smart and not _smart_keepalive_allowed(repo, fresh, commit_now):
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


def keepalive_active_lease(repo, lease, clock=time.time, *, provider_factory=None):
    """L0: send lightweight TFE-only keep-alive on active running Colab leases without touching Jupyter kernel."""
    if not env_flag("GPU_ACTIVE_KEEPALIVE", settings.gpu_active_keepalive):
        return None
    if (
        not isinstance(lease, dict)
        or lease.get("provider") != "colab"
        or lease.get("state") != "active"
        or lease.get("force_stop")
        or lease.get("cleanup_claim_token")
        or not lease.get("account_id")
        or not lease.get("tunnel_endpoint")
    ):
        return None
    if provider_factory is None and not GpuConfig.from_env().real_enabled:
        return None
    now = float(clock())
    deadline = lease.get("deadline_epoch")
    if deadline is not None and now >= float(deadline):
        return None
    from .colab import KEEPALIVE_INTERVAL_SECONDS, KEEPALIVE_RETRY_SECONDS, GoogleColabProvider

    prev_status = lease.get("active_keepalive_status")
    interval = (
        KEEPALIVE_RETRY_SECONDS
        if isinstance(prev_status, dict) and prev_status.get("ok") is False
        else KEEPALIVE_INTERVAL_SECONDS
    )
    last_keepalive = float(lease.get("active_last_keepalive_at") or 0)
    if last_keepalive > 0 and now - last_keepalive < interval:
        return None

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
        import inspect

        params = inspect.signature(provider.keep_alive).parameters
        if "tfe_only" in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            result = provider.keep_alive(lease, run_kernel_exec=False, tfe_only=True)
        else:
            result = provider.keep_alive(lease, run_kernel_exec=False)
    except Exception as exc:
        result = {"ok": False, "tfe": f"err:{type(exc).__name__}", "proxy": "skipped", "kernel_exec": "skipped"}
    finally:
        if (owns_provider or isinstance(provider, GoogleColabProvider)) and hasattr(provider, "close"):
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
            or fresh.get("state") != "active"
            or fresh.get("force_stop")
            or fresh.get("cleanup_claim_token")
            or _lease_identity(fresh) != identity
        ):
            return None
        fresh_deadline = fresh.get("deadline_epoch")
        if fresh_deadline is not None and commit_now >= float(fresh_deadline):
            return None
        repo.update(
            "leases",
            fresh["id"],
            {
                "active_last_keepalive_at": commit_now,
                "active_keepalive_status": {
                    "at": commit_now,
                    "ok": bool(result.get("ok")),
                    "tfe": result.get("tfe"),
                },
            },
            fresh["owner_id"],
        )
    return result


def record_runtime_check_hint(repo, lease_id: str, owner_id: str, *, now: float | None = None) -> bool:
    """Record a throttled (>=10s) hint asking _inspect_runtime to probe Colab early."""
    if not lease_id or not owner_id:
        return False
    current = time.time() if now is None else float(now)
    lease = repo.get("leases", lease_id, owner_id)
    if not lease or lease.get("state") == "deleted":
        return False
    prev = float(lease.get("provider_runtime_check_hint_at") or 0)
    if current - prev < 10.0:
        return False
    repo.update("leases", lease_id, {"provider_runtime_check_hint_at": current}, owner_id)
    return True


def native_proxy_healthy(repo, owner: str = "owner", client=None) -> bool:
    """Lightweight check whether the native ComfyUI tunnel responds healthy right now."""
    try:
        url, headers = backend_for(repo, owner)
        with httpx.Client(timeout=3.0, follow_redirects=False) if client is None else _borrow(client) as c:
            resp = c.get(url + "/system_stats", headers=headers)
            return resp.status_code == 200 and isinstance(resp.json().get("system"), dict)
    except Exception:
        return False


def protect_lease(repo, lease, clock=time.time, *, check_remote=True, provider_factory=None):
    """Fence idle shutdown against new uploads/prompts under the existing dispatch lock."""
    if not lease.get("native_comfy") or lease.get("force_stop"):
        return False
    identity = _lease_identity(lease)
    # Reuse one HTTP client and a single /studio_native/state fetch across plugin sync and busy probe.
    if GpuConfig.from_env().real_enabled and sync_plugins is _default_sync_plugins and busy is _default_busy:
        state_payload = None
        with httpx.Client(timeout=15) as shared_client:
            try:
                _, state_payload = sync_plugins(repo, lease, client=shared_client, return_state=True)
            except Exception as exc:
                repo.update("leases", lease["id"], {"native_plugin_sync_error": type(exc).__name__}, lease["owner_id"])
            upstream_status, busy_reason = busy_state_for(
                repo, lease["owner_id"], client=shared_client, state_payload=state_payload
            )
    else:
        if GpuConfig.from_env().real_enabled:
            try:
                sync_plugins(repo, lease)
            except Exception as exc:
                repo.update("leases", lease["id"], {"native_plugin_sync_error": type(exc).__name__}, lease["owner_id"])
                # Busy probe fails closed if persistence has an error or dirty files.
                # A VPS mirror outage must not replace the last committed manifest.
                pass
        upstream_status, busy_reason = busy_state_for(repo, lease["owner_id"])
    upstream_busy = upstream_status != "idle"
    # An offline tunnel also reports "unknown" (conservative busy). It cannot veto reconciliation of
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
        expired = clock() >= float(lease.get("deadline_epoch") or float("inf"))
        if lease.get("force_stop") or instance.get("force_stop"):
            return False
        if lease.get("cu_fuse") == "soft_daily_limit" and not manual:
            retained = upstream_busy or bool(active_writes(lease, clock())) or clock() - float(lease.get("native_http_heartbeat") or 0) < 30
        elif not manual:
            if restart_active(lease) and not expired:
                retained = True
            elif upstream_status == "busy":
                now_ts = clock()
                inherited_idle = lease.get("idle_deadline_at")
                skip_clock_bump = (
                    busy_reason in {"model_preparation", "plugins"}
                    and inherited_idle is not None
                    and lease.get("native_last_busy") is not None
                    and float(lease.get("native_last_busy") or 0) < now_ts
                    and lease.get("native_last_write") is None
                )
                patch = {"native_last_upstream_busy": True}
                if not skip_clock_bump:
                    patch["native_last_busy"] = now_ts
                lease = repo.update("leases", lease["id"], patch, lease["owner_id"])
                retained = True
            elif upstream_status == "unknown":
                # Unreachable/errored tunnel: retain conservatively without fabricating
                # busy evidence or extending idle clocks.
                if lease.get("native_last_upstream_busy") is True:
                    lease = repo.update("leases", lease["id"], {"native_last_upstream_busy": False}, lease["owner_id"])
                retained = True
            elif clock() - float(lease.get("native_http_heartbeat") or 0) < 30:
                if lease.get("native_last_upstream_busy") is not False:
                    lease = repo.update("leases", lease["id"], {"native_last_upstream_busy": False}, lease["owner_id"])
                retained = True
            else:
                if lease.get("native_last_upstream_busy") is not False:
                    lease = repo.update("leases", lease["id"], {"native_last_upstream_busy": False}, lease["owner_id"])
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


def begin_write(repo, lease, *, allow_stopping=False, timeout=2.0, write_id=None):
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
                if restart_active(fresh):
                    raise APIError("NATIVE_RESTARTING", "原生 ComfyUI 正在重启，请等当前操作完成后重试", 409)
                instance = repo.get("instances", fresh.get("instance_id"), fresh["owner_id"]) or {}
                if (instance.get("native_draining")
                        or instance.get("state") in {"deleted", "releasing", "deleting"}):
                    raise APIError("NATIVE_DRAINING", "实例正在释放，请回到 Studio 重启", 409)
                now = time.time()
                if not allow_stopping and (instance.get("manual_stop_requested")
                        or now >= float(fresh.get("deadline_epoch") or float("inf"))):
                    raise APIError("NATIVE_DRAINING", "会话已到运行时限或正在停止；等待已有任务及同步结束后重启", 409)
                patch = {
                    "native_last_write": now, "native_http_heartbeat": now,
                }
                if write_id:
                    patch["native_writes"] = {**active_writes(fresh, now), write_id: now}
                repo.update("leases", fresh["id"], patch, fresh["owner_id"])
                return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise APIError("SCHEDULER_BUSY", "会话状态繁忙，请稍后重试", 503)
        time.sleep(min(0.02, remaining))


def active_writes(lease, now):
    return {key: stamp for key, stamp in (lease.get("native_writes") or {}).items() if now - stamp < 60}


def update_write(repo, lease, write_id, *, finished=False):
    deadline = time.monotonic() + 2
    while True:
        with repo.store.dispatch_lock() as acquired:
            if acquired:
                fresh = repo.get("leases", lease["id"], lease["owner_id"])
                if not fresh or write_id not in (fresh.get("native_writes") or {}):
                    return
                now = time.time()
                writes = active_writes(fresh, now)
                if finished:
                    writes.pop(write_id, None)
                else:
                    writes[write_id] = now
                repo.update("leases", lease["id"], {
                    "native_writes": writes, "native_last_write": now, "native_http_heartbeat": now,
                }, lease["owner_id"])
                return
        if time.monotonic() >= deadline:
            return
        time.sleep(0.02)


def ready(client=None, *, repo=None, owner="owner"):
    started = time.monotonic()
    unauth_codes: list[int] = []
    unauth_max_s = 0.0
    stats_s = 0.0
    stats_code = None
    state_s = 0.0
    state_code = None
    state_busy = None
    state_reason = None
    try:
        url, headers = backend_for(repo, owner)
        with httpx.Client(timeout=8, follow_redirects=False) if client is None else _borrow(client) as c:
            def _timed_get(path: str, req_headers: dict | None = None):
                t0 = time.monotonic()
                resp = c.get(url + path, headers=req_headers) if req_headers is not None else c.get(url + path)
                return resp, time.monotonic() - t0

            # Service-token protection of the internal tunnel is mandatory.
            # Probe installation paths with GET only: never submit an installation.
            # Access must protect Manager too, including deployments with path-based rules.
            if env_flag("NATIVE_FAST_CONTROL_PLANE", True):
                with ThreadPoolExecutor(max_workers=7) as pool:
                    unauth_futures = [pool.submit(_timed_get, path) for path in UNAUTHENTICATED_TUNNEL_PROBES]
                    stats_future = pool.submit(_timed_get, "/system_stats", headers)
                    state_future = pool.submit(_timed_get, "/studio_native/state", headers)
                    for fut in unauth_futures:
                        unauthenticated, elapsed = fut.result()
                        unauth_max_s = max(unauth_max_s, elapsed)
                        unauth_codes.append(unauthenticated.status_code)
                        if unauthenticated.status_code not in {401, 403, 302}:
                            log.info(
                                "native_ready ok=False reason=unauth_open total_s=%.3f unauth_max_s=%.3f unauth_codes=%s",
                                time.monotonic() - started, unauth_max_s, unauth_codes,
                            )
                            return False
                    response, stats_s = stats_future.result()
                    state, state_s = state_future.result()
            else:
                for path in UNAUTHENTICATED_TUNNEL_PROBES:
                    unauthenticated, elapsed = _timed_get(path)
                    unauth_max_s = max(unauth_max_s, elapsed)
                    unauth_codes.append(unauthenticated.status_code)
                    if unauthenticated.status_code not in {401, 403, 302}:
                        log.info(
                            "native_ready ok=False reason=unauth_open total_s=%.3f unauth_max_s=%.3f unauth_codes=%s",
                            time.monotonic() - started, unauth_max_s, unauth_codes,
                        )
                        return False
                response, stats_s = _timed_get("/system_stats", headers)
                state, state_s = _timed_get("/studio_native/state", headers)
            stats_code = response.status_code
            state_code = state.status_code
            response.raise_for_status()
            state.raise_for_status()
            stats_json = response.json()
            state_json = state.json()
            state_busy = state_json.get("busy")
            state_reason = state_json.get("busy_reason")
            ok = isinstance(stats_json.get("system"), dict) and isinstance(state_busy, bool)
            log.info(
                "native_ready ok=%s total_s=%.3f unauth_max_s=%.3f unauth_codes=%s stats_s=%.3f stats_code=%s state_s=%.3f state_code=%s busy=%s busy_reason=%s",
                ok, time.monotonic() - started, unauth_max_s, unauth_codes, stats_s, stats_code, state_s, state_code, state_busy, state_reason,
            )
            return ok
    except Exception as exc:
        log.info(
            "native_ready ok=False total_s=%.3f unauth_max_s=%.3f unauth_codes=%s stats_s=%.3f stats_code=%s state_s=%.3f state_code=%s err=%s",
            time.monotonic() - started, unauth_max_s, unauth_codes, stats_s, stats_code, state_s, state_code, type(exc).__name__,
        )
        return False


_default_ready = ready


def ready_for(repo=None, owner="owner", client=None):
    if ready is not _default_ready:
        return ready() if client is None else ready(client)
    return ready(client=client, repo=repo, owner=owner)


def sync_plugins(repo, lease, client=None, *, return_state=False):
    """Pull committed metadata over the existing authenticated tunnel; no GPU token on VPS."""
    url, headers = backend_for(repo, lease["owner_id"])
    owner = lease["owner_id"]
    from . import recovery
    from .node_persistence import digest
    drive_account_id = resolve_lease_drive_account_id(repo, owner, lease=lease)
    key = plugin_settings_key(owner, drive_account_id)
    models_key = recovery.local_models_key(owner)
    existing_saved = repo.store.get("settings", key, owner) or {}
    existing_models_doc = repo.store.get("settings", models_key, owner) or {}
    existing_by_inst = existing_models_doc.get("by_instance") if isinstance(existing_models_doc.get("by_instance"), dict) else {}
    inst_id = lease.get("instance_id") or lease.get("id")
    existing_inst_models = existing_by_inst.get(inst_id) if isinstance(existing_by_inst.get(inst_id), dict) else {}
    fetched_telemetry = None
    fetched_telemetry_digest = None
    fetched_inventory = None
    fetched_inventory_digest = None
    expected_lease_id = lease.get("id")
    with httpx.Client(timeout=15) if client is None else _borrow(client) as c:
        response = c.get(url + "/studio_native/state", headers=headers)
        response.raise_for_status()
        state_payload = response.json()
        if not isinstance(state_payload, dict):
            raise ValueError("Invalid runtime state payload")
        if isinstance(expected_lease_id, str) and expected_lease_id and "lease_id" in state_payload:
            remote_lease_id = state_payload.get("lease_id")
            if not isinstance(remote_lease_id, str) or remote_lease_id.strip() != expected_lease_id:
                raise ValueError(
                    f"Runtime state lease_id mismatch: expected {expected_lease_id}, got {remote_lease_id!r}"
                )
        status = state_payload.get("plugins") or {}
        remote_tel_digest = status.get("telemetry_digest")
        if (
            isinstance(remote_tel_digest, str)
            and remote_tel_digest
            and remote_tel_digest != existing_saved.get("telemetry_digest")
        ):
            try:
                tel_resp = c.get(url + "/studio_native/plugin-telemetry", headers=headers, timeout=10)
                if tel_resp.status_code == 200:
                    parsed_tel = tel_resp.json()
                    if isinstance(parsed_tel, dict) and parsed_tel.get("schema") == 1:
                        if not parsed_tel.get("gpu_reported") and lease.get("gpu_reported"):
                            parsed_tel["gpu_reported"] = lease.get("gpu_reported")
                        fetched_telemetry = parsed_tel
                        fetched_telemetry_digest = remote_tel_digest
            except (httpx.HTTPError, ValueError):
                pass
        remote_models_digest = state_payload.get("local_models_digest")
        if (
            isinstance(remote_models_digest, str)
            and remote_models_digest
            and remote_models_digest != existing_inst_models.get("digest")
        ):
            try:
                from app import native_model_api

                inv_headers = native_model_api.headers(url, headers, lease, repo=repo)
                inv_resp = c.get(url + "/studio_native/models/inventory", headers=inv_headers, timeout=10)
                if inv_resp.status_code == 200:
                    parsed_inv = inv_resp.json()
                    if isinstance(parsed_inv, dict) and isinstance(parsed_inv.get("recoverable_models"), list):
                        fetched_inventory = parsed_inv
                        fetched_inventory_digest = remote_models_digest
            except Exception:
                pass
    manifest = status.get("manifest")
    if manifest and (manifest.get("schema") != 1 or manifest.get("generation") != digest(manifest.get("plugins"))):
        raise ValueError("Invalid runtime plugin manifest")
    remote_self_heal = state_payload.get("self_heal") if isinstance(state_payload.get("self_heal"), dict) else None
    identity = _lease_identity(lease)
    # Existing settings table: each update commits the entire manifest in one transaction.
    with repo.store.dispatch_lock() as acquired:
        if not acquired:
            return (status, state_payload) if return_state else status
        if active_workspace_write_lock(repo, owner):
            return (status, state_payload) if return_state else status
        lease_id = lease.get("id")
        verified_live_lease = False
        if isinstance(lease_id, str) and lease_id:
            fresh_lease = repo.get("leases", lease_id, owner)
            live = active_lease(repo, owner)
            if fresh_lease is not None or live is not None:
                if (
                    fresh_lease is None
                    or live is None
                    or live.get("id") != lease_id
                    or _lease_identity(fresh_lease) != identity
                ):
                    return (status, state_payload) if return_state else status
                verified_live_lease = True
        saved = repo.store.get("settings", key, owner) or {}
        if not verified_live_lease and isinstance(saved.get("offline_pending"), dict):
            return (status, state_payload) if return_state else status
        history = saved.get("history", [])
        if manifest and (not history or history[-1]["generation"] != manifest["generation"]):
            history = [*history, manifest][-20:]
        offline_pending = saved.get("offline_pending")
        if manifest and isinstance(offline_pending, dict):
            if offline_pending.get("generation") == manifest.get("generation"):
                offline_pending = None
            elif status.get("pending_generation") and status.get("pending_generation") != offline_pending.get("generation"):
                offline_pending = None
        data = {
            "manifest": manifest or saved.get("manifest"),
            "status": status,
            "history": history,
            "telemetry": fetched_telemetry if fetched_telemetry is not None else saved.get("telemetry"),
            "telemetry_digest": fetched_telemetry_digest if fetched_telemetry_digest is not None else saved.get("telemetry_digest"),
            "offline_pending": offline_pending,
        }
        if drive_account_id is not None:
            data["drive_account_id"] = drive_account_id
        if "pending_commit" in saved:
            data["pending_commit"] = saved.get("pending_commit")
        if saved:
            if any(saved.get(k) != value for k, value in data.items()):
                repo.store.update("settings", key, data, owner)
        else:
            repo.store.create("settings", owner, data, record_id=key)

        # Persist in-VM self-heal telemetry & audit events (verifying lease identity)
        if remote_self_heal is not None:
            fresh_lease = repo.get("leases", lease["id"], owner)
            if fresh_lease and _lease_identity(fresh_lease) == identity:
                prev_heal = fresh_lease.get("native_self_heal") or {}
                if prev_heal != remote_self_heal:
                    repo.update("leases", fresh_lease["id"], {"native_self_heal": remote_self_heal}, owner)
                    job_id = fresh_lease.get("native_bootstrap_job_id") or fresh_lease.get("owner_job_id")
                    if job_id and repo.get("jobs", job_id, owner):
                        for field, label in (
                            ("comfy_restarts", "ComfyUI 进程异常退出，已在同机自动恢复"),
                            ("rclone_restarts", "rclone Drive 挂载进程异常退出，已在同机自动重挂"),
                            ("tunnel_restarts", "cloudflared 隧道进程异常退出，已在同机自动拉起"),
                        ):
                            if int(remote_self_heal.get(field) or 0) > int(prev_heal.get(field) or 0):
                                repo.event(job_id, owner, "running", label)

        # Persist per-instance local models snapshot (verifying active lease has not switched;
        # NEVER overwrite a frozen recovery plan target list while restoring_env is running).
        if fetched_inventory is not None and fetched_inventory_digest:
            live = active_lease(repo, owner)
            if live and live.get("id") == lease["id"] and _lease_identity(live) == identity:
                now_ts = time.time()
                saved_models = repo.store.get("settings", models_key, owner) or {}
                by_inst = dict(saved_models.get("by_instance") or {})
                by_inst[inst_id] = {
                    "lease_id": lease["id"],
                    "digest": fetched_inventory_digest,
                    "recoverable_models": list(fetched_inventory.get("recoverable_models") or []),
                    "wild_models": [str(w) for w in (fetched_inventory.get("wild_models") or []) if isinstance(w, str)],
                    "updated_at": now_ts,
                }
                # Keep only current active instance + at most 1 most recent prior instance
                if len(by_inst) > 2:
                    others = sorted(
                        [(k, v) for k, v in by_inst.items() if k != inst_id],
                        key=lambda kv: float((kv[1] or {}).get("updated_at") or 0.0),
                        reverse=True,
                    )
                    by_inst = {inst_id: by_inst[inst_id], **{k: v for k, v in others[:1]}}
                payload_doc = {"by_instance": by_inst, "updated_at": now_ts}
                if saved_models:
                    repo.store.update("settings", models_key, payload_doc, owner)
                else:
                    repo.store.create("settings", owner, payload_doc, record_id=models_key)

                # Union-append newly added models into an active restoring_env plan without ever dropping frozen targets
                active_rec = recovery.active_plan(repo, owner, pool="native")
                if (
                    active_rec
                    and active_rec.get("state") == "restoring_env"
                    and active_rec.get("successor_instance_id") == inst_id
                ):
                    ns = dict(active_rec.get("native_snapshot") or {})
                    merged = recovery.merge_recoverable_models(
                        ns.get("recoverable_models"),
                        fetched_inventory.get("recoverable_models"),
                    )
                    wild_union = sorted(
                        set(ns.get("wild_models") or [])
                        | {str(w) for w in (fetched_inventory.get("wild_models") or []) if isinstance(w, str)}
                    )
                    if merged != ns.get("recoverable_models") or wild_union != ns.get("wild_models"):
                        ns["recoverable_models"] = merged
                        ns["wild_models"] = wild_union
                        active_rec["native_snapshot"] = ns
                        active_rec["updated_at"] = now_ts
                        recovery.save_plan(repo, owner, active_rec)
    return (status, state_payload) if return_state else status


_default_sync_plugins = sync_plugins
