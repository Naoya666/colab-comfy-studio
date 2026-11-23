"""Durable, lease-scoped native ComfyUI process restarts."""
from __future__ import annotations

import time
import uuid
from contextlib import contextmanager

import httpx

from app.common import APIError

from . import native, runtime_worker
from .drive import DriveStorage
from .manifest import digest
from .native_runtime import _index_models

RECOVERY_SECONDS = 2100
WAIT_HINT = "请等当前操作完成后重试"
RESTART_PHASES = ("checking", "syncing", "rebooting", "recovering")


def _apply_phase_timings(current_op: dict, changes: dict, now: float) -> dict:
    """Record completed restart phase duration and start clock for the next phase."""
    result = {**current_op, **changes}
    old_phase = current_op.get("phase")
    new_phase = result.get("phase")
    timings = dict(current_op.get("phase_timings") or {})
    if isinstance(changes.get("phase_timings"), dict):
        timings.update(changes["phase_timings"])
    if new_phase and new_phase != old_phase:
        if old_phase in RESTART_PHASES:
            started = current_op.get("phase_started_at")
            if not isinstance(started, (int, float)) or isinstance(started, bool):
                started = current_op.get("created_at")
            if isinstance(started, (int, float)) and not isinstance(started, bool):
                prev = timings.get(old_phase)
                base = float(prev) if isinstance(prev, (int, float)) and not isinstance(prev, bool) else 0.0
                timings[old_phase] = round(base + max(0.0, now - float(started)), 1)
        if new_phase in RESTART_PHASES:
            result["phase_started_at"] = now
        elif new_phase in {"succeeded", "failed"}:
            result["phase_started_at"] = None
            result["completed_at"] = now
    result["phase_timings"] = timings
    return result


def _busy(message):
    return APIError("NATIVE_BUSY", f"{message}。{WAIT_HINT}", 409, True)


@contextmanager
def _transition(repo):
    deadline = time.monotonic() + 2
    while True:
        with repo.store.dispatch_lock() as acquired:
            if acquired:
                yield
                return
        if time.monotonic() >= deadline:
            raise APIError("SCHEDULER_BUSY", "会话状态繁忙，请稍后重试", 503, True)
        time.sleep(0.02)


def _validate(repo, lease, instance_id, operation=None, *, check_jobs=True):
    if (not lease or not lease.get("native_comfy") or lease.get("instance_id") != instance_id
            or lease.get("state") == "deleted" or lease.get("native_remote_deleted_verified_at")):
        raise APIError("NATIVE_INSTANCE_GONE", "原生 GPU 实例已释放或已切换", 409)
    if operation and (lease.get("runtime_id") != operation["runtime_id"]
                      or lease.get("deadline_epoch") != operation["lease_deadline"]):
        raise APIError("NATIVE_INSTANCE_GONE", "原生 GPU 租约已切换", 409)
    owner = lease["owner_id"]
    instance = repo.get("instances", instance_id, owner)
    if not instance or instance.get("lease_id") != lease["id"] or instance.get("state") == "deleted":
        raise APIError("NATIVE_INSTANCE_GONE", "原生 GPU 实例已释放或已切换", 409)
    if (instance.get("manual_stop_requested") or lease.get("native_draining") or instance.get("native_draining")
            or time.time() >= float(lease.get("deadline_epoch") or 0)
            or lease.get("state") in {"releasing", "deleting", "cleanup_pending", "quarantined"}
            or instance.get("state") in {"releasing", "deleting", "cleanup_pending", "quarantined"}):
        raise APIError("NATIVE_DRAINING", "原生 GPU 会话正在停止、排空或已到期", 409)
    if not lease.get("native_ready"):
        raise APIError("NATIVE_NOT_READY", "原生 ComfyUI 尚未就绪", 409)
    if lease.get("current_job_id") or instance.get("current_job_id"):
        raise _busy("Studio 生成或启动任务仍在执行")
    if check_jobs:
        from .instances import POOL_NATIVE, job_pool

        for job in repo.list("jobs", owner):
            policy = job.get("execution_policy") or {}
            if (
                (job.get("provider") or policy.get("provider")) == "colab"
                and job.get("state") not in native.TERMINAL
                and not job.get("prep")
                and (
                    job_pool(job) == POOL_NATIVE
                    or (job.get("instance_id") or job.get("requested_instance_id")) == instance_id
                )
            ):
                raise _busy("Studio 生成或启动任务仍在排队")


def _upstream_state(client, url, headers, lease):
    response = client.get(url + "/studio_native/state", headers=headers, timeout=12)
    response.raise_for_status()
    state = response.json()
    if not isinstance(state.get("boot_id"), str) or not state["boot_id"] or not state.get("lease_id"):
        raise APIError("NATIVE_RESTART_UNSUPPORTED", "当前原生运行时未加载重启组件，请在下一次启动的会话中使用", 409)
    if state["lease_id"] != lease["id"]:
        raise APIError("NATIVE_INSTANCE_GONE", "认证隧道已连接到其他原生 GPU 租约", 409)
    return state


def _idle(client, url, headers, lease):
    response = client.get(url + "/queue", headers=headers)
    response.raise_for_status()
    queue = response.json()
    if any(not isinstance(queue.get(key), list) for key in ("queue_running", "queue_pending")):
        raise _busy("无法确认 ComfyUI 生成队列状态")
    if queue["queue_running"] or queue["queue_pending"]:
        raise _busy("ComfyUI 生成仍在执行或排队")
    state = _upstream_state(client, url, headers, lease)
    if state.get("busy") is not False:
        reason = {
            "manager": "Manager 安装任务仍在执行或排队",
            "plugins": "Manager 插件归档尚未同步至 Drive",
            "drive": "Drive 写入或同步尚未完成",
        }.get(state.get("busy_reason"), "无法确认 Manager 队列和 Drive 同步状态")
        raise _busy(reason)
    if state.get("reboot_id"):
        raise _busy("Manager 已受理重启")
    return state


def _fresh(repo, lease):
    return repo.get("leases", lease["id"], lease["owner_id"])


def _save(repo, lease, operation, *, validate=False, **changes):
    with _transition(repo):
        fresh = _fresh(repo, lease)
        if not fresh or (fresh.get("native_restart") or {}).get("id") != operation["id"]:
            return None
        if validate:
            _validate(repo, fresh, operation["instance_id"], operation)
        now = time.time()
        result = _apply_phase_timings(fresh["native_restart"], changes, now)
        patch = {"native_restart": result}
        if changes.get("phase") == "succeeded":
            patch["native_last_busy"] = now
        repo.update("leases", fresh["id"], patch, lease["owner_id"])
        return result


def _fail(repo, lease, operation, error, *, uncertain=False):
    return _save(repo, lease, operation, phase="failed", failed_phase=operation["phase"],
                 error=error, retry_blocked=uncertain, probe_after=time.time() + 15)


def submit(repo, owner, instance_id, key):
    if not isinstance(instance_id, str) or not instance_id or len(instance_id) > 64:
        raise APIError("NATIVE_INPUT", "必须提供当前原生实例 ID", 422)
    if not key or len(key) > 200:
        raise APIError("IDEMPOTENCY_REQUIRED", "必须提供固定幂等键", 422)
    key = digest(key)
    # Admission also guards the Studio queue and instance allocation.
    with _transition(repo), repo.store.admission_lock(timeout=0) as admitted:
        if not admitted:
            raise APIError("SCHEDULER_BUSY", "任务正在入队，请稍后使用相同幂等键重试", 503, True)
        leases = [row for row in repo.list("leases", owner) if row.get("native_comfy")]
        for row in leases:
            previous = (row.get("native_restart_keys") or {}).get(key)
            if previous:
                if row.get("instance_id") != instance_id:
                    raise APIError("IDEMPOTENCY_CONFLICT", "幂等键已用于其他实例", 409)
                return previous
        lease = next((row for row in leases if row.get("instance_id") == instance_id), None)
        _validate(repo, lease, instance_id)
        operation = lease.get("native_restart") or {}
        if operation.get("retry_blocked"):
            raise _busy("正在核对上一次重启的结果")
        if operation.get("phase") == "checking" and operation.get("checking_key") == key:
            return operation["id"]
        if operation.get("phase") == "checking":
            raise _busy("正在确认会话是否空闲")
        keys = dict(lease.get("native_restart_keys") or {})
        if native.restart_active(lease):
            keys[key] = operation["id"]
            repo.update("leases", lease["id"], {"native_restart_keys": keys}, owner)
            return operation["id"]
        now = time.time()
        operation = {
            "id": str(uuid.uuid4()), "instance_id": instance_id, "phase": "checking",
            "runtime_id": lease.get("runtime_id"), "lease_deadline": lease.get("deadline_epoch"),
            "created_at": now, "phase_started_at": now, "phase_timings": {}, "checking_key": key,
        }
        repo.update("leases", lease["id"], {"native_restart": operation}, owner)

    try:
        deadline = time.monotonic() + 8
        while native.active_writes(_fresh(repo, lease) or {}, time.time()):
            if time.monotonic() >= deadline:
                raise _busy("已有生成、安装或 Drive 写入请求仍在处理")
            time.sleep(0.2)
        url, headers = native.backend_for(repo, owner)
        with httpx.Client(timeout=8, follow_redirects=False) as client:
            state = _idle(client, url, headers, lease)
        with _transition(repo):
            fresh = _fresh(repo, lease)
            _validate(repo, fresh, instance_id, operation)
            current = fresh.get("native_restart") or {}
            if current.get("id") != operation["id"] or current.get("phase") != "checking":
                raise _busy("会话状态已更新")
            keys[key] = operation["id"]
            next_op = _apply_phase_timings(
                current, {"phase": "syncing", "boot_id": state["boot_id"]}, time.time()
            )
            repo.update("leases", lease["id"], {
                "native_restart_keys": keys,
                "native_restart": next_op,
            }, owner)
    except (APIError, httpx.HTTPError, ValueError, TypeError, AttributeError) as exc:
        error = exc if isinstance(exc, APIError) else _busy("无法确认 ComfyUI 当前是否空闲")
        _fail(repo, lease, operation, error.message)
        raise error from None
    return operation["id"]


def _sync_models(repo, lease):
    if native.lease_storage(repo, lease)["model_storage_mode"] == "local":
        return []
    owner = lease["owner_id"]
    job = repo.get("jobs", lease.get("native_bootstrap_job_id") or lease["owner_job_id"], owner)
    if not job:
        raise APIError("NATIVE_INDEX_FAILED", "找不到原生实例的 Drive 账号配置", 409)
    drive = DriveStorage(job, repo)
    try:
        models = native.ready_models(repo, owner, drive.account_id)
        cache = drive.model_cache()
        return _index_models({"access_token": cache["access_token"],
                              "models_folder_id": cache["folder_id"], "models": models}, runtime_worker)
    finally:
        drive.close()


def _check_models(client, url, headers, lease, boot_id, models, *, refresh=False):
    if not models:
        return []
    response = client.post(url + "/studio_native/index", headers=headers, timeout=48,
                           json={"lease_id": lease["id"], "boot_id": boot_id,
                                 "models": models, "refresh": refresh})
    response.raise_for_status()
    missing = response.json().get("missing")
    if not isinstance(missing, list):
        raise ValueError("Invalid model index response")
    return missing


def _sync_and_reboot(repo, lease, operation, client, url, headers):
    if time.time() - operation["created_at"] >= RECOVERY_SECONDS:
        return _fail(repo, lease, operation, "同步模型索引超时")
    try:
        aliases = _sync_models(repo, lease)
        missing = _check_models(client, url, headers, lease, operation["boot_id"], aliases, refresh=True)
        if missing:
            return _fail(repo, lease, operation, "Drive 挂载中找不到模型快捷方式：" + str(missing[0]))
    except ValueError as exc:
        error = str(exc) if str(exc).startswith("Drive 模型快捷方式目标冲突：") else "无法验证 Drive 模型索引"
        return _fail(repo, lease, operation, error)
    except Exception as exc:
        if isinstance(exc, APIError):
            message = exc.message
        elif isinstance(getattr(exc, "status_code", getattr(exc, "code", None)), int):
            message = f"Drive 模型索引同步失败（HTTP {getattr(exc, 'status_code', getattr(exc, 'code', 0))}）"
        else:
            message = "无法同步 Drive 模型快捷方式或刷新运行时挂载"
        return _fail(repo, lease, operation, message)

    state = _idle(client, url, headers, lease)
    if state["boot_id"] != operation["boot_id"]:
        return _fail(repo, lease, operation, "同步期间 ComfyUI 进程已切换，请重新检查会话")
    # Persist the send boundary first. A dying worker may leave the outcome
    # unknown, but another worker must never issue this operation again.
    operation = _save(repo, lease, operation, validate=True, phase="rebooting",
                      models=aliases, recover_until=time.time() + RECOVERY_SECONDS)
    if not operation:
        return None
    try:
        response = client.post(url + "/v2/manager/reboot", headers={
            **headers, "Origin": url,
            "X-Studio-Restart-Id": operation["id"], "X-Studio-Restart-Lease": lease["id"],
            "X-Studio-Restart-Boot": operation["boot_id"],
        })
        if 400 <= response.status_code < 500 and response.status_code not in {408, 409, 429}:
            return _fail(repo, lease, operation, f"Manager 拒绝重启请求（HTTP {response.status_code}）")
    except httpx.TransportError:
        pass
    return _save(repo, lease, operation, phase="recovering")


def _recover(repo, lease, operation, client, url, headers):
    review = operation["phase"] == "failed"
    if review and time.time() < operation.get("probe_after", 0):
        return None
    state, healthy = {}, False
    try:
        state = _upstream_state(client, url, headers, lease)
        response = client.get(url + "/system_stats", headers=headers)
        healthy = response.is_success and isinstance(response.json().get("system"), dict)
        if healthy and state["boot_id"] != operation["boot_id"]:
            missing = _check_models(client, url, headers, lease, state["boot_id"], operation["models"])
            if not missing:
                return _save(repo, lease, operation, validate=True, phase="succeeded",
                             retry_blocked=False, failed_phase=None, error=None)
            if time.time() >= operation["recover_until"]:
                return _fail(repo, lease, operation, "加载器列表中找不到模型：" + str(missing[0]))
    except APIError as exc:
        if exc.code == "NATIVE_INSTANCE_GONE":
            raise
    except (httpx.HTTPError, ValueError, TypeError, AttributeError):
        pass
    if review:
        if healthy and (state["boot_id"] != operation["boot_id"] or state.get("reboot_id") is None):
            return _save(repo, lease, operation, validate=True, retry_blocked=False)
        return _save(repo, lease, operation, probe_after=time.time() + 15)
    if time.time() >= operation["recover_until"]:
        reason = "旧 ComfyUI 进程仍在线，等待重启超时" if healthy and state["boot_id"] == operation["boot_id"] else "等待 ComfyUI 恢复超时"
        return _fail(repo, lease, operation, reason,
                     uncertain=not healthy or (state["boot_id"] == operation["boot_id"] and bool(state.get("reboot_id"))))
    return None


def tick(repo):
    for lease in repo.list("leases"):
        operation = lease.get("native_restart") or {}
        phase = operation.get("phase")
        if not native.restart_active(lease) and not operation.get("retry_blocked"):
            continue
        with repo.store.upload_lock("native-restart:" + lease["id"]) as acquired:
            if not acquired:
                continue
            lease = _fresh(repo, lease)
            operation = (lease or {}).get("native_restart") or {}
            phase = operation.get("phase")
            if not lease:
                continue
            if phase == "checking":
                if time.time() - operation["created_at"] > 60:
                    _fail(repo, lease, operation, "检查会话空闲状态超时")
                continue
            if not native.restart_active(lease) and not operation.get("retry_blocked"):
                continue
            try:
                _validate(repo, lease, operation["instance_id"], operation,
                          check_jobs=phase in {"syncing", "rebooting", "recovering"})
                url, headers = native.backend_for(repo, lease["owner_id"])
                with httpx.Client(timeout=8, follow_redirects=False) as client:
                    if phase == "syncing":
                        _sync_and_reboot(repo, lease, operation, client, url, headers)
                    else:
                        if phase == "rebooting":
                            operation = _save(repo, lease, operation, phase="recovering")
                        if operation:
                            _recover(repo, lease, operation, client, url, headers)
            except APIError as exc:
                if exc.code != "SCHEDULER_BUSY":
                    _fail(repo, lease, operation, exc.message)
            except (httpx.HTTPError, ValueError, TypeError, AttributeError):
                if phase == "syncing":
                    _fail(repo, lease, operation, "无法复核 ComfyUI 生成、Manager 或 Drive 同步状态")


def restart_timing_details(state: dict, *, now: float | None = None) -> dict[str, dict]:
    details: dict[str, dict] = {}
    raw_timings = state.get("phase_timings") if isinstance(state.get("phase_timings"), dict) else {}
    for phase in RESTART_PHASES:
        val = raw_timings.get(phase)
        if isinstance(val, (int, float)) and not isinstance(val, bool) and val >= 0:
            details[f"restart_{phase}"] = {
                "seconds": round(float(val), 1),
                "source": "events",
                "status": "completed",
            }
    current = state.get("phase")
    if current in RESTART_PHASES:
        started = state.get("phase_started_at")
        if not isinstance(started, (int, float)) or isinstance(started, bool):
            started = state.get("created_at")
        elapsed = None
        if now is not None and isinstance(started, (int, float)) and not isinstance(started, bool):
            prev = details.get(f"restart_{current}", {}).get("seconds")
            base = float(prev) if isinstance(prev, (int, float)) else 0.0
            elapsed = round(base + max(0.0, now - float(started)), 1)
        details[f"restart_{current}"] = {
            "seconds": elapsed,
            "source": "events",
            "status": "in_progress",
        }
    return details


def public_restart(lease, *, now: float | None = None):
    state = (lease or {}).get("native_restart") or {}
    if not state:
        return None
    current_now = time.time() if now is None else now
    return {
        "restart_id": state["id"],
        "instance_id": state["instance_id"],
        "phase": state["phase"],
        "failed_phase": state.get("failed_phase"),
        "error": state.get("error"),
        "created_at": state.get("created_at"),
        "completed_at": state.get("completed_at"),
        "timing_details": restart_timing_details(state, now=current_now),
    }


def can_restart(repo, lease):
    if not lease or native.restart_active(lease) or (lease.get("native_restart") or {}).get("retry_blocked"):
        return False
    try:
        _validate(repo, lease, lease.get("instance_id"), check_jobs=False)
    except APIError:
        return False
    return True
