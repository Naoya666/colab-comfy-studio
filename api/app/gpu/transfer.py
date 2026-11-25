"""Explicit cross-domain GPU instance transfer coordinator.

Preserves the physical Colab GPU VM (1 VM = 1 instance = 1 lease = 1 CU ledger
segment) while cleanly tearing down the old domain environment inside the VM,
verifying global site-packages purity against the VPS-stored baseline digest,
and atomically handing the lease to the target domain.

Design specification: docs/specs/GPU_INSTANCE_DOMAIN_TRANSFER.md.
"""

from __future__ import annotations

import copy
import logging
import os
import re
import time
import uuid
from typing import Any, Callable, Mapping

from sqlalchemy import select

from app.common import APIError
from app.config import settings
from app.db import ProjectRow, utcnow

from . import ccu, native, native_config, recovery
from .accounts import AccountService
from .colab import RUNTIME_MISSING
from .compute import GpuConfig
from .cu_guard import CuGuardService
from .google import GoogleService
from .instances import (
    POOL_CREATION,
    POOL_EPHEMERAL,
    POOL_NATIVE,
    VALID_POOLS,
    instance_pool_of,
)
from .manifest import digest
from .mcp import MCPTransient
from .repository import GpuEventRow, GpuInstanceRow, GpuJobRow, GpuLeaseRow
from .specs import spec_key

log = logging.getLogger("studio.transfer")

DEFAULT_TRANSFER_HARD_TIMEOUT_SECONDS = 180.0
EPHEMERAL_STANDBY_MINUTES = 10
NATIVE_STANDBY_MINUTES = 10
DEFAULT_CREATION_IDLE_MINUTES = 15
TERMINAL_JOB_STATES = frozenset({"completed", "failed", "cancelled", "rejected"})

_POOL_LABELS = {
    POOL_CREATION: "创作池",
    POOL_NATIVE: "原生 ComfyUI",
    POOL_EPHEMERAL: "冒烟域",
}

_OLD_NATIVE_LEASE_FIELDS = (
    "native_comfy",
    "native_ready",
    "native_draining",
    "native_bootstrap_job_id",
    "native_model_storage",
    "native_idle_minutes",
    "native_last_busy",
    "native_last_write",
    "native_http_heartbeat",
    "native_writes",
    "native_restart",
    "native_self_heal",
    "native_plugin_sync_error",
    "native_last_upstream_busy",
    "native_remote_deleted_verified_at",
    "native_bootstrap_failed",
    "native_drive_account_id",
)


def transfer_hard_timeout_seconds() -> float:
    raw = os.getenv("GPU_TRANSFER_TIMEOUT_SECONDS")
    if raw is not None and raw.strip():
        try:
            val = float(raw.strip())
            if val >= 5.0:
                return val
        except ValueError:
            pass
    return DEFAULT_TRANSFER_HARD_TIMEOUT_SECONDS


def _instance_spec_key(data: Mapping[str, Any] | None) -> dict[str, str] | None:
    if not isinstance(data, Mapping):
        return None
    raw = data.get("runtime_spec_reported") or data.get("runtime_spec_requested")
    if not isinstance(raw, Mapping):
        return None
    inner = raw.get("key") if isinstance(raw.get("key"), Mapping) else raw
    if not isinstance(inner, Mapping):
        return None
    try:
        return spec_key(dict(inner))
    except ValueError:
        if all(k in inner for k in ("variant", "accelerator", "shape")):
            return {k: str(inner[k]) for k in ("variant", "accelerator", "shape")}
        return None


def _is_gpu_instance(data: Mapping[str, Any] | None) -> bool:
    if not isinstance(data, Mapping):
        return False
    spec = _instance_spec_key(data)
    if spec is not None:
        return spec.get("variant") in {"GPU", "VARIANT_GPU"}
    # Isolated mock tests may not attach an explicit spec dict; treat non-CPU mock as GPU.
    if data.get("provider") == "mock":
        return True
    return False


def _vm_allocated_at(inst: Mapping[str, Any], lease: Mapping[str, Any] | None = None) -> float | None:
    for candidate in (
        inst.get("vm_allocated_at"),
        (lease or {}).get("vm_allocated_at"),
        (lease or {}).get("create_started"),
        inst.get("allocation_requested_at"),
    ):
        if isinstance(candidate, (int, float)) and not isinstance(candidate, bool) and float(candidate) > 0:
            return float(candidate)
    return None


def _has_active_job_for_instance_in_tx(session, owner_id: str, instance_id: str, *, exclude_job_id: str | None = None) -> bool:
    for row in session.scalars(select(GpuJobRow).where(GpuJobRow.owner_id == owner_id)).all():
        if row.id == exclude_job_id or row.state in TERMINAL_JOB_STATES:
            continue
        jdata = row.data or {}
        policy = jdata.get("execution_policy") if isinstance(jdata.get("execution_policy"), Mapping) else {}
        if (
            row.instance_id == instance_id
            or jdata.get("instance_id") == instance_id
            or jdata.get("requested_instance_id") == instance_id
            or policy.get("instance_id") == instance_id
            or policy.get("requested_instance_id") == instance_id
        ):
            return True
    return False


def _has_active_recovery_for_instance_in_tx(session, owner_id: str, instance_id: str) -> bool:
    for plan in recovery.list_plans_in_tx(session, owner_id):
        if plan.get("state") not in recovery.ACTIVE_PLAN_STATES:
            continue
        if plan.get("source_instance_id") == instance_id or plan.get("successor_instance_id") == instance_id:
            return True
    return False


def _native_slot_blocker_in_tx(
    session,
    repo,
    owner_id: str,
    *,
    exclude_instance_id: str | None = None,
) -> tuple[str, str] | None:
    """Return ``(error_code, message)`` if the owner's single native slot is occupied."""
    if native.active_workspace_write_lock(repo, owner_id):
        return (
            "NATIVE_WORKSPACE_LOCKED",
            "离线工作流或插件管理正在写入 Drive，请数秒后重试转移",
        )
    for lrow in session.scalars(select(GpuLeaseRow).where(GpuLeaseRow.owner_id == owner_id)).all():
        if lrow.state == "deleted":
            continue
        ldata = lrow.data or {}
        l_inst = lrow.instance_id or ldata.get("instance_id")
        if exclude_instance_id and l_inst == exclude_instance_id:
            continue
        if ldata.get("native_comfy") or ldata.get("instance_pool") == POOL_NATIVE:
            if lrow.state == "quarantined":
                return (
                    "NATIVE_ACTIVE",
                    "原生域存在保护隔离中的旧实例，请先在实例中心处理后再转入原生域",
                )
            return (
                "NATIVE_ACTIVE",
                "原生域已有活跃或释放中的实例，请先停止旧原生实例后再转入",
            )
    for irow in session.scalars(select(GpuInstanceRow).where(GpuInstanceRow.owner_id == owner_id)).all():
        if irow.state == "deleted":
            continue
        if exclude_instance_id and irow.id == exclude_instance_id:
            continue
        idata = irow.data or {}
        if instance_pool_of(idata) == POOL_NATIVE:
            if irow.state == "quarantined":
                return (
                    "NATIVE_ACTIVE",
                    "原生域存在保护隔离中的旧实例，请先在实例中心处理后再转入原生域",
                )
            return (
                "NATIVE_ACTIVE",
                "原生域已有活跃或释放中的实例，请先停止旧原生实例后再转入",
            )
        if irow.state == "transferring" and (idata.get("transfer_state") or {}).get("to_pool") == POOL_NATIVE:
            return (
                "NATIVE_ACTIVE",
                "已有其他 GPU 实例正在转移至原生域，原生域同时仅允许 1 张 GPU",
            )
    for jrow in session.scalars(select(GpuJobRow).where(GpuJobRow.owner_id == owner_id)).all():
        if jrow.state in TERMINAL_JOB_STATES:
            continue
        jdata = jrow.data or {}
        if (jdata.get("snapshot") or {}).get("native_comfy"):
            j_inst = jrow.instance_id or jdata.get("instance_id")
            if exclude_instance_id and j_inst == exclude_instance_id:
                continue
            return (
                "NATIVE_ACTIVE",
                "原生域已有进行中的启动任务，请等待结束或停止后再转入",
            )
    for plan in recovery.list_plans_in_tx(session, owner_id):
        if plan.get("pool") == POOL_NATIVE and plan.get("state") in recovery.ACTIVE_PLAN_STATES:
            return (
                "NATIVE_ACTIVE",
                "原生域存在进行中的自动恢复计划，请先取消自动恢复后再转入",
            )
    return None


def _check_native_config_and_drive(
    repo,
    owner_id: str,
    account_id: str | None,
    provider: str,
) -> tuple[str, str, int] | None:
    """Validate tunnel config and Drive authorization required for transferring to native."""
    if provider == "mock":
        return None
    cfg = GpuConfig.from_env()
    reader = None if native.secret_file is native._default_secret_file else native.secret_file
    ncfg = native_config.resolve(repo, owner_id, use_cache=False, file_reader=reader)
    if (not ncfg.native_comfy_url and not ncfg.native_comfy_backend_url) or not cfg.real_enabled:
        return ("NATIVE_CONFIG", "请先配置原生 ComfyUI 隧道并启用真实 GPU", 503)
    try:
        native.backend_for(repo, owner_id)
        if not ncfg.tunnel_token:
            native.secret_file("NATIVE_COMFY_TUNNEL_TOKEN_FILE")
    except APIError as exc:
        return (exc.code, exc.message, exc.status)
    commit = ncfg.native_comfy_commit
    if not re.fullmatch(r"[0-9a-f]{40}", commit or ""):
        return ("NATIVE_CONFIG", "NATIVE_COMFY_COMMIT 必须锁定 40 位官方提交", 503)
    if not account_id:
        return ("ACCOUNT_REQUIRED", "实例缺少绑定的 Google 账号，无法转入原生域", 409)
    try:
        drive_account_id = AccountService(repo, owner_id).resolve_drive_account_id(account_id)
        drive_status = GoogleService(repo, owner_id, account_id=drive_account_id).status().get("drive") or {}
        if not drive_status.get("connected"):
            return (
                "DRIVE_NOT_CONNECTED",
                "该实例对应存储账号尚未授权 Google Drive，请先完成 Drive 授权后再转入原生域",
                409,
            )
    except Exception:
        return (
            "DRIVE_NOT_CONNECTED",
            "无法验证该实例存储账号的 Google Drive 授权状态，请稍后重试",
            409,
        )
    return None


def _validate_target_options(target_pool: str, body: Mapping[str, Any]) -> dict[str, Any]:
    if target_pool == POOL_CREATION:
        raw = body.get("creation_options")
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise APIError("INVALID_TRANSFER_OPTIONS", "creation_options 必须为对象", 422)
        unknown = set(raw) - {"retain_idle_minutes"}
        if unknown:
            raise APIError("INVALID_TRANSFER_OPTIONS", "creation_options 包含不支持的参数", 422)
        minutes = raw.get("retain_idle_minutes", DEFAULT_CREATION_IDLE_MINUTES)
        if isinstance(minutes, bool) or not isinstance(minutes, (int, float)) or not (1 <= float(minutes) <= 30):
            raise APIError("INVALID_TRANSFER_OPTIONS", "创作池空闲保留时长须在 1~30 分钟之间", 422)
        return {"retain_idle_minutes": int(round(float(minutes)))}

    if target_pool == POOL_EPHEMERAL:
        return {"retain_idle_minutes": EPHEMERAL_STANDBY_MINUTES}

    if target_pool == POOL_NATIVE:
        raw = body.get("native_options")
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise APIError("INVALID_TRANSFER_OPTIONS", "native_options 必须为对象", 422)
        allowed = {
            "model_storage_mode",
            "cache_gb",
            "cache_max_age_minutes",
            "retain_idle_minutes",
            "max_runtime_minutes",
            "plugin_generation",
        }
        if set(raw) - allowed:
            raise APIError("INVALID_TRANSFER_OPTIONS", "native_options 包含不支持的参数", 422)
        merged_storage_input = {**native.model_storage_defaults(), **dict(raw)}
        storage = native.model_storage(merged_storage_input, validate=True)
        timeouts = {
            "retain_idle_minutes": settings.native_comfy_idle_minutes,
            "max_runtime_minutes": settings.native_comfy_max_runtime_minutes,
        }
        for field in timeouts:
            value = raw.get(field, timeouts[field])
            if type(value) is not int or not 1 <= value <= 153722867:
                raise APIError("NATIVE_INPUT", f"{field} 必须为正整数（分钟）", 422)
            timeouts[field] = value
        gen = raw.get("plugin_generation")
        if gen is not None and (not isinstance(gen, str) or not re.fullmatch(r"[0-9a-f]{64}", gen)):
            raise APIError("NATIVE_INPUT", "plugin_generation 必须为历史清单的 64 位 SHA256", 422)
        return {**storage, **timeouts, "plugin_generation": gen}

    raise APIError("INVALID_TRANSFER_TARGET", "目标域无效，仅支持 creation / native / ephemeral", 422)


def _check_base_transferability_in_tx(
    session,
    repo,
    owner_id: str,
    inst_row: GpuInstanceRow,
    lease_row: GpuLeaseRow | None,
    now: float,
) -> tuple[bool, str | None, str | None]:
    """Return ``(ok, error_code, reason_message)`` for base transfer preconditions."""
    idata = inst_row.data or {}
    ldata = (lease_row.data or {}) if lease_row is not None else {}
    provider = str(idata.get("provider") or "mock")
    pool = instance_pool_of(idata)

    if inst_row.state not in {"ready", "retained_idle"}:
        if inst_row.state == "transferring":
            return False, "INSTANCE_TRANSFER_BUSY", "实例正在执行域转移，请等待当前转移完成"
        if inst_row.state in {"busy", "allocating"}:
            return False, "INSTANCE_TRANSFER_BUSY", "请等待当前任务/同步完成"
        return False, "INSTANCE_TRANSFER_UNAVAILABLE", f"实例当前状态（{inst_row.state}）不可转移"
    if provider not in {"colab", "mock"}:
        return False, "INSTANCE_TRANSFER_UNAVAILABLE", "当前实例提供方不支持域转移"
    if not _is_gpu_instance(idata):
        return False, "INSTANCE_TRANSFER_UNAVAILABLE", "仅 GPU 实例支持域转移（CPU 预处理实例不可转移）"
    baseline = idata.get("vm_baseline_digest") or ldata.get("vm_baseline_digest")
    if idata.get("venv_isolated") is not True or not baseline:
        return False, "INSTANCE_TRANSFER_UNAVAILABLE", "旧版运行时环境实例不支持域转移"
    if (
        idata.get("manual_stop_requested")
        or idata.get("force_stop")
        or idata.get("runtime_deleted")
        or idata.get("cleanup_claim_token")
    ):
        return False, "INSTANCE_TRANSFER_UNAVAILABLE", "实例正在停止或已释放，不可转移"
    if idata.get("current_job_id") is not None or inst_row.current_job_id is not None:
        return False, "INSTANCE_TRANSFER_BUSY", "请等待当前任务/同步完成"
    if lease_row is None:
        if provider == "colab":
            return False, "INSTANCE_TRANSFER_UNAVAILABLE", "实例缺少有效 GPU 租约，不可转移"
    else:
        if (
            lease_row.state != "idle"
            or ldata.get("current_job_id") is not None
            or ldata.get("cleanup_claim_token")
            or ldata.get("force_stop")
        ):
            return False, "INSTANCE_TRANSFER_BUSY", "请等待当前任务/同步完成"
        if not ldata.get("handle") and provider == "colab":
            return False, "INSTANCE_TRANSFER_UNAVAILABLE", "实例运行时尚未确认分配，不可转移"
    if _has_active_job_for_instance_in_tx(session, owner_id, inst_row.id):
        return False, "INSTANCE_TRANSFER_BUSY", "请等待当前任务/同步完成"
    if _has_active_recovery_for_instance_in_tx(session, owner_id, inst_row.id):
        return False, "INSTANCE_TRANSFER_BLOCKED", "实例存在进行中的自动恢复计划，不可转移"
    hard = idata.get("hard_deadline_at")
    if hard is not None and now >= float(hard):
        return False, "INSTANCE_TRANSFER_BLOCKED", "实例已到达会话运行上限，不可转移"
    is_native_ready = pool == POOL_NATIVE and bool(ldata.get("native_ready")) and not idata.get("native_bootstrap_failed")
    if inst_row.state == "retained_idle" and not is_native_ready:
        idle_deadlines = [
            float(v)
            for v in (idata.get("idle_deadline_at"), ldata.get("idle_deadline_at"))
            if v is not None
        ]
        if idle_deadlines and now >= min(idle_deadlines):
            return False, "INSTANCE_TRANSFER_UNAVAILABLE", "实例空闲待命已到期，不可转移"
    if provider == "colab":
        account_id = inst_row.account_id or idata.get("account_id")
        guard = CuGuardService(repo, lambda: now)
        guard_state = guard.state_in(session, owner_id, fail_closed_on_sample_error=True)
        try:
            guard.filter_accounts(guard_state, [account_id] if account_id else [])
        except APIError as exc:
            return False, exc.code, exc.message
    if pool == POOL_NATIVE and not idata.get("native_bootstrap_failed"):
        if (
            idata.get("native_draining")
            or ldata.get("native_draining")
            or native.restart_active(ldata)
            or bool(native.active_writes(ldata, now))
        ):
            return False, "INSTANCE_TRANSFER_BUSY", "请等待当前任务/同步完成"
    return True, None, None


def preview_transfer(
    repo,
    owner_id: str,
    instance_id: str,
    *,
    is_admin: bool = False,
    clock: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """Return transfer eligibility, blocked reasons, and target defaults for UI modal (Spec §7.1(2))."""
    now = float(clock())
    with repo.store.db_lock(), repo.store.Session() as session:
        inst = session.scalar(
            select(GpuInstanceRow).where(GpuInstanceRow.id == instance_id, GpuInstanceRow.owner_id == owner_id)
        )
        if inst is None:
            raise APIError("NOT_FOUND", "实例不存在或不属于当前用户", 404)
        idata = copy.deepcopy(inst.data or {})
        lease_id = idata.get("lease_id")
        lease = (
            session.scalar(
                select(GpuLeaseRow).where(GpuLeaseRow.id == lease_id, GpuLeaseRow.owner_id == owner_id)
            )
            if lease_id
            else None
        )
        ldata = copy.deepcopy(lease.data or {}) if lease is not None else {}
        current_pool = instance_pool_of(idata)
        provider = str(idata.get("provider") or "mock")
        account_id = inst.account_id or idata.get("account_id")
        account_alias = idata.get("account_alias_snapshot")
        allocated_at = _vm_allocated_at(idata, ldata)
        elapsed_s = max(0, int(round(now - allocated_at))) if allocated_at is not None else None
        venv_isolated = bool(idata.get("venv_isolated") is True and (idata.get("vm_baseline_digest") or ldata.get("vm_baseline_digest")))
        transferable, blocked_code, blocked_reason = _check_base_transferability_in_tx(
            session, repo, owner_id, inst, lease, now
        )
        native_slot_block = (
            _native_slot_blocker_in_tx(session, repo, owner_id, exclude_instance_id=instance_id)
            if current_pool != POOL_NATIVE
            else None
        )
        version = int(inst.version or 0)

    native_cfg_block = (
        _check_native_config_and_drive(repo, owner_id, account_id, provider)
        if current_pool != POOL_NATIVE
        else None
    )

    targets: dict[str, dict[str, Any]] = {}
    if current_pool != POOL_CREATION:
        targets[POOL_CREATION] = {
            "allowed": bool(transferable),
            "reason": None if transferable else blocked_reason,
            "defaults": {"retain_idle_minutes": DEFAULT_CREATION_IDLE_MINUTES},
        }
    if current_pool != POOL_NATIVE:
        if not transferable:
            n_allowed, n_reason = False, blocked_reason
        elif native_slot_block is not None:
            n_allowed, n_reason = False, native_slot_block[1]
        elif native_cfg_block is not None:
            n_allowed, n_reason = False, native_cfg_block[1]
        else:
            n_allowed, n_reason = True, None
        targets[POOL_NATIVE] = {
            "allowed": n_allowed,
            "reason": n_reason,
            "defaults": {
                **native.model_storage_defaults(),
                **native.timeout_options()["defaults"],
            },
        }
    if current_pool != POOL_EPHEMERAL:
        if not is_admin:
            e_allowed, e_reason = False, "仅管理员可将实例转移至冒烟域"
        elif not transferable:
            e_allowed, e_reason = False, blocked_reason
        else:
            e_allowed, e_reason = True, None
        targets[POOL_EPHEMERAL] = {
            "allowed": e_allowed,
            "reason": e_reason,
            "defaults": {"retain_idle_minutes": EPHEMERAL_STANDBY_MINUTES},
        }

    return {
        "instance_id": instance_id,
        "version": version,
        "current_pool": current_pool,
        "venv_isolated": venv_isolated,
        "native_bootstrap_failed": bool(idata.get("native_bootstrap_failed")),
        "vm_allocated_at": allocated_at,
        "vm_elapsed_seconds": elapsed_s,
        "account_id": account_id,
        "account_alias": account_alias,
        "spec": _instance_spec_key(idata),
        "gpu_reported": idata.get("gpu_reported"),
        "transferable": bool(transferable),
        "blocked_code": blocked_code,
        "blocked_reason": blocked_reason,
        "targets": targets,
    }


def start_transfer(
    repo,
    instance_service,
    owner_id: str,
    instance_id: str,
    body: Mapping[str, Any],
    key: str,
    *,
    is_admin: bool = False,
    clock: Callable[[], float] = time.time,
    native_busy_probe: Callable[[], tuple[str, str | None]] | None = None,
) -> dict[str, Any]:
    """Validate preconditions, probe live native activity outside locks, and atomically initiate transfer."""
    if not isinstance(key, str) or not key.strip() or len(key) > 200:
        raise APIError("IDEMPOTENCY_REQUIRED", "域转移必须提供固定幂等键", 422)
    key = key.strip()
    if not isinstance(body, Mapping) or body.get("confirmed") is not True:
        raise APIError("CONFIRMATION_REQUIRED", "域转移必须明确 confirmed=true", 422)
    target_pool = body.get("target_pool")
    if target_pool not in VALID_POOLS:
        raise APIError("INVALID_TRANSFER_TARGET", "目标域无效，仅支持 creation / native / ephemeral", 422)
    if target_pool == POOL_EPHEMERAL and not is_admin:
        raise APIError("FORBIDDEN", "仅管理员可将实例转移至冒烟域", 403)
    expected_version = body.get("expected_version")
    if expected_version is not None and (isinstance(expected_version, bool) or not isinstance(expected_version, int) or expected_version < 0):
        raise APIError("VALIDATION_ERROR", "expected_version 格式无效", 422)
    normalized_options = _validate_target_options(target_pool, body)
    canonical_body = {
        "target_pool": target_pool,
        "confirmed": True,
        "expected_version": expected_version,
        "options": normalized_options,
    }
    body_hash = digest(canonical_body)

    # 1) Fast read-only preflight + idempotency short-circuit outside write locks.
    op_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"colab-gpu-operation:{owner_id}:{instance_id}:transfer:{key}"))
    with repo.store.db_lock(), repo.store.Session() as session:
        op_model = instance_service._model("operations")
        prior_op = session.scalar(select(op_model).where(op_model.id == op_id))
        if prior_op is not None:
            op_data = prior_op.data or {}
            if op_data.get("body_hash") != body_hash:
                raise APIError("IDEMPOTENCY_CONFLICT", "相同幂等键不能用于不同的转移参数", 409)
            if prior_op.state == "completed" and op_data.get("result") is not None:
                return copy.deepcopy(op_data["result"])
        pre_inst = session.scalar(
            select(GpuInstanceRow).where(GpuInstanceRow.id == instance_id, GpuInstanceRow.owner_id == owner_id)
        )
        if pre_inst is None:
            raise APIError("NOT_FOUND", "实例不存在或不属于当前用户", 404)
        pre_idata = copy.deepcopy(pre_inst.data or {})
        pre_pool = instance_pool_of(pre_idata)
        pre_provider = str(pre_idata.get("provider") or "mock")
        pre_account_id = pre_inst.account_id or pre_idata.get("account_id")
        pre_lease_id = pre_idata.get("lease_id")
        pre_lease = (
            session.scalar(select(GpuLeaseRow).where(GpuLeaseRow.id == pre_lease_id, GpuLeaseRow.owner_id == owner_id))
            if pre_lease_id
            else None
        )
        pre_ldata = copy.deepcopy(pre_lease.data or {}) if pre_lease is not None else {}

    if target_pool == pre_pool:
        raise APIError("INVALID_TRANSFER_TARGET", "目标域不能与实例当前所属域相同", 422)

    # Check native config & Drive authorization outside write locks when target == native.
    if target_pool == POOL_NATIVE:
        cfg_err = _check_native_config_and_drive(repo, owner_id, pre_account_id, pre_provider)
        if cfg_err is not None:
            raise APIError(cfg_err[0], cfg_err[1], cfg_err[2])

    # Lock-free live probe when transferring out of a ready native ComfyUI instance (Spec §4.2(1)).
    pre_native_last_write = pre_ldata.get("native_last_write")
    from_native_ready = pre_pool == POOL_NATIVE and not pre_idata.get("native_bootstrap_failed")
    if from_native_ready:
        now_pre = float(clock())
        if (
            pre_idata.get("native_draining")
            or pre_ldata.get("native_draining")
            or native.restart_active(pre_ldata)
            or bool(native.active_writes(pre_ldata, now_pre))
        ):
            raise APIError("INSTANCE_TRANSFER_BUSY", "请等待当前任务/同步完成", 409)
        if native_busy_probe is not None:
            status, _reason = native_busy_probe()
        elif pre_provider == "colab" or native.busy is not native._default_busy:
            status, _reason = native.busy_state_for(repo, owner_id, fresh=True)
        else:
            status, _reason = "idle", None
        if status != "idle":
            raise APIError("INSTANCE_TRANSFER_BUSY", "请等待当前任务/同步完成", 409)

    # 2) Atomic lock hierarchy: dispatch_lock -> admission_lock -> DB BEGIN IMMEDIATE.
    with repo.store.dispatch_lock() as dispatch_ok:
        if not dispatch_ok:
            raise APIError("SCHEDULER_BUSY", "调度正在同步，请稍后使用相同幂等键重试", 503, True)
        with instance_service._admission(), instance_service._tx() as session:
            operation, prior, done = instance_service._operation(
                session, owner_id, instance_id, "transfer", key, canonical_body
            )
            if done:
                return prior

            now = float(clock())
            inst = instance_service._find(session, "instances", instance_id, owner_id)
            if inst is None:
                raise APIError("NOT_FOUND", "实例不存在或不属于当前用户", 404)
            if expected_version is not None and int(inst.version) != int(expected_version):
                raise APIError("INSTANCE_CONFLICT", "实例状态已变更，请刷新后重试", 409, True)

            idata = instance_service._data(inst)
            from_pool = instance_service._instance_pool(session, inst)
            if target_pool == from_pool:
                raise APIError("INVALID_TRANSFER_TARGET", "目标域不能与实例当前所属域相同", 422)

            lease_id = idata.get("lease_id")
            lease = instance_service._find(session, "leases", lease_id, owner_id) if lease_id else None
            ldata = instance_service._data(lease) if lease is not None else {}

            ok_base, err_code, err_msg = _check_base_transferability_in_tx(
                session, repo, owner_id, inst, lease, now
            )
            if not ok_base:
                raise APIError(err_code or "INSTANCE_TRANSFER_UNAVAILABLE", err_msg or "实例当前不可转移", 409)

            # Native TOCTOU check: reject if any write started after our lock-free probe.
            if from_pool == POOL_NATIVE and not idata.get("native_bootstrap_failed"):
                if ldata.get("native_last_write") != pre_native_last_write:
                    raise APIError("INSTANCE_TRANSFER_BUSY", "请等待当前任务/同步完成", 409)

            # Native single-slot atomic reservation (blocks concurrent double transfers to native!).
            if target_pool == POOL_NATIVE:
                blocker = _native_slot_blocker_in_tx(session, repo, owner_id, exclude_instance_id=instance_id)
                if blocker is not None:
                    raise APIError(blocker[0], blocker[1], 409)

            transfer_id = f"tr-{uuid.uuid4().hex[:12]}"
            transfer_job_id = str(uuid.uuid4())
            transfer_exec_id = str(uuid.uuid4())
            hard_timeout = transfer_hard_timeout_seconds()
            transfer_deadline = now + hard_timeout
            provider = str(idata.get("provider") or "mock")
            account_id = inst.account_id or idata.get("account_id")
            baseline_digest = idata.get("vm_baseline_digest") or ldata.get("vm_baseline_digest")
            from_native_failed = bool(idata.get("native_bootstrap_failed"))

            transfer_state = {
                "transfer_id": transfer_id,
                "from_pool": from_pool,
                "to_pool": target_pool,
                "phase": "cleaning",
                "transfer_job_id": transfer_job_id,
                "execution_id": transfer_exec_id,
                "started_at": now,
                "error_code": None,
                "error_message": None,
            }

            # Transition instance -> transferring (CAS on version).
            idata.update(
                current_job_id=transfer_job_id,
                idle_deadline_at=None,
                last_seen_at=now,
                transfer_state=transfer_state,
            )
            if from_pool == POOL_NATIVE:
                idata.update(native_ready=False, native_draining=True)
            instance_service._set(inst, idata, state="transferring", expected_version=inst.version, session=session)
            post_instance_version = int(inst.version)

            # Transition lease -> active owned by transfer job (retarget handle & fence native proxy immediately).
            if lease is not None:
                handle = ldata.get("handle")
                if isinstance(handle, dict):
                    ldata["handle"] = {
                        **handle,
                        "job_id": transfer_job_id,
                        "owner_job_id": transfer_job_id,
                    }
                prev_deadline = float(ldata.get("deadline_epoch") or 0.0)
                ldata.update(
                    state="active",
                    owner_job_id=transfer_job_id,
                    current_job_id=transfer_job_id,
                    idle_deadline_at=None,
                    last_active_at=now,
                    last_seen=now,
                    deadline_epoch=max(prev_deadline, transfer_deadline),
                )
                if from_pool == POOL_NATIVE:
                    ldata.update(native_ready=False, native_draining=True)
                instance_service._set(lease, ldata, state="active", expected_version=lease.version, session=session)

            # Create durable domain_transfer system job in state="loading".
            job_payload = {
                "domain_transfer": True,
                "transfer_id": transfer_id,
                "from_pool": from_pool,
                "to_pool": target_pool,
                "from_native_bootstrap_failed": from_native_failed,
                "target_options": normalized_options,
                "expected_baseline_digest": baseline_digest,
                "expected_instance_version": post_instance_version,
                "instance_id": instance_id,
                "lease_id": lease.id if lease is not None else None,
                "account_id": account_id,
                "provider": provider,
                "instance_pool": from_pool,
                "model_id": "domain-transfer",
                "workflow_key": "system/domain-transfer",
                "workflow_version": 1,
                "execution_id": transfer_exec_id,
                "state": "loading",
                "stage": "transfer_cleaning",
                "started_epoch": now,
                "started_at": now,
                "transfer_started_at": now,
                "transfer_hard_deadline_at": transfer_deadline,
                "deadline_epoch": transfer_deadline,
                "max_runtime_minutes": max(1, int(round(hard_timeout / 60.0))),
                "cancel_requested": False,
                "output_asset_ids": [],
                "params": {},
                "execution_policy": {"provider": provider, "account_id": account_id},
                "snapshot": {
                    "domain_transfer": True,
                    "from_pool": from_pool,
                    "to_pool": target_pool,
                },
            }
            now_dt = utcnow()
            session.add(
                GpuJobRow(
                    id=transfer_job_id,
                    owner_id=owner_id,
                    project_id=inst.project_id,
                    account_id=account_id,
                    instance_id=instance_id,
                    state="loading",
                    data=job_payload,
                    version=0,
                    idempotency_key=f"domain-transfer:{transfer_id}",
                    body_hash=body_hash,
                    created_at=now_dt,
                    updated_at=now_dt,
                )
            )
            msg = f"开始将 GPU 实例从{_POOL_LABELS.get(from_pool, from_pool)}转移至{_POOL_LABELS.get(target_pool, target_pool)}"
            session.add(
                GpuEventRow(
                    id=str(uuid.uuid4()),
                    owner_id=owner_id,
                    project_id=inst.project_id,
                    state="transferring",
                    version=0,
                    data={
                        "instance_id": instance_id,
                        "job_id": transfer_job_id,
                        "state": "transferring",
                        "message": msg,
                        "transfer_id": transfer_id,
                        "from_pool": from_pool,
                        "to_pool": target_pool,
                    },
                    created_at=now_dt,
                    updated_at=now_dt,
                )
            )
            public_inst = instance_service.public(instance_service._dict(inst))
            result = {
                "instance": public_inst,
                "transfer_job_id": transfer_job_id,
                "transfer_id": transfer_id,
            }
            instance_service._finish_operation(operation, result, session)
            return result


def _quarantine_instance(
    scheduler,
    job: dict[str, Any],
    *,
    reason: str,
    error_code: str,
    message: str,
    is_transfer_job: bool = True,
) -> dict[str, Any]:
    """Place instance + lease into protective quarantine without deleting the VM."""
    owner_id = job["owner_id"]
    instance_id = job.get("instance_id")
    lease_id = job.get("lease_id")
    now = float(scheduler.clock())
    svc = scheduler.instance_service

    with scheduler.db_store.dispatch_lock() as d_ok:
        if not d_ok:
            return job
        with svc._admission(), svc._tx() as session:
            inst = svc._find(session, "instances", instance_id, owner_id) if instance_id else None
            lease = svc._find(session, "leases", lease_id, owner_id) if lease_id else None
            if inst is not None and inst.state != "deleted":
                idata = svc._data(inst)
                ts = dict(idata.get("transfer_state") or {})
                if ts:
                    ts.update(phase="aborted_unsynced", error_code=error_code, error_message=message)
                idata.update(
                    current_job_id=None,
                    native_ready=False,
                    native_draining=True,
                    quarantine_reason=reason,
                    last_seen_at=now,
                    **({"transfer_state": ts} if ts else {}),
                )
                svc._set(inst, idata, state="quarantined", expected_version=inst.version, session=session)
            if lease is not None and lease.state != "deleted":
                ldata = svc._data(lease)
                ldata.update(
                    state="quarantined",
                    current_job_id=None,
                    native_ready=False,
                    native_draining=True,
                    quarantine_reason=reason,
                    last_error=message,
                    last_seen=now,
                )
                svc._set(lease, ldata, state="quarantined", expected_version=lease.version, session=session)
            if inst is not None:
                svc._emit(
                    session,
                    owner_id,
                    inst.project_id,
                    inst.id,
                    job["id"],
                    "quarantined",
                    f"【数据保护隔离】{message}（未自动销毁 GPU，请确认同步完成后再手动停止；注意持续计费风险）",
                    quarantine_reason=reason,
                )

    patch: dict[str, Any] = {
        "error_code": error_code,
        "error_message": message,
        "runtime_deleted": False,
        "stopped_at": now,
    }
    if not is_transfer_job:
        patch["native_failure_cleanup_state"] = "unsynced"
        patch["native_failure_cleaned"] = False
    return scheduler._set_state(job, "failed", message, **patch)


def _fail_transfer_to_releasing(
    scheduler,
    job: dict[str, Any],
    *,
    phase: str,
    error_code: str,
    message: str,
    final_state: str = "failed",
) -> dict[str, Any]:
    """Fail-closed: abort transfer and route VM through normal confirmed deletion."""
    owner_id = job["owner_id"]
    instance_id = job.get("instance_id")
    now = float(scheduler.clock())
    svc = scheduler.instance_service

    with scheduler.db_store.dispatch_lock() as d_ok:
        if not d_ok:
            return job
        with svc._admission(), svc._tx() as session:
            inst = svc._find(session, "instances", instance_id, owner_id) if instance_id else None
            if inst is not None and inst.state != "deleted":
                idata = svc._data(inst)
                ts = dict(idata.get("transfer_state") or {})
                if ts:
                    ts.update(phase=phase, error_code=error_code, error_message=message)
                idata.update(
                    manual_stop_requested=True,
                    retention_mode="save",
                    idle_deadline_at=None,
                    last_seen_at=now,
                    **({"transfer_state": ts} if ts else {}),
                )
                svc._set(inst, idata, state="releasing", expected_version=inst.version, session=session)
                svc._emit(session, owner_id, inst.project_id, inst.id, job["id"], "releasing", message)

    return scheduler._set_state(
        job,
        "cleaning_up",
        message,
        requested_outcome=final_state,
        cleanup_final_state=final_state,
        error_code=error_code if final_state == "failed" else None,
        error_message=message if final_state == "failed" else None,
    )


def _commit_clean_transfer(
    scheduler,
    job: dict[str, Any],
    teardown_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically commit a verified clean domain transfer in a single short DB transaction."""
    owner_id = job["owner_id"]
    instance_id = job["instance_id"]
    lease_id = job.get("lease_id")
    from_pool = job["from_pool"]
    to_pool = job["to_pool"]
    transfer_id = job["transfer_id"]
    opts = dict(job.get("target_options") or {})
    provider = str(job.get("provider") or "mock")
    account_id = job.get("account_id")
    svc = scheduler.instance_service
    repo = scheduler.store

    # Resolve native config & Drive account outside the write transaction when targeting native.
    native_commit = "0" * 40
    drive_account_id = account_id
    if to_pool == POOL_NATIVE:
        if provider == "colab":
            reader = None if native.secret_file is native._default_secret_file else native.secret_file
            ncfg = native_config.resolve(repo, owner_id, use_cache=False, file_reader=reader)
            native_commit = ncfg.native_comfy_commit
        try:
            drive_account_id = AccountService(repo, owner_id).resolve_drive_account_id(account_id) or account_id
        except Exception:
            drive_account_id = account_id

    with scheduler.db_store.dispatch_lock() as d_ok:
        if not d_ok:
            return job
        with svc._admission(), ccu.transaction(repo, owner_id) as session:
            now = float(scheduler.clock())
            inst = svc._find(session, "instances", instance_id, owner_id)
            lease = svc._find(session, "leases", lease_id, owner_id) if lease_id else None
            job_row = session.get(GpuJobRow, job["id"])
            if inst is None or job_row is None or inst.state == "deleted":
                return scheduler._get("jobs", job["id"], owner_id) or job

            idata = svc._data(inst)
            ldata = svc._data(lease) if lease is not None else {}
            jdata = copy.deepcopy(job_row.data or {})
            ts = dict(idata.get("transfer_state") or {})

            # CAS token check: transfer identity must still match.
            if inst.state != "transferring" or ts.get("transfer_id") != transfer_id:
                jdata.update(state="failed", error_code="transfer_state_mismatch", error_message="实例转移状态已变更")
                job_row.state = "failed"
                job_row.data = jdata
                job_row.version += 1
                job_row.updated_at = utcnow()
                return repo._row(job_row)

            # Check concurrent manual stop / hard balance floor fuse BEFORE switching domain!
            if (
                idata.get("manual_stop_requested")
                or idata.get("force_stop")
                or ldata.get("force_stop")
                or jdata.get("cancel_requested")
            ):
                ts.update(phase="cancelled_by_stop", error_code=None, error_message="收到停止请求，已中止域转移")
                idata.update(
                    manual_stop_requested=True,
                    retention_mode="save",
                    idle_deadline_at=None,
                    transfer_state=ts,
                    last_seen_at=now,
                )
                svc._set(inst, idata, state="releasing", expected_version=inst.version, session=session)
                svc._emit(
                    session,
                    owner_id,
                    inst.project_id,
                    inst.id,
                    job["id"],
                    "releasing",
                    "域转移期间收到停止请求，已中止切换并开始释放实例",
                )
                jdata.update(
                    state="cleaning_up",
                    requested_outcome="cancelled",
                    cleanup_final_state="cancelled",
                    **ccu.execution_finished(jdata, now),
                )
                job_row.state = "cleaning_up"
                job_row.data = jdata
                job_row.version += 1
                job_row.updated_at = utcnow()
                return repo._row(job_row)

            expected_ver = jdata.get("expected_instance_version")
            if expected_ver is not None and int(inst.version) != int(expected_ver):
                ts.update(phase="teardown_broken", error_code="transfer_version_conflict", error_message="实例版本发生并发变更")
                idata.update(manual_stop_requested=True, retention_mode="save", idle_deadline_at=None, transfer_state=ts, last_seen_at=now)
                svc._set(inst, idata, state="releasing", expected_version=inst.version, session=session)
                jdata.update(
                    state="cleaning_up",
                    requested_outcome="failed",
                    cleanup_final_state="failed",
                    error_code="transfer_version_conflict",
                    error_message="实例版本发生并发变更，已中止转移并释放",
                    **ccu.execution_finished(jdata, now),
                )
                job_row.state = "cleaning_up"
                job_row.data = jdata
                job_row.version += 1
                job_row.updated_at = utcnow()
                return repo._row(job_row)

            # Belt-and-suspenders native single-slot re-check right before atomic commit.
            if to_pool == POOL_NATIVE:
                blocker = _native_slot_blocker_in_tx(session, repo, owner_id, exclude_instance_id=instance_id)
                if blocker is not None:
                    ts.update(phase="teardown_broken", error_code=blocker[0], error_message=blocker[1])
                    idata.update(manual_stop_requested=True, retention_mode="save", idle_deadline_at=None, transfer_state=ts, last_seen_at=now)
                    svc._set(inst, idata, state="releasing", expected_version=inst.version, session=session)
                    jdata.update(
                        state="cleaning_up",
                        requested_outcome="failed",
                        cleanup_final_state="failed",
                        error_code=blocker[0],
                        error_message=blocker[1],
                        **ccu.execution_finished(jdata, now),
                    )
                    job_row.state = "cleaning_up"
                    job_row.data = jdata
                    job_row.version += 1
                    job_row.updated_at = utcnow()
                    return repo._row(job_row)

            # Common post-teardown cleanup & domain transition metadata.
            domain_gen = int(idata.get("domain_generation") or 0) + 1
            history = list(idata.get("transfer_history") or [])
            history.append(
                {
                    "transfer_id": transfer_id,
                    "from_pool": from_pool,
                    "to_pool": to_pool,
                    "started_at": jdata.get("transfer_started_at"),
                    "completed_at": now,
                    "transfer_job_id": job["id"],
                    "domain_generation": domain_gen,
                }
            )
            idata.update(
                instance_pool=to_pool,
                domain_generation=domain_gen,
                transfer_history=history[-20:],
                creation_runtime=None,
                native_comfy=(to_pool == POOL_NATIVE),
                native_ready=False,
                native_draining=False,
                native_bootstrap_failed=False,
                native_bootstrap_error=None,
                ephemeral_standby=(to_pool == POOL_EPHEMERAL),
                quarantine_reason=None,
                last_seen_at=now,
            )
            # Clear stale lease fields from previous domain.
            if lease is not None:
                for old_key in _OLD_NATIVE_LEASE_FIELDS:
                    ldata.pop(old_key, None)
                ldata.pop("restored_signature", None)
                ldata.pop("quarantine_reason", None)
                ldata.pop("ephemeral_standby", None)
                ldata.update(
                    instance_pool=to_pool,
                    domain_generation=domain_gen,
                    last_seen=now,
                )

            # Update authoritative CU ledger in the exact same transaction (Spec §6.2).
            ccu.record_domain_transfer_in_tx(
                session, owner_id, instance_id, from_pool=from_pool, to_pool=to_pool, at=now
            )

            # Branch A: -> creation
            if to_pool == POOL_CREATION:
                minutes = max(1.0, min(30.0, float(opts.get("retain_idle_minutes") or DEFAULT_CREATION_IDLE_MINUTES)))
                deadline = now + minutes * 60.0
                ts.update(phase="succeeded", completed_at=now, error_code=None, error_message=None)
                idata.update(
                    current_job_id=None,
                    hard_deadline_at=None,
                    retention_mode="retain",
                    retain_idle_minutes=minutes,
                    idle_deadline_at=deadline,
                    transfer_state=ts,
                )
                svc._set(inst, idata, state="retained_idle", expected_version=inst.version, session=session)
                if lease is not None:
                    ldata.update(
                        state="idle",
                        current_job_id=None,
                        generation_timeouts=True,
                        hard_deadline_at=None,
                        idle_deadline_at=deadline,
                        deadline_epoch=deadline,
                        last_active_at=now,
                    )
                    svc._set(lease, ldata, state="idle", expected_version=lease.version, session=session)
                svc._emit(
                    session,
                    owner_id,
                    inst.project_id,
                    inst.id,
                    job["id"],
                    "retained_idle",
                    f"域转移完成：已切入创作池并保留 {int(round(minutes))} 分钟空闲待命",
                    transfer_id=transfer_id,
                    from_pool=from_pool,
                    to_pool=to_pool,
                )
                jdata.update(
                    state="completed",
                    runtime_deleted=False,
                    stopped_at=now,
                    teardown_report=teardown_report,
                    **ccu.execution_finished(jdata, now),
                )
                job_row.state = "completed"
                job_row.data = jdata
                job_row.version += 1
                job_row.updated_at = utcnow()
                return repo._row(job_row)

            # Branch B: -> ephemeral (10-minute smoke standby)
            if to_pool == POOL_EPHEMERAL:
                minutes = float(EPHEMERAL_STANDBY_MINUTES)
                deadline = now + minutes * 60.0
                ts.update(phase="succeeded", completed_at=now, error_code=None, error_message=None)
                idata.update(
                    current_job_id=None,
                    hard_deadline_at=None,
                    retention_mode="retain",
                    retain_idle_minutes=minutes,
                    idle_deadline_at=deadline,
                    ephemeral_standby=True,
                    transfer_state=ts,
                )
                svc._set(inst, idata, state="retained_idle", expected_version=inst.version, session=session)
                if lease is not None:
                    ldata.update(
                        state="idle",
                        current_job_id=None,
                        generation_timeouts=True,
                        hard_deadline_at=None,
                        idle_deadline_at=deadline,
                        deadline_epoch=deadline,
                        ephemeral_standby=True,
                        last_active_at=now,
                    )
                    svc._set(lease, ldata, state="idle", expected_version=lease.version, session=session)
                svc._emit(
                    session,
                    owner_id,
                    inst.project_id,
                    inst.id,
                    job["id"],
                    "retained_idle",
                    "域转移完成：已切入冒烟域，保留 10 分钟等待工作流真实冒烟",
                    transfer_id=transfer_id,
                    from_pool=from_pool,
                    to_pool=to_pool,
                )
                jdata.update(
                    state="completed",
                    runtime_deleted=False,
                    stopped_at=now,
                    teardown_report=teardown_report,
                    **ccu.execution_finished(jdata, now),
                )
                job_row.state = "completed"
                job_row.data = jdata
                job_row.version += 1
                job_row.updated_at = utcnow()
                return repo._row(job_row)

            # Branch C: -> native (single-TX atomic switch + create transferred_native_bootstrap job!)
            storage = native.model_storage(opts)
            idle_min = int(opts.get("retain_idle_minutes") or settings.native_comfy_idle_minutes)
            max_rt = int(opts.get("max_runtime_minutes") or settings.native_comfy_max_runtime_minutes)
            hard_deadline = now + max_rt * 60.0
            norm_spec = _instance_spec_key(idata)
            inst_spec = idata.get("runtime_spec_reported") or idata.get("runtime_spec_requested")
            req_spec = {"key": norm_spec} if norm_spec else (copy.deepcopy(inst_spec) if isinstance(inst_spec, Mapping) else {"provider": provider})

            native_project_id = "native-comfy-" + owner_id
            if session.get(ProjectRow, native_project_id) is None:
                now_dt = utcnow()
                session.add(
                    ProjectRow(
                        id=native_project_id,
                        owner_id=owner_id,
                        project_id=None,
                        state="active",
                        data={"title": "原生 ComfyUI", "system": True},
                        version=0,
                        created_at=now_dt,
                        updated_at=now_dt,
                    )
                )
                session.flush()

            boot_job_id = str(uuid.uuid4())
            boot_policy = {
                "provider": provider,
                "lifecycle": "retain",
                "account_id": account_id,
                "retain_idle_minutes": idle_min,
                "max_runtime_minutes": max_rt,
                **storage,
            }
            from .rates import minute_rate

            rate = minute_rate(req_spec, fallback_rate=scheduler.config.unit_rate)
            boot_payload = {
                "transferred_native_bootstrap": True,
                "transfer_id": transfer_id,
                "selection_reason": "domain_transfer",
                "instance_pool": POOL_NATIVE,
                "project_id": native_project_id,
                "provider": provider,
                "account_id": account_id,
                "requested_account_id": account_id,
                "drive_account_id": drive_account_id,
                "native_drive_account_id": drive_account_id,
                "instance_id": instance_id,
                "lease_id": lease.id if lease is not None else None,
                "auto_select_account": False,
                "requested_instance_id": instance_id,
                "requested_spec": req_spec,
                "initial_requested_spec": req_spec,
                "lifecycle": "retain",
                "execution_policy": boot_policy,
                "model_id": "native-comfy",
                "workflow_key": "system/native-comfy",
                "workflow_version": 1,
                "max_runtime_minutes": max_rt,
                "unit_rate": rate,
                "hard_deadline_at": hard_deadline,
                "deadline_epoch": hard_deadline,
                "native_session_deadline_epoch": hard_deadline,
                "estimated_units": None,
                "started_epoch": now,
                "started_at": now,
                "state": "allocating",
                "cancel_requested": False,
                "output_asset_ids": [],
                "params": {},
                "snapshot": {
                    "native_comfy": True,
                    "native_plugin_generation": opts.get("plugin_generation"),
                    "bound_graph": {},
                    "manifest": {
                        "reviewed": True,
                        "state": "published",
                        "comfy_commit": native_commit,
                        "compute_spec": norm_spec or (req_spec.get("key") if isinstance(req_spec, dict) else {}),
                    },
                },
            }
            now_dt = utcnow()
            session.add(
                GpuJobRow(
                    id=boot_job_id,
                    owner_id=owner_id,
                    project_id=native_project_id,
                    account_id=account_id,
                    instance_id=instance_id,
                    state="allocating",
                    data=boot_payload,
                    version=0,
                    idempotency_key=f"transfer-native-boot:{transfer_id}",
                    body_hash=digest({"transfer_id": transfer_id, "boot_policy": boot_policy}),
                    created_at=now_dt,
                    updated_at=now_dt,
                )
            )

            ts.update(
                phase="native_bootstrapping",
                native_bootstrap_job_id=boot_job_id,
                error_code=None,
                error_message=None,
            )
            idata.update(
                current_job_id=boot_job_id,
                hard_deadline_at=hard_deadline,
                idle_deadline_at=None,
                retention_mode="retain",
                retain_idle_minutes=idle_min,
                selection_reason="domain_transfer",
                native_comfy=True,
                native_ready=False,
                native_bootstrap_failed=False,
                drive_account_id=drive_account_id,
                transfer_state=ts,
            )
            svc._set(inst, idata, state="allocating", expected_version=inst.version, session=session)

            if lease is not None:
                handle = ldata.get("handle")
                if isinstance(handle, dict):
                    ldata["handle"] = {
                        **handle,
                        "job_id": boot_job_id,
                        "owner_job_id": boot_job_id,
                    }
                ldata.update(
                    state="active",
                    owner_job_id=boot_job_id,
                    current_job_id=boot_job_id,
                    instance_bound=True,
                    generation_timeouts=False,
                    native_comfy=True,
                    native_ready=False,
                    native_draining=False,
                    native_bootstrap_job_id=boot_job_id,
                    native_idle_minutes=idle_min,
                    native_model_storage=storage,
                    drive_account_id=drive_account_id,
                    native_drive_account_id=drive_account_id,
                    hard_deadline_at=hard_deadline,
                    deadline_epoch=hard_deadline,
                    idle_deadline_at=None,
                    last_active_at=now,
                )
                svc._set(lease, ldata, state="active", expected_version=lease.version, session=session)

            svc._emit(
                session,
                owner_id,
                inst.project_id,
                inst.id,
                boot_job_id,
                "allocating",
                "旧域清理校验通过，正在原卡拉起原生 ComfyUI",
                transfer_id=transfer_id,
                from_pool=from_pool,
                to_pool=to_pool,
            )
            jdata.update(
                state="completed",
                runtime_deleted=False,
                stopped_at=now,
                native_bootstrap_job_id=boot_job_id,
                teardown_report=teardown_report,
                **ccu.execution_finished(jdata, now),
            )
            job_row.state = "completed"
            job_row.data = jdata
            job_row.version += 1
            job_row.updated_at = utcnow()
            return repo._row(job_row)


def _handle_teardown_result(
    scheduler,
    job: dict[str, Any],
    lease: dict[str, Any] | None,
    executor: Any,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    state = result.get("state")
    report = result.get("teardown_report") if isinstance(result.get("teardown_report"), dict) else None

    if state == "aborted_unsynced" or (report and report.get("aborted_unsynced")):
        detail = str(
            result.get("error_message")
            or result.get("message")
            or (report or {}).get("reason")
            or "检测到原生 ComfyUI 仍有未完成任务或未上传的 Drive VFS 缓存，已中止清理以保护数据"
        )
        return _quarantine_instance(
            scheduler,
            job,
            reason="aborted_unsynced",
            error_code=str(result.get("error_code") or "native_unsynced_cache"),
            message=detail,
            is_transfer_job=True,
        )

    if state == "completed" and result.get("cleaned") is True:
        # Optional Kernel restart to reclaim Jupyter memory (Spec §4.3 Step 3.6).
        if lease and lease.get("handle") and hasattr(executor, "restart_kernel") and callable(executor.restart_kernel):
            try:
                executor.restart_kernel(scheduler._lease_handle(lease))
            except Exception as exc:  # noqa: BLE001 - best effort after verified clean teardown
                log.warning("transfer kernel restart failed (non-fatal): %s", type(exc).__name__)
        return _commit_clean_transfer(scheduler, job, teardown_report=report)

    err_code = str(result.get("error_code") or "teardown_failed")
    err_msg = str(result.get("error_message") or result.get("message") or "远端环境清理或纯净度校验未通过，已按安全策略释放实例")
    return _fail_transfer_to_releasing(
        scheduler,
        job,
        phase="teardown_broken",
        error_code=err_code,
        message=err_msg,
        final_state="failed",
    )


def advance_transfer_job(scheduler, job: dict[str, Any]) -> dict[str, Any]:
    """Advance a ``domain_transfer`` system job one phase per scheduler tick."""
    owner_id = job["owner_id"]
    instance = scheduler._get("instances", job.get("instance_id"), owner_id) if job.get("instance_id") else None
    lease = scheduler._lease(job)
    now = float(scheduler.clock())
    from_pool = job.get("from_pool") or POOL_CREATION
    from_native_failed = bool(job.get("from_native_bootstrap_failed"))
    hard_deadline = float(job.get("transfer_hard_deadline_at") or job.get("deadline_epoch") or 0.0)

    if instance is None or instance.get("state") == "deleted":
        return scheduler._set_state(
            job,
            "failed",
            "目标实例已不存在或已释放，域转移中止",
            error_code="instance_missing",
            error_message="目标实例已不存在或已释放",
        )

    # Immediate hard stop (e.g., CU balance floor fuse or force_stop).
    if instance.get("force_stop") or (lease and lease.get("force_stop")):
        return _fail_transfer_to_releasing(
            scheduler,
            job,
            phase="cancelled_by_stop",
            error_code="force_stopped",
            message="收到强制停止或余额底线熔断，已中止域转移并释放实例",
            final_state="cancelled",
        )

    # Hard timeout check (default 3 minutes, Spec §4.2(2) rule 4).
    if hard_deadline > 0 and now >= hard_deadline:
        if from_pool == POOL_NATIVE and not from_native_failed:
            return _quarantine_instance(
                scheduler,
                job,
                reason="transfer_timeout_unsynced",
                error_code="transfer_timeout",
                message="原生域转移清理超时且无法确认 Drive VFS 缓存是否已全部上传，已转入保护隔离态避免丢数据",
                is_transfer_job=True,
            )
        return _fail_transfer_to_releasing(
            scheduler,
            job,
            phase="teardown_broken",
            error_code="transfer_timeout",
            message="域转移远端清理超过 3 分钟硬超时，已按安全策略中止并释放实例",
            final_state="failed",
        )

    # If user requested manual stop before non-native teardown even launched, release immediately.
    manual_stop = bool(instance.get("manual_stop_requested") or job.get("cancel_requested"))
    if manual_stop and job.get("state") == "loading" and (from_pool != POOL_NATIVE or from_native_failed):
        return _fail_transfer_to_releasing(
            scheduler,
            job,
            phase="cancelled_by_stop",
            error_code="manual_stop",
            message="转移期间收到手动停止请求，已中止域转移并释放实例",
            final_state="cancelled",
        )

    execution_id = job.get("execution_id")
    if not isinstance(execution_id, str) or not execution_id:
        execution_id = str(uuid.uuid4())
        job = scheduler._update("jobs", job["id"], {"execution_id": execution_id}, owner_id)
    executor = scheduler._executor(job)

    if job.get("state") == "loading":
        teardown_payload = {
            "from_pool": from_pool,
            "to_pool": job.get("to_pool"),
            "native_bootstrap_failed": from_native_failed,
            "expected_baseline_digest": (
                job.get("expected_baseline_digest")
                or instance.get("vm_baseline_digest")
                or (lease or {}).get("vm_baseline_digest")
            ),
            "clean_native_only": False,
        }
        start_fn = getattr(executor, "start_teardown", None)
        if callable(start_fn):
            if not lease or not lease.get("handle"):
                return _fail_transfer_to_releasing(
                    scheduler,
                    job,
                    phase="teardown_broken",
                    error_code="gpu_lease_missing",
                    message="域转移缺少有效远端运行时句柄",
                )
            try:
                start_fn(scheduler._lease_handle(lease), job, execution_id, teardown_payload)
            except MCPTransient:
                # Transient network error: keep state="loading" to retry idempotently with same execution_id.
                return job
            except Exception as exc:  # noqa: BLE001
                err_msg = scheduler._safe_error(exc)
                if from_pool == POOL_NATIVE and not from_native_failed:
                    return _quarantine_instance(
                        scheduler,
                        job,
                        reason="teardown_start_error_unsynced",
                        error_code="teardown_start_failed",
                        message=f"启动原生清理脚本失败且无法确认同步状态（{err_msg}），已保护隔离",
                        is_transfer_job=True,
                    )
                return _fail_transfer_to_releasing(
                    scheduler,
                    job,
                    phase="teardown_broken",
                    error_code="teardown_start_failed",
                    message=f"启动远端清理脚本失败：{err_msg}",
                )
            return scheduler._set_state(
                job,
                "running",
                "正在远端 VM 执行旧域清理与纯净度校验",
                execution_id=execution_id,
                stage="transfer_cleaning",
            )
        # Mock executor fast-path (offline unit tests without MCP Jupyter runtime).
        if job.get("provider") != "mock":
            return _fail_transfer_to_releasing(
                scheduler,
                job,
                phase="teardown_broken",
                error_code="teardown_unsupported",
                message="当前执行器不支持受控环境清理，已按安全策略拒绝转移",
            )
        return _handle_teardown_result(
            scheduler,
            job,
            lease,
            executor,
            {"state": "completed", "cleaned": True, "teardown_report": {"cleaned": True, "mock": True}},
        )

    if job.get("state") == "running":
        if float(job.get("poll_retry_at") or 0) > now:
            return job
        if not lease or not lease.get("handle"):
            return _fail_transfer_to_releasing(
                scheduler,
                job,
                phase="teardown_broken",
                error_code="gpu_lease_missing",
                message="域转移轮询期间租约丢失",
            )
        try:
            poll_job = {**job, "execution_id": execution_id}
            result = executor.poll(scheduler._lease_handle(lease), poll_job)
        except MCPTransient:
            retries = int(job.get("poll_retry_count") or 0) + 1
            return scheduler._update(
                "jobs",
                job["id"],
                {"poll_retry_count": retries, "poll_retry_at": now + min(2 ** min(retries, 4), 15)},
                owner_id,
            )
        except Exception as exc:  # noqa: BLE001
            err_msg = scheduler._safe_error(exc)
            if from_pool == POOL_NATIVE and not from_native_failed:
                return _quarantine_instance(
                    scheduler,
                    job,
                    reason="teardown_poll_error_unsynced",
                    error_code="teardown_poll_failed",
                    message=f"轮询原生清理状态失败且无法确认同步状态（{err_msg}），已保护隔离",
                    is_transfer_job=True,
                )
            return _fail_transfer_to_releasing(
                scheduler,
                job,
                phase="teardown_broken",
                error_code="teardown_poll_failed",
                message=f"轮询远端清理状态失败：{err_msg}",
            )

        if job.get("poll_retry_count") or job.get("poll_retry_at"):
            job = scheduler._update("jobs", job["id"], {"poll_retry_count": 0, "poll_retry_at": 0}, owner_id)

        if result.get("state") == "running":
            stage = result.get("stage")
            if stage in {"transfer_cleaning", "transfer_verifying", "teardown_stop_processes", "teardown_clean_files", "teardown_verify_purity"} and stage != job.get("stage"):
                job = scheduler._update("jobs", job["id"], {"stage": stage}, owner_id)
                if instance and isinstance(instance.get("transfer_state"), dict):
                    phase = "verifying" if stage in {"transfer_verifying", "teardown_verify_purity"} else "cleaning"
                    if instance["transfer_state"].get("phase") != phase:
                        scheduler._update(
                            "instances",
                            instance["id"],
                            {"transfer_state": {**instance["transfer_state"], "phase": phase}},
                            owner_id,
                        )
            return job
        return _handle_teardown_result(scheduler, job, lease, executor, result)

    return _fail_transfer_to_releasing(
        scheduler,
        job,
        phase="teardown_broken",
        error_code="transfer_invalid_state",
        message=f"域转移任务处于意外状态：{job.get('state')}",
    )


def maybe_cleanup_failed_native_bootstrap(
    scheduler,
    job: dict[str, Any],
    lease: dict[str, Any] | None,
    instance: dict[str, Any] | None,
    *,
    observation: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Lightweight residual cleanup + purity check before retaining a failed ->native VM for 10 min.

    Returns ``(status, updated_job)`` where status is one of:
    - ``"skip"``: not eligible for failed-bootstrap standby fallback (caller proceeds normally)
    - ``"pending"``: cleanup still running remotely (caller should return and poll next tick)
    - ``"cleaned"``: verified clean (job marked ``native_failure_cleaned=True``, caller calls ``complete()``)
    - ``"failed"``: cleanup/purity check failed (job marked ``native_failure_cleaned=False``, caller releases)
    - ``"quarantined"``: unsynced data found (instance & lease quarantined, job failed, caller returns)
    """
    if not (job.get("transferred_native_bootstrap") or job.get("native_standby_retry")):
        return "skip", job
    if job.get("native_failure_cleanup_state") in {"done", "failed", "unsynced"}:
        return "skip", job
    if not lease or not lease.get("handle") or lease.get("native_ready"):
        return "skip", job
    if (
        job.get("cancel_requested")
        or job.get("error_code") == RUNTIME_MISSING
        or observation == "missing"
        or lease.get("remote_delete_source") == RUNTIME_MISSING
        or lease.get("force_stop")
        or (instance or {}).get("manual_stop_requested")
        or (instance or {}).get("force_stop")
        or (instance or {}).get("runtime_deleted")
    ):
        return "skip", job
    now = float(scheduler.clock())
    hard = (instance or {}).get("hard_deadline_at") or lease.get("hard_deadline_at")
    if hard is not None and now >= float(hard):
        return "skip", job
    if job.get("provider") == "colab":
        if CuGuardService(scheduler.store, scheduler.clock).state(job["owner_id"])["soft_tripped"]:
            return "skip", job

    owner_id = job["owner_id"]
    exec_id = job.get("native_failure_cleanup_exec_id")
    cleanup_running = bool(job.get("native_failure_cleanup_state") == "running" and exec_id)
    if not isinstance(exec_id, str) or not exec_id:
        exec_id = str(uuid.uuid4())
        job = scheduler._update("jobs", job["id"], {"native_failure_cleanup_exec_id": exec_id}, owner_id)
    executor = scheduler._executor(job)
    start_fn = getattr(executor, "start_teardown", None)

    if not cleanup_running:
        if callable(start_fn):
            payload = {
                "from_pool": POOL_NATIVE,
                "to_pool": POOL_NATIVE,
                "clean_native_only": True,
                "native_bootstrap_failed": False,
                "expected_baseline_digest": (
                    (instance or {}).get("vm_baseline_digest") or lease.get("vm_baseline_digest")
                ),
            }
            try:
                start_fn(scheduler._lease_handle(lease), job, exec_id, payload)
            except MCPTransient:
                return "pending", job
            except Exception:  # noqa: BLE001 - fail closed to releasing
                job = scheduler._update(
                    "jobs",
                    job["id"],
                    {"native_failure_cleaned": False, "native_failure_cleanup_state": "failed"},
                    owner_id,
                )
                return "failed", job
            job = scheduler._update(
                "jobs",
                job["id"],
                {
                    "native_failure_cleanup_exec_id": exec_id,
                    "native_failure_cleanup_started_at": now,
                    "native_failure_cleanup_state": "running",
                },
                owner_id,
            )
            scheduler._event(job, "cleaning_up", "原生启动失败，正在清理失败残留以保留 GPU 10 分钟待命", None)
            return "pending", job

        # Mock executor fast-path.
        if job.get("provider") == "mock":
            cleaned = job.get("mock_native_failure_cleaned", True) is True
            job = scheduler._update(
                "jobs",
                job["id"],
                {
                    "native_failure_cleaned": cleaned,
                    "native_failure_cleanup_state": "done" if cleaned else "failed",
                },
                owner_id,
            )
            return ("cleaned" if cleaned else "failed"), job

        job = scheduler._update(
            "jobs",
            job["id"],
            {"native_failure_cleaned": False, "native_failure_cleanup_state": "failed"},
            owner_id,
        )
        return "failed", job

    # Already started: check hard timeout & poll.
    started_at = float(job.get("native_failure_cleanup_started_at") or now)
    if now - started_at >= transfer_hard_timeout_seconds():
        job = scheduler._update(
            "jobs",
            job["id"],
            {"native_failure_cleaned": False, "native_failure_cleanup_state": "failed"},
            owner_id,
        )
        return "failed", job

    try:
        result = executor.poll(scheduler._lease_handle(lease), {**job, "execution_id": exec_id})
    except MCPTransient:
        return "pending", job
    except Exception:  # noqa: BLE001
        job = scheduler._update(
            "jobs",
            job["id"],
            {"native_failure_cleaned": False, "native_failure_cleanup_state": "failed"},
            owner_id,
        )
        return "failed", job

    state = result.get("state")
    if state == "running":
        return "pending", job
    report = result.get("teardown_report") if isinstance(result.get("teardown_report"), dict) else None
    if state == "aborted_unsynced" or (report and report.get("aborted_unsynced")):
        detail = str(result.get("error_message") or result.get("message") or "清理原生失败残留时发现未上传的 Drive 缓存，已保护隔离")
        job = _quarantine_instance(
            scheduler,
            job,
            reason="aborted_unsynced",
            error_code="native_unsynced_cache",
            message=detail,
            is_transfer_job=False,
        )
        return "quarantined", job
    if state == "completed" and result.get("cleaned") is True:
        if hasattr(executor, "restart_kernel") and callable(executor.restart_kernel):
            try:
                executor.restart_kernel(scheduler._lease_handle(lease))
            except Exception:  # noqa: BLE001
                pass
        job = scheduler._update(
            "jobs",
            job["id"],
            {"native_failure_cleaned": True, "native_failure_cleanup_state": "done"},
            owner_id,
        )
        return "cleaned", job

    job = scheduler._update(
        "jobs",
        job["id"],
        {"native_failure_cleaned": False, "native_failure_cleanup_state": "failed"},
        owner_id,
    )
    return "failed", job
