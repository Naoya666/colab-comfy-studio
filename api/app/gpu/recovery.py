"""L3 cross-VM auto-recovery orchestrator and shared L2/L3 safety policies.

Design baseline: docs/specs/GPU_AUTO_RECOVERY.md.
Key invariants:
1. Physical instances remain immutable; each successor gets a new instance_id,
   lease_id, and independent CU ledger segment linked via recovery_group_id.
2. Reclaim takeover happens atomically inside _close_reclaimed_runtime's DB
   transaction so worker restarts never lose snapshots and Creation API never
   prematurely reports execution_error.
3. Single-lock short transactions: eligibility check, plan transition, and job
   admission run under dispatch_lock -> admission_lock with zero external I/O.
4. No double-GPU allocation: when a successor assign POST leaves remote state
   unknown, wait until the active_lease_id completes 60s nbh reconciliation and
   is confirmed deleted before scheduling the next retry attempt.
5. Hard deadline is inherited as an exact absolute timestamp; idle deadline is
   strictly inherited without extension when reclaimed while idle.
6. Original account_id and GPU spec are strictly preserved (never downgrade,
   silently upgrade, or switch accounts).
"""

from __future__ import annotations

import copy
import logging
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Mapping

from sqlalchemy import select

from app.common import APIError
from app.config import settings
from app.db import SettingRow, utcnow

from .colab import RUNTIME_MISSING
from .compute import GpuConfig
from .cu_guard import CuGuardService
from .instances import POOL_CREATION, POOL_EPHEMERAL, POOL_NATIVE, instance_pool_of
from .manifest import digest
from .repository import (
    GpuAccountRow,
    GpuBatchRow,
    GpuEventRow,
    GpuExecutionRow,
    GpuInstanceRow,
    GpuJobRow,
    GpuLeaseRow,
    GpuWorkflowRow,
)

log = logging.getLogger("studio.recovery")

PLAN_KEY_PREFIX = "recovery-plan:"
ACTIVE_PLAN_STATES = frozenset({"scheduled", "dispatching", "cancelling", "restoring_env"})
TERMINAL_PLAN_STATES = frozenset({"recovered", "exhausted", "aborted_fatal", "aborted_vps_outage", "cancelled"})
TERMINAL_JOB_STATES = frozenset({"completed", "failed", "cancelled"})
ACTIVE_CREATION_TAKEOVER_STATES = frozenset({"preparing", "loading", "running", "uploading"})
INACTIVE_LEASE_STATES = frozenset({"deleted"})


@contextmanager
def _wait_dispatch_lock(store, timeout: float = 0.5, step: float = 0.02):
    """Retry non-blocking dispatch_lock briefly so critical state transitions don't silently drop."""
    deadline = time.monotonic() + max(0.0, float(timeout))
    while True:
        with store.dispatch_lock() as acquired:
            if acquired:
                yield True
                return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            yield False
            return
        time.sleep(min(step, remaining))

NON_DETERMINISTIC_L2_ERRORS = frozenset({"gpu_worker_exited", "cuda_device_lost"})
_L2_DETERMINISTIC_MARKERS = (
    "outofmemoryerror",
    "cuda out of memory",
    "hash mismatch",
    "sha-256",
    "checksum",
    "validation",
    "auth",
    "gpu_timeout",
    "exceeded its runtime limit",
)

FATAL_ERROR_CODES = frozenset({
    "colab_auth_failed",
    "colab_quota_exceeded",
    "CU_BALANCE_FLOOR_REACHED",
    "CU_DAILY_LIMIT_REACHED",
    "gpu_spec_invalid",
    "GPU_SPEC_BELOW_WORKFLOW_DEFAULT",
    "NATIVE_CONFIG",
    "NATIVE_DISABLED",
    "WORKFLOW_UNPUBLISHED",
    "ACCOUNT_UNAVAILABLE",
    "ACCOUNT_NOT_FOUND",
    "GOOGLE_NOT_CONNECTED",
    "gpu_spec_mismatch",
    "MODEL_DEPENDENCIES_NOT_READY",
    "MODEL_UNAVAILABLE",
    "REVIEW_REQUIRED",
    "NATIVE_INPUT",
})
FATAL_OUTCOMES = frozenset({
    "DENYLISTED",
    "BANNED",
    "RESTRICTED",
    "QUOTA_EXCEEDED_USAGE_TIME",
    "TOO_MANY",
})
FATAL_MESSAGE_MARKERS = (
    "custom node load failed",
    "outofmemoryerror",
    "cuda out of memory",
    "sha-256",
    "checksum mismatch",
    "hash mismatch",
    "protected pytorch",
    "pytorch/cuda",
    "denylisted",
    "quota_exceeded_usage_time",
)
LOCAL_LOCK_ERROR_CODES = frozenset({"NATIVE_WORKSPACE_LOCKED", "SCHEDULER_BUSY"})
RATE_LIMITED_ERROR_CODES = frozenset({"colab_rate_limited"})
RETRYABLE_ERROR_CODES = frozenset({
    "gpu_capacity_unavailable",
    "colab_assign_timeout",
    "colab_network_error",
    "colab_upstream_unavailable",
    "colab_rate_limited",
    "gpu_provider_unavailable",
    "gpu_allocation_failed",
    "gpu_create_failed",
    "gpu_allocation_aborted",
    "gpu_runtime_inspect_failed",
    "gpu_runtime_missing",
    "colab_invalid_response",
})

_PROCESS_STARTED_AT = time.time()


def _read_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _read_int_env(name: str, default: int, *, minimum: int, maximum: int | None = None) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        val = int(default)
    else:
        try:
            val = int(raw.strip())
        except ValueError:
            val = int(default)
    if val < minimum:
        val = minimum
    if maximum is not None and val > maximum:
        val = maximum
    return val


@dataclass(frozen=True)
class RecoveryConfig:
    enabled: bool = False
    max_rounds: int = 3
    min_remaining_minutes: int = 15
    max_job_retries: int = 1
    backoff_base_seconds: int = 30
    round_timeout_seconds: int = 600
    startup_grace_seconds: int = 30
    max_vps_outage_minutes: int = 18
    active_keepalive: bool = True
    in_vm_self_heal: bool = True

    @classmethod
    def from_env(cls) -> "RecoveryConfig":
        return cls(
            enabled=_read_bool_env("GPU_AUTO_RECOVERY_ENABLED", settings.gpu_auto_recovery_enabled),
            max_rounds=_read_int_env(
                "GPU_AUTO_RECOVERY_MAX_ROUNDS", settings.gpu_auto_recovery_max_rounds, minimum=1, maximum=10
            ),
            min_remaining_minutes=_read_int_env(
                "GPU_AUTO_RECOVERY_MIN_REMAINING_MINUTES", settings.gpu_auto_recovery_min_remaining_minutes, minimum=1
            ),
            max_job_retries=_read_int_env(
                "GPU_AUTO_RECOVERY_MAX_JOB_RETRIES", settings.gpu_auto_recovery_max_job_retries, minimum=0, maximum=3
            ),
            backoff_base_seconds=_read_int_env(
                "GPU_AUTO_RECOVERY_BACKOFF_BASE_SECONDS",
                settings.gpu_auto_recovery_backoff_base_seconds,
                minimum=10,
                maximum=300,
            ),
            round_timeout_seconds=_read_int_env(
                "GPU_AUTO_RECOVERY_ROUND_TIMEOUT_SECONDS",
                settings.gpu_auto_recovery_round_timeout_seconds,
                minimum=60,
                maximum=3600,
            ),
            startup_grace_seconds=_read_int_env(
                "GPU_AUTO_RECOVERY_STARTUP_GRACE_SECONDS",
                settings.gpu_auto_recovery_startup_grace_seconds,
                minimum=0,
                maximum=300,
            ),
            max_vps_outage_minutes=_read_int_env(
                "GPU_AUTO_RECOVERY_MAX_VPS_OUTAGE_MINUTES",
                settings.gpu_auto_recovery_max_vps_outage_minutes,
                minimum=5,
                maximum=60,
            ),
            active_keepalive=_read_bool_env("GPU_ACTIVE_KEEPALIVE", settings.gpu_active_keepalive),
            in_vm_self_heal=_read_bool_env("GPU_IN_VM_SELF_HEAL", settings.gpu_in_vm_self_heal),
        )

    def to_public(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "max_rounds": self.max_rounds,
            "min_remaining_minutes": self.min_remaining_minutes,
            "max_job_retries": self.max_job_retries,
            "backoff_base_seconds": self.backoff_base_seconds,
            "round_timeout_seconds": self.round_timeout_seconds,
            "startup_grace_seconds": self.startup_grace_seconds,
            "max_vps_outage_minutes": self.max_vps_outage_minutes,
            "active_keepalive": self.active_keepalive,
            "in_vm_self_heal": self.in_vm_self_heal,
        }


def plan_key(recovery_group_id: str) -> str:
    return f"{PLAN_KEY_PREFIX}{recovery_group_id}"


def make_group_id(instance_id: str) -> str:
    return f"rec-{uuid.uuid5(uuid.NAMESPACE_URL, f'gpu-recovery-group:{instance_id}').hex[:12]}"


def local_models_key(owner_id: str) -> str:
    return "native-local-models-" + digest(owner_id)[:32]


def is_l2_retryable_failure(error_code: str | None, error_message: str | None = None) -> bool:
    code = str(error_code or "").strip()
    if code not in NON_DETERMINISTIC_L2_ERRORS:
        return False
    msg = str(error_message or "").lower()
    if any(marker in msg for marker in _L2_DETERMINISTIC_MARKERS):
        return False
    return True


def classify_failure(
    error_code: str | None,
    error_message: str | None = None,
    error_details: Mapping[str, Any] | None = None,
) -> str:
    """Classify a successor startup/allocation failure per §5.5."""
    code = str(error_code or "").strip()
    msg = str(error_message or "").lower()
    details = error_details if isinstance(error_details, Mapping) else {}
    outcome = str(details.get("outcome") or "").strip().upper()
    http_status = details.get("http_status")

    if code in LOCAL_LOCK_ERROR_CODES:
        return "local_lock"
    if code in FATAL_ERROR_CODES:
        return "fatal"
    if outcome and (outcome in FATAL_OUTCOMES or any(k in outcome for k in ("DENYLIST", "BANNED", "RESTRICTED", "QUOTA_EXCEEDED", "TOO_MANY"))):
        return "fatal"
    if http_status in {401, 403, 412}:
        return "fatal"
    if any(marker in msg for marker in FATAL_MESSAGE_MARKERS):
        return "fatal"
    if code in RATE_LIMITED_ERROR_CODES or http_status == 429:
        return "rate_limited"
    if code in RETRYABLE_ERROR_CODES or http_status in {500, 502, 503, 504}:
        return "retryable"
    if details.get("category") in {"capacity", "timeout", "network", "upstream"}:
        return "retryable"
    if "no_runtimes" in msg or "quota_denied_requested_variants" in msg:
        return "retryable"
    return "fatal"


def compute_inherited_deadlines(
    *,
    hard_deadline_at: float | None,
    orig_idle_deadline_at: float | None,
    retain_idle_minutes: float | None,
    was_busy: bool,
    ready_at: float,
) -> tuple[float | None, float | None]:
    """Compute exact inherited (hard_deadline_at, idle_deadline_at) per §5.6."""
    hard = float(hard_deadline_at) if hard_deadline_at is not None else None
    if was_busy:
        minutes = max(1.0, float(retain_idle_minutes or 15.0))
        idle = ready_at + 60.0 * minutes
        if hard is not None:
            idle = min(idle, hard)
        return hard, idle
    if orig_idle_deadline_at is not None:
        idle = float(orig_idle_deadline_at)
        if hard is not None:
            idle = min(idle, hard)
        return hard, idle
    return hard, hard


def merge_recoverable_models(
    frozen: list[dict[str, Any]] | None,
    incoming: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Union-merge model lists by (category, name) without dropping frozen target items."""
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for item in frozen or []:
        if not isinstance(item, dict):
            continue
        cat, name = item.get("category"), item.get("name")
        if isinstance(cat, str) and cat and isinstance(name, str) and name:
            merged[(cat, name)] = copy.deepcopy(item)
    for item in incoming or []:
        if not isinstance(item, dict):
            continue
        cat, name = item.get("category"), item.get("name")
        if not (isinstance(cat, str) and cat and isinstance(name, str) and name):
            continue
        key = (cat, name)
        if key not in merged:
            merged[key] = copy.deepcopy(item)
        else:
            prev = merged[key]
            updated = {**prev}
            for field in ("sha256", "md5", "size", "drive_file_id", "account", "source"):
                if item.get(field) and not updated.get(field):
                    updated[field] = copy.deepcopy(item[field])
            prev_ts = float(prev.get("last_used_at") or 0)
            new_ts = float(item.get("last_used_at") or 0)
            if new_ts > prev_ts:
                updated["last_used_at"] = new_ts
            merged[key] = updated
    result = list(merged.values())
    result.sort(key=lambda r: (-float(r.get("last_used_at") or 0.0), str(r.get("category") or ""), str(r.get("name") or "")))
    return result


# ---- Persistence helpers -----------------------------------------------------------

def _plan_from_row(row: SettingRow | None) -> dict[str, Any] | None:
    if row is None or not isinstance(row.data, dict):
        return None
    data = copy.deepcopy(row.data)
    if not data.get("plan_id"):
        return None
    return data


def get_plan_in_tx(session, owner_id: str | None, plan_id: str) -> dict[str, Any] | None:
    key = plan_id if plan_id.startswith(PLAN_KEY_PREFIX) else plan_key(plan_id)
    row = session.get(SettingRow, key)
    if row is None:
        return None
    if owner_id is not None and row.owner_id != owner_id:
        return None
    return _plan_from_row(row)


def save_plan_in_tx(session, owner_id: str, plan: dict[str, Any]) -> dict[str, Any]:
    gid = str(plan["recovery_group_id"])
    key = plan_key(gid)
    row = session.get(SettingRow, key)
    prev_state = None if row is None else row.state
    prev_plan_id = None if row is None else (row.data or {}).get("plan_id")
    payload = copy.deepcopy(plan)
    new_state = str(payload.get("state") or "scheduled")
    if row is None:
        row = SettingRow(
            id=key,
            owner_id=owner_id,
            state=new_state,
            data=payload,
            version=0,
            created_at=utcnow(),
            updated_at=utcnow(),
        )
        session.add(row)
    else:
        row.owner_id = owner_id
        row.state = str(payload.get("state") or row.state or "scheduled")
        row.data = payload
        row.version = int(row.version or 0) + 1
        row.updated_at = utcnow()
    session.flush()
    if (
        (new_state == "scheduled" and (prev_state is None or prev_plan_id != payload.get("plan_id")))
        or (new_state != prev_state and new_state in {"recovered", "exhausted", "aborted_fatal", "aborted_vps_outage"})
    ):
        from app import telegram_notify

        telegram_notify.notify_auto_recovery_in_tx(
            session,
            owner_id,
            plan=payload,
            state=new_state,
            now=payload.get("updated_at") or payload.get("created_at"),
        )
    return payload


def list_plans_in_tx(session, owner_id: str | None = None) -> list[dict[str, Any]]:
    stmt = select(SettingRow).where(SettingRow.id.like(f"{PLAN_KEY_PREFIX}%"))
    if owner_id is not None:
        stmt = stmt.where(SettingRow.owner_id == owner_id)
    rows = session.scalars(stmt).all()
    plans = [p for row in rows if (p := _plan_from_row(row)) is not None]
    plans.sort(key=lambda item: float(item.get("updated_at") or item.get("created_at") or 0.0), reverse=True)
    return plans


def get_plan(repo, owner_id: str | None, plan_id: str) -> dict[str, Any] | None:
    store = repo.store if hasattr(repo, "store") else repo
    if not (hasattr(store, "db_lock") and hasattr(store, "Session")):
        return None
    with store.db_lock(), store.Session() as session:
        return get_plan_in_tx(session, owner_id, plan_id)


def list_plans(repo, owner_id: str | None = None) -> list[dict[str, Any]]:
    store = repo.store if hasattr(repo, "store") else repo
    if not (hasattr(store, "db_lock") and hasattr(store, "Session")):
        return []
    with store.db_lock(), store.Session() as session:
        return list_plans_in_tx(session, owner_id)


def save_plan(repo, owner_id: str, plan: dict[str, Any]) -> dict[str, Any]:
    store = repo.store if hasattr(repo, "store") else repo
    if not (hasattr(store, "db_lock") and hasattr(store, "Session")):
        return copy.deepcopy(plan)
    with store.db_lock(), store.Session.begin() as session:
        store.begin_immediate(session)
        return save_plan_in_tx(session, owner_id, plan)


def active_plan(repo, owner_id: str, pool: str | None = None) -> dict[str, Any] | None:
    for plan in list_plans(repo, owner_id):
        if plan.get("state") not in ACTIVE_PLAN_STATES:
            continue
        if pool is not None and plan.get("pool") != pool:
            continue
        return plan
    return None


def latest_plan_for_pool(repo, owner_id: str, pool: str) -> dict[str, Any] | None:
    plans = [p for p in list_plans(repo, owner_id) if p.get("pool") == pool]
    if not plans:
        return None
    for plan in plans:
        if plan.get("state") in ACTIVE_PLAN_STATES:
            return plan
    return plans[0]


def public_plan(plan: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(plan, Mapping):
        return None
    native_snap = plan.get("native_snapshot") if isinstance(plan.get("native_snapshot"), Mapping) else None
    creation_snap = plan.get("creation_snapshot") if isinstance(plan.get("creation_snapshot"), Mapping) else None
    model_restore = None
    wild_models: list[str] = []
    if native_snap is not None:
        raw_wild = native_snap.get("wild_models") or []
        wild_models = [str(w) for w in raw_wild if isinstance(w, str)]
        raw_restore = native_snap.get("model_restore_state")
        if isinstance(raw_restore, Mapping):
            model_restore = {
                "job_id": native_snap.get("model_restore_job_id"),
                "state": raw_restore.get("state"),
                "total": int(raw_restore.get("total") or 0),
                "completed": int(raw_restore.get("completed") or 0),
                "failed": int(raw_restore.get("failed") or 0),
                "skipped": int(raw_restore.get("skipped") or 0),
                "partial_disk_limit": bool(raw_restore.get("partial_disk_limit")),
                "updated_at": raw_restore.get("updated_at"),
            }
        elif native_snap.get("recoverable_models"):
            model_restore = {
                "job_id": native_snap.get("model_restore_job_id"),
                "state": "pending" if plan.get("state") in ACTIVE_PLAN_STATES else None,
                "total": len(native_snap.get("recoverable_models") or []),
                "completed": 0,
                "failed": 0,
                "skipped": 0,
                "partial_disk_limit": False,
                "updated_at": None,
            }
    return {
        "plan_id": plan.get("plan_id"),
        "recovery_group_id": plan.get("recovery_group_id"),
        "pool": plan.get("pool"),
        "state": plan.get("state"),
        "account_id": plan.get("account_id"),
        "account_alias_snapshot": plan.get("account_alias_snapshot"),
        "requested_spec": plan.get("requested_spec"),
        "retention_mode": plan.get("retention_mode"),
        "retain_idle_minutes": plan.get("retain_idle_minutes"),
        "hard_deadline_at": plan.get("hard_deadline_at"),
        "orig_idle_deadline_at": plan.get("orig_idle_deadline_at"),
        "was_busy": bool(plan.get("was_busy")),
        "rounds_used": int(plan.get("rounds_used") or 0),
        "max_rounds": int(plan.get("max_rounds") or 3),
        "backoff_attempt": int(plan.get("backoff_attempt") or 0),
        "next_retry_at": plan.get("next_retry_at"),
        "round_started_at": plan.get("round_started_at"),
        "source_instance_id": plan.get("source_instance_id"),
        "active_job_id": plan.get("active_job_id"),
        "successor_instance_id": plan.get("successor_instance_id"),
        "last_trigger_reason": plan.get("last_trigger_reason"),
        "last_error_code": plan.get("last_error_code"),
        "last_error_message": plan.get("last_error_message"),
        "model_restore": model_restore,
        "wild_models": wild_models,
        "creation_mode": creation_snap.get("mode") if creation_snap else None,
        "workflow_key": creation_snap.get("workflow_key") if creation_snap else None,
        "workflow_version": creation_snap.get("workflow_version") if creation_snap else None,
        "created_at": plan.get("created_at"),
        "updated_at": plan.get("updated_at"),
        "cancelled_at": plan.get("cancelled_at"),
        "history": list(plan.get("history") or [])[-10:],
    }


def status_summary(repo, owner_id: str) -> dict[str, Any]:
    cfg = RecoveryConfig.from_env()
    plans = list_plans(repo, owner_id)
    active = [public_plan(p) for p in plans if p.get("state") in ACTIVE_PLAN_STATES]
    recent = [public_plan(p) for p in plans[:10]]
    return {
        "config": cfg.to_public(),
        "active_plans": active,
        "recent_plans": recent,
    }


def _emit_event_in_tx(
    session,
    owner_id: str,
    project_id: str | None,
    instance_id: str | None,
    job_id: str | None,
    state: str,
    message: str,
    **extra,
) -> None:
    session.add(
        GpuEventRow(
            id=str(uuid.uuid4()),
            owner_id=owner_id,
            project_id=project_id,
            state=state,
            version=0,
            data={"instance_id": instance_id, "job_id": job_id, "state": state, "message": str(message)[:500], **extra},
            created_at=utcnow(),
            updated_at=utcnow(),
        )
    )


def _account_available_in_tx(session, owner_id: str, account_id: str | None) -> bool:
    if not account_id:
        return False
    rows = session.scalars(select(GpuAccountRow).where(GpuAccountRow.owner_id == owner_id)).all()
    row = next((r for r in rows if r.id == account_id or (r.data or {}).get("account_id") == account_id), None)
    if row is None:
        return False
    data = row.data or {}
    status = data.get("auth_status", data.get("status", data.get("connection_status", "connected")))
    connected = data.get("connected", status == "connected")
    return connected is True and data.get("enabled", True) is True and data.get("allow_allocation", True) is True


def _workflow_eligible_in_tx(session, owner_id: str, workflow_id: str | None, workflow_key: str | None, workflow_version: Any) -> dict | None:
    rows = session.scalars(select(GpuWorkflowRow).where(GpuWorkflowRow.owner_id == owner_id)).all()
    target = None
    for row in rows:
        data = row.data or {}
        if workflow_id and row.id == workflow_id:
            target = row
            break
        if workflow_key and row.workflow_key == workflow_key and str(row.workflow_version) == str(workflow_version):
            target = row
            break
    if target is None:
        return None
    data = dict(target.data or {})
    if target.state != "published" or not data.get("review_approved") or data.get("smoke_test_kind") != "real":
        return None
    return {
        "id": target.id,
        "owner_id": target.owner_id,
        "workflow_key": target.workflow_key,
        "workflow_version": int(target.workflow_version) if str(target.workflow_version).isdigit() else target.workflow_version,
        "state": target.state,
        **data,
    }


def _normalize_spec(raw: Any) -> dict[str, str] | None:
    if not isinstance(raw, Mapping):
        return None
    inner = raw.get("key") if isinstance(raw.get("key"), Mapping) else raw
    if not isinstance(inner, Mapping):
        return None
    try:
        from .specs import spec_key

        return spec_key(dict(inner))
    except ValueError:
        if all(k in inner for k in ("variant", "accelerator", "shape")):
            return {k: str(inner[k]) for k in ("variant", "accelerator", "shape")}
        return None


def evaluate_recovery_eligibility_in_tx(
    repo,
    session,
    *,
    owner_id: str,
    pool: str,
    account_id: str | None,
    retention_mode: str | None,
    hard_deadline_at: float | None,
    orig_idle_deadline_at: float | None,
    was_busy: bool,
    rounds_used: int,
    now: float,
    cfg: RecoveryConfig | None = None,
    manual_stop_requested: bool = False,
    force_stop: bool = False,
    fuse_forced_stop: bool = False,
    native_draining: bool = False,
    was_transferring: bool = False,
    native_bootstrap_failed: bool = False,
    last_successful_probe_at: float | None = None,
    process_started_at: float | None = None,
    check_vps_outage: bool = False,
    vps_outage_now: float | None = None,
    check_concurrency: bool = True,
    active_job_id: str | None = None,
    active_lease_id: str | None = None,
    ignore_job_id: str | None = None,
    ignore_instance_id: str | None = None,
    source_instance_id: str | None = None,
) -> tuple[bool, str, str]:
    """Evaluate all 10 eligibility rules from §5.4.

    Returns ``(eligible, code, message)`` where ``code`` indicates the exact reason
    (e.g. ``"ok"``, ``"disabled"``, ``"startup_grace"``, ``"aborted_vps_outage"``,
    ``"non_candidate"``, ``"exhausted"``, ``"aborted_fatal"``, ``"concurrent_active"``,
    ``"inflight_pending"``).
    """
    cfg = cfg or RecoveryConfig.from_env()
    gpu_cfg = GpuConfig.from_env()

    # 1. Master switches
    if not cfg.enabled or not gpu_cfg.real_enabled:
        return False, "disabled", "自动恢复或真实 GPU 未启用"

    # 4. Pool & policy filter
    if was_transferring:
        return False, "non_candidate", "实例正在执行域转移，不触发跨机自动恢复"
    if native_bootstrap_failed:
        return False, "non_candidate", "原生启动失败待命态不触发跨机自动恢复"
    if pool == POOL_EPHEMERAL:
        return False, "non_candidate", "一次性任务不触发跨机自动恢复"
    if pool == POOL_CREATION and retention_mode not in {"smart", "retain"}:
        return False, "non_candidate", "仅 smart/retain 长驻创作实例支持跨机自动恢复"
    if pool not in {POOL_NATIVE, POOL_CREATION}:
        return False, "non_candidate", "不支持的实例域"

    # 5. Voluntary stop / drain veto
    if manual_stop_requested or force_stop or fuse_forced_stop or native_draining:
        return False, "non_candidate", "实例已主动停止、排空或强制熔断，不触发自动恢复"

    # 3. VPS long outage / maintenance protection (MUST precede startup_grace!)
    if check_vps_outage and last_successful_probe_at is not None:
        outage_ref = now if vps_outage_now is None else float(vps_outage_now)
        outage_s = max(0.0, outage_ref - float(last_successful_probe_at))
        if outage_s >= cfg.max_vps_outage_minutes * 60.0:
            return (
                False,
                "aborted_vps_outage",
                f"VPS 距上次成功通信已超过 {int(outage_s // 60)} 分钟（阈值 {cfg.max_vps_outage_minutes} 分钟），疑似 VPS 停机维护或 VM 失联自释放，已放弃自动重拉",
            )

    # 6. CU soft & hard fuses (permanent vetoes evaluated before startup_grace)
    guard = CuGuardService(repo, lambda: now)
    guard_state = guard.state_in(session, owner_id)
    if guard_state.get("soft_tripped"):
        return False, "aborted_fatal", guard_state.get("daily_message") or "已触发今日 CU 上限温和熔断，停止自动恢复"
    hard_ids = {row["account_id"] for row in guard_state.get("hard_tripped_accounts") or []}
    if account_id and account_id in hard_ids:
        return False, "aborted_fatal", guard_state.get("message") or "原账号已触及余额底线硬熔断，停止自动恢复"

    # 7. Original account availability
    if not _account_available_in_tx(session, owner_id, account_id):
        return False, "aborted_fatal", "原 Google 账号不可用、已停用或未开启自动分配，已终止自动恢复"

    # 8. Round quota
    if int(rounds_used) >= cfg.max_rounds:
        return False, "exhausted", f"已达到单生命周期最大自动恢复轮次（{cfg.max_rounds}/{cfg.max_rounds}）"

    # 9. Remaining effective lifetime threshold
    min_rem = cfg.min_remaining_minutes * 60.0
    if pool == POOL_NATIVE:
        if hard_deadline_at is None or (float(hard_deadline_at) - now) < min_rem:
            return False, "aborted_fatal", f"会话剩余硬截止寿命不足 {cfg.min_remaining_minutes} 分钟，已放弃自动恢复"
        if not was_busy:
            if orig_idle_deadline_at is not None and (float(orig_idle_deadline_at) - now) < min_rem:
                return False, "aborted_fatal", f"空闲剩余保留时长不足 {cfg.min_remaining_minutes} 分钟，已放弃自动恢复"
    else:
        if was_busy:
            if hard_deadline_at is None or (float(hard_deadline_at) - now) < min_rem:
                return False, "aborted_fatal", f"任务剩余有效寿命不足 {cfg.min_remaining_minutes} 分钟，已放弃自动恢复"
        else:
            candidates = [float(d) for d in (orig_idle_deadline_at, hard_deadline_at) if d is not None]
            effective_deadline = min(candidates) if candidates else None
            if effective_deadline is None or (effective_deadline - now) < min_rem:
                return False, "aborted_fatal", f"空闲剩余保留时长不足 {cfg.min_remaining_minutes} 分钟，已放弃自动恢复"

    # 2. Startup grace window (evaluated after permanent vetoes, before transient concurrency/dispatch)
    started_at = _PROCESS_STARTED_AT if process_started_at is None else float(process_started_at)
    if cfg.startup_grace_seconds > 0 and (now - started_at) < cfg.startup_grace_seconds:
        return False, "startup_grace", "watchdog 处于启动宽限期，稍后复核"

    # 10. Concurrency & inflight lease exclusion
    if active_lease_id:
        inflight_lease = session.get(GpuLeaseRow, active_lease_id)
        if inflight_lease is not None and inflight_lease.state != "deleted":
            return False, "inflight_pending", "上次继任分配租约尚未确认释放，等待核对完成"
    if active_job_id:
        inflight_job = session.get(GpuJobRow, active_job_id)
        if active_job_id != ignore_job_id and inflight_job is not None and inflight_job.state not in TERMINAL_JOB_STATES:
            return False, "inflight_pending", "已有在途恢复任务正在推进"
        if inflight_job is not None:
            for lease_row in session.scalars(select(GpuLeaseRow).where(GpuLeaseRow.owner_id == owner_id)).all():
                ldata = lease_row.data or {}
                if ldata.get("owner_job_id") == active_job_id and lease_row.state != "deleted":
                    return False, "inflight_pending", "在途恢复任务的旧租约尚未确认释放"

    if check_concurrency:
        if pool == POOL_NATIVE:
            for lease_row in session.scalars(select(GpuLeaseRow).where(GpuLeaseRow.owner_id == owner_id)).all():
                ldata = lease_row.data or {}
                if ldata.get("native_comfy") and lease_row.state not in {"deleted", "quarantined", "releasing", "deleting"}:
                    return False, "concurrent_active", "检测到已存在活跃原生实例，停止重复恢复"
            for inst_row in session.scalars(select(GpuInstanceRow).where(GpuInstanceRow.owner_id == owner_id)).all():
                if inst_row.id == ignore_instance_id:
                    continue
                idata = inst_row.data or {}
                if inst_row.state == "transferring" and (idata.get("transfer_state") or {}).get("to_pool") == POOL_NATIVE:
                    return False, "concurrent_active", "检测到已有 GPU 实例正在转入原生域，停止重复恢复"
            reclaimed_ids = {i for i in (ignore_instance_id, source_instance_id) if i}
            for job_row in session.scalars(select(GpuJobRow).where(GpuJobRow.owner_id == owner_id)).all():
                if job_row.id == ignore_job_id:
                    continue
                jdata = job_row.data or {}
                if (jdata.get("snapshot") or {}).get("native_comfy") and job_row.state not in TERMINAL_JOB_STATES:
                    if job_row.state in {"cleaning_up", "cleanup_pending"} and (
                        (reclaimed_ids and (jdata.get("instance_id") in reclaimed_ids or job_row.instance_id in reclaimed_ids))
                        or jdata.get("error_code") == RUNTIME_MISSING
                    ):
                        return False, "inflight_pending", "原原生启动任务正在收尾释放，稍后复核"
                    return False, "concurrent_active", "检测到已存在进行中的原生启动任务，停止重复恢复"
        elif pool == POOL_CREATION:
            for inst_row in session.scalars(select(GpuInstanceRow).where(GpuInstanceRow.owner_id == owner_id)).all():
                if inst_row.id == ignore_instance_id:
                    continue
                idata = inst_row.data or {}
                if (
                    inst_row.state in {"allocating", "ready", "busy", "retained_idle"}
                    and instance_pool_of(idata) == POOL_CREATION
                    and idata.get("account_id") == account_id
                    and not idata.get("manual_stop_requested")
                    and not idata.get("runtime_deleted")
                ):
                    return False, "concurrent_active", "原账号下已存在活跃创作池实例，取消重复自动恢复"
            for job_row in session.scalars(select(GpuJobRow).where(GpuJobRow.owner_id == owner_id)).all():
                if job_row.id == ignore_job_id:
                    continue
                if job_row.state not in {"allocating", "preparing", "loading", "running", "uploading"}:
                    continue
                jdata = job_row.data or {}
                if jdata.get("cancel_requested"):
                    continue
                j_acct = jdata.get("account_id") or jdata.get("requested_account_id") or (jdata.get("execution_policy") or {}).get("account_id")
                if j_acct != account_id:
                    continue
                j_pool = jdata.get("instance_pool")
                if j_pool is None:
                    snap = jdata.get("snapshot") or {}
                    if snap.get("native_comfy"):
                        j_pool = POOL_NATIVE
                    elif snap.get("system_smoke") or (jdata.get("execution_policy") or {}).get("lifecycle") == "save":
                        j_pool = POOL_EPHEMERAL
                    else:
                        j_pool = POOL_CREATION
                if j_pool == POOL_CREATION:
                    return False, "concurrent_active", "原账号下已存在推进中的创作池任务，取消重复自动恢复"

    return True, "ok", "满足自动恢复条件"


def _local_outputs_persisted(job_data: Mapping[str, Any]) -> bool:
    """Return True if an uploading job already has outputs persisted locally on VPS."""
    if job_data.get("output_asset_ids") or job_data.get("persisted_outputs"):
        return True
    outputs = job_data.get("outputs")
    if isinstance(outputs, list) and outputs:
        from pathlib import Path

        artifacts_dir = settings.data_dir / "assets" / "artifacts"
        all_local = True
        for item in outputs:
            if not isinstance(item, Mapping):
                all_local = False
                break
            if item.get("content"):
                continue
            storage = item.get("storage")
            if isinstance(storage, Mapping):
                if storage.get("local_path") and Path(str(storage["local_path"])).is_file():
                    continue
                if storage.get("provider") in {"local", "mock"}:
                    obj_key = str(storage.get("object_key") or item.get("artifact_id") or "").strip()
                    if obj_key and "/" not in obj_key and "\\" not in obj_key and obj_key not in {".", ".."}:
                        if (artifacts_dir / f"{obj_key}.data").is_file() or (artifacts_dir / obj_key).is_file():
                            continue
            all_local = False
            break
        if all_local:
            return True
    return False


def _read_local_models_snapshot_in_tx(
    session,
    owner_id: str,
    instance_id: str,
    previous_snapshot: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    row = session.get(SettingRow, local_models_key(owner_id))
    data = (row.data or {}) if row is not None and isinstance(row.data, dict) else {}
    by_instance = data.get("by_instance") if isinstance(data.get("by_instance"), dict) else {}
    inst_entry = by_instance.get(instance_id) if isinstance(by_instance.get(instance_id), dict) else {}

    prev_recoverable = list((previous_snapshot or {}).get("recoverable_models") or [])
    prev_wild = list((previous_snapshot or {}).get("wild_models") or [])

    inc_recoverable = list(inst_entry.get("recoverable_models") or [])
    inc_wild = [str(w) for w in (inst_entry.get("wild_models") or []) if isinstance(w, str)]

    merged_recoverable = merge_recoverable_models(prev_recoverable, inc_recoverable)
    merged_wild = sorted(set(prev_wild) | set(inc_wild))
    return merged_recoverable, merged_wild


def on_runtime_reclaimed_in_tx(
    repo,
    session,
    snapshot: Mapping[str, Any],
    checked_at: float,
    *,
    process_started_at: float | None = None,
) -> dict[str, Any] | None:
    """Atomically evaluate and persist a recovery plan inside _close_reclaimed_runtime's TX.

    Returns ``None`` when L3 is disabled or the instance is not a candidate.
    Otherwise returns ``{"plan": plan, "takeover_job_id": str | None, "recovery_group_id": str}``.
    """
    cfg = RecoveryConfig.from_env()
    if not cfg.enabled:
        return None

    owner_id = str(snapshot["owner_id"])
    instance_id = str(snapshot["instance_id"])
    pool = str(snapshot["pool"])
    account_id = snapshot.get("account_id")
    retention_mode = snapshot.get("retention_mode")
    was_busy = bool(snapshot.get("was_busy"))
    hard_deadline_at = snapshot.get("hard_deadline_at")
    orig_idle_deadline_at = snapshot.get("orig_idle_deadline_at")
    retain_idle_minutes = snapshot.get("retain_idle_minutes")
    requested_spec = _normalize_spec(snapshot.get("requested_spec"))

    # Load existing plan if this reclaimed VM was already part of a recovery chain.
    existing_gid = snapshot.get("recovery_group_id")
    existing_plan = get_plan_in_tx(session, owner_id, existing_gid) if existing_gid else None
    recovery_group_id = str(existing_gid or make_group_id(instance_id))
    rounds_used = int((existing_plan or {}).get("rounds_used") or 0)

    # Preserve initial logical deadlines across multiple recovery rounds!
    if existing_plan:
        if existing_plan.get("hard_deadline_at") is not None:
            hard_deadline_at = float(existing_plan["hard_deadline_at"])
        if not was_busy and orig_idle_deadline_at is None and existing_plan.get("orig_idle_deadline_at") is not None:
            orig_idle_deadline_at = float(existing_plan["orig_idle_deadline_at"])

    # Determine interrupted job for Creation Path B
    interrupted_job_row = None
    interrupted_job_data: dict[str, Any] = {}
    candidate_job_id = snapshot.get("current_job_id")
    if pool == POOL_CREATION and was_busy and candidate_job_id:
        interrupted_job_row = session.get(GpuJobRow, candidate_job_id)
        if interrupted_job_row is not None and interrupted_job_row.owner_id == owner_id:
            interrupted_job_data = dict(interrupted_job_row.data or {})
        else:
            interrupted_job_row = None

    # Check if Path B job explicitly downgraded lifecycle to save/batch (§6.2 Path B(1))
    effective_retention = retention_mode
    if interrupted_job_row is not None:
        jpolicy = interrupted_job_data.get("execution_policy") if isinstance(interrupted_job_data.get("execution_policy"), Mapping) else {}
        job_lc = jpolicy.get("lifecycle") or interrupted_job_data.get("lifecycle")
        if job_lc in {"save", "batch"}:
            effective_retention = job_lc

    # Check if job already has local outputs persisted on VPS or is already finalizing cleanup -> do not re-run GPU (§5.3, §6.2)
    if interrupted_job_row is not None:
        if _local_outputs_persisted(interrupted_job_data):
            return None
        if (
            interrupted_job_row.state in {"cleaning_up", "cleanup_pending"}
            or bool(interrupted_job_data.get("cleanup_final_state"))
            or bool(interrupted_job_data.get("cancel_requested"))
        ):
            return None

    # Check if this instance is even a candidate before persisting anything.
    eligible, code, reason_msg = evaluate_recovery_eligibility_in_tx(
        repo,
        session,
        owner_id=owner_id,
        pool=pool,
        account_id=account_id,
        retention_mode=effective_retention,
        hard_deadline_at=hard_deadline_at,
        orig_idle_deadline_at=orig_idle_deadline_at,
        was_busy=was_busy,
        rounds_used=rounds_used,
        now=checked_at,
        cfg=cfg,
        manual_stop_requested=bool(snapshot.get("manual_stop_requested")),
        force_stop=bool(snapshot.get("force_stop")),
        fuse_forced_stop=bool(snapshot.get("fuse_forced_stop")),
        native_draining=bool(snapshot.get("native_draining")),
        was_transferring=bool(snapshot.get("was_transferring")),
        native_bootstrap_failed=bool(snapshot.get("native_bootstrap_failed")),
        last_successful_probe_at=snapshot.get("last_successful_probe_at"),
        process_started_at=process_started_at,
        check_vps_outage=True,
        vps_outage_now=checked_at,
        check_concurrency=True,
        ignore_job_id=candidate_job_id if pool == POOL_CREATION and was_busy else snapshot.get("owner_job_id"),
        ignore_instance_id=instance_id,
        source_instance_id=instance_id,
    )

    if code in {"disabled", "non_candidate"}:
        # If there was an active plan for a manually stopped/drained successor, cancel it.
        if existing_plan and existing_plan.get("state") in ACTIVE_PLAN_STATES:
            existing_plan["state"] = "cancelled"
            existing_plan["cancelled_at"] = checked_at
            existing_plan["updated_at"] = checked_at
            existing_plan["last_error_message"] = reason_msg
            save_plan_in_tx(session, owner_id, existing_plan)
        return None

    # Build/preserve domain snapshot
    native_snapshot = None
    creation_snapshot = None
    takeover_job_id: str | None = None

    if pool == POOL_NATIVE:
        prev_ns = (existing_plan or {}).get("native_snapshot")
        recoverable, wild = _read_local_models_snapshot_in_tx(session, owner_id, instance_id, previous_snapshot=prev_ns)
        native_snapshot = {
            "comfy_commit": snapshot.get("comfy_commit") or (prev_ns or {}).get("comfy_commit"),
            "model_storage": copy.deepcopy(snapshot.get("model_storage") or (prev_ns or {}).get("model_storage") or {}),
            "plugin_generation": snapshot.get("plugin_generation") or (prev_ns or {}).get("plugin_generation"),
            "recoverable_models": recoverable,
            "wild_models": wild,
            "model_restore_job_id": None,
            "model_restore_state": None,
        }
    elif pool == POOL_CREATION:
        cr = snapshot.get("creation_runtime") if isinstance(snapshot.get("creation_runtime"), Mapping) else {}
        prev_cs = (existing_plan or {}).get("creation_snapshot") or {}
        if (
            was_busy
            and interrupted_job_row is not None
            and interrupted_job_row.state in ACTIVE_CREATION_TAKEOVER_STATES
            and not interrupted_job_data.get("cleanup_final_state")
            and not _local_outputs_persisted(interrupted_job_data)
        ):
            if interrupted_job_data.get("cancel_requested"):
                return None
            retries = int(interrupted_job_data.get("auto_retry_count") or 0)
            if (eligible or code == "startup_grace") and retries >= cfg.max_job_retries:
                eligible = False
                code = "aborted_fatal"
                reason_msg = f"该任务已达到自动重试上限（{cfg.max_job_retries} 次），已停止自动恢复"
            elif eligible or code == "startup_grace":
                takeover_job_id = interrupted_job_row.id
            creation_snapshot = {
                "mode": "retry_job",
                "interrupted_job_id": interrupted_job_row.id,
                "workflow_id": interrupted_job_data.get("workflow_version_id") or interrupted_job_data.get("creation_workflow_id") or prev_cs.get("workflow_id"),
                "workflow_key": interrupted_job_data.get("workflow_key") or cr.get("workflow_key") or prev_cs.get("workflow_key"),
                "workflow_version": interrupted_job_data.get("workflow_version") or cr.get("workflow_version") or prev_cs.get("workflow_version"),
                "fingerprint": interrupted_job_data.get("creation_fingerprint") or cr.get("fingerprint") or prev_cs.get("fingerprint"),
                "project_id": interrupted_job_row.project_id or snapshot.get("project_id") or prev_cs.get("project_id"),
            }
        else:
            # Path A: idle retained warmup recovery
            wf_id = cr.get("workflow_id") or snapshot.get("workflow_version_id") or prev_cs.get("workflow_id")
            wf_key = cr.get("workflow_key") or snapshot.get("workflow_key") or prev_cs.get("workflow_key")
            wf_ver = cr.get("workflow_version") if cr.get("workflow_version") is not None else (snapshot.get("workflow_version") or prev_cs.get("workflow_version"))
            if eligible or code == "startup_grace":
                wf_row = _workflow_eligible_in_tx(session, owner_id, wf_id, wf_key, wf_ver)
                if wf_row is None:
                    eligible = False
                    code = "aborted_fatal"
                    reason_msg = "原工作流版本已撤销或不可执行，已放弃自动再预热"
                else:
                    wf_id = wf_row["id"]
                    wf_key = wf_row["workflow_key"]
                    wf_ver = wf_row["workflow_version"]
            creation_snapshot = {
                "mode": "warmup",
                "interrupted_job_id": None,
                "workflow_id": wf_id,
                "workflow_key": wf_key,
                "workflow_version": wf_ver,
                "fingerprint": cr.get("fingerprint") or prev_cs.get("fingerprint"),
                "project_id": snapshot.get("project_id") or prev_cs.get("project_id"),
            }

    # Startup grace: permanent vetoes already evaluated above; persist scheduled plan until grace ends!
    if code == "startup_grace":
        started_at = _PROCESS_STARTED_AT if process_started_at is None else float(process_started_at)
        next_retry_at = started_at + cfg.startup_grace_seconds
        plan_state = "scheduled"
    elif eligible:
        plan_state = "scheduled"
        next_retry_at = checked_at
    else:
        plan_state = code if code in {"aborted_vps_outage", "exhausted", "aborted_fatal"} else "aborted_fatal"
        next_retry_at = None
        takeover_job_id = None

    plan: dict[str, Any] = {
        "plan_id": recovery_group_id,
        "recovery_group_id": recovery_group_id,
        "owner_id": owner_id,
        "pool": pool,
        "state": plan_state,
        "account_id": account_id,
        "account_alias_snapshot": snapshot.get("account_alias_snapshot") or (existing_plan or {}).get("account_alias_snapshot"),
        "drive_account_id": snapshot.get("drive_account_id") or (existing_plan or {}).get("drive_account_id"),
        "requested_spec": requested_spec or (existing_plan or {}).get("requested_spec"),
        "retention_mode": effective_retention,
        "retain_idle_minutes": retain_idle_minutes or (existing_plan or {}).get("retain_idle_minutes"),
        "hard_deadline_at": hard_deadline_at,
        "orig_idle_deadline_at": orig_idle_deadline_at,
        "was_busy": was_busy,
        "rounds_used": rounds_used,
        "max_rounds": cfg.max_rounds,
        "round_counted_for_instance_id": (existing_plan or {}).get("round_counted_for_instance_id"),
        "round_started_at": checked_at,
        "reclaimed_at": checked_at,
        "last_successful_probe_at": snapshot.get("last_successful_probe_at"),
        "backoff_attempt": 0,
        "dispatch_seq": int((existing_plan or {}).get("dispatch_seq") or 0),
        "next_retry_at": next_retry_at,
        "source_instance_id": instance_id,
        "active_job_id": takeover_job_id,
        "active_lease_id": None,
        "successor_instance_id": None,
        "last_trigger_reason": RUNTIME_MISSING,
        "last_error_code": None if plan_state == "scheduled" else code,
        "last_error_message": None if plan_state == "scheduled" else reason_msg,
        "native_snapshot": native_snapshot,
        "creation_snapshot": creation_snapshot,
        "created_at": (existing_plan or {}).get("created_at") or checked_at,
        "updated_at": checked_at,
        "cancelled_at": None,
        "history": list((existing_plan or {}).get("history") or []),
    }
    save_plan_in_tx(session, owner_id, plan)

    event_msg = (
        f"检测到 Colab 运行时被上游回收，已创建自动恢复计划（第 {rounds_used + 1}/{cfg.max_rounds} 轮）"
        if plan_state == "scheduled"
        else f"检测到 Colab 运行时被上游回收，未执行自动恢复：{reason_msg}"
    )
    _emit_event_in_tx(
        session,
        owner_id,
        snapshot.get("project_id"),
        instance_id,
        takeover_job_id or snapshot.get("owner_job_id"),
        plan_state,
        event_msg,
        recovery_group_id=recovery_group_id,
        recovery_state=plan_state,
    )
    return {
        "plan": plan,
        "takeover_job_id": takeover_job_id,
        "recovery_group_id": recovery_group_id,
    }


def on_successor_allocation_confirmed(
    repo,
    owner_id: str,
    plan_id: str,
    new_instance_id: str,
    lease_id: str | None,
    now: float,
) -> dict[str, Any] | None:
    """Unique round-deduction point: invoked once successor VM reaches preparing->ready (§5.2)."""
    store = repo.store if hasattr(repo, "store") else repo
    with _wait_dispatch_lock(store) as acquired:
        if not acquired:
            return None
        with store.db_lock(), store.Session.begin() as session:
            store.begin_immediate(session)
            plan = get_plan_in_tx(session, owner_id, plan_id)
            if not plan:
                return None
            if plan.get("round_counted_for_instance_id") != new_instance_id:
                plan["rounds_used"] = int(plan.get("rounds_used") or 0) + 1
                plan["round_counted_for_instance_id"] = new_instance_id
            plan["successor_instance_id"] = new_instance_id
            if lease_id:
                plan["active_lease_id"] = lease_id
            plan["updated_at"] = now
            save_plan_in_tx(session, owner_id, plan)

            # Link parent instance -> replaced_by_instance_id
            parent_id = plan.get("source_instance_id")
            if parent_id and parent_id != new_instance_id:
                parent_row = session.get(GpuInstanceRow, parent_id)
                if parent_row is not None and parent_row.owner_id == owner_id:
                    pdata = copy.deepcopy(parent_row.data or {})
                    if pdata.get("replaced_by_instance_id") != new_instance_id:
                        pdata["replaced_by_instance_id"] = new_instance_id
                        pdata.setdefault("recovery_group_id", plan["recovery_group_id"])
                        parent_row.data = pdata
                        parent_row.version = int(parent_row.version or 0) + 1
                        parent_row.updated_at = utcnow()
            return plan


def bind_active_lease_for_plan(repo, owner_id: str, plan_id: str, lease_id: str, now: float) -> None:
    """Track the current successor attempt's lease_id on the recovery plan."""
    store = repo.store if hasattr(repo, "store") else repo
    with store.db_lock(), store.Session.begin() as session:
        store.begin_immediate(session)
        plan = get_plan_in_tx(session, owner_id, plan_id)
        if not plan or plan.get("state") not in {"dispatching", "cancelling"}:
            return
        if plan.get("active_lease_id") != lease_id:
            plan["active_lease_id"] = lease_id
            plan["updated_at"] = now
            save_plan_in_tx(session, owner_id, plan)


def _compute_backoff_delay(cfg: RecoveryConfig, attempt: int, failure_kind: str) -> float:
    base = max(60, cfg.backoff_base_seconds) if failure_kind == "rate_limited" else cfg.backoff_base_seconds
    return float(min(240, base * (2 ** max(0, attempt - 1))))


def maybe_defer_retry_job_after_cleanup(
    repo,
    job: dict[str, Any],
    lease: dict[str, Any] | None,
    now: float,
) -> dict[str, Any] | None:
    """Keep a Path-B retry job alive across single-round allocation backoff once its failed lease is confirmed deleted."""
    plan_id = job.get("recovery_plan_id")
    if not plan_id:
        return None
    owner_id = job["owner_id"]
    cfg = RecoveryConfig.from_env()
    store = repo.store if hasattr(repo, "store") else repo

    with _wait_dispatch_lock(store) as acquired:
        if not acquired:
            return None
        with store.db_lock(), store.Session.begin() as session:
            store.begin_immediate(session)
            plan = get_plan_in_tx(session, owner_id, plan_id)
            if (
                not plan
                or plan.get("pool") != POOL_CREATION
                or (plan.get("creation_snapshot") or {}).get("mode") != "retry_job"
                or plan.get("active_job_id") != job["id"]
                or plan.get("state") != "dispatching"
            ):
                return None

            # Ensure lease (and any other lease owned by this job) is genuinely deleted before scheduling another attempt.
            if lease and lease.get("id"):
                lrow = session.get(GpuLeaseRow, lease["id"])
                if lrow is not None and lrow.state != "deleted":
                    return None
            for lrow in session.scalars(select(GpuLeaseRow).where(GpuLeaseRow.owner_id == owner_id)).all():
                if (lrow.data or {}).get("owner_job_id") == job["id"] and lrow.state != "deleted":
                    return None

            job_row = session.get(GpuJobRow, job["id"])
            if job_row is None or job_row.owner_id != owner_id:
                return None
            jdata = copy.deepcopy(job_row.data or {})
            if jdata.get("cancel_requested"):
                return None

            err_code = jdata.get("error_code") or job.get("error_code")
            err_msg = jdata.get("error_message") or job.get("error_message")
            err_details = jdata.get("error_details") or job.get("error_details")
            kind = classify_failure(err_code, err_msg, err_details)

            if kind == "fatal":
                plan["state"] = "aborted_fatal"
                plan["last_error_code"] = err_code
                plan["last_error_message"] = err_msg
                plan["active_lease_id"] = None
                plan["updated_at"] = now
                save_plan_in_tx(session, owner_id, plan)
                return None

            round_started = float(plan.get("round_started_at") or now)
            if now - round_started >= cfg.round_timeout_seconds:
                plan["state"] = "exhausted"
                plan["last_error_code"] = err_code
                plan["last_error_message"] = f"单轮自动恢复退避超时（{cfg.round_timeout_seconds}s）：{err_msg or err_code}"
                plan["active_lease_id"] = None
                plan["updated_at"] = now
                save_plan_in_tx(session, owner_id, plan)
                return None

            if kind == "local_lock":
                next_attempt = int(plan.get("backoff_attempt") or 0)
                delay = 5.0
            else:
                next_attempt = int(plan.get("backoff_attempt") or 0) + 1
                delay = _compute_backoff_delay(cfg, next_attempt, kind)
            next_retry_at = now + delay
            if next_retry_at - round_started > cfg.round_timeout_seconds:
                if (round_started + cfg.round_timeout_seconds) - now < 5.0:
                    plan["state"] = "exhausted"
                    plan["last_error_code"] = err_code
                    plan["last_error_message"] = f"单轮自动恢复退避超时（{cfg.round_timeout_seconds}s）：{err_msg or err_code}"
                    plan["active_lease_id"] = None
                    plan["updated_at"] = now
                    save_plan_in_tx(session, owner_id, plan)
                    return None
                next_retry_at = round_started + cfg.round_timeout_seconds

            eligible, elig_code, elig_msg = evaluate_recovery_eligibility_in_tx(
                repo,
                session,
                owner_id=owner_id,
                pool=POOL_CREATION,
                account_id=plan.get("account_id"),
                retention_mode=plan.get("retention_mode"),
                hard_deadline_at=plan.get("hard_deadline_at"),
                orig_idle_deadline_at=plan.get("orig_idle_deadline_at"),
                was_busy=True,
                rounds_used=int(plan.get("rounds_used") or 0),
                now=next_retry_at,
                cfg=cfg,
                check_concurrency=False,
                ignore_job_id=job["id"],
            )
            if not eligible and elig_code != "startup_grace":
                plan["state"] = "exhausted" if elig_code == "exhausted" else "aborted_fatal"
                plan["last_error_code"] = elig_code
                plan["last_error_message"] = elig_msg
                plan["active_lease_id"] = None
                plan["updated_at"] = now
                save_plan_in_tx(session, owner_id, plan)
                return None

            # Reset job back to waiting_for_capacity with a brand-new attempt_id & recovery_hold=True.
            old_execution_id = jdata.get("execution_id")
            if old_execution_id:
                exec_row = session.get(GpuExecutionRow, old_execution_id)
                if (
                    exec_row is not None
                    and exec_row.owner_id == owner_id
                    and exec_row.state not in TERMINAL_JOB_STATES
                ):
                    exec_row.state = "failed"
                    exec_row.data = {
                        **(exec_row.data or {}),
                        "state": "failed",
                        "finished_at": now,
                        "error_code": err_code or "auto_recovery_retry",
                    }
                    exec_row.version = int(exec_row.version or 0) + 1
                    exec_row.updated_at = utcnow()
            old_lease_id = jdata.get("lease_id") or (lease or {}).get("id")
            retry_keys = list(jdata.get("attempt_retry_keys") or [])
            if old_lease_id and old_lease_id not in retry_keys:
                retry_keys.append(old_lease_id)
            new_attempt_id = f"att-{uuid.uuid4().hex[:12]}"
            for key in (
                "instance_id",
                "lease_id",
                "execution_id",
                "error_code",
                "error_message",
                "error_details",
                "cleanup_final_state",
                "requested_outcome",
                "cleanup_attempts",
                "cleanup_next_epoch",
                "instance_completion_recorded",
                "stage",
                "stage_started",
                "selection_reason",
                "instance_state",
                "runtime_id",
                "request_id",
                "poll_retry_count",
                "poll_retry_at",
                "claim_token",
                "claim_until",
            ):
                jdata.pop(key, None)
            jdata.update(
                state="waiting_for_capacity",
                attempt_id=new_attempt_id,
                attempt_retry_keys=retry_keys,
                runtime_deleted=False,
                recovery_hold=True,
                retry_not_before_epoch=next_retry_at,
            )
            job_row.state = "waiting_for_capacity"
            job_row.instance_id = None
            job_row.data = jdata
            job_row.version = int(job_row.version or 0) + 1
            job_row.updated_at = utcnow()

            plan["state"] = "scheduled"
            plan["backoff_attempt"] = next_attempt
            plan["next_retry_at"] = next_retry_at
            plan["active_lease_id"] = None
            plan["last_error_code"] = err_code
            plan["last_error_message"] = f"{err_msg or err_code}；{int(round(next_retry_at - now))}s 后重试"
            plan["updated_at"] = now
            save_plan_in_tx(session, owner_id, plan)

            _emit_event_in_tx(
                session,
                owner_id,
                job_row.project_id,
                None,
                job_row.id,
                "waiting_for_capacity",
                f"自动恢复分配暂未成功（{err_code}），将在 {int(round(next_retry_at - now))} 秒后重试",
                recovery_group_id=plan["recovery_group_id"],
            )
            return repo._row(job_row)


def _refresh_batch_in_tx(session, owner_id: str, batch_id: str | None) -> None:
    if not batch_id:
        return
    session.flush()
    batch = session.get(GpuBatchRow, batch_id)
    if batch is None or batch.owner_id != owner_id:
        return
    job_ids = list((batch.data or {}).get("job_ids") or [])
    if not job_ids:
        return
    jobs = list(
        session.scalars(
            select(GpuJobRow).where(GpuJobRow.owner_id == owner_id, GpuJobRow.id.in_(job_ids))
        )
    )
    counts = {st: sum(j.state == st for j in jobs) for st in ("completed", "failed", "cancelled")}
    finished = sum(counts.values())
    if jobs and finished == len(jobs):
        bstate = next((name for name, count in counts.items() if count == len(jobs)), "partial")
    else:
        bstate = "running" if any(j.state != "queued" for j in jobs) else "queued"
    batch.state = bstate
    batch.data = {**(batch.data or {}), "state": bstate, "finished_jobs": finished, "counts": counts}
    batch.version = int(batch.version or 0) + 1
    batch.updated_at = utcnow()


def _cancel_held_job_in_tx(session, owner_id: str, job_id: str | None, *, state: str, message: str, error_code: str | None = None) -> None:
    if not job_id:
        return
    job_row = session.get(GpuJobRow, job_id)
    if job_row is None or job_row.owner_id != owner_id or job_row.state in TERMINAL_JOB_STATES:
        return
    jdata = copy.deepcopy(job_row.data or {})
    if job_row.state == "waiting_for_capacity" and not jdata.get("lease_id") and not jdata.get("instance_id"):
        jdata["recovery_hold"] = False
        jdata["retry_not_before_epoch"] = 0
        jdata["state"] = state
        if state == "cancelled":
            jdata["cancel_requested"] = True
            jdata["requested_outcome"] = "cancelled"
            if error_code:
                jdata["error_code"] = error_code
            if message:
                jdata["error_message"] = message
        else:
            jdata["requested_outcome"] = "failed"
            jdata["error_code"] = error_code or "auto_recovery_aborted"
            jdata["error_message"] = message
        job_row.state = state
        job_row.data = jdata
        job_row.version = int(job_row.version or 0) + 1
        job_row.updated_at = utcnow()
        _emit_event_in_tx(session, owner_id, job_row.project_id, None, job_row.id, state, message)
        _refresh_batch_in_tx(session, owner_id, jdata.get("batch_id"))


def _cancel_scheduled_plan_in_tx(
    session,
    owner_id: str,
    plan: dict[str, Any],
    now: float,
    reason: str,
    *,
    error_code: str | None = None,
) -> None:
    job_id = plan.get("active_job_id")
    _cancel_held_job_in_tx(session, owner_id, job_id, state="cancelled", message=reason, error_code=error_code)
    plan.update(state="cancelled", cancelled_at=now, updated_at=now, next_retry_at=None, last_error_message=reason)
    if error_code:
        plan["last_error_code"] = error_code
    save_plan_in_tx(session, owner_id, plan)
    _emit_event_in_tx(session, owner_id, None, plan.get("source_instance_id"), job_id, "cancelled", reason)


def cancel_plan(
    repo,
    owner_id: str,
    plan_id: str,
    *,
    scheduler=None,
    clock=time.time,
    reason: str = "用户取消自动恢复",
) -> dict[str, Any]:
    """Idempotently cancel an active recovery plan following §7(3) safe-cancel rules."""
    store = repo.store if hasattr(repo, "store") else repo
    now = float(clock())
    job_to_cancel: str | None = None

    with store.dispatch_lock() as acquired:
        if not acquired:
            raise APIError("SCHEDULER_BUSY", "调度正在同步，请稍后重试", 503, True)
        with store.db_lock(), store.Session.begin() as session:
            store.begin_immediate(session)
            plan = get_plan_in_tx(session, owner_id, plan_id)
            if plan is None:
                raise APIError("NOT_FOUND", "自动恢复计划不存在", 404)
            state = plan.get("state")
            if state in TERMINAL_PLAN_STATES:
                return public_plan(plan)

            if state == "scheduled":
                _cancel_scheduled_plan_in_tx(session, owner_id, plan, now, reason)
                return public_plan(plan)

            if state == "restoring_env":
                plan["state"] = "cancelled"
                plan["cancelled_at"] = now
                plan["updated_at"] = now
                plan["last_error_message"] = reason
                ns = plan.get("native_snapshot")
                if isinstance(ns, dict) and isinstance(ns.get("model_restore_state"), dict):
                    ns["model_restore_state"] = {**ns["model_restore_state"], "state": "cancelled", "updated_at": now}
                save_plan_in_tx(session, owner_id, plan)
                _emit_event_in_tx(session, owner_id, None, plan.get("successor_instance_id"), None, "cancelled", "已停止后续本地模型回拉，保留当前就绪实例")
                return public_plan(plan)

            # dispatching or cancelling: safely cancel inflight job and wait for lease deletion confirmation
            active_job_id = plan.get("active_job_id")
            active_lease_id = plan.get("active_lease_id")
            job_row = session.get(GpuJobRow, active_job_id) if active_job_id else None
            job_terminal = job_row is None or job_row.state in TERMINAL_JOB_STATES
            if job_row is not None and not job_terminal:
                jdata = copy.deepcopy(job_row.data or {})
                claim_live = bool(jdata.get("claim_token")) and float(jdata.get("claim_until") or 0) > now
                can_short_circuit = (
                    job_row.state == "waiting_for_capacity"
                    and not jdata.get("lease_id")
                    and not jdata.get("instance_id")
                    and (bool(jdata.get("recovery_hold")) or not claim_live)
                )
                if can_short_circuit:
                    with store.admission_lock(timeout=0.5) as a_acquired:
                        if a_acquired:
                            _cancel_held_job_in_tx(session, owner_id, active_job_id, state="cancelled", message=reason)
                            job_terminal = True
                        else:
                            can_short_circuit = False
                if not can_short_circuit:
                    jdata["cancel_requested"] = True
                    jdata["requested_outcome"] = "cancelled"
                    job_row.data = jdata
                    job_row.version = int(job_row.version or 0) + 1
                    job_row.updated_at = utcnow()
                    job_to_cancel = active_job_id

            lease_deleted = True
            if active_lease_id:
                lrow = session.get(GpuLeaseRow, active_lease_id)
                if lrow is not None and lrow.state != "deleted":
                    lease_deleted = False
            if lease_deleted and active_job_id:
                for lrow in session.scalars(select(GpuLeaseRow).where(GpuLeaseRow.owner_id == owner_id)).all():
                    if (lrow.data or {}).get("owner_job_id") == active_job_id and lrow.state != "deleted":
                        lease_deleted = False
                        plan["active_lease_id"] = lrow.id
                        break

            if job_terminal and lease_deleted:
                plan["state"] = "cancelled"
                plan["cancelled_at"] = now
                plan["next_retry_at"] = None
            else:
                plan["state"] = "cancelling"
                plan["next_retry_at"] = None
            plan["last_error_message"] = reason
            plan["updated_at"] = now
            save_plan_in_tx(session, owner_id, plan)
            result = public_plan(plan)

    if job_to_cancel and scheduler is not None:
        try:
            scheduler.cancel(owner_id, job_to_cancel)
        except Exception:
            pass
    return result


def cancel_plans_for_instance_in_tx(session, owner_id: str, instance_id: str, now: float, *, reason: str = "用户手动停止实例，自动取消恢复计划") -> None:
    """Cancel or transition active recovery plans tied to a manually stopped instance."""
    for plan in list_plans_in_tx(session, owner_id):
        if plan.get("state") not in ACTIVE_PLAN_STATES:
            continue
        if instance_id not in {plan.get("source_instance_id"), plan.get("successor_instance_id")}:
            continue
        if plan["state"] in {"scheduled", "restoring_env"}:
            _cancel_held_job_in_tx(session, owner_id, plan.get("active_job_id"), state="cancelled", message=reason)
            plan["state"] = "cancelled"
            plan["cancelled_at"] = now
        elif plan["state"] == "dispatching":
            plan["state"] = "cancelling"
        plan["next_retry_at"] = None
        plan["last_error_message"] = reason
        plan["updated_at"] = now
        save_plan_in_tx(session, owner_id, plan)


def cancel_plans_for_job(repo, owner_id: str, job_id: str, now: float, *, reason: str = "用户取消任务，自动停止恢复计划") -> None:
    """Cancel or transition any active recovery plan associated with a manually cancelled job."""
    store = repo.store if hasattr(repo, "store") else repo
    with _wait_dispatch_lock(store):
        with store.db_lock(), store.Session.begin() as session:
            store.begin_immediate(session)
            job_row = session.get(GpuJobRow, job_id)
            job_terminal = job_row is None or job_row.state in TERMINAL_JOB_STATES
            for plan in list_plans_in_tx(session, owner_id):
                if plan.get("state") not in ACTIVE_PLAN_STATES:
                    continue
                if plan.get("active_job_id") != job_id:
                    continue
                if plan["state"] == "cancelling":
                    continue
                if plan["state"] == "scheduled":
                    plan["state"] = "cancelled"
                    plan["cancelled_at"] = now
                elif plan["state"] == "dispatching":
                    # Check if job is non-terminal or any associated lease is non-deleted
                    has_live_lease = False
                    active_lease_id = plan.get("active_lease_id")
                    if active_lease_id:
                        lrow = session.get(GpuLeaseRow, active_lease_id)
                        if lrow is not None and lrow.state != "deleted":
                            has_live_lease = True
                    if not has_live_lease:
                        for lrow in session.scalars(select(GpuLeaseRow).where(GpuLeaseRow.owner_id == owner_id)).all():
                            if (lrow.data or {}).get("owner_job_id") == job_id and lrow.state != "deleted":
                                has_live_lease = True
                                plan["active_lease_id"] = lrow.id
                                break
                    if has_live_lease or not job_terminal:
                        plan["state"] = "cancelling"
                    else:
                        plan["state"] = "cancelled"
                        plan["cancelled_at"] = now
                plan["next_retry_at"] = None
                plan["last_error_message"] = reason
                plan["updated_at"] = now
                save_plan_in_tx(session, owner_id, plan)


def cancel_scheduled_native_plans_in_tx(session, owner_id: str, now: float, *, reason: str = "用户手动启动原生 ComfyUI，已取消等待中的自动恢复计划") -> None:
    """Cancel any scheduled/restoring_env native recovery plan when user manually starts native ComfyUI."""
    for plan in list_plans_in_tx(session, owner_id):
        if plan.get("pool") != POOL_NATIVE:
            continue
        if plan.get("state") in {"scheduled", "restoring_env"}:
            plan["state"] = "cancelled"
            plan["cancelled_at"] = now
            plan["next_retry_at"] = None
            plan["last_error_message"] = reason
            plan["updated_at"] = now
            save_plan_in_tx(session, owner_id, plan)


# ---- Native local model background restore integration ----------------------------

def _trigger_native_model_restore(repo, owner_id: str, plan: dict[str, Any], lease: dict[str, Any], now: float) -> tuple[str | None, dict[str, Any]]:
    """Send POST /studio_native/models/restore to the ready successor native VM."""
    from app import native_model_api

    from . import native

    ns = plan.get("native_snapshot") or {}
    models = list(ns.get("recoverable_models") or [])
    if not models:
        return None, {"state": "ready", "total": 0, "completed": 0, "failed": 0, "skipped": 0, "partial_disk_limit": False, "updated_at": now}
    if native.restart_active(lease):
        raise APIError("NATIVE_RESTARTING", "原生 ComfyUI 正在重启，等待重启完成后再回拉模型", 409, True)
    url, auth = native.backend_for(repo, owner_id)
    channel_headers = native_model_api.headers(url, auth, lease, repo=repo)
    context = native_model_api.runtime_context(repo, lease)
    payload = {**context, "models": models}
    res = native_model_api.rpc_sync("POST", url, channel_headers, "/restore", body=payload, timeout=30)
    ident = res.get("id")
    state = {
        "state": "preparing",
        "total": len(models),
        "completed": 0,
        "failed": 0,
        "skipped": 0,
        "partial_disk_limit": False,
        "updated_at": now,
    }
    return ident, state


def _poll_native_model_restore(repo, owner_id: str, lease: dict[str, Any], restore_job_id: str, now: float) -> dict[str, Any] | None:
    from app import native_model_api

    from . import native

    try:
        url, auth = native.backend_for(repo, owner_id)
        channel_headers = native_model_api.headers(url, auth, lease, repo=repo)
        res = native_model_api.rpc_sync("GET", url, channel_headers, "/prepare/" + restore_job_id, timeout=15)
    except Exception:
        return None
    models = res.get("models") if isinstance(res.get("models"), list) else []
    total = len(models)
    completed = sum(1 for m in models if isinstance(m, dict) and m.get("state") == "ready")
    skipped = sum(1 for m in models if isinstance(m, dict) and m.get("state") == "skipped_disk_limit")
    failed = sum(1 for m in models if isinstance(m, dict) and m.get("state") == "failed")
    disk_limit = bool(res.get("partial_disk_limit")) or skipped > 0 or any(
        isinstance(m, dict) and "5 GiB" in str(m.get("error") or "") for m in models
    )
    raw_state = str(res.get("state") or "preparing")
    if raw_state in {"ready", "partial_disk_limit", "failed"}:
        final_state = "partial_disk_limit" if disk_limit else raw_state
    else:
        final_state = "preparing"
    return {
        "state": final_state,
        "total": total,
        "completed": completed,
        "failed": failed,
        "skipped": skipped,
        "partial_disk_limit": disk_limit,
        "updated_at": now,
    }


# ---- Orchestrator tick -------------------------------------------------------------

def _append_history(plan: dict[str, Any], outcome: str, now: float) -> None:
    history = list(plan.get("history") or [])
    history.append({
        "round": int(plan.get("rounds_used") or 0),
        "from_instance_id": plan.get("source_instance_id"),
        "to_instance_id": plan.get("successor_instance_id"),
        "triggered_at": plan.get("round_started_at"),
        "recovered_at": now,
        "trigger_reason": plan.get("last_trigger_reason"),
        "outcome": outcome,
    })
    plan["history"] = history[-20:]


def _dispatch_native_locked(repo, owner_id: str, plan: dict[str, Any], now: float) -> dict[str, Any]:
    from . import native

    ns = plan.get("native_snapshot") or {}
    storage = native.model_storage(ns.get("model_storage") or {})
    retain_minutes = max(1, int(round(float(plan.get("retain_idle_minutes") or settings.native_comfy_idle_minutes))))
    hard_deadline = float(plan["hard_deadline_at"])
    # Estimate integer max_runtime_minutes for validation only; exact deadline is pinned via extra_job_fields.
    approx_runtime = max(1, int((hard_deadline - now) // 60) + 1)
    body: dict[str, Any] = {
        "account_id": plan["account_id"],
        "spec": {"key": plan["requested_spec"]},
        "retain_idle_minutes": retain_minutes,
        "max_runtime_minutes": approx_runtime,
        **storage,
    }
    if ns.get("plugin_generation"):
        body["plugin_generation"] = ns["plugin_generation"]
    idem_key = f"auto-recovery:{plan['plan_id']}:r{int(plan.get('rounds_used') or 0) + 1}:a{int(plan.get('backoff_attempt') or 0)}:d{int(plan.get('dispatch_seq') or 0)}"
    extra = {
        "hard_deadline_at": hard_deadline,
        "deadline_epoch": hard_deadline,
        "native_session_deadline_epoch": hard_deadline,
        "recovery_group_id": plan["recovery_group_id"],
        "recovery_plan_id": plan["plan_id"],
        "recovery_generation": int(plan.get("rounds_used") or 0) + 1,
        "parent_instance_id": plan.get("source_instance_id"),
        "recovery_source": "auto_recovery",
        "recovery_was_busy": bool(plan.get("was_busy")),
        "recovery_orig_idle_deadline_at": plan.get("orig_idle_deadline_at"),
    }
    return native._admit_locked(
        repo,
        owner_id,
        body,
        idem_key,
        extra_job_fields=extra,
        cancel_recovery_plan=False,
    )


def _dispatch_creation_warmup_locked(repo, owner_id: str, plan: dict[str, Any], now: float) -> dict[str, Any]:
    from app import creation_api
    from app.admission import admit_job
    from app.job_schemas import JobInput, Policy

    cs = plan.get("creation_snapshot") or {}
    wf_id = cs.get("workflow_id")
    wf_key = cs.get("workflow_key")
    wf_ver = cs.get("workflow_version")
    eligible_wfs = creation_api.eligible(repo, owner_id)
    wf = next(
        (
            row
            for row in eligible_wfs
            if (wf_id and row["id"] == wf_id)
            or (wf_key and row["workflow_key"] == wf_key and str(row["workflow_version"]) == str(wf_ver))
        ),
        None,
    )
    if wf is None:
        raise APIError("WORKFLOW_UNPUBLISHED", "发布版本已撤销或不可执行", 409)

    project_id = cs.get("project_id")
    if not project_id or not repo.store.get("projects", project_id, owner_id):
        ident = creation_api.default_project_id(owner_id)
        if not repo.store.get("projects", ident, owner_id):
            repo.store.create("projects", owner_id, {"title": "创作 API"}, record_id=ident)
        project_id = ident

    raw_hard = plan.get("hard_deadline_at")
    if raw_hard is None:
        raw_hard = plan.get("orig_idle_deadline_at")
    if raw_hard is None:
        raise APIError("AUTO_RECOVERY_INVALID_DEADLINE", "缺少有效的恢复截止时间", 409)
    hard_deadline = float(raw_hard)
    retain_minutes = max(1, int(round(float(plan.get("retain_idle_minutes") or 15))))
    approx_runtime = max(1, min(240, int((hard_deadline - now) // 60) + 1))
    lifecycle = str(plan.get("retention_mode") or "smart")
    if lifecycle not in {"smart", "retain"}:
        lifecycle = "smart"
    pol = Policy.model_validate({
        "provider": "colab",
        "account_id": plan["account_id"],
        "compute_spec": plan["requested_spec"],
        "lifecycle": lifecycle,
        "retain_idle_minutes": retain_minutes,
        "max_runtime_minutes": approx_runtime,
        "reuse_action": "new",
    }).model_dump()
    raw = JobInput(
        project_id=project_id,
        model_id=wf["model_id"],
        workflow_key=wf["workflow_key"],
        workflow_version=wf["workflow_version"],
        params={},
        execution_policy=pol,
    ).model_dump()
    idem_key = f"auto-recovery-warmup:{plan['plan_id']}:r{int(plan.get('rounds_used') or 0) + 1}:a{int(plan.get('backoff_attempt') or 0)}:d{int(plan.get('dispatch_seq') or 0)}"
    extra = {
        "hard_deadline_at": hard_deadline,
        "deadline_epoch": hard_deadline,
        "recovery_group_id": plan["recovery_group_id"],
        "recovery_plan_id": plan["plan_id"],
        "recovery_generation": int(plan.get("rounds_used") or 0) + 1,
        "parent_instance_id": plan.get("source_instance_id"),
        "recovery_source": "auto_recovery",
        "recovery_was_busy": False,
        "recovery_orig_idle_deadline_at": plan.get("orig_idle_deadline_at"),
        "auto_recovery_warmup": True,
    }
    return admit_job(
        repo.store,
        {"id": owner_id},
        raw,
        idem_key,
        creation={
            "creation_key_id": None,
            "creation_workflow_id": wf["id"],
            "creation_existing_only": False,
            "creation_body_hash": None,
            "warmup": True,
        },
        _lock_held=True,
        extra_job_fields=extra,
    )


def _advance_cancelling(repo, owner_id: str, plan: dict[str, Any], now: float, *, scheduler=None) -> bool:
    store = repo.store if hasattr(repo, "store") else repo
    active_job_id = plan.get("active_job_id")
    active_lease_id = plan.get("active_lease_id")

    job = repo.get("jobs", active_job_id, owner_id) if active_job_id else None
    if job and job.get("state") not in TERMINAL_JOB_STATES and not job.get("cancel_requested"):
        if scheduler is not None:
            try:
                scheduler.cancel(owner_id, active_job_id)
            except Exception:
                pass
        else:
            repo.update("jobs", active_job_id, {"cancel_requested": True, "requested_outcome": "cancelled"}, owner_id)
        return False

    job_terminal = job is None or job.get("state") in TERMINAL_JOB_STATES
    lease_deleted = True
    if active_lease_id:
        lease = repo.get("leases", active_lease_id, owner_id)
        if lease is not None and lease.get("state") != "deleted":
            lease_deleted = False
    if lease_deleted and active_job_id:
        for lease in repo.list("leases", owner_id):
            if lease.get("owner_job_id") == active_job_id and lease.get("state") != "deleted":
                lease_deleted = False
                active_lease_id = lease["id"]
                break

    if not (job_terminal and lease_deleted):
        if active_lease_id and active_lease_id != plan.get("active_lease_id"):
            with store.db_lock(), store.Session.begin() as session:
                store.begin_immediate(session)
                fresh = get_plan_in_tx(session, owner_id, plan["plan_id"])
                if fresh and fresh.get("state") == "cancelling":
                    fresh["active_lease_id"] = active_lease_id
                    fresh["updated_at"] = now
                    save_plan_in_tx(session, owner_id, fresh)
        return False

    with store.dispatch_lock() as acquired:
        if not acquired:
            return False
        with store.db_lock(), store.Session.begin() as session:
            store.begin_immediate(session)
            fresh = get_plan_in_tx(session, owner_id, plan["plan_id"])
            if not fresh or fresh.get("state") != "cancelling":
                return False
            fresh["state"] = "cancelled"
            fresh["cancelled_at"] = now
            fresh["updated_at"] = now
            save_plan_in_tx(session, owner_id, fresh)
            _emit_event_in_tx(
                session,
                owner_id,
                None,
                fresh.get("successor_instance_id") or fresh.get("source_instance_id"),
                active_job_id,
                "cancelled",
                "在途恢复分配已确认释放，自动恢复已安全取消",
            )
            return True


def _advance_dispatching(repo, owner_id: str, plan: dict[str, Any], now: float, cfg: RecoveryConfig) -> bool:
    store = repo.store if hasattr(repo, "store") else repo
    active_job_id = plan.get("active_job_id")
    job = repo.get("jobs", active_job_id, owner_id) if active_job_id else None
    if job is None:
        with store.dispatch_lock() as acquired:
            if not acquired:
                return False
            with store.db_lock(), store.Session.begin() as session:
                store.begin_immediate(session)
                fresh = get_plan_in_tx(session, owner_id, plan["plan_id"])
                if fresh and fresh.get("state") == "dispatching":
                    fresh["state"] = "aborted_fatal"
                    fresh["last_error_message"] = "继任启动任务丢失"
                    fresh["updated_at"] = now
                    save_plan_in_tx(session, owner_id, fresh)
                    return True
        return False

    # Locate lease associated with this dispatch job
    lease = None
    if job.get("lease_id"):
        lease = repo.get("leases", job["lease_id"], owner_id)
    elif plan.get("active_lease_id"):
        lease = repo.get("leases", plan["active_lease_id"], owner_id)
    if lease is None:
        for row in repo.list("leases", owner_id):
            if row.get("owner_job_id") == job["id"] and (
                not job.get("attempt_id") or row.get("attempt_id") == job.get("attempt_id")
            ):
                lease = row
                break

    if lease and lease.get("id") != plan.get("active_lease_id"):
        bind_active_lease_for_plan(repo, owner_id, plan["plan_id"], lease["id"], now)

    job_state = job.get("state")
    if job_state not in TERMINAL_JOB_STATES:
        # Check if user manually stopped successor or cancelled job
        inst_id = job.get("instance_id") or plan.get("successor_instance_id")
        inst = repo.get("instances", inst_id, owner_id) if inst_id else None
        if job.get("cancel_requested") or (inst and inst.get("manual_stop_requested")):
            with store.dispatch_lock() as acquired:
                if acquired:
                    with store.db_lock(), store.Session.begin() as session:
                        store.begin_immediate(session)
                        fresh = get_plan_in_tx(session, owner_id, plan["plan_id"])
                        if fresh and fresh.get("state") == "dispatching":
                            fresh["state"] = "cancelling"
                            fresh["updated_at"] = now
                            save_plan_in_tx(session, owner_id, fresh)
                            return True
        return False

    # Terminal job states
    if job_state == "completed":
        successor_id = job.get("instance_id") or plan.get("successor_instance_id")
        if successor_id:
            on_successor_allocation_confirmed(repo, owner_id, plan["plan_id"], successor_id, (lease or {}).get("id"), now)
        # Native local mode: check if recoverable_models need background restore
        ns = plan.get("native_snapshot") or {}
        is_native_local = (
            plan.get("pool") == POOL_NATIVE
            and (ns.get("model_storage") or {}).get("model_storage_mode") == "local"
            and bool(ns.get("recoverable_models"))
        )
        restore_job_id = None
        restore_state = None
        if is_native_local and lease and lease.get("native_ready"):
            try:
                restore_job_id, restore_state = _trigger_native_model_restore(repo, owner_id, plan, lease, now)
            except Exception as exc:
                log.warning("native model restore trigger failed: %s", type(exc).__name__)
                restore_state = {
                    "state": "pending",
                    "total": len(ns.get("recoverable_models") or []),
                    "completed": 0,
                    "failed": 0,
                    "skipped": 0,
                    "partial_disk_limit": False,
                    "updated_at": now,
                }
        with store.dispatch_lock() as acquired:
            if not acquired:
                return False
            with store.db_lock(), store.Session.begin() as session:
                store.begin_immediate(session)
                fresh = get_plan_in_tx(session, owner_id, plan["plan_id"])
                if not fresh or fresh.get("state") != "dispatching":
                    return False
                if successor_id and fresh.get("round_counted_for_instance_id") != successor_id:
                    fresh["rounds_used"] = int(fresh.get("rounds_used") or 0) + 1
                    fresh["round_counted_for_instance_id"] = successor_id
                    parent_id = fresh.get("source_instance_id")
                    if parent_id and parent_id != successor_id:
                        parent_row = session.get(GpuInstanceRow, parent_id)
                        if parent_row is not None and parent_row.owner_id == owner_id:
                            pdata = copy.deepcopy(parent_row.data or {})
                            if pdata.get("replaced_by_instance_id") != successor_id:
                                pdata["replaced_by_instance_id"] = successor_id
                                pdata.setdefault("recovery_group_id", fresh["recovery_group_id"])
                                parent_row.data = pdata
                                parent_row.version = int(parent_row.version or 0) + 1
                                parent_row.updated_at = utcnow()
                fresh["successor_instance_id"] = successor_id or fresh.get("successor_instance_id")
                fresh["active_lease_id"] = (lease or {}).get("id") or fresh.get("active_lease_id")
                fresh["last_error_code"] = None
                fresh["last_error_message"] = None
                if is_native_local:
                    fresh["state"] = "restoring_env"
                    fresh["model_restore_started_at"] = now
                    fresh["model_restore_last_progress_at"] = now
                    fresh_ns = dict(fresh.get("native_snapshot") or {})
                    fresh_ns["model_restore_job_id"] = restore_job_id
                    fresh_ns["model_restore_state"] = restore_state
                    fresh["native_snapshot"] = fresh_ns
                else:
                    fresh["state"] = "recovered"
                    _append_history(fresh, "recovered", now)
                fresh["updated_at"] = now
                save_plan_in_tx(session, owner_id, fresh)
                if fresh.get("pool") == POOL_NATIVE and (fresh.get("native_snapshot") or {}).get("wild_models"):
                    wild = fresh["native_snapshot"]["wild_models"]
                    _emit_event_in_tx(
                        session,
                        owner_id,
                        None,
                        fresh.get("successor_instance_id"),
                        job["id"],
                        fresh["state"],
                        f"以下 {len(wild)} 个模型上次未登记在 Studio 模型库，无法自动回拉：{', '.join(wild[:10])}",
                    )
                return True

    if job_state == "cancelled":
        if lease and lease.get("state") != "deleted":
            return False
        with store.dispatch_lock() as acquired:
            if not acquired:
                return False
            with store.db_lock(), store.Session.begin() as session:
                store.begin_immediate(session)
                fresh = get_plan_in_tx(session, owner_id, plan["plan_id"])
                if not fresh or fresh.get("state") != "dispatching":
                    return False
                if job.get("error_code") == "CU_BALANCE_FLOOR_REACHED":
                    fresh["state"] = "aborted_fatal"
                    fresh["last_error_code"] = "CU_BALANCE_FLOOR_REACHED"
                    fresh["last_error_message"] = job.get("error_message") or "触发余额底线强制熔断"
                else:
                    fresh["state"] = "cancelled"
                    fresh["cancelled_at"] = now
                fresh["active_lease_id"] = None
                fresh["updated_at"] = now
                save_plan_in_tx(session, owner_id, fresh)
                return True

    # job_state == "failed": MUST confirm lease is deleted before ANY retry! (§5.5 anti-double-GPU law)
    if lease and lease.get("state") != "deleted":
        return False
    # Also check any non-deleted lease for this job_id
    for row in repo.list("leases", owner_id):
        if row.get("owner_job_id") == job["id"] and row.get("state") != "deleted":
            return False

    err_code = job.get("error_code")
    err_msg = job.get("error_message")
    err_details = job.get("error_details")
    kind = classify_failure(err_code, err_msg, err_details)

    with store.dispatch_lock() as acquired:
        if not acquired:
            return False
        with store.db_lock(), store.Session.begin() as session:
            store.begin_immediate(session)
            fresh = get_plan_in_tx(session, owner_id, plan["plan_id"])
            if not fresh or fresh.get("state") != "dispatching":
                return False
            fresh["active_job_id"] = None
            fresh["active_lease_id"] = None
            fresh["last_error_code"] = err_code

            if kind == "fatal":
                fresh["state"] = "aborted_fatal"
                fresh["last_error_message"] = err_msg or f"遇到致命错误（{err_code}），已停止自动恢复"
                fresh["updated_at"] = now
                save_plan_in_tx(session, owner_id, fresh)
                _emit_event_in_tx(
                    session, owner_id, None, fresh.get("source_instance_id"), job["id"],
                    "aborted_fatal", f"自动恢复因致命错误中止：{fresh['last_error_message']}",
                )
                return True

            round_started = float(fresh.get("round_started_at") or now)
            if now - round_started >= cfg.round_timeout_seconds:
                fresh["state"] = "exhausted"
                fresh["last_error_message"] = f"单轮自动恢复退避超时（{cfg.round_timeout_seconds}s）：{err_msg or err_code}"
                fresh["updated_at"] = now
                save_plan_in_tx(session, owner_id, fresh)
                return True

            if kind == "local_lock":
                next_attempt = int(fresh.get("backoff_attempt") or 0)
                delay = 5.0
            else:
                next_attempt = int(fresh.get("backoff_attempt") or 0) + 1
                delay = _compute_backoff_delay(cfg, next_attempt, kind)
            next_retry_at = now + delay
            if next_retry_at - round_started > cfg.round_timeout_seconds:
                if (round_started + cfg.round_timeout_seconds) - now < 5.0:
                    fresh["state"] = "exhausted"
                    fresh["last_error_message"] = f"单轮自动恢复退避超时（{cfg.round_timeout_seconds}s）：{err_msg or err_code}"
                    fresh["updated_at"] = now
                    save_plan_in_tx(session, owner_id, fresh)
                    return True
                next_retry_at = round_started + cfg.round_timeout_seconds

            eligible, elig_code, elig_msg = evaluate_recovery_eligibility_in_tx(
                repo,
                session,
                owner_id=owner_id,
                pool=str(fresh["pool"]),
                account_id=fresh.get("account_id"),
                retention_mode=fresh.get("retention_mode"),
                hard_deadline_at=fresh.get("hard_deadline_at"),
                orig_idle_deadline_at=fresh.get("orig_idle_deadline_at"),
                was_busy=bool(fresh.get("was_busy")),
                rounds_used=int(fresh.get("rounds_used") or 0),
                now=next_retry_at,
                cfg=cfg,
                check_concurrency=False,
            )
            if not eligible and elig_code != "startup_grace":
                fresh["state"] = "exhausted" if elig_code == "exhausted" else "aborted_fatal"
                fresh["last_error_code"] = elig_code
                fresh["last_error_message"] = elig_msg
                fresh["updated_at"] = now
                save_plan_in_tx(session, owner_id, fresh)
                return True

            fresh["state"] = "scheduled"
            fresh["backoff_attempt"] = next_attempt
            fresh["dispatch_seq"] = int(fresh.get("dispatch_seq") or 0) + 1
            fresh["next_retry_at"] = next_retry_at
            fresh["last_error_message"] = f"{err_msg or err_code}；{int(round(next_retry_at - now))}s 后重试"
            fresh["updated_at"] = now
            save_plan_in_tx(session, owner_id, fresh)
            return True


def _advance_restoring_env(repo, owner_id: str, plan: dict[str, Any], now: float, cfg: RecoveryConfig) -> bool:
    store = repo.store if hasattr(repo, "store") else repo
    succ_id = plan.get("successor_instance_id")
    inst = repo.get("instances", succ_id, owner_id) if succ_id else None
    if inst and inst.get("manual_stop_requested"):
        with store.dispatch_lock() as acquired:
            if acquired:
                with store.db_lock(), store.Session.begin() as session:
                    store.begin_immediate(session)
                    fresh = get_plan_in_tx(session, owner_id, plan["plan_id"])
                    if fresh and fresh.get("state") == "restoring_env":
                        fresh["state"] = "cancelled"
                        fresh["cancelled_at"] = now
                        fresh["updated_at"] = now
                        save_plan_in_tx(session, owner_id, fresh)
                        return True
        return False

    def _finish_restoring_env(state: str, code: str, message: str) -> bool:
        with store.dispatch_lock() as acquired:
            if not acquired:
                return False
            with store.db_lock(), store.Session.begin() as session:
                store.begin_immediate(session)
                fresh = get_plan_in_tx(session, owner_id, plan["plan_id"])
                if not fresh or fresh.get("state") != "restoring_env" or any(
                    fresh.get(key) != plan.get(key)
                    for key in ("successor_instance_id", "active_lease_id", "model_restore_started_at")
                ):
                    return False
                snapshot = dict(fresh.get("native_snapshot") or {})
                snapshot["model_restore_state"] = {
                    **(snapshot.get("model_restore_state") or {}), "state": "failed", "updated_at": now,
                }
                fresh.update(
                    state=state,
                    last_error_code=code,
                    last_error_message=message,
                    updated_at=now,
                    native_snapshot=snapshot,
                )
                _append_history(fresh, state, now)
                save_plan_in_tx(session, owner_id, fresh)
                _emit_event_in_tx(session, owner_id, None, succ_id, fresh.get("active_job_id"), state, message)
                return True

    lease_id = plan.get("active_lease_id") or (inst or {}).get("lease_id")
    lease = repo.get("leases", lease_id, owner_id) if lease_id else None
    lease_lost = (
        not lease
        or lease.get("state") in {"deleted", "quarantined"}
        or (inst is not None and inst.get("state") == "deleted")
    )
    if lease_lost:
        return _finish_restoring_env("aborted_fatal", "model_restore_lease_lost", "继任实例租约已结束，本地模型回拉中止")

    ns = dict(plan.get("native_snapshot") or {})
    prev_state = ns.get("model_restore_state") if isinstance(ns.get("model_restore_state"), dict) else {}
    last_progress_at = (
        plan.get("model_restore_last_progress_at")
        or plan.get("model_restore_started_at")
        or plan.get("round_started_at")
        or plan.get("created_at")
        or now
    )
    hard_deadline = plan.get("hard_deadline_at")
    hard_expired = hard_deadline is not None and now >= float(hard_deadline)
    poll_timed_out = now >= float(last_progress_at) + cfg.round_timeout_seconds
    stall_timeout_seconds = max(float(cfg.round_timeout_seconds), 1800.0)

    restore_job_id = ns.get("model_restore_job_id")
    if not restore_job_id:
        if not ns.get("recoverable_models"):
            with store.dispatch_lock() as acquired:
                if acquired:
                    with store.db_lock(), store.Session.begin() as session:
                        store.begin_immediate(session)
                        fresh = get_plan_in_tx(session, owner_id, plan["plan_id"])
                        if fresh and fresh.get("state") == "restoring_env":
                            fresh["state"] = "recovered"
                            _append_history(fresh, "recovered", now)
                            fresh["updated_at"] = now
                            save_plan_in_tx(session, owner_id, fresh)
                            return True
            return False
        try:
            restore_job_id, restore_state = _trigger_native_model_restore(repo, owner_id, plan, lease, now)
        except Exception:
            if poll_timed_out or hard_expired:
                return _finish_restoring_env("exhausted", "model_restore_timeout", "本地模型回拉超过恢复时限")
            return False
        with store.dispatch_lock() as acquired:
            if acquired:
                with store.db_lock(), store.Session.begin() as session:
                    store.begin_immediate(session)
                    fresh = get_plan_in_tx(session, owner_id, plan["plan_id"])
                    if fresh and fresh.get("state") == "restoring_env":
                        fresh_ns = dict(fresh.get("native_snapshot") or {})
                        fresh_ns["model_restore_job_id"] = restore_job_id
                        fresh_ns["model_restore_state"] = restore_state
                        fresh["native_snapshot"] = fresh_ns
                        fresh["model_restore_last_progress_at"] = now
                        fresh["updated_at"] = now
                        save_plan_in_tx(session, owner_id, fresh)
                        return True
        return False

    progress = _poll_native_model_restore(repo, owner_id, lease, restore_job_id, now)
    if progress is None:
        if poll_timed_out or hard_expired:
            return _finish_restoring_env("exhausted", "model_restore_timeout", "本地模型回拉超过恢复时限")
        return False

    if progress["state"] not in {"ready", "partial_disk_limit", "failed"}:
        prev_done = (
            int(prev_state.get("completed") or 0)
            + int(prev_state.get("failed") or 0)
            + int(prev_state.get("skipped") or 0)
        )
        curr_done = (
            int(progress.get("completed") or 0)
            + int(progress.get("failed") or 0)
            + int(progress.get("skipped") or 0)
        )
        made_progress = curr_done > prev_done
        if not made_progress and (hard_expired or now >= float(last_progress_at) + stall_timeout_seconds):
            return _finish_restoring_env("exhausted", "model_restore_timeout", "本地模型回拉长时间无进展，已中止等待")
    else:
        made_progress = True

    with store.dispatch_lock() as acquired:
        if not acquired:
            return False
        with store.db_lock(), store.Session.begin() as session:
            store.begin_immediate(session)
            fresh = get_plan_in_tx(session, owner_id, plan["plan_id"])
            if not fresh or fresh.get("state") != "restoring_env":
                return False
            fresh_ns = dict(fresh.get("native_snapshot") or {})
            fresh_ns["model_restore_state"] = progress
            fresh["native_snapshot"] = fresh_ns
            if made_progress:
                fresh["model_restore_last_progress_at"] = now
            if progress["state"] in {"ready", "partial_disk_limit", "failed"}:
                fresh["state"] = "recovered"
                outcome = "partial_disk_limit" if progress.get("partial_disk_limit") else "recovered"
                _append_history(fresh, outcome, now)
                if progress.get("partial_disk_limit"):
                    _emit_event_in_tx(
                        session,
                        owner_id,
                        None,
                        fresh.get("successor_instance_id"),
                        None,
                        "recovered",
                        f"本地模型后台回拉触及 5 GiB 空间保护线，已完成 {progress['completed']}/{progress['total']} 个模型，跳过其余较早模型",
                    )
            fresh["updated_at"] = now
            save_plan_in_tx(session, owner_id, fresh)
            return True


def _advance_scheduled(
    repo,
    owner_id: str,
    plan: dict[str, Any],
    now: float,
    cfg: RecoveryConfig,
    *,
    process_started_at: float | None = None,
) -> bool:
    from . import native

    store = repo.store if hasattr(repo, "store") else repo
    if not cfg.enabled:
        with store.dispatch_lock() as acquired:
            if not acquired:
                return False
            with store.db_lock(), store.Session.begin() as session:
                store.begin_immediate(session)
                fresh = get_plan_in_tx(session, owner_id, plan["plan_id"])
                if not fresh or fresh.get("state") != "scheduled":
                    return False
                _cancel_scheduled_plan_in_tx(
                    session,
                    owner_id,
                    fresh,
                    now,
                    "跨机自动恢复已关闭，已取消等待中的恢复计划",
                    error_code="disabled",
                )
                return True
    next_retry_at = float(plan.get("next_retry_at") or 0.0)
    round_started = float(plan.get("round_started_at") or now)

    # Even before next_retry_at, enforce round timeout, CU fuses, remaining time, and account state.
    if now < next_retry_at:
        with store.dispatch_lock() as acquired:
            if not acquired:
                return False
            with store.db_lock(), store.Session.begin() as session:
                store.begin_immediate(session)
                fresh = get_plan_in_tx(session, owner_id, plan["plan_id"])
                if not fresh or fresh.get("state") != "scheduled":
                    return False
                if now - round_started >= cfg.round_timeout_seconds:
                    fresh["state"] = "exhausted"
                    fresh["last_error_message"] = f"单轮自动恢复等待超过 {cfg.round_timeout_seconds} 秒，已停止重试"
                    fresh["updated_at"] = now
                    _cancel_held_job_in_tx(
                        session, owner_id, fresh.get("active_job_id"), state="failed",
                        message=fresh["last_error_message"], error_code="auto_recovery_exhausted",
                    )
                    save_plan_in_tx(session, owner_id, fresh)
                    return True
                eligible, code, msg = evaluate_recovery_eligibility_in_tx(
                    repo,
                    session,
                    owner_id=owner_id,
                    pool=str(fresh["pool"]),
                    account_id=fresh.get("account_id"),
                    retention_mode=fresh.get("retention_mode"),
                    hard_deadline_at=fresh.get("hard_deadline_at"),
                    orig_idle_deadline_at=fresh.get("orig_idle_deadline_at"),
                    was_busy=bool(fresh.get("was_busy")),
                    rounds_used=int(fresh.get("rounds_used") or 0),
                    now=now,
                    cfg=cfg,
                    last_successful_probe_at=fresh.get("last_successful_probe_at"),
                    process_started_at=process_started_at,
                    check_vps_outage=fresh.get("reclaimed_at") is not None,
                    vps_outage_now=fresh.get("reclaimed_at"),
                    check_concurrency=True,
                    ignore_job_id=fresh.get("active_job_id"),
                    source_instance_id=fresh.get("source_instance_id"),
                )
                if not eligible and code not in {"startup_grace", "inflight_pending"}:
                    if code == "concurrent_active":
                        fresh["state"] = "cancelled"
                        fresh["cancelled_at"] = now
                    elif code == "aborted_vps_outage":
                        fresh["state"] = "aborted_vps_outage"
                    elif code == "exhausted":
                        fresh["state"] = "exhausted"
                    else:
                        fresh["state"] = "aborted_fatal"
                    fresh["last_error_code"] = code
                    fresh["last_error_message"] = msg
                    fresh["updated_at"] = now
                    _cancel_held_job_in_tx(
                        session, owner_id, fresh.get("active_job_id"),
                        state="cancelled" if fresh["state"] == "cancelled" else "failed",
                        message=msg, error_code=code,
                    )
                    save_plan_in_tx(session, owner_id, fresh)
                    return True
        return False

    # Pre-warm workflow registry bootstrap OUTSIDE dispatch_lock/admission_lock to avoid nested admission_lock!
    if plan.get("pool") == POOL_CREATION and (plan.get("creation_snapshot") or {}).get("mode") != "retry_job":
        try:
            from app.admission import repo_registry

            repo_registry(store, {"id": owner_id})
        except Exception:
            return False

    # Due for dispatch: acquire dispatch_lock -> admission_lock in strict non-reentrant order (§5.4)
    with store.dispatch_lock() as d_acquired:
        if not d_acquired:
            return False
        with store.admission_lock(timeout=0.5) as a_acquired:
            if not a_acquired:
                return False

            # Check native workspace offline write lock symmetrically (§7(2))
            if plan.get("pool") == POOL_NATIVE and native.active_workspace_write_lock(repo, owner_id, now=now):
                with store.db_lock(), store.Session.begin() as session:
                    store.begin_immediate(session)
                    fresh = get_plan_in_tx(session, owner_id, plan["plan_id"])
                    if fresh and fresh.get("state") == "scheduled":
                        fresh["next_retry_at"] = now + 5.0
                        fresh["updated_at"] = now
                        save_plan_in_tx(session, owner_id, fresh)
                return False

            with store.db_lock(), store.Session.begin() as session:
                store.begin_immediate(session)
                fresh = get_plan_in_tx(session, owner_id, plan["plan_id"])
                if not fresh or fresh.get("state") != "scheduled":
                    return False

                if now - float(fresh.get("round_started_at") or now) >= cfg.round_timeout_seconds:
                    fresh["state"] = "exhausted"
                    fresh["last_error_message"] = f"单轮自动恢复等待超过 {cfg.round_timeout_seconds} 秒，已停止重试"
                    fresh["updated_at"] = now
                    _cancel_held_job_in_tx(
                        session, owner_id, fresh.get("active_job_id"), state="failed",
                        message=fresh["last_error_message"], error_code="auto_recovery_exhausted",
                    )
                    save_plan_in_tx(session, owner_id, fresh)
                    return True

                eligible, code, msg = evaluate_recovery_eligibility_in_tx(
                    repo,
                    session,
                    owner_id=owner_id,
                    pool=str(fresh["pool"]),
                    account_id=fresh.get("account_id"),
                    retention_mode=fresh.get("retention_mode"),
                    hard_deadline_at=fresh.get("hard_deadline_at"),
                    orig_idle_deadline_at=fresh.get("orig_idle_deadline_at"),
                    was_busy=bool(fresh.get("was_busy")),
                    rounds_used=int(fresh.get("rounds_used") or 0),
                    now=now,
                    cfg=cfg,
                    last_successful_probe_at=fresh.get("last_successful_probe_at"),
                    process_started_at=process_started_at,
                    check_vps_outage=fresh.get("reclaimed_at") is not None,
                    vps_outage_now=fresh.get("reclaimed_at"),
                    check_concurrency=True,
                    active_job_id=fresh.get("active_job_id"),
                    active_lease_id=fresh.get("active_lease_id"),
                    ignore_job_id=fresh.get("active_job_id") if (fresh.get("creation_snapshot") or {}).get("mode") == "retry_job" else None,
                    source_instance_id=fresh.get("source_instance_id"),
                )
                if code == "startup_grace":
                    started_at = _PROCESS_STARTED_AT if process_started_at is None else float(process_started_at)
                    fresh["next_retry_at"] = started_at + cfg.startup_grace_seconds
                    fresh["updated_at"] = now
                    save_plan_in_tx(session, owner_id, fresh)
                    return False
                if code == "inflight_pending":
                    return False
                if not eligible:
                    if code == "concurrent_active":
                        fresh["state"] = "cancelled"
                        fresh["cancelled_at"] = now
                    elif code == "aborted_vps_outage":
                        fresh["state"] = "aborted_vps_outage"
                    elif code == "exhausted":
                        fresh["state"] = "exhausted"
                    else:
                        fresh["state"] = "aborted_fatal"
                    fresh["last_error_code"] = code
                    fresh["last_error_message"] = msg
                    fresh["updated_at"] = now
                    _cancel_held_job_in_tx(
                        session, owner_id, fresh.get("active_job_id"),
                        state="cancelled" if fresh["state"] == "cancelled" else "failed",
                        message=msg, error_code=code,
                    )
                    save_plan_in_tx(session, owner_id, fresh)
                    return True

                # Path B Creation retry: release recovery_hold atomically in this TX
                if fresh["pool"] == POOL_CREATION and (fresh.get("creation_snapshot") or {}).get("mode") == "retry_job":
                    job_id = fresh.get("active_job_id")
                    job_row = session.get(GpuJobRow, job_id) if job_id else None
                    if job_row is None or job_row.owner_id != owner_id or job_row.state in TERMINAL_JOB_STATES:
                        fresh["state"] = "aborted_fatal"
                        fresh["last_error_message"] = "待重跑任务已结束或不存在"
                        fresh["updated_at"] = now
                        save_plan_in_tx(session, owner_id, fresh)
                        return True
                    jdata = copy.deepcopy(job_row.data or {})
                    if jdata.get("cancel_requested"):
                        fresh["state"] = "cancelled"
                        fresh["cancelled_at"] = now
                        fresh["updated_at"] = now
                        save_plan_in_tx(session, owner_id, fresh)
                        return True
                    jdata["recovery_hold"] = False
                    jdata["retry_not_before_epoch"] = 0
                    jdata["recovery_generation"] = int(fresh.get("rounds_used") or 0) + 1
                    job_row.data = jdata
                    job_row.version = int(job_row.version or 0) + 1
                    job_row.updated_at = utcnow()
                    fresh["state"] = "dispatching"
                    fresh["active_lease_id"] = None
                    fresh["updated_at"] = now
                    save_plan_in_tx(session, owner_id, fresh)
                    return True

                plan_copy = copy.deepcopy(fresh)

            # Still holding dispatch_lock + admission_lock (no nested lock acquisition!):
            # Admit successor job for native or creation Path A warmup.
            try:
                if plan_copy["pool"] == POOL_NATIVE:
                    new_job = _dispatch_native_locked(repo, owner_id, plan_copy, now)
                else:
                    new_job = _dispatch_creation_warmup_locked(repo, owner_id, plan_copy, now)
            except APIError as exc:
                kind = classify_failure(exc.code, exc.message)
                with store.db_lock(), store.Session.begin() as session:
                    store.begin_immediate(session)
                    fresh = get_plan_in_tx(session, owner_id, plan_copy["plan_id"])
                    if not fresh or fresh.get("state") != "scheduled":
                        return False
                    if kind == "local_lock":
                        fresh["dispatch_seq"] = int(fresh.get("dispatch_seq") or 0) + 1
                        fresh["next_retry_at"] = now + 5.0
                        fresh["last_error_code"] = exc.code
                        fresh["last_error_message"] = exc.message
                    elif kind == "fatal":
                        fresh["state"] = "aborted_fatal"
                        fresh["last_error_code"] = exc.code
                        fresh["last_error_message"] = exc.message
                    else:
                        next_attempt = int(fresh.get("backoff_attempt") or 0) + 1
                        delay = _compute_backoff_delay(cfg, next_attempt, kind)
                        fresh["backoff_attempt"] = next_attempt
                        fresh["next_retry_at"] = now + delay
                        fresh["last_error_code"] = exc.code
                        fresh["last_error_message"] = exc.message
                    fresh["updated_at"] = now
                    save_plan_in_tx(session, owner_id, fresh)
                return True
            except Exception as exc:
                with store.db_lock(), store.Session.begin() as session:
                    store.begin_immediate(session)
                    fresh = get_plan_in_tx(session, owner_id, plan_copy["plan_id"])
                    if fresh and fresh.get("state") == "scheduled":
                        fresh["state"] = "aborted_fatal"
                        fresh["last_error_code"] = "auto_recovery_dispatch_failed"
                        fresh["last_error_message"] = f"创建继任启动任务失败：{type(exc).__name__}"
                        fresh["updated_at"] = now
                        save_plan_in_tx(session, owner_id, fresh)
                return True

            with store.db_lock(), store.Session.begin() as session:
                store.begin_immediate(session)
                fresh = get_plan_in_tx(session, owner_id, plan_copy["plan_id"])
                if fresh and fresh.get("state") == "scheduled":
                    fresh["state"] = "dispatching"
                    fresh["active_job_id"] = new_job["id"]
                    fresh["active_lease_id"] = None
                    fresh["updated_at"] = now
                    save_plan_in_tx(session, owner_id, fresh)
            return True


def tick(
    repo,
    *,
    clock=time.time,
    scheduler=None,
    process_started_at: float | None = None,
) -> int:
    """Advance all active L3 recovery plans by one step. Returns number of progressed plans."""
    cfg = RecoveryConfig.from_env()
    plans = [p for p in list_plans(repo) if p.get("state") in ACTIVE_PLAN_STATES]
    if not plans:
        return 0
    now = float(clock())
    progressed = 0
    for plan in plans:
        owner_id = str(plan.get("owner_id") or "")
        state = plan.get("state")
        if not owner_id:
            continue
        try:
            if state == "cancelling":
                if _advance_cancelling(repo, owner_id, plan, now, scheduler=scheduler):
                    progressed += 1
            elif state == "dispatching":
                if _advance_dispatching(repo, owner_id, plan, now, cfg):
                    progressed += 1
            elif state == "restoring_env":
                if _advance_restoring_env(repo, owner_id, plan, now, cfg):
                    progressed += 1
            elif state == "scheduled":
                if _advance_scheduled(repo, owner_id, plan, now, cfg, process_started_at=process_started_at):
                    progressed += 1
        except Exception as exc:
            log.warning("recovery tick failed for plan %s: %s", plan.get("plan_id"), type(exc).__name__)
    return progressed


__all__ = [
    "ACTIVE_PLAN_STATES",
    "TERMINAL_PLAN_STATES",
    "NON_DETERMINISTIC_L2_ERRORS",
    "RecoveryConfig",
    "active_plan",
    "bind_active_lease_for_plan",
    "cancel_plan",
    "cancel_plans_for_instance_in_tx",
    "cancel_plans_for_job",
    "cancel_scheduled_native_plans_in_tx",
    "classify_failure",
    "compute_inherited_deadlines",
    "evaluate_recovery_eligibility_in_tx",
    "get_plan",
    "get_plan_in_tx",
    "is_l2_retryable_failure",
    "latest_plan_for_pool",
    "list_plans",
    "local_models_key",
    "make_group_id",
    "maybe_defer_retry_job_after_cleanup",
    "merge_recoverable_models",
    "on_runtime_reclaimed_in_tx",
    "on_successor_allocation_confirmed",
    "plan_key",
    "public_plan",
    "save_plan",
    "save_plan_in_tx",
    "status_summary",
    "tick",
]
