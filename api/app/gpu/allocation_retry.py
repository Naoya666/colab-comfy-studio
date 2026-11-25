"""Auto-queue GPU allocation retry policy, candidate matrix, and atomic retry deferral.

Covers initial allocation failures (before a runtime becomes ready) across both
Native ComfyUI and Creation pools (including Creation API / Native API auto-start).
"""

from __future__ import annotations

import copy
import logging
import random
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from sqlalchemy import select

from app.db import utcnow

from .compute import GpuConfig
from .cu_guard import CuGuardService
from .instances import POOL_CREATION, POOL_EPHEMERAL, POOL_NATIVE, job_pool
from .rates import estimate as estimate_units
from .rates import length_factor, minute_rate, mode_param
from .repository import GpuAccountRow, GpuCredentialRow, GpuEventRow, GpuExecutionRow, GpuJobRow, GpuLeaseRow
from .specs import (
    check_requirements,
    gpu_spec_order_settings,
    is_below_workflow_default,
    spec_id,
    spec_key,
    spec_rank,
)

log = logging.getLogger("studio.allocation_retry")

SETTINGS_PREFIX = "allocation-retry:"
MIN_INTERVAL_SECONDS = 20
MAX_INTERVAL_SECONDS = 120
DEFAULT_INTERVAL_SECONDS = 45
JITTER_MAX_SECONDS = 5.0
MIN_MAX_RETRIES = 1
MAX_MAX_RETRIES = 60
DEFAULT_MAX_RETRIES = 20
MAX_HISTORY_ENTRIES = 20

GLOBAL_FATAL_CODES = frozenset({
    "colab_rate_limited",
    "CU_DAILY_LIMIT_REACHED",
    "gpu_disabled",
    "REAL_GPU_DISABLED",
    "NATIVE_CONFIG",
    "NATIVE_DISABLED",
    "WORKFLOW_UNPUBLISHED",
    "MODEL_UNAVAILABLE",
    "MODEL_DEPENDENCIES_NOT_READY",
    "REVIEW_REQUIRED",
    "NATIVE_INPUT",
    "GPU_SPEC_BELOW_WORKFLOW_DEFAULT",
    "gpu_spec_mismatch",
    "CREATION_INSTANCE_SPEC_TOO_LOW",
    "gpu_cleanup_refused",
    "gpu_lease_missing",
    "gpu_execution_prepare_failed",
    "creation_lease_unavailable",
})

ACCOUNT_FATAL_CODES = frozenset({
    "colab_auth_failed",
    "colab_quota_exceeded",
    "CU_BALANCE_FLOOR_REACHED",
    "ACCOUNT_UNAVAILABLE",
    "ACCOUNT_NOT_FOUND",
    "ACCOUNT_DISABLED",
    "ACCOUNT_UNAUTHORIZED",
    "GOOGLE_NOT_CONNECTED",
})

ACCOUNT_FATAL_OUTCOMES = frozenset({
    "DENYLISTED",
    "BANNED",
    "RESTRICTED",
    "QUOTA_EXCEEDED_USAGE_TIME",
    "TOO_MANY",
})

RETRYABLE_ALLOCATION_CODES = frozenset({
    "gpu_capacity_unavailable",
    "colab_upstream_unavailable",
    "colab_assign_timeout",
    "colab_network_error",
    "colab_invalid_response",
    "gpu_allocation_failed",
    "gpu_create_failed",
    "gpu_allocation_aborted",
    "gpu_provider_unavailable",
    "gpu_runtime_inspect_failed",
    "gpu_runtime_missing",
})


def _base_store(store):
    return store.store if hasattr(store, "store") else store


def _settings_record_id(owner_id: str) -> str:
    return f"{SETTINGS_PREFIX}{owner_id}"


def _normalize_gpu_pool(items: Any) -> list[dict[str, str]]:
    if items is None:
        return []
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        raise ValueError("gpu_pool 必须是 GPU 规格数组")
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, Mapping):
            raise ValueError("gpu_pool 中的每一项都必须是有效的 GPU 规格对象")
        key = spec_key(dict(item))
        if key["variant"] != "VARIANT_GPU":
            raise ValueError("自动抢卡显卡池仅支持 GPU 规格")
        sid = spec_id(key)
        if sid not in seen:
            seen.add(sid)
            result.append(key)
    return result


def allocation_retry_settings(store, owner_id: str) -> dict[str, Any]:
    base = _base_store(store)
    saved = base.get("settings", _settings_record_id(owner_id), owner_id) if base is not None else None
    enabled = bool(saved.get("enabled", False)) if isinstance(saved, Mapping) else False
    try:
        gpu_pool = _normalize_gpu_pool(saved.get("gpu_pool")) if isinstance(saved, Mapping) else []
    except ValueError:
        gpu_pool = []
    raw_interval = saved.get("retry_interval_seconds") if isinstance(saved, Mapping) else None
    interval = (
        int(raw_interval)
        if isinstance(raw_interval, int) and not isinstance(raw_interval, bool)
        and MIN_INTERVAL_SECONDS <= raw_interval <= MAX_INTERVAL_SECONDS
        else DEFAULT_INTERVAL_SECONDS
    )
    raw_max = saved.get("max_retries") if isinstance(saved, Mapping) else None
    max_retries = (
        int(raw_max)
        if isinstance(raw_max, int) and not isinstance(raw_max, bool)
        and MIN_MAX_RETRIES <= raw_max <= MAX_MAX_RETRIES
        else DEFAULT_MAX_RETRIES
    )
    return {
        "enabled": enabled,
        "gpu_pool": gpu_pool,
        "retry_interval_seconds": interval,
        "max_retries": max_retries,
        "updated_at": saved.get("updated_at") if isinstance(saved, Mapping) else None,
    }


def save_allocation_retry_settings(
    store, owner_id: str, body: Mapping[str, Any], *, now: float | None = None
) -> dict[str, Any]:
    if not isinstance(body, Mapping):
        raise ValueError("配置参数必须是对象")
    unknown = set(body.keys()) - {"enabled", "gpu_pool", "retry_interval_seconds", "max_retries"}
    if unknown:
        raise ValueError("包含不支持的配置字段")
    current = allocation_retry_settings(store, owner_id)
    enabled = body.get("enabled", current["enabled"])
    if type(enabled) is not bool:
        raise ValueError("enabled 必须为布尔值")
    gpu_pool = _normalize_gpu_pool(body.get("gpu_pool", current["gpu_pool"]))
    interval = body.get("retry_interval_seconds", current["retry_interval_seconds"])
    if type(interval) is not int or not (MIN_INTERVAL_SECONDS <= interval <= MAX_INTERVAL_SECONDS):
        raise ValueError(f"重试间隔必须为 {MIN_INTERVAL_SECONDS} ~ {MAX_INTERVAL_SECONDS} 之间的整数秒")
    max_retries = body.get("max_retries", current["max_retries"])
    if type(max_retries) is not int or not (MIN_MAX_RETRIES <= max_retries <= MAX_MAX_RETRIES):
        raise ValueError(f"最大重试次数必须为 {MIN_MAX_RETRIES} ~ {MAX_MAX_RETRIES} 之间的整数")

    base = _base_store(store)
    ident = _settings_record_id(owner_id)
    ts = int(time.time() if now is None else now)
    payload = {
        "enabled": enabled,
        "gpu_pool": gpu_pool,
        "retry_interval_seconds": interval,
        "max_retries": max_retries,
        "updated_at": ts,
    }
    with base.admission_lock() as acquired:
        if not acquired:
            raise RuntimeError("自动排队抢卡策略正在保存，请稍后重试")
        existing = base.get("settings", ident, owner_id)
        if existing:
            base.update("settings", ident, payload, owner_id)
        else:
            base.create("settings", owner_id, payload, record_id=ident)
    return copy.deepcopy(payload)


def format_spec_label(raw: Mapping[str, Any] | None) -> str:
    if not isinstance(raw, Mapping):
        return "GPU"
    inner = raw.get("key") if isinstance(raw.get("key"), Mapping) else raw
    if not isinstance(inner, Mapping):
        return "GPU"
    acc = str(inner.get("accelerator") or "GPU").upper()
    shape = str(inner.get("shape") or "").upper()
    shape_label = "高系统内存" if "HIGHMEM" in shape else "标准系统内存"
    return f"{acc} · {shape_label}"


def _normalize_spec_dict(raw: Any) -> dict[str, str] | None:
    if not isinstance(raw, Mapping):
        return None
    inner = raw.get("key") if isinstance(raw.get("key"), Mapping) else raw
    if not isinstance(inner, Mapping):
        return None
    try:
        key = spec_key(dict(inner))
    except ValueError:
        return None
    return key if key.get("variant") == "VARIANT_GPU" else None


def is_auto_account_job(job: Mapping[str, Any]) -> bool:
    if isinstance(job.get("auto_select_account"), bool):
        return bool(job["auto_select_account"])
    policy = job.get("execution_policy") if isinstance(job.get("execution_policy"), Mapping) else {}
    requested = job.get("requested_account_id") or policy.get("account_id")
    return not bool(requested)


def is_retry_eligible_job(job: Mapping[str, Any]) -> bool:
    if not isinstance(job, Mapping):
        return False
    policy = job.get("execution_policy") if isinstance(job.get("execution_policy"), Mapping) else {}
    provider = job.get("provider") or policy.get("provider") or "mock"
    if provider != "colab":
        return False
    if job.get("recovery_plan_id"):
        return False
    if job.get("cancel_requested"):
        return False
    if job.get("allocation_confirmed_ready") is True:
        return False
    pool = job_pool(job)
    if pool not in {POOL_NATIVE, POOL_CREATION}:
        return False
    if pool == POOL_EPHEMERAL:
        return False
    reuse_action = str(policy.get("reuse_action") or job.get("reuse_action") or "auto")
    if reuse_action == "reuse":
        return False
    if job.get("requested_instance_id") or policy.get("instance_id"):
        return False
    if job.get("selection_reason") in {"reuse_idle", "requested_instance"}:
        return False
    return True


def classify_allocation_error(
    error_code: str | None,
    error_message: str | None = None,
    error_details: Mapping[str, Any] | None = None,
) -> str:
    """Classify an allocation failure for auto-queue retry.

    Returns one of: ``"global_fatal"``, ``"account_fatal"``, ``"spec_ineligible"``, ``"retryable"``.
    """
    code = str(error_code or "").strip()
    msg = str(error_message or "").lower()
    details = error_details if isinstance(error_details, Mapping) else {}
    http_status = details.get("http_status")
    outcome = str(details.get("outcome") or "").strip().upper()
    category = str(details.get("category") or "").strip().lower()

    if code == "colab_rate_limited" or http_status == 429:
        return "global_fatal"
    if code == "gpu_spec_ineligible" or (code == "gpu_spec_invalid" and category == "spec_ineligible"):
        return "spec_ineligible"
    if code in GLOBAL_FATAL_CODES:
        return "global_fatal"
    if code in ACCOUNT_FATAL_CODES:
        return "account_fatal"
    if http_status in {401, 403, 412}:
        return "account_fatal"
    if outcome and (
        outcome in ACCOUNT_FATAL_OUTCOMES
        or any(k in outcome for k in ("DENYLIST", "BANNED", "RESTRICTED", "QUOTA_EXCEEDED", "TOO_MANY"))
    ):
        return "account_fatal"
    if code in RETRYABLE_ALLOCATION_CODES or http_status in {500, 502, 503, 504}:
        return "retryable"
    if category in {"capacity", "timeout", "network", "upstream"}:
        return "retryable"
    if "no_runtimes" in msg or "quota_denied_requested_variants" in msg:
        return "retryable"
    return "global_fatal"


def build_candidate_specs(
    store,
    owner_id: str,
    job: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> list[dict[str, str]]:
    """Build ordered candidate GPU list: [selected_spec, *(gpu_pool sorted low->high by GPU order table)].

    In creation pool, any pool spec below workflow default or failing compute_requirements is filtered out.
    """
    order_table = gpu_spec_order_settings(store, owner_id)["ordered_specs"]
    policy = job.get("execution_policy") if isinstance(job.get("execution_policy"), Mapping) else {}
    manifest = (job.get("snapshot") or {}).get("manifest") if isinstance(job.get("snapshot"), Mapping) else {}
    manifest = manifest if isinstance(manifest, Mapping) else {}

    initial_raw = (
        job.get("initial_requested_spec")
        or job.get("requested_spec")
        or policy.get("compute_spec")
        or manifest.get("compute_spec")
    )
    initial = _normalize_spec_dict(initial_raw)
    initial_sid = spec_id(initial) if initial else None

    pool_raw = cfg.get("gpu_pool") or []
    rest_entries: list[tuple[int, int, dict[str, str]]] = []
    seen: set[str] = {initial_sid} if initial_sid else set()
    for idx, item in enumerate(pool_raw):
        norm = _normalize_spec_dict(item)
        if norm is None:
            continue
        sid = spec_id(norm)
        if sid in seen:
            continue
        seen.add(sid)
        rank = spec_rank(norm, order_table)
        rest_entries.append((rank if rank is not None else 10**6, idx, norm))

    rest_entries.sort(key=lambda entry: (entry[0], entry[1]))
    ordered_rest = [entry[2] for entry in rest_entries]

    pool = job_pool(job)
    candidates: list[dict[str, str]] = [initial] if initial else []
    for candidate in ordered_rest:
        if pool == POOL_CREATION and manifest:
            if is_below_workflow_default(manifest, candidate, order_table):
                continue
            try:
                check_requirements(dict(manifest), {"key": candidate})
            except ValueError:
                continue
        candidates.append(candidate)
    return candidates


def _account_row_usable(row: GpuAccountRow, owner_id: str, project_id: str | None) -> bool:
    if row.owner_id != owner_id:
        return False
    data = row.data or {}
    status = data.get("auth_status", data.get("status", data.get("connection_status", "connected")))
    connected = data.get("connected", status == "connected")
    if connected is not True or data.get("enabled", True) is not True or data.get("disabled", False) is True:
        return False
    if data.get("allow_allocation", True) is not True:
        return False
    allowed = data.get("allowed_projects") or data.get("allowed_project_ids") or data.get("project_ids") or data.get("projects")
    if allowed and project_id and project_id not in set(allowed):
        return False
    acct_provider = data.get("provider")
    if acct_provider not in {None, "colab", "google"}:
        return False
    return True


def _colab_connected_account_ids_in_tx(session, owner_id: str) -> set[str] | None:
    """Return set of account_ids with live Colab credentials, or None if no Colab creds exist in DB (test stub mode)."""
    rows = session.scalars(select(GpuCredentialRow).where(GpuCredentialRow.owner_id == owner_id)).all()
    colab_rows = [r for r in rows if (r.data or {}).get("purpose") == "colab"]
    if not colab_rows:
        return None
    connected: set[str] = set()
    for r in colab_rows:
        data = r.data or {}
        if data.get("revoked"):
            continue
        if data.get("encrypted_refresh_token"):
            aid = r.account_id or data.get("account_id")
            if isinstance(aid, str) and aid:
                connected.add(aid)
    return connected


def _resolve_candidate_accounts_in_tx(
    repo,
    session,
    owner_id: str,
    job: Mapping[str, Any],
    *,
    now: float,
    excluded_ids: set[str],
    preferred_order: Sequence[str] = (),
) -> tuple[list[str], dict[str, str], str | None, dict[str, dict[str, str]]]:
    """Return (ordered_account_ids, alias_map, global_stop_reason, cu_excluded_reasons)."""
    auto_select = is_auto_account_job(job)
    project_id = job.get("project_id")
    acct_rows = session.scalars(select(GpuAccountRow).where(GpuAccountRow.owner_id == owner_id)).all()
    alias_map: dict[str, str] = {}
    by_id: dict[str, GpuAccountRow] = {}
    for row in acct_rows:
        data = row.data or {}
        aid = str(data.get("account_id") or row.id)
        by_id[aid] = row
        alias_map[aid] = str(data.get("alias") or data.get("label") or aid)

    colab_connected = _colab_connected_account_ids_in_tx(session, owner_id)

    if auto_select:
        raw_ids = sorted(
            aid
            for aid, row in by_id.items()
            if aid not in excluded_ids
            and _account_row_usable(row, owner_id, project_id)
            and (colab_connected is None or aid in colab_connected)
        )
    else:
        pinned = job.get("requested_account_id") or job.get("account_id")
        if not isinstance(pinned, str) or not pinned or pinned in excluded_ids:
            raw_ids = []
        else:
            row = by_id.get(pinned)
            if row is not None and _account_row_usable(row, owner_id, project_id) and (
                colab_connected is None or pinned in colab_connected
            ):
                raw_ids = [pinned]
            else:
                raw_ids = []

    guard = CuGuardService(repo, lambda: now)
    guard_state = guard.state_in(session, owner_id)
    if guard_state.get("soft_tripped"):
        return (
            [],
            alias_map,
            str(guard_state.get("daily_message") or "已触发今日 CU 上限熔断，停止自动排队抢卡"),
            {},
        )
    hard_map = {
        str(item["account_id"]): str(guard_state.get("message") or "账号余额已触及底线熔断")
        for item in (guard_state.get("hard_tripped_accounts") or [])
        if isinstance(item, Mapping) and item.get("account_id")
    }
    cu_excluded: dict[str, dict[str, str]] = {}
    filtered: list[str] = []
    for aid in raw_ids:
        if aid in hard_map:
            cu_excluded[aid] = {
                "alias": alias_map.get(aid, aid),
                "code": "CU_BALANCE_FLOOR_REACHED",
                "reason": hard_map[aid],
            }
            continue
        filtered.append(aid)

    # Preserve stable rotation order starting from preferred_order.
    ordered: list[str] = []
    seen: set[str] = set()
    for aid in preferred_order:
        if aid in filtered and aid not in seen:
            seen.add(aid)
            ordered.append(aid)
    for aid in filtered:
        if aid not in seen:
            seen.add(aid)
            ordered.append(aid)
    return ordered, alias_map, None, cu_excluded


def public_allocation_retry(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, Mapping):
        return None
    keys = (
        "enabled",
        "active",
        "phase",
        "attempt",
        "retries_used",
        "max_retries",
        "interval_seconds",
        "next_retry_at",
        "candidate_specs",
        "current_spec",
        "current_account_id",
        "current_account_alias",
        "next_spec",
        "next_account_id",
        "next_account_alias",
        "excluded_accounts",
        "last_error_code",
        "last_error_message",
        "last_failed_spec",
        "last_failed_account_id",
        "last_failed_account_alias",
        "stop_reason",
        "updated_at",
    )
    out = {k: copy.deepcopy(raw[k]) for k in keys if k in raw}
    if "history" in raw and isinstance(raw["history"], list):
        out["history"] = copy.deepcopy(raw["history"][-10:])
    return out or None


def sync_attempt_state(
    repo,
    job: dict[str, Any],
    *,
    now: float,
    spec: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Ensure allocation_retry metadata reflects the currently executing allocation attempt."""
    if not is_retry_eligible_job(job):
        return job
    owner_id = job.get("owner_id")
    if not owner_id:
        return job
    base = _base_store(repo)
    cfg = allocation_retry_settings(base, owner_id)
    existing = job.get("allocation_retry") if isinstance(job.get("allocation_retry"), Mapping) else None
    if not cfg["enabled"] and not existing:
        return job

    candidates = build_candidate_specs(base, owner_id, job, cfg)
    cur_spec = _normalize_spec_dict(spec or job.get("requested_spec")) or (candidates[0] if candidates else None)
    cur_acct = job.get("account_id") or job.get("retry_target_account_id") or job.get("requested_account_id")
    cur_alias = job.get("account_alias_snapshot")
    if not cur_alias and cur_acct:
        acct_row = repo.get("accounts", cur_acct, owner_id) if hasattr(repo, "get") else None
        if isinstance(acct_row, Mapping):
            cur_alias = acct_row.get("alias") or acct_row.get("label") or cur_acct

    if existing is None:
        account_order = [cur_acct] if isinstance(cur_acct, str) and cur_acct else []
        state = {
            "enabled": bool(cfg["enabled"]),
            "active": True,
            "phase": "allocating",
            "attempt": 1,
            "retries_used": 0,
            "max_retries": int(cfg["max_retries"]),
            "interval_seconds": int(cfg["retry_interval_seconds"]),
            "next_retry_at": None,
            "candidate_specs": candidates,
            "account_order": account_order,
            "current_spec": cur_spec,
            "current_account_id": cur_acct,
            "current_account_alias": cur_alias,
            "next_spec": None,
            "next_account_id": None,
            "next_account_alias": None,
            "excluded_accounts": {},
            "ineligible_pairs": [],
            "last_error_code": None,
            "last_error_message": None,
            "last_failed_spec": None,
            "last_failed_account_id": None,
            "last_failed_account_alias": None,
            "stop_reason": None,
            "history": [],
            "updated_at": now,
        }
    else:
        state = copy.deepcopy(dict(existing))
        state.update(
            enabled=bool(cfg["enabled"]),
            active=True,
            phase="allocating",
            max_retries=int(cfg["max_retries"]),
            interval_seconds=int(cfg["retry_interval_seconds"]),
            next_retry_at=None,
            candidate_specs=candidates or state.get("candidate_specs") or [],
            current_spec=cur_spec,
            current_account_id=cur_acct,
            current_account_alias=cur_alias,
            next_spec=None,
            next_account_id=None,
            next_account_alias=None,
            updated_at=now,
        )
        if isinstance(cur_acct, str) and cur_acct and cur_acct not in (state.get("account_order") or []):
            state["account_order"] = [*(state.get("account_order") or []), cur_acct]

    patch: dict[str, Any] = {"allocation_retry": state}
    if not job.get("initial_requested_spec") and cur_spec is not None:
        patch["initial_requested_spec"] = {"key": cur_spec}
    return repo.update("jobs", job["id"], patch, owner_id)


def mark_reconciling(
    repo,
    job: dict[str, Any],
    *,
    now: float,
    error_code: str | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    existing = job.get("allocation_retry")
    if not isinstance(existing, Mapping) or not existing.get("active"):
        return job
    state = copy.deepcopy(dict(existing))
    state["phase"] = "reconciling"
    state["next_retry_at"] = None
    if error_code:
        state["last_error_code"] = error_code
    if error_message:
        state["last_error_message"] = error_message
    state["updated_at"] = now
    return repo.update("jobs", job["id"], {"allocation_retry": state}, job.get("owner_id"))


def mark_succeeded(
    repo,
    job: dict[str, Any],
    *,
    now: float,
    actual_spec: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    existing = job.get("allocation_retry")
    if not isinstance(existing, Mapping):
        return job
    state = copy.deepcopy(dict(existing))
    state["active"] = False
    state["phase"] = "succeeded"
    state["next_retry_at"] = None
    state["next_spec"] = None
    state["next_account_id"] = None
    state["next_account_alias"] = None
    if actual_spec:
        norm = _normalize_spec_dict(actual_spec)
        if norm:
            state["current_spec"] = norm
    if job.get("account_id"):
        state["current_account_id"] = job["account_id"]
    if job.get("account_alias_snapshot"):
        state["current_account_alias"] = job["account_alias_snapshot"]
    state["updated_at"] = now
    return repo.update("jobs", job["id"], {"allocation_retry": state}, job.get("owner_id"))


def mark_cancelled(
    repo,
    job: dict[str, Any],
    *,
    now: float,
    reason: str = "用户手动停止排队抢卡",
) -> dict[str, Any]:
    existing = job.get("allocation_retry")
    if not isinstance(existing, Mapping):
        return job
    state = copy.deepcopy(dict(existing))
    state["active"] = False
    state["phase"] = "cancelled"
    state["next_retry_at"] = None
    state["stop_reason"] = reason
    state["updated_at"] = now
    return repo.update("jobs", job["id"], {"allocation_retry": state}, job.get("owner_id"))


def _pick_next_pair(
    accounts: Sequence[str],
    specs: Sequence[dict[str, str]],
    *,
    last_account_id: str | None,
    last_spec_id: str | None,
    ineligible_pairs: set[str],
) -> tuple[str, dict[str, str]] | None:
    """Account-first, then GPU low->high traversal over (account, spec) matrix."""
    matrix: list[tuple[str, dict[str, str], str]] = []
    for spec in specs:
        sid = spec_id(spec)
        for aid in accounts:
            matrix.append((aid, spec, f"{aid}|{sid}"))
    if not matrix:
        return None

    # Find the exact position of the just-failed (account, spec) pair, or match by spec if account was excluded.
    start_idx = 0
    found_exact = False
    if last_account_id and last_spec_id:
        target = f"{last_account_id}|{last_spec_id}"
        for idx, (_, _, pair_key) in enumerate(matrix):
            if pair_key == target:
                start_idx = (idx + 1) % len(matrix)
                found_exact = True
                break
    if not found_exact and last_spec_id:
        # Failed account was excluded: continue trying remaining accounts for the SAME spec first!
        for idx, (_, spec, _) in enumerate(matrix):
            if spec_id(spec) == last_spec_id:
                start_idx = idx
                break

    for offset in range(len(matrix)):
        aid, spec, pair_key = matrix[(start_idx + offset) % len(matrix)]
        if pair_key not in ineligible_pairs:
            return aid, spec
    return None


def _recalculate_cost_fields(repo, owner_id: str, jdata: Mapping[str, Any], next_spec: dict[str, str]) -> dict[str, Any]:
    cfg = GpuConfig.from_env()
    pool = job_pool(jdata)
    if pool == POOL_NATIVE:
        rate = minute_rate({"key": next_spec}, fallback_rate=cfg.unit_rate)
        return {"unit_rate": rate}
    manifest = (jdata.get("snapshot") or {}).get("manifest") if isinstance(jdata.get("snapshot"), Mapping) else {}
    manifest = manifest if isinstance(manifest, Mapping) else {}
    params = (jdata.get("snapshot") or {}).get("resolved_params") or jdata.get("params") or {}
    picked = mode_param(manifest) if manifest else None
    mode = params.get(picked[0]) if picked else None
    cap = int(jdata.get("max_runtime_minutes") or cfg.max_runtime_minutes)
    units, rate = estimate_units(
        repo,
        owner_id,
        str(jdata.get("workflow_key") or ""),
        manifest,
        next_spec,
        cap,
        mode=mode,
        fallback_rate=cfg.unit_rate,
    )
    if units is not None and rate is not None:
        units = round(min(units * length_factor(manifest, params), cap * rate), 3)
    return {"unit_rate": rate, "estimated_units": units}


def _emit_event_in_tx(
    session,
    owner_id: str,
    project_id: str | None,
    job_id: str,
    state: str,
    message: str,
) -> None:
    session.add(
        GpuEventRow(
            id=str(uuid.uuid4()),
            owner_id=owner_id,
            project_id=project_id,
            state=state,
            version=0,
            data={"job_id": job_id, "state": state, "message": str(message)[:500]},
            created_at=utcnow(),
            updated_at=utcnow(),
        )
    )


def maybe_defer_allocation_retry(
    repo,
    job: dict[str, Any],
    lease: dict[str, Any] | None,
    now: float,
    *,
    jitter_fn: Callable[[], float] | None = None,
) -> dict[str, Any] | None:
    """After an allocation failure's lease is confirmed deleted, defer the job back to
    ``waiting_for_capacity`` for its next retry attempt, or return ``None`` to finalize failure.
    """
    if not is_retry_eligible_job(job):
        return None
    owner_id = job.get("owner_id")
    if not owner_id:
        return None
    store = _base_store(repo)
    if not (hasattr(store, "db_lock") and hasattr(store, "Session")):
        return None

    cfg = allocation_retry_settings(store, owner_id)
    if not cfg["enabled"]:
        return None

    with store.db_lock(), store.Session.begin() as session:
        store.begin_immediate(session)
        job_row = session.get(GpuJobRow, job["id"])
        if job_row is None or job_row.owner_id != owner_id:
            return None
        jdata = copy.deepcopy(job_row.data or {})
        if jdata.get("cancel_requested") or jdata.get("recovery_plan_id") or jdata.get("allocation_confirmed_ready"):
            return None

        # Anti-double-GPU fence: EVERY lease belonging to this job must be confirmed deleted.
        for lrow in session.scalars(select(GpuLeaseRow).where(GpuLeaseRow.owner_id == owner_id)).all():
            if (lrow.data or {}).get("owner_job_id") == job["id"] and lrow.state != "deleted":
                return None

        err_code = jdata.get("error_code") or job.get("error_code")
        err_msg = jdata.get("error_message") or job.get("error_message")
        err_details = jdata.get("error_details") or job.get("error_details")
        err_kind = classify_allocation_error(err_code, err_msg, err_details)

        existing = jdata.get("allocation_retry") if isinstance(jdata.get("allocation_retry"), Mapping) else {}
        state = copy.deepcopy(dict(existing))
        candidates = build_candidate_specs(store, owner_id, jdata, cfg)
        failed_spec = _normalize_spec_dict(jdata.get("requested_spec")) or (candidates[0] if candidates else None)
        failed_spec_id = spec_id(failed_spec) if failed_spec else None
        failed_acct = jdata.get("account_id") or jdata.get("retry_target_account_id") or jdata.get("requested_account_id")
        failed_alias = jdata.get("account_alias_snapshot") or failed_acct

        retries_used = int(state.get("retries_used") or 0)
        attempt = int(state.get("attempt") or 1)
        max_retries = int(cfg["max_retries"])
        interval_seconds = int(cfg["retry_interval_seconds"])
        excluded_accounts: dict[str, dict[str, str]] = copy.deepcopy(state.get("excluded_accounts") or {})
        ineligible_pairs: set[str] = set(state.get("ineligible_pairs") or [])
        account_order: list[str] = list(state.get("account_order") or [])
        if isinstance(failed_acct, str) and failed_acct and failed_acct not in account_order:
            account_order.append(failed_acct)

        history = list(state.get("history") or [])
        history.append({
            "attempt": attempt,
            "account_id": failed_acct,
            "account_alias": failed_alias,
            "spec": failed_spec,
            "error_code": err_code,
            "error_message": str(err_msg or "")[:240],
            "kind": err_kind,
            "at": round(float(now), 1),
        })
        history = history[-MAX_HISTORY_ENTRIES:]

        def _abort(phase: str, stop_reason: str) -> None:
            state.update(
                enabled=True,
                active=False,
                phase=phase,
                attempt=attempt,
                retries_used=retries_used,
                max_retries=max_retries,
                interval_seconds=interval_seconds,
                next_retry_at=None,
                candidate_specs=candidates,
                account_order=account_order,
                excluded_accounts=excluded_accounts,
                ineligible_pairs=sorted(ineligible_pairs),
                last_error_code=err_code,
                last_error_message=err_msg,
                last_failed_spec=failed_spec,
                last_failed_account_id=failed_acct,
                last_failed_account_alias=failed_alias,
                stop_reason=stop_reason,
                history=history,
                updated_at=now,
            )
            jdata["allocation_retry"] = state
            job_row.data = jdata
            job_row.version = int(job_row.version or 0) + 1
            job_row.updated_at = utcnow()
            _emit_event_in_tx(session, owner_id, job_row.project_id, job_row.id, job_row.state, stop_reason)

        if err_kind == "global_fatal":
            reason = (
                "触发 Colab 429 限流，已立即停止自动排队抢卡以保护账号"
                if (err_code == "colab_rate_limited" or (isinstance(err_details, Mapping) and err_details.get("http_status") == 429))
                else f"遇到不可重试错误（{err_code or 'fatal'}），已停止自动排队抢卡"
            )
            _abort("aborted", reason)
            return None

        if err_kind == "account_fatal":
            if isinstance(failed_acct, str) and failed_acct:
                excluded_accounts[failed_acct] = {
                    "alias": str(failed_alias or failed_acct),
                    "code": str(err_code or "account_fatal"),
                    "reason": str(err_msg or "账号异常"),
                }
            if not is_auto_account_job(jdata):
                _abort("aborted", f"所选账号不可用或鉴权/配额受限（{err_code or 'account_fatal'}），已停止自动排队抢卡")
                return None

        if err_kind == "spec_ineligible":
            if isinstance(failed_acct, str) and failed_acct and failed_spec_id:
                ineligible_pairs.add(f"{failed_acct}|{failed_spec_id}")

        if err_kind == "retryable" and retries_used >= max_retries:
            _abort("exhausted", f"已达到最大自动重试次数（{max_retries}/{max_retries}），停止排队抢卡")
            return None

        if not candidates:
            _abort("aborted", "没有符合要求的候选 GPU 规格，已停止自动排队抢卡")
            return None

        accounts, alias_map, global_cu_stop, cu_excluded = _resolve_candidate_accounts_in_tx(
            repo,
            session,
            owner_id,
            jdata,
            now=now,
            excluded_ids=set(excluded_accounts.keys()),
            preferred_order=account_order,
        )
        if cu_excluded:
            excluded_accounts.update(cu_excluded)
        if global_cu_stop:
            _abort("aborted", global_cu_stop)
            return None
        if not accounts:
            _abort("aborted", "当前所有可用 Google 账号均已受限或触及余额底线，已停止自动排队抢卡")
            return None

        next_pair = _pick_next_pair(
            accounts,
            candidates,
            last_account_id=failed_acct if isinstance(failed_acct, str) else None,
            last_spec_id=failed_spec_id,
            ineligible_pairs=ineligible_pairs,
        )
        if next_pair is None:
            _abort("aborted", "所有候选账号均不支持重试显卡池中的任何规格，已停止自动排队抢卡")
            return None

        next_acct, next_spec = next_pair
        next_alias = alias_map.get(next_acct, next_acct)
        if err_kind == "retryable":
            retries_used += 1
        next_attempt = attempt + 1

        if err_kind == "spec_ineligible":
            delay = 0.0
        elif err_kind == "account_fatal":
            delay = 2.0
        else:
            jitter = float(jitter_fn()) if jitter_fn is not None else random.uniform(0.0, JITTER_MAX_SECONDS)
            delay = float(interval_seconds) + max(0.0, min(JITTER_MAX_SECONDS, jitter))
        next_retry_at = float(now) + delay

        # Close any stale non-terminal execution row.
        old_exec_id = jdata.get("execution_id")
        if old_exec_id:
            exec_row = session.get(GpuExecutionRow, old_exec_id)
            if exec_row is not None and exec_row.owner_id == owner_id and exec_row.state not in {"completed", "failed", "cancelled"}:
                exec_row.state = "failed"
                exec_row.data = {
                    **(exec_row.data or {}),
                    "state": "failed",
                    "finished_at": now,
                    "error_code": err_code or "allocation_retry",
                }
                exec_row.version = int(exec_row.version or 0) + 1
                exec_row.updated_at = utcnow()

        old_lease_id = jdata.get("lease_id") or (lease or {}).get("id")
        retry_keys = list(jdata.get("attempt_retry_keys") or [])
        if old_lease_id and old_lease_id not in retry_keys:
            retry_keys.append(old_lease_id)

        if not jdata.get("initial_requested_spec") and failed_spec is not None:
            jdata["initial_requested_spec"] = {"key": failed_spec}

        # Reset transient attempt fields cleanly for the new attempt.
        for field in (
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
            "ccu_before",
            "ccu_before_at",
            "ccu_before_source",
            "started_epoch",
            "started_at",
            "deadline_epoch",
            "task_deadline_epoch",
            "runtime_spec_reported",
            "gpu_reported",
            "gpu_diagnostics",
            "account_id",
            "account_alias_snapshot",
        ):
            jdata.pop(field, None)

        # Native ComfyUI mounts Drive from the allocated compute account (or its delegated Drive);
        # never keep a stale Drive pin from a failed account attempt.
        if job_pool(jdata) == POOL_NATIVE:
            jdata.pop("drive_account_id", None)
            jdata.pop("native_drive_account_id", None)

        # Keep snapshot compute spec aligned so runtime_worker._check_compute_hardware verifies the actual next_spec.
        snapshot = copy.deepcopy(jdata.get("snapshot") or {})
        if job_pool(jdata) == POOL_NATIVE:
            manifest = copy.deepcopy(snapshot.get("manifest") or {})
            manifest["compute_spec"] = copy.deepcopy(next_spec)
            snapshot["manifest"] = manifest
        else:
            snapshot["runtime_compute_spec"] = copy.deepcopy(next_spec)

        cost_patch = _recalculate_cost_fields(repo, owner_id, jdata, next_spec)
        state.update(
            enabled=True,
            active=True,
            phase="waiting",
            attempt=next_attempt,
            retries_used=retries_used,
            max_retries=max_retries,
            interval_seconds=interval_seconds,
            next_retry_at=next_retry_at,
            candidate_specs=candidates,
            account_order=accounts,
            current_spec=failed_spec,
            current_account_id=failed_acct,
            current_account_alias=failed_alias,
            next_spec=next_spec,
            next_account_id=next_acct,
            next_account_alias=next_alias,
            excluded_accounts=excluded_accounts,
            ineligible_pairs=sorted(ineligible_pairs),
            last_error_code=err_code,
            last_error_message=err_msg,
            last_failed_spec=failed_spec,
            last_failed_account_id=failed_acct,
            last_failed_account_alias=failed_alias,
            stop_reason=None,
            history=history,
            updated_at=now,
        )

        new_attempt_id = f"att-{uuid.uuid4().hex[:12]}"
        jdata.update(
            state="waiting_for_capacity",
            attempt_id=new_attempt_id,
            attempt_retry_keys=retry_keys,
            runtime_deleted=False,
            retry_not_before_epoch=next_retry_at,
            retry_target_account_id=next_acct,
            requested_spec={"key": copy.deepcopy(next_spec)},
            snapshot=snapshot,
            allocation_retry=state,
            **cost_patch,
        )
        job_row.state = "waiting_for_capacity"
        job_row.instance_id = None
        job_row.account_id = None
        job_row.data = jdata
        job_row.version = int(job_row.version or 0) + 1
        job_row.updated_at = utcnow()

        next_spec_label = format_spec_label(next_spec)
        if err_kind == "spec_ineligible":
            msg = f"账号「{failed_alias}」无 {format_spec_label(failed_spec)} 申请资格，立即切换尝试 {next_alias} · {next_spec_label}"
        elif err_kind == "account_fatal":
            msg = f"账号「{failed_alias}」异常（{err_code}）已移出本次轮换，即将切换至 {next_alias} · {next_spec_label}"
        else:
            msg = (
                f"自动排队抢卡：第 {retries_used}/{max_retries} 次重试，"
                f"将在 {max(1, int(round(delay)))} 秒后尝试 {next_alias} · {next_spec_label}"
                f"（上次 {format_spec_label(failed_spec)}：{err_code or '无可用卡'}）"
            )
        _emit_event_in_tx(session, owner_id, job_row.project_id, job_row.id, "waiting_for_capacity", msg)
        return repo._row(job_row)


def revalidate_before_allocation(
    repo,
    job: dict[str, Any],
    *,
    now: float,
) -> tuple[dict[str, Any], str | None, str | None]:
    """Re-verify excluded accounts, live account usability, and (account, spec) eligibility
    right before allocating a retry-queued job.

    Returns ``(updated_job, abort_error_code, abort_reason)``.
    """
    existing = job.get("allocation_retry")
    if not isinstance(existing, Mapping) or not existing.get("active"):
        return job, None, None
    owner_id = job.get("owner_id")
    if not owner_id:
        return job, None, None
    store = _base_store(repo)
    if not (hasattr(store, "db_lock") and hasattr(store, "Session")):
        return job, None, None

    cfg = allocation_retry_settings(store, owner_id)
    with store.db_lock(), store.Session.begin() as session:
        store.begin_immediate(session)
        job_row = session.get(GpuJobRow, job["id"])
        if job_row is None or job_row.owner_id != owner_id:
            return job, None, None
        jdata = copy.deepcopy(job_row.data or {})
        state = jdata.get("allocation_retry")
        if not isinstance(state, Mapping) or not state.get("active"):
            return repo._row(job_row), None, None
        if jdata.get("instance_id"):
            return repo._row(job_row), None, None
        state = copy.deepcopy(dict(state))

        excluded_accounts: dict[str, dict[str, str]] = copy.deepcopy(state.get("excluded_accounts") or {})
        ineligible_pairs: set[str] = set(state.get("ineligible_pairs") or [])
        account_order: list[str] = list(state.get("account_order") or [])

        accounts, alias_map, global_cu_stop, cu_excluded = _resolve_candidate_accounts_in_tx(
            repo,
            session,
            owner_id,
            jdata,
            now=now,
            excluded_ids=set(excluded_accounts.keys()),
            preferred_order=account_order,
        )
        if cu_excluded:
            excluded_accounts.update(cu_excluded)

        def _abort(error_code: str, stop_reason: str) -> tuple[dict[str, Any], str, str]:
            state.update(
                active=False,
                phase="aborted",
                next_retry_at=None,
                next_account_id=None,
                next_account_alias=None,
                next_spec=None,
                account_order=accounts,
                excluded_accounts=excluded_accounts,
                ineligible_pairs=sorted(ineligible_pairs),
                stop_reason=stop_reason,
                updated_at=now,
            )
            jdata["allocation_retry"] = state
            jdata.pop("retry_target_account_id", None)
            job_row.data = jdata
            job_row.version = int(job_row.version or 0) + 1
            job_row.updated_at = utcnow()
            _emit_event_in_tx(session, owner_id, job_row.project_id, job_row.id, job_row.state, stop_reason)
            return repo._row(job_row), error_code, stop_reason

        if global_cu_stop:
            return _abort("CU_DAILY_LIMIT_REACHED", global_cu_stop)
        if not accounts:
            return _abort(
                "ACCOUNT_UNAVAILABLE",
                "当前所有可用 Google 账号均已受限、禁用或触及余额底线，已停止自动排队抢卡",
            )

        planned_acct = jdata.get("retry_target_account_id") or jdata.get("requested_account_id")
        planned_spec = _normalize_spec_dict(jdata.get("requested_spec"))
        planned_sid = spec_id(planned_spec) if planned_spec else None

        planned_valid = (
            isinstance(planned_acct, str)
            and planned_acct in accounts
            and (planned_sid is None or f"{planned_acct}|{planned_sid}" not in ineligible_pairs)
        )
        if planned_valid:
            if cu_excluded:
                state["excluded_accounts"] = excluded_accounts
                state["account_order"] = accounts
                state["updated_at"] = now
                jdata["allocation_retry"] = state
                job_row.data = jdata
                job_row.version = int(job_row.version or 0) + 1
                job_row.updated_at = utcnow()
            return repo._row(job_row), None, None

        candidates = build_candidate_specs(store, owner_id, jdata, cfg) or list(state.get("candidate_specs") or [])
        if not candidates:
            return _abort("gpu_spec_ineligible", "没有符合要求的候选 GPU 规格，已停止自动排队抢卡")

        next_pair = _pick_next_pair(
            accounts,
            candidates,
            last_account_id=None,
            last_spec_id=planned_sid,
            ineligible_pairs=ineligible_pairs,
        )
        if next_pair is None:
            return _abort(
                "gpu_spec_ineligible",
                "所有候选账号均不支持重试显卡池中的任何规格，已停止自动排队抢卡",
            )

        next_acct, next_spec = next_pair
        next_alias = alias_map.get(next_acct, next_acct)

        snapshot = copy.deepcopy(jdata.get("snapshot") or {})
        if job_pool(jdata) == POOL_NATIVE:
            manifest = copy.deepcopy(snapshot.get("manifest") or {})
            manifest["compute_spec"] = copy.deepcopy(next_spec)
            snapshot["manifest"] = manifest
        else:
            snapshot["runtime_compute_spec"] = copy.deepcopy(next_spec)

        cost_patch = _recalculate_cost_fields(repo, owner_id, jdata, next_spec)
        state.update(
            account_order=accounts,
            candidate_specs=candidates,
            excluded_accounts=excluded_accounts,
            ineligible_pairs=sorted(ineligible_pairs),
            next_account_id=next_acct,
            next_account_alias=next_alias,
            next_spec=next_spec,
            updated_at=now,
        )
        jdata.update(
            retry_target_account_id=next_acct,
            requested_spec={"key": copy.deepcopy(next_spec)},
            snapshot=snapshot,
            allocation_retry=state,
            **cost_patch,
        )
        job_row.data = jdata
        job_row.version = int(job_row.version or 0) + 1
        job_row.updated_at = utcnow()
        _emit_event_in_tx(
            session,
            owner_id,
            job_row.project_id,
            job_row.id,
            job_row.state,
            f"原计划重试账号已不可用，自动切换尝试 {next_alias} · {format_spec_label(next_spec)}",
        )
        return repo._row(job_row), None, None
