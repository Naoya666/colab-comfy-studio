"""Colab balances and instance CU accounting."""

from __future__ import annotations

import logging
import math
import os
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import func, select, text

from app.config import settings
from app.db import SettingRow

from .rates import minute_rate
from .repository import GpuCcuLedgerRow, GpuInstanceRow, GpuInstanceUsageSegmentRow, GpuJobRow, GpuLeaseRow
from .specs import spec_key

log = logging.getLogger(__name__)

ACTIVE_LEASE_STATES = frozenset(
    {"creating", "active", "busy", "idle", "cleanup_pending", "quarantined", "releasing"}
)
TERMINAL_JOB_STATES = frozenset({"completed", "failed", "cancelled", "expired"})

_BALANCE_KEYS = (
    "paidComputeUnitsBalance",
    "paid_compute_units_balance",
    "currentBalance",
    "current_balance",
    "computeUnitsBalance",
    "balance",
)
_RATE_KEYS = ("consumptionRateHourly", "consumption_rate_hourly", "rateHourly", "usageRateHourly")
_ASSIGNMENT_KEYS = ("assignmentsCount", "assignments_count", "activeAssignments", "active_assignments")


def _number(payload: Any, keys: tuple[str, ...], depth: int = 0) -> float | None:
    """Find the first numeric value for any key, peeking one level into dicts."""
    if not isinstance(payload, dict) or depth > 2:
        return None
    for key in keys:
        value = payload.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            if math.isfinite(value):
                return float(value)
        if isinstance(value, str):
            try:
                number = float(value)
            except ValueError:
                continue
            if math.isfinite(number):
                return number
    for value in payload.values():
        if isinstance(value, dict):
            found = _number(value, keys, depth + 1)
            if found is not None:
                return found
    return None


def normalize_subscription(payload: Any) -> dict[str, Any]:
    """Normalize tunnel and REST subscription payloads to one shape."""
    balance = _number(payload, _BALANCE_KEYS)
    rate = _number(payload, _RATE_KEYS)
    assignments = _number(payload, _ASSIGNMENT_KEYS)
    return {
        "balance": balance,
        "rate": rate,
        "assignments": int(assignments) if assignments is not None else None,
    }


def sampling_enabled() -> bool:
    if os.getenv("GPU_CCU_SAMPLING", "1") != "1":
        return False
    # Tests must never reach Google: the sampler is monkeypatched when exercised.
    if os.getenv("PYTEST_CURRENT_TEST"):
        return False
    if os.getenv("STUDIO_OFFLINE_TEST") == "1":
        return False
    return True


def sample_account_balance(repo, owner_id: str, account_id: str | None, *, timeout: float = 8.0) -> dict | None:
    """Read one account's live compute-unit balance. Returns None on any failure."""
    if not account_id or not owner_id:
        return None
    provider = None
    try:
        from .colab import GoogleColabProvider

        provider = GoogleColabProvider(repo, owner_id, account_id=account_id, timeout=timeout)
        payload = provider.subscription()
        sample = normalize_subscription(payload)
        if sample["balance"] is None:
            return None
        sample["at"] = time.time()
        return sample
    except Exception as exc:  # noqa: BLE001 - sampling is best-effort by contract
        log.warning("ccu sample failed for account %s: %s", account_id[:8], type(exc).__name__)
        return None
    finally:
        if provider is not None:
            try:
                provider.close()
            except Exception:  # noqa: BLE001
                pass


_DEFAULT_SAMPLE_ACCOUNT_BALANCE = sample_account_balance


def active_lease_count(repo, owner_id: str, account_id: str | None, *, exclude_lease_id: str | None = None) -> int:
    if not account_id:
        return 0
    total = 0
    for lease in repo.list("leases", owner_id):
        if lease.get("account_id") != account_id:
            continue
        if lease.get("state") not in ACTIVE_LEASE_STATES:
            continue
        if exclude_lease_id and lease.get("id") == exclude_lease_id:
            continue
        total += 1
    return total


def active_job_count(repo, owner_id: str, account_id: str | None) -> int:
    if not account_id:
        return 0
    total = 0
    for job in repo.list("jobs", owner_id):
        if job.get("account_id") != account_id:
            continue
        if job.get("state") in TERMINAL_JOB_STATES:
            continue
        total += 1
    return total


def local_date(moment=None):
    return datetime.fromtimestamp(time.time() if moment is None else moment, ZoneInfo(settings.studio_timezone)).date().isoformat()


@contextmanager
def transaction(repo, owner):
    with repo.store.db_lock(), repo.store.Session.begin() as session:
        if repo.store.engine.dialect.name == "sqlite":
            session.execute(text("BEGIN IMMEDIATE"))
        else:
            session.execute(text("SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"), {"key": f"studio-cu:{owner}"})
        yield session


def setting(session, key, values=None, owner=None):
    row = session.get(SettingRow, key)
    if values is not None:
        if row is None:
            row = SettingRow(id=key, owner_id=owner)
            session.add(row)
        row.data = values
    return dict(row.data or {}) if row else {}


def enabled_account(data):
    return (data.get("provider", "colab") in {"colab", "google"} and not data.get("is_mock")
            and data.get("enabled") is not False and not data.get("disabled"))


def has_colab_credential(repo, owner, account_id) -> bool:
    if not account_id or not owner:
        return False
    try:
        from .google import GoogleService

        colab = GoogleService(repo, owner, account_id=account_id).status().get("colab") or {}
        return bool(colab.get("connected"))
    except Exception:  # noqa: BLE001 - best-effort capability check
        return False


def accounts(repo, owner):
    return {
        row.get("account_id") or row["id"]: row
        for row in repo.list("accounts", owner)
        if enabled_account(row)
    }


def _snapshot_key(owner, account):
    return "ccu:" + uuid.uuid5(uuid.NAMESPACE_URL, f"{owner}:{account}").hex


def instance_live_rate(session, owner, account, instance_id):
    if account and instance_id:
        snapshot = setting(session, _snapshot_key(owner, account))
        if (not snapshot.get("error") and snapshot.get("active_instance_ids") == [instance_id]
                and (snapshot.get("assignments") or 0) <= 1):
            return snapshot.get("rate_hourly")
    return None


def snapshots(repo, owner):
    directory = accounts(repo, owner)
    colab_flags = {key: has_colab_credential(repo, owner, key) for key in directory}
    real_mode = sample_account_balance is _DEFAULT_SAMPLE_ACCOUNT_BALANCE
    with repo.store.db_lock(), repo.store.Session() as session:
        result = {}
        for key, row in directory.items():
            saved = setting(session, _snapshot_key(owner, key))
            colab_connected = colab_flags[key]
            merged = {
                "alias": row.get("alias"),
                "balance": None,
                "rate_hourly": None,
                "error": None,
                **saved,
                "colab_connected": colab_connected,
            }
            # Suppress legacy stale sample_failed errors on accounts that never had Colab authorized.
            if real_mode and not colab_connected and merged.get("sampled_at") is None:
                merged["error"] = None
            result[key] = merged
        return result


def execution_finished(job, now):
    start = job.get("execution_started_at")
    if start is None or job.get("execution_ended_at") is not None:
        return {}
    return {"execution_ended_at": now, "execution_seconds": max(0, now - start)}


def _segments(session, owner, account=None, states=None):
    model = GpuInstanceUsageSegmentRow
    query = select(model).where(model.owner_id == owner, model.state.in_(states or ("open", "reconciling", "closed")))
    if account is not None:
        query = query.where(model.account_id == account)
    return [row for row in session.scalars(query) if (row.data or {}).get("session_kind")]


def _session_id(owner, instance_id):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"colab-gpu-usage:{owner}:{instance_id}:instance-lifetime:{instance_id}"))


def _kind(instance, job):
    snapshot = job.get("snapshot") or {}
    if instance.get("native_comfy") or snapshot.get("native_comfy"):
        return "native_comfy"
    if job.get("prep"):
        return "prep"
    if snapshot.get("system_smoke") or snapshot.get("workflow_smoke") or job.get("workflow_smoke") or job.get("smoke") or job.get("smoke_test_kind") or "smoke" in str(job.get("workflow_key", "")):
        return "smoke"
    policy = job.get("execution_policy") or {}
    if job.get("creation_fingerprint") or (policy.get("lifecycle") or policy.get("mode") or instance.get("retention_mode")) in {"smart", "retain", "batch"}:
        return "smart_reuse"
    return "single_job"


def _refresh(session, row, now, **patch):
    data = {**row.data, **patch}
    end = data.get("ended_at")
    elapsed = max(0.0, (now if end is None else end) - data["started_at"])
    rate = data.get("unit_rate")
    actual = data["actual_cu"]
    jobs = {j["job_id"]: j for j in data.get("jobs", [])}
    current = {}
    for job in session.scalars(select(GpuJobRow).where(
        GpuJobRow.owner_id == row.owner_id, GpuJobRow.instance_id == row.instance_id,
    ).order_by(GpuJobRow.created_at, GpuJobRow.id).with_for_update()):
        details = job.data or {}
        start = details.get("execution_started_at")
        seconds = details.get("execution_seconds")
        if seconds is None and start is not None:
            seconds = max(0, (now if end is None else end) - start)
        jobs[job.id] = {"job_id": job.id, "workflow_key": details.get("workflow_key"),
                        "state": job.state, "execution_seconds": seconds or 0}
        current[job.id] = job
    if data["session_kind"] == "single_job" and len(jobs) > 1:
        data["session_kind"] = "smart_reuse"
    total = max(elapsed, sum(job["execution_seconds"] for job in jobs.values()))
    remainder = round(actual * 1_000_000)
    for item in jobs.values():
        proportional = data["session_kind"] == "smart_reuse"
        units = round(actual * item["execution_seconds"] / total * 1_000_000) if proportional and total else (
            remainder if not proportional and item["job_id"] == data.get("job_id") else 0
        )
        units = min(units, remainder)
        remainder -= units
        item["actual_cu"] = units / 1_000_000
        job = current.get(item["job_id"])
        if job is not None:
            job_details = {**job.data, "actual_units": item["actual_cu"], "ccu_source": "google_balance_ledger",
                           "ccu_attribution": data.get("attribution")}
            if data.get("ccu_after") is not None:
                job_details.update(ccu_after=data["ccu_after"], ccu_after_at=data["ccu_after_at"])
            if job_details != job.data:
                job.data = job_details
                job.version += 1
    data.update(elapsed_minutes=elapsed / 60, duration_minutes=elapsed / 60,
                estimated_cu=round(elapsed / 60 * rate, 6) if rate is not None else None,
                actual_units=actual, idle_cu=remainder / 1_000_000, jobs=list(jobs.values()))
    data["estimated_units"] = data["estimated_cu"]
    row.data = data


def _seal(session, rows, now, *, all_pending=False):
    for row in rows:
        if row.state == "reconciling" and (all_pending or row.data["reconcile_until"] <= now):
            row.state = "closed"
            _refresh(session, row, now)


def _open_session(session, instance, job, lease, started_at):
    data = instance.data or {}
    account = instance.account_id
    if data.get("provider") != "colab" or not account or not lease.get("handle") or lease.get("account_id") != account:
        return None
    key = _session_id(instance.owner_id, instance.id)
    row = session.get(GpuInstanceUsageSegmentRow, key)
    if row is not None and row.data.get("session_kind"):
        return row
    raw_spec = data.get("runtime_spec_reported") or lease.get("selected_spec") or data.get("runtime_spec_requested")
    try:
        spec = spec_key(raw_spec)
    except ValueError:
        spec = (raw_spec.get("key") or raw_spec) if isinstance(raw_spec, dict) else {}
    rate = minute_rate(spec)
    if instance.state != "deleted":
        instance.data = {**data, "unit_rate": rate}
        instance.version += 1
    if row is None:
        row = GpuInstanceUsageSegmentRow(id=key, owner_id=instance.owner_id, project_id=instance.project_id,
                                         instance_id=instance.id, account_id=account)
        session.add(row)
    row.state = "open"
    row.data = {
        "segment_key": f"instance-lifetime:{instance.id}", "session_kind": _kind(data, job),
        "job_id": job.get("id") or lease.get("owner_job_id"), "started_at": started_at, "local_date": local_date(started_at),
        "spec_key": spec, "gpu_label": gpu_label(spec), "unit_rate": rate, "ccu_before": job.get("ccu_before"),
        "actual_cu": 0.0, "estimated_cu": 0.0 if rate is not None else None, "jobs": [], "idle_cu": 0.0,
    }
    return row


def open_instance_session(repo, job, lease, now):
    instance_id = lease.get("instance_id") or job.get("instance_id")
    if lease.get("provider") != "colab" or not instance_id or not lease.get("account_id") or not lease.get("handle"):
        return
    owner, account = lease["owner_id"], lease["account_id"]
    with transaction(repo, owner) as session:
        instance = session.get(GpuInstanceRow, instance_id, with_for_update=True)
        if not instance or instance.owner_id != owner or instance.state == "deleted":
            return
        row = _open_session(session, instance, job, lease, now)
        if row is not None and row in session.new:
            _seal(session, _segments(session, owner, account, ("reconciling",)), now, all_pending=True)


def gpu_label(spec):
    accel = str(spec.get("accelerator") or "").upper()
    if accel == "T4" and "HIGHMEM" in str(spec.get("shape") or "").upper():
        return "T4 HIGHMEM"
    if accel in {"NONE", "CPU"} or spec.get("variant") in {"CPU", "VARIANT_CPU"}:
        return "CPU"
    return accel or "未知规格"


def close_instance_session(repo, owner, instance_id, ended_at, reason, *, session=None):
    """Start reconciliation in the deletion transaction when one is available."""
    if not instance_id:
        return None
    if session is None:
        with transaction(repo, owner) as current:
            account = close_instance_session(repo, owner, instance_id, ended_at, reason, session=current)
        if account:
            sample_and_record(repo, owner, account, "delete_after")
        return None

    row = session.get(GpuInstanceUsageSegmentRow, _session_id(owner, instance_id))
    if row is None or not row.data.get("session_kind"):
        # Create missing sessions for confirmed runtimes that predate the ledger.
        instance = session.get(GpuInstanceRow, instance_id)
        if not instance or instance.owner_id != owner:
            return None
        lease_id = instance.data.get("lease_id")
        lease = session.get(GpuLeaseRow, lease_id) if lease_id else None
        if not lease or lease.owner_id != owner or lease.instance_id != instance.id:
            return None
        job_id = lease.data.get("owner_job_id")
        job = session.get(GpuJobRow, job_id) if job_id else None
        details = {**job.data, "id": job.id} if job and job.owner_id == owner else {}
        start = details.get("started_epoch") or lease.data.get("create_started") or instance.data.get("allocation_requested_at")
        if start is not None:
            row = _open_session(session, instance, details, lease.data, start)
    if row is None or row.state != "open":
        return None
    row.state = "reconciling"
    _refresh(session, row, ended_at, ended_at=ended_at, reconcile_until=ended_at + 120, close_reason=reason)
    return row.account_id


@contextmanager
def _sample_lock(repo, owner, account, source):
    identity = f"studio-ccu:{owner}:{account}"
    deadline = time.monotonic() + (0 if source == "tick" else 10)
    while True:
        with repo.store.upload_lock(identity) as acquired:
            if acquired:
                yield True
                return
        if time.monotonic() >= deadline:
            yield False
            return
        time.sleep(0.05)


def sample_and_record(repo, owner, account, source, *, sampler=None, now=None):
    if not account or (sampler is None and not sampling_enabled()):
        return None
    real_sampler = sampler is None and sample_account_balance is _DEFAULT_SAMPLE_ACCOUNT_BALANCE
    if real_sampler:
        # Never query Colab CU balance for Drive-only or unlinked accounts (manual or tick).
        if not has_colab_credential(repo, owner, account):
            return None
        if (
            source == "tick"
            and os.getenv("NATIVE_FAST_CONTROL_PLANE", "1").strip().lower() not in {"0", "false", "off", "no"}
        ):
            check_now = time.time() if now is None else float(now)
            with repo.store.db_lock(), repo.store.Session() as session:
                prev = setting(session, _snapshot_key(owner, account))
            if (
                prev.get("error") == "sample_failed"
                and check_now - float(prev.get("attempted_at") or 0) < 300
                and active_job_count(repo, owner, account) == 0
                and active_lease_count(repo, owner, account) == 0
            ):
                return None
    with _sample_lock(repo, owner, account, source) as acquired:
        if not acquired:
            return None
        try:
            sample = (sampler or sample_account_balance)(repo, owner, account) or {}
        except Exception:
            sample = {}
        moment = now if now is not None else sample.get("at", time.time())
        balance = _number(sample, ("balance",))
        with transaction(repo, owner) as session:
            key = _snapshot_key(owner, account)
            previous = setting(session, key)
            if balance is None:
                setting(session, key, {**previous, "error": "sample_failed", "attempted_at": moment}, owner)
                return None
            if moment < (previous.get("sampled_at") or 0):
                return previous
            rows = _segments(session, owner, account, ("open", "reconciling"))
            _seal(session, rows, moment)
            opened = [row for row in rows if row.state == "open"]
            pending = [row for row in rows if row.state == "reconciling"]
            before = previous.get("balance")
            delta = round(max(0, before - balance), 6) if before is not None else 0.0
            event = "baseline" if before is None else "spend" if balance < before else "topup" if balance > before else "snapshot"
            snapshot = {
                "sampled_at": moment, "balance": balance, "rate_hourly": _number(sample, ("rate_hourly", "rate")),
                "assignments": _number(sample, ("assignments",)), "error": None,
                "active_instance_ids": sorted(row.instance_id for row in opened),
                "reconciling_instance_ids": sorted(row.instance_id for row in pending),
            }
            last_written = previous.get("last_written")
            changed = any(snapshot[k] != previous.get(k) for k in ("active_instance_ids", "reconciling_instance_ids", "assignments"))
            if event != "snapshot" or changed or last_written is None or moment - last_written >= 600:
                session.add(GpuCcuLedgerRow(
                    id=uuid.uuid4().hex, owner_id=owner, account_id=account, local_date=local_date(moment), state=event,
                    data={**snapshot, "prev_balance": before, "delta_spent": delta, "source": source},
                ))
                last_written = moment
            setting(session, key, {**snapshot, "last_written": last_written}, owner)
            candidates = opened or pending
            if opened:
                for row in pending:
                    _refresh(session, row, moment, ccu_after=balance, ccu_after_at=moment)
            sole_live_rate = (snapshot["rate_hourly"] if len(opened) == 1
                              and (snapshot["assignments"] or 0) <= 1 else None)
            rates = [minute_rate(row.data.get("spec_key"), live_rate_hourly=sole_live_rate)
                     if row.state == "open" else row.data.get("unit_rate") for row in candidates]
            weights = [rate or 0 for rate in rates]
            total = sum(weights)
            remaining = round(delta * 1_000_000)
            for index, row in enumerate(candidates):
                part = remaining if index == len(candidates) - 1 else min(remaining, round(
                    delta * 1_000_000 * (weights[index] / total if total else 1 / len(candidates)),
                ))
                remaining -= part
                patch = {"unit_rate": rates[index]}
                if row.state == "reconciling":
                    patch.update(ccu_after=balance, ccu_after_at=moment)
                if part:
                    patch.update(actual_cu=round(row.data["actual_cu"] + part / 1_000_000, 6),
                                 attribution="exclusive" if len(candidates) == 1 else "prorated")
                _refresh(session, row, moment, **patch)
            return snapshot


def _detect(snapshots: dict[str, dict], previous: list[dict], now: float) -> list[dict]:
    """An assignment with no job behind it is a leaked runtime, i.e. money on fire."""
    known = {item.get("account_id"): item for item in previous if isinstance(item, dict)}
    anomalies: list[dict] = []
    for account_id, snapshot in snapshots.items():
        assignments = snapshot.get("assignments") or 0
        leaked = (
            assignments > 0
            and not snapshot.get("active_jobs")
            and not snapshot.get("active_leases")
        )
        if not leaked:
            continue
        previous_item = known.get(account_id) or {}
        anomalies.append(
            {
                "account_id": account_id,
                "alias": snapshot.get("alias"),
                "assignments": assignments,
                "balance": snapshot.get("balance"),
                "rate": snapshot.get("rate_hourly"),
                "first_seen": previous_item.get("first_seen") or now,
                "last_seen": now,
                "resolved_at": None,
            }
        )
    active_accounts = {item["account_id"] for item in anomalies}
    for account_id, previous_item in known.items():
        if account_id in active_accounts or previous_item.get("resolved_at"):
            continue
        anomalies.append({**previous_item, "resolved_at": now})
    anomalies.sort(key=lambda item: item.get("last_seen") or 0, reverse=True)
    return anomalies[:20]


def balance_snapshots(repo, owner_id, *, sampler=None, source="manual", now=None):
    real_sampler = sampler is None and sample_account_balance is _DEFAULT_SAMPLE_ACCOUNT_BALANCE
    for account in accounts(repo, owner_id):
        if real_sampler and not has_colab_credential(repo, owner_id, account):
            continue
        sample_and_record(repo, owner_id, account, source, sampler=sampler, now=now)
    current = snapshots(repo, owner_id)
    if real_sampler:
        return {key: value for key, value in current.items() if value.get("colab_connected")}
    return current


def sync_sessions(repo, owner, now):
    """Recover live sessions and confirmed closures across restarts and upgrades."""
    instances = {row["id"]: row for row in repo.list("instances", owner) if row.get("provider") == "colab"}
    if not instances:
        return
    with repo.store.db_lock(), repo.store.Session() as session:
        rows = {row.instance_id: row.state for row in _segments(session, owner)}
    for lease in repo.list("leases", owner):
        instance_id = lease.get("instance_id")
        instance = instances.get(instance_id)
        if not instance or instance.get("lease_id") != lease["id"]:
            continue
        if rows.get(instance_id) == "open" and (instance.get("runtime_deleted") or lease.get("remote_deleted")):
            close_instance_session(repo, owner, instance_id, instance.get("deletion_confirmed_at") or lease.get("remote_deleted_at") or now,
                                   lease.get("remote_delete_source") or "confirmed_delete")
        elif instance_id not in rows and instance["state"] != "deleted" and lease.get("state") != "deleted" and lease.get("handle"):
            job = repo.get("jobs", lease.get("owner_job_id"), owner) or {}
            open_instance_session(repo, job, lease, job.get("started_epoch") or instance.get("allocation_requested_at") or now)


def tick_ccu(repo, owner_id, *, sampler=None, now=None):
    moment = time.time() if now is None else now
    sync_sessions(repo, owner_id, moment)
    values = balance_snapshots(repo, owner_id, sampler=sampler, source="tick", now=now)
    for key, snapshot in values.items():
        snapshot["active_jobs"] = active_job_count(repo, owner_id, key)
        snapshot["active_leases"] = active_lease_count(repo, owner_id, key)
    with transaction(repo, owner_id) as session:
        key = f"cu-anomalies:{owner_id}"
        anomalies = _detect(values, setting(session, key).get("items", []), moment)
        setting(session, key, {"items": anomalies}, owner_id)
        rows = _segments(session, owner_id, states=("open", "reconciling"))
        _seal(session, rows, moment)
        for row in rows:
            if row.state == "open" and (row.data.get("elapsed_minutes") is None or row.account_id not in values or values[row.account_id].get("error")):
                _refresh(session, row, moment)
    return {"snapshots": values, "updated_at": moment, "anomalies": anomalies}


def summarize(values):
    balances = [s["balance"] for s in values.values() if s.get("balance") is not None]
    return {
        "accounts": len(values), "total_balance": round(sum(balances), 6) if balances else None,
        "assignments": sum(int(s.get("assignments") or 0) for s in values.values()),
        "sampled_at": max((s.get("sampled_at") or 0 for s in values.values()), default=0) or None,
    }


def daily_spend(session, owner, date):
    ledger = GpuCcuLedgerRow
    spent = session.scalar(select(func.sum(ledger.data["delta_spent"].as_float())).where(
        ledger.owner_id == owner, ledger.local_date == date, ledger.state == "spend",
    ))
    return round(spent or 0, 6)


def sessions_page(repo, owner, *, date=None, start_date=None, end_date=None, kind="all", limit=10, offset=0, active=False):
    now = time.time()
    model = GpuInstanceUsageSegmentRow
    kind_col = model.data["session_kind"].as_string()
    day_col = model.data["local_date"].as_string()
    conditions = [
        model.owner_id == owner, kind_col.is_not(None),
        model.state.in_(("open",) if active else ("closed", "reconciling")),
    ]
    if date:
        conditions.append(day_col == date)
    if start_date:
        conditions.append(day_col >= start_date)
    if end_date:
        conditions.append(day_col <= end_date)
    if kind != "all":
        conditions.append(kind_col == kind)
    with repo.store.db_lock(), repo.store.Session() as session:
        total, total_actual = session.execute(
            select(func.count(), func.coalesce(func.sum(model.data["actual_cu"].as_float()), 0.0))
            .select_from(model).where(*conditions)
        ).one()
        query = select(model).where(*conditions).order_by(
            model.data["started_at"].as_float().desc(), model.instance_id.desc(),
        ).limit(limit).offset(offset)
        values = []
        for row in session.scalars(query):
            data = row.data
            instance = session.get(GpuInstanceRow, row.instance_id) if active else None
            elapsed = max(0, now - data["started_at"]) / 60 if active else data["duration_minutes"]
            values.append({**data, "instance_id": row.instance_id, "account_id": row.account_id, "state": row.state,
                           "elapsed_minutes": elapsed, "draining": bool(instance and instance.data.get("native_draining")),
                           "estimated_cu": round(elapsed * data["unit_rate"], 6) if data.get("unit_rate") is not None else None})
    return {
        "sessions": values,
        "total": int(total or 0),
        "total_actual_cu": round(float(total_actual or 0.0), 6),
        "limit": limit,
        "offset": offset,
    }


def sessions(repo, owner, *, date=None, start_date=None, end_date=None, kind="all", limit=10, offset=0, active=False):
    page = sessions_page(
        repo, owner, date=date, start_date=start_date, end_date=end_date,
        kind=kind, limit=limit, offset=offset, active=active,
    )
    return page["sessions"], page["total"]


def _collect_daily_entries(session, rows, owner, *, today=None, start_date=None, end_date=None):
    daily = {}
    by_account = {}

    def add(day, account, units):
        bucket = daily.setdefault(day, {"date": day, "actual_cu": 0.0, "sessions": 0})
        bucket["actual_cu"] += units
        if account:
            totals = by_account.setdefault(account, {"today_actual_cu": 0.0, "history_actual_cu": 0.0})
            totals["history_actual_cu"] += units
            if today and day == today:
                totals["today_actual_cu"] += units

    cutover = setting(session, "cu-ledger-cutover").get("epoch")
    ledger = GpuCcuLedgerRow
    ledger_conds = [ledger.owner_id == owner, ledger.state == "spend"]
    if start_date:
        ledger_conds.append(ledger.local_date >= start_date)
    if end_date:
        ledger_conds.append(ledger.local_date <= end_date)
    totals = select(ledger.local_date, ledger.account_id, func.sum(ledger.data["delta_spent"].as_float())).where(
        *ledger_conds,
    ).group_by(ledger.local_date, ledger.account_id)
    for day, account, units in session.execute(totals):
        add(day, account, units)

    segments = GpuInstanceUsageSegmentRow
    day_col = segments.data["local_date"].as_string()
    seg_conds = [segments.owner_id == owner, segments.data["session_kind"].as_string().is_not(None)]
    if start_date:
        seg_conds.append(day_col >= start_date)
    if end_date:
        seg_conds.append(day_col <= end_date)
    counts = select(day_col, func.count()).where(*seg_conds).group_by(day_col)
    for day, count in session.execute(counts):
        if day:
            daily.setdefault(day, {"date": day, "actual_cu": 0.0, "sessions": 0})["sessions"] = int(count or 0)

    legacy_measured = []
    for row in rows:
        if row.get("mock") or row.get("simulated") or row.get("provider") == "mock" or row.get("units_actual") is None:
            continue
        row_moment = row.get("created_epoch") or row.get("stopped_at")
        if row_moment is None:
            row_moment = datetime.fromisoformat(row["created_at"]).timestamp()
        if cutover is None or row_moment < cutover:
            # Legacy usage contributes to history. The daily fuse uses only account ledger debits.
            day = local_date(row_moment)
            if start_date and day < start_date:
                continue
            if end_date and day > end_date:
                continue
            bucket = daily.setdefault(day, {"date": day, "actual_cu": 0.0, "sessions": 0})
            bucket["legacy_actual_cu"] = round(bucket.get("legacy_actual_cu", 0.0) + row["units_actual"], 6)
            legacy_measured.append(row)

    entries = [
        {**entry, "actual_cu": round(entry["actual_cu"] + entry.get("legacy_actual_cu", 0), 6)}
        for _, entry in sorted(daily.items(), reverse=True)
    ]
    today_ledger_cu = round(daily.get(today, {}).get("actual_cu", 0), 6) if today else 0.0
    return entries, by_account, legacy_measured, today_ledger_cu


def daily_history_page(repo, owner, *, start_date=None, end_date=None, limit=10, offset=0):
    rows = repo.list("usage", owner)
    with repo.store.db_lock(), repo.store.Session() as session:
        entries, _, _, _ = _collect_daily_entries(
            session, rows, owner, start_date=start_date, end_date=end_date,
        )
    total = len(entries)
    return {
        "days": entries[offset : offset + limit],
        "total": total,
        "total_actual_cu": round(sum(entry["actual_cu"] for entry in entries), 6),
        "total_sessions": sum(int(entry.get("sessions") or 0) for entry in entries),
        "limit": limit,
        "offset": offset,
    }


def usage_summary(repo, owner):
    from .cu_guard import CuGuardService

    moment = time.time()
    date = local_date(moment)
    rows = repo.list("usage", owner)
    values = snapshots(repo, owner)
    guard_service = CuGuardService(repo, lambda: moment)
    with repo.store.db_lock(), repo.store.Session() as session:
        anomalies = setting(session, f"cu-anomalies:{owner}").get("items", [])
        daily_history, by_account, legacy_measured, today_ledger_cu = _collect_daily_entries(
            session, rows, owner, today=date,
        )
        guard = guard_service.state_in(session, owner, today=today_ledger_cu)
        active_count = session.scalar(select(func.count()).select_from(GpuLeaseRow).where(
            GpuLeaseRow.owner_id == owner, GpuLeaseRow.state != "deleted",
        ))
    active, _ = sessions(repo, owner, active=True, limit=100)
    recent_page = sessions_page(repo, owner)
    recent = recent_page["sessions"]
    recent_count = recent_page["total"]
    below = {row["account_id"] for row in guard["hard_tripped_accounts"]}
    history = round(sum(entry["actual_cu"] for entry in daily_history), 6)
    colab_values = {key: value for key, value in values.items() if value.get("colab_connected")}
    if colab_values:
        ccu_values = colab_values
    else:
        ccu_values = {
            key: value
            for key, value in values.items()
            if value.get("sampled_at") is not None or value.get("balance") is not None
        }
    return {
        "timezone": settings.studio_timezone, "local_date": date,
        "today_actual_cu": guard["today_actual_cu"], "history_actual_cu": history,
        "daily_history": daily_history,
        "active_sessions": active, "recent_sessions": recent, "recent_sessions_count": recent_count,
        "recent_sessions_actual_cu": recent_page["total_actual_cu"], "active_runtimes": active_count,
        "guard": guard,
        "accounts": [{
            "account_id": key, "alias": value.get("alias"), "balance": value.get("balance"),
            "rate_hourly": value.get("rate_hourly"), "assignments": value.get("assignments"),
            "sampled_at": value.get("sampled_at"), "error": value.get("error"), "below_floor": key in below,
            "colab_connected": bool(value.get("colab_connected")),
            **{k: round(by_account.get(key, {}).get(k, 0), 6) for k in ("today_actual_cu", "history_actual_cu")},
        } for key, value in values.items()],
        "actual_units": history, "daily_limit": guard["daily_limit_cu"], "ledger": rows,
        "estimated_units": round(sum(row["estimated_cu"] or 0 for row in active + recent), 6),
        "reconciliation": {
            "measured_jobs": len(legacy_measured),
            "unmeasured_jobs": sum(1 for row in rows if row.get("units_actual") is None and not row.get("mock")),
            "measured_actual_units": round(sum(row["units_actual"] for row in legacy_measured), 6) if legacy_measured else None,
            "variance_units": round(sum(
                row["units_actual"] - (row.get("units_estimate", row.get("units")) or 0) for row in legacy_measured
            ), 6) if legacy_measured else None,
        },
        "ccu": {**summarize(ccu_values), "snapshots": ccu_values, "anomalies": anomalies},
    }
