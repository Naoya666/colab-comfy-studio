"""Persistent instance allocation and lifecycle domain service.

The service is deliberately repository-facing rather than provider-facing.  It
makes the short, durable admission decision for the account and instance,
then leaves provider I/O to the scheduler.  Every state change increments the
row version and uses a conditional SQL UPDATE, so a process restart or a
second scheduler cannot silently steal an instance.
"""

from __future__ import annotations

import copy
import math
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Mapping

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from app.common import APIError
from app.db import utc_isoformat

from .manifest import digest

POOL_NATIVE = "native"
POOL_CREATION = "creation"
POOL_EPHEMERAL = "ephemeral"
VALID_POOLS = frozenset({POOL_NATIVE, POOL_CREATION, POOL_EPHEMERAL})


def job_pool(job: Mapping | None) -> str:
    if not isinstance(job, Mapping):
        return POOL_CREATION
    explicit = job.get("instance_pool")
    if explicit in VALID_POOLS:
        return str(explicit)
    snapshot = job.get("snapshot") if isinstance(job.get("snapshot"), Mapping) else {}
    if (
        snapshot.get("native_comfy") is True
        or job.get("native_comfy") is True
        or job.get("model_id") == "native-comfy"
        or job.get("workflow_key") == "system/native-comfy"
    ):
        return POOL_NATIVE
    if (
        snapshot.get("system_smoke") is True
        or job.get("system_smoke") is True
        or snapshot.get("workflow_smoke") is True
        or job.get("workflow_smoke") is True
        or snapshot.get("prep") is True
        or job.get("prep") is True
        or job.get("model_id") == "system-cuda-smoke"
        or job.get("workflow_key") == "system/cuda-smoke"
    ):
        return POOL_EPHEMERAL
    return POOL_CREATION


def instance_pool_of(instance: Mapping | None) -> str:
    if not isinstance(instance, Mapping):
        return POOL_CREATION
    explicit = instance.get("instance_pool")
    if explicit in VALID_POOLS:
        return str(explicit)
    if instance.get("native_comfy") is True:
        return POOL_NATIVE
    req = instance.get("runtime_spec_requested")
    key = req.get("key", req) if isinstance(req, Mapping) else None
    if isinstance(key, Mapping) and key.get("variant") in {"CPU", "VARIANT_CPU"}:
        return POOL_EPHEMERAL
    return POOL_CREATION


def evaluate_lifetime_shortening(
    instance_data: Mapping,
    *,
    is_busy: bool,
    new_lifecycle: str,
    new_retain_minutes: float | int,
    now: float,
    lease_idle_deadline: float | None = None,
) -> tuple[bool, list[str]]:
    """Dual-criteria lifetime shortening check (Spec §5.2): policy config + actual expiry."""
    reasons: list[str] = []
    inst_mode = str(instance_data.get("retention_mode") or "save")
    raw_inst_retain = instance_data.get("retain_idle_minutes")
    inst_retain_min = float(raw_inst_retain) if raw_inst_retain is not None else None
    new_retain_val = float(new_retain_minutes) if new_retain_minutes is not None else 0.0

    # Standard A: policy configuration shortening
    if inst_mode in {"smart", "retain"} and new_lifecycle in {"save", "batch"}:
        reasons.append("当前实例处于保留策略，本次任务选择节省/批次模式，任务结束后将立即释放该实例")
    elif (
        inst_mode in {"smart", "retain"}
        and new_lifecycle in {"smart", "retain"}
        and inst_retain_min is not None
        and new_retain_val < inst_retain_min
    ):
        reasons.append(f"空闲保留时长将从 {int(round(inst_retain_min))} 分钟缩短为 {int(round(new_retain_val))} 分钟")

    # Standard B: actual expiry timestamp shortening
    if is_busy:
        current_expiry = (
            now + inst_retain_min * 60.0
            if inst_mode in {"smart", "retain"} and inst_retain_min is not None
            else None
        )
    else:
        candidates = [
            float(val)
            for val in (instance_data.get("idle_deadline_at"), lease_idle_deadline)
            if val is not None
        ]
        current_expiry = min(candidates) if candidates else None

    new_expiry = now if new_lifecycle in {"save", "batch"} else now + new_retain_val * 60.0
    if current_expiry is not None and current_expiry > now and new_expiry < current_expiry:
        remaining_min = max(1, int(math.ceil((current_expiry - now) / 60.0)))
        if new_lifecycle in {"save", "batch"}:
            msg = f"该实例当前还有约 {remaining_min} 分钟保留余量，本次任务结束后将提前释放"
        else:
            msg = f"本次任务预计到期时间早于该实例当前剩余保留时间（剩余约 {remaining_min} 分钟）"
        if msg not in reasons:
            reasons.append(msg)

    return bool(reasons), reasons


class InstanceServiceError(RuntimeError):
    """Base class for safe, domain-level allocation errors."""


class AllocationError(InstanceServiceError):
    """The requested account/instance cannot legally be allocated."""


class InstanceConflict(InstanceServiceError):
    """The resource changed state or is owned by another active operation."""


class CreationInstancePending(InstanceConflict):
    """An eligible Studio VM is still starting or finishing an earlier job."""


class InstanceService:
    """Persistent instance state machine used by both API and Scheduler.

    ``repo`` is the GPU ``Repository``.  The implementation only relies on
    its public store/session shape plus the model registry, which lets a
    migration register the v0.2 tables without changing this domain service.
    """

    _INSTANCE_STATES = {"requested", "allocating", "ready", "busy", "retained_idle", "releasing", "deleted", "unknown", "cleanup_pending", "quarantined"}
    _IDLE_STATES = {"ready", "retained_idle"}
    _ACTIVE_CREATION_STATES = {"retained_idle", "ready", "allocating", "busy"}
    _RUNTIME_CHECK_INTERVAL = 45
    _RUNTIME_CHECK_MAX_BACKOFF = 300
    _SENSITIVE = {"token", "access_token", "refresh_token", "connectioninfo", "connection_info", "handle", "secret", "password", "authorization", "signed_url"}
    _PUBLIC = {
        "id", "owner_id", "project_id", "state", "version", "account_id", "account_alias_snapshot", "provider",
        "instance_pool", "instance_label", "runtime_spec_requested", "runtime_spec_reported", "gpu_reported", "gpu_diagnostics", "current_job_id",
        "lease_id", "hard_deadline_at", "idle_deadline_at", "retention_mode", "retain_idle_minutes", "manual_stop_requested",
        "creation_runtime", "created_at", "updated_at", "last_seen_at", "generation", "selection_reason", "runtime_deleted",
        "allocation_key", "budget_reserved", "retention_reserved", "allocation_requested_at", "deletion_confirmed_at",
        "remote_delete_source", "history_cleared_at",
    }

    def __init__(self, repo, clock=time.time, config=None):
        self.repo = repo
        self.store = repo.store
        self.clock = clock
        self.config = config

    # ---- repository/model plumbing -------------------------------------------------
    def _model(self, kind):
        return self.repo._model(kind)

    @contextmanager
    def _tx(self):
        with self.store.db_lock(), self.store.Session.begin() as session:
            yield session

    @contextmanager
    def _admission(self):
        with self.store.admission_lock(timeout=float(self._cfg("admission_timeout_seconds", 1.0))) as acquired:
            if not acquired:
                raise InstanceConflict("资源分配正在进行，请稍后重试")
            yield

    @staticmethod
    def _data(row) -> dict:
        return copy.deepcopy(getattr(row, "data", None) or {})

    def _dict(self, row) -> dict:
        data = self._data(row)
        result = {"id": row.id, "owner_id": row.owner_id, "project_id": row.project_id}
        result.update(data)
        result.update(
            state=row.state,
            version=row.version,
            created_at=utc_isoformat(row.created_at),
            updated_at=utc_isoformat(row.updated_at),
        )
        return result

    def _cfg(self, name, default=None):
        cfg = self.config
        if cfg is None:
            return default
        if isinstance(cfg, Mapping):
            return cfg.get(name, default)
        return getattr(cfg, name, default)

    def _utcnow(self):
        return datetime.now(timezone.utc)

    def _set(self, row, payload: dict, *, state: str | None = None, expected_version: int | None = None, session=None):
        """CAS a row and update its JSON/state/version in one SQL statement."""
        expected = row.version if expected_version is None else expected_version
        values = {"data": copy.deepcopy(payload), "version": expected + 1, "updated_at": self._utcnow()}
        if state is not None:
            values["state"] = state
        for field in ("account_id", "instance_id", "current_job_id"):
            if hasattr(type(row), field) and field in payload:
                values[field] = payload[field]
        stmt = update(type(row)).where(type(row).id == row.id, type(row).version == expected).values(**values)
        result = session.execute(stmt)
        if result.rowcount != 1:
            raise InstanceConflict("实例状态已被其他调度器更新")
        row.data = copy.deepcopy(payload)
        row.version = expected + 1
        row.updated_at = values["updated_at"]
        if state is not None:
            row.state = state
        return row

    def _find(self, session, kind, record_id, owner_id=None):
        model = self._model(kind)
        stmt = select(model).where(model.id == record_id)
        if owner_id is not None:
            stmt = stmt.where(model.owner_id == owner_id)
        return session.scalar(stmt)

    def _rows(self, session, kind, owner_id=None):
        model = self._model(kind)
        stmt = select(model)
        if owner_id is not None:
            stmt = stmt.where(model.owner_id == owner_id)
        return list(session.scalars(stmt).all())

    @staticmethod
    def _value(data, *names, default=None):
        for name in names:
            if name in data:
                return data[name]
        return default

    def _job_policy(self, job):
        return job.get("execution_policy") if isinstance(job.get("execution_policy"), Mapping) else {}

    def _provider(self, job):
        return str(job.get("provider") or self._job_policy(job).get("provider") or "mock")

    def _requested(self, job, name):
        policy = self._job_policy(job)
        return job.get(f"requested_{name}") or job.get(name) or policy.get(f"requested_{name}") or policy.get(name)

    def _attempt_key(self, job):
        job_id = job.get("id")
        if not isinstance(job_id, str) or not job_id:
            raise AllocationError("job id is required for persistent allocation")
        attempt = job.get("attempt_id") or job.get("attempt_number") or job.get("attempt") or 1
        return f"{job_id}:{attempt}", str(uuid.uuid5(uuid.NAMESPACE_URL, f"colab-gpu-instance:{job_id}:{attempt}"))

    # ---- account, budget and selection --------------------------------------------
    def _account_id(self, row):
        return self._value(self._data(row), "account_id", default=row.id)

    def _account_legal(self, row, owner_id, project_id, provider):
        if row.owner_id != owner_id:
            return False
        data = self._data(row)
        status = self._value(data, "auth_status", "status", "connection_status", default="connected")
        connected = self._value(data, "connected", default=status == "connected")
        if connected is not True or data.get("enabled", True) is not True or data.get("allow_allocation", True) is not True:
            return False
        allowed = self._value(data, "allowed_projects", "allowed_project_ids", "project_ids", "projects")
        if allowed and project_id not in set(allowed):
            return False
        account_provider = data.get("provider")
        if provider != "mock" and account_provider not in {None, provider, "colab", "google"}:
            return False
        return True

    def _make_mock_account(self, session, owner_id):
        model = self._model("accounts")
        account_id = f"mock-{uuid.uuid5(uuid.NAMESPACE_URL, f'colab-gpu-mock-account:{owner_id}').hex}"
        existing = session.scalar(select(model).where(model.id == account_id))
        if existing:
            return existing
        now = self._utcnow()
        payload = {
            "account_id": account_id,
            "alias": "Mock account",
            "provider": "mock",
            "connected": True,
            "auth_status": "connected",
            "enabled": True,
            "allow_allocation": True,
            "is_mock": True,
        }
        row = model(id=account_id, owner_id=owner_id, project_id=None, state="connected", data=payload, version=0, created_at=now, updated_at=now)
        session.add(row)
        session.flush()
        return row

    def _legal_accounts(self, session, owner_id, project_id, provider, requested):
        rows = self._rows(session, "accounts", owner_id)
        if requested:
            row = next((r for r in rows if self._account_id(r) == requested or r.id == requested), None)
            if row is None:
                raise AllocationError("指定账号不存在")
            if not self._account_legal(row, owner_id, project_id, provider):
                raise AllocationError("指定账号未连接、已禁用或不允许该项目")
            return [row]
        legal = [r for r in rows if self._account_legal(r, owner_id, project_id, provider)]
        if not legal and provider == "mock":
            legal = [self._make_mock_account(session, owner_id)]
        if not legal:
            raise AllocationError("没有可分配的授权账号")
        return sorted(legal, key=lambda row: self._account_id(row))

    def _cursor_id(self, owner_id, project_id):
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"colab-gpu-allocation-cursor:{owner_id}:{project_id or ''}"))

    def _choose_account(self, session, accounts, owner_id, project_id):
        if len(accounts) == 1:
            return accounts[0]
        model = self._model("allocation_cursors")
        cursor_id = self._cursor_id(owner_id, project_id)
        cursor = session.scalar(select(model).where(model.id == cursor_id))
        if cursor is None:
            now = self._utcnow()
            cursor = model(id=cursor_id, owner_id=owner_id, project_id=project_id, state="active", version=0, data={"next_index": 0}, created_at=now, updated_at=now)
            session.add(cursor)
            session.flush()
        data = self._data(cursor)
        index = int(data.get("next_index", 0)) % len(accounts)
        selected = accounts[index]
        data.update(next_index=(index + 1) % len(accounts), last_account_id=self._account_id(selected))
        self._set(cursor, data, state="active", session=session)
        return selected

    def _instance_pool(self, session, instance) -> str:
        data = self._data(instance)
        explicit = data.get("instance_pool")
        if explicit in VALID_POOLS:
            return str(explicit)
        if data.get("native_comfy") is True:
            return POOL_NATIVE
        req = data.get("runtime_spec_requested")
        key = req.get("key", req) if isinstance(req, Mapping) else None
        if isinstance(key, Mapping) and key.get("variant") in {"CPU", "VARIANT_CPU"}:
            return POOL_EPHEMERAL
        # Legacy fallback: inspect associated job(s) if available in DB session.
        owner_id = getattr(instance, "owner_id", None)
        if session is not None and owner_id:
            candidate_job_ids = []
            cur_job_id = data.get("current_job_id")
            if isinstance(cur_job_id, str) and cur_job_id:
                candidate_job_ids.append(cur_job_id)
            alloc_key = data.get("allocation_key")
            if isinstance(alloc_key, str) and ":" in alloc_key:
                first_id = alloc_key.split(":", 1)[0]
                if first_id and first_id not in candidate_job_ids:
                    candidate_job_ids.append(first_id)
            for jid in candidate_job_ids:
                try:
                    jrow = self._find(session, "jobs", jid, owner_id)
                except ValueError:
                    jrow = None
                if jrow is not None:
                    jpool = job_pool(self._data(jrow))
                    if jpool != POOL_CREATION:
                        return jpool
        return POOL_CREATION

    def _order_table(self, owner_id: str | None) -> list[dict[str, str]]:
        from .specs import DEFAULT_GPU_SPEC_ORDER, gpu_spec_order_settings

        if not owner_id or self.store is None:
            return copy.deepcopy(DEFAULT_GPU_SPEC_ORDER)
        try:
            return gpu_spec_order_settings(self.store, owner_id)["ordered_specs"]
        except Exception:
            return copy.deepcopy(DEFAULT_GPU_SPEC_ORDER)

    @staticmethod
    def _instance_spec(data: Mapping) -> dict | None:
        return data.get("runtime_spec_reported") or data.get("runtime_spec_requested")

    @staticmethod
    def _job_workflow_default_spec(job: Mapping) -> dict | None:
        manifest = (job.get("snapshot") or {}).get("manifest") if isinstance(job.get("snapshot"), Mapping) else None
        if isinstance(manifest, Mapping) and isinstance(manifest.get("compute_spec"), Mapping):
            return dict(manifest["compute_spec"])
        return None

    def _job_requested_spec(self, job: Mapping) -> dict | None:
        policy = self._job_policy(job)
        raw = job.get("requested_spec") or policy.get("requested_spec") or policy.get("compute_spec")
        if isinstance(raw, Mapping):
            return dict(raw)
        return None

    def _lease_is_reusable_idle(self, session, instance, data: Mapping, now: float) -> tuple[bool, dict | None]:
        lease_id = data.get("lease_id")
        if lease_id:
            owner_id = getattr(instance, "owner_id", None)
            lease = self._find(session, "leases", lease_id, owner_id)
            if lease is None or lease.state != "idle":
                return False, None
            lease_data = self._data(lease)
            if lease_data.get("current_job_id") is not None or lease_data.get("cleanup_claim_token"):
                return False, None
            if any(
                lease_data.get(k) is not None and now >= float(lease_data[k])
                for k in ("deadline_epoch", "idle_deadline_at")
            ):
                return False, None
            return True, lease_data
        # Real Colab instances MUST have an active idle lease to be reused;
        # isolated mock unit tests may create mock instances without a lease row.
        if data.get("provider") == "colab":
            return False, None
        return True, None

    def _compatible(
        self,
        session,
        instance,
        job,
        account_id,
        *,
        order_table=None,
        target_baseline=...,
        allow_busy: bool = False,
    ):
        data = self._data(instance)
        jpool = job_pool(job)
        if jpool == POOL_EPHEMERAL:
            return False
        ipool = self._instance_pool(session, instance)
        if ipool != jpool:
            return False
        if self._value(data, "account_id") != account_id:
            return False
        provider = self._provider(job)
        if data.get("provider", provider) != provider:
            return False
        if (
            data.get("native_draining")
            or data.get("manual_stop_requested")
            or data.get("runtime_deleted")
            or data.get("cleanup_claim_token")
        ):
            return False
        is_idle = instance.state in self._IDLE_STATES and data.get("current_job_id") is None
        if allow_busy:
            if instance.state not in self._ACTIVE_CREATION_STATES:
                return False
        else:
            if not is_idle:
                return False

        now = self.clock()
        if instance.state == "retained_idle":
            deadlines = [data[key] for key in ("idle_deadline_at", "hard_deadline_at") if data.get(key) is not None]
            if deadlines and now >= min(map(float, deadlines)):
                return False
        if ipool == POOL_NATIVE:
            if data.get("hard_deadline_at") is not None and now >= float(data["hard_deadline_at"]):
                return False
            if is_idle:
                ok_lease, _ = self._lease_is_reusable_idle(session, instance, data, now)
                if not ok_lease:
                    return False
            return True

        # Creation pool compatibility
        if is_idle:
            ok_lease, _ = self._lease_is_reusable_idle(session, instance, data, now)
            if not ok_lease:
                return False
            if data.get("lease_id") is None and provider == "colab":
                return False

        if (
            data.get("retention_mode") == "batch"
            and job.get("creation_fingerprint")
            and (data.get("creation_runtime") or {}).get("fingerprint")
            and (data.get("creation_runtime") or {}).get("fingerprint") != job.get("creation_fingerprint")
        ):
            return False

        snapshot = job.get("snapshot") if isinstance(job.get("snapshot"), Mapping) else {}
        manifest = snapshot.get("manifest") if isinstance(snapshot.get("manifest"), Mapping) else {}
        is_creation_api = bool(snapshot.get("creation_api"))
        actual = self._instance_spec(data)
        workflow_default = self._job_workflow_default_spec(job)
        requested = self._job_requested_spec(job)
        if target_baseline is ...:
            target_baseline = workflow_default if is_creation_api else (requested or workflow_default)

        if order_table is None:
            order_table = self._order_table(job.get("owner_id") or getattr(instance, "owner_id", None))

        if provider == "colab" or actual is not None or target_baseline is not None or workflow_default is not None:
            if actual is None:
                if provider == "colab":
                    return False
            else:
                from .specs import check_requirements, compare_specs

                if manifest:
                    try:
                        check_requirements(dict(manifest), dict(actual))
                    except ValueError:
                        return False
                try:
                    if workflow_default is not None and compare_specs(actual, workflow_default, order_table) not in {"equal", "higher"}:
                        return False
                    if target_baseline is not None and compare_specs(actual, target_baseline, order_table) not in {"equal", "higher"}:
                        return False
                except ValueError:
                    return False

        if provider == "mock" and actual is None and target_baseline is None:
            signature = job.get("compatibility_signature")
            if signature and data.get("compatibility_signature") != signature:
                return False
        return True

    def _candidate_rank_key(self, session, instance, job, *, target_baseline, order_table):
        from .specs import compare_specs, spec_rank

        data = self._data(instance)
        fingerprint = job.get("creation_fingerprint")
        runtime = data.get("creation_runtime") or {}
        warm_hit = bool(fingerprint and runtime.get("fingerprint") == fingerprint and runtime.get("process"))
        if not warm_hit and fingerprint and data.get("current_job_id"):
            cur_job = self._find(session, "jobs", data["current_job_id"], instance.owner_id)
            if cur_job is not None and self._data(cur_job).get("creation_fingerprint") == fingerprint:
                warm_hit = True
        actual = self._instance_spec(data)
        rel = "equal"
        if actual and target_baseline:
            try:
                rel = compare_specs(actual, target_baseline, order_table)
            except ValueError:
                rel = "unknown"
        tier1 = 0 if warm_hit else 1
        tier2 = 0 if rel == "equal" else 1
        rank = spec_rank(actual, order_table) if actual else None
        tier3 = rank if rank is not None else 999999
        is_idle = instance.state in self._IDLE_STATES and data.get("current_job_id") is None
        tier4_state = 0 if is_idle else 1
        ts_dt = getattr(instance, "updated_at", None) or getattr(instance, "created_at", None)
        ts = ts_dt.timestamp() if ts_dt is not None else 0.0
        return (tier1, tier2, tier3, tier4_state, -ts)

    def _active_creation_instances_in_tx(self, session, owner_id: str, provider: str, account_ids: set[str]):
        now = self.clock()
        active = []
        for inst in self._rows(session, "instances", owner_id):
            if inst.state not in self._ACTIVE_CREATION_STATES:
                continue
            data = self._data(inst)
            if self._instance_pool(session, inst) != POOL_CREATION:
                continue
            if data.get("provider", provider) != provider:
                continue
            if data.get("account_id") not in account_ids:
                continue
            if data.get("manual_stop_requested") or data.get("runtime_deleted") or data.get("cleanup_claim_token"):
                continue
            if inst.state == "retained_idle":
                deadlines = [data[k] for k in ("idle_deadline_at", "hard_deadline_at") if data.get(k) is not None]
                if deadlines and now >= min(map(float, deadlines)):
                    continue
            is_idle = inst.state in self._IDLE_STATES and data.get("current_job_id") is None
            if is_idle and provider == "colab":
                ok_lease, _ = self._lease_is_reusable_idle(session, inst, data, now)
                if not ok_lease:
                    continue
            active.append(inst)
        return active

    def _inflight_compatible_creation_jobs(self, jobs, job, account_ids: set[str], order_table) -> bool:
        from .specs import check_requirements, compare_specs

        manifest = (job.get("snapshot") or {}).get("manifest") or {}
        workflow_default = self._job_workflow_default_spec(job)
        active_states = {
            "queued", "validating", "waiting_for_capacity", "allocating",
            "preparing", "loading", "running", "uploading", "cleaning_up",
        }
        for row in jobs:
            if row.get("id") == job.get("id"):
                continue
            if row.get("cancel_requested") or row.get("state") not in active_states:
                continue
            if job_pool(row) != POOL_CREATION:
                continue
            if row.get("creation_existing_only"):
                continue
            if self._provider(row) != self._provider(job):
                continue
            row_policy = self._job_policy(row)
            row_lifecycle = row_policy.get("lifecycle") or row.get("lifecycle") or "save"
            if row_lifecycle not in {"smart", "retain"} and not (
                row_lifecycle == "batch"
                and row.get("creation_fingerprint")
                and row.get("creation_fingerprint") == job.get("creation_fingerprint")
            ):
                continue
            row_account = row.get("account_id") or row.get("requested_account_id")
            if row_account and row_account not in account_ids:
                continue
            row_spec = row.get("runtime_spec_reported") or row.get("requested_spec") or self._job_workflow_default_spec(row)
            if row_spec is not None and self._provider(job) == "colab":
                try:
                    if manifest:
                        check_requirements(dict(manifest), dict(row_spec))
                    if workflow_default is not None and compare_specs(row_spec, workflow_default, order_table) not in {"equal", "higher"}:
                        continue
                except ValueError:
                    continue
            return True
        return False

    def inspect_creation_api_admission(self, job, jobs) -> dict:
        """Classify Creation API pool state: reusable, low-spec conflict, or empty."""
        owner = job["owner_id"]
        requested = self._requested(job, "account_id") or self._job_policy(job).get("account_id")
        provider = self._provider(job)
        order_table = self._order_table(owner)
        workflow_default = self._job_workflow_default_spec(job)
        with self._tx() as session:
            accounts = self._legal_accounts(session, owner, job["project_id"], provider, requested)
            if provider == "colab":
                from .cu_guard import CuGuardService

                guard = CuGuardService(self.repo, self.clock)
                eligible = guard.filter_accounts(
                    guard.state_in(session, owner), [self._account_id(a) for a in accounts]
                )
                accounts = [a for a in accounts if self._account_id(a) in eligible]
            account_ids = {self._account_id(account) for account in accounts}
            active = self._active_creation_instances_in_tx(session, owner, provider, account_ids)
            compatible = [
                inst
                for inst in active
                if self._compatible(
                    session,
                    inst,
                    job,
                    self._data(inst).get("account_id"),
                    order_table=order_table,
                    target_baseline=workflow_default,
                    allow_busy=True,
                )
            ]
            if compatible:
                compatible.sort(
                    key=lambda inst: self._candidate_rank_key(
                        session, inst, job, target_baseline=workflow_default, order_table=order_table
                    )
                )
                return {"status": "compatible_instance", "instance_id": compatible[0].id}
            releasing = any(
                inst.state == "releasing"
                and self._instance_pool(session, inst) == POOL_CREATION
                and self._data(inst).get("provider", provider) == provider
                and self._data(inst).get("account_id") in account_ids
                for inst in self._rows(session, "instances", owner)
            )
            if active:
                return {"status": "spec_too_low", "instance_id": active[0].id}
            if self._inflight_compatible_creation_jobs(jobs, job, account_ids, order_table):
                return {"status": "inflight_compatible", "instance_id": None}
            if releasing:
                return {"status": "releasing", "instance_id": None}
            return {"status": "empty", "instance_id": None}

    def creation_capacity(self, job, jobs):
        """Check a reusable lease or an earlier job which can leave one for this request."""
        state = self.inspect_creation_api_admission(job, jobs)
        return state["status"] in {"compatible_instance", "inflight_compatible"}

    def _instance_summary(self, session, instance, job, *, order_table) -> dict:
        from .specs import spec_id, spec_key

        data = self._data(instance)
        now = self.clock()
        is_busy = instance.state in {"busy", "allocating"} or data.get("current_job_id") is not None
        actual = self._instance_spec(data)
        norm_spec = None
        sid = None
        if actual:
            try:
                norm_spec = spec_key(dict(actual))
                sid = spec_id(norm_spec)
            except ValueError:
                pass
        lease_idle_deadline = None
        if data.get("lease_id"):
            lease = self._find(session, "leases", data["lease_id"], instance.owner_id)
            if lease is not None:
                lease_idle_deadline = self._data(lease).get("idle_deadline_at")
        deadlines = [
            float(val)
            for val in (data.get("idle_deadline_at"), data.get("hard_deadline_at"), lease_idle_deadline)
            if val is not None
        ]
        expires_at = min(deadlines) if deadlines else None
        remaining = max(0.0, expires_at - now) if (not is_busy and expires_at is not None) else None
        runtime = data.get("creation_runtime") or {}
        fingerprint = job.get("creation_fingerprint")
        loaded_wf_key = runtime.get("workflow_key")
        loaded_wf_ver = runtime.get("workflow_version")
        loaded_fp = runtime.get("fingerprint")
        warm_hit = bool(fingerprint and loaded_fp == fingerprint and runtime.get("process"))
        if is_busy and data.get("current_job_id"):
            cur = self._find(session, "jobs", data["current_job_id"], instance.owner_id)
            if cur is not None:
                cur_data = self._data(cur)
                loaded_wf_key = loaded_wf_key or cur_data.get("workflow_key")
                loaded_wf_ver = loaded_wf_ver if loaded_wf_ver is not None else cur_data.get("workflow_version")
                loaded_fp = loaded_fp or cur_data.get("creation_fingerprint")
                if fingerprint and cur_data.get("creation_fingerprint") == fingerprint:
                    warm_hit = True
        return {
            "id": instance.id,
            "instance_id": instance.id,
            "label": data.get("instance_label"),
            "instance_label": data.get("instance_label"),
            "account_id": data.get("account_id"),
            "account_alias_snapshot": data.get("account_alias_snapshot"),
            "state": instance.state,
            "is_busy": is_busy,
            "spec": norm_spec,
            "spec_id": sid,
            "gpu_reported": data.get("gpu_reported"),
            "retention_mode": data.get("retention_mode") or "save",
            "retain_idle_minutes": data.get("retain_idle_minutes"),
            "idle_deadline_at": data.get("idle_deadline_at"),
            "hard_deadline_at": data.get("hard_deadline_at"),
            "expires_at": expires_at,
            "remaining_seconds": round(remaining, 1) if remaining is not None else None,
            "warm_hit": warm_hit,
            "loaded_workflow_key": loaded_wf_key,
            "loaded_workflow_version": loaded_wf_ver,
            "loaded_fingerprint": loaded_fp,
        }

    def preview_creation_reuse(
        self,
        owner_id: str,
        *,
        project_id: str | None,
        provider: str,
        manifest: Mapping,
        creation_fingerprint: str | None,
        selected_spec: Mapping | None,
        account_id: str | None,
        instance_id: str | None,
        lifecycle: str,
        retain_idle_minutes: int,
    ) -> dict:
        """Authoritative server-side reuse preview for frontend Create page (Spec §4.2 & §5.2)."""
        from .specs import check_requirements, compare_specs, spec_id, spec_key

        order_table = self._order_table(owner_id)
        workflow_default = manifest.get("compute_spec") if isinstance(manifest, Mapping) else None
        norm_wf_default = spec_key(dict(workflow_default)) if isinstance(workflow_default, Mapping) else None
        norm_selected = spec_key(dict(selected_spec)) if isinstance(selected_spec, Mapping) else norm_wf_default

        base_meta = {
            "workflow_default_spec": norm_wf_default,
            "selected_spec": norm_selected,
        }
        if provider == "colab" and norm_selected is not None:
            try:
                check_requirements(dict(manifest), {"key": norm_selected})
            except ValueError as exc:
                return {
                    **base_meta,
                    "decision": "unsupported_spec",
                    "reason": str(exc),
                    "target_instance": None,
                    "shortens_lifetime": False,
                    "shorten_reasons": [],
                    "allowed_actions": [],
                    "force_confirm": True,
                }
            if norm_wf_default is not None and compare_specs(norm_selected, norm_wf_default, order_table) == "lower":
                return {
                    **base_meta,
                    "decision": "unsupported_spec",
                    "reason": f"所选 GPU（{spec_id(norm_selected)}）低于工作流默认最低规格（{spec_id(norm_wf_default)}），不支持降配运行",
                    "target_instance": None,
                    "shortens_lifetime": False,
                    "shorten_reasons": [],
                    "allowed_actions": [],
                    "force_confirm": True,
                }

        probe_job = {
            "id": "preview",
            "owner_id": owner_id,
            "project_id": project_id,
            "provider": provider,
            "instance_pool": POOL_CREATION,
            "creation_fingerprint": creation_fingerprint,
            "requested_spec": {"key": norm_selected} if norm_selected else None,
            "snapshot": {"manifest": dict(manifest)},
            "execution_policy": {"provider": provider, "lifecycle": lifecycle, "retain_idle_minutes": retain_idle_minutes},
        }
        now = self.clock()
        with self._tx() as session:
            try:
                accounts = self._legal_accounts(session, owner_id, project_id, provider, account_id)
            except AllocationError:
                accounts = []
            if provider == "colab" and accounts:
                from .cu_guard import CuGuardService

                guard = CuGuardService(self.repo, self.clock)
                try:
                    eligible = guard.filter_accounts(
                        guard.state_in(session, owner_id), [self._account_id(a) for a in accounts]
                    )
                    accounts = [a for a in accounts if self._account_id(a) in eligible]
                except APIError:
                    accounts = []
            account_ids = {self._account_id(a) for a in accounts}
            active = self._active_creation_instances_in_tx(session, owner_id, provider, account_ids)
            if instance_id:
                active = [inst for inst in active if inst.id == instance_id]

            if not active:
                return {
                    **base_meta,
                    "decision": "create_new",
                    "reason": None,
                    "target_instance": None,
                    "shortens_lifetime": False,
                    "shorten_reasons": [],
                    "allowed_actions": ["create_new"],
                    "force_confirm": False,
                }

            compatible = [
                inst
                for inst in active
                if self._compatible(
                    session,
                    inst,
                    probe_job,
                    self._data(inst).get("account_id"),
                    order_table=order_table,
                    target_baseline=norm_selected,
                    allow_busy=True,
                )
            ]
            if compatible:
                compatible.sort(
                    key=lambda inst: self._candidate_rank_key(
                        session, inst, probe_job, target_baseline=norm_selected, order_table=order_table
                    )
                )
                best = compatible[0]
                best_data = self._data(best)
                summary = self._instance_summary(session, best, probe_job, order_table=order_table)
                is_busy = summary["is_busy"]
                actual = self._instance_spec(best_data)
                rel = "equal"
                if actual and norm_selected:
                    try:
                        rel = compare_specs(actual, norm_selected, order_table)
                    except ValueError:
                        rel = "equal"
                lease_idle_deadline = None
                if best_data.get("lease_id"):
                    lease = self._find(session, "leases", best_data["lease_id"], owner_id)
                    if lease is not None:
                        lease_idle_deadline = self._data(lease).get("idle_deadline_at")
                shortens, shorten_reasons = evaluate_lifetime_shortening(
                    best_data,
                    is_busy=is_busy,
                    new_lifecycle=lifecycle,
                    new_retain_minutes=retain_idle_minutes,
                    now=now,
                    lease_idle_deadline=lease_idle_deadline,
                )
                if rel == "higher":
                    return {
                        **base_meta,
                        "decision": "confirm_higher_spec",
                        "reason": f"创作池中已有更高规格实例（{summary['spec_id'] or summary['gpu_reported'] or 'GPU'}），请选择复用、温和替换或双卡新开",
                        "target_instance": summary,
                        "shortens_lifetime": shortens,
                        "shorten_reasons": shorten_reasons,
                        "allowed_actions": ["reuse", "replace", "new"],
                        "force_confirm": True,
                    }
                if is_busy:
                    return {
                        **base_meta,
                        "decision": "confirm_busy_same_spec",
                        "reason": "同规格创作实例当前正在执行任务，可排队复用或直接新开实例",
                        "target_instance": summary,
                        "shortens_lifetime": shortens,
                        "shorten_reasons": shorten_reasons,
                        "allowed_actions": ["reuse", "new"],
                        "force_confirm": True,
                    }
                if shortens:
                    return {
                        **base_meta,
                        "decision": "confirm_shorten",
                        "reason": "；".join(shorten_reasons),
                        "target_instance": summary,
                        "shortens_lifetime": True,
                        "shorten_reasons": shorten_reasons,
                        "allowed_actions": ["reuse"],
                        "force_confirm": True,
                    }
                return {
                    **base_meta,
                    "decision": "reuse_direct",
                    "reason": None,
                    "target_instance": summary,
                    "shortens_lifetime": False,
                    "shorten_reasons": [],
                    "allowed_actions": ["reuse"],
                    "force_confirm": False,
                }

            # Active creation instances exist, but none meet selected_spec (lower-spec / incompatible)
            active.sort(
                key=lambda inst: (
                    0 if (inst.state in self._IDLE_STATES and self._data(inst).get("current_job_id") is None) else 1,
                    -(
                        (inst.updated_at or inst.created_at).timestamp()
                        if (inst.updated_at or inst.created_at) is not None
                        else 0.0
                    ),
                )
            )
            lower = active[0]
            summary = self._instance_summary(session, lower, probe_job, order_table=order_table)
            return {
                **base_meta,
                "decision": "confirm_lower_spec",
                "reason": f"当前仅有低规格实例（{summary['spec_id'] or summary['gpu_reported'] or 'GPU'}）不满足本次需求，请选择温和停止旧实例后新开，或直接新开双卡",
                "target_instance": summary,
                "shortens_lifetime": False,
                "shorten_reasons": [],
                "allowed_actions": ["replace", "new"],
                "force_confirm": True,
            }

    def _creation_reuse_cost(self, session, job, instance):
        if self._provider(job) != "colab" or not job.get("creation_fingerprint"):
            return job
        from .ccu import instance_live_rate
        from .rates import minute_rate

        data = self._data(instance)
        actual = self._instance_spec(data)
        live_rate = instance_live_rate(session, instance.owner_id, instance.account_id, instance.id)
        rate = minute_rate(actual, live_rate_hourly=live_rate, fallback_rate=self._cfg("unit_rate"))
        before = job.get("unit_rate")
        estimate = job.get("estimated_units")
        if rate is not None and before and estimate is not None:
            estimate = round(estimate * rate / before, 3)
        return {**job, "unit_rate": rate, "estimated_units": estimate}

    def _claim_existing(self, session, instance, job, account, key, stable_id, reason):
        data = self._data(instance)
        if data.get("current_job_id") not in {None, job.get("id")}:
            raise InstanceConflict("实例已被其他任务占用")
        now = self.clock()
        lease_id = data.get("lease_id")
        claimed_deadline = None
        if lease_id:
            lease = self._find(session, "leases", lease_id, instance.owner_id)
            if lease is None or lease.state != "idle":
                raise InstanceConflict("实例租约已不可复用")
            lease_data = self._data(lease)
            if lease_data.get("current_job_id") not in {None, job.get("id")} or lease_data.get("cleanup_claim_token"):
                raise InstanceConflict("实例租约已被占用或正在清理")
            if (
                not lease_data.get("native_ready")
                and lease_data.get("idle_deadline_at") is not None
                and now >= float(lease_data["idle_deadline_at"])
            ):
                raise InstanceConflict("实例租约空闲保留已到期")
            if lease_data.get("deadline_epoch") is not None and now >= float(lease_data["deadline_epoch"]):
                raise InstanceConflict("实例租约已到期")
            handle = lease_data.get("handle")
            lease_data.update(
                state="active",
                owner_job_id=job["id"],
                current_job_id=job["id"],
                idle_deadline_at=None,
                last_active_at=now,
                instance_id=instance.id,
                account_id=self._account_id(account),
            )
            if isinstance(handle, dict):
                lease_data["handle"] = {
                    **handle,
                    "job_id": job["id"],
                    "owner_job_id": job["id"],
                }
            expected_lease_ver = lease.version
            lease_values = {
                "data": copy.deepcopy(lease_data),
                "state": "active",
                "version": expected_lease_ver + 1,
                "updated_at": self._utcnow(),
                "account_id": self._account_id(account),
                "instance_id": instance.id,
            }
            res = session.execute(
                update(type(lease))
                .where(
                    type(lease).id == lease.id,
                    type(lease).version == expected_lease_ver,
                    type(lease).state == "idle",
                )
                .values(**lease_values)
            )
            if res.rowcount != 1:
                raise InstanceConflict("实例租约状态已被其他调度器更新")
            lease.data = copy.deepcopy(lease_data)
            lease.state = "active"
            lease.version = expected_lease_ver + 1
            lease.updated_at = lease_values["updated_at"]
            claimed_deadline = lease_data.get("deadline_epoch")
        elif self._provider(job) == "colab":
            raise InstanceConflict("Colab 实例缺少可复用租约")

        data.update(
            current_job_id=job["id"],
            account_id=self._account_id(account),
            allocation_key=key,
            selection_reason=reason,
            allocation_requested_at=data.get("allocation_requested_at") or now,
            last_seen_at=now,
            manual_stop_requested=False,
            runtime_deleted=False,
            idle_deadline_at=None,
        )
        self._set(instance, data, state="allocating", expected_version=instance.version, session=session)
        job_for_bind = dict(job)
        if lease_id:
            job_for_bind["lease_id"] = lease_id
        if claimed_deadline is not None:
            job_for_bind["deadline_epoch"] = claimed_deadline
        self._bind_job_rows(session, job_for_bind, instance, account)
        self._emit(session, job["owner_id"], instance.project_id, instance.id, job["id"], "allocating", "instance allocated")
        return self._dict(instance)

    def _create_instance(self, session, job, account, key, stable_id, reason):
        model = self._model("instances")
        now = self._utcnow()
        policy = self._job_policy(job)
        pool = job_pool(job)
        retention_mode = "save" if pool == POOL_EPHEMERAL else (
            "retain" if pool == POOL_NATIVE else (job.get("lifecycle") or policy.get("lifecycle") or "save")
        )
        payload = {
            "account_id": self._account_id(account),
            "provider": self._provider(job),
            "instance_pool": pool,
            "current_job_id": job["id"],
            "allocation_key": key,
            "compatibility_signature": job.get("compatibility_signature"),
            "unit_rate": job.get("unit_rate"),
            "runtime_spec_requested": job.get("requested_spec") or policy.get("requested_spec"),
            "retention_mode": retention_mode,
            "selection_reason": reason,
            "allocation_requested_at": self.clock(),
            "generation": 0,
            "manual_stop_requested": False,
            "runtime_deleted": False,
            "last_seen_at": self.clock(),
        }
        if retention_mode in {"smart", "retain"} and policy.get("retain_idle_minutes") is not None:
            payload["retain_idle_minutes"] = float(policy["retain_idle_minutes"])
        if job.get("hard_deadline_at") is not None:
            payload["hard_deadline_at"] = float(job["hard_deadline_at"])
        row = model(id=stable_id, owner_id=job["owner_id"], project_id=job.get("project_id"), state="allocating", data=payload, version=0, created_at=now, updated_at=now)
        session.add(row)
        try:
            session.flush()
        except IntegrityError as exc:
            raise InstanceConflict("实例分配发生竞争，请重试") from exc
        self._bind_job_rows(session, job, row, account)
        self._emit(session, job["owner_id"], row.project_id, row.id, job["id"], "allocating", "instance allocated")
        return self._dict(row)

    def _bind_job_rows(self, session, job, instance, account):
        account_id = self._account_id(account) if account is not None else self._value(self._data(instance), "account_id")
        account_data = self._data(account) if account is not None else {}
        account_alias = self._value(account_data, "alias", "label", default=None) if account is not None else None
        drive_account_id = self._value(account_data, "drive_account_id", default=None) or account_id
        try:
            job_row = self._find(session, "jobs", job["id"], job["owner_id"])
        except ValueError:
            job_row = None
        if job_row is not None:
            data = self._data(job_row)
            existing = data.get("instance_id")
            if existing not in {None, instance.id}:
                raise InstanceConflict("job already belongs to another instance")
            data.update(
                account_id=account_id,
                instance_id=instance.id,
                account_alias_snapshot=account_alias,
                selection_reason=self._data(instance).get("selection_reason"),
            )
            if job.get("creation_fingerprint") and self._provider(job) == "colab":
                data.update(unit_rate=job.get("unit_rate"), estimated_units=job.get("estimated_units"))
            if not data.get("drive_account_id") and drive_account_id:
                data["drive_account_id"] = drive_account_id
            if job.get("lease_id") is not None:
                data["lease_id"] = job["lease_id"]
            if job.get("deadline_epoch") is not None:
                data["deadline_epoch"] = job["deadline_epoch"]
            self._set(job_row, data, state=job_row.state, session=session)
        execution_id = job.get("execution_id")
        if execution_id:
            try:
                execution = self._find(session, "executions", execution_id, job["owner_id"])
            except ValueError:
                execution = None
            if execution is not None:
                data = self._data(execution)
                data.update(instance_id=instance.id, account_id=self._account_id(account))
                self._set(execution, data, state=execution.state, session=session)

    def _validate_new_creation_spec(self, job: Mapping, order_table) -> None:
        if self._provider(job) != "colab":
            return
        from .specs import check_requirements, is_below_workflow_default

        manifest = (job.get("snapshot") or {}).get("manifest") if isinstance(job.get("snapshot"), Mapping) else None
        requested = self._job_requested_spec(job)
        if isinstance(manifest, Mapping) and requested is not None:
            try:
                check_requirements(dict(manifest), dict(requested))
            except ValueError as exc:
                raise APIError("GPU_SPEC_BELOW_WORKFLOW_DEFAULT", str(exc), 422) from exc
            if is_below_workflow_default(manifest, requested, order_table):
                raise APIError(
                    "GPU_SPEC_BELOW_WORKFLOW_DEFAULT",
                    "请求启动的 GPU 规格低于工作流默认最低规格，不支持降配运行",
                    422,
                )

    def _apply_gentle_stop_in_tx(self, session, instance) -> dict:
        if instance.state == "deleted":
            return self._dict(instance)
        if instance.state == "unknown":
            raise InstanceConflict("unknown instance requires reconciliation before stop")
        data = self._data(instance)
        is_busy = data.get("current_job_id") is not None or instance.state in {"allocating", "busy"}
        next_state = instance.state if is_busy else (
            instance.state if instance.state == "cleanup_pending" else "releasing"
        )
        data.update(
            manual_stop_requested=True,
            retention_mode="save",
            idle_deadline_at=None,
            retention_reserved=0.0,
            last_seen_at=self.clock(),
        )
        self._set(instance, data, state=next_state, session=session)
        self._emit(
            session,
            instance.owner_id,
            instance.project_id,
            instance.id,
            data.get("current_job_id"),
            next_state,
            "gentle stop requested",
        )
        return self._dict(instance)

    def request_gentle_stop(self, owner: str, instance_id: str, key: str) -> dict:
        if not isinstance(key, str) or not key:
            raise AllocationError("idempotency key is required")
        body = {"action": "gentle_stop"}
        with self._admission(), self._tx() as session:
            operation, prior, done = self._operation(session, owner, instance_id, "gentle_stop", key, body)
            if done:
                return prior
            instance = self._find(session, "instances", instance_id, owner)
            if instance is None:
                raise KeyError(instance_id)
            result = self._apply_gentle_stop_in_tx(session, instance)
            self._finish_operation(operation, self.public(result), session)
            return result

    def allocate(self, job) -> dict:
        if not isinstance(job, Mapping):
            raise AllocationError("job must be an object")
        owner_id = job.get("owner_id")
        project_id = job.get("project_id")
        if not owner_id:
            raise AllocationError("job owner is required")
        key, stable_id = self._attempt_key(job)
        policy = self._job_policy(job)
        requested_instance = self._requested(job, "instance_id")
        requested_account = self._requested(job, "account_id")
        reuse_action = str(policy.get("reuse_action") or job.get("reuse_action") or "auto")
        wait_delete_id = job.get("wait_for_deleted_instance_id") or policy.get("replace_instance_id")
        provider = self._provider(job)
        pool = job_pool(job)
        is_creation_api = bool((job.get("snapshot") or {}).get("creation_api"))

        with self._admission(), self._tx() as session:
            model = self._model("instances")
            existing = session.scalar(select(model).where(model.owner_id == owner_id, model.id == stable_id))
            if existing is not None and self._data(existing).get("allocation_key") == key:
                return self._dict(existing)
            accounts = self._legal_accounts(session, owner_id, project_id, provider, requested_account)
            if provider == "colab":
                from .cu_guard import CuGuardService

                guard = CuGuardService(self.repo, self.clock)
                keys = [self._account_id(account) for account in accounts]
                if requested_instance and not requested_account:
                    selected = self._find(session, "instances", requested_instance, owner_id)
                    selected_account = selected.account_id if selected else None
                    if selected_account in keys:
                        keys = [selected_account]
                eligible = guard.filter_accounts(guard.state_in(session, owner_id), keys)
                accounts = [account for account in accounts if self._account_id(account) in eligible]
            account_id = self._account_id(accounts[0]) if requested_account else None

            # 1) Ephemeral pool (CUDA smoke, workflow smoke, CPU prep): NEVER reuse, NEVER block others.
            if pool == POOL_EPHEMERAL:
                if requested_instance:
                    raise AllocationError("一次性任务不支持复用实例")
                account = accounts[0] if requested_account else self._choose_account(session, accounts, owner_id, project_id)
                return self._create_instance(
                    session, job, account, key, stable_id, "requested_account" if requested_account else "round_robin"
                )

            # 2) Native pool: NEVER reuse creation/ephemeral instances, NEVER block others.
            if pool == POOL_NATIVE:
                if requested_instance:
                    instance = session.scalar(select(model).where(model.id == requested_instance, model.owner_id == owner_id))
                    if instance is None or self._instance_pool(session, instance) != POOL_NATIVE:
                        raise AllocationError("指定原生实例不存在或不属于原生域")
                    data = self._data(instance)
                    if instance.state in {"unknown", "cleanup_pending", "deleted", "releasing", "quarantined"}:
                        raise AllocationError("指定实例当前不可用")
                    if requested_account and data.get("account_id") != account_id:
                        raise AllocationError("指定实例与账号冲突")
                    account = next((a for a in accounts if self._account_id(a) == data.get("account_id")), None)
                    if account is None or not self._compatible(session, instance, job, self._account_id(account)):
                        raise InstanceConflict("指定原生实例不兼容或已被占用")
                    return self._claim_existing(session, instance, job, account, key, stable_id, "requested_instance")
                account = accounts[0] if requested_account else self._choose_account(session, accounts, owner_id, project_id)
                return self._create_instance(
                    session, job, account, key, stable_id, "requested_account" if requested_account else "round_robin"
                )

            # 3) Creation pool (frontend Create + batches + workbench + Creation API)
            order_table = self._order_table(owner_id)

            # 3a) Gentle replace: wait for old instance to be deleted, then create new instance.
            if reuse_action == "replace" or wait_delete_id:
                if wait_delete_id:
                    old = self._find(session, "instances", wait_delete_id, owner_id)
                    if old is not None and old.state != "deleted":
                        if old.state in {"unknown", "quarantined"}:
                            raise AllocationError("待替换旧实例状态异常，请先在算力页处理")
                        if not self._data(old).get("manual_stop_requested"):
                            self._apply_gentle_stop_in_tx(session, old)
                        raise CreationInstancePending("等待旧实例停止并释放完成后再启动新规格实例")
                self._validate_new_creation_spec(job, order_table)
                account = accounts[0] if requested_account else self._choose_account(session, accounts, owner_id, project_id)
                return self._create_instance(
                    session, job, account, key, stable_id, "requested_account" if requested_account else "replace_new"
                )

            # 3b) Explicit new instance (multi-GPU concurrency allowed on frontend)
            if reuse_action == "new":
                self._validate_new_creation_spec(job, order_table)
                if requested_account:
                    account = accounts[0]
                    reason = "requested_account"
                else:
                    active_accounts = {
                        self._data(r).get("account_id")
                        for r in self._rows(session, "instances", owner_id)
                        if r.state != "deleted" and self._data(r).get("provider") == provider
                    }
                    idle_accounts = [a for a in accounts if self._account_id(a) not in active_accounts]
                    pool_accounts = idle_accounts or accounts
                    account = self._choose_account(session, pool_accounts, owner_id, project_id)
                    reason = "requested_new"
                return self._create_instance(session, job, account, key, stable_id, reason)

            # 3c) Explicit requested instance (`reuse` or legacy `allocation=existing`)
            if requested_instance:
                instance = session.scalar(select(model).where(model.id == requested_instance, model.owner_id == owner_id))
                if instance is None:
                    raise AllocationError("指定实例不存在")
                if self._instance_pool(session, instance) != POOL_CREATION:
                    raise AllocationError("指定实例不属于创作池，不能跨域复用")
                data = self._data(instance)
                if instance.state in {"unknown", "cleanup_pending", "deleted", "releasing", "quarantined"}:
                    raise AllocationError("指定实例当前不可用")
                if requested_account and data.get("account_id") != account_id:
                    raise AllocationError("指定实例与账号冲突")
                account = next((a for a in accounts if self._account_id(a) == data.get("account_id")), None)
                if account is None:
                    raise InstanceConflict("指定实例所属账号当前不可用")
                if reuse_action == "reuse":
                    if not self._compatible(
                        session,
                        instance,
                        job,
                        self._account_id(account),
                        order_table=order_table,
                        target_baseline=self._job_workflow_default_spec(job),
                        allow_busy=True,
                    ):
                        raise InstanceConflict("指定实例不兼容或已被释放")
                    is_busy = instance.state in {"busy", "allocating"} or data.get("current_job_id") not in {None, job.get("id")}
                    if is_busy:
                        raise CreationInstancePending("目标实例正在执行其他任务，排队等待空闲后复用")
                else:
                    if not self._compatible(
                        session,
                        instance,
                        job,
                        self._account_id(account),
                        order_table=order_table,
                        target_baseline=self._job_workflow_default_spec(job),
                        allow_busy=False,
                    ):
                        raise InstanceConflict("指定实例不兼容或已被占用")
                job = self._creation_reuse_cost(session, job, instance)
                return self._claim_existing(session, instance, job, account, key, stable_id, "requested_instance")

            # 3d) Automatic creation pool selection (4-tier priority ranking)
            all_rows = [r for r in self._rows(session, "instances", owner_id) if r.id != stable_id]
            account_ids = {self._account_id(a) for a in accounts}
            workflow_default = self._job_workflow_default_spec(job)
            requested_spec = self._job_requested_spec(job)
            target_baseline = workflow_default if is_creation_api else (requested_spec or workflow_default)

            if is_creation_api:
                active = self._active_creation_instances_in_tx(session, owner_id, provider, account_ids)
                compatible = [
                    r
                    for r in active
                    if r.id != stable_id
                    and self._compatible(
                        session,
                        r,
                        job,
                        self._data(r).get("account_id"),
                        order_table=order_table,
                        target_baseline=workflow_default,
                        allow_busy=True,
                    )
                ]
                if compatible:
                    idle_compatible = [
                        r
                        for r in compatible
                        if r.state in self._IDLE_STATES
                        and self._data(r).get("current_job_id") is None
                        and self._compatible(
                            session,
                            r,
                            job,
                            self._data(r).get("account_id"),
                            order_table=order_table,
                            target_baseline=workflow_default,
                            allow_busy=False,
                        )
                    ]
                    if idle_compatible:
                        idle_compatible.sort(
                            key=lambda row: self._candidate_rank_key(
                                session, row, job, target_baseline=workflow_default, order_table=order_table
                            )
                        )
                        best = idle_compatible[0]
                        best_data = self._data(best)
                        account = next(a for a in accounts if self._account_id(a) == best_data.get("account_id"))
                        job = self._creation_reuse_cost(session, job, best)
                        return self._claim_existing(session, best, job, account, key, stable_id, "reuse_idle")
                    raise CreationInstancePending("等待当前创作实例完成启动或任务")
                # No compatible creation instance: check releasing or lower-spec instances (NEVER auto multi-GPU!)
                releasing = any(
                    r.state == "releasing"
                    and self._instance_pool(session, r) == POOL_CREATION
                    and self._data(r).get("provider", provider) == provider
                    and self._data(r).get("account_id") in account_ids
                    for r in all_rows
                )
                if releasing:
                    raise CreationInstancePending("等待当前创作实例释放完成")
                if active:
                    raise APIError(
                        "CREATION_INSTANCE_SPEC_TOO_LOW",
                        "当前创作实例规格低于该工作流要求，API 不自动多开显卡，请先在算力页停止低规格实例或由前端手动启动高规格实例",
                        409,
                    )
                if job.get("creation_existing_only"):
                    raise AllocationError("没有可复用实例，请先在创作 → API 设置启动工作流")
                self._validate_new_creation_spec(job, order_table)
                account = accounts[0] if requested_account else self._choose_account(session, accounts, owner_id, project_id)
                return self._create_instance(
                    session, job, account, key, stable_id, "requested_account" if requested_account else "round_robin"
                )

            # Frontend creation (`reuse_action == "auto"`): reuse top-ranked idle compatible creation instance, else create new.
            candidates = [
                r
                for r in all_rows
                if r.state in self._IDLE_STATES
                and self._data(r).get("account_id") in account_ids
                and self._compatible(
                    session,
                    r,
                    job,
                    self._data(r).get("account_id"),
                    order_table=order_table,
                    target_baseline=target_baseline,
                    allow_busy=False,
                )
            ]
            if candidates:
                candidates.sort(
                    key=lambda row: self._candidate_rank_key(
                        session, row, job, target_baseline=target_baseline, order_table=order_table
                    )
                )
                instance = candidates[0]
                account = next(a for a in accounts if self._account_id(a) == self._data(instance).get("account_id"))
                job = self._creation_reuse_cost(session, job, instance)
                return self._claim_existing(session, instance, job, account, key, stable_id, "reuse_idle")

            self._validate_new_creation_spec(job, order_table)
            account = accounts[0] if requested_account else self._choose_account(session, accounts, owner_id, project_id)
            return self._create_instance(
                session, job, account, key, stable_id, "requested_account" if requested_account else "round_robin"
            )

    # ---- scheduler lifecycle -------------------------------------------------------
    def _instance_for_job(self, session, job, owner_id=None):
        owner_id = owner_id or job.get("owner_id")
        instance_id = job.get("instance_id")
        if instance_id:
            return self._find(session, "instances", instance_id, owner_id)
        rows = self._rows(session, "instances", owner_id)
        return next((row for row in rows if self._data(row).get("current_job_id") == job.get("id") or self._data(row).get("allocation_key", "").startswith(f"{job.get('id')}:")), None)

    def _transition(self, job, state, message):
        with self._admission(), self._tx() as session:
            instance = self._instance_for_job(session, job)
            if instance is None:
                raise AllocationError("job has no instance")
            if instance.state in {"unknown", "deleted"}:
                raise InstanceConflict("unknown or deleted instance cannot be advanced")
            data = self._data(instance)
            if data.get("current_job_id") not in {None, job.get("id")}:
                raise InstanceConflict("instance is owned by another job")
            data["current_job_id"] = job.get("id")
            data["last_seen_at"] = self.clock()
            self._set(instance, data, state=state, session=session)
            self._emit(session, instance.owner_id, instance.project_id, instance.id, job.get("id"), state, message)
            return self._dict(instance)

    def mark_ready(self, job):
        return self._transition(job, "ready", "instance ready")

    def job_busy(self, job):
        return self._transition(job, "busy", "job started")

    def complete(self, job, success=True):
        from .colab import RUNTIME_MISSING

        with self._admission(), self._tx() as session:
            instance = self._instance_for_job(session, job)
            if instance is None:
                raise AllocationError("job has no instance")
            if instance.state == "deleted":
                return self._dict(instance)
            data = self._data(instance)
            if data.get("current_job_id") not in {None, job.get("id")}:
                raise InstanceConflict("instance is owned by another job")
            pool = self._instance_pool(session, instance)
            if pool == POOL_CREATION and job_pool(job) != POOL_CREATION:
                pool = job_pool(job)
            policy = self._job_policy(job)
            mode = policy.get("lifecycle") or job.get("lifecycle") or data.get("retention_mode") or "save"

            lease_id = data.get("lease_id") or job.get("lease_id")
            lease = self._find(session, "leases", lease_id, instance.owner_id) if lease_id else None
            lease_data = self._data(lease) if lease is not None else {}

            # 1) Highest-priority hard termination conditions (Spec §5.1.1)
            remote_missing = (
                data.get("remote_delete_source") == RUNTIME_MISSING
                or data.get("runtime_deleted") is True
                or lease_data.get("remote_delete_source") == RUNTIME_MISSING
                or job.get("error_code") == RUNTIME_MISSING
            )
            unstarted_vm = False
            if lease is not None:
                unstarted_vm = not lease_data.get("handle") and int(lease_data.get("create_attempts") or 0) == 0
            elif data.get("provider") == "colab" or (not success and instance.state == "allocating"):
                unstarted_vm = True
            force_stopped = bool(data.get("force_stop") or lease_data.get("force_stop"))
            soft_tripped = False
            if data.get("provider") == "colab":
                from .cu_guard import CuGuardService

                soft_tripped = bool(CuGuardService(self.repo, self.clock).state_in(session, instance.owner_id)["soft_tripped"])
            manual_stop = bool(data.get("manual_stop_requested"))
            hard = data.get("hard_deadline_at")
            past_hard = hard is not None and self.clock() >= float(hard)

            hard_terminate = remote_missing or unstarted_vm or force_stopped or soft_tripped or manual_stop or past_hard

            if pool == POOL_EPHEMERAL:
                mode = "save"
            elif pool == POOL_NATIVE or data.get("native_comfy"):
                mode = "save" if (manual_stop or force_stopped or remote_missing or unstarted_vm) else "retain"
            elif hard_terminate:
                mode = "save"

            reuse_batch = (
                not hard_terminate
                and pool == POOL_CREATION
                and success
                and mode == "batch"
                and bool(job.get("creation_fingerprint"))
                and any(
                    row.id != job["id"]
                    and row.state in {"queued", "validating", "waiting_for_capacity"}
                    and not self._data(row).get("cancel_requested")
                    and self._data(row).get("creation_fingerprint") == job["creation_fingerprint"]
                    and self._requested(self._data(row), "account_id") in (None, data.get("account_id"))
                    for row in self._rows(session, "jobs", job["owner_id"])
                )
            )

            if pool == POOL_EPHEMERAL:
                retain = False
            elif pool == POOL_NATIVE or data.get("native_comfy"):
                # Native VM is retained only once booted (success or already marked native_comfy) and no hard stop.
                retain = (
                    (success or bool(data.get("native_comfy")))
                    and not (manual_stop or force_stopped or remote_missing or unstarted_vm or past_hard)
                )
            else:
                # Creation pool: last job's strategy decides; smart/retain survive failure & cancellation!
                retain = (not hard_terminate) and (mode in {"retain", "smart"} or reuse_batch)

            data.update(current_job_id=None, last_seen_at=self.clock(), retention_mode=mode, instance_pool=pool)
            if not (pool == POOL_NATIVE or data.get("native_comfy")):
                data.update(idle_deadline_at=None, retention_reserved=0.0)
            if retain:
                minutes = (
                    policy.get("retain_idle_minutes")
                    or data.get("retain_idle_minutes")
                    or self._cfg("default_idle_minutes", self._cfg("max_idle_minutes", 15))
                )
                minutes = 1.0 if reuse_batch else max(1.0, float(minutes))
                if not (pool == POOL_NATIVE or data.get("native_comfy")) and not job.get("generation_timeouts"):
                    minutes = min(30.0, minutes)
                now = self.clock()
                deadline = min(now + minutes * 60, float(hard)) if hard is not None else now + minutes * 60
                data.update(retain_idle_minutes=minutes, idle_deadline_at=deadline)
            state = "retained_idle" if retain else "releasing"
            self._set(instance, data, state=state, session=session)
            message = "job completed" if success else "job failed"
            self._emit(session, instance.owner_id, instance.project_id, instance.id, job.get("id"), state, message)
            return self._dict(instance)

    def bind_lease(self, job, lease):
        lease_data = lease if isinstance(lease, Mapping) else {}
        lease_id = lease_data.get("id") or lease_data.get("lease_id")
        instance_id = job.get("instance_id") or lease_data.get("instance_id")
        if not lease_id:
            raise AllocationError("lease id is required")
        with self._admission(), self._tx() as session:
            instance = self._find(session, "instances", instance_id, job.get("owner_id")) if instance_id else self._instance_for_job(session, job)
            if instance is None:
                raise AllocationError("job has no instance")
            data = self._data(instance)
            if instance.state in {"deleted", "unknown", "releasing", "cleanup_pending", "quarantined"}:
                raise InstanceConflict("实例正在清理，不能接管租约")
            if data.get("cleanup_claim_token"):
                raise InstanceConflict("实例已被清理流程围栏")
            if data.get("current_job_id") not in {None, job.get("id")}:
                raise InstanceConflict("instance is owned by another job")
            deadline = lease_data.get("hard_deadline_at")
            if deadline is None and not lease_data.get("generation_timeouts"):
                deadline = lease_data.get("deadline_epoch")
            current = data.get("hard_deadline_at")
            if deadline is not None:
                deadline = float(deadline)
                data["hard_deadline_at"] = min(float(current), deadline) if current is not None else deadline
            data.update(lease_id=lease_id, current_job_id=job.get("id"), last_seen_at=self.clock())
            self._set(instance, data, state=instance.state, session=session)
            account = self._account_row(session, data.get("account_id"), instance.owner_id)
            self._bind_job_rows(session, {**dict(job), "lease_id": lease_id}, instance, account)
            try:
                lease_row = self._find(session, "leases", lease_id, instance.owner_id)
            except ValueError:
                lease_row = None
            if lease_row is not None:
                lease_payload = self._data(lease_row)
                lease_payload.update(instance_id=instance.id, account_id=data.get("account_id"))
                self._set(lease_row, lease_payload, state=lease_row.state, session=session)
                if hasattr(lease_row, "instance_id"):
                    lease_row.instance_id = instance.id
                if hasattr(lease_row, "account_id"):
                    lease_row.account_id = data.get("account_id")
            return self._dict(instance)

    def _account_row(self, session, account_id, owner_id):
        rows = self._rows(session, "accounts", owner_id)
        return next((r for r in rows if self._account_id(r) == account_id or r.id == account_id), None)

    # ---- retention, stop and deletion ---------------------------------------------
    def _operation(self, session, owner_id, instance_id, kind, key, body):
        model = self._model("operations")
        op_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"colab-gpu-operation:{owner_id}:{instance_id}:{kind}:{key}"))
        body_hash = digest(body)
        row = session.scalar(select(model).where(model.id == op_id))
        if row is not None:
            data = self._data(row)
            if data.get("body_hash") != body_hash:
                raise InstanceConflict("idempotency key was used with a different body")
            return row, data.get("result"), True
        now = self._utcnow()
        row = model(id=op_id, owner_id=owner_id, project_id=None, state="started", version=0, data={"kind": kind, "idempotency_key": key, "body_hash": body_hash}, created_at=now, updated_at=now)
        session.add(row)
        session.flush()
        return row, None, False

    def _finish_operation(self, row, result, session):
        data = self._data(row)
        data.update(result=copy.deepcopy(result), state="completed")
        self._set(row, data, state="completed", session=session)

    def retention(self, owner, id, mode, idle_minutes, key, expected_version=None):
        if mode not in {"save", "smart", "retain"}:
            raise AllocationError("invalid retention mode")
        if not isinstance(key, str) or not key:
            raise AllocationError("idempotency key is required")
        try:
            minutes = float(idle_minutes)
        except (TypeError, ValueError) as exc:
            raise AllocationError("idle_minutes is invalid") from exc
        if not math.isfinite(minutes) or minutes < 0:
            raise AllocationError("idle_minutes is invalid")
        body = {"mode": mode, "idle_minutes": minutes, "expected_version": expected_version}
        with self._admission(), self._tx() as session:
            operation, prior, done = self._operation(session, owner, id, "retention", key, body)
            if done:
                return prior
            instance = self._find(session, "instances", id, owner)
            if instance is None:
                raise KeyError(id)
            if expected_version is not None and int(instance.version) != int(expected_version):
                raise InstanceConflict("instance version conflict")
            if instance.state in {"releasing", "deleted", "unknown", "cleanup_pending", "quarantined"}:
                raise InstanceConflict("instance can no longer be retained")
            data = self._data(instance)
            lease = self._find(session, "leases", data["lease_id"], owner) if data.get("lease_id") else None
            lease_data = self._data(lease) if lease is not None else {}
            if not lease_data.get("generation_timeouts") and minutes > float(self._cfg("max_idle_minutes", 30)):
                raise AllocationError("idle retention exceeds configured limit")
            hard = data.get("hard_deadline_at")
            if mode in {"retain", "smart"}:
                if data.get("provider") == "colab":
                    from .cu_guard import CuGuardService
                    guard = CuGuardService(self.repo, self.clock)
                    guard.filter_accounts(guard.state_in(session, owner), [data.get("account_id")])
                data["retain_idle_minutes"] = minutes
                deadline = self.clock() + minutes * 60
                data["idle_deadline_at"] = min(deadline, float(hard)) if hard is not None else deadline
                data["retention_mode"] = mode
                state = "retained_idle"
                if not data.get("native_comfy") and not data.get("current_job_id") and lease is not None and lease.state == "idle":
                    if lease_data.get("instance_id") != id or lease_data.get("current_job_id"):
                        raise InstanceConflict("GPU 租约已发生变更")
                    lease_data["idle_deadline_at"] = data["idle_deadline_at"]
                    if lease_data.get("generation_timeouts") and hard is None:
                        lease_data["deadline_epoch"] = data["idle_deadline_at"]
                    self._set(lease, lease_data, session=session)
            else:
                data.update(retention_mode="save", idle_deadline_at=None, retention_reserved=0.0)
                state = "releasing" if data.get("current_job_id") is None else instance.state
            self._set(instance, data, state=state, session=session)
            result = self.public(self._dict(instance))
            self._finish_operation(operation, result, session)
            self._emit(session, owner, instance.project_id, id, data.get("current_job_id"), state, "retention updated")
            return result

    def stop(self, owner, id, key):
        if not isinstance(key, str) or not key:
            raise AllocationError("idempotency key is required")
        body = {"action": "stop"}
        with self._admission(), self._tx() as session:
            operation, prior, done = self._operation(session, owner, id, "stop", key, body)
            if done:
                return prior
            instance = self._find(session, "instances", id, owner)
            if instance is None:
                raise KeyError(id)
            if instance.state == "deleted":
                result = self._dict(instance)
            elif instance.state == "unknown":
                raise InstanceConflict("unknown instance requires reconciliation before stop")
            else:
                data = self._data(instance)
                data.update(
                    manual_stop_requested=True, retention_mode="save", idle_deadline_at=None,
                    retention_reserved=0.0, last_seen_at=self.clock(),
                )
                self._set(instance, data, state="releasing", session=session)
                result = self._dict(instance)
                self._emit(session, owner, instance.project_id, id, data.get("current_job_id"), "releasing", "manual stop requested")
            self._finish_operation(operation, self.public(result), session)
            return result

    def mark_deleted(self, owner, id, expected_version=None, confirmed=False, remote_deleted=False):
        from . import ccu

        with self._admission(), ccu.transaction(self.repo, owner) as session:
            instance = self._find(session, "instances", id, owner)
            if instance is None:
                raise KeyError(id)
            if instance.state == "deleted":
                return self._dict(instance)
            if expected_version is not None and instance.version != expected_version:
                raise InstanceConflict("instance version conflict")
            if confirmed is not True and remote_deleted is not True:
                raise InstanceConflict("remote deletion must be confirmed before marking deleted")
            data = self._data(instance)
            confirmed_at = self.clock()
            data.update(
                current_job_id=None, runtime_deleted=True, remote_deleted=True,
                retention_reserved=0.0, deleted_at=confirmed_at, deletion_confirmed_at=confirmed_at,
                last_seen_at=confirmed_at,
            )
            self._set(instance, data, state="deleted", session=session)
            account = ccu.close_instance_session(self.repo, owner, id, confirmed_at, "confirmed_delete", session=session)
            self._emit(session, owner, instance.project_id, id, None, "deleted", "remote deletion confirmed")
            result = self._dict(instance)
        if account:
            ccu.sample_and_record(self.repo, owner, account, "delete_after")
        return result

    def apply_runtime_inspection(self, lease_snapshot, instance_version, job_snapshot, identity, status, checked_at, claim_token):
        """Commit a Colab observation only while the inspected binding still owns the runtime."""
        from . import ccu
        from .native import _lease_identity

        owner = lease_snapshot["owner_id"]
        lease_id = lease_snapshot["id"]
        job_id = lease_snapshot["owner_job_id"]
        try:
            with self.store.dispatch_lock() as acquired:
                if not acquired:
                    return False
                write = ccu.transaction(self.repo, owner) if status == "missing" else self._tx()
                with self._admission(), write as session:
                    lease = self._find(session, "leases", lease_id, owner)
                    instance = self._find(session, "instances", lease_snapshot["instance_id"], owner)
                    if lease is None or instance is None:
                        return False
                    current = self.repo._row(lease)
                    data = self.repo._row(instance)
                    if (
                        lease.version != lease_snapshot.get("version", 0)
                        or instance.version != instance_version
                        or lease.state == "deleted" or instance.state == "deleted"
                        or digest(_lease_identity(current)) != identity
                        or data.get("lease_id") != lease_id
                        or data.get("provider") != "colab"
                        or data.get("account_id") != current["account_id"]
                        or data.get("current_job_id") not in {None, job_id}
                        or self._data(instance).get("current_job_id") not in {None, job_id}
                        or current.get("current_job_id") not in {None, job_id}
                        or data.get("runtime_id") not in {None, current["runtime_id"]}
                        or data.get("cleanup_claim_token") or current.get("cleanup_claim_token")
                    ):
                        return False
                    if status != "missing":
                        previous = current.get("provider_runtime_check") or {}
                        failures = int(status == "unknown")
                        if failures and previous.get("identity") == identity:
                            failures += int(previous.get("failures") or 0)
                        check = {
                            "identity": identity, "instance_version": instance.version,
                            "status": status, "checked_at": checked_at, "failures": failures,
                            "next_check_at": checked_at + min(
                                self._RUNTIME_CHECK_MAX_BACKOFF, self._RUNTIME_CHECK_INTERVAL * 2 ** min(failures, 3),
                            ),
                        }
                        self._set(lease, {**self._data(lease), "provider_runtime_check": check}, session=session)
                        return True
                    job = self._find(session, "jobs", job_id)
                    if job is not None and job.owner_id != owner:
                        return False
                    job_data = self.repo._row(job) or {}
                    if (
                        job_data.get("lease_id") not in {None, lease_id}
                        or job_data.get("instance_id") not in {None, instance.id}
                        or any(job_data.get(key) != job_snapshot.get(key) for key in (
                            "id", "execution_id", "lease_id", "instance_id", "account_id",
                        ))
                        or (job_data.get("claim_token") not in {None, claim_token}
                            and float(job_data.get("claim_until") or 0) > self.clock())
                    ):
                        return False
                    for other in self._rows(session, "leases", owner):
                        if other.id == lease_id or other.state == "deleted":
                            continue
                        binding = self.repo._row(other)
                        if binding.get("instance_id") == instance.id or (
                            binding.get("account_id") == current["account_id"]
                            and binding.get("runtime_id") == current["runtime_id"]
                        ):
                            return False
                    account = self._close_reclaimed_runtime(session, instance, lease, current, job_data, identity, checked_at)
            if account:
                ccu.sample_and_record(self.repo, owner, account, "delete_after")
            return True
        except InstanceConflict:
            return False

    def _close_reclaimed_runtime(self, session, instance, lease, binding, job, identity, checked_at):
        from . import ccu
        from .colab import RUNTIME_MISSING, RUNTIME_MISSING_MESSAGE

        owner = instance.owner_id
        job_id = binding["owner_job_id"]
        evidence = {
            "schema_version": 1, "owner_id": owner, "account_id": binding["account_id"], "instance_id": instance.id,
            "lease_id": lease.id, "owner_job_id": job_id, "runtime_id": binding["runtime_id"],
            "identity": identity, "checked_at": checked_at, "source": "colab_inspect",
            "lease_version": lease.version, "instance_version": instance.version,
            "instance_generation": self._data(instance).get("generation"),
        }
        remote = {
            "remote_deleted": True, "remote_deleted_at": checked_at, "remote_delete_source": RUNTIME_MISSING,
            "native_ready": False, "native_draining": True, "current_job_id": None, "idle_deadline_at": None,
        }
        self._set(lease, {
            **self._data(lease), **remote, "state": "deleted", "provider_runtime_delete_evidence": evidence,
            "native_remote_deleted_verified_at": checked_at if binding.get("native_comfy") else None,
            "cleanup_next_epoch": 0, "last_seen": checked_at,
        }, state="deleted", session=session)
        data = self._data(instance)
        self._set(instance, {
            **data, **remote, "state": "deleted", "runtime_deleted": True, "budget_reserved": 0.0,
            "retention_reserved": 0.0, "deleted_at": checked_at, "deletion_confirmed_at": checked_at,
            "last_seen_at": checked_at, "retained_until": None,
        }, state="deleted", session=session)
        for related in self._rows(session, "jobs", owner):
            if related.id != job_id and self.repo._row(related).get("instance_id") != instance.id:
                continue
            payload = self._data(related)
            payload.update(instance_state="deleted", runtime_deleted=True, retained_until=None,
                           idle_deadline_at=None, instance_deleted_reason=RUNTIME_MISSING)
            state = related.state
            if related.id == job_id and state not in {"completed", "failed", "cancelled"}:
                outcome = payload.get("cleanup_final_state") if state in {"cleaning_up", "cleanup_pending"} else None
                state = "cleaning_up"
                if outcome not in {"completed", "cancelled"}:
                    payload.update(requested_outcome="failed", cleanup_final_state="failed",
                                   error_code=RUNTIME_MISSING, error_message=RUNTIME_MISSING_MESSAGE)
                payload.update(state=state, **ccu.execution_finished(payload, checked_at))
            self._set(related, payload, state=state, session=session)
        self._emit(session, owner, instance.project_id, instance.id, job_id, "deleted",
                   RUNTIME_MISSING_MESSAGE, reason=RUNTIME_MISSING, source="colab_inspect",
                   checked_at=checked_at, lease_id=lease.id)
        return ccu.close_instance_session(self.repo, owner, instance.id, checked_at, RUNTIME_MISSING, session=session)

    # ---- accounting, events and safe reads ----------------------------------------
    def record_usage(self, owner, payload):
        with self._tx() as session:
            return self._record_usage(session, owner, payload)

    def _record_usage(self, session, owner, payload):
        if not isinstance(payload, Mapping):
            raise ValueError("usage payload must be an object")
        instance_id = payload.get("instance_id")
        if not instance_id:
            raise ValueError("instance_id is required")
        segment_key = payload.get("segment_key") or payload.get("idempotency_key")
        if not segment_key:
            segment_key = digest({k: payload.get(k) for k in ("job_id", "started_at", "ended_at", "kind")})
        segment_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"colab-gpu-usage:{owner}:{instance_id}:{segment_key}"))
        instance = self._find(session, "instances", instance_id, owner)
        if instance is None:
            raise KeyError(instance_id)
        model = self._model("instance_usage_segments")
        existing = session.scalar(select(model).where(model.id == segment_id))
        if existing is not None:
            return self._public_usage(self._dict(existing))
        instance_data = self._data(instance)
        data = {k: copy.deepcopy(v) for k, v in dict(payload).items() if k not in self._SENSITIVE}
        data.update(instance_id=instance_id, account_id=instance_data.get("account_id"), segment_key=str(segment_key))
        requested_at = next(
            (payload.get(name) for name in ("allocation_requested_at", "request_started_at", "requested_at") if payload.get(name) is not None),
            instance_data.get("allocation_requested_at"),
        )
        confirmed_at = next(
            (payload.get(name) for name in ("deletion_confirmed_at", "confirmed_deleted_at", "remote_deleted_at") if payload.get(name) is not None),
            instance_data.get("deletion_confirmed_at") or instance_data.get("deleted_at"),
        )

        def timestamp(value):
            if value is None:
                return None
            if isinstance(value, datetime):
                return value.timestamp()
            try:
                return float(value)
            except (TypeError, ValueError):
                try:
                    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
                except (TypeError, ValueError):
                    return None

        requested_epoch = timestamp(requested_at)
        confirmed_epoch = timestamp(confirmed_at)
        if requested_epoch is not None:
            data["allocation_requested_at"] = requested_at
        if confirmed_epoch is not None:
            data["deletion_confirmed_at"] = confirmed_at
        if requested_epoch is not None and confirmed_epoch is not None:
            total_seconds = max(0.0, confirmed_epoch - requested_epoch)
            data["total_duration_seconds"] = total_seconds
            data["total_duration_minutes"] = total_seconds / 60.0
        # None means not supplied by the platform; it is intentionally not
        # coerced to zero, which would present an actual bill as known.
        data.setdefault("actual_units", None)
        now = self._utcnow()
        row = model(id=segment_id, owner_id=owner, project_id=instance.project_id, state="closed" if data.get("ended_at") is not None else "open", data=data, version=0, created_at=now, updated_at=now)
        session.add(row)
        session.flush()
        return self._public_usage(self._dict(row))

    record_usage_segment = record_usage
    record_usage_once = record_usage

    def _emit(self, session, owner, project_id, instance_id, job_id, state, message, **details):
        try:
            model = self._model("events")
        except ValueError:
            return
        data = {"instance_id": instance_id, "job_id": job_id, "state": state, "message": str(message)[:500], **details}
        now = self._utcnow()
        session.add(model(id=str(uuid.uuid4()), owner_id=owner, project_id=project_id, state=state, version=0, data=data, created_at=now, updated_at=now))

    @classmethod
    def _redact(cls, value):
        if isinstance(value, Mapping):
            return {k: cls._redact(v) for k, v in value.items() if str(k).lower() not in cls._SENSITIVE}
        if isinstance(value, list):
            return [cls._redact(v) for v in value]
        return value

    def _public_usage(self, row):
        allowed = {
            "id", "owner_id", "project_id", "instance_id", "account_id", "job_id", "segment_key",
            "started_at", "ended_at", "allocation_requested_at", "deletion_confirmed_at",
            "total_duration_seconds", "total_duration_minutes", "estimated_units", "actual_units",
            "state", "version", "created_at", "updated_at",
        }
        return {k: self._redact(v) for k, v in row.items() if k in allowed}

    def public(self, instance):
        source = dict(instance) if isinstance(instance, Mapping) else self._dict(instance)
        result = {k: self._redact(copy.deepcopy(source[k])) for k in self._PUBLIC if k in source}
        if "instance_pool" not in result:
            result["instance_pool"] = instance_pool_of(source)
        return result

    def get(self, owner, id):
        with self._tx() as session:
            row = self._find(session, "instances", id, owner)
            return None if row is None else self.public(self._dict(row))

    def list(self, owner, state=None):
        with self._tx() as session:
            rows = self._rows(session, "instances", owner)
            rows.sort(key=lambda row: (row.created_at, row.id), reverse=True)
            return [self.public(self._dict(row)) for row in rows if state is None or row.state == state]

    def clear_history(self, owner):
        with self._admission(), self._tx() as session:
            model = self._model("instances")
            deleted = session.scalars(select(model).where(model.owner_id == owner, model.state == "deleted")).all()
            cleared = 0
            now = self.clock()
            for row in deleted:
                data = self._data(row)
                if data.get("history_cleared_at") is None:
                    self._set(row, {**data, "history_cleared_at": now}, session=session)
                    cleared += 1
            return cleared

    def events(self, owner, id):
        with self._tx() as session:
            rows = self._rows(session, "events", owner)
            return [self._redact(self._dict(row)) for row in rows if self._data(row).get("instance_id") == id]

    def usage(self, owner, id=None):
        with self._tx() as session:
            rows = self._rows(session, "instance_usage_segments", owner)
            return [self._public_usage(self._dict(row)) for row in rows if id is None or self._data(row).get("instance_id") == id]

    # Friendly aliases used by API adapters.
    instance = get
    instances = list
    usage_segments = usage
