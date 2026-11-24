"""Tests for Auto-Queue GPU Allocation Retry across Native ComfyUI, Creation, and API auto-start."""

from __future__ import annotations

import uuid

from app import creation_api, native_api
from app.common import APIError
from app.compute_routes import router as compute_router
from app.config import settings
from app.db import Store
from app.gpu import allocation_retry, native
from app.gpu.colab import TUNNEL_RECONCILE_WINDOW_SECONDS, ColabAllocationError, ColabError
from app.gpu.compute import GpuConfig, MockGPUExecutor
from app.gpu.repository import Repository
from app.gpu.scheduler import Scheduler
from app.gpu.specs import save_gpu_spec_order
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.testclient import TestClient

T4_STD = {"variant": "VARIANT_GPU", "accelerator": "T4", "shape": "SHAPE_STANDARD"}
T4_HM = {"variant": "VARIANT_GPU", "accelerator": "T4", "shape": "SHAPE_HIGHMEM"}
L4_STD = {"variant": "VARIANT_GPU", "accelerator": "L4", "shape": "SHAPE_STANDARD"}
A100_STD = {"variant": "VARIANT_GPU", "accelerator": "A100", "shape": "SHAPE_STANDARD"}
ADMIN = {"Authorization": "Bearer studio-admin"}


class FakeClock:
    def __init__(self, start: float = 1_700_000_000.0):
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += float(seconds)
        return self.now


class ScriptedColabProvider:
    """Deterministic fake Colab provider keyed by account_id."""

    is_mock = False

    def __init__(
        self,
        account_id: str,
        *,
        eligible_by_account: dict[str, list[dict]],
        outcomes: list[dict],
        events: list[tuple[str, str]],
        deleted_handles: list[str],
        clock: FakeClock,
        repo: Repository | None = None,
    ):
        self.account_id = account_id
        self.eligible_by_account = eligible_by_account
        self.outcomes = outcomes
        self.events = events
        self.deleted_handles = deleted_handles
        self.clock = clock
        self.repo = repo
        self._live: dict[str, dict] = {}
        self._reconcile_until: dict[str, float] = {}
        self._late_runtime: set[str] = set()

    def list_specs(self) -> list[dict]:
        specs = self.eligible_by_account.get(self.account_id, [T4_STD, L4_STD])
        return [{"key": dict(s), "eligible": True} for s in specs]

    def create(self, spec: dict, request_id: str, job_id: str, runtime_id: str) -> dict:
        key = spec.get("key", spec)
        acc = key["accelerator"]
        self.events.append((self.account_id, f"{acc}:{key['shape']}"))
        action = self.outcomes.pop(0) if self.outcomes else {"ok": True}
        if action.get("error"):
            exc = action["error"]
            if getattr(exc, "remote_state_unknown", False):
                self._reconcile_until[runtime_id] = self.clock() + TUNNEL_RECONCILE_WINDOW_SECONDS + 0.5
                if action.get("late_runtime"):
                    self._late_runtime.add(runtime_id)
                if self.repo is not None:
                    lease = next((row for row in self.repo.list("leases", "owner") if row.get("runtime_id") == runtime_id), None)
                    if lease:
                        self.repo.update("leases", lease["id"], {"tunnel_nbh": f"nbh-{request_id}"}, "owner")
            raise exc
        handle = {
            "runtime_name": f"runtimes/{runtime_id}",
            "operation_name": f"operations/op-{request_id}",
            "ready": True,
            "runtime_id": runtime_id,
            "request_id": request_id,
            "job_id": job_id,
            "owner_job_id": job_id,
            "owner_id": "owner",
            "spec": dict(key),
        }
        self._live[runtime_id] = handle
        return dict(handle)

    def inspect(self, handle: dict) -> dict:
        rid = handle["runtime_id"]
        if rid in self._reconcile_until:
            if self.clock() < self._reconcile_until[rid]:
                return {"ready": False, "deleted": False, "allocation_unknown": True}
            if rid in self._late_runtime:
                return {"ready": True, "deleted": False, "runtime_name": handle["runtime_name"]}
            return {"ready": False, "deleted": True}
        live = self._live.get(rid)
        if not live:
            return {"ready": False, "deleted": True}
        return {
            "ready": True,
            "deleted": False,
            "runtime_name": live["runtime_name"],
            "runtime_spec": live["spec"],
        }

    def delete(self, handle: dict) -> bool:
        rid = handle["runtime_id"]
        if rid in self._reconcile_until and self.clock() < self._reconcile_until[rid]:
            return False
        self._reconcile_until.pop(rid, None)
        if rid in self._late_runtime:
            self._late_runtime.discard(rid)
            self.deleted_handles.append(rid)
            return True
        if rid in self._live:
            self._live.pop(rid, None)
            self.deleted_handles.append(rid)
        return True


def _make_store():
    store = Store("sqlite:///:memory:")
    repo = Repository(store)
    return store, repo


def _add_account(repo: Repository, account_id: str, alias: str, *, drive_account_id: str | None = None):
    repo.create(
        "accounts",
        "owner",
        {
            "account_id": account_id,
            "alias": alias,
            "provider": "colab",
            "connected": True,
            "enabled": True,
            "allow_allocation": True,
            **({"drive_account_id": drive_account_id} if drive_account_id else {}),
        },
        record_id=account_id,
    )


def _make_native_job(repo: Repository, *, spec: dict, account_id: str | None, max_runtime_minutes: int = 60):
    job, _ = repo.create_job("owner", {"kind": "native-comfy", "spec": {"key": spec}}, f"nat-{uuid.uuid4().hex}")
    return repo.update(
        "jobs",
        job["id"],
        {
            "instance_pool": "native",
            "project_id": "native-comfy-owner",
            "provider": "colab",
            "account_id": account_id,
            "requested_account_id": account_id,
            "auto_select_account": account_id is None,
            "requested_spec": {"key": dict(spec)},
            "initial_requested_spec": {"key": dict(spec)},
            "lifecycle": "retain",
            "execution_policy": {
                "provider": "colab",
                "lifecycle": "retain",
                "account_id": account_id,
                "retain_idle_minutes": 120,
                "max_runtime_minutes": max_runtime_minutes,
                "model_storage_mode": "local",
            },
            "model_id": "native-comfy",
            "workflow_key": "system/native-comfy",
            "workflow_version": 1,
            "max_runtime_minutes": max_runtime_minutes,
            "native_session_deadline_epoch": 1_700_000_000.0 + max_runtime_minutes * 60,
            "snapshot": {
                "native_comfy": True,
                "bound_graph": {},
                "manifest": {
                    "reviewed": True,
                    "state": "published",
                    "comfy_commit": "a" * 40,
                    "compute_spec": dict(spec),
                },
            },
        },
        "owner",
    )


def _make_creation_job(
    repo: Repository,
    *,
    spec: dict,
    workflow_default: dict,
    account_id: str | None,
    batch_id: str | None = None,
    queue_order: int = 0,
    lifecycle: str = "save",
    reuse_action: str = "auto",
    requested_instance_id: str | None = None,
    system_smoke: bool = False,
):
    job, _ = repo.create_job("owner", {"workflow_key": "wf/img", "spec": spec}, f"cr-{uuid.uuid4().hex}")
    return repo.update(
        "jobs",
        job["id"],
        {
            "instance_pool": "ephemeral" if system_smoke else "creation",
            "project_id": "proj-1",
            "provider": "colab",
            "account_id": account_id,
            "requested_account_id": account_id,
            "requested_instance_id": requested_instance_id,
            "auto_select_account": account_id is None,
            "requested_spec": {"key": dict(spec)},
            "initial_requested_spec": {"key": dict(spec)},
            "lifecycle": lifecycle,
            "reuse_action": reuse_action,
            "generation_timeouts": True,
            "creation_fingerprint": None if system_smoke else "fp-wf-img-v1",
            "batch_id": batch_id,
            "queue_order": queue_order,
            "execution_policy": {
                "provider": "colab",
                "lifecycle": lifecycle,
                "account_id": account_id,
                "instance_id": requested_instance_id,
                "reuse_action": reuse_action,
                "compute_spec": dict(spec),
                "retain_idle_minutes": 15,
                "max_runtime_minutes": 30,
            },
            "model_id": "system-cuda-smoke" if system_smoke else "model-1",
            "workflow_key": "system/cuda-smoke" if system_smoke else "wf/img",
            "workflow_version": 1,
            "max_runtime_minutes": 30,
            "snapshot": {
                **({"system_smoke": True} if system_smoke else {}),
                "runtime_compute_spec": dict(spec),
                "bound_graph": {},
                "manifest": {
                    "reviewed": True,
                    "state": "published",
                    "compute_spec": dict(workflow_default),
                },
            },
        },
        "owner",
    )


def _drive_scheduler(
    scheduler: Scheduler,
    repo: Repository,
    job_id: str,
    clock: FakeClock,
    *,
    stop_states: set[str],
    max_ticks: int = 80,
) -> dict:
    for _ in range(max_ticks):
        job = repo.get("jobs", job_id, "owner")
        if job["state"] in stop_states:
            return job
        res = scheduler.tick()
        job = repo.get("jobs", job_id, "owner")
        if job["state"] in stop_states:
            return job
        if not res.get("advanced"):
            # Advance clock past any scheduled retry or cleanup window.
            next_wake = None
            if job["state"] == "waiting_for_capacity" and float(job.get("retry_not_before_epoch") or 0) > clock():
                next_wake = float(job["retry_not_before_epoch"]) + 0.1
            elif job["state"] == "cleanup_pending" and float(job.get("cleanup_next_epoch") or 0) > clock():
                next_wake = float(job["cleanup_next_epoch"]) + 0.1
            if next_wake is not None:
                clock.now = next_wake
            else:
                clock.advance(1.0)
    return repo.get("jobs", job_id, "owner")


def test_settings_api_defaults_and_validation(monkeypatch):
    monkeypatch.setattr(settings, "studio_api_token", "studio-admin")
    store, _ = _make_store()
    app = FastAPI()
    app.state.store = store
    app.include_router(compute_router)

    @app.exception_handler(APIError)
    async def api_error(request, exc):
        return JSONResponse({"error": {"code": exc.code, "message": exc.message}}, status_code=exc.status)

    try:
        with TestClient(app) as client:
            res = client.get("/api/gpu/settings/allocation-retry", headers=ADMIN)
            assert res.status_code == 200
            data = res.json()["data"]
            assert data["enabled"] is False
            assert data["gpu_pool"] == []
            assert data["retry_interval_seconds"] == 45
            assert data["max_retries"] == 20

            bad = client.put(
                "/api/gpu/settings/allocation-retry",
                headers=ADMIN,
                json={"enabled": True, "gpu_pool": [L4_STD], "retry_interval_seconds": 10, "max_retries": 20},
            )
            assert bad.status_code == 422

            bad_max = client.put(
                "/api/gpu/settings/allocation-retry",
                headers=ADMIN,
                json={"enabled": True, "gpu_pool": [L4_STD], "retry_interval_seconds": 30, "max_retries": 99},
            )
            assert bad_max.status_code == 422

            saved = client.put(
                "/api/gpu/settings/allocation-retry",
                headers=ADMIN,
                json={"enabled": True, "gpu_pool": [L4_STD, T4_STD, L4_STD], "retry_interval_seconds": 35, "max_retries": 12},
            )
            assert saved.status_code == 200
            payload = saved.json()["data"]
            assert payload["enabled"] is True
            assert len(payload["gpu_pool"]) == 2
            assert payload["retry_interval_seconds"] == 35
            assert payload["max_retries"] == 12
    finally:
        store.engine.dispose()


def test_candidate_specs_order_and_creation_workflow_floor():
    store, repo = _make_store()
    try:
        # Custom GPU order: T4_STD < L4_STD < T4_HM < A100_STD
        save_gpu_spec_order(store, "owner", [T4_STD, L4_STD, T4_HM, A100_STD])
        cfg = allocation_retry.save_allocation_retry_settings(
            store,
            "owner",
            {"enabled": True, "gpu_pool": [A100_STD, T4_HM, T4_STD, L4_STD], "retry_interval_seconds": 30, "max_retries": 10},
        )
        # Native: selected=T4_HM first, then remaining sorted low->high by GPU order table (T4_STD, L4_STD, A100_STD)
        native_job = _make_native_job(repo, spec=T4_HM, account_id="a1")
        nat_specs = allocation_retry.build_candidate_specs(store, "owner", native_job, cfg)
        assert [f"{s['accelerator']}:{s['shape']}" for s in nat_specs] == [
            "T4:SHAPE_HIGHMEM",
            "T4:SHAPE_STANDARD",
            "L4:SHAPE_STANDARD",
            "A100:SHAPE_STANDARD",
        ]

        # Creation: workflow default = L4_STD, user selected A100_STD -> T4_STD is below workflow default and MUST be filtered out!
        cr_job = _make_creation_job(repo, spec=A100_STD, workflow_default=L4_STD, account_id="a1")
        cr_specs = allocation_retry.build_candidate_specs(store, "owner", cr_job, cfg)
        assert [f"{s['accelerator']}:{s['shape']}" for s in cr_specs] == [
            "A100:SHAPE_STANDARD",
            "L4:SHAPE_STANDARD",
            "T4:SHAPE_HIGHMEM",
        ]

        # Empty gpu_pool: only retries selected spec
        empty_cfg = allocation_retry.save_allocation_retry_settings(
            store, "owner", {"enabled": True, "gpu_pool": [], "retry_interval_seconds": 30, "max_retries": 10}
        )
        only_selected = allocation_retry.build_candidate_specs(store, "owner", native_job, empty_cfg)
        assert only_selected == [T4_HM]
    finally:
        store.engine.dispose()


def test_account_first_then_gpu_rotation_and_deadline_refresh(monkeypatch):
    store, repo = _make_store()
    clock = FakeClock()
    monkeypatch.setattr(allocation_retry.random, "uniform", lambda a, b: 0.0)
    try:
        _add_account(repo, "acc-a", "主号A")
        _add_account(repo, "acc-b", "小号B")
        allocation_retry.save_allocation_retry_settings(
            store,
            "owner",
            {"enabled": True, "gpu_pool": [T4_STD, L4_STD], "retry_interval_seconds": 30, "max_retries": 10},
        )
        events: list[tuple[str, str]] = []
        deleted: list[str] = []
        outcomes = [
            {"error": ColabAllocationError("TUNNEL_DENIED", "NO_RUNTIMES", remote_state_unknown=False)},  # acc-a L4
            {"error": ColabAllocationError("TUNNEL_DENIED", "NO_RUNTIMES", remote_state_unknown=False)},  # acc-b L4
            {"error": ColabAllocationError("TUNNEL_DENIED", "NO_RUNTIMES", remote_state_unknown=False)},  # acc-a T4
            {"ok": True},  # acc-b T4 succeeds!
        ]
        providers: dict[str, ScriptedColabProvider] = {}

        def provider_factory(job):
            aid = job.get("account_id") or "acc-a"
            if aid not in providers:
                providers[aid] = ScriptedColabProvider(
                    aid,
                    eligible_by_account={"acc-a": [T4_STD, L4_STD], "acc-b": [T4_STD, L4_STD]},
                    outcomes=outcomes,
                    events=events,
                    deleted_handles=deleted,
                    clock=clock,
                )
            return providers[aid]

        scheduler = Scheduler(
            repo,
            provider_factory=provider_factory,
            executor_factory=lambda job: MockGPUExecutor(),
            config=GpuConfig(enabled=True, real_enabled=True),
            clock=clock,
        )
        job = _make_native_job(repo, spec=L4_STD, account_id=None, max_runtime_minutes=60)
        initial_deadline = job["native_session_deadline_epoch"]

        result = _drive_scheduler(scheduler, repo, job["id"], clock, stop_states={"loading", "failed"})
        assert result["state"] == "loading"
        # Account-first, then GPU low->high!
        assert events == [
            ("acc-a", "L4:SHAPE_STANDARD"),
            ("acc-b", "L4:SHAPE_STANDARD"),
            ("acc-a", "T4:SHAPE_STANDARD"),
            ("acc-b", "T4:SHAPE_STANDARD"),
        ]
        assert result["account_id"] == "acc-b"
        assert result["drive_account_id"] == "acc-b"  # Not stale acc-a!
        assert result["requested_spec"]["key"] == T4_STD
        assert result["snapshot"]["manifest"]["compute_spec"] == T4_STD
        assert result["allocation_retry"]["phase"] == "succeeded"
        assert result["allocation_retry"]["retries_used"] == 3
        # Queue wait (~90s) must NOT eat the 60-minute session runtime!
        assert result["native_session_deadline_epoch"] > initial_deadline + 80
    finally:
        store.engine.dispose()


def test_rate_limit_429_immediately_aborts_globally():
    store, repo = _make_store()
    clock = FakeClock()
    try:
        _add_account(repo, "acc-a", "主号A")
        _add_account(repo, "acc-b", "小号B")
        allocation_retry.save_allocation_retry_settings(
            store,
            "owner",
            {"enabled": True, "gpu_pool": [T4_STD, L4_STD], "retry_interval_seconds": 30, "max_retries": 10},
        )
        events: list[tuple[str, str]] = []
        outcomes = [
            {
                "error": ColabError(
                    "HTTP 429 rate limited",
                    error_code="colab_rate_limited",
                    category="quota",
                    phase="assign_post",
                    http_status=429,
                    remote_state_unknown=False,
                )
            }
        ]
        provider = ScriptedColabProvider(
            "acc-a",
            eligible_by_account={"acc-a": [L4_STD], "acc-b": [L4_STD]},
            outcomes=outcomes,
            events=events,
            deleted_handles=[],
            clock=clock,
        )
        scheduler = Scheduler(
            repo,
            provider_factory=lambda job: provider,
            executor_factory=lambda job: MockGPUExecutor(),
            config=GpuConfig(enabled=True, real_enabled=True),
            clock=clock,
        )
        job = _make_native_job(repo, spec=L4_STD, account_id=None)
        result = _drive_scheduler(scheduler, repo, job["id"], clock, stop_states={"failed", "loading"})
        assert result["state"] == "failed"
        assert len(events) == 1
        assert result["allocation_retry"]["phase"] == "aborted"
        assert "429" in (result["allocation_retry"].get("stop_reason") or "")
    finally:
        store.engine.dispose()


def test_account_auth_error_excludes_bad_account_and_continues_on_healthy_account():
    store, repo = _make_store()
    clock = FakeClock()
    try:
        _add_account(repo, "acc-a", "坏号A")
        _add_account(repo, "acc-b", "好号B")
        allocation_retry.save_allocation_retry_settings(
            store,
            "owner",
            {"enabled": True, "gpu_pool": [L4_STD], "retry_interval_seconds": 30, "max_retries": 5},
        )
        events: list[tuple[str, str]] = []
        outcomes = [
            {
                "error": ColabError(
                    "HTTP 401 unauthorized",
                    error_code="colab_auth_failed",
                    category="auth",
                    phase="assign_get",
                    http_status=401,
                    remote_state_unknown=False,
                )
            },
            {"ok": True},
        ]
        providers: dict[str, ScriptedColabProvider] = {}

        def provider_factory(job):
            aid = job.get("account_id") or "acc-a"
            if aid not in providers:
                providers[aid] = ScriptedColabProvider(
                    aid,
                    eligible_by_account={"acc-a": [L4_STD], "acc-b": [L4_STD]},
                    outcomes=outcomes,
                    events=events,
                    deleted_handles=[],
                    clock=clock,
                )
            return providers[aid]

        scheduler = Scheduler(
            repo,
            provider_factory=provider_factory,
            executor_factory=lambda job: MockGPUExecutor(),
            config=GpuConfig(enabled=True, real_enabled=True),
            clock=clock,
        )
        job = _make_native_job(repo, spec=L4_STD, account_id=None)
        result = _drive_scheduler(scheduler, repo, job["id"], clock, stop_states={"loading", "failed"})
        assert result["state"] == "loading"
        assert result["account_id"] == "acc-b"
        assert "acc-a" in result["allocation_retry"]["excluded_accounts"]
        assert events == [("acc-a", "L4:SHAPE_STANDARD"), ("acc-b", "L4:SHAPE_STANDARD")]
    finally:
        store.engine.dispose()


def test_unknown_503_waits_60s_reconciliation_and_deletes_late_runtime_before_retry(monkeypatch):
    store, repo = _make_store()
    clock = FakeClock()
    monkeypatch.setattr(allocation_retry.random, "uniform", lambda a, b: 0.0)
    try:
        _add_account(repo, "acc-a", "主号A")
        allocation_retry.save_allocation_retry_settings(
            store,
            "owner",
            {"enabled": True, "gpu_pool": [T4_STD], "retry_interval_seconds": 25, "max_retries": 5},
        )
        events: list[tuple[str, str]] = []
        deleted: list[str] = []
        outcomes = [
            {
                "error": ColabError(
                    "HTTP 503 upstream",
                    error_code="colab_upstream_unavailable",
                    category="upstream",
                    phase="assign_post",
                    http_status=503,
                    remote_state_unknown=True,
                ),
                "late_runtime": True,  # Simulate Colab secretly creating a late VM!
            },
            {"ok": True},
        ]
        provider = ScriptedColabProvider(
            "acc-a",
            eligible_by_account={"acc-a": [T4_STD, L4_STD]},
            outcomes=outcomes,
            events=events,
            deleted_handles=deleted,
            clock=clock,
            repo=repo,
        )
        scheduler = Scheduler(
            repo,
            provider_factory=lambda job: provider,
            executor_factory=lambda job: MockGPUExecutor(),
            config=GpuConfig(enabled=True, real_enabled=True),
            clock=clock,
        )
        job = _make_native_job(repo, spec=L4_STD, account_id="acc-a")
        t0 = clock()
        result = _drive_scheduler(scheduler, repo, job["id"], clock, stop_states={"loading", "failed"})
        assert result["state"] == "loading"
        # Late runtime from attempt 1 MUST have been deleted (never adopted) before attempt 2 started!
        assert len(deleted) == 1
        assert events == [("acc-a", "L4:SHAPE_STANDARD"), ("acc-a", "T4:SHAPE_STANDARD")]
        # At least 60s reconciliation + 25s retry interval must have elapsed!
        assert clock() - t0 >= TUNNEL_RECONCILE_WINDOW_SECONDS + 25.0
    finally:
        store.engine.dispose()


def test_ineligible_spec_on_account_skipped_without_consuming_retry_budget():
    store, repo = _make_store()
    clock = FakeClock()
    try:
        _add_account(repo, "acc-a", "仅T4号")
        _add_account(repo, "acc-b", "有L4号")
        allocation_retry.save_allocation_retry_settings(
            store,
            "owner",
            {"enabled": True, "gpu_pool": [T4_STD, L4_STD], "retry_interval_seconds": 30, "max_retries": 1},
        )
        events: list[tuple[str, str]] = []
        outcomes = [{"ok": True}]
        providers: dict[str, ScriptedColabProvider] = {}

        def provider_factory(job):
            aid = job.get("account_id") or "acc-a"
            if aid not in providers:
                providers[aid] = ScriptedColabProvider(
                    aid,
                    eligible_by_account={"acc-a": [T4_STD], "acc-b": [T4_STD, L4_STD]},
                    outcomes=outcomes,
                    events=events,
                    deleted_handles=[],
                    clock=clock,
                )
            return providers[aid]

        scheduler = Scheduler(
            repo,
            provider_factory=provider_factory,
            executor_factory=lambda job: MockGPUExecutor(),
            config=GpuConfig(enabled=True, real_enabled=True),
            clock=clock,
        )
        job = _make_native_job(repo, spec=L4_STD, account_id=None)
        result = _drive_scheduler(scheduler, repo, job["id"], clock, stop_states={"loading", "failed"})
        assert result["state"] == "loading"
        assert result["account_id"] == "acc-b"
        assert result["requested_spec"]["key"] == L4_STD
        # Skipping acc-a|L4 did not waste retries_used!
        assert result["allocation_retry"]["retries_used"] == 0
    finally:
        store.engine.dispose()


def test_creation_batch_propagates_fallback_spec_and_instant_cancel_works(monkeypatch):
    store, repo = _make_store()
    clock = FakeClock()
    monkeypatch.setattr(allocation_retry.random, "uniform", lambda a, b: 0.0)
    try:
        _add_account(repo, "acc-a", "主号A")
        allocation_retry.save_allocation_retry_settings(
            store,
            "owner",
            {"enabled": True, "gpu_pool": [T4_STD, L4_STD], "retry_interval_seconds": 30, "max_retries": 5},
        )
        events: list[tuple[str, str]] = []
        outcomes = [
            {"error": ColabAllocationError("TUNNEL_DENIED", "NO_RUNTIMES", remote_state_unknown=False)},
            {"ok": True},
        ]
        provider = ScriptedColabProvider(
            "acc-a",
            eligible_by_account={"acc-a": [T4_STD, L4_STD]},
            outcomes=outcomes,
            events=events,
            deleted_handles=[],
            clock=clock,
        )
        scheduler = Scheduler(
            repo,
            provider_factory=lambda job: provider,
            executor_factory=lambda job: MockGPUExecutor(),
            config=GpuConfig(enabled=True, real_enabled=True),
            clock=clock,
        )
        j1 = _make_creation_job(
            repo, spec=L4_STD, workflow_default=T4_STD, account_id="acc-a", batch_id="b1", queue_order=0, lifecycle="batch"
        )
        j2 = _make_creation_job(
            repo, spec=L4_STD, workflow_default=T4_STD, account_id="acc-a", batch_id="b1", queue_order=1, lifecycle="batch"
        )
        res1 = _drive_scheduler(scheduler, repo, j1["id"], clock, stop_states={"loading", "failed"})
        assert res1["state"] == "loading"
        assert res1["requested_spec"]["key"] == T4_STD
        assert res1["snapshot"]["runtime_compute_spec"] == T4_STD
        # Sibling batch job must have stayed parked (zero parallel GPU create) and aligned to T4_STD!
        j2_after = repo.get("jobs", j2["id"], "owner")
        assert j2_after["state"] in {"queued", "validating", "waiting_for_capacity"}
        assert j2_after["requested_spec"]["key"] == T4_STD
        assert j2_after["snapshot"]["runtime_compute_spec"] == T4_STD
        assert len(events) == 2

        # Instant cancel while waiting for next retry!
        outcomes.append({"error": ColabAllocationError("TUNNEL_DENIED", "NO_RUNTIMES", remote_state_unknown=False)})
        j3 = _make_creation_job(repo, spec=L4_STD, workflow_default=T4_STD, account_id="acc-a", reuse_action="new")
        res3 = _drive_scheduler(scheduler, repo, j3["id"], clock, stop_states={"waiting_for_capacity"}, max_ticks=12)
        # Ensure we hit the post-failure waiting_for_capacity (not initial validating -> waiting)
        while res3.get("retry_not_before_epoch") in (None, 0):
            scheduler.tick()
            res3 = repo.get("jobs", j3["id"], "owner")
        cancelled = scheduler.cancel("owner", j3["id"])
        assert cancelled["state"] == "cancelled"
        assert cancelled["allocation_retry"]["phase"] == "cancelled"
    finally:
        store.engine.dispose()


def test_excluded_jobs_never_retry_and_native_auto_account_apis(monkeypatch):
    store, repo = _make_store()
    monkeypatch.delenv("STUDIO_OFFLINE_TEST", raising=False)
    monkeypatch.setenv("GPU_REAL_ENABLED", "1")
    monkeypatch.setenv("NATIVE_COMFY_COMMIT", "a" * 40)
    monkeypatch.setattr(native, "backend", lambda: ("https://internal.example.test", {}))
    monkeypatch.setattr(native, "secret_file", lambda name: "tok")
    monkeypatch.setattr(settings, "native_comfy_url", "https://comfy.example.test")
    monkeypatch.setattr(settings, "native_comfy_backend_url", "https://internal.example.test")
    try:
        _add_account(repo, "acc-a", "主号A")
        _add_account(repo, "acc-b", "小号B")
        allocation_retry.save_allocation_retry_settings(
            store,
            "owner",
            {"enabled": True, "gpu_pool": [T4_STD, L4_STD], "retry_interval_seconds": 30, "max_retries": 5},
        )
        # 1) Smoke job is excluded from allocation retry
        smoke = _make_creation_job(repo, spec=L4_STD, workflow_default=T4_STD, account_id="acc-a", system_smoke=True)
        assert allocation_retry.is_retry_eligible_job(smoke) is False

        # 2) Explicit reuse job is excluded
        reuse_job = _make_creation_job(
            repo, spec=L4_STD, workflow_default=T4_STD, account_id="acc-a", reuse_action="reuse", requested_instance_id="inst-1"
        )
        assert allocation_retry.is_retry_eligible_job(reuse_job) is False

        # 3) Native manual start accepts account_id=None (auto-select)
        admitted = native.admit(
            repo,
            "owner",
            {"account_id": None, "spec": {"key": L4_STD}},
            "idem-native-auto-1",
        )
        assert admitted["auto_select_account"] is True
        assert admitted["account_id"] is None

        # 4) Native API auto_start policy accepts account_id=None and emits auto-select native job
        saved_pol = native_api.save_policy(
            repo,
            "owner",
            {"auto_start": True, "account_id": None, "gpu": L4_STD},
        )
        assert saved_pol["auto_start"] is True
        assert saved_pol["account_id"] is None
        started = native_api.request_start(repo, "owner", None, api_key_label="weak-api")
        assert started["id"] == admitted["id"]  # Reuses active native job cleanly

        # 5) Creation API warm_status exposes allocation_retry
        cr_warm = _make_creation_job(repo, spec=L4_STD, workflow_default=T4_STD, account_id=None)
        cr_warm = repo.update(
            "jobs",
            cr_warm["id"],
            {
                "warmup": True,
                "allocation_retry": {"active": True, "phase": "waiting", "retries_used": 2, "max_retries": 20},
            },
            "owner",
        )
        ws = creation_api.warm_status(repo, cr_warm)
        assert ws["auto_select_account"] is True
        assert ws["allocation_retry"]["retries_used"] == 2
    finally:
        store.engine.dispose()


def test_cancel_during_reconciling_finishes_as_cancelled_without_stale_error(monkeypatch):
    store, repo = _make_store()
    clock = FakeClock()
    monkeypatch.setattr(allocation_retry.random, "uniform", lambda a, b: 0.0)
    try:
        _add_account(repo, "acc-a", "主号A")
        allocation_retry.save_allocation_retry_settings(
            store,
            "owner",
            {"enabled": True, "gpu_pool": [T4_STD], "retry_interval_seconds": 25, "max_retries": 5},
        )
        events: list[tuple[str, str]] = []
        outcomes = [
            {
                "error": ColabError(
                    "HTTP 503 upstream",
                    error_code="colab_upstream_unavailable",
                    category="upstream",
                    phase="assign_post",
                    http_status=503,
                    remote_state_unknown=True,
                ),
            },
        ]
        provider = ScriptedColabProvider(
            "acc-a",
            eligible_by_account={"acc-a": [T4_STD, L4_STD]},
            outcomes=outcomes,
            events=events,
            deleted_handles=[],
            clock=clock,
            repo=repo,
        )
        scheduler = Scheduler(
            repo,
            provider_factory=lambda job: provider,
            executor_factory=lambda job: MockGPUExecutor(),
            config=GpuConfig(enabled=True, real_enabled=True),
            clock=clock,
        )
        job = _make_native_job(repo, spec=L4_STD, account_id="acc-a")
        reconciling = _drive_scheduler(
            scheduler, repo, job["id"], clock, stop_states={"cleanup_pending"}, max_ticks=20
        )
        assert reconciling["state"] == "cleanup_pending"
        assert reconciling["allocation_retry"]["phase"] == "reconciling"
        # User clicks "停止排队" while 503 reconciliation fence is still active!
        cancelled = scheduler.cancel("owner", job["id"])
        assert cancelled["allocation_retry"]["phase"] == "cancelled"
        assert cancelled["cleanup_final_state"] == "cancelled"
        # Advance past 60s reconciliation fence -> must finish as cancelled (not failed) with cleared error.
        final = _drive_scheduler(scheduler, repo, job["id"], clock, stop_states={"cancelled", "failed"})
        assert final["state"] == "cancelled"
        assert final["allocation_retry"]["phase"] == "cancelled"
        assert final["error_code"] is None
        assert final["error_message"] is None
        assert len(events) == 1
    finally:
        store.engine.dispose()
