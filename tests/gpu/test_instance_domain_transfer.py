"""Comprehensive domain transfer state machine, concurrency, fault & ledger tests (Spec §9.1(2))."""

from __future__ import annotations

import time
import uuid
from typing import Any

import pytest
from app.common import APIError, current_user, envelope
from app.config import settings
from app.db import Store
from app.gpu import ccu, native, recovery, transfer
from app.gpu.compute import GpuConfig
from app.gpu.instances import POOL_CREATION, POOL_EPHEMERAL, POOL_NATIVE, InstanceConflict, InstanceService
from app.gpu.mcp import MCPTransient, _uuid4
from app.gpu.repository import Repository
from app.gpu.scheduler import Scheduler
from app.instance_routes import router as instance_router
from app.native_routes import _load_status_snapshot
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

L4 = {"variant": "VARIANT_GPU", "accelerator": "L4", "shape": "SHAPE_STANDARD"}
CPU_SPEC = {"variant": "VARIANT_CPU", "accelerator": "NONE", "shape": "SHAPE_STANDARD"}
BASELINE = "a" * 64


class FakeClock:
    def __init__(self, start: float | None = None):
        self.now = time.time() if start is None else start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += seconds
        return self.now


class FakeTransferExecutor:
    """Offline controllable executor simulating Colab MCP teardown + poll + kernel restart + delete."""

    def __init__(self):
        self.start_calls: list[tuple[str, dict[str, Any]]] = []
        self.poll_calls: list[str] = []
        self.restart_calls: list[str] = []
        self.delete_calls: list[str] = []
        self.poll_script: list[Any] = []
        self.start_error: Exception | None = None
        self.default_poll: dict[str, Any] = {
            "state": "completed",
            "cleaned": True,
            "teardown_report": {"cleaned": True, "baseline_match": True},
        }

    def start_teardown(self, handle, job, execution_id: str, payload: dict[str, Any]):
        execution_id = _uuid4(execution_id)
        self.start_calls.append((execution_id, dict(payload)))
        if self.start_error is not None:
            err = self.start_error
            self.start_error = None
            raise err
        return {"execution_id": execution_id}

    def poll(self, handle, job):
        exec_id = _uuid4(str(job.get("execution_id") or ""))
        self.poll_calls.append(exec_id)
        if self.poll_script:
            nxt = self.poll_script.pop(0)
            if isinstance(nxt, Exception):
                raise nxt
            return nxt
        return dict(self.default_poll)

    def restart_kernel(self, handle):
        self.restart_calls.append(str(handle.get("runtime_id") or ""))

    def delete(self, handle):
        self.delete_calls.append(str(handle.get("runtime_id") or ""))
        return True

    def inspect(self, handle):
        return {"deleted": False, "state": "RUNNING", "runtime_id": handle.get("runtime_id")}

    def observe(self, handle):
        return "present"


def _make_env(tmp_path, monkeypatch, *, start_time: float | None = None):
    clock = FakeClock(start_time)
    store = Store(f"sqlite:///{tmp_path / 'transfer.sqlite'}")
    repo = Repository(store)
    owner = "owner-tr"
    aid = str(uuid.uuid4())
    repo.create(
        "accounts",
        owner,
        {
            "account_id": aid,
            "alias": "MainColab",
            "provider": "colab",
            "enabled": True,
            "auth_status": "connected",
            "allow_allocation": True,
            "budget": {"daily": 100.0, "monthly": 500.0},
        },
        record_id=aid,
    )
    # Seed valid native config & Drive auth so -> native transfers pass preflight cleanly.
    monkeypatch.setenv("GPU_REAL_ENABLED", "1")
    monkeypatch.setenv("GPU_UNIT_RATE", "0.1")
    monkeypatch.setenv("NATIVE_COMFY_COMMIT", "1" * 40)
    monkeypatch.setattr(settings, "native_comfy_url", "https://native.test")
    monkeypatch.setattr(native, "backend_for", lambda *a, **kw: ("https://native.test", {}))
    monkeypatch.setattr(native, "secret_file", lambda _name: "tunnel-secret")
    monkeypatch.setattr(
        transfer.GoogleService,
        "status",
        lambda self: {"colab": {"connected": True}, "drive": {"connected": True}},
    )
    monkeypatch.setattr(native, "busy_state_for", lambda *a, **kw: ("idle", None))

    cfg = GpuConfig(real_enabled=True, unit_rate=0.1, daily_units=100.0, monthly_units=500.0)
    svc = InstanceService(repo, clock=clock, config=cfg)
    executor = FakeTransferExecutor()
    scheduler = Scheduler(
        store,
        config=cfg,
        provider_factory=lambda _job: executor,
        executor_factory=lambda _job: executor,
        clock=clock,
    )
    return clock, store, repo, owner, aid, svc, executor, scheduler


def _seed_instance(
    repo: Repository,
    owner: str,
    aid: str,
    clock: FakeClock,
    *,
    pool: str,
    state: str = "retained_idle",
    spec: dict[str, str] = L4,
    venv_isolated: bool = True,
    baseline: str | None = BASELINE,
    native_ready: bool | None = None,
    native_bootstrap_failed: bool = False,
    native_bootstrap_error: str | None = None,
    idle_seconds: float = 1800.0,
    with_ledger: bool = True,
):
    now = clock()
    rt_id = f"rt-{uuid.uuid4().hex[:8]}"
    is_native = pool == POOL_NATIVE
    n_ready = (is_native and not native_bootstrap_failed) if native_ready is None else native_ready
    hard_dl = (now + 7200.0) if (is_native and not native_bootstrap_failed) else None
    idle_dl = now + idle_seconds
    lease_data: dict[str, Any] = {
        "state": "idle",
        "provider": "colab",
        "account_id": aid,
        "instance_pool": pool,
        "native_comfy": is_native,
        "native_ready": n_ready,
        "native_bootstrap_failed": native_bootstrap_failed,
        "request_id": f"req-{rt_id}",
        "runtime_id": rt_id,
        "owner_job_id": f"seed-job-{rt_id}",
        "vm_allocated_at": now - 120.0,
        "venv_isolated": venv_isolated,
        "vm_baseline_digest": baseline,
        "hard_deadline_at": hard_dl,
        "idle_deadline_at": idle_dl,
        "deadline_epoch": hard_dl or idle_dl,
        "generation_timeouts": not is_native,
        "handle": {"runtime_id": rt_id, "endpoint": f"ep-{rt_id}", "owner_id": owner},
    }
    lease = repo.create("leases", owner, lease_data)
    inst_data: dict[str, Any] = {
        "account_id": aid,
        "account_alias_snapshot": "MainColab",
        "provider": "colab",
        "instance_pool": pool,
        "native_comfy": is_native,
        "native_ready": n_ready,
        "native_bootstrap_failed": native_bootstrap_failed,
        "native_bootstrap_error": native_bootstrap_error,
        "ephemeral_standby": pool == POOL_EPHEMERAL,
        "state": state,
        "retention_mode": "retain",
        "retain_idle_minutes": 30,
        "lease_id": lease["id"],
        "gpu_reported": "NVIDIA L4" if spec == L4 else "CPU",
        "runtime_spec_requested": {"key": spec},
        "runtime_spec_reported": {"key": spec},
        "vm_allocated_at": now - 120.0,
        "venv_isolated": venv_isolated,
        "vm_baseline_digest": baseline,
        "domain_generation": 0,
        "hard_deadline_at": hard_dl,
        "idle_deadline_at": idle_dl,
    }
    inst = repo.create("instances", owner, inst_data)
    lease = repo.update("leases", lease["id"], {"instance_id": inst["id"]}, owner)
    if with_ledger:
        ccu.open_instance_session(
            repo,
            {"id": lease["owner_job_id"], "owner_id": owner, "instance_id": inst["id"], "native_comfy": is_native},
            lease,
            now - 120.0,
        )
    return repo.get("instances", inst["id"], owner), repo.get("leases", lease["id"], owner), rt_id


def _run_transfer_to_completion(
    scheduler: Scheduler,
    svc: InstanceService,
    repo: Repository,
    owner: str,
    instance_id: str,
    target_pool: str,
    *,
    key: str | None = None,
    is_admin: bool = True,
    options: dict[str, Any] | None = None,
    clock: FakeClock,
):
    body: dict[str, Any] = {"target_pool": target_pool, "confirmed": True}
    if target_pool == POOL_CREATION and options:
        body["creation_options"] = options
    if target_pool == POOL_NATIVE and options:
        body["native_options"] = options
    started = transfer.start_transfer(
        repo,
        svc,
        owner,
        instance_id,
        body,
        key or f"key-{uuid.uuid4().hex[:8]}",
        is_admin=is_admin,
        clock=clock,
    )
    job_id = started["transfer_job_id"]
    # Tick 1: loading -> start_teardown -> running
    j1 = transfer.advance_transfer_job(scheduler, repo.get("jobs", job_id, owner))
    assert j1["state"] in {"running", "completed", "failed", "cleaning_up"}
    if j1["state"] == "running":
        j2 = transfer.advance_transfer_job(scheduler, repo.get("jobs", job_id, owner))
    else:
        j2 = j1
    return started, j2


@pytest.mark.parametrize(
    "from_pool,to_pool",
    [
        (POOL_CREATION, POOL_NATIVE),
        (POOL_CREATION, POOL_EPHEMERAL),
        (POOL_EPHEMERAL, POOL_CREATION),
        (POOL_EPHEMERAL, POOL_NATIVE),
        (POOL_NATIVE, POOL_CREATION),
        (POOL_NATIVE, POOL_EPHEMERAL),
    ],
)
def test_all_six_transfer_paths_preserve_vm_and_update_clocks(tmp_path, monkeypatch, from_pool, to_pool):
    clock, store, repo, owner, aid, svc, executor, scheduler = _make_env(tmp_path, monkeypatch)
    inst, lease, rt_id = _seed_instance(repo, owner, aid, clock, pool=from_pool)
    orig_vm_alloc = inst["vm_allocated_at"]

    started, done_job = _run_transfer_to_completion(
        scheduler, svc, repo, owner, inst["id"], to_pool, clock=clock
    )
    assert done_job["state"] == "completed"
    assert executor.restart_calls == [rt_id]

    updated_inst = repo.get("instances", inst["id"], owner)
    updated_lease = repo.get("leases", lease["id"], owner)
    assert updated_inst["instance_pool"] == to_pool
    assert updated_inst["lease_id"] == lease["id"]
    assert updated_lease["handle"]["runtime_id"] == rt_id
    assert updated_inst["vm_allocated_at"] == orig_vm_alloc
    assert updated_inst["domain_generation"] == 1
    assert len(updated_inst["transfer_history"]) == 1
    assert updated_inst["transfer_history"][0]["from_pool"] == from_pool
    assert updated_inst["transfer_history"][0]["to_pool"] == to_pool

    if to_pool == POOL_CREATION:
        assert updated_inst["state"] == "retained_idle"
        assert updated_inst["hard_deadline_at"] is None
        assert updated_inst["idle_deadline_at"] is not None
        assert updated_lease["generation_timeouts"] is True
        assert updated_lease.get("native_comfy") is not True
    elif to_pool == POOL_EPHEMERAL:
        assert updated_inst["state"] == "retained_idle"
        assert updated_inst["ephemeral_standby"] is True
        assert updated_inst["hard_deadline_at"] is None
        assert pytest.approx(updated_inst["idle_deadline_at"] - clock(), abs=2.0) == 600.0
        assert updated_lease["ephemeral_standby"] is True
        assert updated_lease["generation_timeouts"] is True
    elif to_pool == POOL_NATIVE:
        assert updated_inst["state"] == "allocating"
        assert updated_inst["native_comfy"] is True
        assert updated_inst["hard_deadline_at"] is not None
        boot_job_id = done_job["native_bootstrap_job_id"]
        boot_job = repo.get("jobs", boot_job_id, owner)
        assert boot_job["transferred_native_bootstrap"] is True
        assert boot_job["instance_id"] == inst["id"]
        assert boot_job["lease_id"] == lease["id"]
        assert updated_lease["state"] == "active"
        assert updated_lease["owner_job_id"] == boot_job_id

    if from_pool == POOL_NATIVE:
        snap = _load_status_snapshot(repo, owner)
        # Tuple index 0 is lease, index 10 is transferred_out? Verify directly via native_routes helper.
        assert snap[0] is None  # no active native lease after transferring out


def test_multi_hop_transfer_reuses_single_cu_ledger_segment_as_multi_domain(tmp_path, monkeypatch):
    clock, store, repo, owner, aid, svc, executor, scheduler = _make_env(tmp_path, monkeypatch)
    inst, lease, rt_id = _seed_instance(repo, owner, aid, clock, pool=POOL_CREATION)

    # Hop 1: creation -> ephemeral
    clock.advance(60)
    _, j1 = _run_transfer_to_completion(scheduler, svc, repo, owner, inst["id"], POOL_EPHEMERAL, clock=clock)
    assert j1["state"] == "completed"

    # Hop 2: ephemeral -> creation
    clock.advance(60)
    _, j2 = _run_transfer_to_completion(scheduler, svc, repo, owner, inst["id"], POOL_CREATION, clock=clock)
    assert j2["state"] == "completed"

    from app.gpu.repository import GpuInstanceUsageSegmentRow
    from sqlalchemy import select

    with ccu.transaction(repo, owner) as session:
        rows = list(
            session.scalars(
                select(GpuInstanceUsageSegmentRow).where(
                    GpuInstanceUsageSegmentRow.owner_id == owner,
                    GpuInstanceUsageSegmentRow.instance_id == inst["id"],
                )
            ).all()
        )
        assert len(rows) == 1
        row = rows[0]
        assert row.state == "open"
        seg = dict(row.data or {})
    assert seg["segment_key"] == f"instance-lifetime:{inst['id']}"
    assert seg["session_kind"] == "multi_domain"
    assert len(seg["domain_transfers"]) == 2
    assert seg["domain_transfers"][0]["from_pool"] == POOL_CREATION
    assert seg["domain_transfers"][0]["to_pool"] == POOL_EPHEMERAL
    assert seg["domain_transfers"][1]["to_pool"] == POOL_CREATION

    # Still exact same VM & domain_generation == 2.
    final_inst = repo.get("instances", inst["id"], owner)
    assert final_inst["domain_generation"] == 2
    assert repo.get("leases", lease["id"], owner)["handle"]["runtime_id"] == rt_id


def test_concurrent_double_transfer_to_native_blocks_second_with_409_native_active(tmp_path, monkeypatch):
    clock, store, repo, owner, aid, svc, executor, scheduler = _make_env(tmp_path, monkeypatch)
    inst_a, _, _ = _seed_instance(repo, owner, aid, clock, pool=POOL_CREATION)
    inst_b, _, _ = _seed_instance(repo, owner, aid, clock, pool=POOL_CREATION)

    # First transfer atomically reserves the single native slot.
    res_a = transfer.start_transfer(
        repo, svc, owner, inst_a["id"], {"target_pool": POOL_NATIVE, "confirmed": True}, "k-a", is_admin=True, clock=clock
    )
    assert res_a["instance"]["state"] == "transferring"

    # Preview on inst_b immediately reflects native target blocked.
    preview_b = transfer.preview_transfer(repo, owner, inst_b["id"], is_admin=True, clock=clock)
    assert preview_b["targets"][POOL_NATIVE]["allowed"] is False

    # Second concurrent transfer to native MUST fail with 409 NATIVE_ACTIVE.
    with pytest.raises(APIError) as exc:
        transfer.start_transfer(
            repo, svc, owner, inst_b["id"], {"target_pool": POOL_NATIVE, "confirmed": True}, "k-b", is_admin=True, clock=clock
        )
    assert exc.value.status == 409
    assert exc.value.code == "NATIVE_ACTIVE"

    # Offline workspace management is also blocked while a transfer into native is in flight.
    writable, reason = native.workspace_offline_status(repo, owner)
    assert writable is False and reason


def test_concurrent_retention_and_stop_during_transfer_converge_safely(tmp_path, monkeypatch):
    clock, store, repo, owner, aid, svc, executor, scheduler = _make_env(tmp_path, monkeypatch)
    inst, lease, _ = _seed_instance(repo, owner, aid, clock, pool=POOL_CREATION)

    started = transfer.start_transfer(
        repo, svc, owner, inst["id"], {"target_pool": POOL_EPHEMERAL, "confirmed": True}, "k-stop", is_admin=True, clock=clock
    )
    job_id = started["transfer_job_id"]
    # Move transfer job into running state (remote teardown in progress).
    j_run = transfer.advance_transfer_job(scheduler, repo.get("jobs", job_id, owner))
    assert j_run["state"] == "running"

    # 1. PATCH /retention during transferring MUST fail with InstanceConflict (409 at HTTP layer).
    with pytest.raises(InstanceConflict):
        svc.retention(owner, inst["id"], "retain", 20, "ret-during-tr")

    # 2. POST /stop during transferring records manual_stop_requested without corrupting transferring state.
    stop_res = svc.stop(owner, inst["id"], "stop-during-tr")
    assert stop_res["state"] == "transferring"
    assert stop_res["manual_stop_requested"] is True
    assert repo.get("jobs", job_id, owner)["cancel_requested"] is True

    # 3. Even when remote teardown returns completed/cleaned=True, commit sees manual_stop_requested -> releasing!
    j_after = transfer.advance_transfer_job(scheduler, repo.get("jobs", job_id, owner))
    assert j_after["state"] == "cleaning_up"
    assert j_after["requested_outcome"] == "cancelled"
    final_inst = repo.get("instances", inst["id"], owner)
    assert final_inst["state"] == "releasing"
    assert final_inst["instance_pool"] == POOL_CREATION  # NOT switched to ephemeral, NOT resurrected!


def test_worker_restart_idempotent_poll_and_hard_timeout_fail_closed(tmp_path, monkeypatch):
    clock, store, repo, owner, aid, svc, executor, scheduler = _make_env(tmp_path, monkeypatch)
    inst, lease, _ = _seed_instance(repo, owner, aid, clock, pool=POOL_CREATION)

    started = transfer.start_transfer(
        repo, svc, owner, inst["id"], {"target_pool": POOL_EPHEMERAL, "confirmed": True}, "k-restart", is_admin=True, clock=clock
    )
    job_id = started["transfer_job_id"]
    exec_id = repo.get("jobs", job_id, owner)["execution_id"]

    # First tick starts teardown; second tick encounters MCPTransient (lost ack).
    transfer.advance_transfer_job(scheduler, repo.get("jobs", job_id, owner))
    executor.poll_script = [MCPTransient("connection dropped")]
    j_retry = transfer.advance_transfer_job(scheduler, repo.get("jobs", job_id, owner))
    assert j_retry["state"] == "running"

    # Simulate fresh worker restart with a brand-new Scheduler pointing at same DB.
    clock.advance(5)
    restarted_scheduler = Scheduler(
        store,
        config=GpuConfig(real_enabled=True, unit_rate=0.1),
        provider_factory=lambda _job: executor,
        executor_factory=lambda _job: executor,
        clock=clock,
    )
    j_done = transfer.advance_transfer_job(restarted_scheduler, repo.get("jobs", job_id, owner))
    assert j_done["state"] == "completed"
    # Same execution_id reused across worker restart; teardown never launched twice.
    assert len(executor.start_calls) == 1
    assert executor.poll_calls == [exec_id, exec_id]
    assert repo.get("instances", inst["id"], owner)["instance_pool"] == POOL_EPHEMERAL

    # Timeout case A (non-native): > 3 min -> fail-closed to releasing.
    inst2, _, _ = _seed_instance(repo, owner, aid, clock, pool=POOL_CREATION)
    st2 = transfer.start_transfer(
        repo, svc, owner, inst2["id"], {"target_pool": POOL_EPHEMERAL, "confirmed": True}, "k-tmo1", is_admin=True, clock=clock
    )
    transfer.advance_transfer_job(scheduler, repo.get("jobs", st2["transfer_job_id"], owner))
    clock.advance(185)
    j_tmo1 = transfer.advance_transfer_job(scheduler, repo.get("jobs", st2["transfer_job_id"], owner))
    assert j_tmo1["state"] == "cleaning_up"
    assert j_tmo1["error_code"] == "transfer_timeout"
    assert repo.get("instances", inst2["id"], owner)["state"] == "releasing"

    # Timeout case B (native -> creation): > 3 min -> quarantined (NEVER auto-releasing!).
    inst3, lease3, _ = _seed_instance(repo, owner, aid, clock, pool=POOL_NATIVE)
    st3 = transfer.start_transfer(
        repo, svc, owner, inst3["id"], {"target_pool": POOL_CREATION, "confirmed": True}, "k-tmo2", is_admin=True, clock=clock
    )
    transfer.advance_transfer_job(scheduler, repo.get("jobs", st3["transfer_job_id"], owner))
    clock.advance(185)
    j_tmo2 = transfer.advance_transfer_job(scheduler, repo.get("jobs", st3["transfer_job_id"], owner))
    assert j_tmo2["state"] == "failed"
    assert repo.get("instances", inst3["id"], owner)["state"] == "quarantined"
    assert repo.get("leases", lease3["id"], owner)["state"] == "quarantined"


def test_native_aborted_unsynced_quarantines_and_watchdog_never_auto_deletes(tmp_path, monkeypatch):
    clock, store, repo, owner, aid, svc, executor, scheduler = _make_env(tmp_path, monkeypatch)
    inst, lease, rt_id = _seed_instance(repo, owner, aid, clock, pool=POOL_NATIVE)

    executor.poll_script = [
        {
            "state": "aborted_unsynced",
            "cleaned": False,
            "error_code": "native_unsynced_cache",
            "message": "Drive VFS 仍有未上传缓存",
            "teardown_report": {"aborted_unsynced": True},
        }
    ]
    _, done = _run_transfer_to_completion(scheduler, svc, repo, owner, inst["id"], POOL_CREATION, clock=clock)
    assert done["state"] == "failed"
    assert done["runtime_deleted"] is False

    q_inst = repo.get("instances", inst["id"], owner)
    q_lease = repo.get("leases", lease["id"], owner)
    assert q_inst["state"] == "quarantined"
    assert q_inst["quarantine_reason"] == "aborted_unsynced"
    assert q_lease["state"] == "quarantined"
    assert executor.delete_calls == []

    # Even far into the future, watchdog & claim_orphan_lease MUST NOT delete quarantined VM!
    clock.advance(86400)
    scheduler.watchdog()
    assert repo.claim_orphan_lease(owner, lease["id"], inst["id"], "watchdog-tok", clock()) is None
    assert repo.get("instances", inst["id"], owner)["state"] == "quarantined"
    assert executor.delete_calls == []

    # Offline workspace writes stay blocked while quarantined.
    q_writable, q_reason = native.workspace_offline_status(repo, owner)
    assert q_writable is False and q_reason

    # Manual stop by user explicitly transitions quarantined -> releasing with force_stop=True.
    stopped = svc.stop(owner, inst["id"], "manual-release-quarantined")
    assert stopped["state"] == "releasing"
    assert repo.get("instances", inst["id"], owner)["force_stop"] is True
    assert repo.get("leases", lease["id"], owner)["state"] == "releasing"
    assert repo.get("leases", lease["id"], owner)["force_stop"] is True


def test_native_bootstrap_failure_standby_dual_exits_and_expiry(tmp_path, monkeypatch):
    clock, store, repo, owner, aid, svc, executor, scheduler = _make_env(tmp_path, monkeypatch)

    # Helper to create a transferred_native_bootstrap job whose bootstrap failed and triggers cleanup -> 10m standby.
    def make_failed_bootstrap_standby():
        inst, lease, rt_id = _seed_instance(repo, owner, aid, clock, pool=POOL_CREATION)
        _, tr_done = _run_transfer_to_completion(scheduler, svc, repo, owner, inst["id"], POOL_NATIVE, clock=clock)
        boot_job_id = tr_done["native_bootstrap_job_id"]
        boot_job = repo.get("jobs", boot_job_id, owner)
        # Simulate bootstrap failing -> cleanup runs -> cleaned=True -> instance enters 10m standby.
        status1, boot_job = transfer.maybe_cleanup_failed_native_bootstrap(
            scheduler, boot_job, repo.get("leases", lease["id"], owner), repo.get("instances", inst["id"], owner)
        )
        assert status1 == "pending"
        status2, boot_job = transfer.maybe_cleanup_failed_native_bootstrap(
            scheduler, boot_job, repo.get("leases", lease["id"], owner), repo.get("instances", inst["id"], owner)
        )
        assert status2 == "cleaned"
        assert boot_job["native_failure_cleaned"] is True
        boot_job = repo.update("jobs", boot_job_id, {"error_message": "Tunnel handshake timeout"}, owner)
        completed_inst = svc.complete(boot_job, success=False)
        assert completed_inst["state"] == "retained_idle"
        assert completed_inst["native_bootstrap_failed"] is True
        assert completed_inst["native_ready"] is False
        assert pytest.approx(completed_inst["idle_deadline_at"] - clock(), abs=2.0) == 600.0
        # Sync lease idle state like scheduler._cleanup does.
        repo.update(
            "leases",
            lease["id"],
            {
                "state": "idle",
                "current_job_id": None,
                "native_ready": False,
                "native_bootstrap_failed": True,
                "idle_deadline_at": completed_inst["idle_deadline_at"],
                "deadline_epoch": completed_inst["idle_deadline_at"],
            },
            owner,
        )
        repo.update("jobs", boot_job_id, {"state": "failed", "error_message": "Tunnel handshake timeout"}, owner)
        return inst["id"], lease["id"], rt_id

    # --- Exit 1: Transfer out to creation WITHOUT live tunnel probe! ---
    inst1_id, lease1_id, rt1 = make_failed_bootstrap_standby()
    # Even if live native tunnel probe would explode or report busy, Exit 1 MUST skip it!
    monkeypatch.setattr(native, "busy_state_for", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("must skip tunnel probe")))
    # Native status exposes standby_instance and offline workspace is blocked.
    s_writable, s_reason = native.workspace_offline_status(repo, owner)
    assert s_writable is False and s_reason
    # Recovery must NOT schedule L3 recovery for native_bootstrap_failed, transferring, or ephemeral!
    with repo.store.db_lock(), repo.store.Session() as session:
        for case_kwargs in (
            {"pool": POOL_NATIVE, "native_bootstrap_failed": True},
            {"pool": POOL_CREATION, "was_transferring": True},
            {"pool": POOL_EPHEMERAL},
        ):
            eligible, code, _ = recovery.evaluate_recovery_eligibility_in_tx(
                repo,
                session,
                owner_id=owner,
                account_id=aid,
                retention_mode="retain",
                hard_deadline_at=clock() + 3600,
                orig_idle_deadline_at=clock() + 600,
                was_busy=False,
                rounds_used=0,
                now=clock(),
                **case_kwargs,
            )
            assert eligible is False, f"expected recovery ineligible for {case_kwargs}, got {code}"
    _, exit1_job = _run_transfer_to_completion(scheduler, svc, repo, owner, inst1_id, POOL_CREATION, clock=clock)
    assert exit1_job["state"] == "completed"
    inst1_after = repo.get("instances", inst1_id, owner)
    assert inst1_after["instance_pool"] == POOL_CREATION
    assert inst1_after["state"] == "retained_idle"
    assert inst1_after["native_bootstrap_failed"] is False
    assert repo.get("leases", lease1_id, owner)["handle"]["runtime_id"] == rt1
    # Release inst1 so slot is free for Exit 2.
    repo.update("instances", inst1_id, {"state": "deleted", "runtime_deleted": True}, owner)
    repo.update("leases", lease1_id, {"state": "deleted"}, owner)

    # --- Exit 2: Native page in-place retry with new storage & timeout params ---
    monkeypatch.setattr(native, "busy_state_for", lambda *a, **kw: ("idle", None))
    inst2_id, lease2_id, rt2 = make_failed_bootstrap_standby()
    retry_job = native.admit(
        repo,
        owner,
        {
            "account_id": aid,
            "spec": {"key": L4},
            "model_storage_mode": "drive_mount",
            "cache_gb": 25,
            "cache_max_age_minutes": 90,
            "retain_idle_minutes": 45,
            "max_runtime_minutes": 180,
        },
        "retry-standby-key",
    )
    assert retry_job["native_standby_retry"] is True
    assert retry_job["requested_instance_id"] == inst2_id
    assert retry_job["execution_policy"]["model_storage_mode"] == "drive_mount"
    assert retry_job["execution_policy"]["cache_gb"] == 25
    claimed = svc.allocate(retry_job)
    assert claimed["id"] == inst2_id
    assert claimed["state"] in {"allocating", "busy"}
    assert claimed["native_bootstrap_failed"] is False
    assert repo.get("leases", lease2_id, owner)["handle"]["runtime_id"] == rt2
    repo.update("jobs", retry_job["id"], {"state": "completed"}, owner)
    repo.update("instances", inst2_id, {"state": "deleted", "runtime_deleted": True}, owner)
    repo.update("leases", lease2_id, {"state": "deleted"}, owner)

    # --- Exit 3: 10m expiry releases standby automatically; dirty cleanup releases immediately ---
    inst3_id, lease3_id, _ = make_failed_bootstrap_standby()
    # Keepalive allowed before expiry
    assert native._smart_keepalive_allowed(repo, repo.get("leases", lease3_id, owner), clock()) is True
    clock.advance(605)
    scheduler.watchdog()
    assert repo.get("instances", inst3_id, owner)["state"] == "deleted"


def test_transfer_http_auth_and_pre_upgrade_and_cpu_rejection(tmp_path, monkeypatch):
    clock, store, repo, owner, aid, svc, executor, scheduler = _make_env(tmp_path, monkeypatch)
    inst_ok, _, _ = _seed_instance(repo, owner, aid, clock, pool=POOL_CREATION, venv_isolated=True)
    inst_legacy, _, _ = _seed_instance(repo, owner, aid, clock, pool=POOL_CREATION, venv_isolated=False, baseline=None)
    inst_cpu, _, _ = _seed_instance(repo, owner, aid, clock, pool=POOL_EPHEMERAL, spec=CPU_SPEC, venv_isolated=True)

    user_ctx = {"id": owner, "role": "user"}

    app = FastAPI()

    @app.exception_handler(APIError)
    async def api_error(_request: Request, exc: APIError):
        return JSONResponse(envelope(error={"code": exc.code, "message": exc.message}), status_code=exc.status)

    app.state.instance_service = svc
    app.state.repo = repo
    app.dependency_overrides[current_user] = lambda: user_ctx
    app.include_router(instance_router)

    with TestClient(app) as client:
        # 1. Pre-upgrade legacy VM rejected in preview & POST.
        p_leg = client.get(f"/api/gpu/instances/{inst_legacy['id']}/transfer-preview")
        assert p_leg.status_code == 200
        assert p_leg.json()["data"]["transferable"] is False
        assert p_leg.json()["data"]["venv_isolated"] is False
        r_leg = client.post(
            f"/api/gpu/instances/{inst_legacy['id']}/transfer",
            headers={"Idempotency-Key": "tr-leg"},
            json={"target_pool": "native", "confirmed": True},
        )
        assert r_leg.status_code == 409

        # 2. CPU instance rejected.
        p_cpu = client.get(f"/api/gpu/instances/{inst_cpu['id']}/transfer-preview")
        assert p_cpu.json()["data"]["transferable"] is False

        # 3. Non-admin user CANNOT transfer to ephemeral (403).
        r_forbid = client.post(
            f"/api/gpu/instances/{inst_ok['id']}/transfer",
            headers={"Idempotency-Key": "tr-forbid"},
            json={"target_pool": "ephemeral", "confirmed": True},
        )
        assert r_forbid.status_code == 403

        # 4. Non-admin owner CAN transfer to native (202).
        r_native = client.post(
            f"/api/gpu/instances/{inst_ok['id']}/transfer",
            headers={"Idempotency-Key": "tr-user-native"},
            json={"target_pool": "native", "confirmed": True, "expected_version": inst_ok["version"]},
        )
        assert r_native.status_code == 202
        assert r_native.json()["data"]["instance"]["state"] == "transferring"
        # Same idempotency key returns identical result.
        r_dup = client.post(
            f"/api/gpu/instances/{inst_ok['id']}/transfer",
            headers={"Idempotency-Key": "tr-user-native"},
            json={"target_pool": "native", "confirmed": True, "expected_version": inst_ok["version"]},
        )
        assert r_dup.status_code == 202
        assert r_dup.json()["data"]["transfer_job_id"] == r_native.json()["data"]["transfer_job_id"]

        # 5. Admin CAN transfer another instance to ephemeral (202).
        user_ctx["role"] = "admin"
        inst_admin, _, _ = _seed_instance(repo, owner, aid, clock, pool=POOL_CREATION, venv_isolated=True)
        r_eph = client.post(
            f"/api/gpu/instances/{inst_admin['id']}/transfer",
            headers={"Idempotency-Key": "tr-admin-eph"},
            json={"target_pool": "ephemeral", "confirmed": True},
        )
        assert r_eph.status_code == 202
        assert r_eph.json()["data"]["instance"]["state"] == "transferring"
