"""Regression tests for strict FIFO queue order, transactional instance gating, prep wake-up, and admission alias."""

from __future__ import annotations

import uuid

import pytest
from app.admission import admit_job
from app.db import Store
from app.gpu.compute import GpuConfig
from app.gpu.instances import POOL_CREATION, CreationInstancePending, InstanceService
from app.gpu.registry import Registry
from app.gpu.repository import Repository
from app.gpu.scheduler import Scheduler

T4 = {"variant": "VARIANT_GPU", "accelerator": "T4", "shape": "SHAPE_STANDARD"}
L4 = {"variant": "VARIANT_GPU", "accelerator": "L4", "shape": "SHAPE_STANDARD"}
A100 = {"variant": "VARIANT_GPU", "accelerator": "A100", "shape": "SHAPE_STANDARD"}


def _seed_env(tmp_path, *, wf_spec=L4, allowed_projects=None):
    store = Store(f"sqlite:///{tmp_path / 'fifo.sqlite'}")
    repo = Repository(store)
    owner = "owner"
    proj_a = store.create("projects", owner, {"title": "Project A"})
    proj_b = store.create("projects", owner, {"title": "Project B"})
    aid = str(uuid.uuid4())
    acct_data = {
        "account_id": aid,
        "alias": "账号2",
        "provider": "colab",
        "enabled": True,
        "auth_status": "connected",
        "allow_allocation": True,
        "budget": {"daily": 100.0, "monthly": 500.0},
    }
    if allowed_projects is not None:
        acct_data["allowed_projects"] = list(allowed_projects)
    repo.create("accounts", owner, acct_data, record_id=aid)

    repo.create("models", owner, {"model_id": "demo-wf", "enabled": True})
    manifest, graph, info = Registry._template()
    manifest["compute_spec"] = wf_spec
    wf = repo.create(
        "workflows",
        owner,
        {
            "workflow_key": "demo-wf",
            "workflow_version": 1,
            "model_id": "demo-wf",
            "state": "published",
            "review_approved": True,
            "smoke_test_kind": "real",
            "active": True,
            "sha256": "wf-sha-fifo",
            "manifest": manifest,
            "api_graph": graph,
            "object_info": info,
        },
    )
    return store, repo, owner, proj_a, proj_b, aid, wf


def _add_instance(
    repo,
    owner,
    aid,
    *,
    spec=L4,
    state="retained_idle",
    current_job_id=None,
    provider="colab",
    retention_mode="smart",
):
    lease_state = "idle" if state == "retained_idle" and not current_job_id else "active"
    lease = repo.create(
        "leases",
        owner,
        {
            "state": lease_state,
            "provider": provider,
            "account_id": aid,
            "instance_pool": POOL_CREATION,
            "owner_job_id": current_job_id or "prev-job",
            "current_job_id": current_job_id,
            "deadline_epoch": 99999.0,
            "idle_deadline_at": 50000.0 if lease_state == "idle" else None,
            "allocation_confirmed": True,
            "handle": {
                "runtime_name": f"runtimes/rt-{uuid.uuid4().hex[:8]}",
                "runtime_id": f"rt-{uuid.uuid4().hex[:8]}",
                "owner_id": owner,
                "job_id": current_job_id or "prev-job",
                "owner_job_id": current_job_id or "prev-job",
            },
        },
    )
    inst = repo.create(
        "instances",
        owner,
        {
            "instance_pool": POOL_CREATION,
            "provider": provider,
            "account_id": aid,
            "account_alias_snapshot": "账号2",
            "state": state,
            "retention_mode": retention_mode,
            "retain_idle_minutes": 15,
            "lease_id": lease["id"],
            "current_job_id": current_job_id,
            "runtime_spec_requested": {"key": spec},
            "runtime_spec_reported": {"key": spec},
            "gpu_reported": spec["accelerator"],
            "idle_deadline_at": 50000.0 if state == "retained_idle" else None,
            "hard_deadline_at": 99999.0,
        },
    )
    repo.update("leases", lease["id"], {"instance_id": inst["id"]}, owner)
    return inst, lease


def _create_reuse_job(
    repo,
    owner,
    project_id,
    aid,
    inst_id,
    *,
    job_id: str,
    created_epoch: float,
    state: str = "queued",
    spec=L4,
    reuse_action: str = "reuse",
    creation_api: bool = False,
    awaiting_prep: bool = False,
    prep_job_id: str | None = None,
    cancel_requested: bool = False,
    batch_id: str | None = None,
    queue_order: int = 0,
):
    policy = {
        "provider": "colab",
        "account_id": aid,
        "lifecycle": "smart",
        "retain_idle_minutes": 15,
        "reuse_action": reuse_action,
    }
    if inst_id is not None and reuse_action == "reuse":
        policy["instance_id"] = inst_id
    snapshot = {"manifest": {"compute_spec": spec}}
    if creation_api:
        snapshot["creation_api"] = True
    payload = {
        "state": state,
        "project_id": project_id,
        "provider": "colab",
        "instance_pool": POOL_CREATION,
        "account_id": aid,
        "requested_account_id": aid,
        "requested_instance_id": inst_id if reuse_action == "reuse" else None,
        "reuse_action": reuse_action,
        "requested_spec": {"key": spec},
        "snapshot": snapshot,
        "execution_policy": policy,
        "created_epoch": created_epoch,
        "max_runtime_minutes": 30,
        "awaiting_prep": awaiting_prep,
        "cancel_requested": cancel_requested,
        "queue_order": queue_order,
    }
    if prep_job_id is not None:
        payload["prep_job_id"] = prep_job_id
    if batch_id is not None:
        payload["batch_id"] = batch_id
    return repo.create("jobs", owner, payload, project_id=project_id, record_id=job_id)


def test_scheduler_strict_fifo_for_explicit_reuse_queue_across_bursts(tmp_path):
    """Reproduces the 7-job screenshot pattern: J1 (21:04), J2..J4 (21:07), J5..J7 (21:08) must run in strict FIFO."""
    store, repo, owner, proj_a, _, aid, _ = _seed_env(tmp_path)
    inst, lease = _add_instance(repo, owner, aid, spec=L4, state="busy", current_job_id="J0")
    repo.create("executions", owner, {"state": "running", "job_id": "J0", "lease_id": lease["id"]}, record_id="exec-0")
    repo.create(
        "jobs",
        owner,
        {
            "state": "running",
            "project_id": proj_a["id"],
            "provider": "colab",
            "instance_pool": POOL_CREATION,
            "account_id": aid,
            "instance_id": inst["id"],
            "lease_id": lease["id"],
            "execution_id": "exec-0",
            "created_epoch": 100.0,
            "requested_spec": {"key": L4},
            "snapshot": {"manifest": {"compute_spec": L4}},
            "execution_policy": {"provider": "colab", "account_id": aid, "lifecycle": "smart", "retain_idle_minutes": 15},
        },
        project_id=proj_a["id"],
        record_id="J0",
    )

    started_order: list[str] = []
    polls: dict[str, int] = {}

    class FakeProvider:
        def list_specs(self):
            return [{"key": L4, "eligible": True}]

        def inspect(self, _handle):
            return {"ready": True, "deleted": False, "runtime_spec": {"key": L4}}

    class FakeExecutor:
        def start(self, _handle, job, eid):
            started_order.append(job["id"])
            return {"execution_id": eid}

        def poll(self, _handle, job):
            jid = job["id"]
            polls[jid] = polls.get(jid, 0) + 1
            limit = 5 if jid == "J0" else 2
            if polls[jid] >= limit:
                return {
                    "state": "completed",
                    "execution_id": job.get("execution_id"),
                    "outputs": [{"filename": "out.png", "content": b"img", "mime": "image/png"}],
                }
            return {"state": "running", "execution_id": job.get("execution_id")}

    cfg = GpuConfig(enabled=True, real_enabled=True, unit_rate=0.1)
    svc = InstanceService(repo, clock=lambda: 1000.0, config=cfg)
    sched = Scheduler(
        repo,
        provider_factory=lambda _: FakeProvider(),
        executor_factory=lambda _: FakeExecutor(),
        instance_service=svc,
        clock=lambda: 1000.0,
        config=cfg,
    )
    sched._upload = lambda job: sched._set_state(job, "cleaning_up", "uploaded", cleanup_final_state="completed")
    sched._await_prep = lambda _job: False

    sched.tick()  # J0 poll 1
    _create_reuse_job(repo, owner, proj_a["id"], aid, inst["id"], job_id="J1", created_epoch=200.0)
    for _ in range(5):
        sched.tick()
    for idx, jid in enumerate(("J2", "J3", "J4"), start=1):
        _create_reuse_job(repo, owner, proj_a["id"], aid, inst["id"], job_id=jid, created_epoch=300.0 + idx)
    for _ in range(15):
        sched.tick()
    for idx, jid in enumerate(("J5", "J6", "J7"), start=1):
        _create_reuse_job(repo, owner, proj_a["id"], aid, inst["id"], job_id=jid, created_epoch=400.0 + idx)

    for _ in range(220):
        res = sched.tick()
        if not res.get("job_id"):
            break

    assert started_order == ["J1", "J2", "J3", "J4", "J5", "J6", "J7"]


def test_late_job_submitted_during_uploading_cannot_cut_ahead_of_earlier_waiter(tmp_path):
    """A fresh job arriving while J0 is uploading/cleaning_up must never steal the freed instance from J1."""
    store, repo, owner, proj_a, _, aid, _ = _seed_env(tmp_path)
    inst, lease = _add_instance(repo, owner, aid, spec=L4, state="busy", current_job_id="J0")
    repo.create("executions", owner, {"state": "running", "job_id": "J0", "lease_id": lease["id"]}, record_id="exec-0")
    repo.create(
        "jobs",
        owner,
        {
            "state": "running",
            "project_id": proj_a["id"],
            "provider": "colab",
            "instance_pool": POOL_CREATION,
            "account_id": aid,
            "instance_id": inst["id"],
            "lease_id": lease["id"],
            "execution_id": "exec-0",
            "created_epoch": 100.0,
            "requested_spec": {"key": L4},
            "snapshot": {"manifest": {"compute_spec": L4}},
            "execution_policy": {"provider": "colab", "account_id": aid, "lifecycle": "smart", "retain_idle_minutes": 15},
        },
        project_id=proj_a["id"],
        record_id="J0",
    )

    started_order: list[str] = []
    finish_j0 = [False]

    class FakeProvider:
        def list_specs(self):
            return [{"key": L4, "eligible": True}]

        def inspect(self, _handle):
            return {"ready": True, "deleted": False, "runtime_spec": {"key": L4}}

    class FakeExecutor:
        def start(self, _handle, job, eid):
            started_order.append(job["id"])
            return {"execution_id": eid}

        def poll(self, _handle, job):
            if job["id"] == "J0" and finish_j0[0]:
                return {
                    "state": "completed",
                    "execution_id": job.get("execution_id"),
                    "outputs": [{"filename": "out.png", "content": b"img", "mime": "image/png"}],
                }
            return {"state": "running", "execution_id": job.get("execution_id")}

    cfg = GpuConfig(enabled=True, real_enabled=True, unit_rate=0.1)
    svc = InstanceService(repo, clock=lambda: 1000.0, config=cfg)
    sched = Scheduler(
        repo,
        provider_factory=lambda _: FakeProvider(),
        executor_factory=lambda _: FakeExecutor(),
        instance_service=svc,
        clock=lambda: 1000.0,
        config=cfg,
    )
    sched._upload = lambda job: sched._set_state(job, "cleaning_up", "uploaded", cleanup_final_state="completed")
    sched._await_prep = lambda _job: False

    _create_reuse_job(repo, owner, proj_a["id"], aid, inst["id"], job_id="J1_early", created_epoch=200.0)
    for _ in range(5):
        sched.tick()
    assert repo.get("jobs", "J1_early", owner)["state"] == "waiting_for_capacity"

    finish_j0[0] = True
    # Advance J0 to uploading, then submit J2_late right before cleanup completes.
    for _ in range(4):
        sched.tick()
        if repo.get("jobs", "J0", owner)["state"] == "uploading":
            break
    assert repo.get("jobs", "J0", owner)["state"] == "uploading"

    _create_reuse_job(repo, owner, proj_a["id"], aid, inst["id"], job_id="J2_late", created_epoch=300.0)
    for _ in range(25):
        sched.tick()
        if started_order:
            break

    assert started_order == ["J1_early"]


def test_transactional_fifo_gate_across_reuse_creation_api_and_frontend_auto(tmp_path):
    """Even if a later job reaches InstanceService.allocate() first (restart/external free), DB gate blocks it."""
    store, repo, owner, proj_a, _, aid, _ = _seed_env(tmp_path)
    inst, _ = _add_instance(repo, owner, aid, spec=L4, state="retained_idle")
    cfg = GpuConfig(enabled=True, real_enabled=True, unit_rate=0.1)
    svc = InstanceService(repo, clock=lambda: 1000.0, config=cfg)

    j_early = _create_reuse_job(
        repo, owner, proj_a["id"], aid, inst["id"],
        job_id="J_early", created_epoch=200.0, state="waiting_for_capacity",
    )
    j_late_reuse = _create_reuse_job(
        repo, owner, proj_a["id"], aid, inst["id"],
        job_id="J_late_reuse", created_epoch=300.0, state="waiting_for_capacity", reuse_action="reuse",
    )
    j_late_api = _create_reuse_job(
        repo, owner, proj_a["id"], aid, None,
        job_id="J_late_api", created_epoch=310.0, state="waiting_for_capacity", reuse_action="auto", creation_api=True,
    )
    j_late_auto = _create_reuse_job(
        repo, owner, proj_a["id"], aid, None,
        job_id="J_late_auto", created_epoch=320.0, state="waiting_for_capacity", reuse_action="auto",
    )

    for late_job in (j_late_reuse, j_late_api, j_late_auto):
        with pytest.raises(CreationInstancePending) as exc_info:
            svc.allocate(late_job)
        assert exc_info.value.fifo_blocked is True
        assert exc_info.value.blocker_job_id == "J_early"

    # Never silently opened a second VM for frontend auto!
    assert len(repo.list("instances", owner)) == 1

    # Earlier waiter succeeds and claims the instance.
    claimed = svc.allocate(j_early)
    assert claimed["id"] == inst["id"]


def test_external_release_wakes_earlier_waiter_when_later_job_hits_fifo_gate(tmp_path):
    """If an instance is freed externally while J1 is stalled behind J2 in rr, J2 yields and wakes J1 immediately."""
    store, repo, owner, proj_a, _, aid, _ = _seed_env(tmp_path)
    inst, _ = _add_instance(repo, owner, aid, spec=L4, state="retained_idle")
    cfg = GpuConfig(enabled=True, real_enabled=True, unit_rate=0.1)
    svc = InstanceService(repo, clock=lambda: 1000.0, config=cfg)
    sched = Scheduler(repo, instance_service=svc, clock=lambda: 1000.0, config=cfg)
    sched._await_prep = lambda _job: False

    _create_reuse_job(
        repo, owner, proj_a["id"], aid, inst["id"],
        job_id="J1_early", created_epoch=200.0, state="waiting_for_capacity",
    )
    _create_reuse_job(
        repo, owner, proj_a["id"], aid, inst["id"],
        job_id="J2_late", created_epoch=300.0, state="waiting_for_capacity",
    )

    # Artificially put J1_early stalled with a larger seq than J2_late (simulating ring inversion after external release).
    rr = sched._rr_state()
    rr["seq"] = 20
    rr["stalled"]["J1_early"] = (15, 0, 0)
    rr["stalled"]["J2_late"] = (5, 0, 0)

    t1 = sched.tick()
    assert t1["job_id"] == "J2_late"
    assert t1["advanced"] is False
    assert repo.get("jobs", "J2_late", owner)["state"] == "waiting_for_capacity"

    t2 = sched.tick()
    assert t2["job_id"] == "J1_early"
    assert t2["state"] == "allocating"
    assert repo.get("jobs", "J1_early", owner)["instance_id"] == inst["id"]


def test_fifo_gate_does_not_falsely_block_on_ineligible_or_independent_jobs(tmp_path):
    """Ineligible project, incompatible spec, queued/unvalidated, active awaiting_prep, cancelled, or different instance must not block."""
    store, repo, owner, proj_a, proj_b, aid, _ = _seed_env(tmp_path, wf_spec=T4)
    repo.update("accounts", aid, {"allowed_projects": [proj_b["id"]]}, owner)
    inst_t4, _ = _add_instance(repo, owner, aid, spec=T4, state="retained_idle")
    inst_other, _ = _add_instance(repo, owner, aid, spec=L4, state="busy", current_job_id="other-running")
    cfg = GpuConfig(enabled=True, real_enabled=True, unit_rate=0.1)
    svc = InstanceService(repo, clock=lambda: 1000.0, config=cfg)

    # Active CPU prep job
    repo.create("jobs", owner, {"state": "running", "prep": True, "provider": "colab"}, record_id="prep-active")

    # 1) Earlier job in proj_a has NO permission for account (allowed_projects=[proj_b])
    _create_reuse_job(
        repo, owner, proj_a["id"], aid, inst_t4["id"],
        job_id="J_forbidden_project", created_epoch=100.0, state="waiting_for_capacity", spec=T4,
    )
    # 2) Earlier job needs A100 (incompatible with idle T4)
    _create_reuse_job(
        repo, owner, proj_b["id"], aid, inst_t4["id"],
        job_id="J_needs_a100", created_epoch=110.0, state="waiting_for_capacity", spec=A100,
    )
    # 3) Earlier job is still queued (not yet validated/waiting_for_capacity)
    _create_reuse_job(
        repo, owner, proj_b["id"], aid, inst_t4["id"],
        job_id="J_still_queued", created_epoch=120.0, state="queued", spec=T4,
    )
    # 4) Earlier job is waiting on active CPU prep
    _create_reuse_job(
        repo, owner, proj_b["id"], aid, inst_t4["id"],
        job_id="J_awaiting_prep", created_epoch=130.0, state="waiting_for_capacity", spec=T4,
        awaiting_prep=True, prep_job_id="prep-active",
    )
    # 5) Earlier job is cancelled
    _create_reuse_job(
        repo, owner, proj_b["id"], aid, inst_t4["id"],
        job_id="J_cancelled", created_epoch=140.0, state="waiting_for_capacity", spec=T4,
        cancel_requested=True,
    )
    # 6) Earlier job targets a DIFFERENT instance
    _create_reuse_job(
        repo, owner, proj_b["id"], aid, inst_other["id"],
        job_id="J_other_inst", created_epoch=150.0, state="waiting_for_capacity", spec=T4,
    )
    # 7) Earlier job explicitly requests a NEW GPU
    _create_reuse_job(
        repo, owner, proj_b["id"], aid, None,
        job_id="J_new_gpu", created_epoch=160.0, state="waiting_for_capacity", spec=T4,
        reuse_action="new",
    )

    # Later eligible job in proj_b MUST claim inst_t4 without being blocked!
    j_ready = _create_reuse_job(
        repo, owner, proj_b["id"], aid, inst_t4["id"],
        job_id="J_ready", created_epoch=200.0, state="waiting_for_capacity", spec=T4,
    )
    claimed = svc.allocate(j_ready)
    assert claimed["id"] == inst_t4["id"]


def test_awaiting_prep_clears_immediately_after_prep_finishes_even_when_gpu_is_busy(tmp_path):
    """Once prep finishes, J1 leaves Group 3, clears awaiting_prep=False even while GPU is busy, and beats later J2."""
    store, repo, owner, proj_a, _, aid, _ = _seed_env(tmp_path)
    inst, _ = _add_instance(repo, owner, aid, spec=L4, state="busy", current_job_id="J0")
    repo.create("jobs", owner, {"state": "completed", "prep": True, "provider": "colab"}, record_id="prep-done")
    repo.create(
        "jobs",
        owner,
        {"state": "running", "provider": "colab", "instance_id": inst["id"], "created_epoch": 50.0},
        record_id="J0",
    )

    _create_reuse_job(
        repo, owner, proj_a["id"], aid, inst["id"],
        job_id="J1_after_prep", created_epoch=100.0, state="waiting_for_capacity",
        awaiting_prep=True, prep_job_id="prep-done",
    )
    _create_reuse_job(
        repo, owner, proj_a["id"], aid, inst["id"],
        job_id="J2_later", created_epoch=200.0, state="waiting_for_capacity",
    )

    cfg = GpuConfig(enabled=True, real_enabled=True, unit_rate=0.1)
    svc = InstanceService(repo, clock=lambda: 1000.0, config=cfg)
    sched = Scheduler(repo, instance_service=svc, clock=lambda: 1000.0, config=cfg)
    sched._await_prep = lambda _job: False
    sched._run = lambda job: job

    # Tick J0 once so J0 enters stalled; then J1_after_prep must NOT stay stuck in Group 3 because prep-done is completed!
    t1 = sched.tick()
    assert t1["job_id"] == "J0"

    t2 = sched.tick()
    assert t2["job_id"] == "J1_after_prep"
    assert repo.get("jobs", "J1_after_prep", owner)["awaiting_prep"] is False
    assert repo.get("jobs", "J1_after_prep", owner)["state"] == "waiting_for_capacity"

    # Now free the instance externally: J1_after_prep still beats J2_later!
    repo.update("instances", inst["id"], {"state": "retained_idle", "current_job_id": None}, owner)
    repo.update("leases", inst["lease_id"], {"state": "idle", "current_job_id": None}, owner)
    repo.update("jobs", "J0", {"state": "completed"}, owner)

    with pytest.raises(CreationInstancePending) as exc_info:
        svc.allocate(repo.get("jobs", "J2_later", owner))
    assert exc_info.value.blocker_job_id == "J1_after_prep"


def test_admit_job_populates_account_alias_snapshot_while_queued(tmp_path, monkeypatch):
    """Queued jobs with a bound account should expose account_alias_snapshot immediately instead of raw UUID."""
    store, repo, owner, proj_a, _, aid, wf = _seed_env(tmp_path)
    monkeypatch.setenv("GPU_REAL_ENABLED", "1")
    monkeypatch.setattr("app.admission._google_status", lambda *args, **kwargs: {"colab": {"connected": True}})

    body = {
        "project_id": proj_a["id"],
        "model_id": wf["model_id"],
        "workflow_key": wf["workflow_key"],
        "workflow_version": wf["workflow_version"],
        "params": {},
        "asset_ids": [],
        "execution_policy": {
            "provider": "colab",
            "account_id": aid,
            "lifecycle": "smart",
            "retain_idle_minutes": 15,
            "max_runtime_minutes": 30,
        },
    }
    job = admit_job(store, {"id": owner}, body, "idem-alias-1")
    assert job["account_id"] == aid
    assert job["account_alias_snapshot"] == "账号2"

    validated = admit_job(store, {"id": owner}, body, "idem-alias-2", validate_only=True)
    assert validated["account_alias_snapshot"] == "账号2"


def test_stalled_waiting_job_retains_bounded_retry_during_continuous_unrelated_submissions(tmp_path):
    """Under continuous unrelated submissions (no local capacity_epoch bump), a stalled waiting job still gets retried."""
    store, repo, owner, proj_a, _, aid, _ = _seed_env(tmp_path)
    inst, _ = _add_instance(repo, owner, aid, spec=L4, state="busy", current_job_id="external-job")
    cfg = GpuConfig(enabled=True, real_enabled=True, unit_rate=0.1)
    svc = InstanceService(repo, clock=lambda: 1000.0, config=cfg)
    sched = Scheduler(repo, instance_service=svc, clock=lambda: 1000.0, config=cfg)
    sched._await_prep = lambda _job: False

    _create_reuse_job(
        repo, owner, proj_a["id"], aid, inst["id"],
        job_id="J_waiting", created_epoch=100.0, state="waiting_for_capacity",
    )
    # First tick stalls J_waiting while inst is busy.
    t0 = sched.tick()
    assert t0["job_id"] == "J_waiting"
    assert t0["advanced"] is False

    # External process frees inst without notifying this worker's rr["capacity_epoch"].
    repo.update("instances", inst["id"], {"state": "retained_idle", "current_job_id": None}, owner)
    repo.update("leases", inst["lease_id"], {"state": "idle", "current_job_id": None}, owner)

    # Stream of unrelated queued mock jobs arrives every tick; J_waiting must still wake up within bounded deferrals.
    for i in range(12):
        repo.create(
            "jobs",
            owner,
            {"state": "queued", "provider": "mock", "created_epoch": 200.0 + i},
            record_id=f"unrelated-{i}",
        )
        sched.tick()
        if repo.get("jobs", "J_waiting", owner)["state"] == "allocating":
            break

    assert repo.get("jobs", "J_waiting", owner)["state"] == "allocating"
    assert repo.get("jobs", "J_waiting", owner)["instance_id"] == inst["id"]


def test_same_batch_handoff_remains_continuous_even_if_earlier_waiter_becomes_ready(tmp_path):
    """Deliberate policy: once a batch is running on a lease, handoff_lease finishes the batch consecutively."""
    store, repo, owner, proj_a, _, aid, _ = _seed_env(tmp_path)
    inst, lease = _add_instance(repo, owner, aid, spec=L4, state="busy", current_job_id="B0")
    sig = "compat-sig-moss"
    repo.create("executions", owner, {"state": "running", "job_id": "B0", "lease_id": lease["id"], "instance_id": inst["id"], "account_id": aid}, record_id="exec-b0")
    repo.create(
        "jobs",
        owner,
        {
            "state": "cleaning_up",
            "cleanup_final_state": "completed",
            "project_id": proj_a["id"],
            "provider": "colab",
            "instance_pool": POOL_CREATION,
            "account_id": aid,
            "instance_id": inst["id"],
            "lease_id": lease["id"],
            "execution_id": "exec-b0",
            "batch_id": "batch-1",
            "queue_order": 0,
            "compatibility_signature": sig,
            "created_epoch": 300.0,
            "snapshot": {"manifest": {"compute_spec": L4}},
            "execution_policy": {"provider": "colab", "account_id": aid, "lifecycle": "smart", "retain_idle_minutes": 15},
        },
        project_id=proj_a["id"],
        record_id="B0",
    )
    b1 = _create_reuse_job(
        repo, owner, proj_a["id"], aid, inst["id"],
        job_id="B1", created_epoch=300.0, state="waiting_for_capacity", batch_id="batch-1", queue_order=1,
    )
    repo.update("jobs", b1["id"], {"compatibility_signature": sig}, owner)

    # Earlier standalone job became ready while B0 was running.
    _create_reuse_job(
        repo, owner, proj_a["id"], aid, inst["id"],
        job_id="J_earlier", created_epoch=100.0, state="waiting_for_capacity",
    )

    class InspectReadyProvider:
        def inspect(self, _handle):
            return {"ready": True, "deleted": False}

    cfg = GpuConfig(enabled=True, real_enabled=True, unit_rate=0.1)
    svc = InstanceService(repo, clock=lambda: 1000.0, config=cfg)
    sched = Scheduler(
        repo,
        provider_factory=lambda _: InspectReadyProvider(),
        instance_service=svc,
        clock=lambda: 1000.0,
        config=cfg,
    )
    res = sched.tick()
    assert res["job_id"] == "B0"
    assert res["state"] == "completed"
    # Lease handed directly to B1 (same-batch continuity preserved).
    assert repo.get("jobs", "B1", owner)["lease_id"] == lease["id"]
    assert repo.get("jobs", "B1", owner)["instance_id"] == inst["id"]
    assert repo.get("jobs", "J_earlier", owner).get("instance_id") is None

