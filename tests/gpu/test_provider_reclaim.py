"""Reclaim an owned Colab runtime with a real Store and a fake control plane."""

import httpx
import pytest
from app.common import current_user
from app.config import settings
from app.db import Store
from app.gpu import native
from app.gpu.colab import RUNTIME_MISSING, GoogleColabProvider
from app.gpu.compute import GpuConfig
from app.gpu.repository import Repository
from app.gpu.scheduler import Scheduler
from app.main import app
from fastapi.testclient import TestClient


@pytest.fixture
def runtime(tmp_path):
    repo = Repository(Store(f"sqlite:///{tmp_path / 'reclaim.sqlite'}"))
    deadline = 1000 + 1440 * 60
    repo.create("accounts", "owner", {
        "provider": "colab", "enabled": True, "connected": True, "allow_allocation": True,
    }, record_id="account")
    repo.create("instances", "owner", {
        "state": "retained_idle", "provider": "colab", "account_id": "account", "lease_id": "lease",
        "runtime_deleted": False, "current_job_id": None, "generation": 2,
        "retained_until": deadline, "idle_deadline_at": deadline, "hard_deadline_at": deadline,
        "budget_reserved": 5.0, "retention_reserved": 4.0, "allocation_requested_at": 940,
    }, record_id="instance")
    repo.create("leases", "owner", {
        "state": "idle", "provider": "colab", "owner_job_id": "job", "account_id": "account",
        "instance_id": "instance", "native_comfy": True, "native_ready": True, "operation_done": True,
        "runtime_id": "studio-reclaim", "request_id": "request",
        "idle_deadline_at": deadline, "deadline_epoch": deadline,
        "handle": {"runtime_name": "runtimes/studio-reclaim", "runtime_id": "studio-reclaim",
                   "request_id": "request", "owner_id": "owner", "job_id": "old-job"},
    }, record_id="lease")
    repo.create("jobs", "owner", {
        "state": "completed", "provider": "colab", "snapshot": {"native_comfy": True},
        "execution_policy": {"lifecycle": "retain", "retain_idle_minutes": 1440},
        "account_id": "account", "instance_id": "instance", "lease_id": "lease",
        "instance_state": "retained_idle", "runtime_deleted": False, "retained_until": deadline,
        "output_asset_ids": ["saved-image"],
    }, record_id="job")
    repo.store.create("assets", "owner", {"filename": "result.png"}, record_id="saved-image")
    return repo


class ControlPlane:
    def __init__(self, result=None):
        self.result = {"deleted": True} if result is None else result
        self.calls = []

    def __call__(self, job):
        assert (job["owner_id"], job["account_id"], job["id"]) == ("owner", "account", "job")
        return self

    def inspect(self, handle):
        assert handle["job_id"] == handle["owner_job_id"] == "job"
        assert handle["inspection_lease_id"] == "lease"
        self.calls.append(handle["runtime_name"])
        if callable(self.result):
            return self.result()
        return self.result

    def delete(self, handle):
        pytest.fail("a reclaim check must never unassign a runtime")


def scheduler(repo, provider, clock=lambda: 1000):
    return Scheduler(repo, provider_factory=provider, clock=clock,
                     config=GpuConfig(unit_rate=0.1, daily_units=20, monthly_units=20))


def reclaim_events(repo):
    return [event for event in repo.list("events", "owner") if event.get("reason") == RUNTIME_MISSING]


def test_studio_watchdog_only_keeps_smart_colab_leases_alive(runtime):
    repo = runtime
    repo.update("leases", "lease", {"native_comfy": False, "native_ready": False}, "owner")
    repo.update("jobs", "job", {"snapshot": {}, "execution_policy": {"lifecycle": "smart"}}, "owner")
    repo.update("instances", "instance", {"instance_pool": "creation", "retention_mode": "smart", "idle_deadline_at": 1500}, "owner")
    deadline = repo.get("leases", "lease")["deadline_epoch"]
    heartbeats = []

    class Provider(ControlPlane):
        def keep_alive(self, lease, *, run_kernel_exec):
            assert lease["id"] == "lease"
            heartbeats.append(run_kernel_exec)
            return {"ok": True, "kernel_exec": "ok" if run_kernel_exec else "skipped"}

    now = 1000
    watch = scheduler(repo, Provider({"ready": True}), lambda: now)
    watch.watchdog()
    now = 1059
    watch.watchdog()
    assert heartbeats == [True]
    now = 1060
    watch.watchdog()
    now = 1240
    watch.watchdog()
    assert heartbeats == [True, False, True]
    assert repo.get("leases", "lease")["deadline_epoch"] == deadline
    assert repo.get("instances", "instance")["idle_deadline_at"] == 1500

    # Creation-pool `retain` instances also receive watchdog keepalive, decoupled from last job state/policy.
    repo.update("instances", "instance", {"retention_mode": "retain"}, "owner")
    now = 1300
    watch.watchdog()
    assert heartbeats == [True, False, True, False]

    # Even if the last job failed or had a different policy, instance retention_mode drives keepalive.
    repo.update("instances", "instance", {"retention_mode": "smart"}, "owner")
    repo.update("jobs", "job", {"state": "failed", "execution_policy": {"lifecycle": "save"}}, "owner")
    now = 1360
    watch.watchdog()
    assert heartbeats == [True, False, True, False, False]

    # Instance retention_mode=save never receives keepalive.
    repo.update("instances", "instance", {"retention_mode": "save"}, "owner")
    now = 1420
    watch.watchdog()
    assert heartbeats == [True, False, True, False, False]

    # Expired instance never receives keepalive even if retention_mode=smart.
    repo.update("instances", "instance", {"retention_mode": "smart"}, "owner")
    now = 1500
    watch.watchdog()
    assert heartbeats == [True, False, True, False, False]


def test_retained_native_reclaim_reaches_api_and_preserves_success(runtime, monkeypatch):
    repo = runtime
    monkeypatch.setattr(native, "busy", lambda: True)
    monkeypatch.setattr(app.state, "store", repo.store, raising=False)
    monkeypatch.setitem(app.dependency_overrides, current_user, lambda: {"id": "owner", "role": "admin"})
    provider = ControlPlane()
    watch = scheduler(repo, provider)
    monkeypatch.setattr(native, "backend", lambda: ("https://comfy.invalid", {}))
    async_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: async_client(
        transport=httpx.MockTransport(lambda request: httpx.Response(530)), **kwargs,
    ))
    with TestClient(app) as client:
        assert client.get("/api/gpu/instances/instance").json()["data"]["state"] == "retained_idle"
        before = client.get("/api/gpu/native/status").json()["data"]
        assert before["ready"] is False and before["reason"] is None
        assert watch.watchdog()["recovered"] == 1
        assert watch.watchdog()["recovered"] == 0
        instance = client.get("/api/gpu/instances/instance").json()["data"]
        assert instance["state"] == "deleted" and instance["runtime_deleted"] is True
        assert instance["remote_delete_source"] == RUNTIME_MISSING
        assert not client.get("/api/gpu/instances?state=retained_idle").json()["data"]
        lease = repo.get("leases", "lease", "owner")
        assert lease["state"] == "deleted" and lease["native_ready"] is False
        assert lease["provider_runtime_delete_evidence"]["instance_generation"] == 2
        native_status = client.get("/api/gpu/native/status").json()["data"]
        assert (native_status["ready"], native_status["state"], native_status["reason"]) == (
            False, "completed", RUNTIME_MISSING,
        )
        assert native_status["error"] == "Colab 运行时已回收"
        assert native_status["instance_id"] is None
        for job in (client.get("/api/jobs").json()["data"][0], client.get("/api/jobs/job").json()["data"]):
            assert job["state"] == "completed"
            assert job["instance_state"] == "deleted" and job["runtime_deleted"] is True
            assert job["retained_until"] is None and job["output_asset_ids"] == ["saved-image"]
        assert client.get("/api/jobs/job/outputs").json()["data"][0]["filename"] == "result.png"
        monkeypatch.setattr(settings, "native_comfy_url", "https://comfy.invalid")
        monkeypatch.setenv("GPU_REAL_ENABLED", "1")
        monkeypatch.setenv("GPU_UNIT_RATE", "0.1")
        monkeypatch.setenv("NATIVE_COMFY_COMMIT", "1" * 40)
        monkeypatch.setattr(native, "secret_file", lambda _: "offline-test-only")
        spec = {"variant": "VARIANT_GPU", "accelerator": "L4", "shape": "SHAPE_STANDARD"}
        reply = client.post("/api/gpu/native/start", headers={"Idempotency-Key": "relaunch"}, json={
            "account_id": "account", "spec": spec, "instance_id": "instance",
        })
        assert reply.status_code == 409
        reply = client.post("/api/gpu/native/start", headers={"Idempotency-Key": "relaunch"}, json={
            "account_id": "account", "spec": spec,
        })
        assert reply.status_code == 202, reply.text
        assert reply.json()["data"]["state"] == "queued"
        assert all(lease["state"] == "deleted" for lease in repo.list("leases"))
    assert len(reclaim_events(repo)) == len(repo.list("instance_usage_segments")) == len(provider.calls) == 1
    usage = repo.list("instance_usage_segments")[0]
    assert usage["actual_cu"] == 0 and usage["state"] == "reconciling"
    assert usage["session_kind"] == "native_comfy" and usage["close_reason"] == RUNTIME_MISSING


def test_legacy_timestamp_requires_fresh_evidence_and_reuses_backoff_after_restart(runtime):
    repo = runtime
    repo.update("leases", "lease", {"native_ready": False, "native_remote_deleted_verified_at": 999}, "owner")
    provider = ControlPlane({"deleted": True, "allocation_unknown": True})
    assert scheduler(repo, provider).watchdog()["recovered"] == 0
    assert repo.get("instances", "instance")["state"] == "retained_idle"
    provider.result = {"deleted": True}
    assert scheduler(repo, provider, lambda: 1089).watchdog()["recovered"] == 0
    assert len(provider.calls) == 1
    assert scheduler(repo, provider, lambda: 1090).watchdog()["recovered"] == 1
    assert repo.get("leases", "lease")["provider_runtime_delete_evidence"]["checked_at"] == 1090


@pytest.mark.parametrize("status", [401, 403, 429, 503, 530, "timeout", "present", "404", "malformed", "empty"])
def test_exact_cli_assignments_response_controls_reclaim(runtime, monkeypatch, status):
    repo = runtime
    repo.update("leases", "lease", {"tunnel_endpoint": "owned-runtime", "native_ready": False}, "owner")
    calls = []

    def serve(request):
        assert request.method == "GET" and request.url.path == "/tun/m/assignments"
        calls.append(request.method)
        if status == "timeout":
            raise httpx.ReadTimeout("offline timeout")
        if status == "present":
            return httpx.Response(200, json={"assignments": [{"endpoint": "owned-runtime"}]})
        if status == "malformed":
            return httpx.Response(200, json={})
        if status == "empty":
            return httpx.Response(200, json={"assignments": []})
        return httpx.Response(int(status), json={"assignments": []})

    class Google:
        def access_token(self, purpose):
            return "offline-test-only"

    with httpx.Client(transport=httpx.MockTransport(serve)) as client:
        provider = GoogleColabProvider(repo, "owner", account_id="account", client=client, google=Google())
        monkeypatch.setattr(provider, "delete", ControlPlane().delete)
        watch = scheduler(repo, lambda job: provider)
        assert watch.watchdog()["recovered"] == int(status == "empty")
        assert watch.watchdog()["recovered"] == 0
    assert calls == ["GET"]
    assert repo.get("leases", "lease")["state"] == ("deleted" if status == "empty" else "idle")


@pytest.mark.parametrize("kind,patch", [
    ("leases", {"owner_job_id": "replacement-job"}),
    ("leases", {"runtime_id": "replacement-runtime"}),
    ("leases", {"account_id": "replacement-account"}),
    ("leases", {"native_last_write": 1000}),
    ("instances", {"current_job_id": "replacement-job"}),
    ("instances", {"retention_reserved": 8}),
    ("jobs", {"lease_id": "replacement-lease"}),
    ("jobs", {"execution_id": "replacement-execution"}),
])
def test_reclaim_discards_stale_inspection(runtime, kind, patch):
    row_id = {"leases": "lease", "instances": "instance", "jobs": "job"}[kind]

    def inspect():
        runtime.update(kind, row_id, patch, "owner")
        return {"deleted": True}

    assert scheduler(runtime, ControlPlane(inspect)).watchdog()["recovered"] == 0
    assert not reclaim_events(runtime)
    assert runtime.get("instances", "instance")["state"] == "retained_idle"
    assert runtime.get("leases", "lease")["state"] == "idle"


@pytest.mark.parametrize("state,final", [("running", "failed"), ("uploading", "failed"), ("cleaning_up", "completed")])
def test_active_reclaim_uses_cleanup_state_machine(runtime, state, final):
    repo = runtime
    output_ids = ["saved-image"] if final == "completed" else []
    repo.update("jobs", "job", {"state": state, "execution_id": "execution", "output_asset_ids": output_ids,
                                "cleanup_final_state": "completed" if state == "cleaning_up" else None}, "owner")
    repo.create("executions", "owner", {"job_id": "job", "state": state}, record_id="execution")
    repo.update("leases", "lease", {"state": "active", "current_job_id": "job"}, "owner")
    repo.update("instances", "instance", {"state": "busy", "current_job_id": "job"}, "owner")
    watch = scheduler(repo, ControlPlane())
    watch.watchdog()
    job = repo.get("jobs", "job")
    assert job["state"] == repo.get("executions", "execution")["state"] == final
    assert job["output_asset_ids"] == output_ids
    assert job.get("error_code") == (RUNTIME_MISSING if final == "failed" else None)
    assert watch.watchdog()["recovered"] == 0
    assert len(reclaim_events(repo)) == 1


def test_worker_cleanup_checks_before_retaining(runtime):
    runtime.update("jobs", "job", {"state": "cleaning_up", "cleanup_final_state": "completed",
                                   "instance_completion_recorded": True}, "owner")
    watch = scheduler(runtime, ControlPlane())
    assert watch.tick()["state"] == "completed"
    assert runtime.get("instances", "instance")["state"] == "deleted"
    assert len(reclaim_events(runtime)) == 1


def test_second_watchdog_and_retry_after_rollback_are_idempotent(runtime, monkeypatch):
    repo = runtime
    watch = scheduler(repo, ControlPlane())

    def interrupted(*args, **kwargs):
        raise RuntimeError("simulated shutdown before usage and event commit")

    monkeypatch.setattr("app.gpu.ccu.close_instance_session", interrupted)
    with pytest.raises(RuntimeError, match="simulated shutdown"):
        watch.watchdog()
    reopened = Repository(Store(repo.store.url))
    assert reopened.get("instances", "instance")["state"] == "retained_idle"
    assert reopened.get("leases", "lease")["state"] == "idle"
    assert not reclaim_events(reopened)

    monkeypatch.undo()

    def competing_watchdog():
        assert scheduler(reopened, ControlPlane()).watchdog()["recovered"] == 1
        return {"deleted": True}

    assert scheduler(repo, ControlPlane(competing_watchdog)).watchdog()["recovered"] == 0
    assert len(reclaim_events(reopened)) == len(reopened.list("instance_usage_segments")) == 1


@pytest.mark.parametrize("forced", [False, True])
def test_native_cu_fuse_drains_or_deletes_using_confirmation(runtime, monkeypatch, forced):
    from app.common import APIError
    from app.gpu import ccu
    from app.gpu.cu_guard import CuGuardService

    repo = runtime
    now = 1000
    busy = True

    class Provider:
        calls = 0

        def inspect(self, _handle):
            assert not forced, "forced shutdown should immediately use provider deletion"
            return {"ready": True, "deleted": False}

        def delete(self, _handle):
            self.calls += 1
            return self.calls > 1

    provider = Provider()
    monkeypatch.setattr(native, "busy", lambda: busy)
    watch = scheduler(repo, lambda _: provider, lambda: now)
    guard = CuGuardService(repo, lambda: now)
    guard.save("owner", {"min_balance_cu": 5} if forced else {"daily_limit_cu": 1})
    for stamp, balance in [(990, 6), (999, 5)]:
        ccu.sample_and_record(repo, "owner", "account", "tick", sampler=lambda *_: {"balance": balance}, now=stamp)

    watch.watchdog()
    lease = repo.get("leases", "lease", "owner")
    assert lease["native_draining"] is True
    with pytest.raises(APIError, match="实例正在释放"):
        native.begin_write(repo, lease)
    if not forced:
        assert lease["state"] == "idle" and provider.calls == 0
        busy = False
        now += 1
        watch.watchdog()
    assert provider.calls == 1
    assert repo.get("leases", "lease")["state"] == "cleanup_pending"
    assert repo.get("instances", "instance")["state"] != "deleted"
    now += 3
    assert watch.watchdog()["recovered"] == 1
    assert provider.calls == 2
    assert repo.get("leases", "lease")["state"] == "deleted"
    assert repo.get("jobs", "job")["state"] == "completed"
    assert repo.list("instance_usage_segments", "owner")[0]["state"] == "reconciling"


@pytest.mark.parametrize("forced", [False, True])
def test_cu_fuse_respects_current_creation_job_and_cancels_only_for_balance_floor(runtime, forced):
    from app.gpu import ccu
    from app.gpu.cu_guard import CuGuardService

    repo = runtime
    repo.update("leases", "lease", {"native_ready": False, "native_comfy": False, "state": "active"}, "owner")
    repo.update("instances", "instance", {"state": "busy", "current_job_id": "job"}, "owner")
    repo.update("jobs", "job", {"state": "running", "execution_started_at": 900, "snapshot": {},
                                "execution_policy": {"lifecycle": "smart"}}, "owner")
    provider = ControlPlane({"ready": True, "deleted": False})
    provider.inspect = lambda _: {"ready": True, "deleted": False}
    deletes = []
    provider.delete = lambda handle: deletes.append(handle["runtime_name"]) or True

    watch = scheduler(repo, provider)
    guard = CuGuardService(repo, watch.clock)
    guard.save("owner", {"min_balance_cu": 5} if forced else {"daily_limit_cu": 1})
    for stamp, balance in [(990, 6), (999, 5)]:
        ccu.sample_and_record(repo, "owner", "account", "tick", sampler=lambda *_: {"balance": balance}, now=stamp)
    watch.watchdog()
    if not forced:
        assert repo.get("jobs", "job")["state"] == "running"
        assert not deletes
        assert repo.get("instances", "instance")["state"] == "busy"
        repo.transition_job("job", "owner", "cleaning_up", "task complete",
                            {"cleanup_final_state": "completed"}, 1001)
        watch.watchdog()
    job = repo.get("jobs", "job")
    assert job["state"] == ("cancelled" if forced else "completed")
    assert job["execution_seconds"] == (100 if forced else 101)
    assert deletes == ["runtimes/studio-reclaim"]
    assert repo.get("instances", "instance")["state"] == "deleted"
    assert any(event.get("reason") == ("hard_balance_floor" if forced else "soft_daily_limit")
               for event in repo.list("events", "owner"))
