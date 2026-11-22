"""Three-pool hard isolation tests (Spec §2 & §8.1): native / creation / ephemeral."""

from __future__ import annotations

import time
import uuid

import pytest
from app.common import APIError, current_user, get_store
from app.config import settings
from app.db import Store
from app.gpu import native, native_restart
from app.gpu.instances import (
    POOL_CREATION,
    POOL_EPHEMERAL,
    POOL_NATIVE,
    AllocationError,
    InstanceService,
)
from app.gpu.legacy_migration import repair_legacy_identity
from app.gpu.repository import Repository
from app.main import app
from fastapi.testclient import TestClient

L4 = {"variant": "VARIANT_GPU", "accelerator": "L4", "shape": "SHAPE_STANDARD"}


def _setup_owner(tmp_path):
    store = Store(f"sqlite:///{tmp_path / 'isolation.sqlite'}")
    repo = Repository(store)
    owner = "owner-iso"
    aid = str(uuid.uuid4())
    repo.create(
        "accounts",
        owner,
        {
            "account_id": aid,
            "provider": "colab",
            "enabled": True,
            "auth_status": "connected",
            "allow_allocation": True,
            "budget": {"daily": 100.0, "monthly": 500.0},
        },
        record_id=aid,
    )
    return store, repo, owner, aid


def _make_idle_instance(repo, owner, aid, *, pool: str, spec=L4, native_comfy: bool = False, restarting: bool = False):
    deadline = time.time() + 3600.0
    lease_data = {
        "state": "idle",
        "provider": "colab",
        "account_id": aid,
        "instance_pool": pool,
        "native_comfy": native_comfy,
        "native_ready": native_comfy,
        "deadline_epoch": deadline,
        "idle_deadline_at": deadline,
        "handle": {"runtime_id": f"rt-{pool}-{uuid.uuid4().hex[:6]}", "owner_id": owner},
    }
    if restarting:
        lease_data["native_restart"] = {"id": "rst-1", "phase": "rebooting", "started_at": time.time()}
    lease = repo.create("leases", owner, lease_data)
    inst = repo.create(
        "instances",
        owner,
        {
            "account_id": aid,
            "provider": "colab",
            "instance_pool": pool,
            "native_comfy": native_comfy,
            "state": "retained_idle",
            "retention_mode": "retain" if native_comfy else "smart",
            "retain_idle_minutes": 30,
            "lease_id": lease["id"],
            "runtime_spec_requested": {"key": spec},
            "runtime_spec_reported": {"key": spec},
            "hard_deadline_at": deadline,
            "idle_deadline_at": deadline,
        },
    )
    repo.update("leases", lease["id"], {"instance_id": inst["id"]}, owner)
    return inst, lease


def test_native_running_or_restarting_never_blocks_or_gets_reused_by_creation_or_ephemeral(tmp_path):
    store, repo, owner, aid = _setup_owner(tmp_path)
    native_inst, native_lease = _make_idle_instance(
        repo, owner, aid, pool=POOL_NATIVE, native_comfy=True, restarting=True
    )
    service = InstanceService(repo, clock=lambda: 100.0, config={"unit_rate": 0.1, "daily_units": 50.0, "monthly_units": 200.0})

    # 1. guard_generation does NOT block creation/ephemeral during native restart
    native.guard_generation(repo, owner, {"provider": "colab", "account_id": aid})

    # 2. Frontend creation job opens a fresh creation instance (never reuses native)
    creation_job = {
        "id": "job-creation-1",
        "owner_id": owner,
        "project_id": "proj-1",
        "provider": "colab",
        "instance_pool": POOL_CREATION,
        "requested_spec": {"key": L4},
        "snapshot": {"manifest": {"compute_spec": L4}},
        "execution_policy": {"provider": "colab", "account_id": aid, "lifecycle": "smart", "retain_idle_minutes": 20},
    }
    created = service.allocate(creation_job)
    assert created["id"] != native_inst["id"]
    assert created["instance_pool"] == POOL_CREATION
    assert created["selection_reason"] in {"requested_account", "round_robin"}

    # 3. Creation job explicitly targeting native instance is rejected
    with pytest.raises(AllocationError, match="不属于创作池"):
        service.allocate({
            **creation_job,
            "id": "job-creation-bad",
            "requested_instance_id": native_inst["id"],
            "execution_policy": {"provider": "colab", "instance_id": native_inst["id"]},
        })

    # 4. Creation API auto_start=False does NOT treat native instance as creation capacity
    api_job = {
        "id": "job-api-1",
        "owner_id": owner,
        "project_id": "proj-1",
        "provider": "colab",
        "instance_pool": POOL_CREATION,
        "creation_existing_only": True,
        "requested_spec": {"key": L4},
        "snapshot": {"creation_api": True, "manifest": {"compute_spec": L4}},
        "execution_policy": {"provider": "colab", "account_id": aid},
    }
    # Release creation_job's instance so ONLY native_inst is active
    repo.update("instances", created["id"], {"state": "deleted", "runtime_deleted": True}, owner)
    assert not service.creation_capacity(api_job, [])
    inspect = service.inspect_creation_api_admission(api_job, [])
    assert inspect["status"] == "empty"

    # 5. Ephemeral CUDA smoke / workflow smoke opens its own ephemeral instance (never reuses native)
    smoke_job = {
        "id": "job-smoke-1",
        "owner_id": owner,
        "project_id": "proj-1",
        "provider": "colab",
        "instance_pool": POOL_EPHEMERAL,
        "requested_spec": {"key": L4},
        "snapshot": {"system_smoke": True, "manifest": {"executor": "cuda-smoke"}},
        "execution_policy": {"provider": "colab", "account_id": aid, "lifecycle": "save"},
    }
    smoke_inst = service.allocate(smoke_job)
    assert smoke_inst["id"] != native_inst["id"]
    assert smoke_inst["instance_pool"] == POOL_EPHEMERAL
    assert smoke_inst["selection_reason"] in {"requested_account", "round_robin"}


def test_native_admit_never_takes_over_idle_creation_instance(tmp_path, monkeypatch):
    store, repo, owner, aid = _setup_owner(tmp_path)
    creation_inst, _ = _make_idle_instance(repo, owner, aid, pool=POOL_CREATION, native_comfy=False)
    monkeypatch.setattr(settings, "native_comfy_url", "https://native.invalid")
    monkeypatch.setattr(native, "backend_for", lambda *args, **kwargs: ("https://native.invalid", {}))
    monkeypatch.setenv("GPU_REAL_ENABLED", "1")
    monkeypatch.setenv("GPU_UNIT_RATE", "0.1")
    monkeypatch.setenv("NATIVE_COMFY_COMMIT", "1" * 40)
    monkeypatch.setattr(native, "secret_file", lambda _: "token-secret")

    # Explicit creation instance_id must be rejected on native start
    with pytest.raises(APIError) as exc_info:
        native.admit(
            repo,
            owner,
            {"account_id": aid, "spec": L4, "instance_id": creation_inst["id"]},
            "native-bad-key",
        )
    assert exc_info.value.status == 409

    # Normal native start creates a fresh native-pool job without binding to creation_inst
    job = native.admit(repo, owner, {"account_id": aid, "spec": L4}, "native-ok-key")
    assert job["instance_pool"] == POOL_NATIVE
    assert job.get("requested_instance_id") is None
    assert job.get("instance_id") != creation_inst["id"]
    # Creation instance remains untouched
    assert repo.get("instances", creation_inst["id"], owner)["state"] == "retained_idle"


def test_ephemeral_smoke_and_prep_ignore_same_spec_idle_creation_instance(tmp_path):
    store, repo, owner, aid = _setup_owner(tmp_path)
    creation_inst, creation_lease = _make_idle_instance(repo, owner, aid, pool=POOL_CREATION, spec=L4)
    service = InstanceService(repo, clock=lambda: 100.0, config={"unit_rate": 0.1, "daily_units": 50.0, "monthly_units": 200.0})

    for kind, extra_snap in [
        ("cuda-smoke", {"system_smoke": True, "manifest": {"executor": "cuda-smoke"}}),
        ("real-smoke", {"real_smoke": True, "manifest": {"compute_spec": L4}}),
        ("prep", {"prep": True, "manifest": {"compute_spec": L4}}),
    ]:
        ephemeral_job = {
            "id": f"job-{kind}",
            "owner_id": owner,
            "project_id": "proj-1",
            "provider": "colab",
            "instance_pool": POOL_EPHEMERAL,
            "prep": kind == "prep",
            "requested_spec": {"key": L4},
            "snapshot": extra_snap,
            "execution_policy": {"provider": "colab", "account_id": aid, "lifecycle": "smart", "retain_idle_minutes": 30},
        }
        allocated = service.allocate(ephemeral_job)
        assert allocated["id"] != creation_inst["id"]
        assert allocated["instance_pool"] == POOL_EPHEMERAL
        # Completing ephemeral ALWAYS releases itself (even if policy said smart) and leaves creation_inst intact
        repo.create("jobs", owner, {**ephemeral_job, "instance_id": allocated["id"]}, record_id=ephemeral_job["id"])
        completed = service.complete({**ephemeral_job, "instance_id": allocated["id"]}, success=True)
        assert completed["state"] == "releasing"

    assert repo.get("instances", creation_inst["id"], owner)["state"] == "retained_idle"
    assert repo.get("leases", creation_lease["id"], owner)["state"] == "idle"


def test_native_restart_not_blocked_by_queued_creation_job(tmp_path, monkeypatch):
    store, repo, owner, aid = _setup_owner(tmp_path)
    native_inst, native_lease = _make_idle_instance(repo, owner, aid, pool=POOL_NATIVE, native_comfy=True)
    # Queue a creation job on the same account — must NOT block native ComfyUI restart
    repo.create(
        "jobs",
        owner,
        {
            "state": "queued",
            "provider": "colab",
            "instance_pool": POOL_CREATION,
            "account_id": aid,
            "snapshot": {"manifest": {"compute_spec": L4}},
        },
    )
    monkeypatch.setattr(native, "backend", lambda: ("https://native.invalid", {}))
    import httpx

    def handler(request):
        if request.url.path == "/queue":
            return httpx.Response(200, json={"queue_running": [], "queue_pending": []})
        if request.url.path == "/studio_native/state":
            return httpx.Response(200, json={"boot_id": "b1", "lease_id": native_lease["id"], "busy": False})
        return httpx.Response(200, json={})

    client_cls = httpx.Client
    monkeypatch.setattr(
        native_restart.httpx,
        "Client",
        lambda **kwargs: client_cls(transport=httpx.MockTransport(handler), **kwargs),
    )
    restart_id = native_restart.submit(repo, owner, native_inst["id"], "rst-isolated")
    assert restart_id


def test_legacy_instance_pool_backfill_migration(tmp_path):
    store = Store(f"sqlite:///{tmp_path / 'migrate.sqlite'}")
    repo = Repository(store)
    owner = "owner-mig"

    # Pre-migration legacy rows lacking instance_pool
    n_inst = repo.create("instances", owner, {"state": "retained_idle", "native_comfy": True, "provider": "colab"})
    s_inst = repo.create("instances", owner, {"state": "deleted", "provider": "colab"})
    repo.create("jobs", owner, {"state": "completed", "instance_id": s_inst["id"], "snapshot": {"system_smoke": True}})
    p_inst = repo.create("instances", owner, {"state": "deleted", "provider": "colab"})
    repo.create("jobs", owner, {"state": "completed", "instance_id": p_inst["id"], "prep": True, "snapshot": {"prep": True}})
    c_inst = repo.create("instances", owner, {"state": "retained_idle", "provider": "colab"})

    with store.engine.begin() as conn:
        repair_legacy_identity(conn)

    assert repo.get("instances", n_inst["id"], owner)["instance_pool"] == POOL_NATIVE
    assert repo.get("instances", s_inst["id"], owner)["instance_pool"] == POOL_EPHEMERAL
    assert repo.get("instances", p_inst["id"], owner)["instance_pool"] == POOL_EPHEMERAL
    assert repo.get("instances", c_inst["id"], owner)["instance_pool"] == POOL_CREATION


def test_http_cuda_smoke_and_real_smoke_stamp_ephemeral_pool(tmp_path, monkeypatch):
    store, repo, owner, aid = _setup_owner(tmp_path)
    monkeypatch.setenv("GPU_REAL_ENABLED", "1")
    monkeypatch.setenv("GPU_UNIT_RATE", "0.1")
    monkeypatch.setattr(app.state, "store", store, raising=False)
    monkeypatch.setitem(app.dependency_overrides, current_user, lambda: {"id": owner, "role": "admin"})
    monkeypatch.setitem(app.dependency_overrides, get_store, lambda: store)
    monkeypatch.setattr("app.compute_routes.GoogleService.status", lambda self: {"colab": {"connected": True}})
    monkeypatch.setattr(
        "app.compute_routes.GoogleColabProvider.list_specs",
        lambda self: [{"eligible": True, "key": L4, "gpu": "L4", "reference_rate_hourly": 5.0}],
    )
    with TestClient(app) as client:
        resp = client.post(
            "/api/gpu/admin/compute/smoke",
            headers={"Idempotency-Key": "cuda-smoke-iso"},
            json={"confirmed": True, "account_id": aid, "spec": L4},
        )
        assert resp.status_code == 202, resp.text
        job_id = resp.json()["data"]["job_id"]
        assert repo.get("jobs", job_id, owner)["instance_pool"] == POOL_EPHEMERAL
