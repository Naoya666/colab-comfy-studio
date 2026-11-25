"""Workflow smoke 10-min retention, 4-clock deadline recomputation, explicit binding & accept-smoke tests (Spec §9.1(3))."""

from __future__ import annotations

import base64
import json
import time
import uuid
from types import SimpleNamespace

import pytest
from app import review_routes
from app.common import APIError
from app.config import settings
from app.db import Store
from app.gpu import native
from app.gpu.compute import GpuConfig
from app.gpu.instances import POOL_CREATION, POOL_EPHEMERAL, InstanceService
from app.gpu.repository import Repository
from app.storage import save_asset

OWNER = {"id": "owner-smoke", "role": "admin"}
L4 = {"variant": "VARIANT_GPU", "accelerator": "L4", "shape": "SHAPE_STANDARD"}
T4 = {"variant": "VARIANT_GPU", "accelerator": "T4", "shape": "SHAPE_STANDARD"}
COMMIT = "a" * 40
REVISION = "b" * 40
SHA256 = "c" * 64
BASELINE = "d" * 64
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


class _Clock:
    def __init__(self, start: float | None = None):
        self.now = time.time() if start is None else start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += seconds
        return self.now


def _req(headers=None):
    return SimpleNamespace(headers=headers or {}, state=SimpleNamespace(request_id="req-smoke"))


def _body(resp):
    return json.loads(resp.body)["data"]


def _setup_review_env(tmp_path, monkeypatch):
    clock = _Clock()
    store = Store(f"sqlite:///{tmp_path / 'smoke.sqlite'}")
    repo = Repository(store)
    aid = str(uuid.uuid4())
    repo.create(
        "accounts",
        OWNER["id"],
        {
            "account_id": aid,
            "alias": "SmokeColab",
            "provider": "colab",
            "enabled": True,
            "auth_status": "connected",
            "allow_allocation": True,
            "budget": {"daily": 100.0, "monthly": 500.0},
        },
        record_id=aid,
    )
    store.create("projects", OWNER["id"], {"title": "SmokeProject"}, record_id="proj-smoke")
    repo.create(
        "models",
        OWNER["id"],
        {
            "model_id": "real-model",
            "source": "https://huggingface.co/acme/model",
            "revision": REVISION,
            "license": "apache-2.0",
            "resource_profile": {"vram_gb": 8, "ram_gb": 16},
            "enabled": False,
            "is_mock": False,
            "review_approved": True,
            "license_accepted": True,
        },
    )
    manifest = {
        "manifest_version": 1,
        "model_id": "real-model",
        "compute_spec": L4,
        "parameters": {
            "prompt": {"type": "string", "default": "hello", "binding": {"node": "1", "input": "text"}}
        },
        "dependencies": {
            "comfyui_commit": COMMIT,
            "custom_nodes": [],
            "weight_files": [
                {
                    "repo_id": "acme/model",
                    "filename": "model.safetensors",
                    "revision": REVISION,
                    "sha256": SHA256,
                    "target": "checkpoints/model.safetensors",
                }
            ],
        },
    }
    wf = repo.create(
        "workflows",
        OWNER["id"],
        {
            "workflow_key": "smoke-demo",
            "workflow_version": 1,
            "model_id": "real-model",
            "manifest": manifest,
            "api_graph": {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"text": "hello"}}},
            "object_info": {
                "CheckpointLoaderSimple": {"input": {"required": {"text": ["STRING"]}}, "output": ["IMAGE"]}
            },
            "state": "validated",
            "review_approved": True,
            "smoke_test_passed": False,
        },
    )
    cfg = GpuConfig(real_enabled=True, unit_rate=0.1, daily_units=100.0, monthly_units=500.0, max_job_units=50.0)
    monkeypatch.setattr(review_routes.GpuConfig, "from_env", lambda: cfg)
    monkeypatch.setattr(review_routes.GoogleService, "status", lambda self: {"colab": {"connected": True}})
    monkeypatch.setattr(
        review_routes.GoogleColabProvider,
        "list_specs",
        lambda self: [
            {"eligible": True, "key": L4, "label": "L4"},
            {"eligible": True, "key": T4, "label": "T4"},
        ],
    )
    svc = InstanceService(repo, clock=clock, config=cfg)
    return clock, store, repo, aid, wf, svc, cfg


def _seed_ephemeral_standby(repo: Repository, aid: str, clock: _Clock, *, spec=L4, pool=POOL_EPHEMERAL, ttl=600.0):
    now = clock()
    dl = now + ttl
    lease = repo.create(
        "leases",
        OWNER["id"],
        {
            "state": "idle",
            "provider": "colab",
            "account_id": aid,
            "instance_pool": pool,
            "ephemeral_standby": pool == POOL_EPHEMERAL,
            "generation_timeouts": True,
            "hard_deadline_at": None,
            "idle_deadline_at": dl,
            "deadline_epoch": dl,
            "vm_allocated_at": now - 100.0,
            "venv_isolated": True,
            "vm_baseline_digest": BASELINE,
            "handle": {"runtime_id": f"rt-{uuid.uuid4().hex[:6]}", "owner_id": OWNER["id"]},
        },
    )
    inst = repo.create(
        "instances",
        OWNER["id"],
        {
            "account_id": aid,
            "account_alias_snapshot": "SmokeColab",
            "provider": "colab",
            "instance_pool": pool,
            "ephemeral_standby": pool == POOL_EPHEMERAL,
            "state": "retained_idle",
            "retention_mode": "retain",
            "retain_idle_minutes": 10,
            "lease_id": lease["id"],
            "gpu_reported": "NVIDIA L4" if spec == L4 else "NVIDIA T4",
            "runtime_spec_requested": {"key": spec},
            "runtime_spec_reported": {"key": spec},
            "vm_allocated_at": now - 100.0,
            "venv_isolated": True,
            "vm_baseline_digest": BASELINE,
            "hard_deadline_at": None,
            "idle_deadline_at": dl,
        },
    )
    repo.update("leases", lease["id"], {"instance_id": inst["id"]}, OWNER["id"])
    return repo.get("instances", inst["id"], OWNER["id"]), repo.get("leases", lease["id"], OWNER["id"])


def test_smoke_default_releases_immediately_and_retain_keeps_10m_on_success_and_clean_failure(tmp_path, monkeypatch):
    clock, store, repo, aid, wf, svc, _ = _setup_review_env(tmp_path, monkeypatch)

    def run_smoke_complete(*, retain: bool, success: bool, clean_after_smoke: bool | None = None):
        job_id = str(uuid.uuid4())
        job = {
            "id": job_id,
            "owner_id": OWNER["id"],
            "project_id": "proj-smoke",
            "provider": "colab",
            "instance_pool": POOL_EPHEMERAL,
            "retain_after_smoke": retain,
            "requested_spec": {"key": L4},
            "snapshot": {"workflow_smoke": True, "real_smoke": True, "manifest": {"compute_spec": L4}},
            "execution_policy": {"provider": "colab", "account_id": aid, "lifecycle": "save"},
        }
        if clean_after_smoke is not None:
            job["clean_after_smoke"] = clean_after_smoke
        inst = svc.allocate(job)
        lease = repo.create(
            "leases",
            OWNER["id"],
            {
                "state": "active",
                "provider": "colab",
                "account_id": aid,
                "instance_id": inst["id"],
                "instance_pool": POOL_EPHEMERAL,
                "owner_job_id": job_id,
                "current_job_id": job_id,
                "handle": {"runtime_id": f"rt-smk-{job_id[:6]}", "owner_id": OWNER["id"]},
            },
        )
        repo.update(
            "instances",
            inst["id"],
            {
                "state": "busy",
                "lease_id": lease["id"],
                "venv_isolated": True,
                "vm_baseline_digest": BASELINE,
            },
            OWNER["id"],
        )
        job["instance_id"] = inst["id"]
        job["lease_id"] = lease["id"]
        repo.create("jobs", OWNER["id"], job, record_id=job_id)
        return svc.complete(job, success=success)

    # 1. Default retain_after_smoke=False -> immediately releasing.
    r_default = run_smoke_complete(retain=False, success=True, clean_after_smoke=True)
    assert r_default["state"] == "releasing"

    # 2. retain_after_smoke=True + success + clean_after_smoke=True -> 10m retained_idle standby + watchdog keepalive allowed.
    r_ok = run_smoke_complete(retain=True, success=True, clean_after_smoke=True)
    assert r_ok["state"] == "retained_idle"
    assert r_ok["ephemeral_standby"] is True
    assert r_ok["hard_deadline_at"] is None
    assert pytest.approx(r_ok["idle_deadline_at"] - clock(), abs=2.0) == 600.0
    r_ok_lease = repo.update(
        "leases",
        r_ok["lease_id"],
        {
            "state": "idle",
            "current_job_id": None,
            "idle_deadline_at": r_ok["idle_deadline_at"],
            "deadline_epoch": r_ok["idle_deadline_at"],
        },
        OWNER["id"],
    )
    assert native._smart_keepalive_allowed(repo, r_ok_lease, clock()) is True

    # 3. retain_after_smoke=True + clean failure -> ALSO retains 10m!
    r_clean_fail = run_smoke_complete(retain=True, success=False, clean_after_smoke=True)
    assert r_clean_fail["state"] == "retained_idle"
    assert r_clean_fail["ephemeral_standby"] is True

    # 4. retain_after_smoke=True + dirty failure -> Fail-Closed releasing!
    r_dirty_fail = run_smoke_complete(retain=True, success=False, clean_after_smoke=False)
    assert r_dirty_fail["state"] == "releasing"


def test_ephemeral_standby_claimed_at_minute_9_recomputes_lease_deadline(tmp_path, monkeypatch):
    clock, store, repo, aid, wf, svc, cfg = _setup_review_env(tmp_path, monkeypatch)
    t0 = clock()
    inst, lease = _seed_ephemeral_standby(repo, aid, clock, spec=L4, ttl=600.0)
    assert lease["deadline_epoch"] == t0 + 600.0

    # Advance to minute 9 (only 60s left on old standby countdown!).
    clock.advance(540.0)
    t_claim = clock()

    smoke_job = {
        "id": "job-min9",
        "owner_id": OWNER["id"],
        "project_id": "proj-smoke",
        "provider": "colab",
        "instance_pool": POOL_EPHEMERAL,
        "requested_instance_id": inst["id"],
        "max_runtime_minutes": 5,
        "retain_after_smoke": True,
        "requested_spec": {"key": L4},
        "snapshot": {"workflow_smoke": True, "real_smoke": True, "manifest": {"compute_spec": L4}},
        "execution_policy": {"provider": "colab", "account_id": aid, "instance_id": inst["id"], "lifecycle": "save"},
    }
    allocated = svc.allocate(smoke_job)
    assert allocated["id"] == inst["id"]
    assert allocated["state"] in {"allocating", "busy"}
    assert allocated["idle_deadline_at"] is None
    # Verify lease deadline_epoch was recomputed to t_claim + 300s (t0 + 840s), NOT stuck at t0 + 600s!
    updated_lease = repo.get("leases", lease["id"], OWNER["id"])
    assert pytest.approx(updated_lease["deadline_epoch"], abs=1.0) == t_claim + 300.0
    assert updated_lease["deadline_epoch"] > t0 + 600.0
    assert updated_lease["idle_deadline_at"] is None

    # At t0 + 650 (50s past the old standby deadline), lease deadline is still 190s in the future!
    repo.create("jobs", OWNER["id"], {**smoke_job, "instance_id": inst["id"], "lease_id": lease["id"]}, record_id="job-min9")
    clock.advance(110.0)
    assert clock() > t0 + 600.0
    assert clock() < updated_lease["deadline_epoch"]


def test_stale_requested_instance_id_returns_409_and_never_opens_new_gpu(tmp_path, monkeypatch):
    clock, store, repo, aid, wf, svc, _ = _setup_review_env(tmp_path, monkeypatch)

    # Case 1: Expired standby instance.
    expired_inst, _ = _seed_ephemeral_standby(repo, aid, clock, spec=L4, ttl=-10.0)
    # Case 2: Transferred out to creation pool.
    creation_inst, _ = _seed_ephemeral_standby(repo, aid, clock, spec=L4, pool=POOL_CREATION, ttl=600.0)
    # Case 3: Spec-mismatched ephemeral standby (T4 instead of L4).
    t4_inst, _ = _seed_ephemeral_standby(repo, aid, clock, spec=T4, ttl=600.0)

    jobs_before = len(repo.list("jobs", OWNER["id"]))
    for bad_id in (expired_inst["id"], creation_inst["id"], t4_inst["id"], "nonexistent-id"):
        with pytest.raises(APIError) as exc:
            review_routes.real_smoke(
                wf["id"],
                {
                    "project_id": "proj-smoke",
                    "account_id": aid,
                    "confirmed": True,
                    "spec": {"key": L4},
                    "retain_after_smoke": True,
                    "requested_instance_id": bad_id,
                },
                _req({"Idempotency-Key": f"smoke-bad-{bad_id}"}),
                OWNER,
                store,
            )
        assert exc.value.status == 409
        assert exc.value.code == "EPHEMERAL_INSTANCE_UNAVAILABLE"

    # Zero paid smoke jobs created!
    assert len(repo.list("jobs", OWNER["id"])) == jobs_before

    # Valid matching standby succeeds and binds requested_instance_id + retain_after_smoke!
    good_inst, _ = _seed_ephemeral_standby(repo, aid, clock, spec=L4, ttl=600.0)
    act = _body(review_routes.workflow_activation(wf["id"], _req(), OWNER, store))
    assert act["standby_ephemeral_instance"]["instance_id"] == good_inst["id"]
    assert any(item["instance_id"] == creation_inst["id"] for item in act["transferable_other_instances"])

    resp = review_routes.real_smoke(
        wf["id"],
        {
            "project_id": "proj-smoke",
            "account_id": aid,
            "confirmed": True,
            "spec": {"key": L4},
            "retain_after_smoke": True,
            "requested_instance_id": good_inst["id"],
        },
        _req({"Idempotency-Key": "smoke-good-standby"}),
        OWNER,
        store,
    )
    job = _body(resp)
    stored_job = repo.get("jobs", job["id"], OWNER["id"])
    assert stored_job["requested_instance_id"] == good_inst["id"]
    assert stored_job["retain_after_smoke"] is True


def test_accept_smoke_succeeds_during_standby_and_after_transfer_to_creation(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    clock, store, repo, aid, wf, svc, _ = _setup_review_env(tmp_path, monkeypatch)
    inst, lease = _seed_ephemeral_standby(repo, aid, clock, spec=L4, ttl=600.0)
    model = next(m for m in repo.list("models", OWNER["id"]) if m["model_id"] == "real-model")

    # Create a completed real smoke job with pinned gpu_diagnostics, completed execution & verified PNG output.
    asset = save_asset(store, OWNER["id"], "proj-smoke", PNG, "smoke.png", metadata={"real_smoke": True})
    pinned_diag = {
        "driver": "550.54.15",
        "device_name": "NVIDIA L4",
        "devices": [{"name": "NVIDIA L4", "memory_mib": 23034}],
        "cuda_available": True,
    }
    execution = repo.create("executions", OWNER["id"], {"state": "completed", "instance_id": inst["id"]})
    job = repo.create(
        "jobs",
        OWNER["id"],
        {
            "project_id": "proj-smoke",
            "provider": "colab",
            "account_id": aid,
            "instance_id": inst["id"],
            "lease_id": lease["id"],
            "execution_id": execution["id"],
            "target_workflow_id": wf["id"],
            "target_model_id": wf["model_id"],
            "workflow_smoke": True,
            "workflow_version_id": wf["id"],
            "workflow_key": wf["workflow_key"],
            "workflow_version": wf["workflow_version"],
            "model_id": wf["model_id"],
            "requested_spec": {"key": L4},
            "state": "completed",
            "runtime_deleted": False,
            "retain_after_smoke": True,
            "gpu_reported": "NVIDIA L4",
            "gpu_diagnostics": pinned_diag,
            "output_asset_ids": [asset["id"]],
            "smoke_compatibility_signature": review_routes._smoke_compatibility_signature(
                wf["id"], wf, model["revision"]
            ),
            "snapshot": {
                "real_smoke": True,
                "workflow_smoke": True,
                "workflow_version_id": wf["id"],
                "manifest": wf["manifest"],
            },
        },
    )
    # 1. Lease is still idle on ephemeral standby (not deleted!): accept_smoke MUST succeed.
    res1 = _body(review_routes.accept_smoke(wf["id"], {"job_id": job["id"], "confirmed": True}, _req(), OWNER, store))
    assert res1["smoke_test_passed"] is True
    assert repo.get("models", model["id"], OWNER["id"])["tested_compute_specs"]["actual_gpu"] == pinned_diag

    # 2. Now transfer instance & lease to creation pool and even overwrite mutable lease fields.
    repo.update("workflows", wf["id"], {"smoke_test_passed": False, "smoke_evidence": None}, OWNER["id"])
    repo.update("models", model["id"], {"enabled": False, "tested_compute_specs": None}, OWNER["id"])
    repo.update("instances", inst["id"], {"instance_pool": POOL_CREATION, "ephemeral_standby": False}, OWNER["id"])
    repo.update(
        "leases",
        lease["id"],
        {
            "instance_pool": POOL_CREATION,
            "owner_job_id": "some-other-creation-job",
            "gpu_diagnostics": {"device_name": "MUTATED", "devices": [{"name": "MUTATED"}]},
        },
        OWNER["id"],
    )
    # accept_smoke looks up lease by job.lease_id and strictly prefers pinned job.gpu_diagnostics!
    res2 = _body(review_routes.accept_smoke(wf["id"], {"job_id": job["id"], "confirmed": True}, _req(), OWNER, store))
    assert res2["smoke_test_passed"] is True
    assert repo.get("models", model["id"], OWNER["id"])["tested_compute_specs"]["actual_gpu"] == pinned_diag
