"""Bidirectional creation-pool reuse, 4-tier ranking, gentle replace, lifecycle, and atomic lease CAS (Spec §8.3 & §8.4)."""

from __future__ import annotations

import concurrent.futures
import copy
import time
import uuid

import pytest
from app.common import current_user, get_store
from app.config import settings
from app.db import Store
from app.gpu.compute import GpuConfig
from app.gpu.instances import (
    POOL_CREATION,
    CreationInstancePending,
    InstanceService,
)
from app.gpu.mcp import MCPGPUExecutor
from app.gpu.registry import Registry
from app.gpu.repository import Repository
from app.gpu.scheduler import Scheduler
from app.main import app
from fastapi.testclient import TestClient

T4 = {"variant": "VARIANT_GPU", "accelerator": "T4", "shape": "SHAPE_STANDARD"}
L4 = {"variant": "VARIANT_GPU", "accelerator": "L4", "shape": "SHAPE_STANDARD"}
A100 = {"variant": "VARIANT_GPU", "accelerator": "A100", "shape": "SHAPE_STANDARD"}


def _seed_env(tmp_path, *, wf_spec=L4, two_accounts=False):
    store = Store(f"sqlite:///{tmp_path / 'reuse.sqlite'}")
    repo = Repository(store)
    owner = "owner"
    proj = store.create("projects", owner, {"title": "Creation Reuse"})
    aid1 = str(uuid.uuid4())
    repo.create(
        "accounts",
        owner,
        {
            "account_id": aid1,
            "alias": "acct-1",
            "provider": "colab",
            "enabled": True,
            "auth_status": "connected",
            "allow_allocation": True,
            "budget": {"daily": 100.0, "monthly": 500.0},
        },
        record_id=aid1,
    )
    aid2 = None
    if two_accounts:
        aid2 = str(uuid.uuid4())
        repo.create(
            "accounts",
            owner,
            {
                "account_id": aid2,
                "alias": "acct-2",
                "provider": "colab",
                "enabled": True,
                "auth_status": "connected",
                "allow_allocation": True,
                "budget": {"daily": 100.0, "monthly": 500.0},
            },
            record_id=aid2,
        )
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
            "sha256": "wf-sha-1",
            "manifest": manifest,
            "api_graph": graph,
            "object_info": info,
        },
    )
    return store, repo, owner, proj, aid1, aid2, wf, graph


def _add_creation_instance(
    repo,
    owner,
    aid,
    *,
    spec,
    state="retained_idle",
    retention_mode="smart",
    retain_idle_minutes=30,
    idle_deadline_at=2000.0,
    hard_deadline_at=5000.0,
    fingerprint=None,
    process=None,
    current_job_id=None,
):
    lease_state = "idle" if state == "retained_idle" and not current_job_id else "active"
    lease = repo.create(
        "leases",
        owner,
        {
            "state": lease_state,
            "provider": "colab",
            "account_id": aid,
            "instance_pool": POOL_CREATION,
            "owner_job_id": current_job_id or "prev-job",
            "current_job_id": current_job_id,
            "deadline_epoch": hard_deadline_at,
            "idle_deadline_at": idle_deadline_at if lease_state == "idle" else None,
            "handle": {"runtime_id": f"rt-{uuid.uuid4().hex[:8]}", "owner_id": owner, "job_id": current_job_id or "prev-job"},
        },
    )
    runtime = None
    if fingerprint and process:
        runtime = {
            "fingerprint": fingerprint,
            "process": process,
            "lease_id": lease["id"],
            "workflow_key": "demo-wf",
            "workflow_version": 1,
        }
    inst = repo.create(
        "instances",
        owner,
        {
            "instance_pool": POOL_CREATION,
            "provider": "colab",
            "account_id": aid,
            "state": state,
            "retention_mode": retention_mode,
            "retain_idle_minutes": retain_idle_minutes,
            "lease_id": lease["id"],
            "current_job_id": current_job_id,
            "runtime_spec_requested": {"key": spec},
            "runtime_spec_reported": {"key": spec},
            "gpu_reported": spec["accelerator"],
            "idle_deadline_at": idle_deadline_at if state == "retained_idle" else None,
            "hard_deadline_at": hard_deadline_at,
            **({"creation_runtime": runtime} if runtime else {}),
        },
    )
    if runtime:
        runtime["instance_id"] = inst["id"]
        repo.update("instances", inst["id"], {"creation_runtime": runtime}, owner)
    repo.update("leases", lease["id"], {"instance_id": inst["id"]}, owner)
    return inst, lease


def test_four_tier_candidate_priority_ranking(tmp_path):
    """Tier 1 (warm_hit) > Tier 2 (exact_match) > Tier 3 (higher_spec lowest rank) > Tier 4 (idle/recent)."""
    store, repo, owner, proj, aid, _, wf, _ = _seed_env(tmp_path, wf_spec=T4)
    fp = "fp-t4-workflow"
    inst_a100, _ = _add_creation_instance(repo, owner, aid, spec=A100)
    inst_l4_cold, _ = _add_creation_instance(repo, owner, aid, spec=L4)
    inst_t4_cold, _ = _add_creation_instance(repo, owner, aid, spec=T4)
    inst_l4_warm, _ = _add_creation_instance(repo, owner, aid, spec=L4, fingerprint=fp, process="4242:1")

    service = InstanceService(repo, clock=lambda: 100.0, config={"unit_rate": 0.1, "daily_units": 50.0, "monthly_units": 200.0})

    def make_api_job(jid):
        return {
            "id": jid,
            "owner_id": owner,
            "project_id": proj["id"],
            "provider": "colab",
            "instance_pool": POOL_CREATION,
            "creation_fingerprint": fp,
            "requested_spec": {"key": T4},
            "snapshot": {"creation_api": True, "manifest": {"compute_spec": T4}},
            "execution_policy": {"provider": "colab", "account_id": aid, "lifecycle": "smart", "retain_idle_minutes": 15},
        }

    # 1. Warmed L4 beats matched cold T4 (Tier 1 > Tier 2)
    j1 = make_api_job("j1")
    repo.create("jobs", owner, j1, record_id="j1")
    picked1 = service.allocate(j1)
    assert picked1["id"] == inst_l4_warm["id"]

    # 2. Matched cold T4 beats unwarmed L4 and A100 (Tier 2 > Tier 3)
    j2 = make_api_job("j2")
    repo.create("jobs", owner, j2, record_id="j2")
    picked2 = service.allocate(j2)
    assert picked2["id"] == inst_t4_cold["id"]

    # 3. Unwarmed L4 beats unwarmed A100 (Tier 3: lowest higher rank first)
    j3 = make_api_job("j3")
    repo.create("jobs", owner, j3, record_id="j3")
    picked3 = service.allocate(j3)
    assert picked3["id"] == inst_l4_cold["id"]

    # 4. Finally A100 is picked
    j4 = make_api_job("j4")
    repo.create("jobs", owner, j4, record_id="j4")
    picked4 = service.allocate(j4)
    assert picked4["id"] == inst_a100["id"]


def test_creation_api_rejects_lower_spec_pool_regardless_of_auto_start(tmp_path, monkeypatch):
    """When only T4 exists and API requests L4 workflow, reject 409 CREATION_INSTANCE_SPEC_TOO_LOW (never open 2nd GPU)."""
    store, repo, owner, proj, aid, _, wf, graph = _seed_env(tmp_path, wf_spec=L4)
    future = time.time() + 3600.0
    _add_creation_instance(repo, owner, aid, spec=T4, idle_deadline_at=future, hard_deadline_at=future + 3600.0)

    monkeypatch.setattr(settings, "creation_comfy_url", "https://creation.test")
    monkeypatch.setattr(settings, "studio_api_token", "admin-test")
    monkeypatch.setenv("GPU_REAL_ENABLED", "1")
    monkeypatch.setenv("GPU_UNIT_RATE", "0.2")
    monkeypatch.setattr(app.state, "store", store, raising=False)
    monkeypatch.setitem(app.dependency_overrides, current_user, lambda: {"id": owner, "role": "admin"})
    monkeypatch.setitem(app.dependency_overrides, get_store, lambda: store)
    monkeypatch.setattr("app.admission._google_status", lambda *args, **kwargs: {"colab": {"connected": True}})
    monkeypatch.setattr("app.creation_api._google_status", lambda *args, **kwargs: {"colab": {"connected": True}})

    with TestClient(app) as client:
        admin_auth = {"Authorization": "Bearer admin-test"}
        key = client.post("/api/gpu/creation-api/keys", headers=admin_auth, json={"name": "Tavern", "workflow_ids": [wf["id"]]}).json()["data"]
        cfg = client.get("/api/gpu/creation-api", headers=admin_auth).json()["data"]["config"]
        api_headers = {"Authorization": f"Bearer {key['token']}", "Host": "creation.test"}

        for auto_start in (False, True):
            assert client.put("/api/gpu/creation-api/config", headers=admin_auth, json={**cfg, "auto_start": auto_start}).status_code == 200
            resp = client.post("/prompt", headers=api_headers, json={"prompt": copy.deepcopy(graph)})
            assert resp.status_code == 409, resp.text
            assert resp.json()["error"]["type"] == "CREATION_INSTANCE_SPEC_TOO_LOW"

        # Warmup endpoint also refuses to silently open a 2nd GPU when T4 is active
        warm = client.post("/api/gpu/creation-api/warmups", headers=admin_auth, json={"workflow_id": wf["id"]})
        assert warm.status_code == 409, warm.text
        assert warm.json()["error"]["code"] == "CREATION_INSTANCE_SPEC_TOO_LOW"


def test_bidirectional_frontend_and_api_reuse_plus_resident_service_and_prep_skip(tmp_path, monkeypatch):
    """Forward (API -> Frontend hot_reuse skipping prep -> API) AND reverse (Frontend running/idle -> API auto_start=False)."""
    store, repo, owner, proj, aid, _, wf, graph = _seed_env(tmp_path, wf_spec=L4)
    monkeypatch.setattr(settings, "creation_comfy_url", "https://creation.test")
    monkeypatch.setattr(settings, "studio_api_token", "admin-test")
    monkeypatch.setenv("GPU_REAL_ENABLED", "1")
    monkeypatch.setenv("GPU_UNIT_RATE", "0.2")
    monkeypatch.setattr(app.state, "store", store, raising=False)
    monkeypatch.setitem(app.dependency_overrides, current_user, lambda: {"id": owner, "role": "admin"})
    monkeypatch.setitem(app.dependency_overrides, get_store, lambda: store)
    monkeypatch.setattr("app.admission._google_status", lambda *args, **kwargs: {"colab": {"connected": True}})
    monkeypatch.setattr("app.creation_api._google_status", lambda *args, **kwargs: {"colab": {"connected": True}})

    with TestClient(app) as client:
        admin_auth = {"Authorization": "Bearer admin-test"}
        key = client.post("/api/gpu/creation-api/keys", headers=admin_auth, json={"name": "Tavern", "workflow_ids": [wf["id"]]}).json()["data"]
        api_headers = {"Authorization": f"Bearer {key['token']}", "Host": "creation.test"}

        # 1. Frontend submits smart L4 job (in-flight / queued)
        fe1 = client.post(
            "/api/gpu/jobs",
            headers={"Idempotency-Key": "fe-smart-1"},
            json={
                "project_id": proj["id"],
                "model_id": "demo-wf",
                "workflow_key": "demo-wf",
                "workflow_version": 1,
                "params": {"seed": 11},
                "execution_policy": {"provider": "colab", "account_id": aid, "lifecycle": "smart", "retain_idle_minutes": 30},
            },
        )
        assert fe1.status_code == 202, fe1.text
        fe1_id = fe1.json()["data"]["id"]
        fe1_row = repo.get("jobs", fe1_id, owner)
        assert fe1_row["instance_pool"] == POOL_CREATION
        assert fe1_row.get("creation_fingerprint")

        # 2. Reverse flow (while frontend smart job is in-flight): Creation API with auto_start=False SUCCEEDS!
        api_inflight = client.post("/prompt", headers=api_headers, json={"prompt": copy.deepcopy(graph)})
        assert api_inflight.status_code == 200, api_inflight.text
        api_inflight_id = api_inflight.json()["prompt_id"]
        assert repo.get("jobs", api_inflight_id, owner)["creation_fingerprint"] == fe1_row["creation_fingerprint"]

        # 3. Simulate fe1 finished & warmed L4 instance with matching creation_runtime
        soon = time.time() + 1800.0
        inst, lease = _add_creation_instance(
            repo,
            owner,
            aid,
            spec=L4,
            state="retained_idle",
            retention_mode="smart",
            retain_idle_minutes=30,
            idle_deadline_at=soon,
            hard_deadline_at=soon + 3600.0,
            fingerprint=fe1_row["creation_fingerprint"],
            process="7777:1",
        )
        repo.update("jobs", fe1_id, {"state": "completed", "instance_id": inst["id"]}, owner)
        repo.update("jobs", api_inflight_id, {"state": "cancelled"}, owner)

        # Admin status exposes pool_instances including frontend-started instance
        status_data = client.get("/api/gpu/creation-api", headers=admin_auth).json()["data"]
        assert any(row["id"] == inst["id"] for row in status_data["pool_instances"])

        # 4. Forward flow: frontend submits modified prompt -> reuse-preview reports reuse_direct + warm_hit,
        #    _await_prep skips CPU prep, allocate reuses same inst, and MCP runner sends resident service + None compute_spec!
        preview = client.post(
            "/api/gpu/jobs/reuse-preview",
            json={
                "workflow_key": "demo-wf",
                "workflow_version": 1,
                "project_id": proj["id"],
                "provider": "colab",
                "account_id": aid,
                "lifecycle": "smart",
                "retain_idle_minutes": 30,
            },
        ).json()["data"]
        assert preview["decision"] == "reuse_direct"
        assert preview["target_instance"]["instance_id"] == inst["id"]
        assert preview["target_instance"]["warm_hit"] is True
        assert preview["force_confirm"] is False

        fe2 = client.post(
            "/api/gpu/jobs",
            headers={"Idempotency-Key": "fe-smart-2"},
            json={
                "project_id": proj["id"],
                "model_id": "demo-wf",
                "workflow_key": "demo-wf",
                "workflow_version": 1,
                "params": {"seed": 99},
                "execution_policy": {"provider": "colab", "account_id": aid, "lifecycle": "smart", "retain_idle_minutes": 30},
            },
        )
        assert fe2.status_code == 202, fe2.text
        fe2_row = repo.get("jobs", fe2.json()["data"]["id"], owner)

        # Scheduler._await_prep must skip CPU prep because warm creation instance already has matching fingerprint!
        sched = Scheduler(repo, clock=lambda: 100.0)
        assert sched._await_prep(fe2_row) is False

        # Allocate reuses the exact same instance and atomically claims the lease
        svc = InstanceService(repo, clock=lambda: 100.0, config={"unit_rate": 0.1, "daily_units": 50.0, "monthly_units": 200.0})
        claimed = svc.allocate(fe2_row)
        assert claimed["id"] == inst["id"]
        assert claimed["selection_reason"] == "reuse_idle"
        assert repo.get("leases", lease["id"], owner)["state"] == "active"
        assert repo.get("leases", lease["id"], owner)["current_job_id"] == fe2_row["id"]

        # Verify MCPGPUExecutor passes resident service flags and strips runtime_compute_spec on reuse
        fe2_fresh = repo.get("jobs", fe2_row["id"], owner)
        payload = MCPGPUExecutor._payload(fe2_fresh, "exec-fe2", "/tmp/job")
        assert payload["instance_id"] == inst["id"]
        assert payload["use_resident_service"] is True
        assert payload["snapshot"]["runtime_compute_spec"] is None


def test_reuse_preview_all_decisions_and_dual_criteria_shortening(tmp_path, monkeypatch):
    store, repo, owner, proj, aid, _, wf, _ = _seed_env(tmp_path, wf_spec=T4)
    monkeypatch.setenv("GPU_REAL_ENABLED", "1")
    monkeypatch.setattr(app.state, "store", store, raising=False)
    monkeypatch.setitem(app.dependency_overrides, current_user, lambda: {"id": owner, "role": "admin"})
    monkeypatch.setitem(app.dependency_overrides, get_store, lambda: store)
    monkeypatch.setattr("app.admission._google_status", lambda *args, **kwargs: {"colab": {"connected": True}})

    with TestClient(app) as client:
        def preview(**overrides):
            body = {
                "workflow_key": "demo-wf",
                "workflow_version": 1,
                "project_id": proj["id"],
                "provider": "colab",
                "account_id": aid,
                "lifecycle": "smart",
                "retain_idle_minutes": 30,
                **overrides,
            }
            resp = client.post("/api/gpu/jobs/reuse-preview", json=body)
            assert resp.status_code == 200, resp.text
            return resp.json()["data"]

        # 1. Empty pool -> create_new
        assert preview()["decision"] == "create_new"

        # 2. Idle T4 with smart 30m (remaining 20m) -> smart 30m gives reuse_direct (not shortening!)
        now_ts = time.time()
        t4_inst, t4_lease = _add_creation_instance(
            repo, owner, aid, spec=T4, state="retained_idle", retention_mode="smart",
            retain_idle_minutes=30, idle_deadline_at=now_ts + 20 * 60, hard_deadline_at=now_ts + 7200,
        )
        assert preview(lifecycle="smart", retain_idle_minutes=30)["decision"] == "reuse_direct"

        # 3. Criterion A shortening: smart 30m -> save AND smart 30m -> smart 10m
        p_save = preview(lifecycle="save")
        assert p_save["decision"] == "confirm_shorten"
        assert p_save["shortens_lifetime"] is True
        assert p_save["force_confirm"] is True

        p_short = preview(lifecycle="smart", retain_idle_minutes=10)
        assert p_short["decision"] == "confirm_shorten"
        assert p_short["shortens_lifetime"] is True

        # Criterion B shortening: same retain_idle_minutes=15, but idle_deadline_at still has 45m left!
        repo.update("instances", t4_inst["id"], {"retain_idle_minutes": 15, "idle_deadline_at": now_ts + 45 * 60}, owner)
        repo.update("leases", t4_lease["id"], {"idle_deadline_at": now_ts + 45 * 60}, owner)
        p_crit_b = preview(lifecycle="smart", retain_idle_minutes=15)
        assert p_crit_b["decision"] == "confirm_shorten"
        assert p_crit_b["shortens_lifetime"] is True
        repo.update("instances", t4_inst["id"], {"retain_idle_minutes": 30, "idle_deadline_at": now_ts + 20 * 60}, owner)
        repo.update("leases", t4_lease["id"], {"idle_deadline_at": now_ts + 20 * 60}, owner)

        # 4. Requesting L4 when only T4 exists -> confirm_lower_spec (2 actions: replace, new)
        p_low = preview(compute_spec=L4)
        assert p_low["decision"] == "confirm_lower_spec"
        assert p_low["allowed_actions"] == ["replace", "new"]
        assert p_low["force_confirm"] is True

        # 5. Same spec busy -> confirm_busy_same_spec (2 actions: reuse, new)
        repo.update("instances", t4_inst["id"], {"state": "busy", "current_job_id": "busy-j1"}, owner)
        p_busy = preview(lifecycle="smart", retain_idle_minutes=30)
        assert p_busy["decision"] == "confirm_busy_same_spec"
        assert p_busy["allowed_actions"] == ["reuse", "new"]
        assert p_busy["force_confirm"] is True

        # 6. Higher spec (L4 idle) while requesting T4 + shortening policy (save) -> confirm_higher_spec + shortens_lifetime=True (ONE combined modal!)
        repo.update("instances", t4_inst["id"], {"state": "deleted", "runtime_deleted": True, "current_job_id": None}, owner)
        _add_creation_instance(
            repo, owner, aid, spec=L4, state="retained_idle", retention_mode="smart",
            retain_idle_minutes=30, idle_deadline_at=now_ts + 20 * 60, hard_deadline_at=now_ts + 7200,
        )
        p_high = preview(compute_spec=T4, lifecycle="save")
        assert p_high["decision"] == "confirm_higher_spec"
        assert p_high["allowed_actions"] == ["reuse", "replace", "new"]
        assert p_high["shortens_lifetime"] is True
        assert p_high["force_confirm"] is True


def test_frontend_reuse_action_new_and_gentle_replace(tmp_path):
    store, repo, owner, proj, aid1, aid2, wf, _ = _seed_env(tmp_path, wf_spec=L4, two_accounts=True)
    # Busy T4 on acct-1
    old_t4, _ = _add_creation_instance(
        repo, owner, aid1, spec=T4, state="busy", current_job_id="running-old",
        retention_mode="smart", retain_idle_minutes=30,
    )
    service = InstanceService(repo, clock=lambda: 100.0, config={"unit_rate": 0.1, "daily_units": 50.0, "monthly_units": 200.0})

    # 1. reuse_action="new" opens a 2nd GPU and prefers idle account (aid2) over busy aid1!
    j_new = {
        "id": "job-multi-gpu",
        "owner_id": owner,
        "project_id": proj["id"],
        "provider": "colab",
        "instance_pool": POOL_CREATION,
        "requested_spec": {"key": L4},
        "snapshot": {"manifest": {"compute_spec": L4}},
        "execution_policy": {"provider": "colab", "reuse_action": "new", "lifecycle": "smart", "retain_idle_minutes": 15},
    }
    second = service.allocate(j_new)
    assert second["id"] != old_t4["id"]
    assert second["account_id"] == aid2
    assert second["selection_reason"] == "requested_new"
    repo.update("instances", second["id"], {"state": "deleted", "runtime_deleted": True}, owner)

    # 2. reuse_action="replace": gentle stop marks manual_stop_requested=True without killing running-old;
    #    new job waits until old_t4 is deleted, then allocates fresh L4!
    j_rep = {
        "id": "job-rep",
        "owner_id": owner,
        "project_id": proj["id"],
        "provider": "colab",
        "instance_pool": POOL_CREATION,
        "requested_spec": {"key": L4},
        "wait_for_deleted_instance_id": old_t4["id"],
        "snapshot": {"manifest": {"compute_spec": L4}},
        "execution_policy": {
            "provider": "colab",
            "account_id": aid1,
            "reuse_action": "replace",
            "replace_instance_id": old_t4["id"],
            "lifecycle": "smart",
            "retain_idle_minutes": 15,
        },
    }
    with pytest.raises(CreationInstancePending):
        service.allocate(j_rep)
    old_row = repo.get("instances", old_t4["id"], owner)
    assert old_row["manual_stop_requested"] is True
    assert old_row["state"] == "busy"

    # The API must wait for the draining card even when another account is available.
    api_job = {
        "id": "api-during-replace",
        "owner_id": owner,
        "project_id": proj["id"],
        "requested_spec": {"key": L4},
        "snapshot": {"manifest": {"compute_spec": L4}, "creation_api": True},
        "execution_policy": {"provider": "colab"},
    }
    assert service.inspect_creation_api_admission(api_job, [])["status"] == "releasing"
    with pytest.raises(CreationInstancePending):
        service.allocate(api_job)

    # Running job finishes -> complete() sees manual_stop_requested=True and transitions to releasing
    done_old = service.complete({"id": "running-old", "owner_id": owner, "instance_id": old_t4["id"],
                                 "execution_policy": {"lifecycle": "smart", "retain_idle_minutes": 30}}, success=True)
    assert done_old["state"] == "releasing"
    # Still releasing -> still waiting!
    with pytest.raises(CreationInstancePending):
        service.allocate(j_rep)

    # Remote VM deleted -> new job now allocates fresh L4!
    repo.update("instances", old_t4["id"], {"state": "deleted", "runtime_deleted": True}, owner)
    replaced = service.allocate(j_rep)
    assert replaced["id"] != old_t4["id"]
    assert replaced["selection_reason"] in {"requested_account", "replace_new"}


def test_last_job_wins_and_failure_or_cancel_does_not_destroy_smart_or_retain(tmp_path):
    store, repo, owner, proj, aid, _, wf, _ = _seed_env(tmp_path, wf_spec=L4)
    now = [1000.0]
    service = InstanceService(repo, clock=lambda: now[0], config={"unit_rate": 0.1, "daily_units": 50.0, "monthly_units": 200.0})

    inst, lease = _add_creation_instance(
        repo, owner, aid, spec=L4, state="busy", current_job_id="j1",
        retention_mode="smart", retain_idle_minutes=30, hard_deadline_at=99999.0,
    )
    repo.update("leases", lease["id"], {"create_attempts": 1}, owner)

    # 1. smart 30m -> completes -> retained_idle for 30m
    res1 = service.complete(
        {"id": "j1", "owner_id": owner, "instance_id": inst["id"], "execution_policy": {"lifecycle": "smart", "retain_idle_minutes": 30}},
        success=True,
    )
    assert res1["state"] == "retained_idle"
    assert res1["idle_deadline_at"] == 1000.0 + 30 * 60
    assert res1["retain_idle_minutes"] == 30

    # 2. Next job smart 10m FAILS (success=False) -> still retained_idle with NEW 10m deadline!
    now[0] = 1200.0
    repo.update("instances", inst["id"], {"state": "busy", "current_job_id": "j2"}, owner)
    repo.update("leases", lease["id"], {"state": "active", "current_job_id": "j2"}, owner)
    res2 = service.complete(
        {"id": "j2", "owner_id": owner, "instance_id": inst["id"], "execution_policy": {"lifecycle": "smart", "retain_idle_minutes": 10}},
        success=False,
    )
    assert res2["state"] == "retained_idle"
    assert res2["idle_deadline_at"] == 1200.0 + 10 * 60
    assert res2["retain_idle_minutes"] == 10

    # 3. Next job retain 25m CANCELLED (success=False) -> still retained_idle with 25m deadline!
    now[0] = 1300.0
    repo.update("instances", inst["id"], {"state": "busy", "current_job_id": "j3"}, owner)
    repo.update("leases", lease["id"], {"state": "active", "current_job_id": "j3"}, owner)
    res3 = service.complete(
        {"id": "j3", "owner_id": owner, "instance_id": inst["id"], "execution_policy": {"lifecycle": "retain", "retain_idle_minutes": 25}},
        success=False,
    )
    assert res3["state"] == "retained_idle"
    assert res3["retention_mode"] == "retain"
    assert res3["idle_deadline_at"] == 1300.0 + 25 * 60

    # 4. Next job save succeeds -> releases!
    now[0] = 1400.0
    repo.update("instances", inst["id"], {"state": "busy", "current_job_id": "j4"}, owner)
    repo.update("leases", lease["id"], {"state": "active", "current_job_id": "j4"}, owner)
    res4 = service.complete(
        {"id": "j4", "owner_id": owner, "instance_id": inst["id"], "execution_policy": {"lifecycle": "save"}},
        success=True,
    )
    assert res4["state"] == "releasing"


@pytest.mark.parametrize("interrupt_confirmed", [True, False])
def test_scheduler_reuses_cancelled_creation_job_vm_only_after_confirmed_interrupt(tmp_path, interrupt_confirmed):
    store, repo, owner, proj, aid, _, wf, _ = _seed_env(tmp_path, wf_spec=L4)
    inst, lease = _add_creation_instance(
        repo, owner, aid, spec=L4, state="busy", current_job_id="j-cancel",
        retention_mode="smart", retain_idle_minutes=20, hard_deadline_at=99999.0,
    )
    repo.update("leases", lease["id"], {"create_attempts": 1}, owner)
    repo.create("executions", owner, {"job_id": "j-cancel", "state": "running"}, record_id="exec-cancel")
    repo.create(
        "jobs",
        owner,
        {
            "state": "cleaning_up",
            "cleanup_final_state": "cancelled",
            "provider": "colab",
            "instance_pool": POOL_CREATION,
            "account_id": aid,
            "instance_id": inst["id"],
            "lease_id": lease["id"],
            "execution_id": "exec-cancel",
            "execution_policy": {"provider": "colab", "lifecycle": "smart", "retain_idle_minutes": 20},
        },
        record_id="j-cancel",
    )

    deleted = []
    interrupted = []

    class FakeProvider:
        def inspect(self, handle):
            return {"ready": True, "deleted": False}

        def delete(self, handle):
            deleted.append(handle)
            return True

    class FakeExecutor:
        def interrupt_creation_service(self, handle, job):
            interrupted.append(job["id"])
            return interrupt_confirmed

    sched = Scheduler(
        repo,
        provider_factory=lambda _: FakeProvider(),
        executor_factory=lambda _: FakeExecutor(),
        clock=lambda: 2000.0,
        config=GpuConfig(unit_rate=0.1),
    )

    result = sched.tick()
    assert result["state"] == "cancelled"
    assert interrupted == ["j-cancel"]
    assert len(deleted) == (0 if interrupt_confirmed else 1)
    assert repo.get("instances", inst["id"], owner)["state"] == ("retained_idle" if interrupt_confirmed else "deleted")
    assert repo.get("leases", lease["id"], owner)["state"] == ("idle" if interrupt_confirmed else "deleted")
    assert repo.get("executions", "exec-cancel", owner)["state"] == "cancelled"


def test_atomic_allocate_lease_cas_concurrency_and_fail_closed_on_missing_lease(tmp_path):
    store, repo, owner, proj, aid, _, wf, _ = _seed_env(tmp_path, wf_spec=L4)
    inst, lease = _add_creation_instance(repo, owner, aid, spec=L4, state="retained_idle", idle_deadline_at=5000.0)
    service = InstanceService(repo, clock=lambda: 100.0, config={"unit_rate": 0.1, "daily_units": 50.0, "monthly_units": 200.0})

    def make_req(jid):
        row = {
            "id": jid,
            "owner_id": owner,
            "project_id": proj["id"],
            "provider": "colab",
            "instance_pool": POOL_CREATION,
            "requested_instance_id": inst["id"],
            "requested_spec": {"key": L4},
            "snapshot": {"manifest": {"compute_spec": L4}},
            "execution_policy": {"provider": "colab", "account_id": aid, "instance_id": inst["id"], "lifecycle": "smart"},
        }
        repo.create("jobs", owner, row, record_id=jid)
        return row

    jobs = [make_req("race-1"), make_req("race-2")]

    def try_alloc(j):
        try:
            return ("ok", service.allocate(j)["id"])
        except Exception as exc:
            return ("err", type(exc).__name__)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(try_alloc, jobs))

    oks = [r for r in results if r[0] == "ok"]
    errs = [r for r in results if r[0] == "err"]
    assert len(oks) == 1 and oks[0][1] == inst["id"]
    assert len(errs) == 1
    assert len(repo.list("instances", owner)) == 1  # Never silently created a 2nd VM!

    # Scheduler._advance fail-closed: reuse_idle without lease MUST fail instead of calling provider.create
    orphan = repo.create(
        "jobs",
        owner,
        {
            "state": "allocating",
            "provider": "colab",
            "instance_pool": POOL_CREATION,
            "account_id": aid,
            "instance_id": inst["id"],
            "selection_reason": "reuse_idle",
            "lease_id": None,
            "snapshot": {"manifest": {"compute_spec": L4}},
            "execution_policy": {"provider": "colab", "account_id": aid},
        },
        record_id="orphan-reuse",
    )

    class ForbiddenCreateProvider:
        def create(self, *args, **kwargs):
            pytest.fail("reuse_idle without lease must NEVER call provider.create")

    sched = Scheduler(repo, provider_factory=lambda _: ForbiddenCreateProvider(), clock=lambda: 100.0)
    step1 = sched._advance(orphan)
    assert step1["state"] in {"cleaning_up", "failed"}
    assert step1["error_code"] == "creation_lease_unavailable"
    assert len(repo.list("instances", owner)) == 1  # Never created a 2nd VM!
