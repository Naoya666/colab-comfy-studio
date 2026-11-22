"""GPU spec order table, upstream discovery, and workflow floor checks (Spec §3 & §8.2)."""

from __future__ import annotations

import copy
import uuid

from app.common import current_user, get_store
from app.config import settings
from app.db import Store
from app.gpu.registry import Registry
from app.gpu.repository import Repository
from app.gpu.specs import (
    DEFAULT_GPU_SPEC_ORDER,
    compare_specs,
    gpu_spec_order_settings,
    is_below_workflow_default,
    record_discovered_gpu_specs,
    spec_id,
    spec_key,
)
from app.main import app
from fastapi.testclient import TestClient

T4_STD = {"variant": "VARIANT_GPU", "accelerator": "T4", "shape": "SHAPE_STANDARD"}
T4_HM = {"variant": "VARIANT_GPU", "accelerator": "T4", "shape": "SHAPE_HIGHMEM"}
L4_STD = {"variant": "VARIANT_GPU", "accelerator": "L4", "shape": "SHAPE_STANDARD"}
L4_HM = {"variant": "VARIANT_GPU", "accelerator": "L4", "shape": "SHAPE_HIGHMEM"}
A100_STD = {"variant": "VARIANT_GPU", "accelerator": "A100", "shape": "SHAPE_STANDARD"}
A100_HM = {"variant": "VARIANT_GPU", "accelerator": "A100", "shape": "SHAPE_HIGHMEM"}
G4_STD = {"variant": "VARIANT_GPU", "accelerator": "G4", "shape": "SHAPE_STANDARD"}
H100_STD = {"variant": "VARIANT_GPU", "accelerator": "H100", "shape": "SHAPE_STANDARD"}
B200_STD = {"variant": "VARIANT_GPU", "accelerator": "B200", "shape": "SHAPE_STANDARD"}


def test_default_gpu_spec_order_and_highmem_normalization():
    # L4 is high-mem-only in Colab and normalizes to SHAPE_STANDARD
    assert spec_key(L4_HM) == L4_STD
    assert spec_id(L4_HM) == "L4:SHAPE_STANDARD"
    assert compare_specs(L4_HM, L4_STD) == "equal"

    chain = [T4_STD, T4_HM, L4_STD, A100_STD, A100_HM, G4_STD, H100_STD]
    for i in range(len(chain)):
        assert compare_specs(chain[i], chain[i]) == "equal"
        for j in range(i + 1, len(chain)):
            assert compare_specs(chain[i], chain[j]) == "lower"
            assert compare_specs(chain[j], chain[i]) == "higher"

    # Unranked GPU only matches itself (strict safety)
    assert compare_specs(B200_STD, B200_STD) == "equal"
    assert compare_specs(B200_STD, L4_STD) == "unknown"
    assert compare_specs(L4_STD, B200_STD) == "unknown"

    manifest_l4 = {"compute_spec": L4_STD}
    assert is_below_workflow_default(manifest_l4, T4_STD) is True
    assert is_below_workflow_default(manifest_l4, T4_HM) is True
    assert is_below_workflow_default(manifest_l4, L4_STD) is False
    assert is_below_workflow_default(manifest_l4, A100_STD) is False


def test_gpu_order_routes_edit_reset_and_discover(tmp_path, monkeypatch):
    store = Store(f"sqlite:///{tmp_path / 'gpu_order.sqlite'}")
    repo = Repository(store)
    owner = "owner-order"
    aid = str(uuid.uuid4())
    repo.create(
        "accounts",
        owner,
        {"account_id": aid, "provider": "colab", "enabled": True, "auth_status": "connected", "allow_allocation": True},
        record_id=aid,
    )
    monkeypatch.setenv("GPU_REAL_ENABLED", "1")
    monkeypatch.setattr(app.state, "store", store, raising=False)
    monkeypatch.setitem(app.dependency_overrides, current_user, lambda: {"id": owner, "role": "admin"})
    monkeypatch.setitem(app.dependency_overrides, get_store, lambda: store)
    monkeypatch.setattr("app.compute_routes.GoogleService.status", lambda self: {"colab": {"connected": True}})
    monkeypatch.setattr(
        "app.compute_routes.GoogleColabProvider.list_specs",
        lambda self: [
            {"eligible": True, "key": T4_STD},
            {"eligible": True, "key": L4_STD},
            {"eligible": True, "key": B200_STD},
        ],
    )

    with TestClient(app) as client:
        # 1. Default order
        initial = client.get("/api/gpu/settings/gpu-order").json()["data"]
        assert initial["ordered_specs"] == DEFAULT_GPU_SPEC_ORDER
        assert initial["discovered_specs"] == []

        # 2. Upstream discovery finds B200 and leaves it unranked in discovered_specs
        disc = client.post("/api/gpu/settings/gpu-order/discover").json()["data"]
        assert disc["checked_accounts"] == 1
        assert [spec_id(item["spec"]) for item in disc["discovered_specs"]] == ["B200:SHAPE_STANDARD"]
        assert len(disc["newly_discovered"]) == 1

        # 3. Admin adds B200 at top of order and swaps L4/A100 order to test custom ranking
        custom_order = [T4_STD, L4_STD, T4_HM, A100_STD, B200_STD]
        saved = client.put("/api/gpu/settings/gpu-order", json={"ordered_specs": custom_order})
        assert saved.status_code == 200, saved.text
        saved_data = saved.json()["data"]
        assert saved_data["ordered_specs"] == custom_order
        assert saved_data["discovered_specs"] == []  # B200 was promoted into ordered_specs

        # Verify custom order is persisted and effective
        loaded = gpu_spec_order_settings(store, owner)["ordered_specs"]
        assert compare_specs(B200_STD, A100_STD, loaded) == "higher"
        assert compare_specs(T4_HM, L4_STD, loaded) == "higher"

        # 4. Reset restores defaults and moves custom B200 back to discovered_specs
        reset = client.post("/api/gpu/settings/gpu-order/reset").json()["data"]
        assert reset["ordered_specs"] == DEFAULT_GPU_SPEC_ORDER
        assert [spec_id(item["spec"]) for item in reset["discovered_specs"]] == ["B200:SHAPE_STANDARD"]

        # 5. Empty or non-GPU order rejected with 422
        assert client.put("/api/gpu/settings/gpu-order", json={"ordered_specs": []}).status_code == 422
        assert client.put(
            "/api/gpu/settings/gpu-order",
            json={"ordered_specs": [{"variant": "VARIANT_DEFAULT", "accelerator": "", "shape": "SHAPE_STANDARD"}]},
        ).status_code == 422


def test_frontend_and_creation_api_reject_spec_below_workflow_default_422(tmp_path, monkeypatch):
    store = Store(f"sqlite:///{tmp_path / 'floor.sqlite'}")
    repo = Repository(store)
    owner = "owner"
    aid = str(uuid.uuid4())
    proj = store.create("projects", owner, {"title": "Floor Test"})
    repo.create(
        "accounts",
        owner,
        {"account_id": aid, "provider": "colab", "enabled": True, "auth_status": "connected", "allow_allocation": True},
        record_id=aid,
    )
    repo.create("models", owner, {"model_id": "floor-demo", "enabled": True})
    manifest, graph, info = Registry._template()
    manifest["compute_spec"] = L4_STD
    manifest["requirements"] = {"min_vram_gb": 12}
    wf = repo.create(
        "workflows",
        owner,
        {
            "workflow_key": "floor-demo",
            "workflow_version": 1,
            "model_id": "floor-demo",
            "state": "published",
            "review_approved": True,
            "smoke_test_kind": "real",
            "active": True,
            "sha256": "floor-sha",
            "manifest": manifest,
            "api_graph": graph,
            "object_info": info,
        },
    )

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
        # 1. Frontend POST /api/gpu/jobs with T4 for L4 workflow -> 422 GPU_SPEC_BELOW_WORKFLOW_DEFAULT
        bad_job = client.post(
            "/api/gpu/jobs",
            headers={"Idempotency-Key": "floor-t4-1"},
            json={
                "project_id": proj["id"],
                "model_id": "floor-demo",
                "workflow_key": "floor-demo",
                "workflow_version": 1,
                "params": {"seed": 1},
                "execution_policy": {
                    "provider": "colab",
                    "account_id": aid,
                    "compute_spec": T4_STD,
                    "lifecycle": "smart",
                    "retain_idle_minutes": 15,
                },
            },
        )
        assert bad_job.status_code == 422, bad_job.text
        assert bad_job.json()["error"]["code"] == "GPU_SPEC_BELOW_WORKFLOW_DEFAULT"

        # 2. Frontend reuse-preview with T4 for L4 workflow -> decision == unsupported_spec
        preview = client.post(
            "/api/gpu/jobs/reuse-preview",
            json={
                "workflow_key": "floor-demo",
                "workflow_version": 1,
                "project_id": proj["id"],
                "provider": "colab",
                "account_id": aid,
                "compute_spec": T4_STD,
                "lifecycle": "smart",
                "retain_idle_minutes": 15,
            },
        )
        assert preview.status_code == 200, preview.text
        assert preview.json()["data"]["decision"] == "unsupported_spec"

        # 3. Creation API with auto_start=True and cfg.gpu=T4 for L4 workflow:
        #    When pool is empty -> 422 GPU_SPEC_BELOW_WORKFLOW_DEFAULT
        admin_auth = {"Authorization": "Bearer admin-test"}
        key = client.post(
            "/api/gpu/creation-api/keys",
            headers=admin_auth,
            json={"name": "Tavern", "workflow_ids": [wf["id"]]},
        ).json()["data"]
        cfg = client.get("/api/gpu/creation-api", headers=admin_auth).json()["data"]["config"]
        assert client.put(
            "/api/gpu/creation-api/config",
            headers=admin_auth,
            json={**cfg, "auto_start": True, "gpu": T4_STD},
        ).status_code == 200
        api_headers = {"Authorization": f"Bearer {key['token']}", "Host": "creation.test"}
        bad_api = client.post("/prompt", headers=api_headers, json={"prompt": copy.deepcopy(graph)})
        assert bad_api.status_code == 422, bad_api.text
        assert bad_api.json()["error"]["type"] == "GPU_SPEC_BELOW_WORKFLOW_DEFAULT"

        # 4. BUT if an idle L4 creation instance already exists, Creation API reuses it even though cfg.gpu=T4!
        lease = repo.create(
            "leases",
            owner,
            {"state": "idle", "provider": "colab", "account_id": aid, "deadline_epoch": 9e9, "idle_deadline_at": 9e9},
        )
        repo.create(
            "instances",
            owner,
            {
                "instance_pool": "creation",
                "provider": "colab",
                "account_id": aid,
                "state": "retained_idle",
                "lease_id": lease["id"],
                "runtime_spec_requested": {"key": L4_STD},
                "hard_deadline_at": 9e9,
                "idle_deadline_at": 9e9,
            },
        )
        ok_api = client.post("/prompt", headers=api_headers, json={"prompt": copy.deepcopy(graph)})
        assert ok_api.status_code == 200, ok_api.text
        assert ok_api.json()["prompt_id"]


def test_record_discovered_specs_ignores_ineligible_and_deduplicates(tmp_path):
    store = Store(f"sqlite:///{tmp_path / 'disc.sqlite'}")
    settings_data, newly = record_discovered_gpu_specs(
        store,
        "owner-1",
        [
            {"eligible": False, "key": B200_STD},
            {"eligible": True, "key": L4_STD},
            {"eligible": True, "key": {"variant": "VARIANT_GPU", "accelerator": "RTX6000", "shape": "SHAPE_STANDARD"}},
            {"eligible": True, "key": {"variant": "VARIANT_GPU", "accelerator": "RTX6000", "shape": "SHAPE_STANDARD"}},
        ],
        now=1000,
    )
    assert len(newly) == 1
    assert spec_id(newly[0]["spec"]) == "RTX6000:SHAPE_STANDARD"
    assert [spec_id(item["spec"]) for item in settings_data["discovered_specs"]] == ["RTX6000:SHAPE_STANDARD"]
