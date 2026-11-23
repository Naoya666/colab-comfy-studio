"""Tests for project management, default project, system project hiding,
creation API target project binding, cross-project reference assets, and asset moving.
"""

from __future__ import annotations

import io
import time

import pytest
from app import creation_api
from app.common import current_user
from app.config import settings
from app.db import Store
from app.gpu.inputs import resolve_input_images
from app.gpu.repository import Repository
from app.main import app
from app.storage import save_asset
from fastapi.testclient import TestClient
from PIL import Image


def _png_bytes(color=(12, 34, 56)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color=color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    app.state.store = Store("sqlite:///" + str(tmp_path / "projects.db"))
    app.dependency_overrides[current_user] = lambda: {"id": "owner", "role": "admin"}
    with TestClient(app) as session:
        yield session
    app.dependency_overrides.clear()


def _seed_published_workflow(store: Store) -> dict:
    repo = Repository(store)
    repo.create(
        "models",
        "owner",
        {"model_id": "real-m1", "display_name": "Real Model", "enabled": True, "is_mock": False},
        record_id="model-real-m1",
    )
    api_graph = {
        "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "hello"}},
        "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}},
    }
    object_info = {
        "CLIPTextEncode": {"input": {"required": {"text": ["STRING", {"default": "hello"}]}}},
        "SaveImage": {"input": {"required": {"images": ["IMAGE"]}}},
    }
    manifest = {
        "workflow_key": "wf-proj-test",
        "workflow_version": 1,
        "model_id": "real-m1",
        "compute_spec": {"variant": "VARIANT_GPU", "accelerator": "L4", "shape": "SHAPE_STANDARD"},
        "parameters": {
            "prompt": {"type": "string", "default": "hello", "binding": {"node": "1", "input": "text"}},
        },
        "dependencies": {
            "comfyui_commit": "423cbd20511becb1f16122ab518e92e58bfa03d9",
            "custom_nodes": [],
            "weight_files": [],
        },
    }
    return repo.create(
        "workflows",
        "owner",
        {
            "workflow_key": "wf-proj-test",
            "workflow_version": 1,
            "model_id": "real-m1",
            "state": "published",
            "active": True,
            "review_approved": True,
            "smoke_test_passed": True,
            "smoke_test_kind": "real",
            "manifest": manifest,
            "api_graph": api_graph,
            "object_info": object_info,
            "sha256": "a" * 64,
        },
        record_id="wf-proj-test-1",
    )


def test_projects_hide_system_and_seed_and_manage_default(client):
    store = app.state.store
    store.create("projects", "owner", {"title": "原生 ComfyUI", "system": True}, record_id="native-comfy-owner")
    store.create("projects", "owner", {"title": "GPU 可行性检查", "system": True}, record_id="system-smoke-owner")

    # Initial fetch hides system projects and seeds one default project.
    initial = client.get("/api/gpu/projects").json()["data"]
    assert len(initial) == 1
    default_proj = initial[0]
    assert default_proj["title"] == "默认项目"
    assert default_proj["is_default"] is True

    # include_system=true still reveals system projects for diagnostics.
    with_system = client.get("/api/gpu/projects?include_system=true").json()["data"]
    assert {"native-comfy-owner", "system-smoke-owner", default_proj["id"]}.issubset({p["id"] for p in with_system})

    # Deleting a system project is rejected.
    sys_del = client.delete("/api/gpu/projects/native-comfy-owner")
    assert sys_del.status_code == 409
    assert sys_del.json()["code"] == "PROJECT_SYSTEM_PROTECTED"

    # Creating a second project keeps the default project sorted first.
    second = client.post("/api/gpu/projects", json={"title": "角色 A"}).json()["data"]
    listed = client.get("/api/gpu/projects").json()["data"]
    assert [p["id"] for p in listed] == [default_proj["id"], second["id"]]

    # Cannot delete the default project while other projects exist.
    blocked_default = client.delete(f"/api/gpu/projects/{default_proj['id']}")
    assert blocked_default.status_code == 409
    assert blocked_default.json()["code"] == "PROJECT_IS_DEFAULT"

    # Switch default to second project -> now second sorts first and original can be deleted.
    promoted = client.post(f"/api/gpu/projects/{second['id']}/default")
    assert promoted.status_code == 200
    assert promoted.json()["data"]["is_default"] is True
    listed = client.get("/api/gpu/projects").json()["data"]
    assert [p["id"] for p in listed] == [second["id"], default_proj["id"]]
    assert listed[0]["is_default"] is True
    assert listed[1].get("is_default") is False

    deleted_old = client.delete(f"/api/gpu/projects/{default_proj['id']}")
    assert deleted_old.status_code == 200


def test_creation_api_project_binding_warmup_and_delete_protection(client, monkeypatch):
    store = app.state.store
    wf = _seed_published_workflow(store)
    store.create("projects", "owner", {"title": "原生 ComfyUI", "system": True}, record_id="native-comfy-owner")
    main_proj = client.post("/api/gpu/projects", json={"title": "主项目"}).json()["data"]
    other_proj = client.post("/api/gpu/projects", json={"title": "副项目"}).json()["data"]

    # Reject binding to a system project.
    status_before = client.get("/api/gpu/creation-api").json()["data"]
    cfg = status_before["config"]
    assert cfg["project_id"] is None
    assert status_before["default_project_id"] == creation_api.default_project_id("owner")

    bad_sys = client.put("/api/gpu/creation-api/config", json={**cfg, "project_id": "native-comfy-owner"})
    assert bad_sys.status_code == 409
    assert bad_sys.json()["code"] == "CREATION_PROJECT_UNAVAILABLE"

    # Bind to main_proj.
    saved = client.put("/api/gpu/creation-api/config", json={**cfg, "project_id": main_proj["id"]})
    assert saved.status_code == 200
    assert saved.json()["data"]["project_id"] == main_proj["id"]
    assert creation_api.get_project(store, "owner") == main_proj["id"]

    # Deleting a project bound to Creation API is blocked (option B).
    blocked = client.delete(f"/api/gpu/projects/{main_proj['id']}")
    assert blocked.status_code == 409
    assert blocked.json()["code"] == "PROJECT_BOUND_BY_CREATION_API"

    # Both formal creation API submission and warmup job land in the bound project.
    monkeypatch.setenv("GPU_REAL_ENABLED", "1")
    monkeypatch.setattr(creation_api, "admit_job", lambda st, user, raw, key, **kw: {
        "id": "stub-job",
        "owner_id": user["id"],
        "project_id": raw["project_id"],
        "state": "queued",
        "workflow_key": raw["workflow_key"],
        "workflow_version": raw["workflow_version"],
        "warmup": bool((kw.get("creation") or {}).get("warmup")),
    })
    warm_job = creation_api.submit(store, "owner", wf, {})
    assert warm_job["project_id"] == main_proj["id"]
    assert warm_job["warmup"] is True

    formal_job = creation_api.submit(store, "owner", wf, {"prompt": "hi"}, key={"id": "k1", "owner_id": "owner"})
    assert formal_job["project_id"] == main_proj["id"]
    assert formal_job["warmup"] is False

    # Rebind back to default Creation API project -> main_proj can now be deleted.
    rebound = client.put("/api/gpu/creation-api/config", json={**cfg, "project_id": None})
    assert rebound.status_code == 200
    assert rebound.json()["data"]["project_id"] is None
    assert creation_api.get_project(store, "owner") == creation_api.default_project_id("owner")
    assert client.delete(f"/api/gpu/projects/{main_proj['id']}").status_code == 200

    # Runtime fallback: if bound project is externally archived, get_project falls back safely.
    assert client.put("/api/gpu/creation-api/config", json={**cfg, "project_id": other_proj["id"]}).status_code == 200
    store.update("projects", other_proj["id"], {"archived_at": time.time()}, "owner")
    assert creation_api.get_project(store, "owner") == creation_api.default_project_id("owner")


def test_asset_move_preserves_cross_project_job_metadata_and_rejects_system_target(client):
    store = app.state.store
    store.create("projects", "owner", {"title": "原生 ComfyUI", "system": True}, record_id="native-comfy-owner")
    proj_a = client.post("/api/gpu/projects", json={"title": "项目 A"}).json()["data"]
    proj_b = client.post("/api/gpu/projects", json={"title": "项目 B"}).json()["data"]

    repo = Repository(store)
    repo.create(
        "jobs",
        "owner",
        {
            "state": "completed",
            "batch_id": "batch-move-1",
            "workflow_key": "flux-dev",
            "workflow_version": 2,
            "gpu_reported": "L4",
            "provider": "colab",
        },
        project_id=proj_a["id"],
        record_id="job-a1",
    )
    png = _png_bytes()
    a1 = save_asset(store, "owner", proj_a["id"], png, "one.png", source_job_id="job-a1")
    a2 = save_asset(store, "owner", proj_a["id"], png, "two.png", source_job_id="job-a1")

    # Cannot move into a system project.
    bad_move = client.post(
        "/api/assets/batch-move",
        json={"asset_ids": [a1["id"]], "target_project_id": "native-comfy-owner"},
    )
    assert bad_move.status_code == 409
    assert bad_move.json()["code"] == "PROJECT_SYSTEM_PROTECTED"

    # Batch move both assets to proj_b; source job stays in proj_a.
    moved = client.post(
        "/api/assets/batch-move",
        json={"asset_ids": [a1["id"], a2["id"]], "target_project_id": proj_b["id"]},
    )
    assert moved.status_code == 200
    assert moved.json()["data"] == {"moved": 2, "ids": [a1["id"], a2["id"]], "target_project_id": proj_b["id"]}

    # Target project gallery still enriches batch_id / workflow_key / gpu_reported from job in proj_a.
    gal_a = client.get(f"/api/gpu/gallery?project_id={proj_a['id']}").json()["data"]
    gal_b = client.get(f"/api/gpu/gallery?project_id={proj_b['id']}").json()["data"]
    gal_all = client.get("/api/gpu/gallery").json()["data"]
    assert gal_a == []
    assert len(gal_b) == 2
    assert len(gal_all) == 2
    assert all(item["batch_id"] == "batch-move-1" for item in gal_b)
    assert all(item["workflow_key"] == "flux-dev" for item in gal_b)
    assert all(item["gpu_reported"] == "L4" for item in gal_b)
    assert repo.get("jobs", "job-a1", "owner")["project_id"] == proj_a["id"]

    # Single PATCH move back to proj_a works too.
    patch_res = client.patch(f"/api/assets/{a1['id']}", json={"project_id": proj_a["id"]})
    assert patch_res.status_code == 200
    assert patch_res.json()["data"]["project_id"] == proj_a["id"]


def test_cross_project_reference_assets_allowed_and_listable(client):
    store = app.state.store
    proj_a = client.post("/api/gpu/projects", json={"title": "项目 A"}).json()["data"]
    proj_b = client.post("/api/gpu/projects", json={"title": "项目 B"}).json()["data"]
    png = _png_bytes((99, 88, 77))

    upload_a = client.post(
        "/api/assets",
        data={"project_id": proj_a["id"]},
        files={"file": ("ref-a.png", png, "image/png")},
    )
    assert upload_a.status_code == 201
    asset_a = upload_a.json()["data"]

    upload_b = client.post(
        "/api/assets",
        data={"project_id": proj_b["id"]},
        files={"file": ("ref-b.png", png, "image/png")},
    )
    assert upload_b.status_code == 201
    asset_b = upload_b.json()["data"]

    # Per-project list isolates; omitting project_id returns all projects' assets.
    only_a = client.get(f"/api/assets?project_id={proj_a['id']}").json()["data"]
    all_refs = client.get("/api/assets").json()["data"]
    assert [a["id"] for a in only_a] == [asset_a["id"]]
    assert {a["id"] for a in all_refs} == {asset_a["id"], asset_b["id"]}

    # Resolving an image input from proj_a while running a job in proj_b succeeds.
    manifest = {"parameters": {"ref_image": {"type": "image"}}}
    resolved, files = resolve_input_images(
        lambda aid: store.get("assets", aid, "owner"),
        proj_b["id"],
        manifest,
        {"ref_image": asset_a["id"]},
    )
    assert resolved["ref_image"] == "input_ref_image.jpg"
    assert len(files) == 1


def test_jobs_mark_system_flag_and_clear_all_skips_system_by_default(client):
    store = app.state.store
    proj = client.post("/api/gpu/projects", json={"title": "工作项目"}).json()["data"]
    store.create("projects", "owner", {"title": "原生 ComfyUI", "system": True}, record_id="native-comfy-owner")
    repo = Repository(store)
    repo.create(
        "jobs",
        "owner",
        {"state": "completed", "provider": "colab", "workflow_key": "flux-dev"},
        project_id=proj["id"],
        record_id="job-user",
    )
    repo.create(
        "jobs",
        "owner",
        {
            "state": "completed",
            "provider": "colab",
            "workflow_key": "system/native-comfy",
            "snapshot": {"native_comfy": True},
        },
        project_id="native-comfy-owner",
        record_id="job-native-sys",
    )

    all_jobs = {j["id"]: j for j in client.get("/api/gpu/jobs").json()["data"]}
    assert all_jobs["job-user"]["system"] is False
    assert all_jobs["job-native-sys"]["system"] is True

    # Clearing all projects without include_system clears user job only.
    cleared = client.post("/api/gpu/jobs/history/clear", json={"scope": "all_terminal"})
    assert cleared.status_code == 200
    assert cleared.json()["data"] == {"cleared": 1}
    remaining = [j["id"] for j in client.get("/api/gpu/jobs").json()["data"]]
    assert remaining == ["job-native-sys"]

    # Clearing with include_system=True clears the system job too.
    cleared_sys = client.post(
        "/api/gpu/jobs/history/clear",
        json={"scope": "all_terminal", "include_system": True},
    )
    assert cleared_sys.status_code == 200
    assert cleared_sys.json()["data"] == {"cleared": 1}
    assert client.get("/api/gpu/jobs").json()["data"] == []
