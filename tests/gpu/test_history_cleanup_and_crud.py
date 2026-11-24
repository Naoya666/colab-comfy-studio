import pytest
from app.common import current_user
from app.config import settings
from app.db import Store
from app.gpu.repository import Repository
from app.main import app
from app.storage import save_asset
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    app.state.store = Store("sqlite:///" + str(tmp_path / "ux.db"))
    app.dependency_overrides[current_user] = lambda: {"id": "owner", "role": "admin"}
    with TestClient(app) as session:
        yield session
    app.dependency_overrides.clear()


def test_job_history_clear_scopes_and_preserves_active_and_detail(client):
    project = client.post("/api/gpu/projects", json={"title": "Demo"}).json()["data"]
    pid = project["id"]
    repo = Repository(app.state.store)
    repo.create("jobs", "owner", {"state": "running", "provider": "colab"}, project_id=pid, record_id="j-run")
    repo.create("jobs", "owner", {"state": "failed", "provider": "colab"}, project_id=pid, record_id="j-fail")
    repo.create("jobs", "owner", {"state": "completed", "provider": "mock"}, project_id=pid, record_id="j-mock")
    repo.create("jobs", "owner", {"state": "completed", "provider": "colab"}, project_id=pid, record_id="j-done")

    # Clear only failed/cancelled
    res = client.post("/api/gpu/jobs/history/clear", json={"project_id": pid, "scope": "failed_cancelled"})
    assert res.status_code == 200
    assert res.json()["data"] == {"cleared": 1}
    ids = {j["id"] for j in client.get(f"/api/gpu/jobs?project_id={pid}").json()["data"]}
    assert ids == {"j-run", "j-mock", "j-done"}

    # Detail & include_cleared still return cleared jobs
    assert client.get("/api/gpu/jobs/j-fail").status_code == 200
    all_ids = {j["id"] for j in client.get(f"/api/gpu/jobs?project_id={pid}&include_cleared=true").json()["data"]}
    assert "j-fail" in all_ids

    # Clear mock
    res = client.post("/api/gpu/jobs/history/clear", json={"project_id": pid, "scope": "mock"})
    assert res.json()["data"] == {"cleared": 1}
    ids = {j["id"] for j in client.get(f"/api/gpu/jobs?project_id={pid}").json()["data"]}
    assert ids == {"j-run", "j-done"}

    # Clear all terminal leaves active running job intact
    res = client.post("/api/gpu/jobs/history/clear", json={"project_id": pid, "scope": "all_terminal"})
    assert res.json()["data"] == {"cleared": 1}
    ids = {j["id"] for j in client.get(f"/api/gpu/jobs?project_id={pid}").json()["data"]}
    assert ids == {"j-run"}


def test_project_archive_delete_blocks_when_active_jobs_exist(client):
    project = client.post("/api/gpu/projects", json={"title": "Temp"}).json()["data"]
    pid = project["id"]
    repo = Repository(app.state.store)
    repo.create("jobs", "owner", {"state": "running", "provider": "mock"}, project_id=pid, record_id="j-active")

    blocked = client.delete(f"/api/gpu/projects/{pid}")
    assert blocked.status_code == 409

    repo.update("jobs", "j-active", {"state": "completed"}, "owner")
    deleted = client.delete(f"/api/gpu/projects/{pid}")
    assert deleted.status_code == 200
    assert deleted.json()["data"]["deleted"] is True

    remaining = [p["id"] for p in client.get("/api/gpu/projects").json()["data"]]
    assert pid not in remaining


def test_project_rename_preserves_atelier_binding(client):
    project = client.post(
        "/api/gpu/projects", json={"title": "Before", "atelier_project_id": "atelier-project"}
    ).json()["data"]
    path = f"/api/gpu/projects/{project['id']}"

    renamed = client.patch(path, json={"title": "After"})
    assert renamed.status_code == 200
    assert renamed.json()["data"]["title"] == "After"
    assert renamed.json()["data"]["atelier_project_id"] == "atelier-project"

    cleared = client.patch(path, json={"title": "After", "atelier_project_id": ""})
    assert cleared.status_code == 200
    assert cleared.json()["data"]["atelier_project_id"] == ""


def test_preset_update_and_delete(client):
    repo = Repository(app.state.store)
    wf = repo.create(
        "workflows",
        "owner",
        {
            "workflow_key": "test-wf",
            "workflow_version": 1,
            "model_id": "m1",
            "state": "published",
            "manifest": {"parameters": {"prompt": {"type": "string", "default": ""}}},
        },
        record_id="wf-1",
    )
    created = client.post(
        "/api/gpu/presets",
        json={"name": "P1", "workflow_version_id": wf["id"], "overrides": {"prompt": "hi"}},
    ).json()["data"]

    updated = client.patch(
        f"/api/gpu/presets/{created['id']}",
        json={"name": "P1-v2", "overrides": {"prompt": "hello"}},
    )
    assert updated.status_code == 200
    assert updated.json()["data"]["name"] == "P1-v2"
    assert updated.json()["data"]["overrides"] == {"prompt": "hello"}

    removed = client.delete(f"/api/gpu/presets/{created['id']}")
    assert removed.status_code == 200
    assert all(p["id"] != created["id"] for p in client.get("/api/gpu/presets").json()["data"])


def test_gallery_batch_enrichment_and_batch_delete_assets(client):
    project = client.post("/api/gpu/projects", json={"title": "Gal"}).json()["data"]
    pid = project["id"]
    repo = Repository(app.state.store)
    repo.create(
        "jobs",
        "owner",
        {"state": "completed", "batch_id": "batch-xyz", "workflow_key": "flux-dev", "provider": "colab"},
        project_id=pid,
        record_id="job-1",
    )
    # 1x1 valid PNG
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (4, 4), color=(10, 20, 30)).save(buf, format="PNG")
    png = buf.getvalue()
    a1 = save_asset(app.state.store, "owner", pid, png, "one.png", source_job_id="job-1")
    a2 = save_asset(app.state.store, "owner", pid, png, "two.png", source_job_id="job-1")

    gallery = client.get(f"/api/gpu/gallery?project_id={pid}").json()["data"]
    assert len(gallery) == 2
    assert all(item["batch_id"] == "batch-xyz" for item in gallery)
    assert all(item["workflow_key"] == "flux-dev" for item in gallery)

    res = client.post("/api/assets/batch-delete", json={"asset_ids": [a1["id"], a2["id"]]})
    assert res.status_code == 200
    assert res.json()["data"]["deleted"] == 2
    assert client.get(f"/api/gpu/gallery?project_id={pid}").json()["data"] == []


def test_creation_api_warmups_clear(client):
    repo = Repository(app.state.store)
    repo.create("jobs", "owner", {"state": "failed", "warmup": True}, record_id="w-fail")
    repo.create("jobs", "owner", {"state": "running", "warmup": True}, record_id="w-run")

    before = client.get("/api/gpu/creation-api").json()["data"]["warmups"]
    assert {w["id"] for w in before} == {"w-fail", "w-run"}

    cleared = client.post("/api/gpu/creation-api/warmups/clear")
    assert cleared.status_code == 200
    assert cleared.json()["data"] == {"cleared": 1}

    after = client.get("/api/gpu/creation-api").json()["data"]["warmups"]
    assert {w["id"] for w in after} == {"w-run"}
