import copy
import uuid
from datetime import datetime, timezone

from app.config import settings
from app.creation_api import check
from app.db import Store
from app.gpu.registry import Registry
from app.gpu.repository import Repository
from app.main import app
from fastapi.testclient import TestClient


def publish(repo, owner, version, *, fixed="approved"):
    manifest, graph, info = Registry._template()
    manifest["compute_spec"] = {"variant": "VARIANT_GPU", "accelerator": "A100", "shape": "SHAPE_STANDARD"}
    # Identify the exposed positions from the real legacy Studio manifest.
    from app.gpu.manifest import _binding, _canonical_parameters

    definitions, _ = _canonical_parameters(manifest)
    seed_node, seed_input = _binding("seed", definitions["seed"]["binding"])
    output = graph[seed_node]
    output["inputs"]["filename_prefix"] = fixed
    info[output["class_type"]]["input"]["required"]["filename_prefix"] = ["STRING"]
    row = repo.create("workflows", owner, {
        "workflow_key": "creation-demo", "workflow_version": version, "model_id": "creation-demo",
        "state": "published", "review_approved": True, "smoke_test_kind": "real", "active": False,
        "sha256": "test", "manifest": manifest, "api_graph": graph, "object_info": info,
    })
    return row, seed_node, seed_input


def test_match_keeps_fixed_inputs_and_reports_ambiguity(tmp_path):
    repo = Repository(Store(f"sqlite:///{tmp_path / 'db'}"))
    first, node, field = publish(repo, "owner", 1)
    second, _, _ = publish(repo, "owner", 2)
    graph = copy.deepcopy(first["api_graph"])
    graph[node]["inputs"][field] = 1234
    graph[node]["_meta"] = {"title": "Display title"}
    assert check(graph, [first])["matches"][0]["params"]["seed"] == 1234
    result = check(graph, [first, second])
    assert result["status"] == "ambiguous"
    assert {item["workflow"]["id"] for item in result["matches"]} == {first["id"], second["id"]}
    graph[node]["inputs"][field] = True
    assert "seed" in check(graph, [first])["differences"][0]["reason"]
    graph[node]["inputs"][field] = 1234
    output = graph[node]
    output["inputs"]["filename_prefix"] = "other"
    assert "filename_prefix" in check(graph, [first])["differences"][0]["reason"]


def test_host_keys_saved_config_and_output_scope(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "creation_comfy_url", "https://creation.test")
    monkeypatch.setattr(settings, "studio_api_token", "admin-test")
    dburl = f"sqlite:///{tmp_path / 'db'}"
    store = Store(dburl)
    monkeypatch.setattr(app.state, "store", store, raising=False)
    repo = Repository(store)
    first, _, _ = publish(repo, "owner", 1)
    second, _, _ = publish(repo, "owner", 2, fixed="second")
    other, _, _ = publish(repo, "another-owner", 1)
    repo.create("models", "owner", {"model_id": "creation-demo", "enabled": True})
    base = "/api/gpu/creation-api"
    admin_auth = {"Authorization": "Bearer admin-test"}
    with TestClient(app) as client:
        status = client.get(base, headers=admin_auth).json()["data"]
        cfg = {**status["config"], "auto_start": True, "lifecycle": "smart", "retain_idle_minutes": 7}
        assert client.put(base + "/config", headers=admin_auth, json=cfg).json()["data"] == cfg
        bad = client.put(base + "/config", headers=admin_auth, json={**cfg, "retain_idle_minutes": 31})
        assert bad.status_code == 422
        assert Store(dburl).get("settings", "creation-api:owner", "owner")["policy"] == cfg

        def key(ids):
            return client.post(base + "/keys", headers=admin_auth, json={"name": "Tavern", "workflow_ids": ids})

        assert key([other["id"]]).status_code == 409
        row = key([first["id"], second["id"]]).json()["data"]
        other_key = key([first["id"]]).json()["data"]
        assert "token" not in client.get(base, headers=admin_auth).json()["data"]["keys"][0]
        auth = {"Authorization": "Bearer " + row["token"], "Host": "creation.test"}
        other_auth = {**auth, "Authorization": "Bearer " + other_key["token"]}
        assert client.get("/system_stats", headers={"Host": "creation.test"}).status_code == 401
        assert client.get("/system_stats", headers=auth).json()["devices"][0]["name"] == "L4"
        assert client.get(base, headers={**admin_auth, "Host": "creation.test"}).status_code == 404
        pending = repo.create("jobs", "owner", {"state": "queued", "creation_key_id": row["id"],
                                                        "creation_workflow_id": first["id"]})
        assert client.get("/history/" + pending["id"], headers=auth).json() == {}
        assert client.get("/history/" + pending["id"], headers=other_auth).status_code == 404
        assert client.post("/interrupt", headers=auth).status_code == 409
        job = repo.create("jobs", "owner", {"state": "completed", "creation_key_id": row["id"],
                                                    "creation_workflow_id": first["id"]})
        media = uuid.uuid4().hex
        assets = tmp_path / "assets"
        assets.mkdir()
        (assets / "image.png").write_bytes(b"verified-test-bytes")
        store.create("assets", "owner", {"source_job_id": job["id"], "filename": "image.png",
                                        "storage_key": "image.png", "mime": "image/png",
                                        "comfy_node": "42", "comfy_category": "images"}, record_id=media)
        repo.update("jobs", job["id"], {"output_asset_ids": [media]}, "owner")
        history = client.get("/history/" + job["id"], headers=auth).json()
        assert history[job["id"]]["outputs"] == {
            "42": {"images": [{"filename": "image.png", "subfolder": media, "type": "output"}]},
        }
        view = "/view?filename=image.png&subfolder=" + media
        assert client.get(view, headers=auth).content == b"verified-test-bytes"
        assert client.get(view, headers=other_auth).status_code == 404
        assert client.put(base + "/keys/" + row["id"], headers=admin_auth,
                          json={"name": "Tavern", "workflow_ids": [second["id"]]}).status_code == 200
        assert client.get(view, headers=auth).status_code == 404
        assert client.delete(base + "/keys/" + row["id"], headers=admin_auth).status_code == 200
        assert client.get(view, headers=auth).status_code == 401


def test_expired_lease_clears_warm_readiness(tmp_path):
    from app.creation_api import warm_status

    repo = Repository(Store(f"sqlite:///{tmp_path / 'db'}"))
    runtime = {"fingerprint": "fp", "lease_id": "lease", "process": "200:122"}
    instance = repo.create("instances", "owner", {"state": "retained_idle",
                           "lease_id": "lease", "hard_deadline_at": datetime.now(timezone.utc).timestamp() + 60})
    runtime["instance_id"] = instance["id"]
    repo.update("instances", instance["id"], {"creation_runtime": runtime}, "owner")
    job = {"state": "completed", "owner_id": "owner", "instance_id": instance["id"],
           "creation_fingerprint": "fp", "creation_runtime": runtime}
    assert warm_status(repo, job)["ready"]
    repo.update("instances", instance["id"], {"idle_deadline_at": 1}, "owner")
    assert not warm_status(repo, job)["ready"]


def test_admission_binds_published_snapshot_and_deduplicates_only_retries(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "creation_comfy_url", "https://creation.test")
    monkeypatch.setattr(settings, "studio_api_token", "admin-test")
    monkeypatch.setenv("GPU_REAL_ENABLED", "1")
    monkeypatch.setenv("GPU_UNIT_RATE", "0.2")
    store = Store(f"sqlite:///{tmp_path / 'db'}")
    monkeypatch.setattr(app.state, "store", store, raising=False)
    monkeypatch.setattr("app.admission._google_status",
                        lambda *args: {"colab": {"connected": True}})
    repo = Repository(store)
    ident = str(uuid.uuid4())
    repo.create("accounts", "owner", {"account_id": ident, "provider": "colab", "enabled": True,
                                     "auth_status": "connected", "allow_allocation": True}, record_id=ident)
    first, node, field = publish(repo, "owner", 1)
    second, _, _ = publish(repo, "owner", 2, fixed="second")
    repo.create("models", "owner", {"model_id": "creation-demo", "enabled": True})
    base = "/api/gpu/creation-api"
    admin_auth = {"Authorization": "Bearer admin-test"}
    with TestClient(app) as client:
        key = client.post(base + "/keys", headers=admin_auth, json={
            "name": "Tavern", "workflow_ids": [first["id"], second["id"]],
        }).json()["data"]
        auth = {"Authorization": "Bearer " + key["token"], "Host": "creation.test"}
        graph = copy.deepcopy(first["api_graph"])
        graph[node]["inputs"][field] = 153
        graph[node]["_meta"] = {"title": "Ignored in the running snapshot"}
        blocked = client.post("/prompt", headers=auth, json={"prompt": graph})
        assert blocked.status_code == 409, blocked.text
        assert blocked.json()["error"]["type"] == "CREATION_START_REQUIRED"
        cfg = client.get(base, headers=admin_auth).json()["data"]["config"]
        assert client.put(base + "/config", headers=admin_auth,
                          json={**cfg, "auto_start": True}).status_code == 200

        def send(prompt, headers):
            return client.post("/prompt", headers=headers, json={"prompt": prompt})

        with_key = {**auth, "Idempotency-Key": "one-request"}
        accepted = send(graph, with_key)
        assert accepted.status_code == 200, accepted.text
        ident = accepted.json()["prompt_id"]
        assert send(graph, with_key).json()["prompt_id"] == ident
        assert send(graph, auth).json()["prompt_id"] != ident
        second_job = send(second["api_graph"], auth).json()["prompt_id"]
        assert repo.get("jobs", second_job, "owner")["creation_workflow_id"] == second["id"]
        graph[node]["inputs"][field] = 245
        assert send(graph, with_key).status_code == 409
        job = repo.get("jobs", ident, "owner")
        assert job["creation_workflow_id"] == first["id"]
        assert job["snapshot"]["bound_graph"][node]["inputs"][field] == 153
        assert "_meta" not in job["snapshot"]["bound_graph"][node]
        assert job["requested_spec"]["key"]["accelerator"] == "L4"
        assert job["snapshot"]["manifest"]["compute_spec"]["accelerator"] == "A100"
        assert len([row for row in repo.list("jobs", "owner") if row.get("creation_key_id")]) == 3


def test_existing_only_admission_uses_retained_lease_with_different_gpu(tmp_path):
    from app.gpu.instances import InstanceService

    repo = Repository(Store(f"sqlite:///{tmp_path / 'db'}"))
    service = InstanceService(repo, clock=lambda: 100)
    owner, aid = "owner", str(uuid.uuid4())
    repo.create("accounts", owner, {"account_id": aid, "provider": "colab", "auth_status": "connected"},
                record_id=aid)
    lease = repo.create("leases", owner, {"state": "idle", "deadline_epoch": 200, "idle_deadline_at": 190})
    l4 = {"accelerator": "L4", "shape": "SHAPE_STANDARD", "variant": "VARIANT_GPU"}
    a100 = {**l4, "accelerator": "A100"}
    instance = repo.create("instances", owner, {
        "state": "retained_idle", "account_id": aid, "provider": "colab", "lease_id": lease["id"],
        "runtime_spec_requested": {"key": l4}, "hard_deadline_at": 200, "idle_deadline_at": 190,
    })
    job = {
        "id": str(uuid.uuid4()), "owner_id": owner, "project_id": "project",
        "requested_spec": {"key": a100}, "creation_existing_only": True, "creation_fingerprint": "approved",
        "snapshot": {"creation_api": True, "manifest": {"compute_spec": a100}},
        "execution_policy": {"provider": "colab", "account_id": aid},
    }
    assert service.creation_capacity(job, [])
    with service._tx() as session:
        row = service._find(session, "instances", instance["id"], owner)
        assert service._compatible(session, row, job, aid)
    repo.update("leases", lease["id"], {"idle_deadline_at": 99}, owner)
    assert not service.creation_capacity(job, [])


def test_batch_retains_only_for_compatible_queued_creation_request(tmp_path):
    from app.gpu.instances import InstanceService

    repo = Repository(Store(f"sqlite:///{tmp_path / 'db'}"))
    service = InstanceService(repo, clock=lambda: 100)
    owner = "owner"
    first = repo.create("jobs", owner, {"state": "running", "creation_fingerprint": "prepared"})
    instance = repo.create("instances", owner, {"state": "busy", "current_job_id": first["id"],
                                                "hard_deadline_at": 140})
    queued = repo.create("jobs", owner, {"state": "queued", "creation_fingerprint": "prepared"})
    job = {**first, "instance_id": instance["id"], "execution_policy": {"lifecycle": "batch"}}
    waiting = service.complete(job)
    assert waiting["state"] == "retained_idle"
    assert waiting["idle_deadline_at"] == 140
    repo.update("jobs", queued["id"], {"state": "cancelled"}, owner)
    assert service.complete(job)["state"] == "releasing"


def test_task_deadline_does_not_follow_retained_lease(tmp_path):
    from app.gpu.scheduler import Scheduler

    store = Store(f"sqlite:///{tmp_path / 'db'}")
    job = {"id": str(uuid.uuid4()), "owner_id": "owner", "deadline_epoch": 300, "task_deadline_epoch": 150}
    assert not Scheduler(store, clock=lambda: 149)._expired(job)
    assert Scheduler(store, clock=lambda: 151)._expired(job)
    assert Scheduler(store, clock=lambda: 301)._expired({**job, "task_deadline_epoch": 400})
