import pytest
from app.common import current_user
from app.config import settings
from app.db import Store
from app.gpu.repository import Repository
from app.main import app
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    app.state.store = Store("sqlite:///" + str(tmp_path / "fixed-prompts.db"))
    app.dependency_overrides[current_user] = lambda: {"id": "owner", "role": "admin"}
    with TestClient(app) as session:
        yield session
    app.dependency_overrides.clear()


def _seed_workflow(repo: Repository, *, owner="owner", wf_id="wf-1", key="test-wf", version=1, model_id="m1"):
    if not any(m.get("model_id") == model_id for m in repo.list("models", owner)):
        repo.create(
            "models",
            owner,
            {"model_id": model_id, "display_name": "Mock Model", "enabled": True, "is_mock": True},
        )
    return repo.create(
        "workflows",
        owner,
        {
            "workflow_key": key,
            "workflow_version": version,
            "model_id": model_id,
            "state": "published",
            "active": True,
            "sha256": "0" * 64,
            "manifest": {
                "parameters": {
                    "prompt": {
                        "type": "string",
                        "required": True,
                        "binding": {"node": "1", "input": "text"},
                    },
                    "negative_prompt": {
                        "type": "string",
                        "default": "",
                        "binding": {"node": "2", "input": "text"},
                    },
                    "width": {
                        "type": "integer",
                        "default": 768,
                        "min": 256,
                        "max": 2048,
                        "binding": {"node": "3", "input": "width"},
                    },
                    "seed": {
                        "type": "integer",
                        "default": -1,
                        "min": 0,
                        "max": 2_000_000_000,
                        "binding": {"node": "4", "input": "seed"},
                    },
                }
            },
            "api_graph": {
                "1": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
                "2": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
                "3": {"class_type": "EmptyLatentImage", "inputs": {"width": 768, "height": 768}},
                "4": {"class_type": "KSampler", "inputs": {"seed": 1}},
            },
            "object_info": {
                "CLIPTextEncode": {"input": {"required": {"text": ["STRING", {}]}}, "output": ["CONDITIONING"]},
                "EmptyLatentImage": {
                    "input": {"required": {"width": ["INT", {}], "height": ["INT", {}]}},
                    "output": ["LATENT"],
                },
                "KSampler": {"input": {"required": {"seed": ["INT", {}]}}, "output": ["LATENT"]},
            },
        },
        record_id=wf_id,
    )


def test_fixed_prompts_crud_scoping_and_validation(client):
    p1 = client.post("/api/gpu/projects", json={"title": "Proj 1"}).json()["data"]
    p2 = client.post("/api/gpu/projects", json={"title": "Proj 2"}).json()["data"]
    repo = Repository(app.state.store)
    wf1 = _seed_workflow(repo, wf_id="wf-1", key="wf-a", version=1)
    wf2 = _seed_workflow(repo, wf_id="wf-2", key="wf-a", version=2)

    # Default shape
    res = client.get(f"/api/gpu/projects/{p1['id']}/workflows/{wf1['id']}/fixed-prompts")
    assert res.status_code == 200
    data = res.json()["data"]
    assert data["positive"] == {"enabled": True, "text": "", "position": "prefix", "separator": "comma"}
    assert data["negative"] == {"enabled": True, "text": "", "position": "prefix", "separator": "comma"}

    # Save on (p1, wf1)
    payload = {
        "positive": {"enabled": True, "text": "masterpiece, lora_trigger", "position": "prefix", "separator": "comma"},
        "negative": {"enabled": True, "text": "lowres, blurry", "position": "suffix", "separator": "newline"},
    }
    saved = client.put(f"/api/gpu/projects/{p1['id']}/workflows/{wf1['id']}/fixed-prompts", json=payload)
    assert saved.status_code == 200
    assert saved.json()["data"]["positive"] == payload["positive"]
    assert saved.json()["data"]["negative"] == payload["negative"]

    # Read back on (p1, wf1)
    fetched = client.get(f"/api/gpu/projects/{p1['id']}/workflows/{wf1['id']}/fixed-prompts").json()["data"]
    assert fetched["positive"]["text"] == "masterpiece, lora_trigger"
    assert fetched["negative"]["position"] == "suffix"
    assert fetched["negative"]["separator"] == "newline"

    # Scoped strictly by (project_id, workflow_id): p2/wf1 and p1/wf2 stay isolated
    assert client.get(f"/api/gpu/projects/{p2['id']}/workflows/{wf1['id']}/fixed-prompts").json()["data"]["positive"]["text"] == ""
    assert client.get(f"/api/gpu/projects/{p1['id']}/workflows/{wf2['id']}/fixed-prompts").json()["data"]["positive"]["text"] == ""

    # Validation rejects bad separator / position / extra fields
    bad_sep = client.put(
        f"/api/gpu/projects/{p1['id']}/workflows/{wf1['id']}/fixed-prompts",
        json={"positive": {"separator": "tab"}},
    )
    assert bad_sep.status_code == 422

    bad_pos = client.put(
        f"/api/gpu/projects/{p1['id']}/workflows/{wf1['id']}/fixed-prompts",
        json={"negative": {"position": "middle"}},
    )
    assert bad_pos.status_code == 422

    bad_extra = client.put(
        f"/api/gpu/projects/{p1['id']}/workflows/{wf1['id']}/fixed-prompts",
        json={"unknown": {}},
    )
    assert bad_extra.status_code == 422

    # Ownership check blocks other users
    app.dependency_overrides[current_user] = lambda: {"id": "other-user", "role": "user"}
    denied = client.get(f"/api/gpu/projects/{p1['id']}/workflows/{wf1['id']}/fixed-prompts")
    assert denied.status_code == 404


def test_presets_support_partial_overrides_and_fixed_prompts(client):
    repo = Repository(app.state.store)
    wf = _seed_workflow(repo, wf_id="wf-preset")

    # Save preset omitting required `prompt` while including `fixed_prompts` and `width`
    fixed = {
        "positive": {"enabled": True, "text": "anime_style", "position": "prefix", "separator": "comma"},
        "negative": {"enabled": True, "text": "bad hands", "position": "suffix", "separator": "space"},
    }
    created_resp = client.post(
        "/api/gpu/presets",
        json={
            "name": "Anime Style",
            "workflow_version_id": wf["id"],
            "overrides": {"width": 1024},
            "fixed_prompts": fixed,
        },
    )
    assert created_resp.status_code == 201
    created = created_resp.json()["data"]
    assert created["overrides"] == {"width": 1024}
    assert created["fixed_prompts"] == fixed

    # Patch updates fixed_prompts and partial overrides
    updated_resp = client.patch(
        f"/api/gpu/presets/{created['id']}",
        json={
            "overrides": {"width": 896},
            "fixed_prompts": {
                **fixed,
                "positive": {**fixed["positive"], "text": "anime_style_v2"},
            },
        },
    )
    assert updated_resp.status_code == 200
    updated = updated_resp.json()["data"]
    assert updated["overrides"] == {"width": 896}
    assert updated["fixed_prompts"]["positive"]["text"] == "anime_style_v2"

    # Invalid parameter value or unknown key still rejected
    bad_val = client.post(
        "/api/gpu/presets",
        json={"name": "Bad", "workflow_version_id": wf["id"], "overrides": {"width": 99999}},
    )
    assert bad_val.status_code == 422

    bad_key = client.post(
        "/api/gpu/presets",
        json={"name": "BadKey", "workflow_version_id": wf["id"], "overrides": {"nonexistent": 1}},
    )
    assert bad_key.status_code == 422


def test_jobs_and_batches_persist_prompt_composition_without_polluting_legacy_jobs(client):
    project = client.post("/api/gpu/projects", json={"title": "Compose"}).json()["data"]
    repo = Repository(app.state.store)
    _seed_workflow(repo, wf_id="wf-job", key="compose-wf", version=1)

    composition = {
        "positive": {
            "param_key": "prompt",
            "main": "a girl under cherry blossoms",
            "fixed": "lora_trigger",
            "enabled": True,
            "position": "prefix",
            "separator": "comma",
        },
        "negative": {
            "param_key": "negative_prompt",
            "main": "watermark",
            "fixed": "lowres",
            "enabled": True,
            "position": "suffix",
            "separator": "comma",
        },
    }

    # Single job with prompt_composition
    single_resp = client.post(
        "/api/gpu/jobs",
        headers={"Idempotency-Key": "job-comp-1"},
        json={
            "project_id": project["id"],
            "model_id": "m1",
            "workflow_key": "compose-wf",
            "workflow_version": 1,
            "params": {
                "prompt": "lora_trigger, a girl under cherry blossoms",
                "negative_prompt": "watermark, lowres",
                "width": 768,
            },
            "prompt_composition": composition,
            "execution_policy": {"provider": "mock", "max_runtime_minutes": 10},
        },
    )
    assert single_resp.status_code == 202
    single = single_resp.json()["data"]
    assert single["resolved_params"]["prompt"] == "lora_trigger, a girl under cherry blossoms"
    assert single["prompt_composition"] == composition

    detail = client.get(f"/api/gpu/jobs/{single['id']}").json()["data"]
    assert detail["prompt_composition"] == composition

    # Legacy job without prompt_composition omits the field cleanly
    legacy_resp = client.post(
        "/api/gpu/jobs",
        headers={"Idempotency-Key": "job-legacy-1"},
        json={
            "project_id": project["id"],
            "model_id": "m1",
            "workflow_key": "compose-wf",
            "workflow_version": 1,
            "params": {"prompt": "plain prompt", "width": 768},
            "execution_policy": {"provider": "mock", "max_runtime_minutes": 10},
        },
    )
    assert legacy_resp.status_code == 202
    assert "prompt_composition" not in legacy_resp.json()["data"]

    # Batch jobs preserve prompt_composition
    batch_resp = client.post(
        "/api/gpu/batches",
        headers={"Idempotency-Key": "batch-comp-1"},
        json={
            "mode": "save",
            "jobs": [
                {
                    "project_id": project["id"],
                    "model_id": "m1",
                    "workflow_key": "compose-wf",
                    "workflow_version": 1,
                    "params": {
                        "prompt": "lora_trigger, a girl under cherry blossoms",
                        "negative_prompt": "watermark, lowres",
                        "width": 768,
                        "seed": 100 + idx,
                    },
                    "prompt_composition": composition,
                    "execution_policy": {"provider": "mock", "max_runtime_minutes": 10},
                }
                for idx in range(2)
            ],
        },
    )
    assert batch_resp.status_code == 202
    batch_id = batch_resp.json()["data"]["id"]
    batch_status = client.get(f"/api/gpu/batches/{batch_id}").json()["data"]
    assert len(batch_status["jobs"]) == 2
    for job in batch_status["jobs"]:
        assert job["prompt_composition"] == composition
        assert job["resolved_params"]["prompt"] == "lora_trigger, a girl under cherry blossoms"


def test_model_library_patch_trigger_words(client):
    repo = Repository(app.state.store)
    model = repo.create(
        "model_library",
        "owner",
        {
            "display_name": "My Custom LoRA",
            "filename": "custom_lora.safetensors",
            "model_type": "loras",
            "relative_path": "custom_lora.safetensors",
            "architecture": "sdxl",
            "source": "drive",
            "state": "ready",
            "trigger_words": [],
            "tags": [],
        },
    )

    resp = client.patch(
        f"/api/gpu/model-library/{model['id']}",
        json={"trigger_words": [" camera motion ", "Anime-Motion", "camera motion", ""]},
    )
    assert resp.status_code == 200
    assert resp.json()["data"]["trigger_words"] == ["camera motion", "Anime-Motion"]

    bad = client.patch(
        f"/api/gpu/model-library/{model['id']}",
        json={"trigger_words": "not-a-list"},
    )
    assert bad.status_code == 422
