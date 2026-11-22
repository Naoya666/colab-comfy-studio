import copy
from concurrent.futures import ThreadPoolExecutor

import pytest
from app.db import Store
from app.gpu.registry import Registry
from app.gpu.repository import Repository, ensure_schema
from sqlalchemy import inspect


def _store(tmp_path):
    store = Store(f"sqlite:///{tmp_path / 'gpu.db'}")
    ensure_schema(store)
    return store


def test_independent_store_has_basic_tables_and_crud(tmp_path):
    store = _store(tmp_path)
    tables = set(inspect(store.engine).get_table_names())
    assert {"users", "projects", "assets", "settings", "provider_credentials"}.issubset(tables)
    row = store.create("projects", "owner", {"title": "x", "owner_id": "spoof"})
    assert row["owner_id"] == "owner"
    assert store.get("projects", row["id"], "other") is None
    assert store.update("projects", row["id"], {"title": "y"}, "owner")["title"] == "y"
    assert store.delete("projects", row["id"], "owner") is True


def test_gpu_repository_uses_isolated_tables_and_acl(tmp_path):
    store = _store(tmp_path)
    tables = set(inspect(store.engine).get_table_names())
    assert {"gpu_models", "gpu_workflows", "gpu_jobs", "gpu_events"}.issubset(tables)
    repo = Repository(store)
    row = repo.create("models", "owner", {"name": "demo", "enabled": True})
    assert repo.get("models", row["id"], "owner")["name"] == "demo"
    assert repo.get("models", row["id"], "other") is None
    assert repo.update("models", row["id"], {"enabled": False}, "owner")["enabled"] is False


def test_create_job_is_owner_scoped_and_rejects_different_raw_body(tmp_path):
    repo = Repository(_store(tmp_path))
    body = {"workflow_key": "demo", "workflow_version": 1, "params": {"seed": -1}}
    first, created = repo.create_job("owner", body, "same-key")
    assert created is True
    second, created = repo.create_job("owner", body, "same-key")
    assert not created and first["id"] == second["id"]
    assert repo.create_job("other", {**body, "params": {"seed": 2}}, "same-key")[1] is True
    try:
        repo.create_job("owner", {**body, "params": {"seed": 3}}, "same-key")
    except ValueError:
        pass
    else:
        raise AssertionError("same owner+key with another body must be rejected")


def test_create_job_is_atomic_under_concurrency(tmp_path):
    repo = Repository(_store(tmp_path))
    body = {"workflow_key": "demo", "workflow_version": 1, "params": {"seed": -1}}

    def submit(_):
        return repo.create_job("owner", body, "concurrent")

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(submit, range(6)))
    assert len({item[0]["id"] for item in results}) == 1 and sum(item[1] for item in results) == 1


def test_event_is_append_only_and_owner_scoped(tmp_path):
    repo = Repository(_store(tmp_path))
    job, _ = repo.create_job("owner", {"workflow_key": "demo"}, "event-key")
    event = repo.event(job["id"], "owner", "running", "started", 0.25)
    assert event["state"] == "running" and repo.list("events", "owner")[0]["message"] == "started"


def test_registry_bootstrap_marks_demo_published_and_real_models_disabled(tmp_path):
    registry = Registry(Repository(_store(tmp_path)))
    result = registry.bootstrap("owner")
    assert result["demo_workflow"]["state"] == "published"
    models = {m["model_id"]: m for m in registry.models("owner")}
    assert models["demo-mock"]["enabled"] is True
    for model_id in ("moss", "krea", "h3"):
        assert models[model_id]["enabled"] is False and models[model_id]["disabled_reason"]


def test_registry_versions_are_immutable_and_rollback_switches_active(tmp_path):
    repo = Repository(_store(tmp_path))
    registry = Registry(repo)
    registry.bootstrap("owner")
    bundle = registry.get_workflow("owner", "demo-mock", 1)
    imported = registry.import_workflow("owner", bundle)
    assert imported["state"] == "draft"
    validated = registry.validate("owner", imported["id"])
    assert validated["state"] == "validated"
    repo.update("workflows", imported["id"], {"smoke_test_passed": True, "smoke_test_kind": "mock"}, "owner")
    published = registry.publish("owner", imported["id"], confirmed=True)
    assert published["state"] == "published" and published["workflow_version"] != bundle["workflow_version"]
    assert registry.rollback("owner", "demo-mock", 1)["active"] is True


def test_registry_import_respects_semver_manifest_and_auto_increments_legacy_versions(tmp_path):
    repo = Repository(_store(tmp_path))
    registry = Registry(repo)
    registry.bootstrap("owner")
    bundle = registry.get_workflow("owner", "demo-mock", 1)
    semver = {**bundle, "manifest": {**bundle["manifest"], "version": "2.4.0"}}
    imported = registry.import_workflow("owner", semver)
    assert imported["workflow_version"] == "2.4.0"
    legacy = registry.import_workflow("owner", bundle)
    assert legacy["workflow_version"] == 2


def test_registry_validate_allows_disabled_real_model_but_publish_requires_license(tmp_path):
    repo = Repository(_store(tmp_path))
    registry = Registry(repo)
    repo.create("models", "owner", {"model_id": "real", "enabled": False, "is_mock": False})
    source = registry._template()
    row = repo.create(
        "workflows",
        "owner",
        {
            "workflow_key": "real",
            "workflow_version": 1,
            "model_id": "real",
            "manifest": source[0],
            "api_graph": source[1],
            "object_info": source[2],
            "editable_graph": source[1],
            "state": "draft",
        },
    )
    assert registry.validate("owner", row["id"])["state"] == "validated"
    repo.update("models", repo.list("models", "owner")[0]["id"], {"enabled": True}, "owner")
    repo.update(
        "workflows",
        row["id"],
        {"smoke_test_passed": True, "smoke_test_kind": "real", "review_approved": True},
        "owner",
    )
    with pytest.raises(ValueError, match="许可|license"):
        registry.publish("owner", row["id"], confirmed=True)


def _real_publish_fixture(tmp_path):
    from app import review_routes

    repo = Repository(_store(tmp_path))
    registry = Registry(repo)
    repo.create(
        "models",
        "owner",
        {
            "model_id": "real",
            "revision": "b" * 40,
            "enabled": True,
            "is_mock": False,
            "license_status": "approved",
        },
    )
    manifest, graph, info = registry._template()
    manifest = copy.deepcopy(manifest)
    manifest["model_id"] = "real"
    manifest["dependencies"] = {
        "comfyui_commit": "a" * 40,
        "custom_nodes": [],
        "weight_files": [],
    }
    graph = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"text": "demo", "steps": 1, "seed": 1},
        }
    }
    info = {
        "CheckpointLoaderSimple": {
            "input": {"required": {"text": ["STRING"], "steps": ["INT"], "seed": ["INT"]}}
        }
    }
    row = repo.create(
        "workflows",
        "owner",
        {
            "workflow_key": "real",
            "workflow_version": 1,
            "model_id": "real",
            "manifest": manifest,
            "api_graph": graph,
            "object_info": info,
            "state": "draft",
        },
    )
    registry.validate("owner", row["id"])
    row = repo.get("workflows", row["id"], "owner")
    repo.update(
        "workflows",
        row["id"],
        {
            "review_approved": True,
            "smoke_test_passed": True,
            "smoke_test_kind": "real",
        },
        "owner",
    )
    row = repo.get("workflows", row["id"], "owner")
    signature = review_routes._smoke_compatibility_signature(row["id"], row, "b" * 40)
    job, _ = repo.create_job("owner", {"workflow_smoke": True}, "real-smoke")
    repo.update(
        "jobs",
        job["id"],
        {
            "state": "completed",
            "provider": "colab",
            "workflow_smoke": True,
            "target_workflow_id": row["id"],
            "target_model_id": "real",
            "workflow_key": "real",
            "workflow_version": 1,
            "smoke_compatibility_signature": signature,
            "smoke_accepted": True,
            "acceptance_state": "accepted",
        },
        "owner",
    )
    repo.update("workflows", row["id"], {"smoke_job_id": job["id"]}, "owner")
    return repo, registry, row["id"], job["id"]


def test_registry_rejects_historical_real_smoke_flags_without_current_job(tmp_path):
    repo, registry, workflow_id, _ = _real_publish_fixture(tmp_path)
    repo.update("workflows", workflow_id, {"smoke_job_id": "missing"}, "owner")

    with pytest.raises(ValueError, match="当前匹配|smoke"):
        registry.publish("owner", workflow_id, confirmed=True)


def test_registry_publishes_real_workflow_with_current_accepted_smoke_fixture(tmp_path):
    _, registry, workflow_id, _ = _real_publish_fixture(tmp_path)

    published = registry.publish("owner", workflow_id, confirmed=True)

    assert published["state"] == "published"
    assert published["active"] is True


def test_workflow_owner_key_version_is_database_unique(tmp_path):
    repo = Repository(_store(tmp_path))
    data = {"workflow_key": "same", "workflow_version": 1, "model_id": "demo"}
    repo.create("workflows", "owner", data)
    with pytest.raises(ValueError, match="唯一|存在"):
        repo.create("workflows", "owner", data)
    assert repo.create("workflows", "other", data)["owner_id"] == "other"


def test_registry_bootstrap_is_idempotent_under_concurrency(tmp_path):
    repo = Repository(_store(tmp_path))

    def bootstrap(_):
        return Registry(repo).bootstrap("owner")

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(bootstrap, range(6)))
    assert len(results) == 6
    registry = Registry(repo)
    assert len([row for row in registry.models("owner") if row["model_id"] == "demo-mock"]) == 1
    assert len([row for row in registry.workflows("owner") if row["workflow_key"] == "demo-mock"]) == 1


def _publish_fixture_with_graph(tmp_path, graph, info, key="anima-hires"):
    from app import review_routes

    repo = Repository(_store(tmp_path))
    registry = Registry(repo)
    repo.create(
        "models",
        "owner",
        {
            "model_id": "real",
            "revision": "b" * 40,
            "enabled": True,
            "is_mock": False,
            "license_status": "approved",
        },
    )
    manifest = {
        "workflow_key": key,
        "model_id": "real",
        "version": "1.0.0",
        "parameters": {
            "prompt": {"type": "string", "binding": {"node": "20", "input": "text"}, "default": "demo"},
            "seed": {"type": "integer", "binding": {"node": "40", "input": "noise_seed"}, "default": 1, "min": 0, "max": 9999},
        },
        "dependencies": {
            "comfyui_commit": "a" * 40,
            "custom_nodes": [],
            "weight_files": [],
        },
    }
    row = repo.create(
        "workflows",
        "owner",
        {
            "workflow_key": key,
            "workflow_version": "1.0.0",
            "model_id": "real",
            "manifest": manifest,
            "api_graph": graph,
            "object_info": info,
            "state": "draft",
        },
    )
    registry.validate("owner", row["id"])
    row = repo.get("workflows", row["id"], "owner")
    repo.update(
        "workflows",
        row["id"],
        {"review_approved": True, "smoke_test_passed": True, "smoke_test_kind": "real"},
        "owner",
    )
    row = repo.get("workflows", row["id"], "owner")
    signature = review_routes._smoke_compatibility_signature(row["id"], row, "b" * 40)
    job, _ = repo.create_job("owner", {"workflow_smoke": True}, f"real-smoke-{key}")
    repo.update(
        "jobs",
        job["id"],
        {
            "state": "completed",
            "provider": "colab",
            "workflow_smoke": True,
            "target_workflow_id": row["id"],
            "target_model_id": "real",
            "workflow_key": key,
            "workflow_version": "1.0.0",
            "smoke_compatibility_signature": signature,
            "smoke_accepted": True,
            "acceptance_state": "accepted",
        },
        "owner",
    )
    repo.update("workflows", row["id"], {"smoke_job_id": job["id"]}, "owner")
    return registry, row["id"]


def test_registry_allows_core_two_pass_upscale_nodes_and_rejects_glsl_and_custom_nodes(tmp_path):
    graph = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "m.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "c.safetensors", "type": "stable_diffusion", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "v.safetensors"}},
        "10": {"class_type": "LoraLoader", "inputs": {"model": ["1", 0], "clip": ["2", 0], "lora_name": "l1.safetensors", "strength_model": 0.5, "strength_clip": 0.5}},
        "20": {"class_type": "CLIPTextEncode", "inputs": {"text": "demo", "clip": ["10", 1]}},
        "21": {"class_type": "CLIPTextEncode", "inputs": {"text": "bad", "clip": ["10", 1]}},
        "30": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1536, "batch_size": 1}},
        "40": {
            "class_type": "KSamplerAdvanced",
            "inputs": {
                "model": ["10", 0], "add_noise": "enable", "noise_seed": 1, "steps": 12, "cfg": 1.0,
                "sampler_name": "er_sde", "scheduler": "simple", "positive": ["20", 0], "negative": ["21", 0],
                "latent_image": ["30", 0], "start_at_step": 1, "end_at_step": 100, "return_with_leftover_noise": "disable",
            },
        },
        "41": {"class_type": "VAEDecode", "inputs": {"samples": ["40", 0], "vae": ["3", 0]}},
        "50": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "4xUltrasharp_4xUltrasharpV10.pt"}},
        "51": {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["50", 0], "image": ["41", 0]}},
        "52": {"class_type": "ImageScaleBy", "inputs": {"image": ["51", 0], "upscale_method": "lanczos", "scale_by": 0.38}},
        "53": {"class_type": "VAEEncode", "inputs": {"pixels": ["52", 0], "vae": ["3", 0]}},
        "60": {
            "class_type": "KSamplerAdvanced",
            "inputs": {
                "model": ["10", 0], "add_noise": "enable", "noise_seed": 2, "steps": 12, "cfg": 1.0,
                "sampler_name": "euler_ancestral", "scheduler": "karras", "positive": ["20", 0], "negative": ["21", 0],
                "latent_image": ["53", 0], "start_at_step": 7, "end_at_step": 100, "return_with_leftover_noise": "disable",
            },
        },
        "61": {"class_type": "VAEDecode", "inputs": {"samples": ["60", 0], "vae": ["3", 0]}},
        "62": {"class_type": "SaveImage", "inputs": {"images": ["61", 0], "filename_prefix": "Anima-HiRes"}},
    }
    info = {
        "UNETLoader": {"input": {"required": {"unet_name": [["m.safetensors"]], "weight_dtype": [["default"]]}}, "output": ["MODEL"]},
        "CLIPLoader": {"input": {"required": {"clip_name": [["c.safetensors"]], "type": [["stable_diffusion"]]}, "optional": {"device": [["default"]]}}, "output": ["CLIP"]},
        "VAELoader": {"input": {"required": {"vae_name": [["v.safetensors"]]}}, "output": ["VAE"]},
        "LoraLoader": {"input": {"required": {"model": ["MODEL"], "clip": ["CLIP"], "lora_name": [["l1.safetensors"]], "strength_model": ["FLOAT"], "strength_clip": ["FLOAT"]}}, "output": ["MODEL", "CLIP"]},
        "CLIPTextEncode": {"input": {"required": {"text": ["STRING"], "clip": ["CLIP"]}}, "output": ["CONDITIONING"]},
        "EmptyLatentImage": {"input": {"required": {"width": ["INT"], "height": ["INT"], "batch_size": ["INT"]}}, "output": ["LATENT"]},
        "KSamplerAdvanced": {
            "input": {
                "required": {
                    "model": ["MODEL"], "add_noise": [["enable", "disable"]], "noise_seed": ["INT"],
                    "steps": ["INT"], "cfg": ["FLOAT"], "sampler_name": [["er_sde", "euler_ancestral"]],
                    "scheduler": [["simple", "karras"]], "positive": ["CONDITIONING"], "negative": ["CONDITIONING"],
                    "latent_image": ["LATENT"], "start_at_step": ["INT"], "end_at_step": ["INT"],
                    "return_with_leftover_noise": [["disable", "enable"]],
                }
            },
            "output": ["LATENT"],
        },
        "VAEDecode": {"input": {"required": {"samples": ["LATENT"], "vae": ["VAE"]}}, "output": ["IMAGE"]},
        "UpscaleModelLoader": {"input": {"required": {"model_name": [["4xUltrasharp_4xUltrasharpV10.pt"]]}}, "output": ["UPSCALE_MODEL"]},
        "ImageUpscaleWithModel": {"input": {"required": {"upscale_model": ["UPSCALE_MODEL"], "image": ["IMAGE"]}}, "output": ["IMAGE"]},
        "ImageScaleBy": {"input": {"required": {"image": ["IMAGE"], "upscale_method": [["lanczos"]], "scale_by": ["FLOAT"]}}, "output": ["IMAGE"]},
        "VAEEncode": {"input": {"required": {"pixels": ["IMAGE"], "vae": ["VAE"]}}, "output": ["LATENT"]},
        "SaveImage": {"input": {"required": {"images": ["IMAGE"], "filename_prefix": ["STRING"]}}, "output": ["IMAGE"]},
    }
    registry, wf_id = _publish_fixture_with_graph(tmp_path / "ok", graph, info, key="anima-hires")
    published = registry.publish("owner", wf_id, confirmed=True)
    assert published["state"] == "published"

    for disallowed_class in ("GLSLShader", "Lora Loader Stack (rgthree)"):
        bad_graph = copy.deepcopy(graph)
        bad_info = copy.deepcopy(info)
        bad_graph["99"] = {"class_type": disallowed_class, "inputs": {"image": ["61", 0]}}
        bad_info[disallowed_class] = {"input": {"required": {"image": ["IMAGE"]}}, "output": ["IMAGE"]}
        bad_reg, bad_id = _publish_fixture_with_graph(tmp_path / disallowed_class.replace(" ", "_"), bad_graph, bad_info, key="bad")
        with pytest.raises(ValueError, match="allowlist"):
            bad_reg.publish("owner", bad_id, confirmed=True)
