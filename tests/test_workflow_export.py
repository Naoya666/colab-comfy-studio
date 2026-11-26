"""Tests for published workflow export, Civitai/manual portability (Scheme A), auto model registration, and version conflicts."""
from __future__ import annotations

import copy
import json
from contextlib import contextmanager

import httpx
import pytest
from app.common import current_user
from app.config import settings
from app.db import Store
from app.gpu.model_library import ModelLibrary
from app.gpu.model_sources import SourceHTTP
from app.gpu.repository import Repository
from app.gpu.workflow_export import build_workflow_export, suggest_next_version
from app.main import app
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

SHA_HF = "1" * 64
SHA_CIV = "2" * 64
SHA_PRIV = "3" * 64
COMMIT_A = "a" * 40
COMMIT_B = "b" * 40


def _sample_graph():
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "base_hf.safetensors"},
        },
        "2": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["1", 0],
                "lora_name": "civitai_style.safetensors",
                "strength_model": 0.8,
            },
        },
        "3": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["2", 0],
                "lora_name": "private_char.safetensors",
                "strength_model": 0.6,
            },
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["1", 1], "text": "hello"},
        },
    }


def _sample_object_info():
    return {
        "CheckpointLoaderSimple": {
            "input": {"required": {"ckpt_name": [["base_hf.safetensors"]]}},
            "output": ["MODEL", "CLIP", "VAE"],
        },
        "LoraLoaderModelOnly": {
            "input": {
                "required": {
                    "model": ["MODEL"],
                    "lora_name": [["civitai_style.safetensors", "private_char.safetensors"]],
                    "strength_model": ["FLOAT", {"default": 1.0, "min": -3.0, "max": 3.0}],
                }
            },
            "output": ["MODEL"],
        },
        "CLIPTextEncode": {
            "input": {"required": {"clip": ["CLIP"], "text": ["STRING"]}},
            "output": ["CONDITIONING"],
        },
    }


def _sample_manifest(weights: list[dict], *, version: str | None = "1.0.0", model_id: str = "custom-civitai-base"):
    manifest = {
        "workflow_key": "shared-style-t2i",
        "model_id": model_id,
        "title": "Shared Style T2I",
        "compute_spec": {"variant": "VARIANT_GPU", "accelerator": "L4", "shape": "SHAPE_STANDARD"},
        "max_runtime_minutes": 20,
        "parameters": {
            "prompt": {
                "type": "string",
                "label": "提示词",
                "default": "hello",
                "binding": {"node": "4", "input": "text"},
            }
        },
        "dependencies": {
            "comfyui_commit": COMMIT_A,
            "custom_nodes": [],
            "weight_files": copy.deepcopy(weights),
        },
    }
    if version is not None:
        manifest["version"] = version
    return manifest


@contextmanager
def _client_for(tmp_path, monkeypatch, *, owner_id: str = "owner", role: str = "admin", store: Store | None = None):
    monkeypatch.setattr(settings, "encryption_key", Fernet.generate_key().decode())
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    active_store = store or Store(f"sqlite:///{tmp_path / 'studio.db'}")
    app.state.store = active_store
    app.dependency_overrides[current_user] = lambda: {"id": owner_id, "role": role}
    try:
        with TestClient(app) as client:
            yield client, active_store
    finally:
        app.dependency_overrides.clear()


def _seed_published_workflow(repo: Repository, owner_id: str = "owner", *, workflow_version="1.0.0"):
    lib_hf = repo.create(
        "model_library",
        owner_id,
        {
            "display_name": "Base HF Checkpoint",
            "filename": "base_hf.safetensors",
            "model_type": "checkpoints",
            "relative_path": "base_hf.safetensors",
            "architecture": "sdxl",
            "sha256": SHA_HF,
            "expected_sha256": SHA_HF,
            "size": 1024000,
            "source": "huggingface",
            "source_url": f"https://huggingface.co/org/repo/resolve/{COMMIT_B}/base_hf.safetensors",
            "source_identity": {
                "repo_id": "org/repo",
                "revision": COMMIT_B,
                "filename": "base_hf.safetensors",
            },
            "drive_file_id": "drive-secret-hf-1",
            "state": "ready",
        },
    )
    lib_civ = repo.create(
        "model_library",
        owner_id,
        {
            "display_name": "Civitai Watercolor Style LoRA",
            "filename": "civitai_style.safetensors",
            "model_type": "loras",
            "relative_path": "civitai_style.safetensors",
            "architecture": "sdxl",
            "sha256": SHA_CIV,
            "expected_sha256": SHA_CIV,
            "size": 204800,
            "source": "civitai",
            "source_url": "https://civitai.com/models/12345?modelVersionId=67890",
            "download_url": "https://civitai.com/api/download/models/67890",
            "source_identity": {"model_id": 12345, "version_id": 67890, "file_id": 999},
            "version_name": "v2.0",
            "drive_file_id": "drive-secret-civ-2",
            "state": "ready",
        },
    )
    lib_priv = repo.create(
        "model_library",
        owner_id,
        {
            "display_name": "My Private Character LoRA",
            "filename": "private_char.safetensors",
            "model_type": "loras",
            "relative_path": "private_char.safetensors",
            "architecture": "sdxl",
            "sha256": SHA_PRIV,
            "expected_sha256": SHA_PRIV,
            "size": 51200,
            "source": "upload",
            "drive_file_id": "drive-secret-priv-3",
            "state": "ready",
        },
    )
    model = repo.create(
        "models",
        owner_id,
        {
            "model_id": "custom-civitai-base",
            "display_name": "Custom Civitai Base Model",
            "library_model_id": lib_civ["id"],
            "source_kind": "civitai",
            "source": "https://civitai.com/models/12345?modelVersionId=67890",
            "source_identity": {"model_id": 12345, "version_id": 67890, "file_id": 999},
            "drive_file_id": "drive-secret-civ-2",
            "model_sha256": SHA_CIV,
            "license": "Fair AI Public License 1.0-SD",
            "modality": "image",
            "resource_profile": {"min_vram_gb": 12},
            "enabled": True,
            "review_approved": True,
            "license_accepted": True,
            "license_review_approved": True,
        },
    )
    weights = [
        {"source": "drive", "drive_file_id": lib_hf["drive_file_id"], "sha256": SHA_HF, "size": 1024000, "target": "checkpoints/base_hf.safetensors"},
        {"source": "drive", "drive_file_id": lib_civ["drive_file_id"], "sha256": SHA_CIV, "size": 204800, "target": "loras/civitai_style.safetensors"},
        {"source": "drive", "drive_file_id": lib_priv["drive_file_id"], "sha256": SHA_PRIV, "size": 51200, "target": "loras/private_char.safetensors"},
    ]
    manifest = _sample_manifest(weights, version=workflow_version if isinstance(workflow_version, str) else None)
    # Simulate review_workflow writing derived top-level fields onto manifest
    manifest.update(
        node_allowlist=["CheckpointLoaderSimple", "LoraLoaderModelOnly", "CLIPTextEncode"],
        comfyui_commit=COMMIT_A,
        comfy_commit=COMMIT_A,
        custom_nodes=[],
        weight_files=copy.deepcopy(weights),
        state="published",
        reviewed=True,
    )
    wf = repo.create(
        "workflows",
        owner_id,
        {
            "workflow_key": "shared-style-t2i",
            "workflow_version": workflow_version,
            "model_id": model["model_id"],
            "manifest": manifest,
            "api_graph": _sample_graph(),
            "object_info": _sample_object_info(),
            "editable_graph": _sample_graph(),
            "state": "published",
            "active": True,
            "review_approved": True,
            "reviewed_by": owner_id,
            "smoke_test_passed": True,
            "smoke_test_kind": "real",
            "smoke_job_id": "job-smoke-1",
            "drive_files": {"manifest": {"id": "drive-wf-secret"}},
            "image_extraction": {"metadata": [{"keyword": "secret", "text": "private"}]},
        },
    )
    return wf, model, (lib_hf, lib_civ, lib_priv)


def test_export_rejects_draft_validated_and_agent_tokens(tmp_path, monkeypatch):
    with _client_for(tmp_path, monkeypatch) as (client, store):
        repo = Repository(store)
        draft = repo.create(
            "workflows",
            "owner",
            {
                "workflow_key": "draft-only",
                "workflow_version": "1.0.0",
                "model_id": "demo-mock",
                "manifest": _sample_manifest([], model_id="demo-mock"),
                "api_graph": _sample_graph(),
                "object_info": _sample_object_info(),
                "state": "draft",
                "active": False,
            },
        )
        res = client.get(f"/api/gpu/workflows/{draft['id']}/export")
        assert res.status_code == 409
        assert res.json()["code"] == "WORKFLOW_NOT_PUBLISHED"

        validated = repo.update("workflows", draft["id"], {"state": "validated"}, "owner")
        res2 = client.get(f"/api/gpu/workflows/{validated['id']}/export")
        assert res2.status_code == 409
        assert res2.json()["code"] == "WORKFLOW_NOT_PUBLISHED"

    # Agent token forbidden even on published workflow
    store = Store(f"sqlite:///{tmp_path / 'agent.db'}")
    repo = Repository(store)
    wf, _, _ = _seed_published_workflow(repo, "owner")
    with _client_for(tmp_path, monkeypatch, owner_id="owner", role="agent", store=store) as (client, _):
        res = client.get(f"/api/gpu/workflows/{wf['id']}/export")
        assert res.status_code == 403


def test_export_sanitizes_drive_ids_restores_hf_preserves_civitai_and_manual_clues(tmp_path, monkeypatch):
    with _client_for(tmp_path, monkeypatch) as (client, store):
        repo = Repository(store)
        wf, _, _ = _seed_published_workflow(repo, "owner", workflow_version=2)

        res = client.get(f"/api/gpu/workflows/{wf['id']}/export")
        assert res.status_code == 200
        data = res.json()["data"]
        assert data["filename"] == "shared-style-t2i-v2.0.0.json"

        bundle = data["bundle"]
        serialized = json.dumps(bundle)
        # Zero private Drive IDs or instance-only approval fields leaked
        assert "drive-secret-" not in serialized
        assert "drive_file_id" not in serialized
        assert "library_model_id" not in serialized
        assert "smoke_job_id" not in serialized
        assert "image_extraction" not in bundle
        assert "drive_files" not in bundle

        manifest = bundle["manifest"]
        assert manifest["version"] == "2.0.0"
        for stripped in ("node_allowlist", "comfyui_commit", "comfy_commit", "custom_nodes", "weight_files", "state", "reviewed"):
            assert stripped not in manifest

        # Model metadata is exported cleanly without approval/enablement flags
        exported_model = bundle["model"]
        assert exported_model["model_id"] == "custom-civitai-base"
        assert exported_model["source"] == "https://civitai.com/models/12345?modelVersionId=67890"
        assert exported_model["source_kind"] == "civitai"
        assert exported_model["model_sha256"] == SHA_CIV
        assert exported_model["license"] == "Fair AI Public License 1.0-SD"
        assert "enabled" not in exported_model
        assert "review_approved" not in exported_model

        # Weights are classified into restored HF, Civitai, and manual with display_name + sha256
        weights = {w["target"]: w for w in manifest["dependencies"]["weight_files"]}
        assert weights["checkpoints/base_hf.safetensors"]["repo_id"] == "org/repo"
        assert weights["checkpoints/base_hf.safetensors"]["revision"] == COMMIT_B
        assert weights["checkpoints/base_hf.safetensors"]["sha256"] == SHA_HF

        civ = weights["loras/civitai_style.safetensors"]
        assert civ["source"] == "civitai"
        assert civ["source_url"] == "https://civitai.com/models/12345?modelVersionId=67890"
        assert civ["source_identity"] == {"model_id": 12345, "version_id": 67890, "file_id": 999}
        assert civ["display_name"] == "Civitai Watercolor Style LoRA"
        assert civ["sha256"] == SHA_CIV

        priv = weights["loras/private_char.safetensors"]
        assert priv["source"] == "manual"
        assert priv["display_name"] == "My Private Character LoRA"
        assert priv["filename"] == "private_char.safetensors"
        assert priv["sha256"] == SHA_PRIV

        summary = data["summary"]
        assert summary["stripped_drive_ids"] == 3
        assert summary["huggingface_weights"] == 1
        assert summary["restored_huggingface_weights"] == 1
        assert summary["civitai_weights"] == 1
        assert summary["manual_rebind_weights"] == 1
        assert len(summary["rebind_models"]) == 2
        assert len(summary["warnings"]) == 3


def test_import_auto_registers_unreviewed_model_and_detects_version_conflicts(tmp_path, monkeypatch):
    store = Store(f"sqlite:///{tmp_path / 'shared.db'}")
    repo = Repository(store)
    wf_a, _, _ = _seed_published_workflow(repo, "exporter", workflow_version="1.0.0")
    exported = build_workflow_export(repo, "exporter", wf_a)
    bundle = copy.deepcopy(exported["bundle"])
    # Tamper with bundle.model trying to smuggle enabled=True / review_approved=True
    bundle["model"]["enabled"] = True
    bundle["model"]["review_approved"] = True
    bundle["model"]["drive_file_id"] = "evil-drive-id"

    with _client_for(tmp_path, monkeypatch, owner_id="recipient", store=store) as (client, _):
        # 1. First import succeeds, auto-registers model as disabled & unreviewed
        res = client.post("/api/gpu/workflows/import", json=bundle)
        assert res.status_code == 201
        imported = res.json()["data"]
        assert imported["state"] == "draft"
        assert imported["model_auto_registered"] is True
        assert imported["import_weight_summary"]["civitai"] == 1
        assert imported["import_weight_summary"]["manual"] == 1
        assert imported["import_weight_summary"]["requires_rebind"] is True

        models = repo.list("models", "recipient")
        reg = next(m for m in models if m["model_id"] == "custom-civitai-base")
        assert reg["enabled"] is False
        assert reg["review_approved"] is False
        assert reg["license_accepted"] is False
        assert "drive_file_id" not in reg
        assert reg["model_sha256"] == SHA_CIV

        # 2. Re-importing exact same (workflow_key, version) returns 409 with structured conflict detail
        conflict_res = client.post("/api/gpu/workflows/import", json=bundle)
        assert conflict_res.status_code == 409
        err = conflict_res.json()["error"]
        assert err["code"] == "WORKFLOW_VERSION_CONFLICT"
        detail = err["detail"]
        assert detail["workflow_key"] == "shared-style-t2i"
        assert detail["workflow_version"] == "1.0.0"
        assert detail["suggested_version"] == "1.0.1"
        assert detail["suggested_workflow_key"] == "shared-style-t2i-imported"

        # 3. Soft-delete v1.0.0 and verify conflict still catches deleted row!
        repo.update("workflows", imported["id"], {"state": "deleted"}, "recipient")
        del_conflict = client.post("/api/gpu/workflows/import", json=bundle)
        assert del_conflict.status_code == 409
        assert del_conflict.json()["error"]["detail"]["existing_state"] == "deleted"
        assert del_conflict.json()["error"]["detail"]["suggested_version"] == "1.0.1"

        # 4. Importing with suggested version succeeds and does not duplicate model row
        resolved_bundle = copy.deepcopy(bundle)
        resolved_bundle["manifest"]["version"] = detail["suggested_version"]
        ok_res = client.post("/api/gpu/workflows/import", json=resolved_bundle)
        assert ok_res.status_code == 201
        assert ok_res.json()["data"]["workflow_version"] == "1.0.1"
        assert ok_res.json()["data"]["model_auto_registered"] is False
        assert sum(1 for m in repo.list("models", "recipient") if m["model_id"] == "custom-civitai-base") == 1


def test_scheme_a_civitai_and_manual_prepare_rebind_and_model_review_flow(tmp_path, monkeypatch):
    store = Store(f"sqlite:///{tmp_path / 'flow.db'}")
    repo = Repository(store)
    wf_a, _, _ = _seed_published_workflow(repo, "exporter", workflow_version="1.0.0")
    bundle = build_workflow_export(repo, "exporter", wf_a)["bundle"]

    with _client_for(tmp_path, monkeypatch, owner_id="recipient", store=store) as (client, _):
        imp = client.post("/api/gpu/workflows/import", json=bundle)
        assert imp.status_code == 201
        wf_id = imp.json()["data"]["id"]

        # Dependencies scan exposes Civitai URL, display_name, and manual weight clues
        dep_res = client.get(f"/api/gpu/workflows/{wf_id}/model-dependencies")
        assert dep_res.status_code == 200
        deps = {d["reference"]: d for d in dep_res.json()["data"]["dependencies"]}
        assert deps["civitai_style.safetensors"]["display_name"] == "Civitai Watercolor Style LoRA"
        assert deps["civitai_style.safetensors"]["source_hint"] == "https://civitai.com/models/12345?modelVersionId=67890"
        assert deps["civitai_style.safetensors"]["weight"]["source"] == "civitai"
        assert deps["private_char.safetensors"]["display_name"] == "My Private Character LoRA"
        assert deps["private_char.safetensors"]["weight"]["source"] == "manual"

        # Activation explicitly blocks until Civitai/manual weights are bound into a new draft
        act_res = client.get(f"/api/gpu/workflows/{wf_id}/activation")
        assert act_res.status_code == 200
        assert any("Civitai" in b for b in act_res.json()["data"]["blockers"])

        # Direct workflow review also gives clear guidance instead of cryptic error
        # First pre-seed ready model_library entries for recipient and test prepare_dependencies for Civitai
        def civitai_handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/v1/model-versions/67890":
                return httpx.Response(
                    200,
                    json={
                        "id": 67890,
                        "modelId": 12345,
                        "name": "v2.0",
                        "baseModel": "SDXL 1.0",
                        "model": {"name": "Civitai Watercolor Style LoRA", "type": "LORA"},
                        "files": [
                            {
                                "id": 999,
                                "name": "civitai_style.safetensors",
                                "sizeKB": 200,
                                "hashes": {"SHA256": SHA_CIV.upper()},
                                "downloadUrl": "https://civitai.com/api/download/models/67890",
                            }
                        ],
                    },
                )
            return httpx.Response(404)

        mock_http = SourceHTTP(
            {},
            client=httpx.Client(transport=httpx.MockTransport(civitai_handler)),
            resolver=lambda _h: ["93.184.216.34"],
        )
        lib = ModelLibrary(repo, "recipient", http=mock_http)
        try:
            prep = lib.prepare_dependencies(wf_id)
            # Civitai dep resolved & enqueued automatically; manual dep stays unresolved with clear hint
            refreshed = {d["reference"]: d for d in lib.dependency_record(wf_id)["dependencies"]}
            civ_model_id = refreshed["civitai_style.safetensors"]["model_id"]
            assert civ_model_id is not None
            assert refreshed["private_char.safetensors"]["model_id"] is None
            assert "私有存储" in (refreshed["private_char.safetensors"].get("source_error") or "")
            assert refreshed["private_char.safetensors"]["id"] in prep["unresolved"]

            # Trying export_bound_draft before binding manual weight fails clearly
            repo.update("model_library", civ_model_id, {"state": "ready", "sha256": SHA_CIV, "size": 204800, "drive_file_id": "recip-drive-civ"}, "recipient")
            hf_lib = repo.create(
                "model_library",
                "recipient",
                {
                    "display_name": "Base HF",
                    "filename": "base_hf.safetensors",
                    "model_type": "checkpoints",
                    "relative_path": "base_hf.safetensors",
                    "architecture": "sdxl",
                    "sha256": SHA_HF,
                    "size": 1024000,
                    "source": "huggingface",
                    "drive_file_id": "recip-drive-hf",
                    "state": "ready",
                },
            )
            lib.bind(wf_id, refreshed["base_hf.safetensors"]["id"], hf_lib["id"], confirm_unknown=True)
            with pytest.raises(ValueError, match="未绑定"):
                lib.export_bound_draft(wf_id)

            # Now recipient imports & binds the manual model too -> export_bound_draft succeeds!
            priv_lib = repo.create(
                "model_library",
                "recipient",
                {
                    "display_name": "Recipient Private LoRA",
                    "filename": "private_char.safetensors",
                    "model_type": "loras",
                    "relative_path": "private_char.safetensors",
                    "architecture": "sdxl",
                    "sha256": SHA_PRIV,
                    "size": 51200,
                    "source": "upload",
                    "drive_file_id": "recip-drive-priv",
                    "state": "ready",
                },
            )
            lib.bind(wf_id, refreshed["private_char.safetensors"]["id"], priv_lib["id"], confirm_unknown=True)
            bound_draft = lib.export_bound_draft(wf_id)
            bound_weights = bound_draft["manifest"]["dependencies"]["weight_files"]
            assert all(w["source"] == "drive" for w in bound_weights)
            assert {w["drive_file_id"] for w in bound_weights} == {"recip-drive-hf", "recip-drive-civ", "recip-drive-priv"}
        finally:
            lib.close()

        # Model review for the auto-registered Civitai model verifies against recipient's ready library file by SHA-256
        monkeypatch.setattr(
            ModelLibrary,
            "check_remote",
            lambda self, ident: self.get(ident),
        )
        rev_model = client.post(
            "/api/gpu/models/custom-civitai-base/review",
            json={"confirmed": True, "license_accepted": True},
        )
        assert rev_model.status_code == 200
        reviewed_model = rev_model.json()["data"]
        assert reviewed_model["review_approved"] is True
        assert reviewed_model["library_model_id"] == civ_model_id
        assert reviewed_model["drive_file_id"] == "recip-drive-civ"

        # Workflow review on the bound draft now succeeds too!
        rev_wf = client.post(
            f"/api/gpu/workflows/{bound_draft['id']}/review",
            json={"confirmed": True},
        )
        assert rev_wf.status_code == 200
        assert rev_wf.json()["data"]["review_approved"] is True


def test_suggest_next_version_skips_existing_versions():
    assert suggest_next_version("1.0.0", {"1.0.0"}) == "1.0.1"
    assert suggest_next_version("1.0.0", {"1.0.0", "1.0.1", "1.0.2"}) == "1.0.3"
    assert suggest_next_version("2.1.0-beta.1", {"2.1.0-beta.1"}) == "2.1.1"
    assert suggest_next_version("3", {"3", "4"}) == "5"


def test_prepare_dependencies_civitai_requires_exact_file_id_when_sha256_missing(tmp_path):
    """When a Civitai weight specifies file_id without SHA-256, never auto-bind a single mismatched file."""
    store = Store(f"sqlite:///{tmp_path / 'civitai_fid.db'}")
    repo = Repository(store)
    wf = repo.create(
        "workflows",
        "owner",
        {
            "workflow_key": "wf-civitai-fid",
            "workflow_version": 1,
            "model_id": "m-fid",
            "state": "draft",
            "api_graph": {
                "1": {
                    "class_type": "LoraLoader",
                    "inputs": {"lora_name": "style.safetensors"},
                }
            },
            "manifest": {
                "dependencies": {
                    "weight_files": [
                        {
                            "source": "civitai",
                            "target": "loras/style.safetensors",
                            "filename": "style.safetensors",
                            "source_url": "https://civitai.com/models/12345?modelVersionId=67890",
                            "source_identity": {
                                "model_id": 12345,
                                "version_id": 67890,
                                "file_id": 10,
                            },
                        }
                    ]
                }
            },
        },
    )

    returned_file_id = {"value": 20}

    def civitai_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/model-versions/67890":
            return httpx.Response(
                200,
                json={
                    "id": 67890,
                    "modelId": 12345,
                    "name": "v1.0",
                    "baseModel": "SDXL 1.0",
                    "model": {"name": "Style LoRA", "type": "LORA"},
                    "files": [
                        {
                            "id": returned_file_id["value"],
                            "name": "other_variant.safetensors",
                            "sizeKB": 128,
                            "hashes": {},
                            "downloadUrl": "https://civitai.com/api/download/models/67890",
                        }
                    ],
                },
            )
        return httpx.Response(404)

    mock_http = SourceHTTP(
        {},
        client=httpx.Client(transport=httpx.MockTransport(civitai_handler)),
        resolver=lambda _h: ["93.184.216.34"],
    )
    lib = ModelLibrary(repo, "owner", http=mock_http)
    try:
        # 1. Single candidate has file_id=20 while dependency specifies file_id=10 -> MUST NOT auto-bind!
        prep_mismatch = lib.prepare_dependencies(wf["id"])
        assert prep_mismatch["tasks"] == []
        deps_after = lib.dependency_record(wf["id"])["dependencies"]
        assert len(deps_after) == 1
        assert deps_after[0]["id"] in prep_mismatch["unresolved"]
        assert deps_after[0].get("model_id") is None
        assert "不匹配" in (deps_after[0].get("source_error") or "")
        assert lib.models() == []

        # 2. Exact file_id=10 match -> binds and enqueues cleanly.
        returned_file_id["value"] = 10
        prep_match = lib.prepare_dependencies(wf["id"])
        assert len(prep_match["tasks"]) == 1
        assert prep_match["unresolved"] == []
        deps_matched = lib.dependency_record(wf["id"])["dependencies"]
        assert deps_matched[0]["model_id"] is not None
        assert deps_matched[0]["confirmed"] is True
    finally:
        lib.close()
        store.engine.dispose()


def test_workflow_export_recognizes_civitai_red_urls():
    from app.gpu.workflow_export import _is_civitai_url, sanitize_import_model

    assert _is_civitai_url("https://civitai.com/models/123")
    assert _is_civitai_url("https://civitai.red/models/2771337?modelVersionId=3270971")
    assert _is_civitai_url("https://www.civitai.red/api/download/models/3270971")
    assert not _is_civitai_url("https://evil-civitai.red/models/2771337")

    imported = sanitize_import_model(
        {
            "display_name": "Red LoRA",
            "source": "https://civitai.red/models/2771337?modelVersionId=3270971",
        },
        "red-lora",
    )
    assert imported is not None
    assert imported["source_kind"] == "civitai"
    assert imported["source"] == "https://civitai.red/models/2771337?modelVersionId=3270971"
