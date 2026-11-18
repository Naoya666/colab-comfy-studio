"""Local fakes only: selection, allocation failures, instance isolation and compatibility."""

import copy
from types import SimpleNamespace

import pytest
from app.db import Store
from app.gpu.colab import ColabAllocationError
from app.gpu.compute import ComputeError
from app.gpu.instances import InstanceService
from app.gpu.registry import Registry
from app.gpu.repository import Repository
from app.gpu.runtime_worker import WorkerError, _check_compute_hardware
from app.gpu.scheduler import Scheduler
from app.gpu.specs import compatibility_signature, resolve_compute
from app.job_schemas import Policy


def key(gpu="L4", shape="SHAPE_STANDARD"):
    return {"variant": "VARIANT_GPU", "accelerator": gpu, "shape": shape}


def spec(gpu="L4", **extras):
    return {"key": key(gpu), "eligible": True, **extras}


def test_default_override_does_not_mutate_workflow_and_normalizes_aliases():
    manifest = {"compute_spec": key()}
    before = copy.deepcopy(manifest)
    default, warnings = resolve_compute(manifest, None)
    assert default["key"] == key() and warnings == []
    chosen, warnings = resolve_compute(
        manifest, {"variant": "GPU", "accelerator": "a100", "shape": "STANDARD"}
    )
    assert chosen["key"] == key("A100") and warnings
    assert manifest == before
    assert resolve_compute({}, None) == (None, [])
    assert "compute_spec" not in Policy().model_dump()


@pytest.mark.parametrize(
    "bad", [key(shape="made-up"), {**key(), "variant": "CPU"}, {"accelerator": "A100"}, {**key(), "extra": 1}]
)
def test_invalid_specs_fail_closed(bad):
    with pytest.raises(ValueError):
        resolve_compute({"compute_spec": key()}, bad)


def test_scheduler_selects_requested_exact_spec_without_fallback():
    scheduler = Scheduler(Repository(Store("sqlite:///:memory:")))
    job = {"requested_spec": {"key": key("A100")}, "snapshot": {"manifest": {"compute_spec": key()}}}
    provider = SimpleNamespace(list_specs=lambda: [spec(), spec("A100")])
    assert scheduler._select_spec(job, provider)["key"] == key("A100")
    provider.list_specs = lambda: [spec()]
    with pytest.raises(ComputeError, match="未切换"):
        scheduler._select_spec(job, provider)
    provider.list_specs = lambda: [spec("A100", eligible=False)]
    with pytest.raises(ComputeError):
        scheduler._select_spec(job, provider)


def test_explicit_incompatibility_and_actual_memory_gate():
    from app.gpu.specs import check_requirements

    with pytest.raises(ValueError, match="低于最低"):
        check_requirements({"compute_requirements": {"min_vram_gb": 48}}, spec("A100", vram_gb=40))
    with pytest.raises(ValueError, match="架构限制"):
        resolve_compute({"compute_requirements": {"accelerators": ["A100"]}}, key())
    with pytest.raises(WorkerError, match="实际 GPU 显存"):
        _check_compute_hardware(
            {"compute_requirements": {"min_vram_gb": 48}}, {"devices": [{"memory_mib": 40960}]}
        )
    _check_compute_hardware(
        {"compute_requirements": {"min_vram_gb": 24}}, {"devices": [{"memory_mib": 40960}]}
    )


def test_same_spec_reuse_and_busy_or_different_spec_rejected():
    service = InstanceService(Repository(Store("sqlite:///:memory:")))
    instance = SimpleNamespace(
        state="retained_idle",
        data={"account_id": "a", "runtime_spec_requested": spec(), "compatibility_signature": "same"},
    )
    job = {"requested_spec": spec(), "compatibility_signature": "same"}
    with service.store.Session() as session:
        assert service._compatible(session, instance, job, "a")
        assert not service._compatible(session, instance, {**job, "requested_spec": spec("A100")}, "a")
        assert not service._compatible(session, instance, job, "other-account")
        instance.data["current_job_id"] = "other-job"
        assert not service._compatible(session, instance, job, "a")


def test_cache_signature_separates_gpu_not_parameters():
    a = {"manifest": {"compute_spec": key()}, "resolved_params": {"seed": 1}}
    b = {**a, "resolved_params": {"seed": 2}}
    assert compatibility_signature("colab", a, spec()) == compatibility_signature("colab", b, spec())
    assert compatibility_signature("colab", a, spec()) != compatibility_signature("colab", a, spec("A100"))


def test_allocation_failure_keeps_reason_and_never_selects_another_gpu(monkeypatch):
    scheduler = Scheduler(Repository(Store("sqlite:///:memory:")))
    captured = []
    monkeypatch.setattr(
        scheduler, "_begin_failure", lambda job, code, message: captured.append((code, message))
    )
    scheduler._allocation_failed({"requested_spec": spec("A100")}, ColabAllocationError(8, "NO_RUNTIMES"))
    assert captured[0][0] == "gpu_capacity_unavailable"
    assert "NO_RUNTIMES" in captured[0][1]


def test_image_extraction_enters_existing_registry_as_unreviewed_draft():
    registry = Registry(Repository(Store("sqlite:///:memory:")))
    workflow = {"nodes": [{"id": 1, "type": "UntrustedNode"}], "links": []}
    prompt = {"1": {"class_type": "UntrustedNode", "inputs": {"ckpt_name": "model.safetensors"}}}
    row = registry.import_workflow(
        "owner", {"workflow_key": "from-image", "image_extraction": {"workflow": workflow, "prompt": prompt}}
    )
    assert row["state"] == "draft" and row["active"] is False and row["smoke_test_passed"] is False
    assert row["api_graph"] == prompt and row["editable_graph"] == workflow
    assert row["image_extraction"]["workflow"] == workflow and row["pending_configuration"]
    assert registry.repo.list("jobs") == [] and registry.repo.list("leases") == []
    with pytest.raises(ValueError, match="尚待配置"):
        registry.validate("owner", row["id"])
    with pytest.raises(ValueError):
        registry.publish("owner", row["id"], confirmed=True)


@pytest.mark.parametrize("which", ["workflow", "prompt"])
def test_partial_image_draft_preserved(which):
    payload = {
        "workflow": {"nodes": [{"id": 1, "type": "Node"}]},
        "prompt": {"1": {"class_type": "Node", "inputs": {}}},
    }
    registry = Registry(Repository(Store("sqlite:///:memory:")))
    row = registry.import_workflow(
        "owner", {"workflow_key": which, "image_extraction": {which: payload[which]}}
    )
    assert row["image_extraction"][which] == payload[which]
    if which == "workflow":
        assert row["api_graph"] == {}  # never runs visual workflow as API JSON


@pytest.mark.parametrize(
    "extraction", [{}, {"workflow": {"1": {"class_type": "Node", "inputs": {}}}}, {"prompt": {"nodes": []}}]
)
def test_bad_image_import_json_cannot_bypass_format_checks(extraction):
    registry = Registry(Repository(Store("sqlite:///:memory:")))
    with pytest.raises(ValueError):
        registry.import_workflow("owner", {"workflow_key": "bad", "image_extraction": extraction})


def test_runtime_memory_is_not_assumed_80gb_and_wrong_device_stops_before_load():
    evidence = {"devices": [{"name": "NVIDIA A100-SXM4-40GB", "memory_mib": 40536}]}
    _check_compute_hardware({"compute_requirements": {"min_vram_gb": 40}}, evidence, key("A100"))
    with pytest.raises(WorkerError, match="最低 80"):
        _check_compute_hardware({"compute_requirements": {"min_vram_gb": 80}}, evidence, key("A100"))
    with pytest.raises(WorkerError, match="不一致"):
        _check_compute_hardware({}, evidence, key("L4"))


def test_cu_estimate_uses_runtime_spec_and_unknown_gpu_requires_explicit_rate():
    from app.admission import _job_estimate
    from app.gpu.compute import GpuConfig
    from app.gpu.rates import minute_rate

    repo = Repository(Store("sqlite:///:memory:"))
    workflow = {"workflow_key": "w", "manifest": {"compute_spec": key(), "typical_minutes": 1}}
    cfg = GpuConfig(unit_rate=0.1)
    policy = {"provider": "colab", "max_runtime_minutes": 10}
    default = _job_estimate(repo, "o", workflow, policy, cfg, {})
    override = _job_estimate(repo, "o", workflow, {**policy, "resolved_compute_spec": key("A100")}, cfg, {})
    assert override["unit_rate"] > default["unit_rate"]
    with pytest.raises(ValueError, match="无已知估算费率"):
        minute_rate(key("unknown-gpu"))
    assert minute_rate(key("G4"), fallback_rate=0.2) == 0.2


def test_cli_actual_spec_is_never_copied_from_request():
    from app.gpu.colab import GoogleColabProvider

    assert GoogleColabProvider._assignment_spec({"endpoint": "e"}) is None
    assert GoogleColabProvider._assignment_spec(
        {"variant": 1, "accelerator": "A100", "machineShape": 0}
    ) == key("A100")


@pytest.mark.parametrize("gpu", ["L4", "A100"])
def test_selected_spec_is_sent_to_real_provider_http_adapter_with_mock_transport(gpu, monkeypatch):
    from app.config import settings
    from cryptography.fernet import Fernet

    monkeypatch.setattr(settings, "encryption_key", Fernet.generate_key().decode())
    import json

    import httpx

    from tests.gpu.test_colab_contract import make

    sent = []

    def handler(request):
        sent.append(json.loads(request.content)["runtimeSpec"])
        return httpx.Response(
            200,
            json={
                "name": "operations/op-1",
                "done": True,
                "response": {"name": "runtimes/atelier-runtime-1"},
            },
        )

    provider = make(handler)
    scheduler = Scheduler(Repository(Store("sqlite:///:memory:")))
    selected = scheduler._select_spec(
        {"snapshot": {"manifest": {"compute_spec": key()}}, "requested_spec": spec(gpu)},
        SimpleNamespace(list_specs=lambda: [spec(), spec("A100")]),
    )
    provider.create(selected, "11111111-1111-4111-8111-111111111111", "job-1", "atelier-runtime-1")
    assert sent == [key(gpu)]


def test_real_job_admission_persists_override_and_keeps_default_artifact(monkeypatch):
    from app import admission
    from app.gpu import ccu
    from app.gpu.compute import GpuConfig
    from app.job_schemas import JobInput

    store = Store("sqlite:///:memory:")
    repo = Repository(store)
    registry = Registry(repo)
    bootstrap = registry.bootstrap("owner")
    wf = bootstrap["demo_workflow"]
    model = next(m for m in repo.list("models", "owner") if m["model_id"] == "demo-mock")
    repo.update("models", model["id"], {"is_mock": False}, "owner")
    # Test fixture is a reviewed real job at the admission boundary; no runtime is created.
    from app.gpu.repository import GpuWorkflowRow
    from sqlalchemy import update

    manifest = {**wf["manifest"], "compute_spec": key()}
    with store.Session.begin() as session:
        data = {**wf, "manifest": manifest, "review_approved": True, "smoke_test_kind": "real"}
        session.execute(update(GpuWorkflowRow).where(GpuWorkflowRow.id == wf["id"]).values(data=data))
    project = store.create("projects", "owner", {"title": "test"})
    monkeypatch.setattr(admission.GpuConfig, "from_env", lambda: GpuConfig(real_enabled=True, unit_rate=0.1))
    monkeypatch.setattr(admission, "_google_status", lambda *a: {"colab": {"connected": True}})
    monkeypatch.setattr(ccu, "maybe_sample_balance", lambda *a: None)
    body = JobInput(
        project_id=project["id"],
        model_id="demo-mock",
        workflow_key="demo-mock",
        workflow_version=1,
        execution_policy={"provider": "colab", "compute_spec": key("A100")},
    ).model_dump()
    job = admission.admit_job(store, {"id": "owner"}, body, "runtime-override")
    assert job["requested_spec"]["key"] == key("A100")
    assert job["snapshot"]["manifest"]["compute_spec"] == key()
    assert registry.get_workflow("owner", "demo-mock", 1)["manifest"]["compute_spec"] == key()
    assert repo.list("leases") == []


def test_image_import_http_handlers_preserve_original_and_save_incomplete_config():
    import json

    from app import main

    user = {"id": "owner", "role": "admin"}
    request = SimpleNamespace(state=SimpleNamespace(request_id="local-test"))
    store = Store("sqlite:///:memory:")
    original = {"nodes": [{"id": 1, "type": "UnknownCustomNode"}]}
    response = main.import_workflow(
        {"workflow_key": "png-draft", "image_extraction": {"workflow": original}}, request, user, store
    )
    row = json.loads(response.body)["data"]
    assert row["state"] == "draft"
    assert row["image_import_summary"]["node_types"] == ["UnknownCustomNode"]
    body = {"model_id": "unconfigured", "manifest": {"parameters": {}}, "api_graph": {}, "object_info": {}}
    response = main.configure_image_draft(row["id"], body, request, user, store)
    configured = json.loads(response.body)["data"]
    assert configured["image_extraction"]["workflow"] == original
    assert configured["pending_configuration"] and not configured["review_approved"]
    repo = Repository(store)
    repo.update("workflows", row["id"], {"state": "validated"}, "owner")
    from app.common import APIError

    with pytest.raises(APIError, match="只能配置"):
        main.configure_image_draft(row["id"], body, request, user, store)
    assert repo.list("jobs") == [] and repo.list("leases") == []
