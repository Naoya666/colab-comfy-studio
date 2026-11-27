"""Human review gates and the real-model smoke-test workflow.

This router is deliberately the only admission path for an unenabled real
model.  It records review evidence, creates only an immutable queued job, and
lets the Scheduler (never this module) perform provider side effects.
"""

from __future__ import annotations

import copy
import hashlib
import math
import re
import time
from collections.abc import Mapping
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, Request

from .common import APIError, admin, current_user, get_store, ok, own
from .compute_routes import _require_account
from .gpu.colab import GoogleColabProvider
from .gpu.compute import GpuConfig
from .gpu.google import GoogleService
from .gpu.inputs import resolve_input_images
from .gpu.manifest import bind_parameters, digest, resolve_parameters, validate_manifest
from .gpu.prep import create_prep_job, prep_idempotency_key
from .gpu.rates import estimate as estimate_units
from .gpu.rates import mode_param
from .gpu.repository import Repository
from .gpu.runtime_worker import REVIEWED_CUSTOM_NODES, _reviewed_custom_nodes, _validate_weights
from .storage import read_asset

router = APIRouter(prefix="/api/gpu")

_COMMIT = re.compile(r"^[0-9a-fA-F]{40}$")
_SOURCE_HOSTS = {"huggingface.co", "www.huggingface.co", "github.com", "www.github.com"}
_DANGEROUS = {
    "python",
    "exec",
    "eval",
    "shell",
    "subprocess",
    "command",
    "bash",
    "powershell",
    "httprequest",
    "download",
    "websocket",
}
# A custom node class is admissible only with its exact reviewed repository pin.
_REQUIRED_CUSTOM_NODE_PINS = {
    "ConditioningKrea2Rebalance": (
        "https://github.com/huwhitememes/comfyui-krea2-conditioning",
        "729cda4fade982988a375b01928f515458407a5c",
    ),
}


def _confirmed(body: dict) -> None:
    if not isinstance(body, dict) or body.get("confirmed") is not True:
        raise APIError("CONFIRMATION_REQUIRED", "此操作必须明确confirmed=true", 403)


def _admin(user: dict) -> None:
    # Keep this explicit at every write boundary: agent tokens are never an
    # acceptable substitute for a human administrator review.
    admin(user)


def _fixed_commit(value, field: str) -> str:
    if not isinstance(value, str) or not _COMMIT.fullmatch(value):
        raise APIError("REVIEW_INVALID", f"{field}必须是固定的40位commit")
    return value.lower()


def _source(value) -> str:
    if not isinstance(value, str) or len(value) > 2048:
        raise APIError("MODEL_REVIEW_INVALID", "模型source必须是可信的HTTPS来源")
    parsed = urlsplit(value)
    host = (parsed.hostname or "").lower()
    if (
        parsed.scheme != "https"
        or host not in _SOURCE_HOSTS
        or parsed.username
        or parsed.password
        or not parsed.path.strip("/")
    ):
        raise APIError("MODEL_REVIEW_INVALID", "模型source只允许huggingface.co或github.com的HTTPS地址")
    return value


def _resource_profile(value) -> dict:
    if not isinstance(value, Mapping) or not value:
        raise APIError("MODEL_REVIEW_INVALID", "resource_profile必须包含资源数字")
    numeric = 0

    def visit(item):
        nonlocal numeric
        if isinstance(item, bool):
            raise APIError("MODEL_REVIEW_INVALID", "resource_profile数字不合理")
        if isinstance(item, (int, float)):
            if not math.isfinite(float(item)) or float(item) <= 0 or float(item) > 65536:
                raise APIError("MODEL_REVIEW_INVALID", "resource_profile数字不合理")
            numeric += 1
            return
        if isinstance(item, Mapping):
            for child in item.values():
                visit(child)
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
            return
        if not isinstance(item, str):
            raise APIError("MODEL_REVIEW_INVALID", "resource_profile格式无效")

    visit(value)
    if not numeric:
        raise APIError("MODEL_REVIEW_INVALID", "resource_profile必须包含资源数字")
    return copy.deepcopy(dict(value))


def _model(repo: Repository, owner: str, model_id: str) -> dict:
    row = repo.get("models", model_id, owner)
    if row:
        return row
    row = next((item for item in repo.list("models", owner) if item.get("model_id") == model_id), None)
    if not row:
        raise APIError("MODEL_NOT_FOUND", "工作流绑定的模型不存在", 404)
    return row


def _reviewed_model(model: dict) -> None:
    if not (model.get("review_approved") is True and model.get("license_accepted") is True):
        raise APIError("MODEL_NOT_REVIEWED", "模型尚未完成管理员来源与许可审查")


def _review_manifest(row: dict) -> tuple[dict, list[str], str, list[dict]]:
    try:
        manifest = validate_manifest(row.get("manifest"), row.get("api_graph"), row.get("object_info"))
    except (TypeError, ValueError) as exc:
        raise APIError("WORKFLOW_INVALID", str(exc)[:300]) from exc

    dependencies = manifest.get("dependencies")
    if not isinstance(dependencies, Mapping):
        raise APIError("REVIEW_INVALID", "manifest.dependencies缺失")
    commit = _fixed_commit(dependencies.get("comfyui_commit"), "dependencies.comfyui_commit")
    try:
        custom_nodes = _reviewed_custom_nodes(dependencies.get("custom_nodes"))
    except Exception as exc:
        raise APIError("REVIEW_INVALID", "custom_nodes 只允许已审核并锁定提交的仓库") from exc
    weights = dependencies.get("weight_files")
    if isinstance(weights, list) and any(
        isinstance(item, Mapping) and item.get("source") in {"civitai", "manual"}
        for item in weights
    ):
        raise APIError(
            "REVIEW_INVALID",
            "工作流包含 Civitai 或待重绑的模型权重，请先在「模型依赖」中准备并绑定模型，再点击「将当前绑定生成新草稿」后启用",
        )
    try:
        checked = _validate_weights({"weight_files": weights})
    except Exception as exc:
        raise APIError("REVIEW_INVALID", str(exc)[:300]) from exc
    for item in checked:
        target = item.get("target")
        if not isinstance(target, str) or not target or "\\" in target:
            raise APIError("REVIEW_INVALID", "weight target路径不安全")

    classes = set()
    for node in row.get("api_graph", {}).values():
        if not isinstance(node, Mapping) or not isinstance(node.get("class_type"), str):
            raise APIError("WORKFLOW_INVALID", "api_graph节点类型无效")
        class_type = node["class_type"]
        compact = class_type.lower().replace("_", "")
        if any(token in compact for token in _DANGEROUS):
            raise APIError("REVIEW_INVALID", f"危险节点类型禁止使用: {class_type}")
        classes.add(class_type)
    if not classes:
        raise APIError("WORKFLOW_INVALID", "api_graph不能为空")
    pinned_nodes = {(item["repo"], item["commit"]): item["dir"] for item in custom_nodes}
    for class_type, pin in _REQUIRED_CUSTOM_NODE_PINS.items():
        reviewed_dir = REVIEWED_CUSTOM_NODES.get(pin)
        if class_type in classes and pinned_nodes.get(pin) != reviewed_dir:
            raise APIError("REVIEW_INVALID", f"节点 {class_type} 必须绑定匹配的已审查custom node pin")
    return manifest, sorted(classes), commit, checked


def _reviewed_snapshot(row: dict) -> dict:
    manifest, allowlist, commit, weights = _review_manifest(row)
    snapshot_manifest = copy.deepcopy(manifest)
    # These top-level fields are the adapter contract.  Keep both commit names
    # during the transition so runtime_worker and older Comfy adapters agree.
    snapshot_manifest.update(
        {
            "state": "published",
            "reviewed": True,
            "node_allowlist": allowlist,
            "comfyui_commit": commit,
            "comfy_commit": commit,
            "custom_nodes": list(manifest["dependencies"].get("custom_nodes") or []),
            "weight_files": weights,
        }
    )
    return {
        "manifest": snapshot_manifest,
        "api_graph": copy.deepcopy(row["api_graph"]),
        "object_info": copy.deepcopy(row["object_info"]),
        "node_allowlist": allowlist,
        "comfyui_commit": commit,
        "weight_files": weights,
    }


def _workflow_digest(row: dict) -> str:
    """Recompute the digest from the stored artifact, never from client input."""
    try:
        return digest(
            {"manifest": row["manifest"], "api_graph": row["api_graph"], "object_info": row["object_info"]}
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise APIError("WORKFLOW_INVALID", "工作流artifact不完整，无法计算固定摘要") from exc


def _job_request(workflow_id: str, row: dict, body: Mapping) -> dict:
    """Build the exact persisted idempotency body without external validation."""
    return {
        "workflow_id": workflow_id,
        "project_id": body.get("project_id"),
        "model_id": row.get("model_id"),
        "workflow_key": row.get("workflow_key"),
        "workflow_version": row.get("workflow_version"),
        "params": copy.deepcopy(body.get("params", {})),
        "asset_ids": copy.deepcopy(body.get("asset_ids", [])),
        "spec": copy.deepcopy(body.get("spec")),
        "execution_policy": {
            "provider": "colab",
            "account_id": body.get("account_id"),
            "shutdown_after_job": True,
            "max_runtime_minutes": 5,
            "idle_grace_minutes": 0,
        },
    }


def _idempotency_key(request: Request) -> str:
    key = request.headers.get("Idempotency-Key")
    if not key or len(key) > 200:
        raise APIError("IDEMPOTENCY_REQUIRED", "真实冒烟必须提供固定Idempotency-Key")
    return key


def _spec_key(value):
    if not isinstance(value, Mapping):
        return None
    key = value.get("key")
    return key if isinstance(key, Mapping) else None


def _eligible_spec(specs, requested_key):
    if not isinstance(specs, list):
        raise APIError("GPU_UNAVAILABLE", "Colab规格响应无效")
    for spec in specs:
        if not isinstance(spec, Mapping) or spec.get("eligible") is not True:
            continue
        key = spec.get("key")
        if not isinstance(key, Mapping) or key.get("variant") not in {"GPU", "VARIANT_GPU"}:
            continue
        if dict(key) == dict(requested_key):
            return copy.deepcopy(dict(spec))
    raise APIError("GPU_UNAVAILABLE", "请求的GPU规格不是当前账户实际可用规格")


def _require_workflow_spec(manifest, requested_key):
    expected = manifest.get("compute_spec")
    if not isinstance(expected, Mapping):
        return  # Preserve explicit administrator smoke for legacy manifests.
    def normalized(key):
        return (
            str(key.get("variant", "")).upper().removeprefix("VARIANT_"),
            str(key.get("accelerator", "")).upper(),
            str(key.get("shape", "STANDARD")).upper().removeprefix("SHAPE_"),
        )
    if normalized(expected) != normalized(requested_key):
        raise APIError("WORKFLOW_GPU_SPEC_MISMATCH", "冒烟GPU必须匹配工作流声明的型号和内存规格，不会自动升级显卡")


def _runtime_cap(row: Mapping) -> int:
    """Return the server-enforced runtime ceiling for this workflow."""
    declared = (row.get("manifest") or {}).get("max_runtime_minutes", 5)
    if isinstance(declared, bool) or not isinstance(declared, int) or declared < 1:
        raise APIError("WORKFLOW_INVALID", "manifest.max_runtime_minutes必须是正整数")
    return declared


def activation_inputs(manifest: Mapping | None) -> list[dict]:
    """Image slots and other values the smoke cannot fill from a default.

    Optional images stay visible so the operator can attach them, but they are
    not required. Parameters that already have a default are omitted: the smoke
    resolver fills those, and asking again would look like the form wiped them.
    """
    parameters = manifest.get("parameters") if isinstance(manifest, Mapping) else None
    if not isinstance(parameters, Mapping):
        return []
    rows = []
    for name, spec in parameters.items():
        if not isinstance(spec, Mapping):
            continue
        kind = str(spec.get("type") or "string")
        optional_image = kind == "image" and spec.get("required") is False
        needs_value = "default" not in spec and (kind == "image" or spec.get("required") is True)
        if kind != "image" and not needs_value:
            continue
        rows.append(
            {
                "key": str(name),
                "label": str(spec.get("label") or name),
                "description": str(spec.get("description") or ""),
                "type": kind,
                "required": bool(needs_value and not optional_image),
            }
        )
    return rows


def _smoke_compatibility_signature(
    workflow_id: str,
    row: Mapping,
    model_revision: str | None,
    *,
    snapshot_parts: Mapping | None = None,
) -> str:
    """Fingerprint the reviewed artifact, not the user-supplied smoke params.

    The workflow id is deliberately only one part of this identity.  A draft
    can be replaced in place during the review flow, and old jobs may not have
    been stamped with this field at all, so the template digest, model
    revision, dependency pin and exact weight list are retained as well.
    """
    manifest = row.get("manifest")
    if not isinstance(manifest, Mapping):
        raise APIError("WORKFLOW_INVALID", "工作流artifact缺少manifest")
    dependencies = manifest.get("dependencies")
    if not isinstance(dependencies, Mapping):
        raise APIError("WORKFLOW_INVALID", "manifest.dependencies缺失")
    weights = dependencies.get("weight_files")
    if snapshot_parts and isinstance(snapshot_parts.get("weight_files"), list):
        weights = snapshot_parts["weight_files"]
    template_hash = row.get("sha256")
    if not isinstance(template_hash, str) or not template_hash:
        template_hash = _workflow_digest(dict(row))
    identity = {
        "workflow_id": workflow_id,
        "workflow_key": row.get("workflow_key"),
        "workflow_version": str(row.get("workflow_version")),
        "model_id": row.get("model_id"),
        "template_sha256": template_hash,
        "model_revision": model_revision,
        "comfyui_commit": dependencies.get("comfyui_commit"),
        "custom_nodes": copy.deepcopy(dependencies.get("custom_nodes") or []),
        "weight_files": copy.deepcopy(weights),
    }
    return digest(identity)


def _job_snapshot(job: Mapping) -> Mapping:
    """Return a safe immutable snapshot view for old and malformed jobs."""
    snapshot = job.get("snapshot")
    return snapshot if isinstance(snapshot, Mapping) else {}


def _job_smoke_signature(job: Mapping) -> str | None:
    """Read an explicit fingerprint, or reconstruct a complete legacy one."""
    direct = job.get("smoke_compatibility_signature")
    if isinstance(direct, str) and direct:
        return direct
    snapshot = _job_snapshot(job)
    stamped = snapshot.get("smoke_compatibility_signature")
    if isinstance(stamped, str) and stamped:
        return stamped
    return None


def _legacy_smoke_identity(job: Mapping) -> dict | None:
    """Extract every immutable identity field from a pre-fingerprint job."""
    snapshot = _job_snapshot(job)
    manifest = snapshot.get("manifest")
    manifest = manifest if isinstance(manifest, Mapping) else {}
    dependencies = snapshot.get("dependency_lock")
    dependencies = dependencies if isinstance(dependencies, Mapping) else {}
    manifest_dependencies = manifest.get("dependencies")
    manifest_dependencies = manifest_dependencies if isinstance(manifest_dependencies, Mapping) else {}

    def first(*values):
        for value in values:
            if value is not None:
                return copy.deepcopy(value)
        return None

    identity = {
        "workflow_id": first(job.get("target_workflow_id"), snapshot.get("target_workflow_id")),
        "workflow_key": first(job.get("workflow_key"), snapshot.get("workflow_key")),
        "workflow_version": first(job.get("workflow_version"), snapshot.get("workflow_version")),
        "model_id": first(job.get("target_model_id"), job.get("model_id"), snapshot.get("model_id")),
        "template_sha256": first(
            snapshot.get("workflow_sha256"), job.get("workflow_sha256"), snapshot.get("sha256")
        ),
        "model_revision": first(snapshot.get("model_revision"), job.get("model_revision")),
        "comfyui_commit": first(
            dependencies.get("comfyui_commit"),
            snapshot.get("comfyui_commit"),
            manifest.get("comfyui_commit"),
            manifest_dependencies.get("comfyui_commit"),
        ),
        "custom_nodes": first(
            snapshot.get("custom_nodes"),
            manifest.get("custom_nodes"),
            manifest_dependencies.get("custom_nodes"),
        ),
        "weight_files": first(
            dependencies.get("weight_files"),
            snapshot.get("weight_files"),
            manifest.get("weight_files"),
            manifest_dependencies.get("weight_files"),
        ),
    }
    if any(value is None for value in identity.values()):
        return None
    return identity


def _smoke_signatures(workflow_id: str, row: Mapping, model: Mapping) -> list[str]:
    """Fingerprints activation, acceptance and publish must all recognize.

    The queued smoke job is stamped from the reviewed weight list. Reading only
    the raw manifest misses that job, so the wizard looks idle and a second
    click is easy to mistake for a failed submit.
    """
    signatures: list[str] = []
    revision = model.get("revision")
    try:
        signatures.append(_smoke_compatibility_signature(workflow_id, row, revision))
    except APIError:
        pass
    if row.get("review_approved") is True:
        try:
            reviewed = _reviewed_snapshot(row)
            reviewed_signature = _smoke_compatibility_signature(
                workflow_id, row, revision, snapshot_parts=reviewed
            )
            if reviewed_signature not in signatures:
                signatures.append(reviewed_signature)
        except APIError:
            pass
    return signatures


def _job_matches_any_smoke(job: Mapping, workflow_id: str, row: Mapping, model: Mapping) -> bool:
    return any(
        _job_matches_smoke(job, workflow_id, row, model, signature)
        for signature in _smoke_signatures(workflow_id, row, model)
    )


def _job_matches_smoke(job: Mapping, workflow_id: str, row: Mapping, model: Mapping, current: str) -> bool:
    """Match current reviewed identity; an explicit mismatch is terminal."""
    if not isinstance(job, Mapping) or not isinstance(row, Mapping) or not isinstance(model, Mapping):
        return False
    if job.get("target_workflow_id") != workflow_id:
        return False
    if (job.get("target_model_id") or job.get("model_id")) != row.get("model_id"):
        return False
    if job.get("workflow_key") != row.get("workflow_key"):
        return False
    if str(job.get("workflow_version")) != str(row.get("workflow_version")):
        return False

    # A present fingerprint is authoritative.  Never let a stale explicit
    # fingerprint fall through to weaker legacy evidence.
    if job.get("smoke_compatibility_signature") is not None:
        return job.get("smoke_compatibility_signature") == current
    snapshot = _job_snapshot(job)
    if snapshot.get("smoke_compatibility_signature") is not None:
        return snapshot.get("smoke_compatibility_signature") == current

    # Legacy jobs remain reusable only when all immutable fields are present
    # and equal.  In particular, custom_nodes cannot be omitted from the
    # comparison or replaced by a truthiness-based default.
    legacy = _legacy_smoke_identity(job)
    if legacy is None:
        return False
    manifest = row.get("manifest")
    if not isinstance(manifest, Mapping):
        return False
    dependencies = manifest.get("dependencies")
    if not isinstance(dependencies, Mapping):
        return False
    current_identity = {
        "workflow_id": workflow_id,
        "workflow_key": row.get("workflow_key"),
        "workflow_version": row.get("workflow_version"),
        "model_id": row.get("model_id"),
        "template_sha256": row.get("sha256") or _workflow_digest(dict(row)),
        "model_revision": model.get("revision"),
        "comfyui_commit": dependencies.get("comfyui_commit"),
        "custom_nodes": copy.deepcopy(dependencies.get("custom_nodes")),
        "weight_files": copy.deepcopy(dependencies.get("weight_files")),
    }
    return legacy == current_identity


def _reusable_smoke_job(repo: Repository, owner: str, workflow_id: str, row: Mapping, model: Mapping, signature: str):
    for job in repo.list("jobs", owner):
        if job.get("state") not in {"queued", "running", "completed"}:
            continue
        if job.get("smoke_accepted") is True:
            continue
        if _job_matches_smoke(job, workflow_id, row, model, signature):
            return job
    return None


def _smoke_response(job: Mapping, request: Request, *, resumed: bool):
    return ok(
        {
            "id": job["id"],
            "job_id": job["id"],
            "state": job["state"],
            "estimated_units": job.get("estimated_units"),
            "resumed": resumed,
        },
        request,
        202,
    )


@router.post("/models/{model_id}/review")
def review_model(
    model_id: str, body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)
):  # noqa: B008
    _admin(user)
    _confirmed(body)
    if body.get("license_accepted") is not True:
        raise APIError("LICENSE_CONFIRMATION_REQUIRED", "必须明确接受模型许可", 403)
    repo = Repository(st)
    model = _model(repo, user["id"], model_id)
    model_patch: dict = {}
    if model.get("library_model_id"):
        from .gpu.model_library import ModelLibrary
        from .gpu.model_sources import ModelError
        lib = ModelLibrary(repo, user["id"])
        try:
            entry = lib.check_remote(model["library_model_id"])
            if entry.get("state") != "ready" or entry.get("sha256") != model.get("model_sha256"):
                raise ModelError("模型库文件身份已变化或未就绪，请重新注册工作流模型")
        except ModelError as exc:
            raise APIError("MODEL_REVIEW_INVALID", str(exc)) from exc
        finally:
            lib.close()
    elif model.get("model_sha256") and not (
        isinstance(model.get("revision"), str)
        and _COMMIT.fullmatch(model["revision"])
        and isinstance(model.get("source"), str)
        and (urlsplit(model["source"]).hostname or "").lower() in _SOURCE_HOSTS
    ):
        from .gpu.model_library import ModelLibrary
        from .gpu.model_sources import ModelError
        lib = ModelLibrary(repo, user["id"])
        try:
            expected_sha = str(model["model_sha256"]).lower()
            matched = next(
                (
                    item for item in lib.models()
                    if item.get("state") == "ready"
                    and isinstance(item.get("sha256"), str)
                    and item["sha256"].lower() == expected_sha
                ),
                None,
            )
            if not matched:
                raise ModelError("请先在工作流的「模型依赖」中下载并准备对应的模型库文件（SHA-256 匹配），再完成模型审查")
            entry = lib.check_remote(matched["id"])
            if entry.get("state") != "ready" or (entry.get("sha256") or "").lower() != expected_sha:
                raise ModelError("模型库文件身份已变化或未就绪，请重新准备对应模型")
            model_patch["library_model_id"] = matched["id"]
            if entry.get("drive_file_id"):
                model_patch["drive_file_id"] = entry["drive_file_id"]
        except ModelError as exc:
            raise APIError("MODEL_REVIEW_INVALID", str(exc)) from exc
        finally:
            lib.close()
    else:
        _source(model.get("source"))
        _fixed_commit(model.get("revision"), "revision")
    license_name = model.get("license")
    if not isinstance(license_name, str) or not license_name.strip():
        raise APIError("MODEL_REVIEW_INVALID", "模型license不能为空")
    _resource_profile(model.get("resource_profile"))
    reviewed = repo.update(
        "models",
        model["id"],
        {
            **model_patch,
            "review_approved": True,
            "license_accepted": True,
            "license_review_approved": True,
            "reviewed_by": user["id"],
            "reviewed_at": time.time(),
            # Review is not enablement.  Only accept-smoke may flip this bit.
            "enabled": False,
            "is_mock": False,
            "disabled_reason": "已通过人工审查，等待已授权真实GPU冒烟验收",
        },
        user["id"],
    )
    return ok(reviewed, request)


@router.post("/workflows/{workflow_id}/review")
def review_workflow(
    workflow_id: str, body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)
):  # noqa: B008
    _admin(user)
    _confirmed(body)
    repo = Repository(st)
    row = own(repo, "workflows", workflow_id, user)
    if row.get("state") not in {"draft", "validated"}:
        raise APIError("WORKFLOW_REVIEW_INVALID", "只能审查draft或validated工作流")
    model = _model(repo, user["id"], row.get("model_id"))
    _reviewed_model(model)
    manifest, allowlist, commit, weights = _review_manifest(row)
    reviewed_manifest = copy.deepcopy(manifest)
    reviewed_manifest.update(
        {
            "node_allowlist": allowlist,
            "comfyui_commit": commit,
            "comfy_commit": commit,
            "custom_nodes": list(manifest["dependencies"].get("custom_nodes") or []),
            "weight_files": weights,
        }
    )
    reviewed = repo.update(
        "workflows",
        workflow_id,
        {
            "manifest": reviewed_manifest,
            "sha256": digest(
                {
                    "manifest": reviewed_manifest,
                    "api_graph": row["api_graph"],
                    "object_info": row["object_info"],
                }
            ),
            "review_approved": True,
            "reviewed_by": user["id"],
            "reviewed_at": time.time(),
        },
        user["id"],
    )
    return ok(reviewed, request)


@router.post("/workflows/{workflow_id}/real-smoke")
def real_smoke(
    workflow_id: str, body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)
):  # noqa: B008
    _admin(user)
    _confirmed(body)
    if (
        not isinstance(body, dict)
        or not isinstance(body.get("project_id"), str)
        or not isinstance(body.get("params", {}), dict)
    ):
        raise APIError("VALIDATION_ERROR", "project_id和params格式无效")
    asset_ids = body.get("asset_ids", [])
    if (
        not isinstance(asset_ids, list)
        or len(asset_ids) > 10
        or any(not isinstance(value, str) for value in asset_ids)
    ):
        raise APIError("VALIDATION_ERROR", "asset_ids格式无效")
    retain_after_smoke = body.get("retain_after_smoke", False)
    if not isinstance(retain_after_smoke, bool):
        raise APIError("VALIDATION_ERROR", "retain_after_smoke必须为布尔值")
    raw_req_inst = body.get("requested_instance_id")
    if raw_req_inst is None or raw_req_inst == "":
        requested_instance_id = None
    elif isinstance(raw_req_inst, str) and raw_req_inst.strip():
        requested_instance_id = raw_req_inst.strip()
    else:
        raise APIError("VALIDATION_ERROR", "requested_instance_id格式无效")
    requested_key = _spec_key(body.get("spec"))
    if requested_key is None:
        raise APIError("GPU_SPEC_REQUIRED", "必须提供来自当前账户的spec.key")
    key = _idempotency_key(request)
    account_id = body.get("account_id")
    if not isinstance(account_id, str) or not account_id.strip():
        raise APIError("ACCOUNT_REQUIRED", "真实冒烟必须明确选择Google账号", 422)
    repo = Repository(st)
    _require_account(repo, user["id"], account_id)
    row = own(repo, "workflows", workflow_id, user)
    if row.get("state") not in {"draft", "validated"}:
        raise APIError("WORKFLOW_REVIEW_INVALID", "真实冒烟只能用于未发布的draft或validated工作流")
    _require_workflow_spec(row.get("manifest") or {}, requested_key)
    model = _model(repo, user["id"], row.get("model_id"))
    _reviewed_model(model)
    snapshot_parts = _reviewed_snapshot(row) if row.get("review_approved") is True else None
    if snapshot_parts is None:
        raise APIError("WORKFLOW_NOT_REVIEWED", "工作流尚未完成管理员审查")
    try:
        params = resolve_parameters(row["manifest"], body.get("params", {}))
    except (TypeError, ValueError) as exc:
        raise APIError("WORKFLOW_INVALID", str(exc)[:300]) from exc
    project = own(st, "projects", body["project_id"], user)
    bind_params, input_files = resolve_input_images(
        lambda aid: own(st, "assets", aid, user), project["id"], row["manifest"], params
    )
    try:
        bound = bind_parameters(row["manifest"], row["api_graph"], bind_params)
    except (TypeError, ValueError) as exc:
        raise APIError("WORKFLOW_INVALID", str(exc)[:300]) from exc
    for asset_id in asset_ids:
        own(st, "assets", asset_id, user)

    cfg = GpuConfig.from_env()
    runtime = _runtime_cap(row)
    raw = {
        "project_id": project["id"],
        "model_id": row["model_id"],
        "workflow_key": row["workflow_key"],
        "workflow_version": row["workflow_version"],
        "params": copy.deepcopy(body.get("params", {})),
        "asset_ids": copy.deepcopy(asset_ids),
        "loras": copy.deepcopy(body.get("loras") or []),
        "spec": copy.deepcopy(body["spec"]),
        "retain_after_smoke": retain_after_smoke,
        "requested_instance_id": requested_instance_id,
        "execution_policy": {
            "provider": "colab",
            "account_id": account_id,
            "lifecycle": "retain" if retain_after_smoke else "save",
            "shutdown_after_job": not retain_after_smoke,
            "retain_idle_minutes": 10 if retain_after_smoke else 0,
            "max_runtime_minutes": runtime,
            "idle_grace_minutes": 0,
        },
    }
    previous = next(
        (item for item in repo.list("jobs", user["id"]) if item.get("idempotency_key") == key), None
    )
    if previous:
        if previous.get("body_hash") != digest(raw):
            raise APIError("IDEMPOTENCY_CONFLICT", "相同幂等键不能用于不同冒烟参数", 409)
        return ok(
            {
                "id": previous["id"],
                "job_id": previous["id"],
                "state": previous["state"],
                "estimated_units": previous.get("estimated_units"),
            },
            request,
            202,
        )

    if not cfg.real_enabled:
        raise APIError("REAL_GPU_DISABLED", "GPU_REAL_ENABLED尚未启用")
    from .gpu.cu_guard import CuGuardService
    CuGuardService(repo).check(user["id"], "colab", account_id)
    google = GoogleService(repo, user["id"], account_id=account_id)
    if not google.status().get("colab", {}).get("connected"):
        raise APIError("GOOGLE_NOT_CONNECTED", "请先完成个人Colab OAuth授权")
    chosen_spec = _eligible_spec(
        GoogleColabProvider(repo, user["id"], account_id=account_id).list_specs(), requested_key
    )
    picked = mode_param(row["manifest"])
    shown, rate = estimate_units(repo, user["id"], row["workflow_key"], row["manifest"], chosen_spec.get("key"), runtime,
                                 mode=params.get(picked[0]) if picked else None)
    snapshot = {
        **snapshot_parts,
        "bound_graph": bound,
        "input_files": input_files,
        "resolved_params": params,
        "workflow_sha256": row.get("sha256"),
        "model_revision": model.get("revision"),
        "dependency_lock": {
            "comfyui_commit": snapshot_parts["comfyui_commit"],
            "weight_files": snapshot_parts["weight_files"],
        },
        "workflow_smoke": True,
        "system_smoke": False,
        "target_workflow_id": workflow_id,
        "target_model_id": row["model_id"],
    }
    loras = body.get("loras") or []
    if loras:
        # Smoke runs may only use the curated (pinned, SFW) list, never owner-uploaded files.
        from .gpu import loras as lora_registry
        from .main import _apply_loras

        curated_ids = {c["id"] for c in lora_registry.CURATED if row["workflow_key"] in c["workflows"]}
        if not isinstance(loras, list) or len(loras) > 3 or any(
            not isinstance(p, dict) or p.get("id") not in curated_ids for p in loras
        ):
            raise APIError("VALIDATION_ERROR", "冒烟只允许使用精选 LoRA")
        _apply_loras(repo, user, row, loras, snapshot, "colab")
    with st.dispatch_lock() as acquired:
        if not acquired:
            raise APIError("SCHEDULER_BUSY", "调度正在同步，请稍后用相同幂等键重试", 503, True)
        compatibility = _smoke_compatibility_signature(
            workflow_id,
            row,
            model.get("revision"),
            snapshot_parts=snapshot_parts,
        )
        reusable = _reusable_smoke_job(repo, user["id"], workflow_id, row, model, compatibility)
        if reusable:
            return _smoke_response(reusable, request, resumed=True)
        if requested_instance_id:
            with st.admission_lock(timeout=1.0) as adm_ok:
                if not adm_ok:
                    raise APIError("SCHEDULER_BUSY", "资源分配正在进行，请稍后使用相同幂等键重试", 503, True)
                _validate_requested_ephemeral_standby(
                    repo,
                    user["id"],
                    requested_instance_id,
                    account_id=account_id,
                    expected_spec_key=requested_key,
                    now=time.time(),
                )
        from .gpu.native import guard_generation

        guard_generation(repo, user["id"], raw["execution_policy"])
        job, created = repo.create_job(user["id"], raw, key)
        if created:
            job = repo.update(
                "jobs",
                job["id"],
                {
                    "project_id": project["id"],
                    "provider": "colab",
                    "instance_pool": "ephemeral",
                    "account_id": account_id,
                    "requested_account_id": account_id,
                    "requested_instance_id": requested_instance_id,
                    "retain_after_smoke": retain_after_smoke,
                    "generation_timeouts": bool(requested_instance_id or retain_after_smoke),
                    "execution_policy": raw["execution_policy"],
                    "model_id": row["model_id"],
                    "workflow_key": row["workflow_key"],
                    "workflow_version": row["workflow_version"],
                    "target_workflow_id": workflow_id,
                    "target_model_id": row["model_id"],
                    "snapshot": snapshot,
                    "params": copy.deepcopy(body.get("params", {})),
                    "resolved_seed": params.get("seed"),
                    "requested_spec": chosen_spec,
                    "max_runtime_minutes": runtime,
                    "estimated_units": shown,
                    "unit_rate": rate,
                    "workflow_smoke": True,
                    "system_smoke": False,
                    "smoke_compatibility_signature": compatibility,
                },
                user["id"],
            )
            smoke_msg = (
                f"已进入队列，最长 {runtime} 分钟，完成后保留 GPU 10 分钟"
                if retain_after_smoke
                else f"已进入队列，最长 {runtime} 分钟，完成后立即释放 GPU"
            )
            repo.event(
                job["id"],
                user["id"],
                "queued",
                smoke_msg,
            )
    return _smoke_response(job, request, resumed=not created)


def _prepare_account_ids(repo, owner_id, requested_account_id):
    """Bind cache preparation to an owned compute account and its Drive account."""
    from .gpu.accounts import AccountService

    accounts = AccountService(repo, owner_id)
    account_id = requested_account_id
    if account_id is None:
        choices = [row for row in accounts.list() if row.get("enabled", True)
                   and row.get("allow_allocation")
                   and row.get("authorization", {}).get("colab", {}).get("connected")]
        if len(choices) != 1:
            raise APIError("ACCOUNT_REQUIRED", "缓存准备需要明确选择Google账号", 422)
        account_id = choices[0]["account_id"]
    if not isinstance(account_id, str) or not account_id.strip():
        raise APIError("ACCOUNT_INVALID", "account_id必须为非空字符串", 422)
    account_id = account_id.strip()
    _require_account(repo, owner_id, account_id)
    if not GoogleService(repo, owner_id, account_id=account_id).status().get("colab", {}).get("connected"):
        raise APIError("GOOGLE_NOT_CONNECTED", "所选账号尚未完成Colab OAuth授权")
    drive_account_id = accounts.resolve_drive_account_id(account_id)
    if not GoogleService(repo, owner_id, account_id=drive_account_id).status().get("drive", {}).get("connected"):
        raise APIError("DRIVE_NOT_CONNECTED", "所选存储账号尚未完成Drive OAuth授权")
    return account_id, drive_account_id


@router.post("/workflows/{workflow_id}/prepare")
def prepare_workflow(
    workflow_id: str, body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)
):  # noqa: B008
    """Queue a CPU-only job that builds the env archive and fills the weight cache in Drive."""
    _admin(user)
    _confirmed(body)
    repo = Repository(st)
    row = own(repo, "workflows", workflow_id, user)
    if row.get("review_approved") is not True:
        raise APIError("WORKFLOW_NOT_REVIEWED", "工作流尚未完成管理员审查")
    cfg = GpuConfig.from_env()
    if not cfg.real_enabled:
        raise APIError("REAL_GPU_DISABLED", "GPU_REAL_ENABLED尚未启用")
    account_id, drive_account_id = _prepare_account_ids(repo, user["id"], body.get("account_id"))
    project_id = body.get("project_id")
    if not isinstance(project_id, str):
        raise APIError("VALIDATION_ERROR", "project_id必填")
    own(st, "projects", project_id, user)
    meta = {
        "workflow_sha256": row.get("sha256"),
        "model_id": row["model_id"],
        "workflow_key": row["workflow_key"],
        "workflow_version": row["workflow_version"],
        "account_id": account_id,
        "drive_account_id": drive_account_id,
    }
    key = prep_idempotency_key(row.get("sha256") or workflow_id, account_id + ":" + _idempotency_key(request))
    with st.dispatch_lock() as acquired:
        if not acquired:
            raise APIError("SCHEDULER_BUSY", "调度正在同步，请稍后用相同幂等键重试", 503, True)
        job = create_prep_job(repo, user["id"], project_id, _reviewed_snapshot(row), meta, key)
    return ok({"id": job["id"], "job_id": job["id"], "state": job["state"]}, request, 202)


@router.post("/loras/prepare")
def prepare_loras(body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)):  # noqa: B008
    """CPU-only: copy the curated H3 LoRAs from Hugging Face into the Drive model cache."""
    from .gpu import loras as lora_registry

    _admin(user)
    _confirmed(body)
    repo = Repository(st)
    rows = [w for w in repo.list("workflows", user["id"]) if w.get("workflow_key") in lora_registry.H3_WORKFLOWS
            and w.get("state") == "published" and w.get("review_approved") is True]
    if not rows:
        raise APIError("WORKFLOW_NOT_REVIEWED", "没有已发布的 H3 工作流")
    project_id = body.get("project_id")
    if not isinstance(project_id, str):
        raise APIError("VALIDATION_ERROR", "project_id必填")
    own(st, "projects", project_id, user)
    row = rows[0]
    snapshot = _reviewed_snapshot(row)
    curated = [lora_registry.curated_weight(c) for c in lora_registry.CURATED]
    snapshot["manifest"] = {**snapshot["manifest"], "weight_files": curated}
    snapshot["weight_files"] = curated
    meta = {"workflow_sha256": row.get("sha256"), "model_id": row["model_id"], "workflow_key": row["workflow_key"],
            "workflow_version": row["workflow_version"]}
    key = prep_idempotency_key("curated-loras", _idempotency_key(request))
    with st.dispatch_lock() as acquired:
        if not acquired:
            raise APIError("SCHEDULER_BUSY", "调度正在同步，请稍后用相同幂等键重试", 503, True)
        job = create_prep_job(repo, user["id"], project_id, snapshot, meta, key)
    return ok({"id": job["id"], "job_id": job["id"], "state": job["state"]}, request, 202)


def _validate_requested_ephemeral_standby(
    repo: Repository,
    owner_id: str,
    instance_id: str,
    *,
    account_id: str,
    expected_spec_key: dict,
    now: float,
) -> dict:
    """Fail-closed validation for explicit ephemeral standby binding (Spec §5.2)."""
    from .gpu.instances import POOL_EPHEMERAL, instance_pool_of
    from .gpu.specs import spec_key as _norm_spec_key

    msg = "冒烟域待命实例已失效或被占用，未自动新开 GPU，请刷新后确认是否新开实例"
    inst = repo.get("instances", instance_id, owner_id)
    if inst is None or instance_pool_of(inst) != POOL_EPHEMERAL:
        raise APIError("EPHEMERAL_INSTANCE_UNAVAILABLE", msg, 409)
    if (
        inst.get("state") not in {"ready", "retained_idle"}
        or inst.get("current_job_id") is not None
        or inst.get("manual_stop_requested")
        or inst.get("force_stop")
        or inst.get("runtime_deleted")
        or inst.get("cleanup_claim_token")
        or inst.get("account_id") != account_id
        or inst.get("provider") != "colab"
    ):
        raise APIError("EPHEMERAL_INSTANCE_UNAVAILABLE", msg, 409)
    deadlines = [float(inst[k]) for k in ("idle_deadline_at", "hard_deadline_at") if inst.get(k) is not None]
    if deadlines and now >= min(deadlines):
        raise APIError("EPHEMERAL_INSTANCE_UNAVAILABLE", msg, 409)
    lease_id = inst.get("lease_id")
    lease = repo.get("leases", lease_id, owner_id) if lease_id else None
    if (
        lease is None
        or lease.get("state") != "idle"
        or lease.get("current_job_id") is not None
        or lease.get("cleanup_claim_token")
        or lease.get("force_stop")
        or not lease.get("handle")
    ):
        raise APIError("EPHEMERAL_INSTANCE_UNAVAILABLE", msg, 409)
    lease_deadlines = [float(lease[k]) for k in ("idle_deadline_at", "deadline_epoch") if lease.get(k) is not None]
    if lease_deadlines and now >= min(lease_deadlines):
        raise APIError("EPHEMERAL_INSTANCE_UNAVAILABLE", msg, 409)
    actual_spec = inst.get("runtime_spec_reported") or inst.get("runtime_spec_requested")
    try:
        if not isinstance(actual_spec, Mapping) or _norm_spec_key(dict(actual_spec)) != _norm_spec_key(dict(expected_spec_key)):
            raise APIError("EPHEMERAL_INSTANCE_UNAVAILABLE", msg, 409)
    except ValueError as exc:
        raise APIError("EPHEMERAL_INSTANCE_UNAVAILABLE", msg, 409) from exc
    terminal = {"completed", "failed", "cancelled", "rejected"}
    for existing_job in repo.list("jobs", owner_id):
        if existing_job.get("state") in terminal:
            continue
        if (
            existing_job.get("instance_id") == instance_id
            or existing_job.get("requested_instance_id") == instance_id
        ):
            raise APIError("EPHEMERAL_INSTANCE_UNAVAILABLE", msg, 409)
    return inst


def _activation_instance_hints(repo: Repository, owner_id: str, manifest: Mapping | None, *, now: float) -> tuple[dict | None, list[dict]]:
    """Compute exact-spec-matching standby ephemeral card and transferable other-domain cards (Spec §5.2)."""
    from .gpu.instances import POOL_CREATION, POOL_EPHEMERAL, POOL_NATIVE, instance_pool_of
    from .gpu.specs import spec_key as _norm_spec_key
    from .gpu.transfer import _is_gpu_instance, _vm_allocated_at

    raw_spec = (manifest or {}).get("compute_spec") if isinstance(manifest, Mapping) else None
    if not isinstance(raw_spec, Mapping):
        return None, []
    try:
        target_spec = _norm_spec_key(dict(raw_spec))
    except ValueError:
        return None, []

    terminal = {"completed", "failed", "cancelled", "rejected"}
    busy_inst_ids: set[str] = set()
    for j in repo.list("jobs", owner_id):
        if j.get("state") in terminal:
            continue
        for key in ("instance_id", "requested_instance_id"):
            if j.get(key):
                busy_inst_ids.add(str(j[key]))

    standby_candidates: list[tuple[float, dict]] = []
    transferable_others: list[dict] = []
    leases_by_id = {ls["id"]: ls for ls in repo.list("leases", owner_id) if ls.get("id")}

    for inst in repo.list("instances", owner_id):
        if inst.get("state") not in {"ready", "retained_idle"}:
            continue
        if (
            inst.get("current_job_id") is not None
            or inst.get("id") in busy_inst_ids
            or inst.get("manual_stop_requested")
            or inst.get("force_stop")
            or inst.get("runtime_deleted")
            or inst.get("cleanup_claim_token")
            or not _is_gpu_instance(inst)
        ):
            continue
        actual_raw = inst.get("runtime_spec_reported") or inst.get("runtime_spec_requested")
        if not isinstance(actual_raw, Mapping):
            continue
        try:
            actual_spec = _norm_spec_key(dict(actual_raw))
        except ValueError:
            continue
        if actual_spec != target_spec:
            continue
        pool = instance_pool_of(inst)
        lease = leases_by_id.get(inst.get("lease_id")) if inst.get("lease_id") else None
        if inst.get("provider") == "colab":
            if (
                lease is None
                or lease.get("state") != "idle"
                or lease.get("current_job_id") is not None
                or lease.get("cleanup_claim_token")
                or lease.get("force_stop")
                or not lease.get("handle")
            ):
                continue
        is_native_ready = pool == POOL_NATIVE and bool((lease or {}).get("native_ready")) and not inst.get("native_bootstrap_failed")
        hard = inst.get("hard_deadline_at")
        if hard is not None and now >= float(hard):
            continue
        live_native_idle = None
        if is_native_ready:
            from .gpu import native as native_gpu

            live_native_idle = native_gpu.effective_idle_deadline(lease, instance=inst)
        idle_deadline = live_native_idle or inst.get("idle_deadline_at") or (lease or {}).get("idle_deadline_at")
        if not is_native_ready and idle_deadline is not None and now >= float(idle_deadline):
            continue
        if pool == POOL_EPHEMERAL:
            rem = max(0, int(round(float(idle_deadline) - now))) if idle_deadline is not None else None
            standby_candidates.append(
                (
                    float(idle_deadline or 0.0),
                    {
                        "instance_id": inst["id"],
                        "account_id": inst.get("account_id"),
                        "account_alias": inst.get("account_alias_snapshot"),
                        "spec": actual_spec,
                        "gpu_reported": inst.get("gpu_reported"),
                        "idle_deadline_at": idle_deadline,
                        "remaining_seconds": rem,
                    },
                )
            )
        elif pool in {POOL_CREATION, POOL_NATIVE}:
            baseline = inst.get("vm_baseline_digest") or (lease or {}).get("vm_baseline_digest")
            if inst.get("venv_isolated") is not True or not baseline:
                continue
            if pool == POOL_NATIVE and not inst.get("native_bootstrap_failed"):
                if inst.get("native_draining") or (lease or {}).get("native_draining"):
                    continue
            transferable_others.append(
                {
                    "instance_id": inst["id"],
                    "instance_pool": pool,
                    "account_id": inst.get("account_id"),
                    "account_alias": inst.get("account_alias_snapshot"),
                    "spec": actual_spec,
                    "gpu_reported": inst.get("gpu_reported"),
                    "idle_deadline_at": idle_deadline,
                    "vm_allocated_at": _vm_allocated_at(inst, lease),
                    "native_bootstrap_failed": bool(inst.get("native_bootstrap_failed")),
                }
            )

    standby_candidates.sort(key=lambda item: item[0], reverse=True)
    best_standby = standby_candidates[0][1] if standby_candidates else None
    return best_standby, transferable_others


def _job_gpu_diagnostics(job: dict, leases: list[dict]) -> dict | None:
    # Only server-persisted scheduler/provider evidence is accepted.  Never use
    # a diagnostic value from the request body or the accept endpoint body.
    # Strictly prefer diagnostics pinned onto the job itself over mutable lease fields (Spec §5.3(3)).
    candidates = [job.get(name) for name in ("gpu_diagnostics", "gpu_diagnostic", "runtime_diagnostics")]
    snapshot = job.get("snapshot") or {}
    candidates.extend(
        snapshot.get(name) for name in ("gpu_diagnostics", "gpu_diagnostic", "runtime_diagnostics")
    )
    for value in candidates:
        if isinstance(value, Mapping) and value:
            return copy.deepcopy(dict(value))
    # Legacy fallback only for untransferred leases still owned by this smoke job.
    for lease in leases:
        if lease.get("owner_job_id") != job.get("id") or int(lease.get("domain_generation") or 0) > 0:
            continue
        for name in ("gpu_diagnostics", "gpu_diagnostic", "runtime_diagnostics", "selected_spec"):
            value = lease.get(name)
            if isinstance(value, Mapping) and value:
                return copy.deepcopy(dict(value))
    return None


@router.post("/workflows/{workflow_id}/accept-smoke")
def accept_smoke(
    workflow_id: str, body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)
):  # noqa: B008
    _admin(user)
    _confirmed(body)
    if (
        not isinstance(body, dict)
        or set(body) - {"job_id", "confirmed"}
        or not isinstance(body.get("job_id"), str)
    ):
        raise APIError("INVALID_SMOKE_EVIDENCE", "只能提交服务器生成的job_id，不能提交客户端测试证据")
    repo = Repository(st)
    row = own(repo, "workflows", workflow_id, user)
    if row.get("state") not in {"draft", "validated"}:
        raise APIError("WORKFLOW_ALREADY_PUBLISHED", "已发布版本不能覆写冒烟验收")
    model = _model(repo, user["id"], row.get("model_id"))
    _reviewed_model(model)
    if row.get("review_approved") is not True:
        raise APIError("WORKFLOW_NOT_REVIEWED", "工作流尚未完成管理员审查")
    job = own(repo, "jobs", body["job_id"], user)
    snapshot = _job_snapshot(job)
    if (
        job.get("state") != "completed"
        or job.get("provider") != "colab"
        or snapshot.get("workflow_smoke") is not True
    ):
        raise APIError("SMOKE_NOT_COMPLETE", "只能验收已完成的服务器真实工作流冒烟任务")
    if job.get("target_workflow_id") != workflow_id or job.get("target_model_id") != row.get("model_id"):
        raise APIError("SMOKE_TARGET_MISMATCH", "冒烟任务目标与当前工作流/模型不匹配")
    if not _job_matches_any_smoke(job, workflow_id, row, model):
        raise APIError("SMOKE_TARGET_MISMATCH", "冒烟任务证据与当前工作流artifact不匹配")
    output_ids = job.get("output_asset_ids")
    if not isinstance(output_ids, list) or not output_ids:
        raise APIError("SMOKE_OUTPUT_INVALID", "真实冒烟没有可验收输出")
    for asset_id in output_ids:
        asset = own(st, "assets", asset_id, user)
        try:
            content = read_asset(asset)
        except (OSError, ValueError) as exc:
            raise APIError("SMOKE_OUTPUT_INVALID", "冒烟输出文件不可读取") from exc
        if asset.get("size") != len(content) or asset.get("sha256") != hashlib.sha256(content).hexdigest():
            raise APIError("SMOKE_OUTPUT_INVALID", "冒烟输出SHA校验失败")
    job_lease_id = job.get("lease_id")
    leases = []
    if job_lease_id:
        found_lease = repo.get("leases", job_lease_id, user["id"])
        if found_lease is not None:
            leases = [found_lease]
    if not leases:
        leases = [lease for lease in repo.list("leases", user["id"]) if lease.get("owner_job_id") == job["id"]]
    exec_id = job.get("execution_id")
    execution = repo.get("executions", exec_id, user["id"]) if exec_id else None
    execution_completed = bool(execution and execution.get("state") == "completed")

    def _lease_released_or_detached(lease: dict) -> bool:
        if lease.get("state") == "deleted":
            return True
        if not execution_completed:
            return False
        if lease.get("current_job_id") == job["id"]:
            return False
        if lease.get("state") == "active" and lease.get("owner_job_id") == job["id"]:
            return False
        inst_id = lease.get("instance_id") or job.get("instance_id")
        if inst_id:
            inst = repo.get("instances", inst_id, user["id"])
            if inst is not None and inst.get("current_job_id") == job["id"]:
                return False
        return True

    if not leases or not all(_lease_released_or_detached(lease) for lease in leases):
        raise APIError("GPU_NOT_RELEASED", "GPU租约尚未由服务器确认释放")
    diagnostics = _job_gpu_diagnostics(job, leases)
    requested = job.get("requested_spec")
    if not isinstance(requested, Mapping) or diagnostics is None:
        raise APIError("GPU_EVIDENCE_MISSING", "缺少服务器保存的实际GPU诊断证据")
    tested = {"requested_spec": copy.deepcopy(dict(requested)), "actual_gpu": diagnostics}
    updated_workflow = repo.update(
        "workflows",
        workflow_id,
        {
            "smoke_test_passed": True,
            "smoke_test_kind": "real",
            "smoke_job_id": job["id"],
        },
        user["id"],
    )
    # Keep accepted jobs out of future resume scans while preserving the
    # underlying scheduler state (completed).  Publishing remains a separate
    # registry gate and is not performed here.
    repo.update(
        "jobs",
        job["id"],
        {"smoke_accepted": True, "acceptance_state": "accepted"},
        user["id"],
    )
    repo.update(
        "models",
        model["id"],
        {
            "enabled": True,
            "disabled_reason": None,
            "tested_compute_specs": tested,
            "enabled_at": time.time(),
        },
        user["id"],
    )
    return ok(updated_workflow, request)


@router.get("/workflows/{workflow_id}/activation")
@router.post("/workflows/{workflow_id}/activation")
def workflow_activation(
    workflow_id: str,
    request: Request,
    user=Depends(current_user),
    st=Depends(get_store),
    body: dict | None = None,
):  # noqa: B008
    """Everything the web wizard needs to walk a built-in draft to a usable workflow.

    Read-only: it never approves, allocates GPU or writes evidence.  The wizard still calls
    the individual reviewed endpoints, one human confirmation at a time.
    """
    _admin(user)
    repo = Repository(st)
    account_id = None
    if body is not None:
        if not isinstance(body, dict) or not isinstance(body.get("account_id"), str) or not body["account_id"]:
            raise APIError("ACCOUNT_REQUIRED", "工作流启用必须明确选择Google账号", 422)
        account_id = body["account_id"]
        _require_account(repo, user["id"], account_id)
        google = GoogleService(repo, user["id"], account_id=account_id)
        if not google.status().get("colab", {}).get("connected"):
            raise APIError("GOOGLE_NOT_CONNECTED", "所选Google账号尚未完成Colab授权", 409)
    row = own(repo, "workflows", workflow_id, user)
    model = next((item for item in repo.list("models", user["id"]) if item.get("model_id") == row.get("model_id")), None)
    cfg = GpuConfig.from_env()
    try:
        runtime_cap = _runtime_cap(row)
    except APIError as exc:
        runtime_cap = None
        # Keep activation read-only and useful for a malformed draft; the
        # write endpoint will return the same validation error before charging.
        runtime_error = exc.message
    else:
        runtime_error = None
    jobs = [
        job
        for job in repo.list("jobs", user["id"])
        if _job_snapshot(job).get("workflow_smoke") is True
        and model is not None
        and _job_matches_any_smoke(job, workflow_id, row, model)
    ]

    def job_priority(job):
        accepted = (
            job.get("state") == "completed"
            and bool(job.get("smoke_accepted"))
            and row.get("smoke_job_id") == job.get("id")
            and row.get("smoke_test_kind") == "real"
            and row.get("smoke_test_passed") is True
        )
        states = {"completed": 1, "running": 2, "queued": 3, "failed": 4, "cancelled": 5}
        return 0 if accepted else states.get(job.get("state"), 6)

    # Prefer a successful pending artifact over a newer failed retry.  The
    # second stable sort keeps the newest job within the same lifecycle state.
    jobs.sort(key=lambda job: (job.get("created_at") or "", str(job.get("id"))), reverse=True)
    jobs.sort(key=job_priority)
    smoke = jobs[0] if jobs else None
    smoke_state = (smoke or {}).get("state")
    published = row.get("state") == "published"
    validated = row.get("state") in {"validated", "published"}
    model_reviewed = bool(model and model.get("review_approved") and model.get("license_accepted"))
    reviewed = bool(row.get("review_approved"))
    smoke_completed = smoke_state == "completed"
    smoke_accepted = bool(
        smoke
        and smoke_completed
        and row.get("smoke_test_passed") is True
        and row.get("smoke_test_kind") == "real"
        and row.get("smoke_job_id") == smoke.get("id")
        and (smoke.get("smoke_accepted") is True or row.get("smoke_job_id") == smoke.get("id"))
    )
    # The workflow flag is only meaningful when it points at this current,
    # identity-matched completed job; an old flag cannot make activation done.
    smoke_done = smoke_accepted
    smoke_acceptance_state = None
    if smoke is not None:
        if smoke_accepted:
            smoke_acceptance_state = "published" if published else "accepted"
        elif smoke_completed:
            smoke_acceptance_state = "pending"
        elif smoke_state in {"queued", "running"}:
            smoke_acceptance_state = "pending"
        else:
            smoke_acceptance_state = "failed"
    blockers = []
    if model is None:
        blockers.append("找不到该工作流绑定的模型记录")
    else:
        has_source_or_identity = bool(
            (isinstance(model.get("source"), str) and model.get("source").strip())
            or model.get("library_model_id")
            or model.get("model_sha256")
        )
        if not (has_source_or_identity and model.get("license")):
            blockers.append("模型缺少来源或许可字段，无法完成审查")
    raw_weights = (
        ((row.get("manifest") or {}).get("dependencies") or {}).get("weight_files")
        or (row.get("manifest") or {}).get("weight_files")
        or []
    )
    if isinstance(raw_weights, list) and any(
        isinstance(item, Mapping) and item.get("source") in {"civitai", "manual"}
        for item in raw_weights
    ):
        blockers.append("工作流包含 Civitai 或待重绑的模型权重，请先展开「模型依赖」准备并绑定模型，再点击「将当前绑定生成新草稿」后启用")
    if not cfg.real_enabled:
        blockers.append("服务端未开启真实 GPU（GPU_REAL_ENABLED）")
    if runtime_error:
        blockers.append(runtime_error)

    steps = [
        {"key": "validate", "label": "校验模板", "done": validated},
        {"key": "model_review", "label": "确认模型来源与许可", "done": model_reviewed},
        {"key": "workflow_review", "label": "锁定权重与节点", "done": reviewed},
        {
            "key": "real_smoke",
            "label": "真实 GPU 冒烟" if runtime_cap is None else f"真实 GPU 冒烟（≤{runtime_cap} 分钟）",
            "done": smoke_done or smoke_completed,
            "running": smoke_state in {"queued", "running"},
            "failed": smoke_state == "failed",
        },
        {"key": "accept_smoke", "label": "验收冒烟结果", "done": smoke_done},
        {"key": "publish", "label": "发布并设为当前", "done": published},
    ]
    for step in steps:
        if not step["done"] and not step.get("running") and not step.get("failed"):
            step["next"] = True
            break
    standby_ephemeral_instance, transferable_other_instances = _activation_instance_hints(
        repo, user["id"], row.get("manifest"), now=time.time()
    )
    return ok(
        {
            "workflow": {
                "id": row["id"],
                "workflow_key": row.get("workflow_key"),
                "version": row.get("workflow_version"),
                "state": row.get("state"),
                "active": row.get("active"),
                "account_id": account_id,
            },
            "model": None
            if model is None
            else {
                "model_id": model.get("model_id"),
                "review_approved": bool(model.get("review_approved")),
                "license_accepted": bool(model.get("license_accepted")),
                "enabled": bool(model.get("enabled")),
                "license": model.get("license"),
                "source": model.get("source"),
            },
            "steps": steps,
            "smoke": None
            if smoke is None
            else {
                "job_id": smoke.get("id"),
                "project_id": smoke.get("project_id"),
                "state": smoke_state,
                "error_code": smoke.get("error_code"),
                "error_message": smoke.get("error_message"),
                "acceptance_state": smoke_acceptance_state,
                "max_runtime_minutes": smoke.get("max_runtime_minutes") or runtime_cap,
                "reusable": bool(smoke_state in {"queued", "running", "completed"} and not smoke_accepted),
            },
            "required_inputs": activation_inputs(row.get("manifest")),
            "smoke_job_completed": smoke_completed,
            "smoke_kind": row.get("smoke_test_kind"),
            "real_enabled": bool(cfg.real_enabled),
            "max_runtime_minutes": runtime_cap,
            "blockers": blockers,
            "done": published and bool(row.get("active")) and smoke_done,
            "account_id": account_id,
            "standby_ephemeral_instance": standby_ephemeral_instance,
            "transferable_other_instances": transferable_other_instances,
        },
        request,
    )


__all__ = ["accept_smoke", "real_smoke", "review_model", "review_workflow", "router", "workflow_activation"]
