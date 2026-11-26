"""Portable Studio workflow bundle export, model metadata sanitization, and version conflict helpers."""

from __future__ import annotations

import copy
import re
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlsplit

from .model_library import external_source
from .model_sources import CIVITAI_HOSTS, SHA, ModelError, https_url, normalize_hf_repo_type

_SEMVER_RE = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:(-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?"
    r"(?:(\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$"
)
_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_WORKFLOW_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$")

_REVIEW_DERIVED_MANIFEST_KEYS = (
    "node_allowlist",
    "comfyui_commit",
    "comfy_commit",
    "custom_nodes",
    "weight_files",
    "state",
    "reviewed",
)

_EXPORTABLE_MODEL_FIELDS = (
    "model_id",
    "display_name",
    "revision",
    "source",
    "license",
    "modality",
    "resource_profile",
    "variants",
)

_CIVITAI_IDENTITY_KEYS = ("model_id", "version_id", "file_id")
_HF_IDENTITY_KEYS = ("repo_type", "repo_id", "revision", "filename")


def _safe_persisted_url(raw: Any) -> str | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return https_url(raw.strip(), persisted=True)
    except ModelError:
        return None


def _is_civitai_url(url: str | None) -> bool:
    if not url:
        return False
    host = (urlsplit(url).hostname or "").lower()
    return any(host == domain or host.endswith(f".{domain}") for domain in CIVITAI_HOSTS)


def _civitai_url_from_identity(identity: dict[str, Any] | None) -> str | None:
    if not isinstance(identity, dict):
        return None
    model_id = identity.get("model_id")
    version_id = identity.get("version_id")
    if model_id is not None and str(model_id).isdigit():
        base = f"https://civitai.com/models/{model_id}"
        if version_id is not None and str(version_id).isdigit():
            return f"{base}?modelVersionId={version_id}"
        return base
    return None


def _safe_source_identity(source_kind: str | None, raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    if source_kind == "civitai":
        return {k: raw[k] for k in _CIVITAI_IDENTITY_KEYS if k in raw and raw[k] is not None}
    if source_kind == "huggingface":
        cleaned = {k: raw[k] for k in _HF_IDENTITY_KEYS if k in raw and raw[k] is not None}
        if cleaned.get("repo_type") in {None, "", "model", "models"}:
            cleaned.pop("repo_type", None)
        elif "repo_type" in cleaned:
            try:
                cleaned["repo_type"] = normalize_hf_repo_type(cleaned["repo_type"])
            except ModelError:
                cleaned.pop("repo_type", None)
        return cleaned
    return {}


def _find_library_match(weight: dict[str, Any], lib_models: list[dict[str, Any]]) -> dict[str, Any] | None:
    drive_id = weight.get("drive_file_id")
    sha = weight.get("sha256")
    sha_lower = sha.lower() if isinstance(sha, str) and SHA.fullmatch(sha) else None
    target = str(weight.get("target") or "")
    target_name = PurePosixPath(target).name if target else str(weight.get("filename") or "")

    def name_matches(row: dict[str, Any]) -> bool:
        if not target_name:
            return False
        return row.get("filename") == target_name or (
            isinstance(row.get("relative_path"), str) and target.endswith("/" + row["relative_path"])
        )

    if isinstance(drive_id, str) and drive_id:
        by_drive = [m for m in lib_models if m.get("drive_file_id") == drive_id]
        if by_drive:
            exact = [m for m in by_drive if name_matches(m)]
            return exact[0] if exact else by_drive[0]

    if sha_lower:
        by_sha = [
            m for m in lib_models
            if (m.get("sha256") or m.get("expected_sha256") or "").lower() == sha_lower
        ]
        if by_sha:
            exact = [m for m in by_sha if name_matches(m)]
            return exact[0] if exact else by_sha[0]

    return None


def _library_public_url(row: dict[str, Any] | None) -> str | None:
    if not isinstance(row, dict):
        return None
    ext = external_source(row)
    candidates = [
        row.get("source_url"),
        row.get("original_source_url"),
        ext.get("url") if isinstance(ext, dict) else None,
        ext.get("download_url") if isinstance(ext, dict) else None,
        row.get("download_url"),
    ]
    for candidate in candidates:
        cleaned = _safe_persisted_url(candidate)
        if cleaned:
            return cleaned
    return None


def _sanitize_weight_entry(
    weight: dict[str, Any],
    lib_models: list[dict[str, Any]],
) -> tuple[dict[str, Any], str, bool]:
    """Convert one weight entry into a portable, Drive-ID-free representation.

    Returns ``(clean_weight, bucket, stripped_drive_id)`` where ``bucket`` is one of:
    ``"huggingface"``, ``"restored_huggingface"``, ``"civitai"``, or ``"manual"``.
    """
    raw = copy.deepcopy(weight)
    had_drive_id = bool(raw.get("drive_file_id") or raw.get("source") == "drive")
    lib_row = _find_library_match(raw, lib_models)

    target = str(raw.get("target") or "")
    base_name = PurePosixPath(target).name if target else str(raw.get("filename") or "")
    sha = raw.get("sha256")
    if not (isinstance(sha, str) and SHA.fullmatch(sha)) and lib_row:
        sha = lib_row.get("sha256") or lib_row.get("expected_sha256")
    sha_clean = sha.lower() if isinstance(sha, str) and SHA.fullmatch(sha) else None

    size = raw.get("size")
    if (isinstance(size, bool) or not isinstance(size, int) or size <= 0) and lib_row:
        cand_size = lib_row.get("size") or lib_row.get("expected_size")
        size = cand_size if isinstance(cand_size, int) and not isinstance(cand_size, bool) and cand_size > 0 else None

    display_name = (
        str(raw.get("display_name") or "").strip()
        or str((lib_row or {}).get("display_name") or "").strip()
        or base_name
    )
    filename = str((lib_row or {}).get("filename") or raw.get("filename") or base_name).strip()
    architecture = str(raw.get("architecture") or (lib_row or {}).get("architecture") or "").strip()
    if architecture.lower() == "unknown":
        architecture = ""

    raw_source = raw.get("source")

    # 1. Existing pinned Hugging Face weight
    if raw_source in {None, "huggingface"} and all(
        isinstance(raw.get(k), str) and raw.get(k) for k in ("repo_id", "filename", "revision")
    ):
        entry: dict[str, Any] = {
            "repo_id": raw["repo_id"],
            "filename": raw["filename"],
            "revision": raw["revision"],
            "target": target or raw["filename"],
        }
        if raw.get("repo_type") and raw.get("repo_type") not in {"model", "models"}:
            try:
                entry["repo_type"] = normalize_hf_repo_type(raw["repo_type"])
            except ModelError:
                pass
        if sha_clean:
            entry["sha256"] = sha_clean
        if size is not None:
            entry["size"] = size
        if display_name and display_name != base_name:
            entry["display_name"] = display_name
        if architecture:
            entry["architecture"] = architecture
        return entry, "huggingface", False

    # 2. Restore pinned Hugging Face source from exporter's model library
    if lib_row and lib_row.get("source") == "huggingface":
        ident = _safe_source_identity("huggingface", lib_row.get("source_identity"))
        if (
            isinstance(ident.get("repo_id"), str)
            and isinstance(ident.get("filename"), str)
            and isinstance(ident.get("revision"), str)
            and _COMMIT_RE.fullmatch(ident["revision"])
        ):
            entry = {
                "repo_id": ident["repo_id"],
                "filename": ident["filename"],
                "revision": ident["revision"],
                "target": target or filename or ident["filename"],
            }
            if ident.get("repo_type"):
                entry["repo_type"] = ident["repo_type"]
            if sha_clean:
                entry["sha256"] = sha_clean
            if size is not None:
                entry["size"] = size
            if display_name and display_name != base_name:
                entry["display_name"] = display_name
            if architecture:
                entry["architecture"] = architecture
            return entry, "restored_huggingface", had_drive_id

    # 3. Civitai source (direct in weight, from model library, or bound external_source)
    raw_url = _safe_persisted_url(raw.get("source_url") or raw.get("url") or raw.get("download_url"))
    lib_url = _library_public_url(lib_row)
    is_civitai = (
        raw_source == "civitai"
        or (lib_row is not None and lib_row.get("source") == "civitai")
        or _is_civitai_url(raw_url)
        or _is_civitai_url(lib_url)
    )
    if is_civitai:
        identity = _safe_source_identity(
            "civitai",
            raw.get("source_identity") or (lib_row or {}).get("source_identity"),
        )
        civitai_url = (
            (raw_url if _is_civitai_url(raw_url) else None)
            or (lib_url if _is_civitai_url(lib_url) else None)
            or _civitai_url_from_identity(identity)
            or raw_url
            or lib_url
        )
        entry = {
            "source": "civitai",
            "target": target or filename,
            "filename": filename or base_name,
            "display_name": display_name or filename or base_name,
        }
        if civitai_url:
            entry["source_url"] = civitai_url
        if identity:
            entry["source_identity"] = identity
        if sha_clean:
            entry["sha256"] = sha_clean
        if size is not None:
            entry["size"] = size
        version_name = str(raw.get("version_name") or (lib_row or {}).get("version_name") or "").strip()
        if version_name:
            entry["version_name"] = version_name
        if architecture:
            entry["architecture"] = architecture
        return entry, "civitai", had_drive_id

    # 4. Private Drive / manual / other non-HF weights -> strip drive_file_id and preserve lookup clues
    entry = {
        "source": "manual",
        "target": target or filename,
        "filename": filename or base_name,
        "display_name": display_name or filename or base_name,
    }
    hint_url = raw_url or lib_url
    if hint_url:
        entry["source_url"] = hint_url
    if sha_clean:
        entry["sha256"] = sha_clean
    if size is not None:
        entry["size"] = size
    if architecture:
        entry["architecture"] = architecture
    return entry, "manual", had_drive_id


def _sanitize_exported_model(
    model_row: dict[str, Any] | None,
    lib_models: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not isinstance(model_row, dict) or not model_row.get("model_id"):
        return None
    exported: dict[str, Any] = {}
    for field in _EXPORTABLE_MODEL_FIELDS:
        val = model_row.get(field)
        if val is not None and val != "":
            exported[field] = copy.deepcopy(val)

    # Enrich from exporter's model_library if this model family came from a library entry,
    # without ever copying drive_file_id or library_model_id.
    lib_row = None
    if model_row.get("library_model_id"):
        lib_row = next((m for m in lib_models if m.get("id") == model_row["library_model_id"]), None)
    if lib_row is None and model_row.get("model_sha256"):
        sha_low = str(model_row["model_sha256"]).lower()
        lib_row = next((m for m in lib_models if (m.get("sha256") or "").lower() == sha_low), None)

    if exported.get("source"):
        safe_src = _safe_persisted_url(exported["source"])
        if safe_src:
            exported["source"] = safe_src
        else:
            exported.pop("source", None)
    if not exported.get("source") and lib_row:
        lib_url = _library_public_url(lib_row)
        if lib_url:
            exported["source"] = lib_url

    source_kind = model_row.get("source_kind") or (lib_row or {}).get("source")
    if source_kind in {"huggingface", "civitai", "https"}:
        exported["source_kind"] = source_kind
    elif _is_civitai_url(exported.get("source")):
        exported["source_kind"] = "civitai"

    ident = _safe_source_identity(
        exported.get("source_kind"),
        model_row.get("source_identity") or (lib_row or {}).get("source_identity"),
    )
    if ident:
        exported["source_identity"] = ident

    sha = model_row.get("model_sha256") or (lib_row or {}).get("sha256")
    if isinstance(sha, str) and SHA.fullmatch(sha):
        exported["model_sha256"] = sha.lower()

    if not exported.get("display_name"):
        exported["display_name"] = str((lib_row or {}).get("display_name") or exported["model_id"])

    return exported


def sanitize_import_model(raw_model: Any, model_id: str) -> dict[str, Any] | None:
    """Build a strictly unreviewed, disabled models-table payload from an imported bundle."""
    if not isinstance(raw_model, dict) or not isinstance(model_id, str) or not model_id.strip():
        return None
    if model_id.strip() == "unconfigured":
        return None
    payload: dict[str, Any] = {"model_id": model_id.strip()}
    for field in _EXPORTABLE_MODEL_FIELDS:
        if field == "model_id":
            continue
        val = raw_model.get(field)
        if val is None or val == "":
            continue
        if field == "source":
            safe_src = _safe_persisted_url(val)
            if safe_src:
                payload["source"] = safe_src
        elif field == "display_name":
            payload["display_name"] = str(val)[:300]
        elif field == "license":
            payload["license"] = str(val)[:200]
        elif field == "revision":
            if isinstance(val, str) and len(val) <= 128:
                payload["revision"] = val.strip()
        elif field == "modality":
            if isinstance(val, str) and len(val) <= 64:
                payload["modality"] = val.strip()
        elif field == "resource_profile":
            if isinstance(val, dict):
                payload["resource_profile"] = copy.deepcopy(val)
        elif field == "variants":
            if isinstance(val, list):
                payload["variants"] = [str(v)[:200] for v in val[:20]]

    payload.setdefault("display_name", model_id.strip())

    source_kind = raw_model.get("source_kind")
    if source_kind in {"huggingface", "civitai", "https"}:
        payload["source_kind"] = source_kind
    elif _is_civitai_url(payload.get("source")):
        payload["source_kind"] = "civitai"

    ident = _safe_source_identity(payload.get("source_kind"), raw_model.get("source_identity"))
    if ident:
        payload["source_identity"] = ident

    sha = raw_model.get("model_sha256")
    if isinstance(sha, str) and SHA.fullmatch(sha):
        payload["model_sha256"] = sha.lower()

    payload.update(
        enabled=False,
        is_mock=False,
        review_approved=False,
        license_accepted=False,
        license_review_approved=False,
        disabled_reason="导入工作流自动登记，待管理员来源审查、许可验证和真实 GPU 冒烟测试",
    )
    return payload


def summarize_portable_weights(weights: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Summarize portable weight sources in a workflow manifest for UI hints."""
    hf = civitai = manual = drive = 0
    rebind_models: list[dict[str, Any]] = []
    for item in weights or []:
        if not isinstance(item, dict):
            continue
        source = item.get("source")
        if source == "civitai":
            civitai += 1
            rebind_models.append({
                "target": item.get("target") or item.get("filename") or "",
                "display_name": item.get("display_name") or item.get("filename") or item.get("target") or "",
                "filename": item.get("filename") or PurePosixPath(str(item.get("target") or "")).name,
                "source": "civitai",
                **({"source_url": item["source_url"]} if item.get("source_url") else {}),
                **({"sha256": item["sha256"]} if item.get("sha256") else {}),
            })
        elif source == "manual":
            manual += 1
            rebind_models.append({
                "target": item.get("target") or item.get("filename") or "",
                "display_name": item.get("display_name") or item.get("filename") or item.get("target") or "",
                "filename": item.get("filename") or PurePosixPath(str(item.get("target") or "")).name,
                "source": "manual",
                **({"source_url": item["source_url"]} if item.get("source_url") else {}),
                **({"sha256": item["sha256"]} if item.get("sha256") else {}),
            })
        elif source == "drive":
            drive += 1
        elif source in {None, "huggingface"} and all(item.get(k) for k in ("repo_id", "filename", "revision")):
            hf += 1
    return {
        "total": hf + civitai + manual + drive,
        "huggingface": hf,
        "civitai": civitai,
        "manual": manual,
        "drive": drive,
        "requires_rebind": civitai + manual > 0,
        "rebind_models": rebind_models,
    }


def build_workflow_export(repo, owner_id: str, row: dict[str, Any]) -> dict[str, Any]:
    """Build a sanitized, portable Studio Workflow Bundle and export summary for a published workflow."""
    if row.get("state") != "published":
        raise ValueError("仅支持导出已发布的工作流版本")

    lib_models = repo.list("model_library", owner_id)
    model_row = next(
        (m for m in repo.list("models", owner_id) if m.get("model_id") == row.get("model_id")),
        None,
    )

    manifest = copy.deepcopy(row.get("manifest") or {})
    raw_weights = (
        (manifest.get("dependencies") or {}).get("weight_files")
        or manifest.get("weight_files")
        or []
    )
    for key in _REVIEW_DERIVED_MANIFEST_KEYS:
        manifest.pop(key, None)

    workflow_version = row.get("workflow_version")
    if isinstance(workflow_version, str) and _SEMVER_RE.fullmatch(workflow_version):
        export_version = workflow_version
        manifest["version"] = export_version
    elif isinstance(workflow_version, int) and not isinstance(workflow_version, bool) and workflow_version > 0:
        export_version = f"{workflow_version}.0.0"
        manifest["version"] = export_version
    else:
        existing_ver = manifest.get("version")
        export_version = existing_ver if isinstance(existing_ver, str) and _SEMVER_RE.fullmatch(existing_ver) else "1.0.0"
        manifest["version"] = export_version

    manifest["workflow_key"] = row["workflow_key"]
    manifest["model_id"] = row["model_id"]

    clean_weights: list[dict[str, Any]] = []
    hf_count = restored_hf_count = civitai_count = manual_count = stripped_drive = 0
    rebind_models: list[dict[str, Any]] = []

    for item in raw_weights:
        if not isinstance(item, dict):
            continue
        cleaned, bucket, had_drive = _sanitize_weight_entry(item, lib_models)
        clean_weights.append(cleaned)
        if had_drive:
            stripped_drive += 1
        if bucket == "huggingface":
            hf_count += 1
        elif bucket == "restored_huggingface":
            hf_count += 1
            restored_hf_count += 1
        elif bucket == "civitai":
            civitai_count += 1
            rebind_models.append({
                "target": cleaned.get("target") or "",
                "display_name": cleaned.get("display_name") or cleaned.get("filename") or "",
                "filename": cleaned.get("filename") or "",
                "source": "civitai",
                **({"source_url": cleaned["source_url"]} if cleaned.get("source_url") else {}),
                **({"sha256": cleaned["sha256"]} if cleaned.get("sha256") else {}),
            })
        else:
            manual_count += 1
            rebind_models.append({
                "target": cleaned.get("target") or "",
                "display_name": cleaned.get("display_name") or cleaned.get("filename") or "",
                "filename": cleaned.get("filename") or "",
                "source": "manual",
                **({"source_url": cleaned["source_url"]} if cleaned.get("source_url") else {}),
                **({"sha256": cleaned["sha256"]} if cleaned.get("sha256") else {}),
            })

    if isinstance(manifest.get("dependencies"), dict) or clean_weights:
        deps = copy.deepcopy(manifest.get("dependencies") or {})
        deps["weight_files"] = clean_weights
        manifest["dependencies"] = deps

    exported_model = _sanitize_exported_model(model_row, lib_models)

    bundle: dict[str, Any] = {
        "workflow_key": row["workflow_key"],
        "model_id": row["model_id"],
        **({"model": exported_model} if exported_model else {}),
        "manifest": manifest,
        "api_graph": copy.deepcopy(row.get("api_graph") or {}),
        "object_info": copy.deepcopy(row.get("object_info") or {}),
    }
    editable_graph = row.get("editable_graph")
    if isinstance(editable_graph, dict) and editable_graph and editable_graph != bundle["api_graph"]:
        bundle["editable_graph"] = copy.deepcopy(editable_graph)

    warnings: list[str] = []
    if stripped_drive > 0:
        warnings.append(f"已自动移除 {stripped_drive} 个个人 Google Drive 文件 ID，避免泄露私有路径。")
    if civitai_count > 0:
        warnings.append(
            f"包含 {civitai_count} 个 Civitai 来源模型（已保留模型名称、页面链接与 SHA-256），"
            "接收方导入后需在「模型依赖」一键准备并生成新草稿再启用。"
        )
    if manual_count > 0:
        warnings.append(
            f"包含 {manual_count} 个私有/非公开来源模型（已保留模型名称与 SHA-256），"
            "接收方导入后需自行在模型库查找/上传并重新绑定后生成新草稿。"
        )

    safe_key = _SAFE_FILENAME_RE.sub("-", str(row["workflow_key"])).strip("-.") or "workflow"
    safe_ver = _SAFE_FILENAME_RE.sub("-", str(export_version)).strip("-.") or "1.0.0"
    filename = f"{safe_key}-v{safe_ver}.json"

    return {
        "filename": filename,
        "bundle": bundle,
        "summary": {
            "workflow_key": row["workflow_key"],
            "workflow_version": export_version,
            "model_id": row["model_id"],
            "has_model_metadata": exported_model is not None,
            "total_weights": len(clean_weights),
            "huggingface_weights": hf_count,
            "restored_huggingface_weights": restored_hf_count,
            "civitai_weights": civitai_count,
            "manual_rebind_weights": manual_count,
            "stripped_drive_ids": stripped_drive,
            "warnings": warnings,
            "rebind_models": rebind_models,
        },
    }


def suggest_next_version(version: str, existing_versions: set[str]) -> str:
    """Return the lowest non-colliding SemVer bump for ``version``."""
    match = _SEMVER_RE.fullmatch(version.strip())
    if match:
        major, minor, patch = int(match.group(1)), int(match.group(2)), int(match.group(3))
        candidate = f"{major}.{minor}.{patch + 1}"
        while candidate in existing_versions:
            patch += 1
            candidate = f"{major}.{minor}.{patch + 1}"
        return candidate
    if version.isdigit():
        num = int(version) + 1
        while str(num) in existing_versions:
            num += 1
        return str(num)
    n = 1
    candidate = f"1.0.{n}"
    while candidate in existing_versions:
        n += 1
        candidate = f"1.0.{n}"
    return candidate


def suggest_workflow_key(key: str, existing_keys: set[str]) -> str:
    """Return a clean non-colliding workflow_key variant."""
    base = re.sub(r"-imported(?:-\d+)?$", "", key.strip()) or "workflow"
    candidate = f"{base}-imported"
    if candidate not in existing_keys and _WORKFLOW_KEY_RE.fullmatch(candidate):
        return candidate
    index = 2
    while True:
        candidate = f"{base}-imported-{index}"
        if candidate not in existing_keys and len(candidate) <= 120:
            return candidate
        index += 1


def find_workflow_version_conflict(repo, owner_id: str, key: str, version: str | None) -> dict[str, Any] | None:
    """Detect (workflow_key, workflow_version) collision across active and soft-deleted rows."""
    if not key or version is None:
        return None
    target_version = str(version)
    all_rows = repo.list("workflows", owner_id, include_deleted=True)
    existing_keys = {str(r.get("workflow_key")) for r in all_rows if r.get("workflow_key")}
    same_key_rows = [r for r in all_rows if r.get("workflow_key") == key]
    existing_versions = {str(r.get("workflow_version")) for r in same_key_rows if r.get("workflow_version") is not None}
    colliding = next((r for r in same_key_rows if str(r.get("workflow_version")) == target_version), None)
    if colliding is None:
        return None
    return {
        "workflow_key": key,
        "workflow_version": target_version,
        "existing_state": colliding.get("state"),
        "existing_versions": sorted(existing_versions),
        "suggested_version": suggest_next_version(target_version, existing_versions),
        "suggested_workflow_key": suggest_workflow_key(key, existing_keys),
    }
