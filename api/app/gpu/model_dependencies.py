"""Conservative, extensible ComfyUI model references; original graphs are immutable."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from pathlib import PurePosixPath

try:
    from .model_sources import SHA, ModelError, https_url, relative_path
except ImportError:  # Standalone Colab bootstrap.
    from model_sources import SHA, ModelError, https_url, relative_path

# field -> model category; widget positions are explicit, never guessed from a suffix.
RULES = {
    "CheckpointLoaderSimple": {"ckpt_name": "checkpoints"},
    "CheckpointLoader": {"ckpt_name": "checkpoints"},
    "LoraLoader": {"lora_name": "loras"}, "LoraLoaderModelOnly": {"lora_name": "loras"},
    "Lora Loader Stack (rgthree)": {f"lora_{i:02d}": "loras" for i in range(1, 5)},
    "VAELoader": {"vae_name": "vae"}, "UNETLoader": {"unet_name": "diffusion_models"},
    "DiffusionModelLoader": {"unet_name": "diffusion_models"},
    "UnetLoaderGGUF": {"unet_name": "diffusion_models"},
    "CLIPLoader": {"clip_name": "text_encoders"},
    "DualCLIPLoader": {"clip_name1": "text_encoders", "clip_name2": "text_encoders"},
    "TripleCLIPLoader": {"clip_name1": "text_encoders", "clip_name2": "text_encoders", "clip_name3": "text_encoders"},
    "QuadrupleCLIPLoader": {f"clip_name{i}": "text_encoders" for i in range(1, 5)},
    "CLIPVisionLoader": {"clip_name": "clip_vision"},
    "ControlNetLoader": {"control_net_name": "controlnet"},
    "DiffControlNetLoader": {"control_net_name": "controlnet"},
    "UpscaleModelLoader": {"model_name": "upscale_models"},
}
# Only schemas with known stable widget order are decoded without object_info.
WIDGETS = {
    "CheckpointLoaderSimple": ["ckpt_name"],
    "CheckpointLoader": ["config_name", "ckpt_name"],
    "LoraLoader": ["lora_name", "strength_model", "strength_clip"],
    "LoraLoaderModelOnly": ["lora_name", "strength_model"],
    "Lora Loader Stack (rgthree)": [
        "lora_01", "strength_01", "lora_02", "strength_02",
        "lora_03", "strength_03", "lora_04", "strength_04",
    ],
    "VAELoader": ["vae_name"],
    "UNETLoader": ["unet_name", "weight_dtype"],
    "DiffusionModelLoader": ["unet_name", "weight_dtype"],
    "UnetLoaderGGUF": ["unet_name"],
    "CLIPLoader": ["clip_name", "type", "device"],
    "DualCLIPLoader": ["clip_name1", "clip_name2", "type", "device"],
    "TripleCLIPLoader": ["clip_name1", "clip_name2", "clip_name3"],
    "QuadrupleCLIPLoader": ["clip_name1", "clip_name2", "clip_name3", "clip_name4"],
    "CLIPVisionLoader": ["clip_name"],
    "ControlNetLoader": ["control_net_name"],
    "DiffControlNetLoader": ["control_net_name"],
    "UpscaleModelLoader": ["model_name"],
    "CLIPTextEncode": ["text"],
    "CLIPTextEncodeSDXL": ["width", "height", "crop_w", "crop_h", "target_width", "target_height", "text_g", "text_l"],
}
_WIDGET_PRIMITIVES = {"COMBO", "INT", "FLOAT", "STRING", "BOOLEAN"}
_POWER_LORA_LOADERS = {"Power Lora Loader (rgthree)", "Power Lora Loader"}


def _widget_fields(node_type, node_types):
    if node_type in WIDGETS:
        return WIDGETS[node_type]
    cls = (node_types or {}).get(node_type)
    input_types_fn = getattr(cls, "INPUT_TYPES", None)
    if not callable(input_types_fn):
        return []
    try:
        spec_map = input_types_fn()
    except Exception:
        return []
    if not isinstance(spec_map, dict):
        return []
    fields = []
    for section in ("required", "optional"):
        entries = spec_map.get(section)
        if not isinstance(entries, dict):
            continue
        for name, spec in entries.items():
            if not isinstance(spec, (list, tuple)) or not spec:
                continue
            kind = spec[0]
            meta = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            if meta.get("forceInput"):
                continue
            if isinstance(kind, (list, tuple)) or kind in _WIDGET_PRIMITIVES:
                fields.append(name)
                if kind == "INT" and (meta.get("control_after_generate") or name in {"seed", "noise_seed"}):
                    fields.append(f"__control_after_generate_{name}")
    return fields


def _visual_node_inputs(node, node_types):
    node_type = node.get("type")
    values = node.get("widgets_values")
    if isinstance(values, dict):
        inputs = dict(values)
    elif node_type in _POWER_LORA_LOADERS and isinstance(values, list):
        lora_rows = [item for item in values if isinstance(item, dict) and "lora" in item]
        inputs = {f"lora_{idx}": item for idx, item in enumerate(lora_rows, 1)}
    elif isinstance(values, list):
        fields = _widget_fields(node_type, node_types)
        inputs = dict(zip(fields, values))
    else:
        inputs = {}
    for slot in node.get("inputs") or ():
        if isinstance(slot, dict) and slot.get("link") is not None and slot.get("name") in inputs:
            inputs.pop(slot["name"], None)
    return inputs
DIRECTORIES = {x: x for x in ("checkpoints", "loras", "vae", "diffusion_models", "text_encoders",
                              "clip_vision", "controlnet", "upscale_models", "embeddings", "latent_upscale_models")}
DIRECTORIES.update(unet="diffusion_models", clip="text_encoders")


def directories():
    result = {**DIRECTORIES, **json.loads(os.getenv("MODEL_LIBRARY_DIRECTORIES", "{}"))}
    return {k: relative_path(v) for k, v in result.items()}


def rules():
    return {**RULES, **json.loads(os.getenv("MODEL_DEPENDENCY_RULES", "{}"))}


def weights_of(workflow):
    manifest = workflow.get("manifest") or {}
    return manifest.get("weight_files") or (manifest.get("dependencies") or {}).get("weight_files") or []


def scan(workflow, *, node_types=None):
    graph = workflow.get("api_graph") or (workflow.get("image_extraction") or {}).get("prompt")
    visual = (workflow.get("image_extraction") or {}).get("workflow") or workflow.get("editable_graph") or {}
    nodes = [(str(i), n.get("class_type"), n.get("inputs") or {}, False)
             for i, n in (graph or {}).items() if isinstance(n, dict)]
    if not graph and isinstance(visual, dict):
        for n in visual.get("nodes") or []:
            if not isinstance(n, dict) or "id" not in n:
                continue
            inputs = _visual_node_inputs(n, node_types)
            nodes.append((str(n["id"]), n.get("type"), inputs, n.get("mode") in {2, 4}))
    dependencies = []
    mapping = rules()

    def append(node_id, node_type, field, reference, category, required, **extra):
        if not isinstance(reference, str) or reference.lower() in {"none", "none.safetensors", "disabled", ""}:
            return
        ident = hashlib.sha256(f"{node_id}:{field}:{reference}".encode()).hexdigest()[:24]
        dependencies.append({"id": ident, "node_id": node_id, "node_type": node_type, "field": field,
                             "reference": reference, "model_type": category, "architecture": "unknown",
                             "required": required, "status": "unresolved", "model_id": None, **extra})

    for node_id, node_type, inputs, bypassed in nodes:
        if node_type in {"Power Lora Loader (rgthree)", "Power Lora Loader"}:
            for field, value in inputs.items():
                if re.fullmatch(r"lora_\d+", field) and isinstance(value, dict):
                    strengths = {f"{field}.{key}": value[key] for key in ("strength", "strengthTwo")
                                 if isinstance(value.get(key), (int, float))}
                    active = value.get("on") is not False and any(v != 0 for v in strengths.values())
                    append(node_id, node_type, f"{field}.lora", value.get("lora"), "loras", not bypassed and active,
                           bypassed=bypassed, enabled_field=f"{field}.on", strengths=strengths)
            continue
        rule = mapping.get(node_type, {})
        for field, category in rule.items():
            required = not bypassed
            if category == "loras":
                strength_keys = ("strength_model", "strength_clip")
                if node_type == "Lora Loader Stack (rgthree)" and re.fullmatch(r"lora_\d+", field):
                    strength_keys = (f"strength_{field.split('_', 1)[1]}",)
                strengths = [inputs[k] for k in strength_keys if k in inputs]
                if strengths and all(isinstance(v, (int, float)) and v == 0 for v in strengths):
                    required = False
            append(node_id, node_type, field, inputs.get(field), category, required,
                   bypassed=bypassed, strengths={k: v for k, v in inputs.items() if k in (strength_keys if category == "loras" else ())})
        if node_type in {"CLIPTextEncode", "CLIPTextEncodeSDXL"}:
            for field in ("text", "text_g", "text_l"):
                if isinstance(inputs.get(field), str):
                    for name in re.findall(r"\bembedding:([^\s,():]+)", inputs[field]):
                        append(node_id, node_type, field, name, "embeddings", not bypassed, embedding=True)
        # Unknown loaders remain visible without treating arbitrary strings as filenames.
        if not rule and node_type and re.search(r"(load|lora)", node_type, re.I) and node_type not in {"LoadImage", "LoadVideo"}:
            append(node_id, node_type, "?", "需补充节点识别规则", "unknown", not bypassed, status="manual_review")
    # ComfyUI publishes top-level model metadata and node templates can provide it per node.
    visual_nodes = {str(n.get("id")): n for n in visual.get("nodes", []) if isinstance(n, dict)} if isinstance(visual, dict) else {}
    global_hints = visual.get("models", []) if isinstance(visual, dict) else []
    directory_map = directories()
    for dep in dependencies:
        node = visual_nodes.get(dep["node_id"], {})
        local_hints = (node.get("properties") or {}).get("models") or []
        hints = [
            (hint, scoped)
            for candidates, scoped in ((local_hints, True), (global_hints, False)) if isinstance(candidates, list)
            for hint in candidates
        ]
        candidates = []
        for hint, scoped in hints:
            if not isinstance(hint, dict) or hint.get("name") != dep["reference"]:
                continue
            directory = hint.get("directory")
            if not (directory is None and scoped) and (
                not isinstance(directory, str)
                or directory_map.get(directory.removeprefix("models/"), directory.removeprefix("models/"))
                != directory_map.get(dep["model_type"])
            ):
                continue
            candidates.append(hint)
        urls, digests, sizes = set(), set(), set()
        for hint in candidates:
            if isinstance(hint.get("url"), str):
                try:
                    urls.add(https_url(hint["url"]))
                except ModelError:
                    dep["source_error"] = "节点来源链接无效，需手动补充"
            digest = hint.get("sha256")
            if not digest and str(hint.get("hash_type", "")).lower().replace("-", "") == "sha256":
                digest = hint.get("hash")
            if isinstance(digest, str) and SHA.fullmatch(digest):
                digests.add(digest.lower())
            if type(hint.get("size")) is int and hint["size"] > 0:
                sizes.add(hint["size"])
            if isinstance(hint.get("architecture"), str):
                dep["architecture"] = hint["architecture"]
        if len(urls) > 1 or len(digests) > 1 or len(sizes) > 1:
            dep["source_conflict"] = True
        if len(urls) == 1:
            dep["source_hint"] = next(iter(urls))
        if len(digests) == 1:
            dep["expected_sha256"] = next(iter(digests))
        if len(sizes) == 1:
            dep["expected_size"] = next(iter(sizes))
    # Reviewed legacy manifests supply exact source identity; no filename-only source inference.
    for dep in dependencies:
        if dep["status"] == "manual_review":
            continue
        matches = [w for w in weights_of(workflow)
                   if w.get("target") == f"{directories().get(dep['model_type'], dep['model_type'])}/{dep['reference']}" ]
        if len(matches) == 1:
            dep["weight"] = copy.deepcopy(matches[0])
            dep["expected_sha256"] = matches[0].get("sha256")
            dep["architecture"] = matches[0].get("architecture") or dep["architecture"]
    return dependencies


def identity_matches(dep, model):
    sha = dep.get("expected_sha256")
    if sha and (model.get("sha256") or model.get("expected_sha256")) == sha.lower():
        return True
    weight = dep.get("weight") or {}
    identity = model.get("source_identity") or {}
    if weight.get("source") == "drive":
        return weight.get("drive_file_id") == model.get("drive_file_id")
    return bool(weight.get("repo_id") and all(weight.get(k) == identity.get(k) for k in ("repo_id", "revision", "filename")))


def match(dependencies, models, previous=()):
    previous = {d["id"]: d for d in previous}
    for dep in dependencies:
        old = previous.get(dep["id"], {})
        bound = next((m for m in models if m["id"] == old.get("model_id")), None)
        if bound:
            dep.update({k: old[k] for k in ("model_id", "confirmed", "strengths", "disabled", "replacement", "required") if k in old})
            dep["status"] = "matched" if bound.get("state") == "ready" else bound.get("state", "pending")
            continue
        if dep["status"] == "manual_review":
            continue
        dep["model_id"] = None
        exact = [m for m in models if m.get("model_type") == dep["model_type"] and identity_matches(dep, m)]
        if exact:
            model = sorted(exact, key=lambda m: m.get("state") != "ready")[0]
            dep.update(model_id=model["id"], status="matched" if model.get("state") == "ready" else model.get("state", "pending"))
        else:
            dep["candidates"] = [m["id"] for m in models if m.get("model_type") == dep["model_type"]
                                 and PurePosixPath(dep["reference"]).name == m.get("filename")]
            dep["status"] = "needs_confirmation" if dep["candidates"] else "unresolved"
    return dependencies


def bind(dep, model, *, confirm_unknown=False, strengths=None, disabled=False):
    if model.get("model_type") != dep["model_type"]:
        raise ModelError("模型类型不兼容")
    expected = str(dep.get("architecture") or "unknown").lower()
    actual = str(model.get("architecture") or "unknown").lower()
    if expected != "unknown" and actual != "unknown" and expected != actual:
        raise ModelError("模型架构不兼容，禁止替换")
    if (expected == "unknown" or actual == "unknown") and not confirm_unknown:
        raise ModelError("架构未知，请明确确认兼容性")
    if strengths is not None:
        allowed = set(dep.get("strengths") or {})
        if not set(strengths).issubset(allowed) or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not -3 <= v <= 3 for v in strengths.values()):
            raise ModelError("节点不支持该 LoRA 权重字段或权重不在 -3 到 3 范围")
    if disabled and not dep.get("strengths"):
        raise ModelError("当前节点不支持通过强度关闭 LoRA")
    merged_strengths = {**(dep.get("strengths") or {}), **(strengths or {})}
    required = dep.get("required", True)
    if dep["model_type"] == "loras" and merged_strengths:
        required = not dep.get("bypassed", False) and not disabled and any(v != 0 for v in merged_strengths.values())
    return {**dep, "model_id": model["id"], "confirmed": True, "required": required,
            "replacement": not identity_matches(dep, model), "disabled": disabled,
            "strengths": {**(dep.get("strengths") or {}), **(strengths or {})}}


def set_input(inputs, path, value):
    parts = path.split(".")
    for part in parts[:-1]:
        if not isinstance(inputs.get(part), dict):
            raise ModelError("节点输入结构已变化，请重新扫描依赖")
        inputs = inputs[part]
    inputs[parts[-1]] = value


def target_for(dep, model=None):
    reference = relative_path(dep["reference"])
    if dep.get("embedding") and not PurePosixPath(reference).suffix and model:
        reference += PurePosixPath(model["filename"]).suffix
    return f"{directories()[dep['model_type']]}/{reference}"


def runtime_copy(snapshot, dependencies, models, approved=None):
    """Use Drive copies only for approved identities. Replacements require a fresh review.

    A saved binding changes neither the reviewed artifact nor its trust stamp.
    """
    result = copy.deepcopy(snapshot)
    by_id = {m["id"]: m for m in models}
    weights = result["manifest"].get("weight_files") or (result["manifest"].get("dependencies") or {}).get("weight_files") or []
    required_targets = {target_for(d, by_id.get(d.get("model_id"))) for d in dependencies
                        if d.get("required", True) and not d.get("disabled") and d.get("model_type") in directories()}
    optional_targets = {target_for(d, by_id.get(d.get("model_id"))) for d in dependencies
                        if (not d.get("required", True) or d.get("disabled")) and d.get("model_type") in directories()}
    weights = [w for w in weights if w.get("target") not in optional_targets - required_targets]
    for dep in dependencies:
        model = by_id.get(dep.get("model_id"))
        if dep.get("enabled_field") and dep.get("confirmed"):
            set_input(result["bound_graph"][dep["node_id"]]["inputs"], dep["enabled_field"], not dep.get("disabled", False))
        if dep.get("disabled"):
            for field in dep.get("strengths") or {}:
                set_input(result["bound_graph"][dep["node_id"]]["inputs"], field, 0)
            continue
        if not model or not dep.get("required", True):
            continue
        if model.get("state") != "ready":
            raise ModelError(f"模型 {dep['reference']} 尚未在 Drive 完成准备")
        if not identity_matches(dep, model) and model.get("sha256") not in (approved or {}).get((dep["node_id"], dep["field"]), set()):
            raise ModelError("模型替换改变已审核身份，请先将绑定导出为新草稿并完成审核和运行验证")
        target = target_for(dep, model)
        replaced = False
        for index, weight in enumerate(weights):
            if weight.get("target") == target:
                replaced = True
                weights[index] = {"source": "drive", "drive_file_id": model["drive_file_id"], "sha256": model["sha256"],
                                  "size": model["size"], "target": target}
        if not replaced:
            raise ModelError("模型槽位未在已审核权重清单中声明，请生成绑定草稿并审核")
        if dep.get("strengths"):
            for field, value in dep["strengths"].items():
                set_input(result["bound_graph"][dep["node_id"]]["inputs"], field, value)
    result["manifest"]["weight_files"] = weights
    return result


def approved_alternates(repo, owner, workflow):
    """Only model identities with accepted real smoke evidence in this workflow family.

    Graph topology, environment pins and parameter schema must match. Merely
    labeling a model SDXL or importing it does not grant runtime approval.
    """
    from app import review_routes

    def skeleton(row):
        graph = copy.deepcopy(row.get("api_graph") or {})
        for d in scan(row):
            if d.get("embedding") or d.get("status") == "manual_review":
                continue
            inputs = graph.get(d["node_id"], {}).get("inputs")
            if isinstance(inputs, dict):
                set_input(inputs, d["field"], "<reviewed-model-slot>")
                for strength in d.get("strengths") or {}:
                    set_input(inputs, strength, 0)
        return graph

    baseline = skeleton(workflow)
    manifest = workflow.get("manifest") or {}
    pins = manifest.get("dependencies") or {}
    result = {}
    for row in repo.list("workflows", owner):
        if (row.get("workflow_key") != workflow.get("workflow_key") or row.get("state") != "published"
                or not row.get("review_approved") or row.get("smoke_test_kind") != "real"):
            continue
        other = row.get("manifest") or {}
        other_pins = other.get("dependencies") or {}
        if (any(pins.get(k) != other_pins.get(k) for k in ("comfyui_commit", "custom_nodes"))
                or manifest.get("parameters") != other.get("parameters") or skeleton(row) != baseline):
            continue
        job = repo.get("jobs", row.get("smoke_job_id"), owner) if row.get("smoke_job_id") else None
        family = next((m for m in repo.list("models", owner) if m.get("model_id") == row.get("model_id")), None)
        if (not job or not family or job.get("state") != "completed" or job.get("provider") != "colab"
                or job.get("smoke_accepted") is not True or job.get("acceptance_state") != "accepted"
                or not review_routes._job_matches_any_smoke(job, row["id"], row, family)):
            continue
        for dep in scan(row):
            digest = dep.get("expected_sha256")
            if digest:
                result.setdefault((dep["node_id"], dep["field"]), set()).add(digest)
    return result
