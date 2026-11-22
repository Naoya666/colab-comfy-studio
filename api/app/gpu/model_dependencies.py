"""Conservative, extensible ComfyUI model references; original graphs are immutable."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from pathlib import PurePosixPath

try:
    from .model_sources import SHA, ModelError, hf_download_url, https_url, normalize_hf_repo_type, relative_path
except ImportError:  # Standalone Colab bootstrap.
    from model_sources import SHA, ModelError, hf_download_url, https_url, normalize_hf_repo_type, relative_path

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
_WEIGHT_FILE_RE = re.compile(r"\.(?:safetensors|gguf|onnx|pt|pth|ckpt|bin)$", re.I)
_HINT_LOADER_RE = re.compile(
    r"(?:load|lora|checkpoint|ckpt|unet|diffusion|vae|clip|controlnet|upscale|gguf|model|adapter|pulid|instantid|embedding|facerestore|reactor|animatediff)",
    re.I,
)
_NON_MODEL_NODE_TYPES = {
    "LoadImage", "LoadVideo", "LoadAudio", "LoadImageMask", "LoadImageOutput",
    "SaveImage", "PreviewImage", "SaveAnimatedWEBP", "SaveAnimatedPNG", "SaveVideo",
    "CLIPTextEncode", "CLIPTextEncodeSDXL", "CLIPTextEncodeFlux",
    "PrimitiveNode", "Note", "MarkdownNote", "Reroute",
    "VHS_LoadVideo", "VHS_LoadVideoPath", "VHS_LoadAudio", "VHS_LoadAudioUpload", "VHS_VideoCombine",
}
_IGNORED_REFS = {"none", "none.safetensors", "disabled", ""}


def _is_api_prompt_graph(value):
    return (
        isinstance(value, dict)
        and bool(value)
        and all(
            isinstance(node, dict) and isinstance(node.get("class_type"), str) and isinstance(node.get("inputs"), dict)
            for node in value.values()
        )
    )


def _is_visual_workflow(value):
    return (
        isinstance(value, dict)
        and isinstance(value.get("nodes"), list)
        and all(
            isinstance(node, dict)
            and isinstance(node.get("id"), (int, str))
            and isinstance(node.get("type"), str)
            for node in value["nodes"]
        )
        and (value.get("links") is None or isinstance(value.get("links"), list))
    )


def _guess_model_type_from_node(node_type, field="", ref=""):
    haystack = f"{node_type or ''} {field or ''}".lower()
    ref_lower = (ref or "").lower()
    if "lora" in haystack:
        return "loras"
    if "upscal" in haystack:
        return "upscale_models"
    if "controlnet" in haystack or "control_net" in haystack:
        return "controlnet"
    if "clip_vision" in haystack or "clipvision" in haystack:
        return "clip_vision"
    if "text_encoder" in haystack or "clip" in haystack or "t5" in haystack or "umt5" in haystack:
        return "text_encoders"
    if "vae" in haystack:
        return "vae"
    if "unet" in haystack or "diffusion" in haystack or "gguf" in haystack:
        return "diffusion_models"
    if "checkpoint" in haystack or "ckpt" in haystack:
        return "checkpoints"
    if "embedding" in haystack:
        return "embeddings"
    if ref_lower.endswith(".gguf"):
        return "diffusion_models"
    return "unknown"


def _extract_weight_strings(value, *, prefix="", depth=0):
    if depth > 6:
        return
    if isinstance(value, str):
        cleaned = value.strip()
        if (
            cleaned
            and len(cleaned) <= 300
            and "\n" not in cleaned
            and "\r" not in cleaned
            and not cleaned.lower().startswith(("http://", "https://"))
            and cleaned.lower() not in _IGNORED_REFS
            and _WEIGHT_FILE_RE.search(cleaned)
        ):
            yield prefix or "widget", cleaned
    elif isinstance(value, list):
        for idx, item in enumerate(value[:64]):
            yield from _extract_weight_strings(item, prefix=f"{prefix}[{idx}]" if prefix else f"widget[{idx}]", depth=depth + 1)
    elif isinstance(value, dict):
        for key, item in list(value.items())[:64]:
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _extract_weight_strings(item, prefix=next_prefix, depth=depth + 1)


def source_url_of(dep):
    hint = dep.get("source_hint")
    if isinstance(hint, str) and hint:
        return hint
    weight = dep.get("weight")
    if isinstance(weight, dict):
        if weight.get("repo_id") and weight.get("revision") and weight.get("filename"):
            try:
                return hf_download_url(
                    weight["repo_id"],
                    weight["revision"],
                    weight["filename"],
                    normalize_hf_repo_type(weight.get("repo_type")),
                )
            except ModelError:
                return None
        for key in ("source_url", "download_url", "url"):
            val = weight.get(key)
            if isinstance(val, str) and val:
                try:
                    return https_url(val)
                except ModelError:
                    return None
    return None


def _widget_fields(node_type, node_types, *, allowed_types=None):
    if node_type in WIDGETS:
        return WIDGETS[node_type]
    if allowed_types is not None and node_type not in allowed_types:
        return []
    cls = (node_types or {}).get(node_type)
    input_types_fn = getattr(cls, "INPUT_TYPES", None)
    if not callable(input_types_fn):
        return []
    try:
        spec_map = input_types_fn()
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
                if isinstance(kind, (list, tuple)) or (isinstance(kind, str) and kind in _WIDGET_PRIMITIVES):
                    fields.append(name)
                    if kind == "INT" and (meta.get("control_after_generate") or name in {"seed", "noise_seed"}):
                        fields.append(f"__control_after_generate_{name}")
        return fields
    except Exception:
        return []


def _visual_node_inputs(node, node_types, *, allowed_types=None, link_values=None, passthrough_links=frozenset()):
    node_type = node.get("type")
    values = node.get("widgets_values")
    if isinstance(values, dict):
        inputs = dict(values)
    elif node_type in _POWER_LORA_LOADERS and isinstance(values, list):
        lora_rows = [item for item in values if isinstance(item, dict) and "lora" in item]
        inputs = {f"lora_{idx}": item for idx, item in enumerate(lora_rows, 1)}
    elif isinstance(values, list):
        fields = _widget_fields(node_type, node_types, allowed_types=allowed_types)
        if not fields and isinstance(node.get("inputs"), list):
            fields = [
                slot["name"]
                for slot in node["inputs"]
                if isinstance(slot, dict) and isinstance(slot.get("widget"), dict) and isinstance(slot.get("name"), str)
            ]
        inputs = dict(zip(fields, values))
    else:
        inputs = {}
    for slot in node.get("inputs") or ():
        if not isinstance(slot, dict) or slot.get("link") is None:
            continue
        lid = slot["link"]
        sname = slot.get("name")
        if link_values and lid in link_values and isinstance(sname, str):
            inputs[sname] = link_values[lid]
        elif lid in passthrough_links:
            # Unlinked subgraph parameter without an outer widget slot retains the inner node's default.
            continue
        elif sname in inputs:
            inputs.pop(sname, None)
    return inputs


def _subgraph_link_values(outer_node, subgraph, parent_link_values, parent_passthrough_links=frozenset()):
    values = outer_node.get("widgets_values")
    slots = [s for s in (outer_node.get("inputs") or ()) if isinstance(s, dict)]
    slots_by_name = {s["name"]: s for s in slots if isinstance(s.get("name"), str)}
    explicit_widget_slots = [s for s in slots if isinstance(s.get("widget"), dict)]
    slot_values = {}
    externally_linked = set()

    if explicit_widget_slots or isinstance(values, dict):
        widget_idx = 0
        for slot in slots:
            name = slot.get("name")
            is_widget = isinstance(slot.get("widget"), dict)
            widget_val = None
            has_widget_val = False
            if is_widget and isinstance(values, list):
                if widget_idx < len(values):
                    widget_val = values[widget_idx]
                    has_widget_val = True
                widget_idx += 1
            elif isinstance(values, dict) and isinstance(name, str) and name in values:
                widget_val = values[name]
                has_widget_val = True
            link_id = slot.get("link")
            if link_id is not None:
                if parent_link_values and link_id in parent_link_values and isinstance(name, str):
                    slot_values[name] = parent_link_values[link_id]
                elif link_id not in parent_passthrough_links and isinstance(name, str):
                    externally_linked.add(name)
            elif has_widget_val and isinstance(name, str):
                slot_values[name] = widget_val
    else:
        for slot in slots:
            name = slot.get("name")
            link_id = slot.get("link")
            if link_id is not None and isinstance(name, str):
                if parent_link_values and link_id in parent_link_values:
                    slot_values[name] = parent_link_values[link_id]
                elif link_id not in parent_passthrough_links:
                    externally_linked.add(name)
        if isinstance(values, list):
            widget_idx = 0
            for sg_in in subgraph.get("inputs") or ():
                if not isinstance(sg_in, dict):
                    continue
                in_name = sg_in.get("name")
                if not isinstance(in_name, str) or in_name in slots_by_name:
                    continue
                if sg_in.get("type") in _WIDGET_PRIMITIVES and widget_idx < len(values):
                    slot_values[in_name] = values[widget_idx]
                    widget_idx += 1

    inner_links = {}
    passthrough_links = set()
    for sg_in in subgraph.get("inputs") or ():
        if not isinstance(sg_in, dict):
            continue
        in_name = sg_in.get("name")
        lids = sg_in.get("linkIds") or ()
        if in_name in slot_values:
            for lid in lids:
                inner_links[lid] = slot_values[in_name]
        elif in_name not in externally_linked:
            passthrough_links.update(lids)
    return inner_links, frozenset(passthrough_links)


def _expand_visual_nodes(
    container,
    subgraphs,
    node_types,
    *,
    allowed_types=None,
    prefix="",
    parent_bypassed=False,
    link_values=None,
    passthrough_links=frozenset(),
    stack=frozenset(),
):
    for n in container.get("nodes") or ():
        if not isinstance(n, dict) or "id" not in n:
            continue
        node_id = f"{prefix}{n['id']}"
        bypassed = parent_bypassed or (n.get("mode") in {2, 4})
        node_type = n.get("type")
        if isinstance(node_type, str) and node_type in subgraphs and node_type not in stack and len(stack) < 8:
            sg = subgraphs[node_type]
            inner_links, inner_passthrough = _subgraph_link_values(n, sg, link_values, passthrough_links)
            yield from _expand_visual_nodes(
                sg,
                subgraphs,
                node_types,
                allowed_types=allowed_types,
                prefix=f"{node_id}:",
                parent_bypassed=bypassed,
                link_values=inner_links,
                passthrough_links=inner_passthrough,
                stack=stack | {node_type},
            )
        else:
            inputs = _visual_node_inputs(
                n, node_types, allowed_types=allowed_types, link_values=link_values, passthrough_links=passthrough_links,
            )
            yield node_id, node_type, inputs, bypassed, n


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


def _resolve_hint_category(raw_dir, directory_map, *, node_type="", field="", ref=""):
    if isinstance(raw_dir, str) and raw_dir.strip():
        cleaned = raw_dir.strip().removeprefix("models/").strip("/")
        if cleaned in directory_map:
            return directory_map[cleaned]
    return _guess_model_type_from_node(node_type, field, ref)


def _apply_hint_metadata(dep, candidates):
    urls, digests, sizes = set(), set(), set()
    for hint in candidates:
        if isinstance(hint.get("url"), str) and hint["url"].strip():
            try:
                urls.add(https_url(hint["url"].strip()))
            except ModelError:
                dep["source_error"] = "节点来源链接无效，需手动补充"
        digest = hint.get("sha256")
        if not digest and str(hint.get("hash_type", "")).lower().replace("-", "") == "sha256":
            digest = hint.get("hash")
        if isinstance(digest, str) and SHA.fullmatch(digest):
            digests.add(digest.lower())
        if type(hint.get("size")) is int and hint["size"] > 0:
            sizes.add(hint["size"])
        if isinstance(hint.get("architecture"), str) and hint["architecture"].strip():
            dep["architecture"] = hint["architecture"].strip()
    if len(urls) > 1 or len(digests) > 1 or len(sizes) > 1:
        dep["source_conflict"] = True
    if len(urls) == 1:
        dep["source_hint"] = next(iter(urls))
    if len(digests) == 1:
        dep["expected_sha256"] = next(iter(digests))
    if len(sizes) == 1:
        dep["expected_size"] = next(iter(sizes))


def scan(workflow, *, node_types=None, include_hints=False):
    graph = workflow.get("api_graph") or (workflow.get("image_extraction") or {}).get("prompt")
    visual = (workflow.get("image_extraction") or {}).get("workflow") or workflow.get("editable_graph") or {}
    nodes = [(str(i), n.get("class_type"), n.get("inputs") or {}, False)
             for i, n in (graph or {}).items() if isinstance(n, dict)]
    mapping = rules()
    visual_nodes = {}
    visual_meta = {}
    if isinstance(visual, dict) and isinstance(visual.get("nodes"), list):
        defs = visual.get("definitions")
        raw_subgraphs = (defs.get("subgraphs") or ()) if isinstance(defs, dict) else ()
        subgraphs = {
            sg["id"]: sg
            for sg in raw_subgraphs
            if isinstance(sg, dict) and isinstance(sg.get("id"), str) and isinstance(sg.get("nodes"), list)
        }
        expanded = list(_expand_visual_nodes(visual, subgraphs, node_types, allowed_types=mapping))
        if not graph:
            for node_id, node_type, inputs, bypassed, raw_node in expanded:
                nodes.append((node_id, node_type, inputs, bypassed))
                visual_nodes[node_id] = raw_node
                visual_meta[node_id] = (node_type, bypassed)
        else:
            for node_id, node_type, _inputs, bypassed, raw_node in expanded:
                visual_nodes[node_id] = raw_node
                visual_meta[node_id] = (node_type, bypassed)
            for n in visual.get("nodes", []):
                if isinstance(n, dict) and "id" in n:
                    visual_nodes.setdefault(str(n["id"]), n)
                    visual_meta.setdefault(str(n["id"]), (n.get("type"), n.get("mode") in {2, 4}))
    dependencies = []

    def append(node_id, node_type, field, reference, category, required, **extra):
        if not isinstance(reference, str) or reference.lower() in _IGNORED_REFS:
            return
        ident = hashlib.sha256(f"{node_id}:{field}:{reference}".encode()).hexdigest()[:24]
        dependencies.append({"id": ident, "node_id": node_id, "node_type": node_type, "field": field,
                             "reference": reference, "model_type": category, "architecture": "unknown",
                             "required": required, "status": "unresolved", "model_id": None, **extra})

    for node_id, node_type, inputs, bypassed in nodes:
        if node_type in _POWER_LORA_LOADERS:
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
        _apply_hint_metadata(dep, candidates)

    if include_hints:
        known_node_refs = {
            (d["node_id"], d["reference"])
            for d in dependencies
            if d["status"] != "manual_review"
        }
        known_global_refs = {
            (d["reference"], directory_map.get(d["model_type"], d["model_type"]))
            for d in dependencies
            if d["status"] != "manual_review"
        }
        resolved_unknown_nodes = set()

        # Fallback 1: explicit node properties.models & top-level workflow.models metadata.
        for node_id, raw_node in visual_nodes.items():
            local_hints = (raw_node.get("properties") or {}).get("models") or []
            if not isinstance(local_hints, list):
                continue
            node_type, bypassed = visual_meta.get(node_id, (raw_node.get("type") or "UnknownNode", raw_node.get("mode") in {2, 4}))
            grouped = {}
            for hint in local_hints:
                if not isinstance(hint, dict):
                    continue
                name = hint.get("name")
                if not isinstance(name, str) or name.strip().lower() in _IGNORED_REFS:
                    continue
                name = name.strip()
                if (node_id, name) in known_node_refs:
                    continue
                grouped.setdefault(name, []).append(hint)
            for name, hint_items in grouped.items():
                first_dir = next((h.get("directory") for h in hint_items if isinstance(h.get("directory"), str)), None)
                category = _resolve_hint_category(first_dir, directory_map, node_type=node_type, field="properties.models", ref=name)
                append(
                    node_id,
                    node_type,
                    "properties.models",
                    name,
                    category,
                    not bypassed,
                    bypassed=bypassed,
                    detected_via="metadata",
                )
                dep = dependencies[-1]
                _apply_hint_metadata(dep, hint_items)
                known_node_refs.add((node_id, name))
                known_global_refs.add((name, directory_map.get(category, category)))
                resolved_unknown_nodes.add(node_id)

        if isinstance(global_hints, list):
            grouped_global = {}
            for hint in global_hints:
                if not isinstance(hint, dict):
                    continue
                name = hint.get("name")
                if not isinstance(name, str) or name.strip().lower() in _IGNORED_REFS:
                    continue
                name = name.strip()
                category = _resolve_hint_category(hint.get("directory"), directory_map, field="workflow.models", ref=name)
                key = (name, directory_map.get(category, category))
                if key in known_global_refs:
                    continue
                grouped_global.setdefault((name, category), []).append(hint)
            for (name, category), hint_items in grouped_global.items():
                append(
                    "workflow",
                    "工作流元数据",
                    "models",
                    name,
                    category,
                    True,
                    detected_via="metadata",
                )
                dep = dependencies[-1]
                _apply_hint_metadata(dep, hint_items)
                known_global_refs.add((name, directory_map.get(category, category)))

        # Fallback 2: weight filename clues inside unrecognized loader nodes.
        for node_id, node_type, inputs, bypassed in nodes:
            if not node_type or node_type in mapping or node_type in _POWER_LORA_LOADERS or node_type in _NON_MODEL_NODE_TYPES:
                continue
            if not _HINT_LOADER_RE.search(node_type):
                continue
            raw_node = visual_nodes.get(node_id) or {}
            extracted = list(_extract_weight_strings(inputs))
            if not extracted and raw_node.get("widgets_values") is not None:
                extracted = list(_extract_weight_strings(raw_node.get("widgets_values"), prefix="widgets"))
            seen_refs = set()
            for field, ref in extracted:
                if ref in seen_refs or (node_id, ref) in known_node_refs:
                    continue
                seen_refs.add(ref)
                category = _guess_model_type_from_node(node_type, field, ref)
                append(
                    node_id,
                    node_type,
                    field,
                    ref,
                    category,
                    not bypassed,
                    bypassed=bypassed,
                    detected_via="heuristic",
                )
                known_node_refs.add((node_id, ref))
                resolved_unknown_nodes.add(node_id)

        if resolved_unknown_nodes:
            dependencies = [
                d for d in dependencies
                if not (d["status"] == "manual_review" and d["node_id"] in resolved_unknown_nodes)
            ]

    # Reviewed legacy manifests supply exact source identity; no filename-only source inference.
    for dep in dependencies:
        if dep["status"] == "manual_review":
            continue
        matches = [w for w in weights_of(workflow)
                   if w.get("target") == f"{directory_map.get(dep['model_type'], dep['model_type'])}/{dep['reference']}"]
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


def _type_compatible(dep_type, model_type):
    return dep_type == "unknown" or model_type == dep_type


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
        exact = [m for m in models if _type_compatible(dep["model_type"], m.get("model_type")) and identity_matches(dep, m)]
        if exact:
            model = sorted(exact, key=lambda m: m.get("state") != "ready")[0]
            dep.update(model_id=model["id"], status="matched" if model.get("state") == "ready" else model.get("state", "pending"))
        else:
            dep["candidates"] = [m["id"] for m in models if _type_compatible(dep["model_type"], m.get("model_type"))
                                 and PurePosixPath(dep["reference"]).name == m.get("filename")]
            dep["status"] = "needs_confirmation" if dep["candidates"] else "unresolved"
    return dependencies


def _normalize_inspect_payload(payload):
    if not isinstance(payload, dict) or not payload:
        raise ModelError("请提供有效的工作流 JSON 对象")
    warnings = []
    # Studio bundle or image extraction wrapper
    if any(k in payload for k in ("api_graph", "editable_graph", "image_extraction", "manifest")):
        normalized = {}
        if isinstance(payload.get("manifest"), dict):
            normalized["manifest"] = payload["manifest"]
        if isinstance(payload.get("api_graph"), dict):
            normalized["api_graph"] = payload["api_graph"]
        if isinstance(payload.get("editable_graph"), dict):
            normalized["editable_graph"] = payload["editable_graph"]
        if isinstance(payload.get("image_extraction"), dict):
            normalized["image_extraction"] = payload["image_extraction"]
        api_graph = normalized.get("api_graph") or (normalized.get("image_extraction") or {}).get("prompt")
        visual = (normalized.get("image_extraction") or {}).get("workflow") or normalized.get("editable_graph")
        if not api_graph and not visual:
            raise ModelError("工作流包中未找到可识别的节点图（缺少 api_graph / editable_graph / workflow）")
        node_count = len(api_graph) if isinstance(api_graph, dict) else len((visual or {}).get("nodes") or [])
        fmt = "bundle" if "manifest" in normalized else ("both" if api_graph and visual else ("api" if api_graph else "visual"))
    elif "workflow" in payload or "prompt" in payload:
        visual = payload.get("workflow")
        prompt = payload.get("prompt")
        if visual is not None and not _is_visual_workflow(visual):
            warnings.append("workflow 字段不是有效的可视化工作流结构，已跳过")
            visual = None
        if prompt is not None and not _is_api_prompt_graph(prompt):
            warnings.append("prompt 字段不是有效的 API 工作流结构，已跳过")
            prompt = None
        if not visual and not prompt:
            raise ModelError("未检测到有效的 ComfyUI workflow 或 API prompt")
        normalized = {"image_extraction": {"workflow": visual, "prompt": prompt}}
        node_count = len(prompt) if isinstance(prompt, dict) else len(visual["nodes"])
        fmt = "both" if visual and prompt else ("api" if prompt else "visual")
    elif _is_visual_workflow(payload):
        normalized = {"editable_graph": payload}
        node_count = len(payload["nodes"])
        fmt = "visual"
    elif _is_api_prompt_graph(payload):
        normalized = {"api_graph": payload}
        node_count = len(payload)
        fmt = "api"
    else:
        raise ModelError("无法识别的工作流 JSON 格式，请导入 ComfyUI 工作流 JSON、API prompt JSON 或 PNG 原图")

    if node_count > 10000:
        raise ModelError("工作流节点数量超过 10000 上限")
    return normalized, fmt, node_count, warnings


def inspect_workflow(payload, models, *, node_types=None):
    """Stateless dependency inspection for the Model Library without creating any workflow draft."""
    normalized, fmt, node_count, warnings = _normalize_inspect_payload(payload)
    scanned = scan(normalized, node_types=node_types, include_hints=True)
    matched = match(scanned, models)
    for dep in matched:
        dep.setdefault("detected_via", "rule")
        if not dep.get("source_hint"):
            derived = source_url_of(dep)
            if derived:
                dep["source_hint"] = derived
    in_progress_states = {"pending", "resolving", "downloading", "verifying", "uploading"}
    ready_count = sum(1 for d in matched if d["status"] == "matched")
    in_progress_count = sum(1 for d in matched if d["status"] in in_progress_states)
    needs_confirm_count = sum(1 for d in matched if d["status"] == "needs_confirmation")
    unresolved_count = sum(1 for d in matched if d["status"] == "unresolved")
    manual_review_count = sum(1 for d in matched if d["status"] == "manual_review")
    with_source_count = sum(1 for d in matched if bool(d.get("source_hint")))
    optional_count = sum(1 for d in matched if not d.get("required", True) or d.get("disabled"))
    complete = all(not d.get("required", True) or d.get("disabled") or d["status"] == "matched" for d in matched)
    return {
        "format": fmt,
        "node_count": node_count,
        "dependencies": matched,
        "models_ready": complete,
        "warnings": warnings,
        "summary": {
            "total": len(matched),
            "ready": ready_count,
            "in_progress": in_progress_count,
            "needs_confirmation": needs_confirm_count,
            "unresolved": unresolved_count,
            "manual_review": manual_review_count,
            "with_source_hint": with_source_count,
            "optional": optional_count,
        },
    }


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
