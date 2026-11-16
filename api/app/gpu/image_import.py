"""Treat image-extracted JSON as unconfigured data, never executable or reviewed."""

import copy
import re


def missing_configuration(source: dict) -> list[str]:
    from .manifest import validate_manifest

    missing = []
    manifest = source.get("manifest") or {}
    dependencies = manifest.get("dependencies") or {}
    if not isinstance(dependencies, dict):
        dependencies = {}
    if not source.get("model_id") or source["model_id"] == "unconfigured":
        missing.append("模型关联 model_id")
    if not source.get("api_graph"):
        missing.append("API prompt / api_graph")
    if not source.get("object_info"):
        missing.append("object_info 和节点依赖审查")
    if not dependencies.get("comfyui_commit"):
        missing.append("ComfyUI 固定版本 dependencies.comfyui_commit")
    if not dependencies.get("weight_files"):
        missing.append("模型清单、来源、大小与 SHA-256：dependencies.weight_files")
    if "parameters" not in manifest:
        missing.append("参数映射 parameters（无可调参数时显式填写空对象）")
    if not manifest.get("compute_spec"):
        missing.append("默认 GPU compute_spec")
    try:
        validate_manifest(manifest, source.get("api_graph") or {}, source.get("object_info") or {})
    except ValueError as exc:
        missing.append("静态校验：" + str(exc)[:200])
    return missing


def extraction_bundle(bundle: dict) -> dict:
    extracted = bundle.get("image_extraction")
    if not isinstance(extracted, dict):
        raise ValueError("image_extraction 必须是对象")
    workflow, prompt = extracted.get("workflow"), extracted.get("prompt")
    if workflow is None and prompt is None:
        raise ValueError("图片中没有有效 ComfyUI 工作流")
    if workflow is not None:
        if not isinstance(workflow, dict) or not isinstance(workflow.get("nodes"), list):
            raise ValueError("workflow 必须为可视化节点格式")
        if (
            len(workflow["nodes"]) > 10000
            or not all(
                isinstance(n, dict) and isinstance(n.get("id"), (str, int)) and isinstance(n.get("type"), str)
                for n in workflow["nodes"]
            )
            or ("links" in workflow and not isinstance(workflow["links"], list))
        ):
            raise ValueError("workflow 节点或连线格式无效")
    if prompt is not None and (
        not isinstance(prompt, dict)
        or not prompt
        or len(prompt) > 10000
        or not all(
            isinstance(n, dict) and isinstance(n.get("class_type"), str) and isinstance(n.get("inputs"), dict)
            for n in prompt.values()
        )
    ):
        raise ValueError("prompt 必须为 API 执行格式，不能使用可视化 workflow")
    nodes = list(prompt.values()) if prompt else (workflow or {}).get("nodes", [])
    types = sorted({n.get("class_type") or n.get("type") for n in nodes})
    files = set()

    def walk(value, depth=0):
        if depth > 64:
            raise ValueError("图片 JSON 嵌套过深")
        if isinstance(value, str) and re.search(r"\.(safetensors|ckpt|pt|pth|bin|gguf|onnx)$", value, re.I):
            files.add(value)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item, depth + 1)
        elif isinstance(value, list):
            for item in value:
                walk(item, depth + 1)

    walk(workflow)
    walk(prompt)
    return {
        "workflow_key": bundle.get("workflow_key"),
        "model_id": "unconfigured",
        "manifest": {},
        "api_graph": copy.deepcopy(prompt or {}),
        "object_info": {},
        "editable_graph": copy.deepcopy(workflow if workflow is not None else prompt),
        "image_extraction": copy.deepcopy({"workflow": workflow, "prompt": prompt, **({"metadata": extracted["metadata"]} if "metadata" in extracted else {})}),
        "image_import_summary": {"node_count": len(nodes), "node_types": types, "model_files": sorted(files)},
        "pending_configuration": missing_configuration({"api_graph": prompt}),
    }
