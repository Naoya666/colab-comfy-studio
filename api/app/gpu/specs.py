"""Runtime-only compute selection; never rewrites a reviewed workflow artifact."""

from __future__ import annotations

import math

from .colab import ColabError, GoogleColabProvider
from .manifest import digest


def spec_key(value: dict) -> dict:
    if not isinstance(value, dict):
        raise ValueError("compute_spec 必须是包含 variant、accelerator、shape 的对象")
    try:
        key = GoogleColabProvider._spec(value.get("key", value))
    except ColabError as exc:
        raise ValueError(str(exc)) from exc
    key["accelerator"] = key["accelerator"].upper()
    return key


def check_requirements(manifest: dict, spec: dict) -> list[str]:
    key = spec_key(spec)
    if key["variant"] != "VARIANT_GPU":
        raise ValueError("请选择 GPU 计算规格")
    req = manifest.get("compute_requirements") or {}
    if not isinstance(req, dict):
        raise ValueError("compute_requirements 必须为对象")
    gpu = key["accelerator"]
    allowed = req.get("accelerators")
    if allowed is not None and (
        not isinstance(allowed, list) or not all(isinstance(g, str) for g in allowed)
    ):
        raise ValueError("compute_requirements.accelerators 必须为 GPU 架构 / 型号列表")
    if allowed and gpu not in allowed:
        raise ValueError(f"{gpu} 不满足工作流 GPU 架构限制")
    # Only trust explicitly reported capacity, never assume A100 means 80 GB.
    for field, label in [("min_vram_gb", "显存"), ("min_ram_gb", "系统 RAM")]:
        minimum = req.get(field)
        capacity = spec.get("vram_gb" if field == "min_vram_gb" else "ram_gb")
        if minimum is not None:
            if (
                isinstance(minimum, bool)
                or not isinstance(minimum, (float, int))
                or not math.isfinite(minimum)
                or minimum <= 0
            ):
                raise ValueError(f"{field} 必须为正数")
            if capacity is not None and float(capacity) < minimum:
                raise ValueError(f"{gpu} {label} {capacity} GB 低于最低要求 {minimum} GB")
            if capacity is None:
                # The worker checks actual hardware before restoring any environment/models.
                continue
    verified = manifest.get("verified_compute_specs") or []
    if not verified and manifest.get("compute_spec"):
        verified = [manifest["compute_spec"]]
    return (
        []
        if any(spec_key(s) == key for s in verified)
        else ["此 GPU 规格尚未经过该工作流验证，可能运行失败；不会自动修改参数或切换 GPU。"]
    )


def resolve_compute(manifest: dict, override: dict | None) -> tuple[dict | None, list[str]]:
    value = override if override is not None else manifest.get("compute_spec")
    if value is None:
        return None, []
    key = spec_key(value)
    return {"key": key}, check_requirements(manifest, {"key": key})


def compatibility_signature(
    provider: str, snapshot: dict, requested: dict | None, workflow_key=None, workflow_version=None
) -> str:
    return digest(
        {
            "provider": provider,
            "environment": {
                k: v
                for k, v in snapshot.items()
                if k not in {"resolved_params", "bound_graph", "input_files"}
            },
            "requested_spec": requested,
            "workflow_key": workflow_key,
            "workflow_version": workflow_version,
        }
    )
