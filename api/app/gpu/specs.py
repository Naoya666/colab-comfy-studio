"""Runtime-only compute selection; never rewrites a reviewed workflow artifact."""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

from .colab import _HIGH_MEM_ONLY_ACCELERATORS, ColabError, GoogleColabProvider
from .manifest import digest

DEFAULT_GPU_SPEC_ORDER: list[dict[str, str]] = [
    {"variant": "VARIANT_GPU", "accelerator": "T4", "shape": "SHAPE_STANDARD"},
    {"variant": "VARIANT_GPU", "accelerator": "T4", "shape": "SHAPE_HIGHMEM"},
    {"variant": "VARIANT_GPU", "accelerator": "L4", "shape": "SHAPE_STANDARD"},
    {"variant": "VARIANT_GPU", "accelerator": "A100", "shape": "SHAPE_STANDARD"},
    {"variant": "VARIANT_GPU", "accelerator": "A100", "shape": "SHAPE_HIGHMEM"},
    {"variant": "VARIANT_GPU", "accelerator": "G4", "shape": "SHAPE_STANDARD"},
    {"variant": "VARIANT_GPU", "accelerator": "G4", "shape": "SHAPE_HIGHMEM"},
    {"variant": "VARIANT_GPU", "accelerator": "H100", "shape": "SHAPE_STANDARD"},
    {"variant": "VARIANT_GPU", "accelerator": "H100", "shape": "SHAPE_HIGHMEM"},
]


def spec_key(value: dict) -> dict:
    if not isinstance(value, dict):
        raise ValueError("compute_spec 必须是包含 variant、accelerator、shape 的对象")
    try:
        key = GoogleColabProvider._spec(value.get("key", value))
    except ColabError as exc:
        raise ValueError(str(exc)) from exc
    key["accelerator"] = key["accelerator"].upper()
    if key["accelerator"] in _HIGH_MEM_ONLY_ACCELERATORS:
        key["shape"] = "SHAPE_STANDARD"
    return key


def spec_id(value: dict) -> str:
    key = spec_key(value)
    return f"{key['accelerator']}:{key['shape']}"


def _order_index_map(order_table: Sequence[Mapping] | None) -> dict[str, int]:
    source = DEFAULT_GPU_SPEC_ORDER if order_table is None else order_table
    mapping: dict[str, int] = {}
    for item in source:
        try:
            key = spec_key(dict(item))
        except ValueError:
            continue
        if key["variant"] != "VARIANT_GPU":
            continue
        sid = f"{key['accelerator']}:{key['shape']}"
        if sid not in mapping:
            mapping[sid] = len(mapping)
    return mapping


def spec_rank(value: dict | None, order_table: Sequence[Mapping] | None = None) -> int | None:
    if not isinstance(value, Mapping):
        return None
    try:
        key = spec_key(dict(value))
    except ValueError:
        return None
    if key["variant"] != "VARIANT_GPU":
        return None
    return _order_index_map(order_table).get(f"{key['accelerator']}:{key['shape']}")


def compare_specs(
    actual_spec: dict,
    required_spec: dict,
    order_table: Sequence[Mapping] | None = None,
) -> Literal["lower", "equal", "higher", "unknown"]:
    actual_key = spec_key(dict(actual_spec))
    required_key = spec_key(dict(required_spec))
    if actual_key == required_key:
        return "equal"
    if actual_key["variant"] != "VARIANT_GPU" or required_key["variant"] != "VARIANT_GPU":
        return "unknown"
    ranks = _order_index_map(order_table)
    actual_rank = ranks.get(f"{actual_key['accelerator']}:{actual_key['shape']}")
    required_rank = ranks.get(f"{required_key['accelerator']}:{required_key['shape']}")
    if actual_rank is None or required_rank is None:
        return "unknown"
    if actual_rank < required_rank:
        return "lower"
    if actual_rank > required_rank:
        return "higher"
    return "equal"


def is_below_workflow_default(
    manifest: Mapping | None,
    candidate_spec: Mapping | None,
    order_table: Sequence[Mapping] | None = None,
) -> bool:
    if not isinstance(manifest, Mapping) or not isinstance(candidate_spec, Mapping):
        return False
    workflow_default = manifest.get("compute_spec")
    if not isinstance(workflow_default, Mapping):
        return False
    return compare_specs(dict(candidate_spec), dict(workflow_default), order_table) not in {"equal", "higher"}


def _base_store(store):
    return store.store if hasattr(store, "store") else store


def _gpu_order_record_id(owner_id: str) -> str:
    return f"gpu-spec-order:{owner_id}"


def _normalize_gpu_order_list(items: Iterable[Mapping]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in items:
        key = spec_key(dict(item))
        if key["variant"] != "VARIANT_GPU":
            raise ValueError("GPU 规格顺序表仅支持 VARIANT_GPU 规格")
        sid = f"{key['accelerator']}:{key['shape']}"
        if sid not in seen:
            seen.add(sid)
            result.append(key)
    return result


def _normalize_discovered_list(items: Iterable, ordered_ids: set[str]) -> list[dict]:
    result: list[dict] = []
    seen: set[str] = set()
    for item in items or []:
        if not isinstance(item, Mapping):
            continue
        raw_spec = item.get("spec") if isinstance(item.get("spec"), Mapping) else item
        try:
            key = spec_key(dict(raw_spec))
        except ValueError:
            continue
        if key["variant"] != "VARIANT_GPU":
            continue
        sid = f"{key['accelerator']}:{key['shape']}"
        if sid in ordered_ids or sid in seen:
            continue
        seen.add(sid)
        discovered_at = item.get("discovered_at")
        result.append(
            {
                "spec": key,
                "discovered_at": int(discovered_at) if isinstance(discovered_at, (int, float)) else 0,
            }
        )
    return result


def gpu_spec_order_settings(store, owner_id: str) -> dict:
    base = _base_store(store)
    saved = base.get("settings", _gpu_order_record_id(owner_id), owner_id) if base is not None else None
    raw_ordered = saved.get("ordered_specs") if isinstance(saved, Mapping) else None
    if isinstance(raw_ordered, list) and raw_ordered:
        try:
            ordered = _normalize_gpu_order_list(raw_ordered)
        except ValueError:
            ordered = copy.deepcopy(DEFAULT_GPU_SPEC_ORDER)
    else:
        ordered = copy.deepcopy(DEFAULT_GPU_SPEC_ORDER)
    ordered_ids = {spec_id(item) for item in ordered}
    discovered = _normalize_discovered_list(
        saved.get("discovered_specs") if isinstance(saved, Mapping) else [], ordered_ids
    )
    return {
        "ordered_specs": ordered,
        "discovered_specs": discovered,
        "defaults": copy.deepcopy(DEFAULT_GPU_SPEC_ORDER),
        "updated_at": saved.get("updated_at") if isinstance(saved, Mapping) else None,
    }


def save_gpu_spec_order(
    store, owner_id: str, ordered_specs: Sequence[Mapping], *, now: float | None = None
) -> dict:
    if not isinstance(ordered_specs, Sequence) or isinstance(ordered_specs, (str, bytes)):
        raise ValueError("ordered_specs 必须是 GPU 规格数组")
    normalized = _normalize_gpu_order_list(ordered_specs)
    if not normalized:
        raise ValueError("GPU 规格顺序表不能为空")
    ordered_ids = {spec_id(item) for item in normalized}
    base = _base_store(store)
    ident = _gpu_order_record_id(owner_id)
    ts = int(time.time() if now is None else now)
    with base.admission_lock() as acquired:
        if not acquired:
            raise RuntimeError("GPU 规格顺序表正在更新，请稍后重试")
        existing = base.get("settings", ident, owner_id)
        discovered = _normalize_discovered_list(
            existing.get("discovered_specs") if isinstance(existing, Mapping) else [], ordered_ids
        )
        payload = {
            "ordered_specs": normalized,
            "discovered_specs": discovered,
            "updated_at": ts,
        }
        if existing:
            base.update("settings", ident, payload, owner_id)
        else:
            base.create("settings", owner_id, payload, record_id=ident)
    return {**payload, "defaults": copy.deepcopy(DEFAULT_GPU_SPEC_ORDER)}


def reset_gpu_spec_order(store, owner_id: str, *, now: float | None = None) -> dict:
    defaults = copy.deepcopy(DEFAULT_GPU_SPEC_ORDER)
    default_ids = {spec_id(item) for item in defaults}
    base = _base_store(store)
    ident = _gpu_order_record_id(owner_id)
    ts = int(time.time() if now is None else now)
    with base.admission_lock() as acquired:
        if not acquired:
            raise RuntimeError("GPU 规格顺序表正在更新，请稍后重试")
        existing = base.get("settings", ident, owner_id)
        # Preserve discovered specs as well as any custom specs that were not in defaults.
        prev_discovered = list(existing.get("discovered_specs") or []) if isinstance(existing, Mapping) else []
        prev_ordered = list(existing.get("ordered_specs") or []) if isinstance(existing, Mapping) else []
        for item in prev_ordered:
            try:
                key = spec_key(dict(item))
            except ValueError:
                continue
            if key["variant"] == "VARIANT_GPU" and spec_id(key) not in default_ids:
                prev_discovered.append({"spec": key, "discovered_at": ts})
        discovered = _normalize_discovered_list(prev_discovered, default_ids)
        payload = {
            "ordered_specs": defaults,
            "discovered_specs": discovered,
            "updated_at": ts,
        }
        if existing:
            base.update("settings", ident, payload, owner_id)
        else:
            base.create("settings", owner_id, payload, record_id=ident)
    return {**payload, "defaults": copy.deepcopy(DEFAULT_GPU_SPEC_ORDER)}


def record_discovered_gpu_specs(
    store, owner_id: str, raw_specs: Iterable[Mapping], *, now: float | None = None
) -> tuple[dict, list[dict]]:
    base = _base_store(store)
    ident = _gpu_order_record_id(owner_id)
    ts = int(time.time() if now is None else now)
    candidates: list[dict[str, str]] = []
    for item in raw_specs or []:
        if not isinstance(item, Mapping):
            continue
        if "eligible" in item and item.get("eligible") is not True:
            continue
        raw = item.get("key") if isinstance(item.get("key"), Mapping) else item
        try:
            key = spec_key(dict(raw))
        except ValueError:
            continue
        if key["variant"] == "VARIANT_GPU":
            candidates.append(key)

    newly_discovered: list[dict] = []
    with base.admission_lock() as acquired:
        if not acquired:
            return gpu_spec_order_settings(base, owner_id), []
        existing = base.get("settings", ident, owner_id)
        current = gpu_spec_order_settings(base, owner_id)
        ordered = current["ordered_specs"]
        ordered_ids = {spec_id(item) for item in ordered}
        discovered = list(current["discovered_specs"])
        known_ids = ordered_ids | {spec_id(entry["spec"]) for entry in discovered}
        for key in candidates:
            sid = spec_id(key)
            if sid in known_ids:
                continue
            known_ids.add(sid)
            entry = {"spec": key, "discovered_at": ts}
            discovered.append(entry)
            newly_discovered.append(entry)
        if newly_discovered:
            payload = {
                "ordered_specs": ordered,
                "discovered_specs": discovered,
                "updated_at": ts,
            }
            if existing:
                base.update("settings", ident, payload, owner_id)
            else:
                base.create("settings", owner_id, payload, record_id=ident)
            current = {**payload, "defaults": copy.deepcopy(DEFAULT_GPU_SPEC_ORDER)}
    return current, newly_discovered


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
