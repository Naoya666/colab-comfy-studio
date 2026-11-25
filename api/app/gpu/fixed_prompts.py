"""Per-(project, workflow) fixed prompt persistence and validation."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

POSITIONS = frozenset({"prefix", "suffix"})
SEPARATORS = frozenset({"comma", "space", "newline"})
MAX_FIXED_PROMPT_LENGTH = 20_000
_SLOT_KEYS = frozenset({"enabled", "text", "position", "separator"})
_CONFIG_KEYS = frozenset({"positive", "negative"})


def default_slot() -> dict[str, Any]:
    return {
        "enabled": True,
        "text": "",
        "position": "prefix",
        "separator": "comma",
    }


def default_fixed_prompts() -> dict[str, dict[str, Any]]:
    return {
        "positive": default_slot(),
        "negative": default_slot(),
    }


def normalize_slot(raw: Any, *, field_name: str = "slot") -> dict[str, Any]:
    if raw is None:
        return default_slot()
    if not isinstance(raw, Mapping):
        raise ValueError(f"{field_name} 必须是对象")
    extra = set(raw.keys()) - _SLOT_KEYS
    if extra:
        raise ValueError(f"{field_name} 包含不支持的字段: {sorted(extra)!r}")

    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError(f"{field_name}.enabled 必须是布尔值")

    text = raw.get("text", "")
    if not isinstance(text, str):
        raise ValueError(f"{field_name}.text 必须是字符串")
    if len(text) > MAX_FIXED_PROMPT_LENGTH:
        raise ValueError(f"{field_name}.text 长度不能超过 {MAX_FIXED_PROMPT_LENGTH} 个字符")

    position = raw.get("position", "prefix")
    if not isinstance(position, str) or position not in POSITIONS:
        raise ValueError(f"{field_name}.position 仅支持 prefix 或 suffix")

    separator = raw.get("separator", "comma")
    if not isinstance(separator, str) or separator not in SEPARATORS:
        raise ValueError(f"{field_name}.separator 仅支持 comma、space 或 newline")

    return {
        "enabled": enabled,
        "text": text,
        "position": position,
        "separator": separator,
    }


def normalize_fixed_prompts(raw: Any) -> dict[str, dict[str, Any]]:
    if raw is None:
        return default_fixed_prompts()
    if not isinstance(raw, Mapping):
        raise ValueError("fixed_prompts 必须是对象")
    extra = set(raw.keys()) - _CONFIG_KEYS
    if extra:
        raise ValueError(f"fixed_prompts 包含不支持的字段: {sorted(extra)!r}")
    return {
        "positive": normalize_slot(raw.get("positive"), field_name="positive"),
        "negative": normalize_slot(raw.get("negative"), field_name="negative"),
    }


def _record_id(owner_id: str, project_id: str, workflow_id: str) -> str:
    return hashlib.sha256(f"fixed-prompts:{owner_id}:{project_id}:{workflow_id}".encode()).hexdigest()


def get_fixed_prompts(store, owner_id: str, project_id: str, workflow_id: str) -> dict[str, Any]:
    row = store.get("settings", _record_id(owner_id, project_id, workflow_id), owner_id)
    if not isinstance(row, Mapping):
        config = default_fixed_prompts()
    else:
        try:
            config = normalize_fixed_prompts(
                {
                    "positive": row.get("positive"),
                    "negative": row.get("negative"),
                }
            )
        except ValueError:
            config = default_fixed_prompts()
    return {
        "project_id": project_id,
        "workflow_id": workflow_id,
        **config,
    }


def save_fixed_prompts(store, owner_id: str, project_id: str, workflow_id: str, raw: Any) -> dict[str, Any]:
    config = normalize_fixed_prompts(raw)
    ident = _record_id(owner_id, project_id, workflow_id)
    payload = {
        "kind": "fixed_prompts",
        "project_id": project_id,
        "workflow_id": workflow_id,
        **config,
    }
    if store.get("settings", ident, owner_id):
        store.update("settings", ident, payload, owner_id)
    else:
        store.create("settings", owner_id, payload, record_id=ident)
    return {
        "project_id": project_id,
        "workflow_id": workflow_id,
        **config,
    }
