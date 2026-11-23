"""Offline workspace management for native ComfyUI workflows, plugins, and build caches.

All reads and writes talk directly to Google Drive (drive.file scope) and the VPS database
without starting a Colab GPU runtime or consuming CU. Every mutation strictly enforces an
offline write lock mutually exclusive with native ComfyUI leases and startup admission.
"""
from __future__ import annotations

import json
import posixpath
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime
from typing import Any

from app.common import APIError

from . import native
from .drive import FOLDER_MIME, DriveError, DriveStorage
from .drive_usage import (
    _bulk_list_app_entries,
    _find_owner_folder_readonly,
    invalidate_drive_usage_cache,
    resolve_owner_drive_account,
)
from .google import GoogleError
from .node_persistence import digest

MAX_WORKFLOW_JSON_BYTES = 20 * 1024 * 1024
ENV_CACHE_OVERSIZED_BYTES = 500 * 1024 * 1024
CURRENT_ENV_CACHE_SCHEMA = 3
DEFAULT_KEEP_HISTORY = 5
MAX_HISTORY_ENTRIES = 20
OVERVIEW_CACHE_TTL_SECONDS = 30.0
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

_BUILTIN_VIRTUAL_NODES = frozenset(
    {
        "Note",
        "MarkdownNote",
        "Reroute",
        "PrimitiveNode",
        "PrimitiveString",
        "PrimitiveInt",
        "PrimitiveFloat",
        "PrimitiveBoolean",
        "GraphInput",
        "GraphOutput",
    }
)
_BUILTIN_CNR_IDS = frozenset({"comfy-core"})

_OVERVIEW_LOCK = threading.Lock()
_OVERVIEW_CACHE: dict[tuple[str, str, int], tuple[float, dict[str, Any]]] = {}
_HISTORY_CACHE_LOCK = threading.Lock()
_HISTORY_JSON_CACHE: dict[str, dict[str, Any]] = {}


def invalidate_workspace_cache(owner_id: str | None = None) -> None:
    with _OVERVIEW_LOCK:
        if owner_id is None:
            _OVERVIEW_CACHE.clear()
            return
        for key in [k for k in _OVERVIEW_CACHE if k[0] == owner_id]:
            _OVERVIEW_CACHE.pop(key, None)


def _settings_key(owner_id: str) -> str:
    return "native-plugins-" + digest(owner_id)[:32]


def _int_or_zero(value: Any) -> int:
    if value is None or isinstance(value, bool):
        return 0
    try:
        parsed = int(value)
        return parsed if parsed >= 0 else 0
    except (TypeError, ValueError):
        return 0


def _iso_to_epoch(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        raw = value.strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        return datetime.fromisoformat(raw).timestamp()
    except ValueError:
        return None


def _validate_workflow_relpath(raw: Any) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise APIError("WORKSPACE_INVALID_PATH", "工作流路径无效", 422)
    cleaned = raw.strip()
    if cleaned.startswith("/") or "\\" in cleaned:
        raise APIError("WORKSPACE_INVALID_PATH", "工作流路径无效", 422)
    norm = posixpath.normpath(cleaned)
    parts = norm.split("/")
    if norm in {".", ".."} or norm.startswith("../") or any(not p or p in {".", ".."} or len(p) > 200 for p in parts):
        raise APIError("WORKSPACE_INVALID_PATH", "工作流路径无效", 422)
    if not norm.lower().endswith(".json"):
        raise APIError("WORKSPACE_INVALID_PATH", "仅支持 .json 工作流文件", 422)
    return norm


def _validate_plugin_path(raw: Any) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise APIError("WORKSPACE_INVALID_PLUGIN", "插件名称无效", 422)
    cleaned = raw.strip()
    if cleaned.startswith("/") or "\\" in cleaned:
        raise APIError("WORKSPACE_INVALID_PLUGIN", "插件名称无效", 422)
    norm = posixpath.normpath(cleaned)
    parts = norm.split("/")
    if (
        norm in {".", ".."}
        or norm.startswith("../")
        or len(parts) > 2
        or any(not p or p in {".", ".."} or len(p) > 200 for p in parts)
    ):
        raise APIError("WORKSPACE_INVALID_PLUGIN", "插件名称无效", 422)
    return norm


def _normalize_alnum(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _strip_plugin_decorators(path: str) -> str:
    base = path
    if base.startswith(".disabled/"):
        base = base[len(".disabled/") :]
    elif base.startswith("disabled/"):
        base = base[len("disabled/") :]
    if base.endswith(".disabled"):
        base = base[: -len(".disabled")]
    if base.endswith(".py"):
        base = base[:-3]
    return base


def _build_plugin_lookup(plugin_paths: list[str]) -> dict[str, str]:
    """Map normalized cnr_id/aux_id variants to exact installed/disabled plugin paths."""
    lookup: dict[str, str] = {}
    for p in plugin_paths:
        core = _strip_plugin_decorators(p)
        variants = {
            p.lower(),
            core.lower(),
            _normalize_alnum(core),
        }
        compact = _normalize_alnum(core)
        if compact.startswith("comfyui") and len(compact) > 7:
            variants.add(compact[7:])
        for v in variants:
            if v and v not in lookup:
                lookup[v] = p
    return lookup


def _match_plugin_by_meta(cnr_id: Any, aux_id: Any, lookup: dict[str, str]) -> str | None:
    candidates: list[str] = []
    if isinstance(cnr_id, str) and cnr_id.strip() and cnr_id.strip() not in _BUILTIN_CNR_IDS:
        candidates.append(cnr_id.strip())
    if isinstance(aux_id, str) and aux_id.strip():
        raw = aux_id.strip()
        candidates.append(raw)
        if "/" in raw:
            candidates.append(raw.rsplit("/", 1)[-1])
    for cand in candidates:
        low = cand.lower()
        if low in lookup:
            return lookup[low]
        compact = _normalize_alnum(cand)
        if compact in lookup:
            return lookup[compact]
        if compact.startswith("comfyui") and compact[7:] in lookup:
            return lookup[compact[7:]]
    return None


def extract_workflow_dependencies(
    workflow: Any,
    *,
    telemetry_node_map: dict[str, str],
    plugin_lookup: dict[str, str],
    known_plugins: set[str],
) -> dict[str, Any]:
    """Extract custom node plugin dependencies from UI graph JSON (incl. subgraphs) or API prompt JSON."""
    if not isinstance(workflow, dict):
        return {"node_count": 0, "plugins": [], "plugin_nodes": {}, "unresolved_nodes": []}

    raw_nodes: list[tuple[str, int, Any, Any]] = []
    subgraph_ids: set[str] = set()

    def _collect_graph(graph: dict[str, Any]) -> None:
        defs = graph.get("definitions")
        if isinstance(defs, dict):
            subgraphs = defs.get("subgraphs")
            if isinstance(subgraphs, list):
                for sg in subgraphs:
                    if isinstance(sg, dict):
                        sg_id = sg.get("id")
                        if isinstance(sg_id, str) and sg_id:
                            subgraph_ids.add(sg_id)
                        _collect_graph(sg)
        nodes = graph.get("nodes")
        if isinstance(nodes, list):
            for node in nodes:
                if not isinstance(node, dict):
                    continue
                ntype = node.get("type")
                if not isinstance(ntype, str) or not ntype:
                    continue
                props = node.get("properties") if isinstance(node.get("properties"), dict) else {}
                mode = node.get("mode", 0)
                mode_int = mode if isinstance(mode, int) and not isinstance(mode, bool) else 0
                raw_nodes.append((ntype, mode_int, props.get("cnr_id"), props.get("aux_id")))

    if isinstance(workflow.get("nodes"), list):
        _collect_graph(workflow)
    else:
        for item in workflow.values():
            if isinstance(item, dict) and isinstance(item.get("class_type"), str) and item["class_type"]:
                raw_nodes.append((item["class_type"], 0, None, None))

    plugin_nodes: dict[str, set[str]] = {}
    unresolved_seen: set[tuple[str, str, str]] = set()
    unresolved_nodes: list[dict[str, Any]] = []
    counted_nodes = 0

    for ntype, _mode, cnr_id, aux_id in raw_nodes:
        if ntype in subgraph_ids:
            continue
        counted_nodes += 1
        if ntype in _BUILTIN_VIRTUAL_NODES:
            continue

        # Tier 1: exact runtime node_map from telemetry
        mapped = telemetry_node_map.get(ntype)
        if mapped:
            target = mapped if mapped in known_plugins else plugin_lookup.get(_normalize_alnum(_strip_plugin_decorators(mapped)))
            if target:
                plugin_nodes.setdefault(target, set()).add(ntype)
                continue

        # Tier 2: static properties cnr_id / aux_id (catches frontend virtual nodes like KJNodes GetNode/SetNode & day-1 fallback)
        meta_match = _match_plugin_by_meta(cnr_id, aux_id, plugin_lookup)
        if meta_match:
            plugin_nodes.setdefault(meta_match, set()).add(ntype)
            continue

        # Ignore explicit built-in core nodes
        if isinstance(cnr_id, str) and cnr_id in _BUILTIN_CNR_IDS and not aux_id:
            continue

        # Tier 3: explicitly tagged third-party node not matched to any installed/disabled plugin
        has_third_party_hint = bool(
            (isinstance(cnr_id, str) and cnr_id.strip() and cnr_id.strip() not in _BUILTIN_CNR_IDS)
            or (isinstance(aux_id, str) and aux_id.strip())
        )
        if has_third_party_hint:
            key = (ntype, str(cnr_id or ""), str(aux_id or ""))
            if key not in unresolved_seen:
                unresolved_seen.add(key)
                unresolved_nodes.append(
                    {
                        "type": ntype,
                        "cnr_id": cnr_id if isinstance(cnr_id, str) else None,
                        "aux_id": aux_id if isinstance(aux_id, str) else None,
                    }
                )

    unresolved_nodes.sort(key=lambda item: item["type"])
    return {
        "node_count": counted_nodes,
        "plugins": sorted(plugin_nodes),
        "plugin_nodes": {k: sorted(v) for k, v in sorted(plugin_nodes.items())},
        "unresolved_nodes": unresolved_nodes,
    }


def _strip_manifest_plugin_entry(raw: dict[str, Any]) -> dict[str, Any]:
    """Keep only canonical node_persistence manifest plugin fields."""
    entry = dict(raw)
    for extra_key in ("disabled_at", "disabled_reason", "source_generation", "origin"):
        entry.pop(extra_key, None)
    return entry


def _validate_manifest_doc(doc: Any, *, label: str = "manifest") -> dict[str, Any]:
    if not isinstance(doc, dict) or doc.get("schema") != 1 or not isinstance(doc.get("plugins"), list):
        raise APIError("WORKSPACE_MANIFEST_INVALID", f"Drive 插件清单（{label}）格式无效", 502)
    expected = digest(doc["plugins"])
    if doc.get("generation") != expected:
        raise APIError("WORKSPACE_MANIFEST_INVALID", f"Drive 插件清单（{label}）校验和不一致", 502)
    return doc


def _parse_disabled_doc(doc: Any) -> list[dict[str, Any]]:
    if not isinstance(doc, dict) or doc.get("schema") != 1 or not isinstance(doc.get("plugins"), list):
        return []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in doc["plugins"]:
        if not isinstance(item, dict):
            continue
        entry = item.get("entry")
        if not isinstance(entry, dict):
            continue
        path = entry.get("path")
        archive = entry.get("archive")
        if not isinstance(path, str) or not path or not isinstance(archive, str) or not _HEX64_RE.fullmatch(archive):
            continue
        if path in seen:
            continue
        seen.add(path)
        out.append(
            {
                "entry": _strip_manifest_plugin_entry(entry),
                "disabled_at": _iso_to_epoch(item.get("disabled_at")),
                "source_generation": item.get("source_generation") if isinstance(item.get("source_generation"), str) else None,
                "origin": item.get("origin") if isinstance(item.get("origin"), str) else "studio_offline",
            }
        )
    out.sort(key=lambda row: row["entry"]["path"])
    return out


class _WorkspaceDrive:
    """Thin Google Drive v3 helper built on DriveStorage for native-comfy/ management."""

    def __init__(self, storage: DriveStorage, owner_id: str):
        self.storage = storage
        self.owner_id = owner_id

    def prime_token(self) -> None:
        self.storage.google.access_token("drive")

    def find_owner_folder(self, *, create: bool = False) -> str | None:
        if create:
            return self.storage._owner_folder()
        return _find_owner_folder_readonly(self.storage, self.owner_id)

    def list_children(self, parent_id: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        page_token: str | None = None
        while True:
            params: dict[str, str] = {
                "q": f"'{self.storage._q(parent_id)}' in parents and trashed = false",
                "pageSize": "1000",
                "fields": "nextPageToken,files(id,name,mimeType,size,modifiedTime,parents)",
            }
            if page_token:
                params["pageToken"] = page_token
            response = self.storage._request("GET", self.storage.base_url + "/files", params=params)
            payload = self.storage._ensure_ok(response, "目录列表查询")
            for item in payload.get("files") or []:
                if isinstance(item, dict) and item.get("id"):
                    out.append(item)
            page_token = payload.get("nextPageToken")
            if not page_token:
                break
        return out

    def find_child_folder(self, parent_id: str, name: str, *, create: bool = False) -> str | None:
        if create:
            return self.storage._folder(name, parent_id)
        return self.storage._find_folder(name, parent_id)

    def find_folder_path(self, root_id: str, parts: tuple[str, ...], *, create: bool = False) -> str | None:
        cur = root_id
        for part in parts:
            nxt = self.find_child_folder(cur, part, create=create)
            if not nxt:
                return None
            cur = nxt
        return cur

    def find_files_by_name(self, parent_id: str, name: str) -> list[dict[str, Any]]:
        q = (
            f"name = '{self.storage._q(name)}' and mimeType != '{FOLDER_MIME}' "
            f"and trashed = false and '{self.storage._q(parent_id)}' in parents"
        )
        response = self.storage._request(
            "GET",
            self.storage.base_url + "/files",
            params={"q": q, "pageSize": "20", "fields": "files(id,name,mimeType,size,modifiedTime,parents)"},
        )
        payload = self.storage._ensure_ok(response, "文件查询")
        files = [item for item in (payload.get("files") or []) if isinstance(item, dict) and item.get("id")]
        files.sort(key=lambda item: _iso_to_epoch(item.get("modifiedTime")) or 0.0, reverse=True)
        return files

    def download_bytes(self, file_id: str, *, max_bytes: int = MAX_WORKFLOW_JSON_BYTES, known_size: int | None = None) -> bytes:
        if known_size is not None and known_size > max_bytes:
            raise APIError("WORKSPACE_FILE_TOO_LARGE", f"文件超过大小上限（{max_bytes // (1024 * 1024)} MiB）", 413)
        response = self.storage._request(
            "GET",
            self.storage.base_url + f"/files/{file_id}",
            params={"alt": "media"},
            headers={"Accept": "application/octet-stream"},
        )
        if response.status_code < 200 or response.status_code >= 300:
            raise DriveError(f"Drive 文件下载失败（HTTP {response.status_code}）")
        data = response.content
        if len(data) > max_bytes:
            raise APIError("WORKSPACE_FILE_TOO_LARGE", f"文件超过大小上限（{max_bytes // (1024 * 1024)} MiB）", 413)
        return data

    def download_json(self, file_id: str, *, max_bytes: int = MAX_WORKFLOW_JSON_BYTES, known_size: int | None = None) -> Any:
        raw = self.download_bytes(file_id, max_bytes=max_bytes, known_size=known_size)
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise ValueError("JSON 解析失败") from exc

    def write_json_inplace(self, parent_id: str, name: str, payload: dict[str, Any]) -> str:
        """Update existing Drive file in place (preventing duplicate same-name files) or create atomically."""
        raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        existing = self.find_files_by_name(parent_id, name)
        if existing:
            primary_id = existing[0]["id"]
            response = self.storage._request(
                "PATCH",
                self.storage.upload_url + f"/files/{primary_id}",
                params={"uploadType": "media", "fields": "id,name,size,modifiedTime"},
                content=raw,
                headers={"Content-Type": "application/json; charset=utf-8"},
                retryable=True,
            )
            self.storage._ensure_ok(response, f"更新文件 {name}")
            # Clean up any legacy duplicate files with the exact same name in this folder.
            for dup in existing[1:]:
                try:
                    self.trash_file(dup["id"])
                except (DriveError, GoogleError):
                    pass
            return primary_id

        boundary = "studio-ws-" + uuid.uuid4().hex
        meta_bytes = json.dumps(
            {"name": name, "parents": [parent_id], "mimeType": "application/json"},
            ensure_ascii=False,
        ).encode("utf-8")
        body = (
            f"--{boundary}\r\n"
            f"Content-Type: application/json; charset=UTF-8\r\n\r\n"
        ).encode("utf-8") + meta_bytes + (
            f"\r\n--{boundary}\r\n"
            f"Content-Type: application/json; charset=UTF-8\r\n\r\n"
        ).encode("utf-8") + raw + f"\r\n--{boundary}--\r\n".encode("utf-8")
        response = self.storage._request(
            "POST",
            self.storage.upload_url + "/files",
            params={"uploadType": "multipart", "fields": "id,name,size,modifiedTime"},
            content=body,
            headers={"Content-Type": f"multipart/related; boundary={boundary}"},
            retryable=False,
        )
        created = self.storage._ensure_ok(response, f"创建文件 {name}")
        file_id = created.get("id")
        if not isinstance(file_id, str) or not file_id:
            raise DriveError(f"Drive 创建文件 {name} 响应无效")
        return file_id

    def move_file(self, file_id: str, src_parent: str, dst_parent: str) -> None:
        response = self.storage._request(
            "PATCH",
            self.storage.base_url + f"/files/{file_id}",
            params={"addParents": dst_parent, "removeParents": src_parent, "fields": "id,parents"},
            json_body={},
            retryable=False,
        )
        self.storage._ensure_ok(response, "移动工作流文件")

    def trash_file(self, file_id: str) -> None:
        response = self.storage._request(
            "PATCH",
            self.storage.base_url + f"/files/{file_id}",
            params={"fields": "id,trashed"},
            json_body={"trashed": True},
            retryable=True,
        )
        self.storage._ensure_ok(response, "移入回收站")

    def delete_file_permanent(self, file_id: str) -> None:
        response = self.storage._request(
            "DELETE",
            self.storage.base_url + f"/files/{file_id}",
            retryable=False,
        )
        if response.status_code in {200, 204, 404}:
            return
        raise DriveError(f"Drive 删除文件失败（HTTP {response.status_code}）")

    def locate_workflow_file(self, root_folder_id: str, relpath: str) -> tuple[dict[str, Any], str] | None:
        parts = relpath.split("/")
        cur = root_folder_id
        for folder_name in parts[:-1]:
            cur = self.find_child_folder(cur, folder_name, create=False)
            if not cur:
                return None
        matches = self.find_files_by_name(cur, parts[-1])
        if not matches:
            return None
        return matches[0], cur


def _open_drive(repo, owner_id: str, drive: DriveStorage | None = None) -> tuple[DriveStorage | None, bool, str | None, str | None]:
    if drive is not None:
        return drive, False, drive.account_id, drive.account_id or "默认 Drive 账号"
    account_id, account_alias = resolve_owner_drive_account(repo, owner_id)
    if account_id is None and account_alias is None:
        return None, False, None, None
    storage = DriveStorage(
        {"id": "native-workspace", "owner_id": owner_id},
        repo,
        account_id=account_id,
    )
    return storage, True, account_id, account_alias


@contextmanager
def _offline_write_guard(repo, owner_id: str, *, reason: str):
    lock_id = native.acquire_workspace_write_lock(repo, owner_id, reason=reason)
    try:
        yield lock_id
    finally:
        native.release_workspace_write_lock(repo, owner_id, lock_id)


def _tree_find_child(children_map: dict[str, list[dict[str, Any]]], parent_id: str, name: str, *, is_folder: bool) -> dict[str, Any] | None:
    matches = []
    for item in children_map.get(parent_id, ()):
        if item.get("name") != name:
            continue
        mime = str(item.get("mimeType") or "")
        if is_folder and mime == FOLDER_MIME:
            matches.append(item)
        elif not is_folder and mime != FOLDER_MIME:
            matches.append(item)
    if not matches:
        return None
    matches.sort(key=lambda row: _iso_to_epoch(row.get("modifiedTime")) or 0.0, reverse=True)
    return matches[0]


def _tree_walk_json(children_map: dict[str, list[dict[str, Any]]], root_id: str | None) -> list[dict[str, Any]]:
    if not root_id:
        return []
    results: list[dict[str, Any]] = []
    seen_folders: set[str] = set()

    def _walk(fid: str, prefix: str) -> None:
        if fid in seen_folders:
            return
        seen_folders.add(fid)
        items = sorted(children_map.get(fid, ()), key=lambda x: str(x.get("name") or ""))
        for item in items:
            iid = str(item.get("id") or "")
            name = str(item.get("name") or "")
            mime = str(item.get("mimeType") or "")
            if not iid or not name:
                continue
            if mime == FOLDER_MIME:
                _walk(iid, f"{prefix}{name}/")
            elif name.lower().endswith(".json"):
                results.append(
                    {
                        "id": iid,
                        "path": f"{prefix}{name}",
                        "parent_id": fid,
                        "size_bytes": _int_or_zero(item.get("size")),
                        "modified_at": _iso_to_epoch(item.get("modifiedTime")),
                    }
                )

    _walk(root_id, "")
    return results


def _read_plugins_bundle(
    wdrive: _WorkspaceDrive,
    children_map: dict[str, list[dict[str, Any]]],
    plugins_fid: str | None,
    saved_settings: dict[str, Any],
) -> dict[str, Any]:
    head_manifest: dict[str, Any] | None = None
    pending_manifest: dict[str, Any] | None = None
    disabled_entries: list[dict[str, Any]] = []
    objects_map: dict[str, dict[str, Any]] = {}  # sha256 -> {id, name, size_bytes, modified_at, duplicates}
    history_entries: list[dict[str, Any]] = []

    if plugins_fid:
        wdrive.prime_token()
        head_file = _tree_find_child(children_map, plugins_fid, "head.json", is_folder=False)
        pending_file = _tree_find_child(children_map, plugins_fid, "pending.json", is_folder=False)
        disabled_file = _tree_find_child(children_map, plugins_fid, "disabled.json", is_folder=False)

        objects_dir = _tree_find_child(children_map, plugins_fid, "objects", is_folder=True)
        if objects_dir:
            for item in children_map.get(objects_dir["id"], ()):
                name = str(item.get("name") or "")
                if not name.endswith(".tar.gz"):
                    continue
                sha = name[: -len(".tar.gz")]
                if not _HEX64_RE.fullmatch(sha):
                    continue
                entry = {
                    "id": str(item["id"]),
                    "name": name,
                    "sha256": sha,
                    "size_bytes": _int_or_zero(item.get("size")),
                    "modified_at": _iso_to_epoch(item.get("modifiedTime")),
                }
                if sha not in objects_map:
                    objects_map[sha] = {**entry, "duplicate_ids": []}
                else:
                    objects_map[sha]["duplicate_ids"].append(entry["id"])

        def _safe_load(file_meta: dict[str, Any] | None) -> Any:
            if not file_meta:
                return None
            try:
                return wdrive.download_json(file_meta["id"], max_bytes=4 * 1024 * 1024, known_size=_int_or_zero(file_meta.get("size")))
            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=4) as pool:
            f_head = pool.submit(_safe_load, head_file)
            f_pending = pool.submit(_safe_load, pending_file)
            f_disabled = pool.submit(_safe_load, disabled_file)
            raw_head = f_head.result()
            raw_pending = f_pending.result()
            raw_disabled = f_disabled.result()

        if isinstance(raw_head, dict) and raw_head.get("schema") == 1 and isinstance(raw_head.get("plugins"), list):
            head_manifest = raw_head
        if isinstance(raw_pending, dict) and raw_pending.get("schema") == 1 and isinstance(raw_pending.get("plugins"), list):
            pending_manifest = raw_pending
        disabled_entries = _parse_disabled_doc(raw_disabled)

        history_dir = _tree_find_child(children_map, plugins_fid, "history", is_folder=True)
        if history_dir:
            hist_files = [
                item
                for item in children_map.get(history_dir["id"], ())
                if str(item.get("name") or "").endswith(".json")
                and _HEX64_RE.fullmatch(str(item.get("name") or "")[: -len(".json")])
            ]
            hist_files.sort(key=lambda row: _iso_to_epoch(row.get("modifiedTime")) or 0.0, reverse=True)
            to_fetch = hist_files[:MAX_HISTORY_ENTRIES]

            def _load_hist(meta: dict[str, Any]) -> dict[str, Any] | None:
                gen = str(meta["name"])[: -len(".json")]
                with _HISTORY_CACHE_LOCK:
                    cached = _HISTORY_JSON_CACHE.get(gen)
                if cached is not None:
                    return cached
                doc = _safe_load(meta)
                if isinstance(doc, dict) and doc.get("schema") == 1 and isinstance(doc.get("plugins"), list):
                    if not doc.get("backed_up_at"):
                        doc = {**doc, "backed_up_at": _iso_to_epoch(meta.get("modifiedTime")) or 0.0}
                    with _HISTORY_CACHE_LOCK:
                        _HISTORY_JSON_CACHE[gen] = doc
                    return doc
                return None

            with ThreadPoolExecutor(max_workers=6) as pool:
                for loaded in pool.map(_load_hist, to_fetch):
                    if loaded is not None:
                        history_entries.append(loaded)

    if head_manifest is None and isinstance(saved_settings.get("manifest"), dict):
        head_manifest = saved_settings["manifest"]
    if not history_entries and isinstance(saved_settings.get("history"), list):
        history_entries = [h for h in saved_settings["history"] if isinstance(h, dict) and isinstance(h.get("plugins"), list)]

    # Deduplicate history by generation and sort newest first
    dedup_hist: dict[str, dict[str, Any]] = {}
    for h in history_entries:
        gen = h.get("generation")
        if isinstance(gen, str) and _HEX64_RE.fullmatch(gen):
            if gen not in dedup_hist or float(h.get("backed_up_at") or 0) > float(dedup_hist[gen].get("backed_up_at") or 0):
                dedup_hist[gen] = h
    sorted_history = sorted(dedup_hist.values(), key=lambda item: float(item.get("backed_up_at") or 0.0), reverse=True)[:MAX_HISTORY_ENTRIES]

    effective_manifest = pending_manifest or head_manifest or {"schema": 1, "plugins": [], "generation": digest([])}
    return {
        "head": head_manifest,
        "pending": pending_manifest,
        "effective": effective_manifest,
        "disabled": disabled_entries,
        "history": sorted_history,
        "objects": objects_map,
    }


def _compute_orphan_objects(
    bundle: dict[str, Any],
    *,
    keep_history: int = DEFAULT_KEEP_HISTORY,
) -> dict[str, Any]:
    keep = max(1, min(int(keep_history), MAX_HISTORY_ENTRIES))
    protected: set[str] = set()
    for manifest in (bundle.get("head"), bundle.get("pending"), bundle.get("effective")):
        if isinstance(manifest, dict):
            for p in manifest.get("plugins") or []:
                if isinstance(p, dict) and isinstance(p.get("archive"), str):
                    protected.add(p["archive"])
    for d in bundle.get("disabled") or []:
        arch = (d.get("entry") or {}).get("archive")
        if isinstance(arch, str):
            protected.add(arch)
    for h in (bundle.get("history") or [])[:keep]:
        for p in h.get("plugins") or []:
            if isinstance(p, dict) and isinstance(p.get("archive"), str):
                protected.add(p["archive"])

    objects_map: dict[str, dict[str, Any]] = bundle.get("objects") or {}
    orphan_items = []
    protected_bytes = 0
    for sha, meta in sorted(objects_map.items()):
        if sha in protected:
            protected_bytes += meta["size_bytes"]
        else:
            orphan_items.append(
                {
                    "id": meta["id"],
                    "sha256": sha,
                    "name": meta["name"],
                    "size_bytes": meta["size_bytes"],
                    "modified_at": meta["modified_at"],
                    "duplicate_ids": meta.get("duplicate_ids") or [],
                }
            )
    orphan_items.sort(key=lambda item: item["size_bytes"], reverse=True)
    return {
        "keep_history": keep,
        "total_objects_count": len(objects_map),
        "total_objects_bytes": sum(m["size_bytes"] for m in objects_map.values()),
        "protected_count": len(objects_map) - len(orphan_items),
        "protected_bytes": protected_bytes,
        "orphan_count": len(orphan_items),
        "orphan_bytes": sum(item["size_bytes"] for item in orphan_items),
        "items": orphan_items,
    }


def _read_env_caches(
    wdrive: _WorkspaceDrive,
    children_map: dict[str, list[dict[str, Any]]],
    env_cache_fid: str | None,
) -> list[dict[str, Any]]:
    if not env_cache_fid:
        return []
    by_key: dict[str, dict[str, Any]] = {}
    for item in children_map.get(env_cache_fid, ()):
        name = str(item.get("name") or "")
        iid = str(item.get("id") or "")
        if not iid:
            continue
        if name.endswith(".tar.gz"):
            key = name[: -len(".tar.gz")]
            slot = by_key.setdefault(key, {"key": key, "tar_files": [], "meta_files": []})
            slot["tar_files"].append(item)
        elif name.endswith(".meta.json"):
            key = name[: -len(".meta.json")]
            slot = by_key.setdefault(key, {"key": key, "tar_files": [], "meta_files": []})
            slot["meta_files"].append(item)

    wdrive.prime_token()

    def _hydrate(slot: dict[str, Any]) -> dict[str, Any]:
        key = slot["key"]
        tar_files = sorted(slot["tar_files"], key=lambda r: _iso_to_epoch(r.get("modifiedTime")) or 0.0, reverse=True)
        meta_files = sorted(slot["meta_files"], key=lambda r: _iso_to_epoch(r.get("modifiedTime")) or 0.0, reverse=True)
        tar_primary = tar_files[0] if tar_files else None
        meta_primary = meta_files[0] if meta_files else None

        meta_doc: dict[str, Any] = {}
        if meta_primary:
            try:
                parsed = wdrive.download_json(meta_primary["id"], max_bytes=2 * 1024 * 1024)
                if isinstance(parsed, dict):
                    meta_doc = parsed
            except Exception:
                meta_doc = {}

        size_bytes = _int_or_zero(tar_primary.get("size")) if tar_primary else _int_or_zero(meta_doc.get("size"))
        schema = meta_doc.get("schema")
        if not isinstance(schema, int):
            m = re.match(r"^native-env-v(\d+)-", key)
            schema = int(m.group(1)) if m else None
        spec = meta_doc.get("spec") if isinstance(meta_doc.get("spec"), dict) else {}
        created_at = (
            _iso_to_epoch(meta_doc.get("created_at"))
            or (_iso_to_epoch(tar_primary.get("modifiedTime")) if tar_primary else None)
            or (_iso_to_epoch(meta_primary.get("modifiedTime")) if meta_primary else None)
        )
        complete = bool(tar_primary and meta_primary)
        flags: list[str] = []
        if not complete:
            flags.append("incomplete")
        if schema is not None and schema < CURRENT_ENV_CACHE_SCHEMA:
            flags.append("legacy_schema")
        if size_bytes > ENV_CACHE_OVERSIZED_BYTES:
            flags.append("oversized")
        if len(tar_files) > 1 or len(meta_files) > 1:
            flags.append("duplicate_files")

        return {
            "key": key,
            "schema": schema,
            "size_bytes": size_bytes,
            "created_at": created_at,
            "complete": complete,
            "flags": flags,
            "comfy_commit": spec.get("comfy_commit"),
            "python": spec.get("python"),
            "gpu_tag": spec.get("gpu"),
            "added_count": meta_doc.get("added_count"),
            "overlay_count": meta_doc.get("overlay_count"),
            "file_ids": [str(f["id"]) for f in [*tar_files, *meta_files]],
        }

    with ThreadPoolExecutor(max_workers=6) as pool:
        caches = list(pool.map(_hydrate, by_key.values()))
    caches.sort(key=lambda item: float(item.get("created_at") or 0.0), reverse=True)
    return caches


def _studio_published_workflows(
    repo,
    owner_id: str,
    *,
    telemetry_node_map: dict[str, str],
    plugin_lookup: dict[str, str],
    known_plugins: set[str],
) -> list[dict[str, Any]]:
    rows = repo.list("workflows", owner_id)
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("state") != "published" and not row.get("active"):
            continue
        graph = row.get("api_graph") or row.get("workflow") or {}
        deps = extract_workflow_dependencies(
            graph,
            telemetry_node_map=telemetry_node_map,
            plugin_lookup=plugin_lookup,
            known_plugins=known_plugins,
        )
        plugins_set = set(deps["plugins"])
        manifest = row.get("manifest") if isinstance(row.get("manifest"), dict) else {}
        for cn in manifest.get("custom_nodes") or []:
            if isinstance(cn, dict):
                matched = _match_plugin_by_meta(cn.get("dir"), cn.get("repo"), plugin_lookup)
                if matched:
                    plugins_set.add(matched)
        name = row.get("workflow_key") or row.get("name") or row.get("id")
        out.append(
            {
                "id": str(row.get("id") or name),
                "source": "studio_db",
                "path": str(name),
                "size_bytes": 0,
                "modified_at": _iso_to_epoch(row.get("updated_at") or row.get("created_at")),
                "node_count": deps["node_count"],
                "plugins": sorted(plugins_set),
                "plugin_nodes": deps["plugin_nodes"],
                "unresolved_nodes": deps["unresolved_nodes"],
                "readonly": True,
            }
        )
    out.sort(key=lambda item: item["path"])
    return out


def _scan_drive_workflows(
    wdrive: _WorkspaceDrive,
    children_map: dict[str, list[dict[str, Any]]],
    native_comfy_fid: str | None,
    *,
    telemetry_node_map: dict[str, str],
    plugin_lookup: dict[str, str],
    known_plugins: set[str],
) -> dict[str, list[dict[str, Any]]]:
    if not native_comfy_fid:
        return {"active": [], "archived": [], "warmup": []}

    user_dir = _tree_find_child(children_map, native_comfy_fid, "user", is_folder=True)
    default_dir = _tree_find_child(children_map, user_dir["id"], "default", is_folder=True) if user_dir else None
    comfy_wf_dir = _tree_find_child(children_map, default_dir["id"], "workflows", is_folder=True) if default_dir else None
    archived_dir = _tree_find_child(children_map, native_comfy_fid, "workflows-archived", is_folder=True)
    warmup_dir = _tree_find_child(children_map, native_comfy_fid, "warmup", is_folder=True)

    tasks: list[tuple[str, dict[str, Any]]] = []
    for meta in _tree_walk_json(children_map, comfy_wf_dir["id"] if comfy_wf_dir else None):
        tasks.append(("comfy", meta))
    for meta in _tree_walk_json(children_map, archived_dir["id"] if archived_dir else None):
        tasks.append(("archived", meta))
    for meta in _tree_walk_json(children_map, warmup_dir["id"] if warmup_dir else None):
        tasks.append(("warmup", meta))

    if not tasks:
        return {"active": [], "archived": [], "warmup": []}

    wdrive.prime_token()

    def _inspect(pair: tuple[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
        source, meta = pair
        parse_error: str | None = None
        deps = {"node_count": 0, "plugins": [], "plugin_nodes": {}, "unresolved_nodes": []}
        try:
            doc = wdrive.download_json(meta["id"], max_bytes=MAX_WORKFLOW_JSON_BYTES, known_size=meta["size_bytes"])
            deps = extract_workflow_dependencies(
                doc,
                telemetry_node_map=telemetry_node_map,
                plugin_lookup=plugin_lookup,
                known_plugins=known_plugins,
            )
        except APIError as exc:
            parse_error = exc.message
        except Exception:
            parse_error = "工作流 JSON 解析失败"
        return source, {
            "id": meta["id"],
            "source": source,
            "path": meta["path"],
            "size_bytes": meta["size_bytes"],
            "modified_at": meta["modified_at"],
            "node_count": deps["node_count"],
            "plugins": deps["plugins"],
            "plugin_nodes": deps["plugin_nodes"],
            "unresolved_nodes": deps["unresolved_nodes"],
            "parse_error": parse_error,
        }

    buckets: dict[str, list[dict[str, Any]]] = {"active": [], "archived": [], "warmup": []}
    with ThreadPoolExecutor(max_workers=6) as pool:
        for source, item in pool.map(_inspect, tasks):
            if source == "comfy":
                buckets["active"].append(item)
            elif source == "archived":
                buckets["archived"].append(item)
            else:
                buckets["warmup"].append(item)
    for bucket in buckets.values():
        bucket.sort(key=lambda item: item["path"])
    return buckets


def _build_overview_payload(
    repo,
    owner_id: str,
    wdrive: _WorkspaceDrive,
    *,
    account_id: str | None,
    account_alias: str | None,
    keep_history: int = DEFAULT_KEEP_HISTORY,
) -> dict[str, Any]:
    saved = repo.store.get("settings", _settings_key(owner_id), owner_id) or {}
    telemetry = saved.get("telemetry") if isinstance(saved.get("telemetry"), dict) else None
    telemetry_node_map = telemetry.get("node_map") if isinstance((telemetry or {}).get("node_map"), dict) else {}
    import_times = telemetry.get("import_times") if isinstance((telemetry or {}).get("import_times"), dict) else {}
    telemetry_plugins = telemetry.get("plugins") if isinstance((telemetry or {}).get("plugins"), dict) else {}

    owner_fid = wdrive.find_owner_folder(create=False)
    children_map: dict[str, list[dict[str, Any]]] = {}
    native_comfy_fid: str | None = None
    plugins_fid: str | None = None
    env_cache_fid: str | None = None

    if owner_fid:
        children_map, _ = _bulk_list_app_entries(wdrive.storage)
        nc_dir = _tree_find_child(children_map, owner_fid, "native-comfy", is_folder=True)
        if nc_dir:
            native_comfy_fid = nc_dir["id"]
            pl_dir = _tree_find_child(children_map, native_comfy_fid, "plugins", is_folder=True)
            if pl_dir:
                plugins_fid = pl_dir["id"]
            ec_dir = _tree_find_child(children_map, native_comfy_fid, "env-cache", is_folder=True)
            if ec_dir:
                env_cache_fid = ec_dir["id"]

    bundle = _read_plugins_bundle(wdrive, children_map, plugins_fid, saved)
    effective_plugins = [p for p in (bundle["effective"].get("plugins") or []) if isinstance(p, dict) and p.get("path")]
    disabled_list = bundle["disabled"]
    objects_map = bundle["objects"]

    all_plugin_paths = [p["path"] for p in effective_plugins] + [d["entry"]["path"] for d in disabled_list]
    known_plugins = set(all_plugin_paths)
    plugin_lookup = _build_plugin_lookup(all_plugin_paths)

    wf_buckets = _scan_drive_workflows(
        wdrive,
        children_map,
        native_comfy_fid,
        telemetry_node_map=telemetry_node_map,
        plugin_lookup=plugin_lookup,
        known_plugins=known_plugins,
    )
    studio_wfs = _studio_published_workflows(
        repo,
        owner_id,
        telemetry_node_map=telemetry_node_map,
        plugin_lookup=plugin_lookup,
        known_plugins=known_plugins,
    )

    used_by_active: dict[str, list[dict[str, str]]] = {}
    used_by_archived: dict[str, list[dict[str, str]]] = {}
    for wf in [*wf_buckets["active"], *wf_buckets["warmup"], *studio_wfs]:
        for p in wf["plugins"]:
            used_by_active.setdefault(p, []).append({"source": wf["source"], "path": wf["path"]})
    for wf in wf_buckets["archived"]:
        for p in wf["plugins"]:
            used_by_archived.setdefault(p, []).append({"source": wf["source"], "path": wf["path"]})

    def _decorate_plugin(entry: dict[str, Any], *, is_disabled_list: bool, disabled_meta: dict[str, Any] | None = None) -> dict[str, Any]:
        path = str(entry["path"])
        archive = str(entry.get("archive") or "")
        obj = objects_map.get(archive)
        archive_size = obj["size_bytes"] if obj else 0
        archive_present = obj is not None
        active_refs = used_by_active.get(path, [])
        archived_refs = used_by_archived.get(path, [])
        t_meta = telemetry_plugins.get(path) or telemetry_plugins.get(_strip_plugin_decorators(path)) or {}
        imp_sec = import_times.get(path)
        if imp_sec is None:
            imp_sec = import_times.get(_strip_plugin_decorators(path))
        js_count = t_meta.get("js_count") if isinstance(t_meta.get("js_count"), int) else None
        node_count = t_meta.get("node_count") if isinstance(t_meta.get("node_count"), int) else None

        if is_disabled_list:
            category = "disabled"
        elif entry.get("enabled") is False:
            category = "manager_disabled"
        elif active_refs:
            category = "in_use"
        elif archived_refs:
            category = "archived_only"
        elif js_count is None or js_count > 0:
            category = "unused_ui"
        else:
            category = "unused"

        return {
            "path": path,
            "version": entry.get("version"),
            "repo": entry.get("repo"),
            "commit": entry.get("commit"),
            "enabled": bool(entry.get("enabled", True)) and not is_disabled_list,
            "archive": archive,
            "archive_size_bytes": archive_size,
            "archive_present": archive_present,
            "status_category": category,
            "used_by_active": active_refs,
            "used_by_archived": archived_refs,
            "import_seconds": imp_sec,
            "node_count": node_count,
            "js_count": js_count,
            "has_requirements": t_meta.get("has_requirements"),
            "has_install_py": t_meta.get("has_install_py"),
            "disabled_at": (disabled_meta or {}).get("disabled_at"),
            "origin": (disabled_meta or {}).get("origin"),
        }

    active_plugin_rows = [_decorate_plugin(p, is_disabled_list=False) for p in effective_plugins]
    disabled_plugin_rows = [_decorate_plugin(d["entry"], is_disabled_list=True, disabled_meta=d) for d in disabled_list]

    head_gen = (bundle["head"] or {}).get("generation")
    pending_gen = (bundle["pending"] or {}).get("generation")
    effective_gen = bundle["effective"].get("generation")
    has_pending_changes = bool(pending_gen and head_gen != pending_gen)

    history_rows = []
    for h in bundle["history"]:
        gen = str(h.get("generation") or "")
        h_plugins = [p for p in (h.get("plugins") or []) if isinstance(p, dict)]
        missing = [str(p.get("path") or p.get("archive")) for p in h_plugins if str(p.get("archive") or "") not in objects_map]
        total_size = sum(objects_map[str(p.get("archive"))]["size_bytes"] for p in h_plugins if str(p.get("archive")) in objects_map)
        history_rows.append(
            {
                "generation": gen,
                "backed_up_at": _iso_to_epoch(h.get("backed_up_at")),
                "plugin_count": len(h_plugins),
                "total_archive_bytes": total_size,
                "is_head": bool(head_gen and gen == head_gen),
                "is_pending": bool(pending_gen and gen == pending_gen),
                "restorable": len(missing) == 0,
                "missing_archives": missing,
            }
        )

    env_caches = _read_env_caches(wdrive, children_map, env_cache_fid)
    orphan_objects = _compute_orphan_objects(bundle, keep_history=keep_history)

    return {
        "drive_connected": True,
        "drive_account_id": account_id,
        "drive_account_alias": account_alias,
        "cached_at": time.time(),
        "plugin_state": {
            "head_generation": head_gen,
            "pending_generation": pending_gen,
            "effective_generation": effective_gen,
            "has_pending_offline_changes": has_pending_changes,
            "telemetry_available": telemetry is not None,
            "telemetry_collected_at": _iso_to_epoch((telemetry or {}).get("collected_at")),
            "telemetry_gpu": (telemetry or {}).get("gpu_reported"),
            "telemetry_generation": (telemetry or {}).get("generation"),
        },
        "workflows": {
            "active": wf_buckets["active"],
            "archived": wf_buckets["archived"],
            "warmup": wf_buckets["warmup"],
            "studio_published": studio_wfs,
        },
        "plugins": {
            "active": active_plugin_rows,
            "disabled": disabled_plugin_rows,
            "history": history_rows,
            "total_active_archive_bytes": sum(r["archive_size_bytes"] for r in active_plugin_rows),
            "total_active_import_seconds": round(
                sum(float(r["import_seconds"]) for r in active_plugin_rows if isinstance(r["import_seconds"], (int, float))),
                2,
            ),
        },
        "env_caches": {
            "oversized_threshold_bytes": ENV_CACHE_OVERSIZED_BYTES,
            "current_schema": CURRENT_ENV_CACHE_SCHEMA,
            "total_bytes": sum(c["size_bytes"] for c in env_caches),
            "items": env_caches,
        },
        "orphan_objects": orphan_objects,
    }


def get_workspace_overview(
    repo,
    owner_id: str,
    *,
    refresh: bool = False,
    keep_history: int = DEFAULT_KEEP_HISTORY,
    drive: DriveStorage | None = None,
) -> dict[str, Any]:
    keep = max(1, min(int(keep_history), MAX_HISTORY_ENTRIES))
    offline_writable, lock_reason = native.workspace_offline_status(repo, owner_id)
    storage, owns_storage, account_id, account_alias = _open_drive(repo, owner_id, drive=drive)
    if storage is None:
        return {
            "offline_writable": offline_writable,
            "lock_reason": lock_reason,
            "drive_connected": False,
            "drive_account_id": None,
            "drive_account_alias": None,
            "cached_at": time.time(),
            "message": "尚未连接 Google Drive 账号，请先在「设置 → Google 账号」完成 Drive 授权。",
            "plugin_state": {
                "head_generation": None,
                "pending_generation": None,
                "effective_generation": None,
                "has_pending_offline_changes": False,
                "telemetry_available": False,
                "telemetry_collected_at": None,
                "telemetry_gpu": None,
                "telemetry_generation": None,
            },
            "workflows": {"active": [], "archived": [], "warmup": [], "studio_published": []},
            "plugins": {"active": [], "disabled": [], "history": [], "total_active_archive_bytes": 0, "total_active_import_seconds": 0.0},
            "env_caches": {"oversized_threshold_bytes": ENV_CACHE_OVERSIZED_BYTES, "current_schema": CURRENT_ENV_CACHE_SCHEMA, "total_bytes": 0, "items": []},
            "orphan_objects": {"keep_history": keep, "total_objects_count": 0, "total_objects_bytes": 0, "protected_count": 0, "protected_bytes": 0, "orphan_count": 0, "orphan_bytes": 0, "items": []},
        }

    cache_key = (owner_id, str(account_id or "default"), keep)
    now = time.time()
    if not refresh and drive is None:
        with _OVERVIEW_LOCK:
            cached = _OVERVIEW_CACHE.get(cache_key)
            if cached and now - cached[0] < OVERVIEW_CACHE_TTL_SECONDS:
                if owns_storage:
                    storage.close()
                return {**cached[1], "offline_writable": offline_writable, "lock_reason": lock_reason}

    try:
        wdrive = _WorkspaceDrive(storage, owner_id)
        payload = _build_overview_payload(
            repo,
            owner_id,
            wdrive,
            account_id=account_id,
            account_alias=account_alias,
            keep_history=keep,
        )
        if drive is None:
            with _OVERVIEW_LOCK:
                _OVERVIEW_CACHE[cache_key] = (time.time(), payload)
        return {**payload, "offline_writable": offline_writable, "lock_reason": lock_reason}
    except (DriveError, GoogleError) as exc:
        raise APIError("WORKSPACE_DRIVE_ERROR", f"读取 Google Drive 工作区失败：{exc}", 502) from exc
    finally:
        if owns_storage:
            storage.close()


def _commit_plugin_manifest(
    repo,
    owner_id: str,
    wdrive: _WorkspaceDrive,
    plugins_fid: str,
    *,
    new_plugins: list[dict[str, Any]],
    new_disabled: list[dict[str, Any]],
    disabled_changed: bool,
    base_snapshot: Any,
    existing_history: list[dict[str, Any]],
) -> dict[str, Any]:
    normalized_plugins = sorted(
        [_strip_manifest_plugin_entry(p) for p in new_plugins],
        key=lambda item: item["path"],
    )
    new_gen = digest(normalized_plugins)
    now = time.time()
    manifest = {
        "schema": 1,
        "plugins": normalized_plugins,
        "snapshot": base_snapshot if isinstance(base_snapshot, dict) else {"git_custom_nodes": {}},
        "backed_up_at": now,
        "generation": new_gen,
    }

    if disabled_changed:
        disabled_payload = {
            "schema": 1,
            "updated_at": now,
            "plugins": sorted(new_disabled, key=lambda row: row["entry"]["path"]),
        }
        wdrive.write_json_inplace(plugins_fid, "disabled.json", disabled_payload)

    history_fid = wdrive.find_child_folder(plugins_fid, "history", create=True)
    assert history_fid is not None
    if not wdrive.find_files_by_name(history_fid, f"{new_gen}.json"):
        wdrive.write_json_inplace(history_fid, f"{new_gen}.json", manifest)
    with _HISTORY_CACHE_LOCK:
        _HISTORY_JSON_CACHE[new_gen] = manifest

    wdrive.write_json_inplace(plugins_fid, "pending.json", manifest)

    # Mirror offline pending state to VPS settings so status/UI reflects immediately.
    key = _settings_key(owner_id)
    with repo.store.dispatch_lock() as acquired:
        if acquired:
            saved = repo.store.get("settings", key, owner_id) or {}
            history = list(saved.get("history") or [])
            if not history:
                history = list(existing_history)
            if not history or history[-1].get("generation") != new_gen:
                history = [*history, manifest][-MAX_HISTORY_ENTRIES:]
            status = dict(saved.get("status") or {})
            head_gen = (saved.get("manifest") or {}).get("generation")
            status["pending_generation"] = new_gen
            status["restart_required"] = bool(head_gen and head_gen != new_gen)
            data = {
                **saved,
                "status": status,
                "history": history,
                "offline_pending": {
                    "generation": new_gen,
                    "updated_at": now,
                    "manifest": manifest,
                },
            }
            if saved:
                repo.store.update("settings", key, data, owner_id)
            else:
                repo.store.create("settings", owner_id, data, record_id=key)

    invalidate_workspace_cache(owner_id)
    invalidate_drive_usage_cache(owner_id)
    return manifest


def _execute_plugin_mutation(
    repo,
    owner_id: str,
    wdrive: _WorkspaceDrive,
    *,
    action: str,
    paths: list[str],
    force_in_use: bool = False,
) -> dict[str, Any]:
    if action not in {"disable", "enable", "remove", "purge_disabled"}:
        raise APIError("WORKSPACE_INVALID_ACTION", "不支持的插件操作类型", 422)
    clean_paths = list(dict.fromkeys(_validate_plugin_path(p) for p in paths))
    if not clean_paths:
        raise APIError("WORKSPACE_INVALID_PLUGIN", "请至少选择一个插件", 422)

    owner_fid = wdrive.find_owner_folder(create=False)
    if not owner_fid:
        raise APIError("WORKSPACE_NOT_FOUND", "Drive 中尚无原生 ComfyUI 插件记录", 404)
    nc_fid = wdrive.find_child_folder(owner_fid, "native-comfy", create=False)
    plugins_fid = wdrive.find_child_folder(nc_fid, "plugins", create=False) if nc_fid else None
    if not plugins_fid:
        raise APIError("WORKSPACE_NOT_FOUND", "Drive 中尚无原生 ComfyUI 插件清单", 404)

    saved = repo.store.get("settings", _settings_key(owner_id), owner_id) or {}
    children_map, _ = _bulk_list_app_entries(wdrive.storage)
    bundle = _read_plugins_bundle(wdrive, children_map, plugins_fid, saved)

    base_manifest = bundle["pending"] or bundle["head"]
    if base_manifest is None:
        raise APIError("WORKSPACE_NOT_FOUND", "尚未找到可编辑的插件清单（head.json / pending.json）", 404)
    _validate_manifest_doc(base_manifest, label="pending/head")

    active_by_path = {str(p["path"]): _strip_manifest_plugin_entry(p) for p in base_manifest["plugins"] if isinstance(p, dict) and p.get("path")}
    disabled_by_path = {d["entry"]["path"]: d for d in bundle["disabled"]}
    objects_map = bundle["objects"]

    # Check active workflow protection when disabling or removing active plugins
    if action in {"disable", "remove"} and not force_in_use:
        overview = _build_overview_payload(
            repo,
            owner_id,
            wdrive,
            account_id=wdrive.storage.account_id,
            account_alias=wdrive.storage.account_id,
        )
        in_use_conflicts: list[dict[str, Any]] = []
        active_lookup = {row["path"]: row for row in overview["plugins"]["active"]}
        for path in clean_paths:
            row = active_lookup.get(path)
            if row and row["used_by_active"]:
                in_use_conflicts.append({"path": path, "used_by_active": row["used_by_active"]})
        if in_use_conflicts:
            names = "、".join(item["path"] for item in in_use_conflicts[:5])
            raise APIError(
                "PLUGIN_IN_USE",
                f"以下插件仍被活跃工作流引用，需二次确认强制操作：{names}",
                409,
            )

    now = time.time()
    changed_paths: list[str] = []
    disabled_changed = False

    if action == "disable":
        for path in clean_paths:
            entry = active_by_path.pop(path, None)
            if entry is None:
                continue
            origin = "manager_migrated" if entry.get("enabled") is False else "studio_offline"
            disabled_by_path[path] = {
                "entry": entry,
                "disabled_at": now,
                "source_generation": base_manifest["generation"],
                "origin": origin,
            }
            disabled_changed = True
            changed_paths.append(path)
    elif action == "enable":
        for path in clean_paths:
            record = disabled_by_path.get(path)
            if record is None:
                continue
            entry = record["entry"]
            archive = str(entry.get("archive") or "")
            if archive not in objects_map:
                raise APIError(
                    "PLUGIN_ARCHIVE_MISSING",
                    f"插件 {path} 的归档对象已不存在，无法离线恢复，请重新在 Manager 安装",
                    409,
                )
            active_by_path[path] = _strip_manifest_plugin_entry(entry)
            disabled_by_path.pop(path, None)
            disabled_changed = True
            changed_paths.append(path)
    elif action == "remove":
        for path in clean_paths:
            removed_any = False
            if path in active_by_path:
                active_by_path.pop(path)
                removed_any = True
            if path in disabled_by_path:
                disabled_by_path.pop(path)
                disabled_changed = True
                removed_any = True
            if removed_any:
                changed_paths.append(path)
    elif action == "purge_disabled":
        for path in clean_paths:
            if path in disabled_by_path:
                disabled_by_path.pop(path)
                disabled_changed = True
                changed_paths.append(path)

    if not changed_paths:
        return {
            "action": action,
            "changed_paths": [],
            "generation": base_manifest["generation"],
            "active_count": len(active_by_path),
            "disabled_count": len(disabled_by_path),
            "no_op": True,
        }

    if action == "purge_disabled":
        wdrive.write_json_inplace(
            plugins_fid,
            "disabled.json",
            {
                "schema": 1,
                "updated_at": now,
                "plugins": sorted(disabled_by_path.values(), key=lambda row: row["entry"]["path"]),
            },
        )
        invalidate_workspace_cache(owner_id)
        invalidate_drive_usage_cache(owner_id)
        return {
            "action": action,
            "changed_paths": changed_paths,
            "generation": base_manifest["generation"],
            "active_count": len(active_by_path),
            "disabled_count": len(disabled_by_path),
            "no_op": False,
        }

    manifest = _commit_plugin_manifest(
        repo,
        owner_id,
        wdrive,
        plugins_fid,
        new_plugins=list(active_by_path.values()),
        new_disabled=list(disabled_by_path.values()),
        disabled_changed=disabled_changed,
        base_snapshot=base_manifest.get("snapshot"),
        existing_history=bundle["history"],
    )
    return {
        "action": action,
        "changed_paths": changed_paths,
        "generation": manifest["generation"],
        "active_count": len(manifest["plugins"]),
        "disabled_count": len(disabled_by_path),
        "no_op": False,
    }


def apply_plugin_action(
    repo,
    owner_id: str,
    *,
    action: str,
    paths: list[str],
    force_in_use: bool = False,
    drive: DriveStorage | None = None,
) -> dict[str, Any]:
    with _offline_write_guard(repo, owner_id, reason=f"plugins:{action}"):
        storage, owns_storage, _, _ = _open_drive(repo, owner_id, drive=drive)
        if storage is None:
            raise APIError("DRIVE_NOT_CONNECTED", "尚未连接 Google Drive 账号", 503)
        try:
            wdrive = _WorkspaceDrive(storage, owner_id)
            return _execute_plugin_mutation(
                repo,
                owner_id,
                wdrive,
                action=action,
                paths=paths,
                force_in_use=force_in_use,
            )
        except APIError:
            raise
        except (DriveError, GoogleError) as exc:
            raise APIError("WORKSPACE_DRIVE_ERROR", f"写入插件清单失败：{exc}", 502) from exc
        finally:
            if owns_storage:
                storage.close()


def apply_workflow_action(
    repo,
    owner_id: str,
    *,
    action: str,
    items: list[dict[str, Any]],
    overwrite: bool = False,
    also_plugins: dict[str, Any] | None = None,
    drive: DriveStorage | None = None,
) -> dict[str, Any]:
    if action not in {"archive", "restore", "trash"}:
        raise APIError("WORKSPACE_INVALID_ACTION", "不支持的工作流操作类型", 422)
    if not isinstance(items, list) or not items:
        raise APIError("WORKSPACE_INVALID_INPUT", "请至少选择一个工作流", 422)

    validated: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        if not isinstance(item, dict):
            raise APIError("WORKSPACE_INVALID_INPUT", "工作流条目格式无效", 422)
        source = item.get("source")
        relpath = _validate_workflow_relpath(item.get("path"))
        if action == "archive" and source != "comfy":
            raise APIError("WORKSPACE_INVALID_ACTION", "仅支持停用归档原生工作流目录中的文件", 422)
        if action == "restore" and source != "archived":
            raise APIError("WORKSPACE_INVALID_ACTION", "仅支持从已停用归档目录恢复工作流", 422)
        if action == "trash" and source not in {"comfy", "archived", "warmup"}:
            raise APIError("WORKSPACE_INVALID_ACTION", "仅支持删除 Drive 中的原生/归档/预热工作流", 422)
        pair = (str(source), relpath)
        if pair not in seen:
            seen.add(pair)
            validated.append(pair)

    if also_plugins is not None:
        if not isinstance(also_plugins, dict):
            raise APIError("WORKSPACE_INVALID_INPUT", "联动插件参数格式无效", 422)
        if also_plugins.get("action") not in {"disable", "remove"}:
            raise APIError("WORKSPACE_INVALID_ACTION", "联动插件仅支持离线停用或卸载", 422)
        if not isinstance(also_plugins.get("paths"), list):
            raise APIError("WORKSPACE_INVALID_INPUT", "联动插件列表必须为数组", 422)

    with _offline_write_guard(repo, owner_id, reason=f"workflows:{action}"):
        storage, owns_storage, _, _ = _open_drive(repo, owner_id, drive=drive)
        if storage is None:
            raise APIError("DRIVE_NOT_CONNECTED", "尚未连接 Google Drive 账号", 503)
        try:
            wdrive = _WorkspaceDrive(storage, owner_id)
            owner_fid = wdrive.find_owner_folder(create=False)
            if not owner_fid:
                raise APIError("WORKSPACE_NOT_FOUND", "Drive 中尚无原生 ComfyUI 目录", 404)
            nc_fid = wdrive.find_child_folder(owner_fid, "native-comfy", create=False)
            if not nc_fid:
                raise APIError("WORKSPACE_NOT_FOUND", "Drive 中尚无原生 ComfyUI 目录", 404)

            source_roots: dict[str, str | None] = {
                "comfy": wdrive.find_folder_path(nc_fid, ("user", "default", "workflows"), create=(action == "restore")),
                "archived": wdrive.find_folder_path(nc_fid, ("workflows-archived",), create=(action == "archive")),
                "warmup": wdrive.find_folder_path(nc_fid, ("warmup",), create=False),
            }

            processed: list[dict[str, Any]] = []
            for source, relpath in validated:
                src_root = source_roots.get(source)
                if not src_root:
                    raise APIError("WORKSPACE_NOT_FOUND", f"未找到工作流：{relpath}", 404)
                located = wdrive.locate_workflow_file(src_root, relpath)
                if not located:
                    raise APIError("WORKSPACE_NOT_FOUND", f"未找到工作流：{relpath}", 404)
                file_meta, src_parent = located

                if action == "trash":
                    wdrive.trash_file(file_meta["id"])
                    processed.append({"source": source, "path": relpath, "action": "trash"})
                else:
                    dst_key = "archived" if action == "archive" else "comfy"
                    dst_root = source_roots[dst_key]
                    assert dst_root is not None
                    parts = relpath.split("/")
                    dst_parent = wdrive.find_folder_path(dst_root, tuple(parts[:-1]), create=True)
                    assert dst_parent is not None
                    existing = wdrive.find_files_by_name(dst_parent, parts[-1])
                    if existing:
                        if not overwrite:
                            raise APIError(
                                "WORKFLOW_CONFLICT",
                                f"目标目录已存在同名工作流：{relpath}",
                                409,
                            )
                        for dup in existing:
                            wdrive.trash_file(dup["id"])
                    wdrive.move_file(file_meta["id"], src_parent, dst_parent)
                    processed.append({"source": source, "path": relpath, "action": action, "target_source": dst_key})

            invalidate_workspace_cache(owner_id)
            invalidate_drive_usage_cache(owner_id)

            plugin_result = None
            if also_plugins and also_plugins.get("paths"):
                plugin_result = _execute_plugin_mutation(
                    repo,
                    owner_id,
                    wdrive,
                    action=str(also_plugins["action"]),
                    paths=list(also_plugins["paths"]),
                    force_in_use=bool(also_plugins.get("force_in_use", False)),
                )

            return {
                "action": action,
                "processed": processed,
                "plugins_result": plugin_result,
            }
        except APIError:
            raise
        except (DriveError, GoogleError) as exc:
            raise APIError("WORKSPACE_DRIVE_ERROR", f"工作流操作失败：{exc}", 502) from exc
        finally:
            if owns_storage:
                storage.close()


def rollback_plugins(
    repo,
    owner_id: str,
    *,
    generation: str,
    drive: DriveStorage | None = None,
) -> dict[str, Any]:
    if not isinstance(generation, str) or not _HEX64_RE.fullmatch(generation):
        raise APIError("WORKSPACE_INVALID_GENERATION", "目标快照 generation 必须为 64 位 SHA256", 422)

    with _offline_write_guard(repo, owner_id, reason="plugins:rollback"):
        storage, owns_storage, _, _ = _open_drive(repo, owner_id, drive=drive)
        if storage is None:
            raise APIError("DRIVE_NOT_CONNECTED", "尚未连接 Google Drive 账号", 503)
        try:
            wdrive = _WorkspaceDrive(storage, owner_id)
            owner_fid = wdrive.find_owner_folder(create=False)
            nc_fid = wdrive.find_child_folder(owner_fid, "native-comfy", create=False) if owner_fid else None
            plugins_fid = wdrive.find_child_folder(nc_fid, "plugins", create=False) if nc_fid else None
            if not plugins_fid:
                raise APIError("WORKSPACE_NOT_FOUND", "Drive 中尚无插件历史记录", 404)

            history_fid = wdrive.find_child_folder(plugins_fid, "history", create=False)
            if not history_fid:
                raise APIError("WORKSPACE_NOT_FOUND", "未找到历史快照目录", 404)
            hist_files = wdrive.find_files_by_name(history_fid, f"{generation}.json")
            if not hist_files:
                raise APIError("WORKSPACE_NOT_FOUND", f"未找到指定历史快照：{generation[:8]}", 404)

            target_manifest = _validate_manifest_doc(
                wdrive.download_json(hist_files[0]["id"], max_bytes=4 * 1024 * 1024),
                label=f"history/{generation[:8]}",
            )
            if target_manifest["generation"] != generation:
                raise APIError("WORKSPACE_MANIFEST_INVALID", "历史快照 generation 与文件内容不匹配", 502)

            saved = repo.store.get("settings", _settings_key(owner_id), owner_id) or {}
            children_map, _ = _bulk_list_app_entries(wdrive.storage)
            bundle = _read_plugins_bundle(wdrive, children_map, plugins_fid, saved)
            objects_map = bundle["objects"]

            missing = [
                str(p.get("path") or p.get("archive"))
                for p in target_manifest["plugins"]
                if isinstance(p, dict) and str(p.get("archive") or "") not in objects_map
            ]
            if missing:
                raise APIError(
                    "ROLLBACK_ARCHIVE_MISSING",
                    f"该历史快照缺少以下插件归档（可能已被清理），无法回滚：{'、'.join(missing[:5])}",
                    409,
                )

            restored_paths = {str(p["path"]) for p in target_manifest["plugins"] if isinstance(p, dict) and p.get("path")}
            remaining_disabled = [d for d in bundle["disabled"] if d["entry"]["path"] not in restored_paths]
            disabled_changed = len(remaining_disabled) != len(bundle["disabled"])

            manifest = _commit_plugin_manifest(
                repo,
                owner_id,
                wdrive,
                plugins_fid,
                new_plugins=target_manifest["plugins"],
                new_disabled=remaining_disabled,
                disabled_changed=disabled_changed,
                base_snapshot=target_manifest.get("snapshot"),
                existing_history=bundle["history"],
            )
            return {
                "generation": manifest["generation"],
                "plugin_count": len(manifest["plugins"]),
            }
        except APIError:
            raise
        except (DriveError, GoogleError) as exc:
            raise APIError("WORKSPACE_DRIVE_ERROR", f"回滚插件快照失败：{exc}", 502) from exc
        finally:
            if owns_storage:
                storage.close()


def delete_env_caches(
    repo,
    owner_id: str,
    *,
    keys: list[str] | None = None,
    delete_all: bool = False,
    drive: DriveStorage | None = None,
) -> dict[str, Any]:
    clean_keys = [k.strip() for k in (keys or []) if isinstance(k, str) and k.strip()]
    if not delete_all and not clean_keys:
        raise APIError("WORKSPACE_INVALID_INPUT", "请指定要删除的缓存 Key 或选择清空全部缓存", 422)
    for k in clean_keys:
        if "/" in k or "\\" in k or ".." in k:
            raise APIError("WORKSPACE_INVALID_INPUT", "缓存 Key 格式无效", 422)

    with _offline_write_guard(repo, owner_id, reason="env_cache:delete"):
        storage, owns_storage, _, _ = _open_drive(repo, owner_id, drive=drive)
        if storage is None:
            raise APIError("DRIVE_NOT_CONNECTED", "尚未连接 Google Drive 账号", 503)
        try:
            wdrive = _WorkspaceDrive(storage, owner_id)
            owner_fid = wdrive.find_owner_folder(create=False)
            nc_fid = wdrive.find_child_folder(owner_fid, "native-comfy", create=False) if owner_fid else None
            ec_fid = wdrive.find_child_folder(nc_fid, "env-cache", create=False) if nc_fid else None
            if not ec_fid:
                return {"deleted_keys": [], "deleted_files": 0, "freed_bytes": 0}

            files = wdrive.list_children(ec_fid)
            target_set = set(clean_keys)
            deleted_keys: set[str] = set()
            deleted_files = 0
            freed_bytes = 0
            for item in files:
                name = str(item.get("name") or "")
                key = None
                if name.endswith(".tar.gz"):
                    key = name[: -len(".tar.gz")]
                elif name.endswith(".meta.json"):
                    key = name[: -len(".meta.json")]
                if not delete_all and (key is None or key not in target_set):
                    continue
                if key is None and not delete_all:
                    continue
                wdrive.delete_file_permanent(str(item["id"]))
                deleted_files += 1
                freed_bytes += _int_or_zero(item.get("size"))
                if key:
                    deleted_keys.add(key)

            invalidate_workspace_cache(owner_id)
            invalidate_drive_usage_cache(owner_id)
            return {
                "deleted_keys": sorted(deleted_keys),
                "deleted_files": deleted_files,
                "freed_bytes": freed_bytes,
            }
        except APIError:
            raise
        except (DriveError, GoogleError) as exc:
            raise APIError("WORKSPACE_DRIVE_ERROR", f"清理构建缓存失败：{exc}", 502) from exc
        finally:
            if owns_storage:
                storage.close()


def cleanup_orphan_objects(
    repo,
    owner_id: str,
    *,
    keep_history: int = DEFAULT_KEEP_HISTORY,
    dry_run: bool = False,
    drive: DriveStorage | None = None,
) -> dict[str, Any]:
    if type(keep_history) is not int or not 1 <= keep_history <= MAX_HISTORY_ENTRIES:
        raise APIError("WORKSPACE_INVALID_INPUT", f"keep_history 必须为 1 到 {MAX_HISTORY_ENTRIES} 之间的整数", 422)

    def _run_scan_and_delete(storage: DriveStorage) -> dict[str, Any]:
        wdrive = _WorkspaceDrive(storage, owner_id)
        owner_fid = wdrive.find_owner_folder(create=False)
        if not owner_fid:
            return {"dry_run": dry_run, "keep_history": keep_history, "deleted_count": 0, "freed_bytes": 0, "items": []}
        saved = repo.store.get("settings", _settings_key(owner_id), owner_id) or {}
        children_map, _ = _bulk_list_app_entries(wdrive.storage)
        nc_dir = _tree_find_child(children_map, owner_fid, "native-comfy", is_folder=True)
        pl_dir = _tree_find_child(children_map, nc_dir["id"], "plugins", is_folder=True) if nc_dir else None
        if not pl_dir:
            return {"dry_run": dry_run, "keep_history": keep_history, "deleted_count": 0, "freed_bytes": 0, "items": []}

        bundle = _read_plugins_bundle(wdrive, children_map, pl_dir["id"], saved)
        report = _compute_orphan_objects(bundle, keep_history=keep_history)
        if dry_run:
            return {
                "dry_run": True,
                "keep_history": report["keep_history"],
                "deleted_count": report["orphan_count"],
                "freed_bytes": report["orphan_bytes"],
                "protected_count": report["protected_count"],
                "items": report["items"],
            }

        deleted_count = 0
        freed_bytes = 0
        for item in report["items"]:
            wdrive.delete_file_permanent(item["id"])
            for dup_id in item.get("duplicate_ids") or []:
                wdrive.delete_file_permanent(dup_id)
            deleted_count += 1
            freed_bytes += item["size_bytes"]

        invalidate_workspace_cache(owner_id)
        invalidate_drive_usage_cache(owner_id)
        return {
            "dry_run": False,
            "keep_history": report["keep_history"],
            "deleted_count": deleted_count,
            "freed_bytes": freed_bytes,
            "protected_count": report["protected_count"],
            "items": report["items"],
        }

    if dry_run:
        storage, owns_storage, _, _ = _open_drive(repo, owner_id, drive=drive)
        if storage is None:
            raise APIError("DRIVE_NOT_CONNECTED", "尚未连接 Google Drive 账号", 503)
        try:
            return _run_scan_and_delete(storage)
        except (DriveError, GoogleError) as exc:
            raise APIError("WORKSPACE_DRIVE_ERROR", f"扫描孤立归档失败：{exc}", 502) from exc
        finally:
            if owns_storage:
                storage.close()

    with _offline_write_guard(repo, owner_id, reason="orphan_objects:cleanup"):
        storage, owns_storage, _, _ = _open_drive(repo, owner_id, drive=drive)
        if storage is None:
            raise APIError("DRIVE_NOT_CONNECTED", "尚未连接 Google Drive 账号", 503)
        try:
            return _run_scan_and_delete(storage)
        except APIError:
            raise
        except (DriveError, GoogleError) as exc:
            raise APIError("WORKSPACE_DRIVE_ERROR", f"清理孤立归档失败：{exc}", 502) from exc
        finally:
            if owns_storage:
                storage.close()
