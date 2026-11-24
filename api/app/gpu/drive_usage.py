"""Read-only Google Drive storage breakdown under AI-Studio/<owner>/."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Any

from .accounts import AccountService
from .drive import FOLDER_MIME, DriveError, DriveStorage
from .google import GoogleError
from .model_library import ModelLibrary

SHORTCUT_MIME = "application/vnd.google-apps.shortcut"
CACHE_TTL_SECONDS = 300.0
MAX_BULK_PAGES = 25

_CACHE_LOCK = threading.Lock()
_CACHE: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}

_CATEGORY_META: tuple[tuple[str, str, str], ...] = (
    ("models_indexed", "模型库已索引", "已在统一模型库登记并追踪的权重文件"),
    ("models_other", "models 目录其他文件", "Drive models/ 中未纳入模型库索引的文件或旧缓存"),
    ("native_env_cache", "原生环境缓存", "native-comfy/env-cache/ 中的 Python 环境增量包"),
    ("native_plugins", "自定义节点归档", "native-comfy/plugins/ 与旧版 custom_nodes 归档"),
    ("native_outputs", "原生 ComfyUI 作品", "native-comfy/output/ 中生成的图片与视频"),
    ("native_workspace", "原生工作流与素材", "native-comfy/user、input 与预热工作流"),
    ("studio_other", "其他 Studio 文件", "runtime-cache、创作输出、工作流快照等其余目录"),
)


def invalidate_drive_usage_cache(owner_id: str | None = None) -> None:
    with _CACHE_LOCK:
        if owner_id is None:
            _CACHE.clear()
            return
        for key in [k for k in _CACHE if k[0] == owner_id]:
            _CACHE.pop(key, None)


def resolve_owner_drive_account(repo, owner_id: str, preferred_account_id: str | None = None) -> tuple[str | None, str | None]:
    """Resolve the Drive storage account id and alias for read-only Drive browsing."""
    accounts = AccountService(repo, owner_id)
    candidates: list[str] = []
    if preferred_account_id:
        candidates.append(preferred_account_id)
    lib = ModelLibrary(repo, owner_id)
    try:
        saved = lib.storage_settings().get("account_id")
        if saved:
            candidates.append(saved)
    finally:
        lib.close()

    from . import native

    lease = native.active_lease(repo, owner_id)
    if lease:
        lease_acc = lease.get("drive_account_id") or lease.get("account_id")
        if lease_acc:
            candidates.append(lease_acc)

    account_list = accounts.list()
    by_id = {row["account_id"]: row for row in account_list}

    for candidate in candidates:
        resolved = accounts.resolve_drive_account_id(candidate)
        if not resolved:
            continue
        row = by_id.get(resolved)
        if row and (row.get("authorization") or {}).get("drive", {}).get("connected"):
            return resolved, row.get("alias") or resolved

    connected = [
        row for row in account_list
        if row.get("enabled", True) is not False
        and (row.get("authorization") or {}).get("drive", {}).get("connected")
    ]
    if connected:
        resolved = accounts.resolve_drive_account_id(connected[0]["account_id"])
        row = by_id.get(resolved or "", connected[0])
        return row["account_id"], row.get("alias") or row["account_id"]

    # Legacy single-account fallback (without explicit account rows)
    from .google import GoogleService

    if GoogleService(repo, owner_id).status().get("drive", {}).get("connected"):
        return None, "默认 Drive 账号"
    return None, None


def _int_or_none(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
        return parsed if parsed >= 0 else None
    except (TypeError, ValueError):
        return None


def _classify_path(rel_parts: tuple[str, ...], file_id: str, indexed_ids: set[str]) -> str:
    if not rel_parts:
        return "studio_other"
    top = rel_parts[0]
    if top == "models":
        return "models_indexed" if file_id in indexed_ids else "models_other"
    if top == "native-comfy":
        sub = rel_parts[1] if len(rel_parts) > 1 else ""
        if sub == "env-cache":
            return "native_env_cache"
        if sub in {"plugins", "custom_nodes"}:
            return "native_plugins"
        if sub == "output":
            return "native_outputs"
        return "native_workspace"
    return "studio_other"


def _fetch_quota(drive: DriveStorage) -> dict[str, int | None] | None:
    try:
        response = drive._request(
            "GET",
            drive.base_url + "/about",
            params={"fields": "storageQuota"},
        )
        if response.status_code < 200 or response.status_code >= 300:
            return None
        payload = drive._json(response)
        raw = payload.get("storageQuota")
        if not isinstance(raw, dict):
            return None
        return {
            "limit": _int_or_none(raw.get("limit")),
            "usage": _int_or_none(raw.get("usage")),
            "usage_in_drive": _int_or_none(raw.get("usageInDrive")),
            "usage_in_drive_trash": _int_or_none(raw.get("usageInDriveTrash")),
        }
    except (DriveError, GoogleError):
        return None


def _find_owner_folder_readonly(drive: DriveStorage, owner_id: str) -> str | None:
    root_id = None
    for row in drive._repo_files():
        if row.get("kind") == "drive_root" and drive._account_matches(row):
            remote_id = row.get("remote_id") or row.get("root_id")
            if isinstance(remote_id, str) and remote_id:
                root_id = remote_id
                break
    if not root_id:
        root_id = drive._find_folder("AI-Studio", "root")
    if not root_id:
        return None
    return drive._find_folder(owner_id, root_id)


def _bulk_list_app_entries(drive: DriveStorage) -> tuple[dict[str, list[dict[str, Any]]], bool]:
    """Fetch all app-visible Drive entries in bulk (drive.file scope keeps this tiny and fast)."""
    children: dict[str, list[dict[str, Any]]] = defaultdict(list)
    page_token: str | None = None
    pages = 0
    truncated = False
    while True:
        if pages >= MAX_BULK_PAGES:
            truncated = True
            break
        params: dict[str, str] = {
            "q": "trashed = false",
            "pageSize": "1000",
            "fields": "nextPageToken,incompleteSearch,files(id,name,mimeType,size,parents,modifiedTime)",
        }
        if page_token:
            params["pageToken"] = page_token
        response = drive._request("GET", drive.base_url + "/files", params=params)
        payload = drive._ensure_ok(response, "空间统计批量读取")
        truncated = truncated or payload.get("incompleteSearch") is True
        pages += 1
        for item in payload.get("files") or []:
            if not isinstance(item, dict):
                continue
            parents = item.get("parents")
            if isinstance(parents, list):
                for parent_id in parents:
                    if isinstance(parent_id, str) and parent_id:
                        children[parent_id].append(item)
        page_token = payload.get("nextPageToken")
        if not page_token:
            break
    return children, truncated


def collect_drive_usage(
    repo,
    owner_id: str,
    *,
    force_refresh: bool = False,
    drive: DriveStorage | None = None,
) -> dict[str, Any]:
    """Return read-only Drive quota and AI-Studio/<owner>/ category breakdown."""
    account_id: str | None
    account_alias: str | None
    if drive is not None:
        account_id = drive.account_id
        account_alias = account_id or "默认 Drive 账号"
    else:
        account_id, account_alias = resolve_owner_drive_account(repo, owner_id)
        if account_id is None and account_alias is None:
            return {
                "connected": False,
                "account_id": None,
                "account_alias": None,
                "cached_at": time.time(),
                "quota": None,
                "studio_total_bytes": 0,
                "studio_file_count": 0,
                "truncated": False,
                "categories": [
                    {"key": key, "label": label, "description": desc, "bytes": 0, "count": 0}
                    for key, label, desc in _CATEGORY_META
                ],
                "message": "尚未连接 Google Drive 账号，请先在「设置」中完成授权。",
            }

    cache_key = (owner_id, str(account_id or "default"))
    now = time.time()
    if not force_refresh and drive is None:
        with _CACHE_LOCK:
            cached = _CACHE.get(cache_key)
            if cached and now - cached[0] < CACHE_TTL_SECONDS:
                return cached[1]

    lib = ModelLibrary(repo, owner_id)
    try:
        indexed_ids = {
            str(m["drive_file_id"])
            for m in lib.models()
            if m.get("drive_file_id")
        }
    finally:
        lib.close()

    storage = drive or DriveStorage(
        {"id": "drive-usage", "owner_id": owner_id},
        repo,
        account_id=account_id,
    )
    owns_storage = drive is None
    try:
        quota = _fetch_quota(storage)
        owner_folder = _find_owner_folder_readonly(storage, owner_id)
        totals: dict[str, dict[str, int]] = {
            key: {"bytes": 0, "count": 0} for key, _, _ in _CATEGORY_META
        }
        seen_files: set[str] = set()
        seen_folders: set[str] = set()
        truncated = False

        if owner_folder:
            children_map, truncated = _bulk_list_app_entries(storage)
            queue: deque[tuple[str, tuple[str, ...]]] = deque([(owner_folder, ())])
            while queue:
                folder_id, prefix = queue.popleft()
                if folder_id in seen_folders:
                    continue
                seen_folders.add(folder_id)
                for item in children_map.get(folder_id, ()):
                    fid = str(item.get("id") or "")
                    name = str(item.get("name") or "")
                    mime = str(item.get("mimeType") or "")
                    if not fid:
                        continue
                    if mime == FOLDER_MIME:
                        if prefix == ("models",) and name == "native-index":
                            continue
                        queue.append((fid, (*prefix, name)))
                        continue
                    if mime == SHORTCUT_MIME or mime.startswith("application/vnd.google-apps."):
                        continue
                    if fid in seen_files:
                        continue
                    seen_files.add(fid)
                    size = _int_or_none(item.get("size")) or 0
                    bucket = _classify_path((*prefix, name), fid, indexed_ids)
                    totals[bucket]["bytes"] += size
                    totals[bucket]["count"] += 1

        categories = [
            {
                "key": key,
                "label": label,
                "description": desc,
                "bytes": totals[key]["bytes"],
                "count": totals[key]["count"],
            }
            for key, label, desc in _CATEGORY_META
        ]
        studio_total_bytes = sum(item["bytes"] for item in categories)
        studio_file_count = sum(item["count"] for item in categories)
        result = {
            "connected": True,
            "account_id": account_id,
            "account_alias": account_alias,
            "cached_at": time.time(),
            "quota": quota,
            "studio_total_bytes": studio_total_bytes,
            "studio_file_count": studio_file_count,
            "truncated": truncated,
            "categories": categories,
        }
        if drive is None:
            with _CACHE_LOCK:
                _CACHE[cache_key] = (result["cached_at"], result)
        return result
    except (DriveError, GoogleError) as exc:
        return {
            "connected": False,
            "account_id": account_id,
            "account_alias": account_alias,
            "cached_at": time.time(),
            "quota": None,
            "studio_total_bytes": 0,
            "studio_file_count": 0,
            "truncated": False,
            "categories": [
                {"key": key, "label": label, "description": desc, "bytes": 0, "count": 0}
                for key, label, desc in _CATEGORY_META
            ],
            "message": str(exc)[:200],
        }
    finally:
        if owns_storage and storage._owns_client:
            storage.client.close()
