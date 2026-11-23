"""Browse, favorite, thumbnail-cache, stream, import, and trash Native ComfyUI Drive outputs."""

from __future__ import annotations

import hashlib
import io
import re
import threading
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
from PIL import Image

from app.common import APIError, own
from app.config import settings
from app.presenters import public_asset
from app.storage import save_asset

from .drive import FOLDER_MIME, DriveError, DriveStorage
from .drive_usage import invalidate_drive_usage_cache, resolve_owner_drive_account
from .google import GoogleError

FILE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,200}$")
MEDIA_MIMES = frozenset({
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
    "video/mp4",
    "video/webm",
})
EXT_TO_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
}

LIST_CACHE_TTL = 60.0
MAX_OUTPUT_FILES = 3000
MAX_OUTPUT_FOLDERS = 250
MAX_OUTPUT_DEPTH = 6
MAX_BULK_PAGES = 25
MAX_THUMB_SOURCE_BYTES = 32 * 1024 * 1024

_LIST_LOCK = threading.Lock()
_LIST_CACHE: dict[tuple[str, str | None], tuple[float, list[dict[str, Any]], str | None]] = {}
_SCAN_LOCKS_GUARD = threading.Lock()
_SCAN_LOCKS: dict[tuple[str, str | None], threading.Lock] = {}


def _scan_lock_for(cache_key: tuple[str, str | None]) -> threading.Lock:
    with _SCAN_LOCKS_GUARD:
        lock = _SCAN_LOCKS.get(cache_key)
        if lock is None:
            lock = threading.Lock()
            _SCAN_LOCKS[cache_key] = lock
        return lock


def invalidate_native_outputs_cache(owner_id: str | None = None) -> None:
    with _LIST_LOCK:
        if owner_id is None:
            _LIST_CACHE.clear()
            return
        for key in [k for k in _LIST_CACHE if k[0] == owner_id]:
            _LIST_CACHE.pop(key, None)


def _favorites_key(owner_id: str) -> str:
    return "native-output-favorites:" + hashlib.sha256(owner_id.encode()).hexdigest()[:32]


def _load_favorites(store, owner_id: str) -> set[str]:
    row = store.get("settings", _favorites_key(owner_id), owner_id) or {}
    ids = row.get("ids")
    if isinstance(ids, dict):
        return {str(k) for k, v in ids.items() if v}
    if isinstance(ids, list):
        return {str(k) for k in ids if isinstance(k, str)}
    return set()


def _save_favorites(store, owner_id: str, fav_ids: set[str]) -> None:
    key = _favorites_key(owner_id)
    payload = {"kind": "native_output_favorites", "ids": {fid: True for fid in sorted(fav_ids)}}
    existing = store.get("settings", key, owner_id)
    if existing:
        store.update("settings", key, payload, owner_id)
    else:
        store.create("settings", owner_id, payload, record_id=key)


def _infer_media_mime(name: str, mime: str) -> str | None:
    if mime in MEDIA_MIMES:
        return mime
    suffix = Path(name).suffix.lower()
    return EXT_TO_MIME.get(suffix)


def _int_or_none(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
        return parsed if parsed >= 0 else None
    except (TypeError, ValueError):
        return None


def _thumb_dir(owner_id: str) -> Path:
    owner_hash = hashlib.sha256(owner_id.encode()).hexdigest()[:20]
    path = (settings.data_dir / "native-thumb-cache" / owner_hash).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _thumb_path(owner_id: str, file_id: str, revision: str) -> Path:
    safe_rev = re.sub(r"[^A-Za-z0-9_-]+", "", revision)[:32] or "v0"
    root = _thumb_dir(owner_id)
    target = (root / f"{file_id}-{safe_rev}.webp").resolve()
    if not target.is_relative_to(root):
        raise APIError("INVALID_FILE_ID", "无效的缩略图路径", 400)
    return target


def _clear_thumbs(owner_id: str, file_id: str) -> None:
    root = _thumb_dir(owner_id)
    for item in root.glob(f"{file_id}-*.webp"):
        item.unlink(missing_ok=True)


def _find_owner_folder(drive: DriveStorage, owner_id: str) -> str | None:
    root_id = None
    files = drive._repo_files()
    for row in files:
        if row.get("kind") == "drive_folder" and row.get("folder_key") == "owner" and drive._account_matches(row):
            remote_id = row.get("remote_id")
            if isinstance(remote_id, str) and remote_id:
                return remote_id
    for row in files:
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


def _find_output_folder(drive: DriveStorage, owner_id: str) -> str | None:
    owner_folder = _find_owner_folder(drive, owner_id)
    if not owner_folder:
        return None
    native_folder = drive._find_folder("native-comfy", owner_folder)
    if not native_folder:
        return None
    return drive._find_folder("output", native_folder)


def _build_output_item(
    entry: dict[str, Any],
    subfolder: str,
    account_id: str | None,
    account_alias: str | None,
) -> dict[str, Any] | None:
    fid = str(entry.get("id") or "")
    name = str(entry.get("name") or "")
    raw_mime = str(entry.get("mimeType") or "")
    if not fid or not name:
        return None
    mime = _infer_media_mime(name, raw_mime)
    if not mime:
        return None
    img_meta = entry.get("imageMediaMetadata") if isinstance(entry.get("imageMediaMetadata"), dict) else {}
    vid_meta = entry.get("videoMediaMetadata") if isinstance(entry.get("videoMediaMetadata"), dict) else {}
    width = _int_or_none(img_meta.get("width") or vid_meta.get("width"))
    height = _int_or_none(img_meta.get("height") or vid_meta.get("height"))
    rel_path = f"{subfolder}/{name}" if subfolder else name
    md5 = entry.get("md5Checksum")
    modified = entry.get("modifiedTime") or entry.get("createdTime")
    rev = str(md5 or modified or "0")
    return {
        "id": fid,
        "filename": name,
        "subfolder": subfolder,
        "path": rel_path,
        "mime": mime,
        "size": _int_or_none(entry.get("size")) or 0,
        "width": width,
        "height": height,
        "md5": md5,
        "modified_at": modified,
        "created_at": entry.get("createdTime"),
        "thumbnail_link": entry.get("thumbnailLink"),
        "revision": rev,
        "thumbnail_url": f"/api/gpu/native/outputs/{quote(fid, safe='')}/thumbnail?v={quote(rev[:16], safe='')}",
        "content_url": f"/api/gpu/native/outputs/{quote(fid, safe='')}/content",
        "account_id": account_id,
        "account_alias": account_alias,
    }


def _bulk_scan_output_tree(
    drive: DriveStorage,
    owner_folder: str,
    account_id: str | None,
    account_alias: str | None,
) -> tuple[list[dict[str, Any]], str | None] | None:
    """Fetch app-visible Drive entries in bulk (1-2 API calls instead of N subfolder BFS calls)."""
    fields = (
        "nextPageToken,files("
        "id,name,mimeType,size,parents,md5Checksum,createdTime,modifiedTime,thumbnailLink,"
        "imageMediaMetadata(width,height),videoMediaMetadata(width,height,durationMillis)"
        ")"
    )
    children: dict[str, list[dict[str, Any]]] = defaultdict(list)
    page_token: str | None = None
    pages = 0
    while True:
        if pages >= MAX_BULK_PAGES:
            return None
        params: dict[str, str] = {
            "q": "trashed = false",
            "pageSize": "1000",
            "fields": fields,
        }
        if page_token:
            params["pageToken"] = page_token
        response = drive._request("GET", drive.base_url + "/files", params=params)
        payload = drive._ensure_ok(response, "原生作品目录读取")
        pages += 1
        for entry in payload.get("files") or []:
            if not isinstance(entry, dict):
                continue
            parents = entry.get("parents")
            if isinstance(parents, list):
                for parent_id in parents:
                    if isinstance(parent_id, str) and parent_id:
                        children[parent_id].append(entry)
        page_token = payload.get("nextPageToken")
        if not page_token:
            break

    native_folder = next(
        (e["id"] for e in children.get(owner_folder, ()) if e.get("mimeType") == FOLDER_MIME and e.get("name") == "native-comfy" and e.get("id")),
        None,
    )
    if not native_folder:
        return [], None
    output_folder = next(
        (e["id"] for e in children.get(native_folder, ()) if e.get("mimeType") == FOLDER_MIME and e.get("name") == "output" and e.get("id")),
        None,
    )
    if not output_folder:
        return [], None

    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    queue: deque[tuple[str, str, int]] = deque([(output_folder, "", 0)])
    visited_folders = 0
    while queue and visited_folders < MAX_OUTPUT_FOLDERS:
        folder_id, subfolder, depth = queue.popleft()
        visited_folders += 1
        for entry in children.get(folder_id, ()):
            fid = str(entry.get("id") or "")
            name = str(entry.get("name") or "")
            if not fid or not name or fid in seen:
                continue
            if entry.get("mimeType") == FOLDER_MIME:
                if depth < MAX_OUTPUT_DEPTH:
                    next_sub = f"{subfolder}/{name}" if subfolder else name
                    queue.append((fid, next_sub, depth + 1))
                continue
            built = _build_output_item(entry, subfolder, account_id, account_alias)
            if built is not None:
                seen.add(fid)
                items.append(built)
    items.sort(key=lambda row: str(row.get("modified_at") or ""), reverse=True)
    return items[:MAX_OUTPUT_FILES], output_folder


def _scan_output_tree(
    drive: DriveStorage,
    owner_id: str,
    account_id: str | None,
    account_alias: str | None,
    *,
    force_refresh: bool = False,
) -> tuple[list[dict[str, Any]], str | None, float]:
    cache_key = (owner_id, account_id)
    now = time.time()
    if not force_refresh:
        with _LIST_LOCK:
            cached = _LIST_CACHE.get(cache_key)
            if cached and now - cached[0] < LIST_CACHE_TTL:
                return cached[1], cached[2], cached[0]

    # Single-flight lock per (owner, account) prevents concurrent thumbnail requests
    # from triggering duplicate Drive directory scans on a cold cache.
    with _scan_lock_for(cache_key):
        now = time.time()
        if not force_refresh:
            with _LIST_LOCK:
                cached = _LIST_CACHE.get(cache_key)
                if cached and now - cached[0] < LIST_CACHE_TTL:
                    return cached[1], cached[2], cached[0]

        owner_folder = _find_owner_folder(drive, owner_id)
        if not owner_folder:
            with _LIST_LOCK:
                _LIST_CACHE[cache_key] = (now, [], None)
            return [], None, now

        bulk = _bulk_scan_output_tree(drive, owner_folder, account_id, account_alias)
        if bulk is not None:
            items, output_folder = bulk
            scanned_at = time.time()
            with _LIST_LOCK:
                _LIST_CACHE[cache_key] = (scanned_at, items, output_folder)
            return items, output_folder, scanned_at

        native_folder = drive._find_folder("native-comfy", owner_folder)
        output_folder = drive._find_folder("output", native_folder) if native_folder else None
        if not output_folder:
            with _LIST_LOCK:
                _LIST_CACHE[cache_key] = (now, [], None)
            return [], None, now

        items: list[dict[str, Any]] = []
        seen: set[str] = set()
        queue: deque[tuple[str, str, int]] = deque([(output_folder, "", 0)])
        visited_folders = 0
        fields = (
            "nextPageToken,files("
            "id,name,mimeType,size,md5Checksum,createdTime,modifiedTime,thumbnailLink,"
            "imageMediaMetadata(width,height),videoMediaMetadata(width,height,durationMillis)"
            ")"
        )

        while queue and visited_folders < MAX_OUTPUT_FOLDERS and len(items) < MAX_OUTPUT_FILES:
            folder_id, subfolder, depth = queue.popleft()
            visited_folders += 1
            page_token: str | None = None
            while len(items) < MAX_OUTPUT_FILES:
                params: dict[str, str] = {
                    "q": f"'{drive._q(folder_id)}' in parents and trashed = false",
                    "pageSize": "200",
                    "orderBy": "modifiedTime desc",
                    "fields": fields,
                }
                if page_token:
                    params["pageToken"] = page_token
                response = drive._request("GET", drive.base_url + "/files", params=params)
                payload = drive._ensure_ok(response, "原生作品目录读取")
                for entry in payload.get("files") or []:
                    if not isinstance(entry, dict):
                        continue
                    fid = str(entry.get("id") or "")
                    name = str(entry.get("name") or "")
                    raw_mime = str(entry.get("mimeType") or "")
                    if not fid or not name or fid in seen:
                        continue
                    if raw_mime == FOLDER_MIME:
                        if depth < MAX_OUTPUT_DEPTH:
                            next_sub = f"{subfolder}/{name}" if subfolder else name
                            queue.append((fid, next_sub, depth + 1))
                        continue
                    built = _build_output_item(entry, subfolder, account_id, account_alias)
                    if built is None:
                        continue
                    seen.add(fid)
                    items.append(built)
                    if len(items) >= MAX_OUTPUT_FILES:
                        break
                page_token = payload.get("nextPageToken")
                if not page_token:
                    break

        items.sort(key=lambda row: str(row.get("modified_at") or ""), reverse=True)
        scanned_at = time.time()
        with _LIST_LOCK:
            _LIST_CACHE[cache_key] = (scanned_at, items, output_folder)
        return items, output_folder, scanned_at


def _public_output_item(item: dict[str, Any], favorites: set[str]) -> dict[str, Any]:
    return {
        "id": item["id"],
        "filename": item["filename"],
        "subfolder": item["subfolder"],
        "path": item["path"],
        "mime": item["mime"],
        "size": item["size"],
        "width": item.get("width"),
        "height": item.get("height"),
        "md5": item.get("md5"),
        "modified_at": item.get("modified_at"),
        "created_at": item.get("created_at"),
        "favorite": item["id"] in favorites,
        "thumbnail_url": item["thumbnail_url"],
        "content_url": item["content_url"],
        "account_id": item.get("account_id"),
        "account_alias": item.get("account_alias"),
    }


def _open_drive(repo, owner_id: str, *, drive: DriveStorage | None = None) -> tuple[DriveStorage, str | None, str | None, bool]:
    if drive is not None:
        return drive, drive.account_id, drive.account_id or "默认 Drive 账号", False
    account_id, account_alias = resolve_owner_drive_account(repo, owner_id)
    if account_id is None and account_alias is None:
        raise APIError("DRIVE_NOT_CONNECTED", "尚未连接 Google Drive 账号，请先在「设置」中完成授权", 409)
    storage = DriveStorage({"id": "native-outputs", "owner_id": owner_id}, repo, account_id=account_id)
    return storage, account_id, account_alias, True


def list_native_outputs(
    repo,
    owner_id: str,
    *,
    refresh: bool = False,
    limit: int = 48,
    offset: int = 0,
    media_type: str = "all",
    favorite: str = "all",
    query: str = "",
    subfolder: str = "",
    drive: DriveStorage | None = None,
) -> dict[str, Any]:
    try:
        storage, account_id, account_alias, owns = _open_drive(repo, owner_id, drive=drive)
    except APIError as exc:
        if exc.code == "DRIVE_NOT_CONNECTED":
            return {
                "connected": False,
                "account_id": None,
                "account_alias": None,
                "items": [],
                "total": 0,
                "total_all": 0,
                "subfolders": [],
                "has_more": False,
                "cached_at": time.time(),
                "message": exc.message,
            }
        raise
    try:
        raw_items, _, cached_at = _scan_output_tree(
            storage, owner_id, account_id, account_alias, force_refresh=refresh,
        )
        favorites = _load_favorites(repo.store, owner_id)
        subfolders = sorted({row["subfolder"] for row in raw_items if row.get("subfolder")})
        q = query.strip().lower()
        filtered = []
        for row in raw_items:
            is_fav = row["id"] in favorites
            if media_type == "image" and not row["mime"].startswith("image/"):
                continue
            if media_type == "video" and not row["mime"].startswith("video/"):
                continue
            if favorite == "only" and not is_fav:
                continue
            if favorite == "unfavorited" and is_fav:
                continue
            if subfolder and row["subfolder"] != subfolder:
                continue
            if q and q not in f"{row['filename']} {row['subfolder']}".lower():
                continue
            filtered.append(_public_output_item(row, favorites))
        bounded_limit = max(1, min(int(limit), 200))
        bounded_offset = max(0, int(offset))
        page = filtered[bounded_offset : bounded_offset + bounded_limit]
        return {
            "connected": True,
            "account_id": account_id,
            "account_alias": account_alias,
            "items": page,
            "total": len(filtered),
            "total_all": len(raw_items),
            "subfolders": subfolders,
            "has_more": bounded_offset + len(page) < len(filtered),
            "cached_at": cached_at,
        }
    except (DriveError, GoogleError) as exc:
        return {
            "connected": False,
            "account_id": account_id,
            "account_alias": account_alias,
            "items": [],
            "total": 0,
            "total_all": 0,
            "subfolders": [],
            "has_more": False,
            "cached_at": time.time(),
            "message": str(exc)[:200],
        }
    finally:
        if owns and storage._owns_client:
            storage.client.close()


def resolve_output_item(
    repo,
    owner_id: str,
    file_id: str,
    *,
    drive: DriveStorage | None = None,
) -> tuple[dict[str, Any], DriveStorage, bool]:
    """Verify that file_id resides inside AI-Studio/<owner>/native-comfy/output/."""
    if not FILE_ID_RE.fullmatch(file_id or ""):
        raise APIError("INVALID_FILE_ID", "无效的原生作品文件 ID", 400)
    storage, account_id, account_alias, owns = _open_drive(repo, owner_id, drive=drive)
    try:
        items, _, _ = _scan_output_tree(storage, owner_id, account_id, account_alias, force_refresh=False)
        match = next((row for row in items if row["id"] == file_id), None)
        if match is None:
            items, _, _ = _scan_output_tree(storage, owner_id, account_id, account_alias, force_refresh=True)
            match = next((row for row in items if row["id"] == file_id), None)
        if match is None:
            if owns and storage._owns_client:
                storage.client.close()
            raise APIError("NOT_FOUND", "原生作品不存在或不在 native-comfy/output 目录内", 404)
        return match, storage, owns
    except BaseException:
        if owns and storage._owns_client:
            storage.client.close()
        raise


def set_native_output_favorite(
    repo,
    owner_id: str,
    file_id: str,
    favorite: bool,
    *,
    drive: DriveStorage | None = None,
) -> dict[str, Any]:
    item, storage, owns = resolve_output_item(repo, owner_id, file_id, drive=drive)
    try:
        favs = _load_favorites(repo.store, owner_id)
        if favorite:
            favs.add(file_id)
        else:
            favs.discard(file_id)
        _save_favorites(repo.store, owner_id, favs)
        return _public_output_item(item, favs)
    finally:
        if owns and storage._owns_client:
            storage.client.close()


def _encode_webp_thumbnail(raw_bytes: bytes) -> bytes:
    with Image.open(io.BytesIO(raw_bytes)) as image:
        image.thumbnail((480, 480))
        out = io.BytesIO()
        mode = "RGBA" if image.mode in {"RGBA", "LA"} else "RGB"
        image.convert(mode).save(out, "WEBP", quality=80)
        return out.getvalue()


def get_native_output_thumbnail(
    repo,
    owner_id: str,
    file_id: str,
    *,
    drive: DriveStorage | None = None,
) -> tuple[bytes, str]:
    item, storage, owns = resolve_output_item(repo, owner_id, file_id, drive=drive)
    try:
        cache_file = _thumb_path(owner_id, file_id, str(item.get("revision") or "0"))
        if cache_file.is_file():
            return cache_file.read_bytes(), "image/webp"

        raw_thumb: bytes | None = None
        thumb_link = item.get("thumbnail_link")
        if isinstance(thumb_link, str) and thumb_link.startswith("https://"):
            resized_link = re.sub(r"=s\d+$", "=s480", thumb_link)
            token = storage.google.access_token("drive")
            for headers in ({"Authorization": "Bearer " + token}, {}):
                try:
                    resp = storage.client.get(resized_link, headers=headers, timeout=15.0, follow_redirects=True)
                    if resp.status_code == 200 and resp.content and len(resp.content) <= MAX_THUMB_SOURCE_BYTES:
                        ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
                        if ctype.startswith("image/"):
                            raw_thumb = resp.content
                            break
                except httpx.HTTPError:
                    continue

        if raw_thumb is None and item["mime"].startswith("image/"):
            if int(item.get("size") or 0) <= MAX_THUMB_SOURCE_BYTES:
                resp = storage._request(
                    "GET",
                    f"{storage.base_url}/files/{quote(file_id, safe='')}",
                    params={"alt": "media"},
                    headers={"Accept": "*/*"},
                )
                if resp.status_code == 200 and resp.content and len(resp.content) <= MAX_THUMB_SOURCE_BYTES:
                    raw_thumb = resp.content

        if raw_thumb is None:
            raise APIError("THUMBNAIL_UNAVAILABLE", "暂时无法生成该原生作品的缩略图", 404)

        try:
            webp = _encode_webp_thumbnail(raw_thumb)
        except Exception as exc:  # noqa: BLE001
            raise APIError("THUMBNAIL_UNAVAILABLE", "无法解码原生作品缩略图", 404) from exc

        tmp = cache_file.with_name(f"{cache_file.name}.{uuid.uuid4().hex}.tmp")
        try:
            tmp.write_bytes(webp)
            tmp.replace(cache_file)
        except OSError:
            tmp.unlink(missing_ok=True)
        return webp, "image/webp"
    finally:
        if owns and storage._owns_client:
            storage.client.close()


def download_native_output_bytes(
    repo,
    owner_id: str,
    file_id: str,
    *,
    max_bytes: int | None = None,
    drive: DriveStorage | None = None,
) -> tuple[dict[str, Any], bytes]:
    item, storage, owns = resolve_output_item(repo, owner_id, file_id, drive=drive)
    try:
        limit = max_bytes or settings.max_upload_bytes
        declared = int(item.get("size") or 0)
        if declared > limit:
            raise APIError("PAYLOAD_TOO_LARGE", "原生作品超过大小限制，无法导入", 413)
        token = storage.google.access_token("drive")
        url = f"{storage.base_url}/files/{quote(file_id, safe='')}?alt=media"
        buf = bytearray()
        with storage.client.stream("GET", url, headers={"Authorization": "Bearer " + token}, timeout=60.0) as resp:
            if resp.status_code < 200 or resp.status_code >= 300:
                raise APIError("DRIVE_DOWNLOAD_FAILED", f"读取 Drive 原生作品失败（HTTP {resp.status_code}）", 502)
            for chunk in resp.iter_bytes(65536):
                buf.extend(chunk)
                if len(buf) > limit:
                    raise APIError("PAYLOAD_TOO_LARGE", "原生作品超过大小限制，无法导入", 413)
        return item, bytes(buf)
    finally:
        if owns and storage._owns_client:
            storage.client.close()


def import_native_output_as_reference(
    repo,
    owner_id: str,
    file_id: str,
    project_id: str,
    *,
    drive: DriveStorage | None = None,
) -> dict[str, Any]:
    own(repo.store, "projects", project_id, {"id": owner_id})
    item, content = download_native_output_bytes(
        repo, owner_id, file_id, max_bytes=settings.max_upload_bytes, drive=drive,
    )
    asset = save_asset(
        repo.store,
        owner_id,
        project_id,
        content,
        item["filename"],
        metadata={
            "native_drive_file_id": item["id"],
            "native_subfolder": item["subfolder"],
            "native_source": "drive",
        },
    )
    return public_asset(asset)


def _verify_output_parent_chain(storage: DriveStorage, owner_id: str, file_id: str) -> None:
    output_folder = _find_output_folder(storage, owner_id)
    current_id = file_id
    # The gallery scans at most MAX_OUTPUT_DEPTH folders beneath the output root.
    for depth in range(MAX_OUTPUT_DEPTH + 1):
        if not output_folder or not FILE_ID_RE.fullmatch(current_id):
            break
        response = storage._request(
            "GET",
            f"{storage.base_url}/files/{quote(current_id, safe='')}",
            params={"fields": "name,parents,trashed,mimeType"},
        )
        if response.status_code == 404:
            break
        metadata = storage._ensure_ok(response, "原生作品目录验证")
        mime = metadata.get("mimeType") or ""
        expected_type = _infer_media_mime(metadata.get("name") or "", mime) if depth == 0 else mime == FOLDER_MIME
        if metadata.get("trashed") or not expected_type:
            break
        parents = metadata.get("parents")
        if not isinstance(parents, list) or len(parents) != 1 or not isinstance(parents[0], str):
            break
        current_id = parents[0]
        if current_id == output_folder:
            return
    raise APIError("NOT_FOUND", "原生作品不存在或不在 native-comfy/output 目录内", 404)


def trash_native_output(
    repo,
    owner_id: str,
    file_id: str,
    *,
    drive: DriveStorage | None = None,
) -> dict[str, Any]:
    item, storage, owns = resolve_output_item(repo, owner_id, file_id, drive=drive)
    try:
        _verify_output_parent_chain(storage, owner_id, file_id)
        response = storage._request(
            "PATCH",
            f"{storage.base_url}/files/{quote(file_id, safe='')}",
            json_body={"trashed": True},
            params={"fields": "id,trashed"},
            retryable=False,
        )
        if response.status_code != 404:
            storage._ensure_ok(response, "移入回收站")
        favs = _load_favorites(repo.store, owner_id)
        if file_id in favs:
            favs.discard(file_id)
            _save_favorites(repo.store, owner_id, favs)
        _clear_thumbs(owner_id, file_id)
        invalidate_native_outputs_cache(owner_id)
        invalidate_drive_usage_cache(owner_id)
        return {"deleted": True, "id": file_id, "filename": item["filename"]}
    finally:
        if owns and storage._owns_client:
            storage.client.close()
