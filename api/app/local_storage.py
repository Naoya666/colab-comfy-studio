"""Local VPS/host storage breakdown, safe previewed cleanup, SQLite VACUUM, and snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import text

from .asset_routes import _asset_thumb_dir, _delete_asset_thumbs
from .batch_zip import STREAM_CHUNK_BYTES, open_zip_spool
from .common import APIError, own
from .config import settings
from .gpu.model_library import ACTIVE as ACTIVE_MODEL_DOWNLOAD_STATES
from .gpu.model_library import transfer_lock
from .gpu.repository import Repository
from .storage import asset_path, delete_asset_file

_TERMINAL_JOB_STATES = frozenset({
    "completed",
    "succeeded",
    "failed",
    "failed_final",
    "cancelled",
    "cancelled_local",
    "blocked",
})
_TERMINAL_EXECUTION_STATES = frozenset({"completed", "succeeded", "failed", "cancelled"})
_PARTIAL_GRACE_SECONDS = 600.0  # Never touch .partial uploads newer than 10 minutes
_SCRATCH_GRACE_SECONDS = 60.0
_MAX_OLDER_THAN_DAYS = 3650
_CLEANUP_ACTIONS = frozenset({
    "thumb_cache",
    "temp_and_orphans",
    "diagnostics",
    "unfavorited_outputs",
})


def _parse_iso_epoch(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return None


def _is_diagnostic_asset(asset: dict) -> bool:
    if str(asset.get("mime") or "").lower() == "application/json":
        return True
    if asset.get("system_smoke") is True:
        return True
    meta = asset.get("metadata")
    return isinstance(meta, dict) and meta.get("system_smoke") is True


def _is_generated_asset(asset: dict) -> bool:
    return bool(asset.get("source_job_id") or asset.get("gpu_job_id"))


def _safe_stat_size(path: Path) -> tuple[int, bool]:
    try:
        if path.is_symlink():
            return 0, False
        st = path.lstat()
        return int(st.st_size), True
    except OSError:
        return 0, False


def _iter_regular_files(root: Path) -> list[tuple[Path, int, float]]:
    """Walk regular non-symlink files under root safely."""
    if not root.is_dir() or root.is_symlink():
        return []
    found: list[tuple[Path, int, float]] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dir_p = Path(dirpath)
        # Prune symlinked subdirectories defensively
        dirnames[:] = [d for d in dirnames if not (dir_p / d).is_symlink()]
        for name in filenames:
            fp = dir_p / name
            try:
                if fp.is_symlink():
                    continue
                st = fp.lstat()
                if not os.path.isfile(fp):
                    continue
                found.append((fp, int(st.st_size), float(st.st_mtime)))
            except OSError:
                continue
    return found


def _sqlite_db_path(st) -> Path | None:
    engine = getattr(st, "engine", None)
    if engine is None or engine.dialect.name != "sqlite":
        return None
    database = engine.url.database
    if not database or database == ":memory:":
        return None
    return Path(database).expanduser().resolve()


def _sqlite_storage_info(st) -> dict[str, Any]:
    engine = getattr(st, "engine", None)
    dialect = engine.dialect.name if engine is not None else "unknown"
    db_path = _sqlite_db_path(st)
    if db_path is None:
        reason = (
            "当前为内存测试数据库，不支持空间回收与快照下载。"
            if dialect == "sqlite"
            else "当前使用外部数据库（如 PostgreSQL），请使用 pg_dump -Fc 进行备份维护。"
        )
        return {
            "dialect": dialect,
            "supported": False,
            "db_bytes": 0,
            "wal_bytes": 0,
            "shm_bytes": 0,
            "total_bytes": 0,
            "reclaimable_bytes": 0,
            "page_size": 0,
            "page_count": 0,
            "freelist_count": 0,
            "message": reason,
        }

    db_bytes, _ = _safe_stat_size(db_path)
    wal_bytes, _ = _safe_stat_size(db_path.with_name(db_path.name + "-wal"))
    shm_bytes, _ = _safe_stat_size(db_path.with_name(db_path.name + "-shm"))
    page_size = page_count = freelist_count = 0
    try:
        with engine.connect() as conn:
            page_size = int(conn.execute(text("PRAGMA page_size")).scalar() or 0)
            page_count = int(conn.execute(text("PRAGMA page_count")).scalar() or 0)
            freelist_count = int(conn.execute(text("PRAGMA freelist_count")).scalar() or 0)
    except Exception:
        pass
    return {
        "dialect": "sqlite",
        "supported": True,
        "filename": db_path.name,
        "db_bytes": db_bytes,
        "wal_bytes": wal_bytes,
        "shm_bytes": shm_bytes,
        "total_bytes": db_bytes + wal_bytes + shm_bytes,
        "reclaimable_bytes": max(0, freelist_count * page_size),
        "page_size": page_size,
        "page_count": page_count,
        "freelist_count": freelist_count,
        "message": None,
    }


def _load_known_asset_paths(st, owner_id: str) -> set[Path]:
    known: set[Path] = set()
    for asset in st.list("assets", owner_id):
        try:
            known.add(asset_path(asset))
        except ValueError:
            continue
    return known


def _collect_active_protections(
    st,
    owner_id: str,
    known_asset_ids: set[str],
    raw_assets: list[dict] | None = None,
) -> dict[str, set[str]]:
    repo = Repository(st)
    active_job_ids: set[str] = set()
    active_execution_ids: set[str] = set()
    protected_asset_ids: set[str] = set()
    active_scratch_names: set[str] = set()

    for job in repo.list("jobs", owner_id):
        state = str(job.get("state") or "").lower()
        if state in _TERMINAL_JOB_STATES:
            continue
        active_job_ids.add(str(job["id"]))
        exec_id = job.get("execution_id")
        if isinstance(exec_id, str) and exec_id:
            active_execution_ids.add(exec_id)
        for field in ("asset_ids", "output_asset_ids"):
            vals = job.get(field)
            if isinstance(vals, list):
                for aid in vals:
                    if isinstance(aid, str) and aid in known_asset_ids:
                        protected_asset_ids.add(aid)
        for param_map in (
            job.get("params"),
            (job.get("snapshot") or {}).get("resolved_params") if isinstance(job.get("snapshot"), dict) else None,
        ):
            if isinstance(param_map, dict):
                for val in param_map.values():
                    if isinstance(val, str) and val in known_asset_ids:
                        protected_asset_ids.add(val)

    for execution in repo.list("executions", owner_id):
        state = str(execution.get("state") or "").lower()
        if state not in _TERMINAL_EXECUTION_STATES:
            active_execution_ids.add(str(execution["id"]))

    if raw_assets is not None:
        for asset in raw_assets:
            aid = str(asset.get("id") or "")
            if not aid or aid not in known_asset_ids:
                continue
            src_job_id = str(asset.get("source_job_id") or asset.get("gpu_job_id") or "")
            exec_id = str(asset.get("execution_id") or "")
            if (src_job_id and src_job_id in active_job_ids) or (exec_id and exec_id in active_execution_ids):
                protected_asset_ids.add(aid)

    try:
        for task in repo.list_by_states("model_downloads", ACTIVE_MODEL_DOWNLOAD_STATES):
            tid = task.get("id")
            if isinstance(tid, str) and tid:
                active_scratch_names.add("transfer-" + hashlib.sha256(tid.encode()).hexdigest())
    except Exception:
        pass

    return {
        "active_job_ids": active_job_ids,
        "active_execution_ids": active_execution_ids,
        "protected_asset_ids": protected_asset_ids,
        "active_scratch_names": active_scratch_names,
    }


def _scan_thumb_caches(owner_id: str) -> list[tuple[Path, int, float]]:
    files: list[tuple[Path, int, float]] = []
    studio_thumb_root = _asset_thumb_dir()
    files.extend(_iter_regular_files(studio_thumb_root))
    owner_hash = hashlib.sha256(owner_id.encode()).hexdigest()[:20]
    native_thumb_root = (settings.data_dir / "native-thumb-cache" / owner_hash).resolve()
    files.extend(_iter_regular_files(native_thumb_root))
    return files


def _scan_gpu_results(owner_id: str, active_job_ids: set[str]) -> tuple[list[tuple[Path, int, float]], list[Path]]:
    root = (settings.data_dir / "gpu-results" / owner_id).resolve()
    if not root.is_dir() or root.is_symlink():
        return [], []
    reclaimable_files: list[tuple[Path, int, float]] = []
    reclaimable_dirs: list[Path] = []
    for child in root.iterdir():
        if child.is_symlink():
            continue
        if child.is_dir():
            if child.name in active_job_ids:
                continue
            reclaimable_dirs.append(child)
            reclaimable_files.extend(_iter_regular_files(child))
        elif child.is_file():
            size, ok_flag = _safe_stat_size(child)
            if ok_flag:
                reclaimable_files.append((child, size, 0.0))
    return reclaimable_files, reclaimable_dirs


def _scan_temp_and_orphans(
    owner_id: str,
    known_asset_paths: set[Path],
    active_execution_ids: set[str],
    active_scratch_names: set[str],
    *,
    now: float,
    st=None,
) -> dict[str, Any]:
    orphan_candidates: list[tuple[Path, Path, int]] = []
    artifact_candidates: list[tuple[str, Path, Path, int]] = []
    scratch_dirs: list[tuple[Path, int, int]] = []  # (dir_path, bytes, file_count)

    # 1. Orphan files under assets/<owner_id> (including stale .partial files)
    owner_assets_root = (settings.data_dir / "assets" / owner_id).resolve()
    for fp, size, mtime in _iter_regular_files(owner_assets_root):
        try:
            resolved = fp.resolve()
        except OSError:
            continue
        if resolved in known_asset_paths:
            continue
        if fp.name.endswith(".partial") and (now - mtime) < _PARTIAL_GRACE_SECONDS:
            continue
        orphan_candidates.append((fp, resolved, size))

    # 2. Reclaimable artifacts under assets/artifacts (owned by owner_id and not active)
    artifacts_root = (settings.data_dir / "assets" / "artifacts").resolve()
    if artifacts_root.is_dir() and not artifacts_root.is_symlink():
        for entry in artifacts_root.iterdir():
            if entry.is_symlink() or not entry.is_file() or not entry.name.endswith(".json"):
                continue
            try:
                meta = json.loads(entry.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(meta, dict):
                continue
            if meta.get("owner_id") != owner_id:
                continue
            exec_id = str(meta.get("execution_id") or "")
            if exec_id and exec_id in active_execution_ids:
                continue
            key = entry.stem
            data_candidate = (artifacts_root / f"{key}.data").resolve()
            if data_candidate in known_asset_paths:
                continue
            for suffix in (".data", ".tmp", ".json"):
                candidate = artifacts_root / f"{key}{suffix}"
                if candidate.is_file() and not candidate.is_symlink():
                    sz, ok_flag = _safe_stat_size(candidate)
                    if ok_flag:
                        artifact_candidates.append((exec_id, data_candidate, candidate, sz))

    # Re-confirm against DB AFTER disk scan so concurrent uploads are never classified as orphans.
    fresh_known = _load_known_asset_paths(st, owner_id) if (st is not None and (orphan_candidates or artifact_candidates)) else known_asset_paths
    orphan_files: list[tuple[Path, int]] = [
        (fp, size) for fp, resolved, size in orphan_candidates if resolved not in fresh_known
    ]
    artifact_files: list[tuple[Path, int]] = [
        (candidate, sz)
        for _exec_id, data_candidate, candidate, sz in artifact_candidates
        if data_candidate not in fresh_known
    ]
    artifact_meta: dict[Path, tuple[str, Path]] = {
        candidate: (exec_id, data_candidate)
        for exec_id, data_candidate, candidate, _sz in artifact_candidates
        if data_candidate not in fresh_known
    }

    # 3. Inactive model-scratch transfer directories
    scratch_root = Path(os.getenv("MODEL_LIBRARY_SCRATCH", str(settings.data_dir / "model-scratch"))).resolve()
    if scratch_root.is_dir() and not scratch_root.is_symlink():
        for entry in scratch_root.iterdir():
            if entry.is_symlink() or not entry.is_dir():
                continue
            if not entry.name.startswith("transfer-"):
                continue
            if entry.name in active_scratch_names:
                continue
            try:
                st_info = entry.lstat()
                if (now - st_info.st_mtime) < _SCRATCH_GRACE_SECONDS:
                    continue
            except OSError:
                continue
            nested = _iter_regular_files(entry)
            dir_bytes = sum(sz for _, sz, _ in nested)
            scratch_dirs.append((entry, dir_bytes, max(1, len(nested))))

    total_bytes = (
        sum(sz for _, sz in orphan_files)
        + sum(sz for _, sz in artifact_files)
        + sum(sz for _, sz, _ in scratch_dirs)
    )
    total_count = len(orphan_files) + len(artifact_files) + sum(cnt for _, _, cnt in scratch_dirs)
    return {
        "orphan_files": orphan_files,
        "artifact_files": artifact_files,
        "artifact_meta": artifact_meta,
        "scratch_dirs": scratch_dirs,
        "total_bytes": total_bytes,
        "total_count": total_count,
    }


def _build_local_storage_state(st, owner_id: str, *, now: float | None = None) -> dict[str, Any]:
    current = time.time() if now is None else now
    raw_assets = st.list("assets", owner_id)
    raw_projects = st.list("projects", owner_id)
    project_titles = {str(p["id"]): str(p.get("title") or p["id"]) for p in raw_projects}
    known_asset_ids = {str(a["id"]) for a in raw_assets}
    protections = _collect_active_protections(st, owner_id, known_asset_ids, raw_assets)
    protected_asset_ids = protections["protected_asset_ids"]

    known_asset_paths: set[Path] = set()
    outputs_favorite: list[tuple[dict, Path, int]] = []
    outputs_unfavorited: list[tuple[dict, Path, int]] = []
    reference_uploads: list[tuple[dict, Path, int]] = []
    diagnostic_assets: list[tuple[dict, Path, int]] = []
    missing_disk_assets = 0

    project_buckets: dict[str, dict[str, Any]] = {}

    def _project_bucket(pid: str) -> dict[str, Any]:
        key = pid or "__unassigned__"
        bucket = project_buckets.get(key)
        if bucket is None:
            bucket = {
                "project_id": pid,
                "title": project_titles.get(pid) or (pid if pid else "未归属项目"),
                "total_bytes": 0,
                "total_count": 0,
                "favorite_bytes": 0,
                "favorite_count": 0,
                "unfavorited_bytes": 0,
                "unfavorited_count": 0,
                "reference_bytes": 0,
                "reference_count": 0,
            }
            project_buckets[key] = bucket
        return bucket

    for asset in raw_assets:
        try:
            fp = asset_path(asset)
        except ValueError:
            continue
        known_asset_paths.add(fp)
        disk_size, exists = _safe_stat_size(fp)
        if not exists:
            missing_disk_assets += 1
        pid = str(asset.get("project_id") or "")
        if _is_diagnostic_asset(asset):
            diagnostic_assets.append((asset, fp, disk_size))
        elif _is_generated_asset(asset):
            bucket = _project_bucket(pid)
            bucket["total_bytes"] += disk_size
            bucket["total_count"] += 1
            if asset.get("favorite") is True:
                outputs_favorite.append((asset, fp, disk_size))
                bucket["favorite_bytes"] += disk_size
                bucket["favorite_count"] += 1
            else:
                outputs_unfavorited.append((asset, fp, disk_size))
                bucket["unfavorited_bytes"] += disk_size
                bucket["unfavorited_count"] += 1
        else:
            reference_uploads.append((asset, fp, disk_size))
            bucket = _project_bucket(pid)
            bucket["total_bytes"] += disk_size
            bucket["total_count"] += 1
            bucket["reference_bytes"] += disk_size
            bucket["reference_count"] += 1

    thumb_files = _scan_thumb_caches(owner_id)
    gpu_result_files, gpu_result_dirs = _scan_gpu_results(owner_id, protections["active_job_ids"])
    temp_orphans = _scan_temp_and_orphans(
        owner_id,
        known_asset_paths,
        protections["active_execution_ids"],
        protections["active_scratch_names"],
        now=current,
        st=st,
    )
    db_info = _sqlite_storage_info(st)

    data_root = settings.data_dir.resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    try:
        du = shutil.disk_usage(data_root)
        disk = {
            "total_bytes": int(du.total),
            "used_bytes": int(du.used),
            "free_bytes": int(du.free),
            "used_percent": round((du.used / du.total) * 100, 1) if du.total > 0 else 0.0,
        }
    except OSError:
        disk = {
            "total_bytes": 0,
            "used_bytes": 0,
            "free_bytes": 0,
            "used_percent": 0.0,
        }

    fav_bytes = sum(sz for _, _, sz in outputs_favorite)
    unfav_bytes = sum(sz for _, _, sz in outputs_unfavorited)
    ref_bytes = sum(sz for _, _, sz in reference_uploads)
    diag_bytes = sum(sz for _, _, sz in diagnostic_assets) + sum(sz for _, sz, _ in gpu_result_files)
    diag_count = len(diagnostic_assets) + len(gpu_result_files)
    thumb_bytes = sum(sz for _, sz, _ in thumb_files)
    temp_bytes = temp_orphans["total_bytes"]
    db_bytes = db_info["total_bytes"]

    categories = [
        {
            "key": "outputs_favorite",
            "label": "本地作品（已收藏）",
            "description": "你标星收藏的 Studio 生成图片与视频，受保护不会被按天清理",
            "bytes": fav_bytes,
            "count": len(outputs_favorite),
            "cleanable": False,
        },
        {
            "key": "outputs_unfavorited",
            "label": "本地作品（未收藏）",
            "description": "未收藏的 Studio 生成作品，可按保留天数与项目选择性清理",
            "bytes": unfav_bytes,
            "count": len(outputs_unfavorited),
            "cleanable": True,
        },
        {
            "key": "reference_uploads",
            "label": "上传与参考素材",
            "description": "手动上传或从原生图库导入的参考图/音视频，不会被自动清理",
            "bytes": ref_bytes,
            "count": len(reference_uploads),
            "cleanable": False,
        },
        {
            "key": "diagnostics",
            "label": "诊断与任务元数据",
            "description": "系统冒烟诊断 JSON 与已完成任务的本地 gpu-results 记录",
            "bytes": diag_bytes,
            "count": diag_count,
            "cleanable": True,
        },
        {
            "key": "thumb_cache",
            "label": "缩略图缓存",
            "description": "Studio 与原生图库本地 WebP 缩略图缓存，清空后按需自动重建",
            "bytes": thumb_bytes,
            "count": len(thumb_files),
            "cleanable": True,
        },
        {
            "key": "temp_and_orphans",
            "label": "临时与孤立残留",
            "description": "已结束任务的产物中转分块、非活跃模型传输临时目录与孤立磁盘文件",
            "bytes": temp_bytes,
            "count": temp_orphans["total_count"],
            "cleanable": True,
        },
        {
            "key": "database",
            "label": "SQLite 数据库",
            "description": "任务、工作流、作品索引与加密凭据（含 WAL/SHM 日志文件）",
            "bytes": db_bytes,
            "count": 1 if db_info["supported"] else 0,
            "cleanable": False,
        },
    ]

    studio_total_bytes = sum(cat["bytes"] for cat in categories)
    safe_reclaimable_bytes = thumb_bytes + temp_bytes + diag_bytes + db_info["reclaimable_bytes"]

    projects_list = sorted(
        [b for b in project_buckets.values() if b["total_count"] > 0 or b["total_bytes"] > 0],
        key=lambda item: (-item["total_bytes"], -item["total_count"], item["title"]),
    )

    summary = {
        "scanned_at": current,
        "disk": disk,
        "studio_total_bytes": studio_total_bytes,
        "safe_reclaimable_bytes": safe_reclaimable_bytes,
        "unfavorited_bytes": unfav_bytes,
        "unfavorited_count": len(outputs_unfavorited),
        "protected_asset_count": len(protected_asset_ids),
        "missing_disk_assets": missing_disk_assets,
        "categories": categories,
        "projects": projects_list,
        "database": db_info,
    }
    return {
        "summary": summary,
        "outputs_unfavorited": outputs_unfavorited,
        "diagnostic_assets": diagnostic_assets,
        "gpu_result_files": gpu_result_files,
        "gpu_result_dirs": gpu_result_dirs,
        "thumb_files": thumb_files,
        "temp_orphans": temp_orphans,
        "protected_asset_ids": protected_asset_ids,
        "now": current,
    }


def get_local_storage_overview(st, owner_id: str) -> dict[str, Any]:
    return _build_local_storage_state(st, owner_id)["summary"]


def _validate_cleanup_request(st, owner_id: str, action: str, older_than_days: int | None, project_id: str | None) -> tuple[str, int, str | None]:
    if action not in _CLEANUP_ACTIONS:
        raise APIError("VALIDATION_ERROR", "不支持的本地清理操作类型", 422)
    days = 30
    if action == "unfavorited_outputs":
        if older_than_days is None:
            days = 30
        elif isinstance(older_than_days, bool) or not isinstance(older_than_days, int) or not (0 <= older_than_days <= _MAX_OLDER_THAN_DAYS):
            raise APIError("VALIDATION_ERROR", f"older_than_days 必须为 0~{_MAX_OLDER_THAN_DAYS} 的整数", 422)
        else:
            days = older_than_days
    pid: str | None = None
    if project_id is not None and str(project_id).strip():
        pid = str(project_id).strip()
        own(st, "projects", pid, {"id": owner_id})
    return action, days, pid


def _select_unfavorited_targets(
    state: dict[str, Any],
    *,
    older_than_days: int,
    project_id: str | None,
) -> tuple[list[tuple[dict, Path, int]], int]:
    now = state["now"]
    cutoff = now - older_than_days * 86400.0 if older_than_days > 0 else None
    protected = state["protected_asset_ids"]
    targets: list[tuple[dict, Path, int]] = []
    protected_skipped = 0
    for asset, fp, disk_size in state["outputs_unfavorited"]:
        if project_id and str(asset.get("project_id") or "") != project_id:
            continue
        if cutoff is not None:
            created_epoch = _parse_iso_epoch(asset.get("created_at"))
            if created_epoch is None or created_epoch > cutoff:
                continue
        if str(asset["id"]) in protected:
            protected_skipped += 1
            continue
        targets.append((asset, fp, disk_size))
    return targets, protected_skipped


def preview_local_storage_cleanup(
    st,
    owner_id: str,
    *,
    action: str,
    older_than_days: int | None = None,
    project_id: str | None = None,
) -> dict[str, Any]:
    action, days, pid = _validate_cleanup_request(st, owner_id, action, older_than_days, project_id)
    state = _build_local_storage_state(st, owner_id)
    warnings: list[str] = []
    samples: list[dict[str, Any]] = []
    protected_skipped = 0

    if action == "thumb_cache":
        files = state["thumb_files"]
        item_count = len(files)
        reclaimable_bytes = sum(sz for _, sz, _ in files)
        samples = [{"name": fp.name, "bytes": sz} for fp, sz, _ in files[:8]]
    elif action == "temp_and_orphans":
        temp = state["temp_orphans"]
        item_count = temp["total_count"]
        reclaimable_bytes = temp["total_bytes"]
        for fp, sz in temp["orphan_files"][:4]:
            samples.append({"name": fp.name, "bytes": sz, "kind": "orphan"})
        for fp, sz in temp["artifact_files"][:2]:
            samples.append({"name": fp.name, "bytes": sz, "kind": "artifact"})
        for dp, sz, _ in temp["scratch_dirs"][:2]:
            samples.append({"name": dp.name, "bytes": sz, "kind": "scratch"})
    elif action == "diagnostics":
        diag_assets = state["diagnostic_assets"]
        gpu_files = state["gpu_result_files"]
        item_count = len(diag_assets) + len(gpu_files)
        reclaimable_bytes = sum(sz for _, _, sz in diag_assets) + sum(sz for _, sz, _ in gpu_files)
        for asset, _, sz in diag_assets[:5]:
            samples.append({"name": str(asset.get("filename") or asset["id"]), "bytes": sz, "kind": "diagnostic_asset"})
        for fp, sz, _ in gpu_files[:3]:
            samples.append({"name": f"{fp.parent.name}/{fp.name}", "bytes": sz, "kind": "gpu_result"})
    else:  # unfavorited_outputs
        targets, protected_skipped = _select_unfavorited_targets(state, older_than_days=days, project_id=pid)
        item_count = len(targets)
        reclaimable_bytes = sum(sz for _, _, sz in targets)
        samples = [
            {
                "id": asset["id"],
                "name": str(asset.get("filename") or asset["id"]),
                "bytes": sz,
                "created_at": asset.get("created_at"),
            }
            for asset, _, sz in targets[:8]
        ]
        if protected_skipped > 0:
            warnings.append(f"已自动跳过 {protected_skipped} 件正被运行中任务引用的作品。")
        warnings.append("已收藏作品与上传/导入的参考素材不会被删除；删除后无法恢复。")

    return {
        "action": action,
        "older_than_days": days if action == "unfavorited_outputs" else None,
        "project_id": pid,
        "item_count": item_count,
        "reclaimable_bytes": reclaimable_bytes,
        "protected_skipped_count": protected_skipped,
        "samples": samples,
        "warnings": warnings,
    }


def _prune_empty_dirs(root: Path, *, stop_at: Path) -> None:
    try:
        cur = root.resolve()
        boundary = stop_at.resolve()
        while cur != boundary and cur.is_relative_to(boundary):
            if not cur.is_dir() or any(cur.iterdir()):
                break
            cur.rmdir()
            cur = cur.parent
    except OSError:
        pass


def execute_local_storage_cleanup(
    st,
    owner_id: str,
    *,
    action: str,
    older_than_days: int | None = None,
    project_id: str | None = None,
) -> dict[str, Any]:
    action, days, pid = _validate_cleanup_request(st, owner_id, action, older_than_days, project_id)
    state = _build_local_storage_state(st, owner_id)
    deleted_count = 0
    freed_bytes = 0
    skipped_count = 0
    skipped_reasons: list[str] = []

    if action == "thumb_cache":
        for fp, sz, _ in state["thumb_files"]:
            try:
                fp.unlink(missing_ok=True)
                deleted_count += 1
                freed_bytes += sz
            except OSError:
                skipped_count += 1

    elif action == "temp_and_orphans":
        temp = state["temp_orphans"]
        owner_assets_root = (settings.data_dir / "assets" / owner_id).resolve()
        fresh_known_paths = (
            _load_known_asset_paths(st, owner_id)
            if (temp["orphan_files"] or temp["artifact_files"])
            else set()
        )
        for fp, sz in temp["orphan_files"]:
            try:
                if fp.resolve() in fresh_known_paths:
                    skipped_count += 1
                    continue
                fp.unlink(missing_ok=True)
                deleted_count += 1
                freed_bytes += sz
                _prune_empty_dirs(fp.parent, stop_at=owner_assets_root)
            except OSError:
                skipped_count += 1

        fresh_active_exec_ids = (
            _collect_active_protections(st, owner_id, set())["active_execution_ids"]
            if temp["artifact_files"]
            else set()
        )
        artifact_meta = temp.get("artifact_meta") or {}
        for fp, sz in temp["artifact_files"]:
            try:
                exec_id, data_candidate = artifact_meta.get(fp, ("", fp.resolve()))
                if (exec_id and exec_id in fresh_active_exec_ids) or (data_candidate in fresh_known_paths):
                    skipped_count += 1
                    continue
                fp.unlink(missing_ok=True)
                deleted_count += 1
                freed_bytes += sz
            except OSError:
                skipped_count += 1

        if temp["scratch_dirs"]:
            with transfer_lock() as acquired:
                if not acquired:
                    skipped_count += sum(cnt for _, _, cnt in temp["scratch_dirs"])
                    skipped_reasons.append("模型后台传输正在进行，已自动跳过 model-scratch 清理。")
                else:
                    for dp, sz, cnt in temp["scratch_dirs"]:
                        try:
                            shutil.rmtree(dp)
                            deleted_count += cnt
                            freed_bytes += sz
                        except OSError:
                            skipped_count += cnt

    elif action == "diagnostics":
        for asset, _, sz in state["diagnostic_assets"]:
            try:
                delete_asset_file(asset)
                _delete_asset_thumbs(asset)
                st.delete("assets", asset["id"], owner_id)
                deleted_count += 1
                freed_bytes += sz
            except (OSError, KeyError):
                skipped_count += 1
        for dp in state["gpu_result_dirs"]:
            nested = _iter_regular_files(dp)
            dir_bytes = sum(sz for _, sz, _ in nested)
            try:
                shutil.rmtree(dp)
                deleted_count += len(nested)
                freed_bytes += dir_bytes
            except OSError:
                skipped_count += len(nested)

    else:  # unfavorited_outputs
        targets, protected_skipped = _select_unfavorited_targets(state, older_than_days=days, project_id=pid)
        skipped_count += protected_skipped
        if protected_skipped > 0:
            skipped_reasons.append(f"已跳过 {protected_skipped} 件正被活跃任务引用的作品。")
        owner_assets_root = (settings.data_dir / "assets" / owner_id).resolve()
        fresh_protections = (
            _collect_active_protections(
                st,
                owner_id,
                {str(a["id"]) for a, _, _ in targets},
                [a for a, _, _ in targets],
            )
            if targets
            else {"protected_asset_ids": set()}
        )
        fresh_protected = fresh_protections["protected_asset_ids"]
        for asset, fp, sz in targets:
            try:
                if str(asset["id"]) in fresh_protected:
                    skipped_count += 1
                    continue
                fresh_asset = st.get("assets", asset["id"], owner_id)
                if fresh_asset is None or fresh_asset.get("favorite") is True:
                    skipped_count += 1
                    continue
                delete_asset_file(fresh_asset)
                _delete_asset_thumbs(fresh_asset)
                st.delete("assets", asset["id"], owner_id)
                _prune_empty_dirs(fp.parent, stop_at=owner_assets_root)
                deleted_count += 1
                freed_bytes += sz
            except (OSError, KeyError):
                skipped_count += 1

    fresh = get_local_storage_overview(st, owner_id)
    return {
        "action": action,
        "deleted_count": deleted_count,
        "freed_bytes": freed_bytes,
        "skipped_count": skipped_count,
        "skipped_reasons": skipped_reasons,
        "overview": fresh,
    }


def vacuum_local_sqlite(st, owner_id: str) -> dict[str, Any]:
    db_path = _sqlite_db_path(st)
    if db_path is None:
        raise APIError("SQLITE_ONLY", "仅本地文件 SQLite 数据库支持在线空间回收（VACUUM）", 409)
    before_info = _sqlite_storage_info(st)
    before_bytes = before_info["total_bytes"]
    with st.admission_lock() as acquired:
        if not acquired:
            raise APIError("SCHEDULER_BUSY", "后台调度繁忙，请稍后再执行数据库空间回收", 503, True)
        with st.db_lock():
            st.engine.dispose()
            conn = sqlite3.connect(str(db_path), timeout=20.0, isolation_level=None)
            try:
                conn.execute("PRAGMA busy_timeout=20000")
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                conn.execute("VACUUM")
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.Error as exc:
                raise APIError("SQLITE_VACUUM_FAILED", f"SQLite 空间回收失败：{str(exc)[:180]}", 500, True) from exc
            finally:
                conn.close()
    after_info = _sqlite_storage_info(st)
    after_bytes = after_info["total_bytes"]
    return {
        "before_bytes": before_bytes,
        "after_bytes": after_bytes,
        "freed_bytes": max(0, before_bytes - after_bytes),
        "reclaimable_bytes_after": after_info["reclaimable_bytes"],
        "overview": get_local_storage_overview(st, owner_id),
    }


def build_sqlite_snapshot_spool(st):
    """Create a verified consistent SQLite online backup and copy it into a bounded spool."""
    db_path = _sqlite_db_path(st)
    if db_path is None:
        raise APIError("SQLITE_ONLY", "仅本地文件 SQLite 支持网页端快照下载；PostgreSQL 请使用 pg_dump -Fc", 409)
    if not db_path.is_file():
        raise APIError("NOT_FOUND", "数据库文件不存在", 404)

    spool = open_zip_spool()
    try:
        with tempfile.TemporaryDirectory(prefix="studio-db-snap-") as tmp_dir:
            snapshot_path = Path(tmp_dir) / "studio.db"
            source = sqlite3.connect(str(db_path), timeout=30.0)
            target = sqlite3.connect(str(snapshot_path))
            try:
                source.backup(target)
                check = target.execute("PRAGMA integrity_check").fetchone()
                if not check or check[0] != "ok":
                    raise APIError("SNAPSHOT_VERIFY_FAILED", "数据库快照完整性校验失败", 500)
            except sqlite3.Error as exc:
                raise APIError("SNAPSHOT_FAILED", f"生成数据库快照失败：{str(exc)[:180]}", 500, True) from exc
            finally:
                target.close()
                source.close()

            with snapshot_path.open("rb") as src:
                while True:
                    chunk = src.read(STREAM_CHUNK_BYTES)
                    if not chunk:
                        break
                    spool.write(chunk)
        return spool
    except BaseException:
        spool.close()
        raise
