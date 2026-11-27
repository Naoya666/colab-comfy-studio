"""Tests for local storage governance: breakdown, preview/execute cleanup, SQLite VACUUM, and snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone

import pytest
from app.config import settings
from app.db import AssetRow, Store
from app.gpu import model_library
from app.gpu.repository import Repository
from app.main import app
from app.security import sign_session
from app.storage import asset_path, save_asset
from fastapi.testclient import TestClient

from tests.test_image_metadata_download import _make_png_with_workflow_and_color


@pytest.fixture
def file_db_client(tmp_path, monkeypatch):
    """TestClient backed by a real file-based SQLite DB inside tmp_path."""
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "studio_password", "test-pass")
    monkeypatch.setattr(settings, "session_secret", "test-session-secret-local-storage")
    monkeypatch.setattr(settings, "encryption_key", "")
    db_file = tmp_path / "studio.db"
    store = Store(f"sqlite:///{db_file}")
    app.state.store = store
    app.state.repo = Repository(store)
    token = sign_session(
        {
            "uid": "owner",
            "role": "admin",
            "purpose": "studio-session",
            "iat": int(time.time()),
            "exp": int(time.time()) + 3600,
        }
    )
    user_token = sign_session(
        {
            "uid": "owner",
            "role": "user",
            "purpose": "studio-session",
            "iat": int(time.time()),
            "exp": int(time.time()) + 3600,
        }
    )
    headers = {"Authorization": f"Bearer {token}"}
    user_headers = {"Authorization": f"Bearer {user_token}"}
    with TestClient(app) as client:
        yield client, store, tmp_path, headers, user_headers
    store.engine.dispose()


def _set_asset_age_days(store: Store, asset_id: str, days: int) -> None:
    target_dt = datetime.now(timezone.utc) - timedelta(days=days)
    with store.Session.begin() as session:
        row = session.get(AssetRow, asset_id)
        assert row is not None
        row.created_at = target_dt
        row.updated_at = target_dt


def test_local_storage_breakdown_and_safe_cleanups(file_db_client):
    client, store, data_dir, headers, user_headers = file_db_client
    repo = Repository(store)
    png_raw = _make_png_with_workflow_and_color()

    p1 = store.create("projects", "owner", {"title": "主项目"})
    p2 = store.create("projects", "owner", {"title": "测试项目"})

    # 1. Seed assets across all categories
    fav_out = save_asset(store, "owner", p1["id"], png_raw, "fav.png", source_job_id="job-fav")
    store.update("assets", fav_out["id"], {"favorite": True}, "owner")
    _set_asset_age_days(store, fav_out["id"], 60)

    old_unfav_p1 = save_asset(store, "owner", p1["id"], png_raw, "old_p1.png", source_job_id="job-old-1")
    _set_asset_age_days(store, old_unfav_p1["id"], 45)

    old_unfav_p2 = save_asset(store, "owner", p2["id"], png_raw, "old_p2.png", source_job_id="job-old-2")
    _set_asset_age_days(store, old_unfav_p2["id"], 45)

    recent_unfav = save_asset(store, "owner", p1["id"], png_raw, "recent.png", source_job_id="job-recent")
    _set_asset_age_days(store, recent_unfav["id"], 2)

    active_protected_out = save_asset(store, "owner", p1["id"], png_raw, "protected.png", source_job_id="job-protected")
    _set_asset_age_days(store, active_protected_out["id"], 60)

    ref_upload = save_asset(store, "owner", p1["id"], png_raw, "reference.png", source_job_id=None)
    _set_asset_age_days(store, ref_upload["id"], 90)

    diag_asset = save_asset(
        store,
        "owner",
        p1["id"],
        b'{"ok": true, "smoke": "diag"}',
        "smoke-report.json",
        source_job_id="job-smoke",
        metadata={"system_smoke": True},
    )

    # 2. Seed an active job protecting `active_protected_out`, `exec-active`, and `active_job["id"]`
    active_job = repo.create(
        "jobs",
        "owner",
        {
            "project_id": p1["id"],
            "state": "running",
            "execution_id": "exec-active",
            "asset_ids": [active_protected_out["id"]],
        },
        project_id=p1["id"],
    )

    # 3. Seed thumbnail caches (Studio + Native)
    studio_thumb = data_dir / "asset-thumb-cache" / f"{old_unfav_p1['id']}-v1.webp"
    studio_thumb.parent.mkdir(parents=True, exist_ok=True)
    studio_thumb.write_bytes(b"WEBP" + b"x" * 200)

    owner_hash = hashlib.sha256(b"owner").hexdigest()[:20]
    native_thumb = data_dir / "native-thumb-cache" / owner_hash / "drivefile-v1.webp"
    native_thumb.parent.mkdir(parents=True, exist_ok=True)
    native_thumb.write_bytes(b"WEBP" + b"y" * 300)

    # 4. Seed gpu-results (one terminal reclaimable, one active protected)
    term_res = data_dir / "gpu-results" / "owner" / "job-done-1" / "metadata.json"
    term_res.parent.mkdir(parents=True, exist_ok=True)
    term_res.write_bytes(b'{"done": true}')

    act_res = data_dir / "gpu-results" / "owner" / active_job["id"] / "metadata.json"
    act_res.parent.mkdir(parents=True, exist_ok=True)
    act_res.write_bytes(b'{"running": true}')

    # 5. Seed orphans, stale/fresh .partial, terminal/active artifacts, and stale/active model-scratch
    orphan_file = data_dir / "assets" / "owner" / p1["id"] / "orphan-stray.png"
    orphan_file.parent.mkdir(parents=True, exist_ok=True)
    orphan_file.write_bytes(b"ORPHAN" * 40)

    stale_partial = data_dir / "assets" / "owner" / p1["id"] / "old-upload.png.partial"
    stale_partial.write_bytes(b"PARTIAL" * 20)
    old_mtime = time.time() - 3600
    os.utime(stale_partial, (old_mtime, old_mtime))

    fresh_partial = data_dir / "assets" / "owner" / p1["id"] / "live-upload.png.partial"
    fresh_partial.write_bytes(b"FRESH" * 20)

    artifacts_dir = data_dir / "assets" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    lock_file = artifacts_dir / ".artifact.lock"
    lock_file.write_bytes(b"")

    term_key = "a" * 32
    (artifacts_dir / f"{term_key}.json").write_text(json.dumps({"owner_id": "owner", "execution_id": "exec-done"}))
    (artifacts_dir / f"{term_key}.data").write_bytes(b"D" * 512)

    act_key = "b" * 32
    (artifacts_dir / f"{act_key}.json").write_text(json.dumps({"owner_id": "owner", "execution_id": "exec-active"}))
    (artifacts_dir / f"{act_key}.data").write_bytes(b"A" * 512)

    active_dl = repo.create("model_downloads", "owner", {"state": "downloading", "model_id": "m1"})
    active_scratch = model_library.transfer_directory(active_dl["id"])
    active_scratch.mkdir(parents=True, exist_ok=True)
    (active_scratch / "part.bin").write_bytes(b"S" * 100)
    os.utime(active_scratch, (old_mtime, old_mtime))

    stale_scratch = data_dir / "model-scratch" / ("transfer-" + "c" * 64)
    stale_scratch.mkdir(parents=True, exist_ok=True)
    (stale_scratch / "dead.bin").write_bytes(b"Z" * 400)
    os.utime(stale_scratch, (old_mtime, old_mtime))

    # --- Overview check ---
    r_ov = client.get("/api/gpu/storage/local", headers=headers)
    assert r_ov.status_code == 200, r_ov.text
    ov = r_ov.json()["data"]
    cats = {c["key"]: c for c in ov["categories"]}
    assert ov["disk"]["total_bytes"] > 0
    assert ov["database"]["supported"] is True
    assert ov["database"]["dialect"] == "sqlite"
    assert cats["outputs_favorite"]["count"] == 1
    assert cats["outputs_unfavorited"]["count"] == 4
    assert cats["reference_uploads"]["count"] == 1
    assert cats["diagnostics"]["count"] == 2  # diag_asset + term_res (active job's gpu-results excluded)
    assert cats["thumb_cache"]["count"] == 2
    assert cats["temp_and_orphans"]["count"] >= 5  # orphan + stale_partial + 2 term artifact files + stale_scratch
    assert ov["protected_asset_count"] == 1
    assert len(ov["projects"]) == 2

    # --- Non-admin cannot preview or execute cleanup ---
    r_forbid = client.post(
        "/api/gpu/storage/local/cleanup/preview",
        json={"action": "thumb_cache"},
        headers=user_headers,
    )
    assert r_forbid.status_code == 403

    # --- Confirmed=true required for cleanup ---
    r_unconf = client.post(
        "/api/gpu/storage/local/cleanup",
        json={"action": "thumb_cache"},
        headers=headers,
    )
    assert r_unconf.status_code == 422

    # --- Cleanup 1: thumb_cache ---
    p_thumb = client.post("/api/gpu/storage/local/cleanup/preview", json={"action": "thumb_cache"}, headers=headers)
    assert p_thumb.status_code == 200
    assert p_thumb.json()["data"]["item_count"] == 2

    c_thumb = client.post(
        "/api/gpu/storage/local/cleanup",
        json={"action": "thumb_cache", "confirmed": True},
        headers=headers,
    )
    assert c_thumb.status_code == 200
    assert c_thumb.json()["data"]["deleted_count"] == 2
    assert not studio_thumb.exists()
    assert not native_thumb.exists()

    # --- Cleanup 2: temp_and_orphans (preserves active artifacts, fresh partial, lock, active scratch) ---
    c_temp = client.post(
        "/api/gpu/storage/local/cleanup",
        json={"action": "temp_and_orphans", "confirmed": True},
        headers=headers,
    )
    assert c_temp.status_code == 200, c_temp.text
    assert not orphan_file.exists()
    assert not stale_partial.exists()
    assert not (artifacts_dir / f"{term_key}.data").exists()
    assert not (artifacts_dir / f"{term_key}.json").exists()
    assert not stale_scratch.exists()
    # Protected files must remain untouched!
    assert fresh_partial.exists()
    assert lock_file.exists()
    assert (artifacts_dir / f"{act_key}.data").exists()
    assert (artifacts_dir / f"{act_key}.json").exists()
    assert active_scratch.exists()

    # --- Cleanup 3: diagnostics (preserves active job's gpu-results) ---
    c_diag = client.post(
        "/api/gpu/storage/local/cleanup",
        json={"action": "diagnostics", "confirmed": True},
        headers=headers,
    )
    assert c_diag.status_code == 200
    assert store.get("assets", diag_asset["id"], "owner") is None
    assert not term_res.exists()
    assert act_res.exists()

    # --- Cleanup 4: unfavorited_outputs older than 30 days scoped to p1 ---
    p_unfav = client.post(
        "/api/gpu/storage/local/cleanup/preview",
        json={"action": "unfavorited_outputs", "older_than_days": 30, "project_id": p1["id"]},
        headers=headers,
    )
    assert p_unfav.status_code == 200, p_unfav.text
    pv = p_unfav.json()["data"]
    assert pv["item_count"] == 1  # Only old_unfav_p1; fav, recent, ref, p2, and active_protected skipped!
    assert pv["protected_skipped_count"] == 1

    c_unfav = client.post(
        "/api/gpu/storage/local/cleanup",
        json={"action": "unfavorited_outputs", "older_than_days": 30, "project_id": p1["id"], "confirmed": True},
        headers=headers,
    )
    assert c_unfav.status_code == 200, c_unfav.text
    assert c_unfav.json()["data"]["deleted_count"] == 1
    assert store.get("assets", old_unfav_p1["id"], "owner") is None
    assert not asset_path(old_unfav_p1).exists()
    # Strictly preserved assets
    assert store.get("assets", fav_out["id"], "owner") is not None
    assert store.get("assets", recent_unfav["id"], "owner") is not None
    assert store.get("assets", active_protected_out["id"], "owner") is not None
    assert store.get("assets", ref_upload["id"], "owner") is not None
    assert store.get("assets", old_unfav_p2["id"], "owner") is not None


def test_local_storage_sqlite_vacuum_and_consistent_snapshot(file_db_client, tmp_path):
    client, store, _data_dir, headers, user_headers = file_db_client

    # Create a durable project and inflate then delete bulky settings rows to produce real SQLite freelist pages.
    proj = store.create("projects", "owner", {"title": "快照测试项目"})
    bulk_ids = []
    for i in range(40):
        row = store.create("settings", "owner", {"kind": "bloat", "blob": "X" * 8192}, record_id=f"bloat-{i}")
        bulk_ids.append(row["id"])
    for rid in bulk_ids:
        store.delete("settings", rid, "owner")

    ov_before = client.get("/api/gpu/storage/local", headers=headers).json()["data"]
    assert ov_before["database"]["supported"] is True
    assert ov_before["database"]["reclaimable_bytes"] > 0

    # Non-admin blocked from VACUUM and snapshot
    assert client.post("/api/gpu/storage/local/vacuum", json={"confirmed": True}, headers=user_headers).status_code == 403
    assert client.post("/api/gpu/storage/local/snapshot", headers=user_headers).status_code == 403

    # Confirmed required for VACUUM
    assert client.post("/api/gpu/storage/local/vacuum", json={}, headers=headers).status_code == 422

    # Run VACUUM and verify disk space reclaimed & data intact
    r_vac = client.post("/api/gpu/storage/local/vacuum", json={"confirmed": True}, headers=headers)
    assert r_vac.status_code == 200, r_vac.text
    vac_data = r_vac.json()["data"]
    assert vac_data["freed_bytes"] > 0
    assert vac_data["after_bytes"] < vac_data["before_bytes"]
    assert vac_data["reclaimable_bytes_after"] == 0
    assert store.get("projects", proj["id"], "owner")["title"] == "快照测试项目"

    # Request and download consistent SQLite snapshot
    r_snap_ticket = client.post("/api/gpu/storage/local/snapshot", headers=headers)
    assert r_snap_ticket.status_code == 200, r_snap_ticket.text
    snap_ticket = r_snap_ticket.json()["data"]
    assert snap_ticket["filename"].endswith(".db")
    assert ".env" in snap_ticket["notice"]

    r_snap = client.get(snap_ticket["download_url"], headers={**headers, "Accept-Encoding": "gzip"})
    assert r_snap.status_code == 200, r_snap.text
    assert r_snap.headers["content-type"] == "application/x-sqlite3"
    assert "content-encoding" not in r_snap.headers
    assert int(r_snap.headers["content-length"]) == len(r_snap.content)
    assert r_snap.content[:16] == b"SQLite format 3\x00"

    # Verify the downloaded snapshot opens cleanly, passes integrity_check, and contains our project!
    snap_file = tmp_path / "downloaded-snapshot.db"
    snap_file.write_bytes(r_snap.content)
    with sqlite3.connect(str(snap_file)) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        rows = conn.execute("SELECT id, data FROM projects WHERE id = ?", (proj["id"],)).fetchall()
        assert len(rows) == 1
        assert json.loads(rows[0][1])["title"] == "快照测试项目"


def test_local_storage_skips_model_scratch_when_transfer_lock_is_held(file_db_client):
    client, _store, data_dir, headers, _user_headers = file_db_client
    stale_scratch = data_dir / "model-scratch" / ("transfer-" + "d" * 64)
    stale_scratch.mkdir(parents=True, exist_ok=True)
    (stale_scratch / "part.bin").write_bytes(b"B" * 256)
    old_mtime = time.time() - 3600
    os.utime(stale_scratch, (old_mtime, old_mtime))

    with model_library.transfer_lock() as acquired:
        assert acquired is True
        r = client.post(
            "/api/gpu/storage/local/cleanup",
            json={"action": "temp_and_orphans", "confirmed": True},
            headers=headers,
        )
        assert r.status_code == 200, r.text
        data = r.json()["data"]
        assert data["skipped_count"] >= 1
        assert any("model-scratch" in reason for reason in data["skipped_reasons"])
        assert stale_scratch.exists()


def test_local_storage_memory_sqlite_reports_unsupported_for_vacuum_and_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "session_secret", "test-session-secret-mem")
    mem_store = Store("sqlite:///:memory:")
    app.state.store = mem_store
    app.state.repo = Repository(mem_store)
    token = sign_session(
        {
            "uid": "owner",
            "role": "admin",
            "purpose": "studio-session",
            "iat": int(time.time()),
            "exp": int(time.time()) + 3600,
        }
    )
    headers = {"Authorization": f"Bearer {token}"}
    with TestClient(app) as client:
        r_ov = client.get("/api/gpu/storage/local", headers=headers)
        assert r_ov.status_code == 200
        assert r_ov.json()["data"]["database"]["supported"] is False

        r_vac = client.post("/api/gpu/storage/local/vacuum", json={"confirmed": True}, headers=headers)
        assert r_vac.status_code == 409
        assert r_vac.json()["error"]["code"] == "SQLITE_ONLY"

        r_snap = client.post("/api/gpu/storage/local/snapshot", headers=headers)
        assert r_snap.status_code == 409
        assert r_snap.json()["error"]["code"] == "SQLITE_ONLY"
    mem_store.engine.dispose()


def test_orphan_cleanup_protects_concurrent_upload_and_committed_artifact(file_db_client, monkeypatch):
    from app import local_storage

    client, store, data_dir, headers, _user_headers = file_db_client
    png_raw = _make_png_with_workflow_and_color()
    proj = store.create("projects", "owner", {"title": "并发上传保护"})

    # Seed a true stray orphan file.
    stray = data_dir / "assets" / "owner" / proj["id"] / "true-orphan.png"
    stray.parent.mkdir(parents=True, exist_ok=True)
    stray.write_bytes(b"ORPHAN" * 16)

    # Seed a committed artifact asset stored under assets/artifacts/<id>.data (terminal execution).
    artifacts_dir = data_dir / "assets" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    committed_aid = "f" * 32
    (artifacts_dir / f"{committed_aid}.json").write_text(
        json.dumps({"owner_id": "owner", "execution_id": "exec-finished"})
    )
    (artifacts_dir / f"{committed_aid}.data").write_bytes(png_raw)
    committed_asset = store.create(
        "assets",
        "owner",
        {
            "filename": "committed.png",
            "mime": "image/png",
            "storage_key": f"artifacts/{committed_aid}.data",
            "size": len(png_raw),
            "favorite": False,
            "source_job_id": "job-done",
        },
        project_id=proj["id"],
        record_id=committed_aid,
    )

    # Simulate a normal upload occurring right between DB snapshot and disk scan.
    uploaded_holder: list[dict] = []
    orig_iter = local_storage._iter_regular_files

    def racing_iter(root):
        if not uploaded_holder and root == (data_dir / "assets" / "owner").resolve():
            uploaded_holder.append(save_asset(store, "owner", proj["id"], png_raw, "concurrent-ref.png"))
        return orig_iter(root)

    monkeypatch.setattr(local_storage, "_iter_regular_files", racing_iter)

    res = client.post(
        "/api/gpu/storage/local/cleanup",
        json={"action": "temp_and_orphans", "confirmed": True},
        headers=headers,
    )
    assert res.status_code == 200, res.text
    assert len(uploaded_holder) == 1
    concurrent_asset = uploaded_holder[0]

    # Concurrent upload and committed artifact must BOTH remain intact on disk and in DB.
    assert store.get("assets", concurrent_asset["id"], "owner") is not None
    assert asset_path(concurrent_asset).is_file()
    assert store.get("assets", committed_asset["id"], "owner") is not None
    assert (artifacts_dir / f"{committed_aid}.data").is_file()
    # True stray orphan is still cleaned up.
    assert not stray.exists()


def test_unfavorited_cleanup_protects_uploading_job_outputs_via_source_job_id(file_db_client):
    client, store, _data_dir, headers, _user_headers = file_db_client
    repo = Repository(store)
    png_raw = _make_png_with_workflow_and_color()
    proj = store.create("projects", "owner", {"title": "上传中任务保护"})

    uploading_job = repo.create(
        "jobs",
        "owner",
        {
            "project_id": proj["id"],
            "state": "uploading",
            "execution_id": "exec-uploading-1",
            "output_asset_ids": [],
        },
        project_id=proj["id"],
    )
    in_flight_out = save_asset(
        store,
        "owner",
        proj["id"],
        png_raw,
        "in-flight.png",
        source_job_id=uploading_job["id"],
    )

    p_res = client.post(
        "/api/gpu/storage/local/cleanup/preview",
        json={"action": "unfavorited_outputs", "older_than_days": 0, "project_id": proj["id"]},
        headers=headers,
    )
    assert p_res.status_code == 200, p_res.text
    assert p_res.json()["data"]["item_count"] == 0
    assert p_res.json()["data"]["protected_skipped_count"] == 1

    c_res = client.post(
        "/api/gpu/storage/local/cleanup",
        json={"action": "unfavorited_outputs", "older_than_days": 0, "project_id": proj["id"], "confirmed": True},
        headers=headers,
    )
    assert c_res.status_code == 200, c_res.text
    assert c_res.json()["data"]["deleted_count"] == 0
    assert store.get("assets", in_flight_out["id"], "owner") is not None
    assert asset_path(in_flight_out).is_file()
