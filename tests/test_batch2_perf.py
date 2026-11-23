"""Regression tests for Batch 2 performance optimizations."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from app.common import APIError, current_user
from app.db import Store
from app.gpu import native_outputs
from app.gpu.accounts import AccountService
from app.gpu.google import GoogleService
from app.gpu.instances import InstanceService
from app.gpu.registry import Registry
from app.gpu.repository import Repository
from app.history_routes import router as history_router
from app.instance_routes import router as instance_router
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from sqlalchemy import text


def _api_error_handler(request, exc):
    return JSONResponse({"error": {"code": exc.code, "message": exc.message}}, status_code=exc.status)


def test_sqlite_file_pragmas_and_expression_indexes(tmp_path):
    db_file = tmp_path / "perf.db"
    store = Store(f"sqlite:///{db_file}")
    repo = Repository(store)
    with store.engine.connect() as conn:
        assert str(conn.execute(text("PRAGMA journal_mode")).scalar()).lower() == "wal"
        assert int(conn.execute(text("PRAGMA synchronous")).scalar()) == 1  # NORMAL
        assert int(conn.execute(text("PRAGMA temp_store")).scalar()) == 2  # MEMORY
        assert int(conn.execute(text("PRAGMA cache_size")).scalar()) == -16000
        assert int(conn.execute(text("PRAGMA busy_timeout")).scalar()) == 30000

    # Verify event queries hit the SQLite expression index instead of scanning the table.
    job = repo.create("jobs", "owner-1", {"state": "completed", "instance_id": "inst-1"})
    repo.event(job["id"], "owner-1", "running", "step 1")
    repo.event(job["id"], "owner-1", "completed", "done")
    events = repo.list_events_for_job("owner-1", job["id"], asc=True)
    assert len(events) == 2
    assert [e["state"] for e in events] == ["running", "completed"]

    with store.engine.connect() as conn:
        plan_rows = conn.execute(
            text("EXPLAIN QUERY PLAN SELECT * FROM gpu_events WHERE owner_id = :o AND json_extract(data, '$.job_id') = :j"),
            {"o": "owner-1", "j": job["id"]},
        ).fetchall()
    plan_text = " ".join(str(r) for r in plan_rows)
    assert "ix_gpu_events_owner_job_id" in plan_text


def test_instance_history_with_real_store_returns_200_and_assets(tmp_path):
    store = Store(f"sqlite:///{tmp_path / 'hist.db'}")
    repo = Repository(store)
    inst = repo.create("instances", "owner-1", {"state": "deleted", "account_id": "acc-1"})
    asset = store.create("assets", "owner-1", {"filename": "out.png", "mime": "image/png", "status": "ready"})
    job = repo.create(
        "jobs",
        "owner-1",
        {
            "state": "completed",
            "provider": "mock",
            "instance_id": inst["id"],
            "account_id": "acc-1",
            "output_asset_ids": [asset["id"]],
        },
    )
    repo.event(job["id"], "owner-1", "completed", "finished")

    app = FastAPI()
    app.add_exception_handler(APIError, _api_error_handler)
    app.state.store = store
    app.state.repo = repo
    app.state.instance_service = InstanceService(repo)
    app.dependency_overrides[current_user] = lambda: {"id": "owner-1", "role": "admin"}
    app.include_router(history_router)
    app.include_router(instance_router)

    with TestClient(app) as client:
        hist = client.get(f"/api/gpu/instances/{inst['id']}/history")
        assert hist.status_code == 200, hist.text
        data = hist.json()["data"]
        assert len(data["jobs"]) == 1
        assert data["jobs"][0]["artifacts"][0]["filename"] == "out.png"

        detail = client.get("/api/gpu/instances?detail=true")
        assert detail.status_code == 200
        items = detail.json()["data"]
        assert len(items) == 1
        assert len(items[0]["jobs"]) == 1


def test_google_service_status_does_not_allocate_httpx_client():
    store = Store("sqlite:///:memory:")
    repo = Repository(store)
    svc = GoogleService(repo, "owner-1")
    status = svc.status()
    assert "colab" in status and "drive" in status
    assert svc._client is None


def test_registry_ensure_bootstrapped_runs_once_per_store():
    store = Store("sqlite:///:memory:")
    repo = Repository(store)
    reg = Registry(repo)
    calls = 0
    orig = reg.bootstrap

    def counted(owner_id):
        nonlocal calls
        calls += 1
        return orig(owner_id)

    reg.bootstrap = counted
    reg.ensure_bootstrapped("owner-1")
    reg.ensure_bootstrapped("owner-1")
    reg.ensure_bootstrapped("owner-1")
    assert calls == 1


def test_account_service_ensure_default_fast_path_and_batch_list():
    store = Store("sqlite:///:memory:")
    repo = Repository(store)
    svc = AccountService(repo, "owner-1")
    acc = svc.create({"alias": "Primary"})
    repo.create("instances", "owner-1", {"state": "deleted", "account_id": acc["account_id"]})
    assert repo.has_unscoped_legacy_rows("owner-1") is False
    listed = svc.list()
    assert len(listed) == 1
    assert listed[0]["instance_count"] == 1


def test_native_outputs_single_flight_and_disk_thumb_fast_path(tmp_path, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    native_outputs.invalidate_native_outputs_cache()
    scan_calls = 0
    scan_lock = threading.Lock()

    class _DummyDrive:
        base_url = "https://drive.example"
        account_id = "acc-1"
        _owns_client = False

    def fake_find_owner(_drive, _owner):
        return "f-owner"

    def fake_bulk(_drive, _owner_folder, account_id, account_alias):
        nonlocal scan_calls
        time.sleep(0.05)
        with scan_lock:
            scan_calls += 1
        if account_id != "acc-1":
            return [], "other-output"
        return [
            {
                "id": "img-sf-1",
                "filename": "sample.png",
                "subfolder": "",
                "path": "sample.png",
                "mime": "image/png",
                "size": 123,
                "width": 64,
                "height": 64,
                "md5": "rev123",
                "modified_at": "2026-11-23T00:00:00Z",
                "created_at": "2026-11-23T00:00:00Z",
                "thumbnail_link": None,
                "revision": "rev123",
                "thumbnail_url": "/api/gpu/native/outputs/img-sf-1/thumbnail?v=rev123",
                "content_url": "/api/gpu/native/outputs/img-sf-1/content",
                "account_id": account_id,
                "account_alias": account_alias,
            }
        ], "f-out"

    monkeypatch.setattr(native_outputs, "_find_owner_folder", fake_find_owner)
    monkeypatch.setattr(native_outputs, "_bulk_scan_output_tree", fake_bulk)

    drive = _DummyDrive()
    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = [
            pool.submit(native_outputs._scan_output_tree, drive, "owner-sf", "acc-1", "主盘")
            for _ in range(6)
        ]
        results = [f.result() for f in futs]

    assert scan_calls == 1
    assert all(len(r[0]) == 1 and r[0][0]["id"] == "img-sf-1" for r in results)

    # A warm lookup checks the current binding and uses the account's cached directory.
    thumb_file = native_outputs._thumb_path("owner-sf", "img-sf-1", "rev123")
    thumb_file.write_bytes(b"RIFF\x00\x00\x00\x00WEBP")
    binding = ("acc-1", "主盘")
    monkeypatch.setattr(native_outputs, "resolve_owner_drive_account", lambda *_: binding)
    monkeypatch.setattr(native_outputs, "DriveStorage", lambda *_a, **_kw: drive)
    data, ctype = native_outputs.get_native_output_thumbnail(object(), "owner-sf", "img-sf-1")
    assert ctype == "image/webp"
    assert data.startswith(b"RIFF")
    assert scan_calls == 1

    binding = (None, None)
    with pytest.raises(APIError) as disconnected:
        native_outputs.get_native_output_thumbnail(object(), "owner-sf", "img-sf-1")
    assert disconnected.value.code == "DRIVE_NOT_CONNECTED"

    binding = ("acc-2", "另一块盘")
    with pytest.raises(APIError) as switched:
        native_outputs.get_native_output_thumbnail(object(), "owner-sf", "img-sf-1")
    assert switched.value.code == "NOT_FOUND"
    native_outputs.invalidate_native_outputs_cache()


def test_native_outputs_bulk_scan_keeps_newest_across_folders(monkeypatch):
    monkeypatch.setattr(native_outputs, "MAX_OUTPUT_FILES", 2)

    def folder(file_id, name, parent):
        return {"id": file_id, "name": name, "mimeType": native_outputs.FOLDER_MIME, "parents": [parent]}

    def image(file_id, parent, day):
        return {
            "id": file_id, "name": f"{file_id}.png", "mimeType": "image/png", "parents": [parent],
            "modifiedTime": f"2026-11-{day:02d}T00:00:00Z",
        }

    class Drive:
        base_url = "https://drive.example"

        def _request(self, *args, **kwargs):
            return {"files": [
                folder("native", "native-comfy", "owner"), folder("output", "output", "native"),
                image("old", "output", 1), image("recent", "output", 2),
                folder("nested", "nested", "output"), image("newest", "nested", 3),
            ]}

        def _ensure_ok(self, response, _action):
            return response

    items, folder_id = native_outputs._bulk_scan_output_tree(Drive(), "owner", "acc-1", "主盘")
    assert folder_id == "output"
    assert [item["id"] for item in items] == ["newest", "recent"]
    assert items[0]["path"] == "nested/newest.png"
