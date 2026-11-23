"""Regression tests for Batch 1 performance fixes:
- Single-worker Scheduler.tick() fairness / anti-starvation
- Media Cache-Control preservation, local thumbnail disk cache & ETag/304
- SelectiveGZipMiddleware compression for JSON without compressing media
"""

from __future__ import annotations

import io
from pathlib import Path

from app import asset_routes, main
from app.config import settings
from app.db import Store
from app.gpu.repository import Repository
from app.gpu.scheduler import Scheduler
from app.storage import save_asset
from fastapi.testclient import TestClient
from PIL import Image


def _png_bytes(color=(120, 180, 220)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (640, 640), color).save(buf, "PNG")
    return buf.getvalue()


def test_scheduler_tick_does_not_starve_newer_job_behind_backoff_or_running_job():
    store = Store("sqlite://")
    repo = Repository(store)
    clock = [1000.0]

    # Job A: older running job in poll backoff.
    job_a = repo.create(
        "jobs",
        "owner",
        {
            "state": "running",
            "provider": "mock",
            "execution_id": "exec-a",
            "created_epoch": 100.0,
            "poll_retry_at": 1060.0,
        },
    )
    repo.create("leases", "owner", {"owner_job_id": job_a["id"], "state": "active", "handle": {}})

    # Job B: newer queued job that must not starve behind Job A.
    job_b = repo.create(
        "jobs",
        "owner",
        {
            "state": "queued",
            "provider": "mock",
            "created_epoch": 200.0,
        },
    )

    class RunningExecutor:
        def __init__(self):
            self.polled: list[str] = []

        def poll(self, _handle, job):
            self.polled.append(job["id"])
            return {"state": "running", "execution_id": job.get("execution_id")}

    executor = RunningExecutor()
    scheduler = Scheduler(repo, executor_factory=lambda _: executor, clock=lambda: clock[0])

    # Tick 1 skips backed-off Job A and immediately picks Job B.
    tick1 = scheduler.tick()
    assert tick1["job_id"] == job_b["id"]
    assert tick1["advanced"] is True
    assert executor.polled == []

    # Even when Job A's backoff expires and Job A stays 'running', round-robin still advances Job C.
    clock[0] = 1100.0
    job_c = repo.create(
        "jobs",
        "owner",
        {
            "state": "queued",
            "provider": "mock",
            "created_epoch": 300.0,
        },
    )
    # Mark Job B terminal so only Job A (running) and Job C (queued) compete.
    repo.update("jobs", job_b["id"], {"state": "completed"}, "owner")

    t_first = scheduler.tick()
    assert t_first["job_id"] == job_a["id"]
    assert t_first["advanced"] is False
    assert executor.polled == [job_a["id"]]

    t_second = scheduler.tick()
    assert t_second["job_id"] == job_c["id"]
    assert t_second["advanced"] is True


def test_scheduler_cleanup_priority_remains_ahead_of_normal_jobs():
    store = Store("sqlite://")
    repo = Repository(store)
    clock = [1000.0]

    job_normal = repo.create("jobs", "owner", {"state": "queued", "provider": "mock", "created_epoch": 100.0})
    job_cleanup = repo.create("jobs", "owner", {"state": "cleaning_up", "provider": "mock", "created_epoch": 200.0})

    scheduler = Scheduler(repo, clock=lambda: clock[0])
    picked, token = scheduler._claim_next_job()
    try:
        assert picked is not None
        assert picked["id"] == job_cleanup["id"]
        assert picked["id"] != job_normal["id"]
    finally:
        scheduler._release_claim(picked["id"], token, "owner")


def test_scheduler_repolls_running_job_during_continuous_submissions(monkeypatch):
    repo = Repository(Store("sqlite://"))
    running = repo.create("jobs", "owner", {"state": "running", "provider": "mock"})
    scheduler = Scheduler(repo, clock=lambda: 1000.0)
    monkeypatch.setattr(scheduler, "_run", lambda job: job)
    assert scheduler.tick()["job_id"] == running["id"]

    picked = []
    for _ in range(16):
        repo.create("jobs", "owner", {"state": "queued", "provider": "mock"})
        picked.append(scheduler.tick()["job_id"])

    assert picked[0] != running["id"]
    assert running["id"] in picked[:9]


def test_scheduler_many_stalled_and_awaiting_prep_jobs_never_starve_prep_or_new_jobs(monkeypatch):
    # 1) >= 9 stalled awaiting_prep jobs must never starve the prep job they wait on.
    repo = Repository(Store("sqlite://"))
    scheduler = Scheduler(repo, clock=lambda: 1000.0)
    monkeypatch.setattr(scheduler, "_advance", lambda job: job)

    for i in range(12):
        repo.create(
            "jobs",
            "owner",
            {
                "state": "waiting_for_capacity",
                "awaiting_prep": True,
                "provider": "mock",
                "created_epoch": 100.0 + i,
            },
        )
    for _ in range(12):
        scheduler.tick()

    prep_job = repo.create(
        "jobs",
        "owner",
        {"state": "queued", "prep": True, "provider": "mock", "created_epoch": 200.0},
    )
    assert scheduler.tick()["job_id"] == prep_job["id"]

    # 2) >= 9 normal stalled jobs must never form a closed loop that starves a new queued job.
    repo2 = Repository(Store("sqlite://"))
    scheduler2 = Scheduler(repo2, clock=lambda: 1000.0)
    monkeypatch.setattr(scheduler2, "_advance", lambda job: job)

    for i in range(12):
        repo2.create(
            "jobs",
            "owner",
            {"state": "waiting_for_capacity", "provider": "mock", "created_epoch": 100.0 + i},
        )
    for _ in range(12):
        scheduler2.tick()

    fresh_job = repo2.create(
        "jobs",
        "owner",
        {"state": "queued", "provider": "mock", "created_epoch": 200.0},
    )
    picked2 = [scheduler2.tick()["job_id"] for _ in range(20)]
    assert fresh_job["id"] in picked2

    # 3) A stalled running job with cancel_requested preempts a stream of new queued jobs.
    repo3 = Repository(Store("sqlite://"))
    scheduler3 = Scheduler(repo3, clock=lambda: 1000.0)
    monkeypatch.setattr(scheduler3, "_run", lambda job: job)
    running = repo3.create("jobs", "owner", {"state": "running", "provider": "mock", "created_epoch": 100.0})
    assert scheduler3.tick()["job_id"] == running["id"]

    repo3.create("jobs", "owner", {"state": "queued", "provider": "mock", "created_epoch": 200.0})
    repo3.update("jobs", running["id"], {"cancel_requested": True, "requested_outcome": "cancelled"}, "owner")
    cancel_tick = scheduler3.tick()
    assert cancel_tick["job_id"] == running["id"]
    assert cancel_tick["state"] == "cancelled"


def test_media_cache_headers_thumbnail_disk_cache_etag_304_and_gzip(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "studio_api_token", "test-batch1-token")
    store = Store(f"sqlite:///{tmp_path / 'studio.db'}")
    main.app.state.store = store
    main.app.state.repo = Repository(store)

    proj = store.create("projects", "owner", {"title": "p1"})
    asset = save_asset(store, "owner", proj["id"], _png_bytes(), "sample.png")
    headers = {"Authorization": "Bearer test-batch1-token", "Accept-Encoding": "gzip"}

    open_calls = 0
    real_open = Image.open

    def counting_open(*args, **kwargs):
        nonlocal open_calls
        open_calls += 1
        return real_open(*args, **kwargs)

    monkeypatch.setattr(asset_routes.Image, "open", counting_open)

    with TestClient(main.app) as client:
        # 1. JSON API keeps strict private, no-store AND supports gzip for payloads >= 1 KiB.
        r_jobs = client.get("/api/gpu/workflows", headers=headers)
        assert r_jobs.status_code == 200
        assert r_jobs.headers["cache-control"] == "private, no-store"
        assert r_jobs.headers.get("content-encoding") == "gzip"

        # 2. First thumbnail request encodes once and writes disk cache, keeping private cache header.
        r1 = client.get(f"/api/assets/{asset['id']}/thumbnail", headers=headers)
        assert r1.status_code == 200
        assert open_calls == 1
        assert "private" in r1.headers["cache-control"]
        assert "max-age=" in r1.headers["cache-control"]
        assert "no-store" not in r1.headers["cache-control"]
        assert "content-encoding" not in r1.headers  # already-compressed image/webp is not re-gzipped
        etag = r1.headers["etag"]
        assert etag

        # 3. Second thumbnail request hits disk cache (Pillow is NOT called again).
        r2 = client.get(f"/api/assets/{asset['id']}/thumbnail", headers=headers)
        assert r2.status_code == 200
        assert open_calls == 1
        assert r2.content == r1.content

        # 4. Third thumbnail request with If-None-Match returns 304 Not Modified.
        r3 = client.get(f"/api/assets/{asset['id']}/thumbnail", headers={**headers, "If-None-Match": etag})
        assert r3.status_code == 304
        assert open_calls == 1

        # 5. Preview returns inline + private max-age + ETag + 304.
        rp1 = client.get(f"/api/assets/{asset['id']}/preview", headers=headers)
        assert rp1.status_code == 200
        assert "private" in rp1.headers["cache-control"]
        assert "max-age=" in rp1.headers["cache-control"]
        assert rp1.headers["content-disposition"].startswith("inline")
        rp2 = client.get(f"/api/assets/{asset['id']}/preview", headers={**headers, "If-None-Match": rp1.headers["etag"]})
        assert rp2.status_code == 304

        # 6. Deleting asset cleans up disk thumbnail cache.
        thumb_dir = tmp_path / "asset-thumb-cache"
        assert list(thumb_dir.glob(f"{asset['id']}-*.webp"))
        r_del = client.delete(f"/api/assets/{asset['id']}", headers=headers)
        assert r_del.status_code == 200
        assert not list(thumb_dir.glob(f"{asset['id']}-*.webp"))
