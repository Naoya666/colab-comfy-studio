"""End-to-end tests for structured Colab allocation errors, Option-A retries, and 60s backoff alignment."""

from __future__ import annotations

import httpx
import pytest
from app.db import Store
from app.gpu.colab import GoogleColabProvider
from app.gpu.compute import GpuConfig
from app.gpu.google import GoogleOAuthError
from app.gpu.instances import lease_allocation_unconfirmed
from app.gpu.repository import Repository
from app.gpu.scheduler import Scheduler
from app.presenters import public_job


def _setup_tunnel_env(monkeypatch, handler, *, now=1000.0, lifecycle="retain", create_retries=2):
    clock = {"now": float(now)}
    monkeypatch.setattr("app.gpu.colab.time.time", lambda: clock["now"])
    repo = Repository(Store("sqlite://"))
    repo.create(
        "accounts",
        "owner",
        {
            "account_id": "acc",
            "alias": "colab-cli",
            "enabled": True,
            "allow_allocation": True,
            "connected": True,
        },
        record_id="acc",
    )
    job = repo.create(
        "jobs",
        "owner",
        {
            "project_id": "proj",
            "state": "queued",
            "provider": "colab",
            "account_id": "acc",
            "workflow_key": "anima-t2i",
            "workflow_version": 1,
            "max_runtime_minutes": 30,
            "requested_spec": {"key": {"variant": "VARIANT_GPU", "accelerator": "L4", "shape": "SHAPE_STANDARD"}},
            "execution_policy": {
                "provider": "colab",
                "account_id": "acc",
                "lifecycle": lifecycle,
                "retain_idle_minutes": 15,
            },
            "snapshot": {
                "manifest": {
                    "compute_spec": {"variant": "VARIANT_GPU", "accelerator": "L4", "shape": "SHAPE_STANDARD"},
                }
            },
        },
        record_id="job-1",
    )
    client = httpx.Client(transport=httpx.MockTransport(handler))
    google = type("FakeGoogle", (), {"access_token": lambda self, purpose: "fake-access"})()

    def make_provider(_job):
        p = GoogleColabProvider(repo, "owner", account_id="acc", client=client, google=google)
        p._is_cli_tunnel = lambda lease=None: True
        return p

    scheduler = Scheduler(
        repo,
        provider_factory=make_provider,
        config=GpuConfig(enabled=True, real_enabled=True, create_retries=create_retries),
        clock=lambda: clock["now"],
    )
    return repo, scheduler, clock, job["id"], client


def _drive_until(scheduler, repo, job_id, stop_states, *, limit=15):
    for _ in range(limit):
        scheduler.tick()
        job = repo.get("jobs", job_id, "owner")
        if job["state"] in stop_states:
            return job
    return repo.get("jobs", job_id, "owner")


def test_tunnel_post_503_preserves_60s_reconciliation_root_cause_and_aligned_backoff(monkeypatch):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.url.path == "/tun/m/ccu-info":
            return httpx.Response(200, json={"eligibleGpus": ["L4"], "currentBalance": 100})
        if request.url.path == "/tun/m/assign" and request.method == "GET":
            return httpx.Response(200, json={"token": "xsrf-503", "nbh": request.url.params["nbh"]})
        if request.url.path == "/tun/m/assign" and request.method == "POST":
            return httpx.Response(503, text="upstream busy")
        if request.url.path == "/tun/m/assignments":
            return httpx.Response(200, json={"assignments": []})
        raise AssertionError(f"unexpected {request.method} {request.url.path}")

    repo, scheduler, clock, job_id, client = _setup_tunnel_env(monkeypatch, handler)
    try:
        # Advance through validating -> waiting_for_capacity -> allocating (503) -> cleaning_up -> cleanup_pending.
        pending = _drive_until(scheduler, repo, job_id, {"cleanup_pending", "failed"})
        assert pending["state"] == "cleanup_pending"
        # First-tick attribution: real HTTP 503 root cause is preserved (never overwritten by gpu_allocation_aborted).
        assert pending["error_code"] == "colab_upstream_unavailable"
        assert "HTTP 503" in pending["error_message"]
        assert pending["error_details"]["phase"] == "assign_post"
        assert pending["error_details"]["http_status"] == 503
        assert pending["error_details"]["reconciling"] is True
        assert pending["error_details"]["elapsed_seconds"] is not None

        # Instance must NOT be retained idle; backoff must align directly to create_started + 60.5s.
        lease = repo.get("leases", pending["lease_id"], "owner")
        inst = repo.get("instances", pending["instance_id"], "owner")
        assert inst["state"] == "releasing"
        assert lease["state"] == "cleanup_pending"
        expected_ready = float(lease["create_started"]) + 60.5
        assert abs(float(lease["cleanup_next_epoch"]) - expected_ready) < 0.01

        # Ticking before the 60s window expires does not spam delete attempts or network calls.
        calls_before = len(calls)
        clock["now"] = expected_ready - 5.0
        scheduler.tick()
        assert len(calls) == calls_before
        assert repo.get("leases", lease["id"], "owner")["delete_attempts"] == 1

        # Right at the aligned 60s window, reconciliation confirms unassigned and finalizes cleanly.
        clock["now"] = expected_ready
        done = _drive_until(scheduler, repo, job_id, {"failed"})
        assert done["state"] == "failed"
        assert done["error_code"] == "colab_upstream_unavailable"
        assert done["error_details"]["reconciling"] is False
        assert done["error_details"]["reconciliation_result"] == "confirmed_unassigned"
        assert repo.get("leases", lease["id"], "owner")["state"] == "deleted"
        assert repo.get("instances", inst["id"], "owner")["state"] == "deleted"
    finally:
        client.close()


def test_tunnel_post_read_timeout_reconciles_late_runtime_and_records_diagnostics(monkeypatch):
    late_assigned = {"active": True}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/tun/m/ccu-info":
            return httpx.Response(200, json={"eligibleGpus": ["L4"], "currentBalance": 100})
        if request.url.path == "/tun/m/assign" and request.method == "GET":
            nbh = request.url.params["nbh"]
            if late_assigned["active"] and any(p[0] == "POST" for p in seen):
                return httpx.Response(200, json={"endpoint": "gpu-l4-late", "nbh": nbh})
            return httpx.Response(200, json={"token": "xsrf-to", "nbh": nbh})
        if request.url.path == "/tun/m/assign" and request.method == "POST":
            seen.append(("POST", request.url.path))
            raise httpx.ReadTimeout("timed out reading assign")
        if request.url.path == "/tun/m/assignments":
            rows = [{"endpoint": "gpu-l4-late", "nbh": "any"}] if late_assigned["active"] else []
            return httpx.Response(200, json={"assignments": rows})
        if request.url.path == "/tun/m/unassign/gpu-l4-late" and request.method == "GET":
            return httpx.Response(200, json={"token": "xsrf-unassign"})
        if request.url.path == "/tun/m/unassign/gpu-l4-late" and request.method == "POST":
            late_assigned["active"] = False
            return httpx.Response(200, json={})
        raise AssertionError(f"unexpected {request.method} {request.url.path}")

    seen: list[tuple[str, str]] = []
    repo, scheduler, clock, job_id, client = _setup_tunnel_env(monkeypatch, handler)
    try:
        pending = _drive_until(scheduler, repo, job_id, {"cleanup_pending", "failed"})
        assert pending["state"] == "cleanup_pending"
        assert pending["error_code"] == "colab_assign_timeout"
        assert pending["error_details"]["phase"] == "assign_post"
        assert pending["error_details"]["timeout_kind"] == "read"
        assert pending["error_details"]["reconciling"] is True

        lease = repo.get("leases", pending["lease_id"], "owner")
        clock["now"] = float(lease["cleanup_next_epoch"])
        done = _drive_until(scheduler, repo, job_id, {"failed"})
        assert done["state"] == "failed"
        assert done["error_code"] == "colab_assign_timeout"
        assert done["error_details"]["reconciling"] is False
        assert done["error_details"]["reconciliation_result"] == "late_runtime_deleted"
        assert late_assigned["active"] is False
    finally:
        client.close()


@pytest.mark.parametrize(
    "post_response,expected_code,expected_status,expected_outcome",
    [
        (httpx.Response(401, text="unauth"), "colab_auth_failed", 401, None),
        (httpx.Response(403, text="forbidden"), "colab_auth_failed", 403, None),
        (httpx.Response(412, text="quota"), "colab_quota_exceeded", 412, None),
        (httpx.Response(429, text="rate limit"), "colab_rate_limited", 429, None),
        (httpx.Response(200, json={"outcome": 1}), "gpu_capacity_unavailable", 200, "QUOTA_DENIED_REQUESTED_VARIANTS"),
        (httpx.Response(200, json={"outcome": 2}), "colab_quota_exceeded", 200, "QUOTA_EXCEEDED_USAGE_TIME"),
        (httpx.Response(200, json={"outcome": 5}), "colab_auth_failed", 200, "DENYLISTED"),
    ],
)
def test_explicit_post_refusal_skips_60s_wait_and_finalizes_immediately(
    monkeypatch, post_response, expected_code, expected_status, expected_outcome
):
    post_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal post_calls
        if request.url.path == "/tun/m/ccu-info":
            return httpx.Response(200, json={"eligibleGpus": ["L4"], "currentBalance": 100})
        if request.url.path == "/tun/m/assign" and request.method == "GET":
            return httpx.Response(200, json={"token": "xsrf-refuse", "nbh": request.url.params["nbh"]})
        if request.url.path == "/tun/m/assign" and request.method == "POST":
            post_calls += 1
            return post_response
        raise AssertionError(f"explicit refusal must not require remote reconcile call: {request.method} {request.url.path}")

    repo, scheduler, _clock, job_id, client = _setup_tunnel_env(monkeypatch, handler)
    try:
        done = _drive_until(scheduler, repo, job_id, {"failed", "cleanup_pending"})
        assert done["state"] == "failed"
        assert post_calls == 1
        assert done["error_code"] == expected_code
        assert done["error_details"]["phase"] == "assign_post"
        assert done["error_details"]["http_status"] == expected_status
        assert done["error_details"].get("outcome") == expected_outcome
        assert done["error_details"]["reconciling"] is False
        assert done["error_details"]["reconciliation_result"] == "explicit_provider_refusal"
        assert repo.get("leases", done["lease_id"], "owner")["state"] == "deleted"
        assert repo.get("instances", done["instance_id"], "owner")["state"] == "deleted"
    finally:
        client.close()


def test_pre_post_transient_503_retries_once_then_succeeds(monkeypatch):
    get_assign_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal get_assign_calls
        if request.url.path == "/tun/m/ccu-info":
            return httpx.Response(200, json={"eligibleGpus": ["L4"], "currentBalance": 100})
        if request.url.path == "/tun/m/assign" and request.method == "GET":
            get_assign_calls += 1
            if get_assign_calls == 1:
                return httpx.Response(503, text="transient")
            return httpx.Response(200, json={"token": "xsrf-ok", "nbh": request.url.params["nbh"]})
        if request.url.path == "/tun/m/assign" and request.method == "POST":
            return httpx.Response(
                200,
                json={
                    "endpoint": "gpu-l4-ok",
                    "runtimeProxyInfo": {"url": "https://proxy.example", "token": "ptok", "tokenExpiresInSeconds": 3600},
                },
            )
        raise AssertionError(f"unexpected {request.method} {request.url.path}")

    repo, scheduler, _clock, job_id, client = _setup_tunnel_env(monkeypatch, handler)
    try:
        ready_job = _drive_until(scheduler, repo, job_id, {"preparing", "failed"})
        assert ready_job["state"] == "preparing"
        assert get_assign_calls == 2
        lease = repo.get("leases", ready_job["lease_id"], "owner")
        assert lease["state"] == "active"
        assert lease.get("last_create_error") is None
    finally:
        client.close()


def test_pre_post_transient_failure_caps_at_one_retry_and_skips_60s_wait(monkeypatch):
    get_assign_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal get_assign_calls
        if request.url.path == "/tun/m/ccu-info":
            return httpx.Response(200, json={"eligibleGpus": ["L4"], "currentBalance": 100})
        if request.url.path == "/tun/m/assign" and request.method == "GET":
            get_assign_calls += 1
            raise httpx.ConnectError("connection refused")
        raise AssertionError(f"unexpected {request.method} {request.url.path}")

    repo, scheduler, _clock, job_id, client = _setup_tunnel_env(monkeypatch, handler, create_retries=2)
    try:
        done = _drive_until(scheduler, repo, job_id, {"failed", "cleanup_pending"})
        assert done["state"] == "failed"
        # Option A: max 1 automatic retry before POST (total 2 attempts), zero 60s wait.
        assert get_assign_calls == 2
        assert done["error_code"] == "colab_network_error"
        assert done["error_details"]["phase"] == "assign_get"
        assert done["error_details"]["timeout_kind"] == "connect"
        assert done["error_details"]["retry_attempt"] == 2
        assert done["error_details"]["reconciling"] is False
        assert done["error_details"]["reconciliation_result"] == "pre_post_unallocated"
        assert repo.get("instances", done["instance_id"], "owner")["state"] == "deleted"
    finally:
        client.close()


def test_pre_post_403_and_oauth_error_fail_fast_without_retry_or_60s_wait(monkeypatch):
    get_assign_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal get_assign_calls
        if request.url.path == "/tun/m/ccu-info":
            return httpx.Response(200, json={"eligibleGpus": ["L4"], "currentBalance": 100})
        if request.url.path == "/tun/m/assign" and request.method == "GET":
            get_assign_calls += 1
            return httpx.Response(403, text="forbidden")
        raise AssertionError(f"unexpected {request.method} {request.url.path}")

    repo, scheduler, _clock, job_id, client = _setup_tunnel_env(monkeypatch, handler, create_retries=2)
    try:
        done = _drive_until(scheduler, repo, job_id, {"failed", "cleanup_pending"})
        assert done["state"] == "failed"
        assert get_assign_calls == 1
        assert done["error_code"] == "colab_auth_failed"
        assert done["error_details"]["phase"] == "assign_get"
        assert done["error_details"]["http_status"] == 403
        assert done["error_details"]["reconciling"] is False
        assert done["error_details"]["reconciliation_result"] == "pre_post_unallocated"
    finally:
        client.close()


def test_oauth_refresh_error_in_create_classifies_auth_phase_and_skips_60s(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/tun/m/ccu-info":
            return httpx.Response(200, json={"eligibleGpus": ["L4"], "currentBalance": 100})
        raise AssertionError(f"unexpected {request.method} {request.url.path}")

    repo, scheduler, _clock, job_id, client = _setup_tunnel_env(monkeypatch, handler, create_retries=2)
    calls = {"token": 0}

    def broken_provider(_job):
        class FlakyGoogle:
            def access_token(self, purpose):
                calls["token"] += 1
                if calls["token"] <= 2:
                    return "ok-for-specs"
                raise GoogleOAuthError("Google OAuth刷新失败")

        p = GoogleColabProvider(repo, "owner", account_id="acc", client=client, google=FlakyGoogle())
        p._is_cli_tunnel = lambda lease=None: True
        return p

    scheduler.provider_factory = broken_provider
    try:
        done = _drive_until(scheduler, repo, job_id, {"failed", "cleanup_pending"})
        assert done["state"] == "failed"
        assert done["error_code"] == "colab_auth_failed"
        assert done["error_details"]["phase"] == "auth"
        assert done["error_details"]["reconciling"] is False
        assert done["error_details"]["reconciliation_result"] == "pre_post_unallocated"
    finally:
        client.close()


def test_unknown_post_response_without_denial_outcome_keeps_60s_reconciliation(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/tun/m/ccu-info":
            return httpx.Response(200, json={"eligibleGpus": ["L4"], "currentBalance": 100})
        if request.url.path == "/tun/m/assign" and request.method == "GET":
            return httpx.Response(200, json={"token": "xsrf-unk", "nbh": request.url.params["nbh"]})
        if request.url.path == "/tun/m/assign" and request.method == "POST":
            return httpx.Response(200, json={"unexpected": True})
        raise AssertionError(f"unexpected {request.method} {request.url.path}")

    repo, scheduler, _clock, job_id, client = _setup_tunnel_env(monkeypatch, handler)
    try:
        pending = _drive_until(scheduler, repo, job_id, {"cleanup_pending", "failed"})
        assert pending["state"] == "cleanup_pending"
        assert pending["error_code"] == "colab_invalid_response"
        assert pending["error_details"]["phase"] == "assign_post"
        assert pending["error_details"]["http_status"] == 200
        assert pending["error_details"]["reconciling"] is True
    finally:
        client.close()


def test_presenter_allowlists_error_details_and_unconfirmed_guard_covers_new_codes():
    presented = public_job(
        {
            "id": "job-1",
            "provider": "colab",
            "state": "failed",
            "error_code": "colab_upstream_unavailable",
            "error_message": "Colab 服务端暂时不可用（HTTP 503，等待 32.9s）",
            "error_details": {
                "error_code": "colab_upstream_unavailable",
                "category": "upstream",
                "phase": "assign_post",
                "http_status": 503,
                "elapsed_seconds": 32.9,
                "reconciling": False,
                "reconciliation_result": "confirmed_unassigned",
                "secret_token": "must-not-leak",
                "raw_url": "https://private.example",
            },
        }
    )
    assert presented["error_details"] == {
        "error_code": "colab_upstream_unavailable",
        "category": "upstream",
        "phase": "assign_post",
        "http_status": 503,
        "elapsed_seconds": 32.9,
        "reconciling": False,
        "reconciliation_result": "confirmed_unassigned",
    }
    assert "secret_token" not in str(presented)
    assert "private.example" not in str(presented)

    for code in (
        "colab_upstream_unavailable",
        "colab_assign_timeout",
        "colab_network_error",
        "colab_auth_failed",
        "colab_rate_limited",
        "colab_quota_exceeded",
    ):
        assert lease_allocation_unconfirmed("active", {"state": "active"}, {"error_code": code}) is True
