from __future__ import annotations

import httpx
import pytest
from app.db import Store
from app.gpu.colab import ColabError, GoogleColabProvider
from app.gpu.repository import Repository
from app.gpu.scheduler import Scheduler
from app.security import encrypt_secret


class MemoryRepository:
    def __init__(self):
        self.rows = {"credentials": [], "leases": []}
        self.create(
            "credentials",
            "owner-1",
            {
                "provider": "google",
                "purpose": "colab",
                "encrypted_access_token": encrypt_secret("access"),
                "encrypted_refresh_token": encrypt_secret("refresh"),
                "expires_at": 9999999999,
            },
        )

    def create(self, kind, owner_id, data, **kwargs):
        row = {"id": f"{kind}-{len(self.rows.setdefault(kind, [])) + 1}", "owner_id": owner_id, **data}
        self.rows[kind].append(row)
        return dict(row)

    def get(self, kind, record_id, owner_id=None):
        return next(
            (
                dict(r)
                for r in self.rows.get(kind, [])
                if r["id"] == record_id and (owner_id is None or r["owner_id"] == owner_id)
            ),
            None,
        )

    def list(self, kind, owner_id=None, **kwargs):
        return [dict(r) for r in self.rows.get(kind, []) if owner_id is None or r["owner_id"] == owner_id]

    def update(self, kind, record_id, patch, owner_id=None):
        row = self.get(kind, record_id, owner_id)
        if row is None:
            raise KeyError(record_id)
        for current in self.rows[kind]:
            if current["id"] == record_id:
                current.update(patch)
                return dict(current)
        raise KeyError(record_id)


def make(handler):
    repo = MemoryRepository()
    repo.create(
        "leases",
        "owner-1",
        {
            "provider": "colab",
            "runtime_id": "atelier-runtime-1",
            "request_id": "11111111-1111-4111-8111-111111111111",
            "owner_job_id": "job-1",
            "state": "creating",
        },
    )
    return GoogleColabProvider(repo, "owner-1", client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_discovery_read_only_calls():
    paths = []

    def handler(request):
        paths.append((request.method, request.url.path))
        if request.url.path.endswith("runtimespecs"):
            return httpx.Response(200, json={"runtimeSpecs": [{"eligible": True}]})
        return httpx.Response(200, json={"name": "subscription", "tier": "SUBSCRIPTION_TIER_PRO"})

    provider = make(handler)
    assert provider.list_specs()[0]["eligible"] is True
    assert provider.subscription()["tier"] == "SUBSCRIPTION_TIER_PRO"
    assert paths == [("GET", "/v1beta/runtimespecs"), ("GET", "/v1beta/subscription")]


def test_tunnel_subscription_distinguishes_zero_from_missing_balance(monkeypatch):
    responses = iter([
        httpx.Response(200, json={"currentBalance": 0}),
        httpx.Response(200, json={}),
        httpx.Response(404),
    ])

    def handler(request):
        assert request.method == "GET"
        assert request.url.path == "/tun/m/ccu-info"
        return next(responses)

    provider = make(handler)
    monkeypatch.setattr(provider, "_is_cli_tunnel", lambda: True)
    assert provider.subscription()["paidComputeUnitsBalance"] == 0
    assert provider.subscription()["paidComputeUnitsBalance"] is None
    with pytest.raises(ColabError):
        provider.subscription()


def test_create_matches_discovery_and_persists_operation():
    def handler(request):
        assert request.method == "POST" and request.url.path == "/v1beta/runtimes"
        assert request.url.params["requestId"] == "11111111-1111-4111-8111-111111111111"
        assert request.url.params["runtimeId"] == "atelier-runtime-1"
        return httpx.Response(
            200,
            json={
                "name": "operations/op-1",
                "done": True,
                "response": {"name": "runtimes/atelier-runtime-1"},
            },
        )

    provider = make(handler)
    result = provider.create(
        {
            "key": {"variant": "GPU", "accelerator": "T4", "shape": "STANDARD"},
            "eligible": True,
        },
        "11111111-1111-4111-8111-111111111111",
        "job-1",
        "atelier-runtime-1",
    )
    assert result == {
        "runtime_name": "runtimes/atelier-runtime-1",
        "operation_name": "operations/op-1",
        "ready": True,
        "runtime_id": "atelier-runtime-1",
        "request_id": "11111111-1111-4111-8111-111111111111",
        "owner_id": "owner-1",
    }


def test_captured_real_spec_uses_official_enums_and_requires_eligibility():
    captured_specs = [
        {
            "key": {"variant": "VARIANT_GPU", "accelerator": "T4", "shape": "SHAPE_STANDARD"},
            "eligible": True,
        },
        {
            "key": {"variant": "VARIANT_GPU", "accelerator": "H100", "shape": "SHAPE_STANDARD"},
        },
    ]

    assert GoogleColabProvider._spec(captured_specs[0]) == {
        "variant": "VARIANT_GPU",
        "accelerator": "T4",
        "shape": "SHAPE_STANDARD",
    }
    with pytest.raises(ColabError, match="eligible"):
        GoogleColabProvider._spec(captured_specs[1])


def test_captured_real_spec_create_payload_preserves_official_enums():
    captured_spec = {
        "key": {"variant": "VARIANT_GPU", "accelerator": "A100", "shape": "SHAPE_HIGHMEM"},
        "eligible": True,
    }

    def handler(request):
        assert request.method == "POST" and request.url.path == "/v1beta/runtimes"
        assert request.content == (
            b'{"runtimeSpec":{"variant":"VARIANT_GPU","accelerator":"A100","shape":"SHAPE_HIGHMEM"}}'
        )
        return httpx.Response(200, json={"name": "operations/op-captured", "done": True})

    result = make(handler).create(
        captured_spec,
        "11111111-1111-4111-8111-111111111111",
        "job-1",
        "atelier-runtime-1",
    )
    assert result["operation_name"] == "operations/op-captured"


def test_inspect_without_operation_uses_runtime_get():
    paths = []

    def handler(request):
        paths.append(request.url.path)
        assert request.url.path == "/v1beta/runtimes/atelier-runtime-1"
        return httpx.Response(
            200,
            json={
                "name": "runtimes/atelier-runtime-1",
                "connectionInfo": {"url": "https://runtime", "token": "secret"},
            },
        )

    result = make(handler).inspect(
        {
            "runtime_name": "runtimes/atelier-runtime-1",
            "runtime_id": "atelier-runtime-1",
            "owner_id": "owner-1",
        }
    )
    assert result["ready"] is True and "token" not in str(result)
    assert paths == ["/v1beta/runtimes/atelier-runtime-1"]


@pytest.mark.parametrize("ready,expected", [(False, "releasing"), (True, "retained_idle")])
def test_rest_allocation_retention_requires_runtime_readiness(ready, expected):
    repo = Repository(Store("sqlite://"))
    job = repo.create("jobs", "owner", {
        "state": "preparing", "provider": "colab", "lease_id": "lease", "instance_id": "instance",
        "execution_policy": {"lifecycle": "retain", "retain_idle_minutes": 15},
    }, record_id="job")
    repo.create("instances", "owner", {
        "state": "allocating", "provider": "colab", "lease_id": "lease", "current_job_id": "job",
    }, record_id="instance")
    repo.create("leases", "owner", {
        "state": "active", "provider": "colab", "owner_job_id": "job", "create_attempts": 2,
        "operation_name": "operations/create", "operation_done": False,
        "handle": {"runtime_name": "runtimes/studio-test", "operation_name": "operations/create"},
        **({"last_create_error": "previous timeout"} if ready else {}),
    }, record_id="lease")

    class Provider:
        def inspect(self, handle):
            return {"ready": ready, "deleted": False, "operation_pending": not ready}

    scheduler = Scheduler(repo, provider_factory=lambda _: Provider(), clock=lambda: 1000.0)
    result = scheduler._prepare(job)
    lease = repo.get("leases", "lease", "owner")
    assert result["state"] == ("loading" if ready else "preparing")
    assert lease.get("allocation_confirmed") is (True if ready else None)

    # A restarted scheduler can make the decision from the stored allocation evidence.
    if ready:
        repo.update("instances", "instance", {"state": "allocating"}, "owner")
    instance = Scheduler(repo, clock=lambda: 1000.0).instance_service.complete(result, success=ready)
    assert instance["state"] == expected


def test_delete_requires_owner_and_get_404_confirmation():
    paths = []

    def handler(request):
        paths.append((request.method, request.url.path))
        if request.method == "DELETE":
            assert request.url.params["allowMissing"] == "true"
            return httpx.Response(204)
        return httpx.Response(404)

    provider = make(handler)
    assert (
        provider.delete(
            {
                "runtime_name": "runtimes/atelier-runtime-1",
                "runtime_id": "atelier-runtime-1",
                "owner_id": "owner-1",
            }
        )
        is True
    )
    # An already-absent owned runtime is confirmed by GET, without an unnecessary DELETE.
    assert paths == [("GET", "/v1beta/runtimes/atelier-runtime-1")]
    with pytest.raises(Exception, match="归属"):
        provider.delete({"runtime_name": "runtimes/other", "runtime_id": "other", "owner_id": "other-owner"})


def test_create_requests_use_extended_create_timeout_while_reads_stay_fast(monkeypatch):
    from app.gpu.compute import GpuConfig, default_provider_factory

    recorded = []

    def handler(request):
        recorded.append((request.method, request.url.path, request.extensions.get("timeout")))
        if request.url.path == "/tun/m/ccu-info":
            return httpx.Response(200, json={"eligibleGpus": ["L4"]})
        if request.url.path == "/tun/m/assign" and request.method == "GET":
            return httpx.Response(200, json={"token": "xsrf-1"})
        if request.url.path == "/tun/m/assign" and request.method == "POST":
            return httpx.Response(
                200,
                json={
                    "endpoint": "gpu-l4-test",
                    "runtimeProxyInfo": {"url": "https://proxy.colab.test", "token": "ptok", "tokenExpiresInSeconds": 3600},
                },
            )
        return httpx.Response(200, json={"name": "operations/op-1", "done": True})

    provider = make(handler)
    assert provider.create_timeout == 60.0
    monkeypatch.setattr(provider, "_is_cli_tunnel", lambda lease=None: True)
    provider.list_specs()
    provider.create(
        {"key": {"variant": "GPU", "accelerator": "L4", "shape": "STANDARD"}, "eligible": True},
        "11111111-1111-4111-8111-111111111111",
        "job-1",
        "atelier-runtime-1",
    )
    assert recorded[0][1] == "/tun/m/ccu-info"
    assert recorded[0][2]["read"] == 5.0 or recorded[0][2]["read"] <= 10.0
    assert recorded[1][0:2] == ("GET", "/tun/m/assign") and recorded[1][2]["read"] == 60.0
    assert recorded[2][0:2] == ("POST", "/tun/m/assign") and recorded[2][2]["read"] == 60.0

    monkeypatch.setenv("STUDIO_OFFLINE_TEST", "0")
    monkeypatch.setenv("GPU_REAL_ENABLED", "1")
    monkeypatch.setenv("GPU_CREATE_TIMEOUT_SECONDS", "90")
    cfg = GpuConfig.from_env()
    assert cfg.create_timeout_seconds == 90.0
    wired = default_provider_factory({"provider": "colab", "owner_id": "owner-1"}, cfg, store=provider.store)
    try:
        assert wired.create_timeout == 90.0
    finally:
        wired.close()
