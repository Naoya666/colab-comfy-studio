"""Comprehensive tests for three-tier API auth, subpaths, dedicated ports, CORS/CSRF, and Native WS/upload/Manager isolation."""

from __future__ import annotations

import asyncio
import copy
import json
import signal
import socket
import uuid
from contextlib import asynccontextmanager

import httpx
import pytest
from app import native_proxy
from app.api_access import is_native_blocked_path
from app.config import settings
from app.db import Store
from app.gpu import native
from app.gpu.registry import Registry
from app.gpu.repository import Repository
from app.main import app
from app.port_listeners import PortListenerManager, _DedicatedPortASGI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect


def _publish_workflow(repo, owner, version, *, active=True, fixed="approved"):
    manifest, graph, info = Registry._template()
    manifest["compute_spec"] = {"variant": "VARIANT_GPU", "accelerator": "A100", "shape": "SHAPE_STANDARD"}
    from app.gpu.manifest import _binding, _canonical_parameters

    definitions, _ = _canonical_parameters(manifest)
    seed_node, seed_input = _binding("seed", definitions["seed"]["binding"])
    output = graph[seed_node]
    output["inputs"]["filename_prefix"] = fixed
    info[output["class_type"]]["input"]["required"]["filename_prefix"] = ["STRING"]
    row = repo.create("workflows", owner, {
        "workflow_key": "creation-demo",
        "workflow_version": version,
        "model_id": "creation-demo",
        "state": "published",
        "review_approved": True,
        "smoke_test_kind": "real",
        "active": active,
        "sha256": f"test-{version}",
        "manifest": manifest,
        "api_graph": graph,
        "object_info": info,
    })
    return row, seed_node, seed_input


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_creation_three_tiers_subpath_cors_and_active_scope(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "site_url", "https://studio.example.test")
    monkeypatch.setattr(settings, "creation_comfy_url", "")
    monkeypatch.setattr(settings, "studio_api_token", "admin-test")
    monkeypatch.setenv("GPU_REAL_ENABLED", "1")
    monkeypatch.setenv("GPU_UNIT_RATE", "0.2")
    monkeypatch.setattr("app.admission._google_status", lambda *args: {"colab": {"connected": True}})
    monkeypatch.setattr("app.creation_api._google_status", lambda *args: {"colab": {"connected": True}})

    store = Store(f"sqlite:///{tmp_path / 'db'}")
    monkeypatch.setattr(app.state, "store", store, raising=False)
    repo = Repository(store)

    aid = str(uuid.uuid4())
    repo.create("accounts", "owner", {
        "account_id": aid, "provider": "colab", "enabled": True,
        "auth_status": "connected", "allow_allocation": True,
    }, record_id=aid)
    repo.create("models", "owner", {"model_id": "creation-demo", "enabled": True})

    # Two versions differing only in default seed would be ambiguous if both were in scope;
    # workflow_scope="active" selects only the active v2!
    v1, node, field = _publish_workflow(repo, "owner", "1.0.0", active=False, fixed="same")
    v2, _, _ = _publish_workflow(repo, "owner", "2.0.0", active=True, fixed="same")

    admin_auth = {"Authorization": "Bearer admin-test"}
    base = "/api/gpu/creation-api"

    with TestClient(app, base_url="https://studio.example.test") as client:
        status = client.get(base, headers=admin_auth).json()["data"]
        assert status["urls"]["public_site_warning"] is True
        assert status["config"]["auth_mode"] == "strict"

        # Strict mode rejects unauthenticated subpath even without dedicated domain
        assert client.get("/comfy/creation/system_stats").status_code == 401

        # Strong Bearer key works on subpath /comfy/creation WITHOUT needing CREATION_COMFY_URL!
        strong_key = client.post(base + "/keys", headers=admin_auth, json={
            "name": "StrongKey", "workflow_ids": [v1["id"]],
        }).json()["data"]
        strong_headers = {"Authorization": f"Bearer {strong_key['token']}"}
        assert client.get("/comfy/creation/system_stats", headers=strong_headers).status_code == 200
        assert client.get("/comfy/creation/api/object_info", headers=strong_headers).status_code == 200

        # Switch to Tier 2: url_path mode + auto_start
        cfg = {
            **status["config"],
            "auto_start": True,
            "account_id": aid,
            "auth_mode": "url_path",
            "workflow_scope": "active",
        }
        res_save = client.put(base + "/config", headers=admin_auth, json=cfg)
        assert res_save.status_code == 200, res_save.text
        status_t2 = client.get(base, headers=admin_auth).json()["data"]
        code = status_t2["path_code"]
        assert code and status_t2["urls"]["subpath_code_url"].endswith(f"/comfy/c/{code}")

        # Without code on subpath still 401 in url_path mode
        assert client.get("/comfy/creation/system_stats").status_code == 401
        assert client.get("/comfy/c/wrongcode_1234567890/system_stats").status_code == 401

        # Browser CORS preflight & cross-origin direct call (like BaiBai in SillyTavern) succeed with code URL!
        preflight = client.options(
            f"/comfy/c/{code}/prompt",
            headers={
                "Origin": "https://st.sakumoyu.com",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        assert preflight.status_code == 204
        assert preflight.headers["access-control-allow-origin"] == "https://st.sakumoyu.com"

        graph = copy.deepcopy(v2["api_graph"])
        graph[node]["inputs"][field] = 777
        queued = client.post(
            f"/comfy/c/{code}/prompt",
            headers={"Origin": "https://st.sakumoyu.com"},
            json={"prompt": graph},
        )
        assert queued.status_code == 200, queued.text
        assert queued.headers["access-control-allow-origin"] == "https://st.sakumoyu.com"
        prompt_id = queued.json()["prompt_id"]
        job = repo.get("jobs", prompt_id, "owner")
        assert job["creation_workflow_id"] == v2["id"]
        assert job["creation_key_id"] == "__weak_default__"

        # BaiBai queue inspection and JSON-body interrupt work!
        q_resp = client.get(f"/comfy/c/{code}/queue").json()
        assert [item[1] for item in q_resp["queue_pending"]] == [prompt_id]
        assert client.post(f"/comfy/c/{code}/interrupt", json={"prompt_id": prompt_id}).status_code == 200
        assert repo.get("jobs", prompt_id, "owner")["state"].startswith("cancelled")

        # Strong key still coexists and cannot read weak-channel jobs
        assert client.get(f"/comfy/creation/history/{prompt_id}", headers=strong_headers).status_code == 404

        # Rotating code invalidates old code immediately
        rotated = client.post(base + "/reset-code", headers=admin_auth).json()["data"]
        new_code = rotated["path_code"]
        assert new_code != code
        assert client.get(f"/comfy/c/{code}/system_stats").status_code == 401
        assert client.get(f"/comfy/c/{new_code}/system_stats").status_code == 200

        # Switch to Tier 3: local_open mode
        cfg_t3 = {
            **cfg,
            "auth_mode": "local_open",
            "allow_lan": False,
            "allowed_origins": ["https://trusted-st.example.test"],
        }
        assert client.put(base + "/config", headers=admin_auth, json=cfg_t3).status_code == 200

        # Local server-to-server request (no Origin) works without code!
        assert client.get("/comfy/creation/system_stats").status_code == 200
        # Localhost browser Origin works without code!
        local_origin = client.get("/comfy/creation/object_info", headers={"Origin": "http://127.0.0.1:8000"})
        assert local_origin.status_code == 200
        assert local_origin.headers["access-control-allow-origin"] == "http://127.0.0.1:8000"
        # Whitelisted external Origin works!
        whitelisted = client.get("/comfy/creation/system_stats", headers={"Origin": "https://trusted-st.example.test"})
        assert whitelisted.status_code == 200
        # Untrusted external Origin is blocked with 403 and NO CORS allow header!
        evil = client.post("/comfy/creation/prompt", headers={"Origin": "https://evil-ad.example"}, json={"prompt": graph})
        assert evil.status_code == 403
        assert "access-control-allow-origin" not in evil.headers

        # Public X-Forwarded-For IP is rejected in local_open mode!
        pub = client.get("/comfy/creation/system_stats", headers={"X-Forwarded-For": "203.0.113.99"})
        assert pub.status_code == 401
        # Private LAN IP is rejected when allow_lan=False, accepted when allow_lan=True!
        assert client.get("/comfy/creation/system_stats", headers={"X-Forwarded-For": "172.19.0.2"}).status_code == 401
        assert client.put(base + "/config", headers=admin_auth, json={**cfg_t3, "allow_lan": True}).status_code == 200
        assert client.get("/comfy/creation/system_stats", headers={"X-Forwarded-For": "172.19.0.2"}).status_code == 200


def test_native_weak_modes_allow_ws_and_upload_while_blocking_manager(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "site_url", "http://127.0.0.1:8788")
    monkeypatch.setattr(settings, "native_comfy_url", "")
    monkeypatch.setattr(settings, "studio_api_token", "admin-test")
    monkeypatch.setattr(native, "backend", lambda: ("https://internal.example.test", {"CF-Access-Client-Id": "id", "CF-Access-Client-Secret": "sec"}))

    store = Store(f"sqlite:///{tmp_path / 'native.db'}")
    monkeypatch.setattr(app.state, "store", store, raising=False)
    repo = Repository(store)

    upstream_calls = []

    class StreamBytes(httpx.AsyncByteStream):
        def __init__(self, data: bytes):
            self.data = data

        async def __aiter__(self):
            yield self.data

    async def upstream(req: httpx.Request):
        assert req.headers["origin"] == "https://internal.example.test"
        body = await req.aread()
        upstream_calls.append((req.method, req.url.raw_path.decode("ascii"), body))
        if req.url.path in {"/prompt", "/api/prompt"}:
            parsed = json.loads(body)
            payload = json.dumps({"prompt_id": "p-weak-1", "client_id": parsed.get("client_id")}).encode()
            return httpx.Response(200, stream=StreamBytes(payload), headers={"Content-Type": "application/json"})
        if req.url.path == "/upload/image":
            return httpx.Response(200, stream=StreamBytes(b'{"name":"ref.png"}'), headers={"Content-Type": "application/json"})
        return httpx.Response(200, stream=StreamBytes(b'{"system":{"os":"linux"}}'), headers={"Content-Type": "application/json"})

    original_client = httpx.AsyncClient
    monkeypatch.setattr(native_proxy.httpx, "AsyncClient", lambda **kw: original_client(transport=httpx.MockTransport(upstream), **kw))

    class FakeUpstreamWS:
        subprotocol = None

        def __init__(self):
            self.q = asyncio.Queue()

        async def send(self, msg):
            await self.q.put(msg)

        def __aiter__(self):
            return self

        async def __anext__(self):
            return await self.q.get()

    ws_targets = []

    @asynccontextmanager
    async def fake_ws_connect(url, **kwargs):
        ws_targets.append(url)
        yield FakeUpstreamWS()

    monkeypatch.setattr(native_proxy, "connect", fake_ws_connect)

    admin_auth = {"Authorization": "Bearer admin-test"}
    settings_url = "/api/gpu/native/api-settings"

    with TestClient(app, base_url="http://127.0.0.1:8788") as client:
        repo.create("leases", "owner", {
            "state": "idle", "native_ready": True, "native_comfy": True, "instance_id": "inst-1",
        })

        # Strict mode rejects subpath without auth
        assert client.get("/comfy/native/system_stats").status_code == 401

        # Enable url_path mode
        current = client.get(settings_url, headers=admin_auth).json()["data"]["policy"]
        saved = client.put(settings_url, headers=admin_auth, json={**current, "auth_mode": "url_path"})
        assert saved.status_code == 200, saved.text
        meta = client.get(settings_url, headers=admin_auth).json()["data"]
        code = meta["path_code"]
        assert code

        # Full generation endpoints work over /comfy/n/<code>: system_stats, upload/image, prompt, view, queue, interrupt, AND WebSocket /ws!
        assert client.get(f"/comfy/n/{code}/system_stats").status_code == 200
        upload = client.post(f"/comfy/n/{code}/upload/image", content=b"PNGDATA", headers={"Content-Type": "application/octet-stream"})
        assert upload.status_code == 200 and upload.json()["name"] == "ref.png"

        prompt_res = client.post(
            f"/comfy/n/{code}/prompt",
            headers={"Origin": "https://st.sakumoyu.com"},
            json={"prompt": {"1": {"class_type": "KSampler", "inputs": {}}}, "client_id": "ws-client-1"},
        )
        assert prompt_res.status_code == 200
        assert prompt_res.json()["client_id"] == "ws-client-1"
        assert prompt_res.headers["access-control-allow-origin"] == "https://st.sakumoyu.com"

        approved = {"1": {"class_type": "KSampler", "inputs": {}}}
        assert client.put(settings_url, headers=admin_auth, json={
            **saved.json()["data"], "weak_workflow_mode": "approved", "weak_workflow": approved,
        }).status_code == 200
        prompt_calls = len(upstream_calls)
        assert client.post(f"/comfy/n/{code}/api/prompt/", json={
            "prompt": {"1": {"class_type": "Example", "inputs": {}}},
        }).status_code == 403
        assert len(upstream_calls) == prompt_calls

        with client.websocket_connect(f"/comfy/n/{code}/ws?clientId=ws-client-1") as ws:
            ws.send_text('{"ping":1}')
            assert ws.receive_text() == '{"ping":1}'
            ws.send_bytes(b"\x01\x02latent")
            assert ws.receive_bytes() == b"\x01\x02latent"
        assert ws_targets[-1] == "wss://internal.example.test/ws?clientId=ws-client-1"

        # Manager, customnode, reboot, and internal endpoints are HARD BLOCKED in weak modes!
        for forbidden in (
            "/manager/queue/status",
            "/v2/manager/queue/install_model",
            "/api/v2/manager/reboot",
            "/customnode/install",
            "/v2/customnode/install",
            "/reboot",
            "/_studio/session",
        ):
            resp = client.post(f"/comfy/n/{code}{forbidden}", json={})
            assert resp.status_code in {403, 404}, f"expected block on {forbidden}, got {resp.status_code}"
        assert not any("manager" in p or "customnode" in p for _, p, _ in upstream_calls)

        # WebSocket on blocked path is rejected too
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(f"/comfy/n/{code}/manager/ws"):
                pass

        # Enable local_open mode and verify /comfy/native/ws and HTTP work locally without code
        assert client.put(settings_url, headers=admin_auth, json={**current, "auth_mode": "local_open"}).status_code == 200
        assert client.get("/comfy/native/system_stats").status_code == 200
        with client.websocket_connect("/comfy/native/ws?clientId=local-ws") as ws:
            ws.send_text("hello")
            assert ws.receive_text() == "hello"
        # Manager remains blocked in local_open mode too!
        assert client.post("/comfy/native/customnode/install", json={}).status_code == 403


def test_weak_native_path_rejects_ambiguous_representations():
    assert is_native_blocked_path("/files/./image", b"/files/./image")
    assert is_native_blocked_path("/files//image", b"/files//image")
    assert is_native_blocked_path("/files/image", b"/files%2Fimage")
    assert is_native_blocked_path("/files/image", b"/other/image")
    assert not is_native_blocked_path("/files/图像.png", b"/files/%E5%9B%BE%E5%83%8F.png")


@pytest.mark.asyncio
async def test_dedicated_listener_leaves_main_process_signal_handlers_untouched():
    before = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    manager = PortListenerManager(app)
    try:
        await manager.apply("creation", 0)
        assert manager._listeners["creation"].server.started
        assert all(signal.getsignal(sig) == handler for sig, handler in before.items())
    finally:
        await manager.shutdown()


def test_dedicated_port_listeners_and_conflict_validation(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "site_url", "http://127.0.0.1:8788")
    monkeypatch.setattr(settings, "creation_comfy_url", "")
    monkeypatch.setattr(settings, "native_comfy_url", "")
    monkeypatch.setattr(settings, "studio_api_token", "admin-test")

    store = Store(f"sqlite:///{tmp_path / 'ports.db'}")
    monkeypatch.setattr(app.state, "store", store, raising=False)
    repo = Repository(store)
    repo.create("models", "owner", {"model_id": "creation-demo", "enabled": True})
    _publish_workflow(repo, "owner", "1.0.0", active=True)

    c_port = _free_port()
    admin_auth = {"Authorization": "Bearer admin-test"}

    with TestClient(app, base_url="http://127.0.0.1:8788") as client:
        c_cfg = client.get("/api/gpu/creation-api", headers=admin_auth).json()["data"]["config"]
        # Reject studio port conflict
        assert client.put(
            "/api/gpu/creation-api/config", headers=admin_auth, json={**c_cfg, "listen_port": 8788}
        ).status_code == 409

        # Bind Creation API on dedicated local port in local_open mode
        saved = client.put(
            "/api/gpu/creation-api/config",
            headers=admin_auth,
            json={**c_cfg, "auth_mode": "local_open", "listen_port": c_port},
        )
        assert saved.status_code == 200, saved.text
        c_status = client.get("/api/gpu/creation-api", headers=admin_auth).json()["data"]
        assert c_status["listener"]["active"] is True
        assert c_status["listener"]["port"] == c_port
        assert c_status["urls"]["port_url"] == f"http://127.0.0.1:{c_port}"

        # Reject Native API claiming the same port as Creation API
        n_pol = client.get("/api/gpu/native/api-settings", headers=admin_auth).json()["data"]["policy"]
        conflict = client.put(
            "/api/gpu/native/api-settings", headers=admin_auth, json={**n_pol, "listen_port": c_port}
        )
        assert conflict.status_code == 409

        # Exercise the dedicated-port ASGI application directly (respects offline_gate socket.connect ban)
        with TestClient(_DedicatedPortASGI(app, "creation"), base_url=f"http://127.0.0.1:{c_port}") as port_client:
            r = port_client.get("/system_stats")
            assert r.status_code == 200
            assert r.json()["devices"][0]["name"] == "L4"
            code = c_status["path_code"]
            assert port_client.get(f"/t/{code}/system_stats").status_code == 200

        # Disable port and verify listener stops cleanly
        assert client.put(
            "/api/gpu/creation-api/config",
            headers=admin_auth,
            json={**c_cfg, "auth_mode": "local_open", "listen_port": None},
        ).status_code == 200
        assert client.get("/api/gpu/creation-api", headers=admin_auth).json()["data"]["listener"]["active"] is False
