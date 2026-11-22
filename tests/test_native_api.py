"""Exercise the dedicated key contract through Studio and the native proxy."""

import asyncio
import json

import httpx
import pytest
from app import native_api, native_proxy
from app.common import APIError
from app.config import settings
from app.db import Store
from app.gpu import native
from app.gpu.repository import Repository
from app.native_proxy import NativeHostMiddleware
from app.native_routes import router
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.testclient import TestClient

GRAPH = {"1": {"class_type": "CLIPTextEncode", "inputs": {"text": "%prompt%", "clip": ["2", 0]}}}
RENDERED = {"1": {"class_type": "CLIPTextEncode", "inputs": {"text": "test prompt", "clip": ["2", 0]}}}
ADMIN = {"Authorization": "Bearer studio-admin"}
FILE = {"filename": "test.png", "subfolder": "generated", "type": "output"}


class ImageStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b"image"


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setattr(settings, "studio_api_token", "studio-admin")
    monkeypatch.setattr(settings, "native_comfy_url", "https://comfy.example.test")
    monkeypatch.setattr(settings, "site_url", "https://studio.example.test")
    monkeypatch.setattr(native, "backend", lambda: ("https://internal.example.test", {}))
    app = FastAPI()
    app.include_router(router)
    app.add_middleware(NativeHostMiddleware, studio_app=app)
    store = Store("sqlite:///:memory:")
    app.state.repo = Repository(store)
    lease = app.state.repo.create("leases", "owner", {
        "state": "idle", "native_ready": True, "native_comfy": True,
        "instance_id": "instance",
    })

    @app.exception_handler(APIError)
    async def api_error(request, exc):
        return JSONResponse({"error": exc.code}, status_code=exc.status)

    try:
        with TestClient(app, base_url=settings.site_url) as client:
            yield client, app.state.repo, lease
    finally:
        store.engine.dispose()


def make_key(client, name, auto_start=False):
    response = client.post("/api/gpu/native/api-keys", headers=ADMIN, json={
        "name": name, "mode": "approved", "auto_start": auto_start, "workflow": GRAPH,
    })
    assert response.status_code == 201
    key = response.json()["data"]
    return key, {"Authorization": f"Bearer {key['token']}"}


def test_native_key_isolation_and_template(server, monkeypatch):
    client, repo, _ = server
    first, auth = make_key(client, "local")
    _, other = make_key(client, "external")
    used = []

    async def upstream(req):
        assert "Authorization" not in req.headers
        used.append(req.url.path)
        if req.url.path == "/prompt":
            assert json.loads(await req.aread()) == {"prompt": RENDERED}
            return httpx.Response(200, json={"prompt_id": "p-1"})
        if req.url.path == "/history/p-1":
            return httpx.Response(200, json={"p-1": {"outputs": {"1": {"images": [FILE]}}}, "p-2": {"private": True}})
        return httpx.Response(200, stream=ImageStream(), headers={"Content-Type": "image/png"})

    original = httpx.AsyncClient
    monkeypatch.setattr(native_proxy.httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(upstream), **kwargs))

    comfy = settings.native_comfy_url
    assert client.get(f"{comfy}/system_stats").status_code == 401
    assert client.get("/api/gpu/native/api-settings", headers=auth).status_code == 401
    assert client.post(f"{comfy}/interrupt", headers=auth).status_code == 403
    assert client.post(f"{comfy}/api//prompt", headers=auth, json={"prompt": RENDERED}).status_code == 403
    assert client.post(f"{comfy}/prompt", headers=auth, json={"prompt": GRAPH | {
        "2": {"class_type": "ExtraNode", "inputs": {}},
    }}).status_code == 403
    assert client.post(f"{comfy}/api/prompt", headers=auth, json={
        "prompt": RENDERED, "number": -100, "extra_data": {"hidden": True},
    }).json() == {"prompt_id": "p-1"}
    assert client.get(f"{comfy}/history/p-1", headers=other).status_code == 404
    assert client.get(f"{comfy}/history/p-1", headers=auth).json() == {
        "p-1": {"outputs": {"1": {"images": [FILE]}}},
    }
    assert client.get(f"{comfy}/view", headers=other, params=FILE).status_code == 404
    assert client.get(f"{comfy}/view", headers=auth, params={**FILE, "type": "input"}).status_code == 400
    assert client.get(f"{comfy}/view", headers=auth, params=FILE).content == b"image"

    assert used == ["/prompt", "/history/p-1", "/view"]
    listed = client.get("/api/gpu/native/api-settings", headers=ADMIN).json()["data"]
    assert "token" not in str(listed) and "digest" not in str(listed)
    assert repo.store.get(native_api.KEYS, first["id"])["token_digest"] != first["token"]
    assert next(key for key in listed["keys"] if key["id"] == first["id"])["submitted"] == 1
    assert client.delete(f"/api/gpu/native/api-keys/{first['id']}", headers=ADMIN).status_code == 200
    assert client.get(f"{comfy}/_studio/api/status", headers=auth).status_code == 401


def test_automatic_start_timeout_never_submits_workflow(server, monkeypatch):
    client, repo, lease = server
    repo.update("leases", lease["id"], {"native_ready": False}, "owner")
    repo.create("accounts", "owner", {"enabled": True}, record_id="account")
    values = {
        "auto_start": True, "account_id": "account",
        "gpu": {"variant": "VARIANT_GPU", "accelerator": "L4", "shape": "SHAPE_STANDARD"},
        "retain_idle_minutes": 9, "max_runtime_minutes": 65,
        "model_storage_mode": "local", "cache_gb": 33, "cache_max_age_minutes": 115,
    }
    response = client.put("/api/gpu/native/api-settings", headers=ADMIN, json=values)
    assert response.status_code == 200, response.text
    _, auth = make_key(client, "may start", auto_start=True)
    _, read_only = make_key(client, "keep running")
    started = []

    def admit(repo, owner, body, key, *, api_key_id):
        started.append((owner, body, api_key_id))
        return {"id": "starting", "state": "queued"}

    monkeypatch.setattr(native, "admit", admit)
    monkeypatch.setattr(native_api, "START_WAIT_SECONDS", 0)
    comfy = settings.native_comfy_url
    assert client.get(f"{comfy}/_studio/api/status", headers=auth).json() == {"ready": False}
    assert client.get(f"{comfy}/system_stats", headers=auth).status_code == 503
    assert client.post(f"{comfy}/prompt", headers=read_only, json={"prompt": RENDERED}).json()["error"] == "NATIVE_NOT_READY"
    assert started == []
    response = client.post(f"{comfy}/prompt", headers=auth, json={"prompt": RENDERED})
    assert response.status_code == 503
    assert response.json()["error"] == "NATIVE_STARTING"
    assert response.headers["Retry-After"] == "5"
    assert started[0][1] == {
        "account_id": "account", "spec": {"key": values["gpu"]},
        "retain_idle_minutes": 9, "max_runtime_minutes": 65,
        "model_storage_mode": "local", "cache_gb": 33, "cache_max_age_minutes": 115,
    }


def test_workflow_is_submitted_once_when_cold_start_becomes_ready(server, monkeypatch):
    client, repo, lease = server
    repo.update("leases", lease["id"], {"native_ready": False}, "owner")
    _, auth = make_key(client, "cold start", auto_start=True)
    calls = []

    def request_start(repo, owner, key):
        calls.append("start")
        repo.update("leases", lease["id"], {"native_ready": True}, owner)
        return {"id": "started", "state": "completed"}

    async def upstream(req):
        calls.append(req.url.path)
        assert json.loads(await req.aread()) == {"prompt": RENDERED}
        return httpx.Response(200, json={"prompt_id": "cold-start-prompt"})

    monkeypatch.setattr(native_api, "request_start", request_start)
    original = httpx.AsyncClient
    monkeypatch.setattr(native_proxy.httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(upstream), **kwargs))
    response = client.post(f"{settings.native_comfy_url}/prompt", headers=auth, json={"prompt": RENDERED})
    assert response.json() == {"prompt_id": "cold-start-prompt"}
    assert calls == ["start", "/prompt"]


def test_concurrent_auto_start_joins_pending_manual_start(server, monkeypatch):
    client, repo, lease = server
    repo.update("leases", lease["id"], {"native_ready": False, "state": "deleted"}, "owner")
    original = repo.create_job("owner", {"kind": "native-comfy"}, "manual-start")[0]
    repo.update("jobs", original["id"], {"snapshot": {"native_comfy": True}}, "owner")
    monkeypatch.setattr(native.GpuConfig, "from_env", lambda: type("Config", (), {"real_enabled": True})())
    monkeypatch.setattr(native, "secret_file", lambda name: "configured")
    monkeypatch.setenv("NATIVE_COMFY_COMMIT", "a" * 40)

    async def submit_twice():
        return await asyncio.gather(*[
            asyncio.to_thread(native.admit, repo, "owner", {"account_id": "different", "spec": {}},
                              f"auto-{i}", api_key_id="api")
            for i in range(2)
        ], return_exceptions=True)

    outcomes = asyncio.run(submit_twice())
    assert any(isinstance(result, dict) for result in outcomes)
    for result in outcomes:
        if isinstance(result, APIError):
            assert result.code == "SCHEDULER_BUSY"
        else:
            assert result["id"] == original["id"]
    assert len(repo.list("jobs", "owner")) == 1


@pytest.mark.parametrize("mode", ["browser", "api"])
@pytest.mark.parametrize("download_ok", [True, False])
def test_local_proxy_prepares_before_real_prompt_and_keeps_client_contract(server, monkeypatch, mode, download_ok):
    from app.security import sign_session

    client, repo, lease = server
    repo.update("leases", lease["id"], {"native_model_storage": {"model_storage_mode": "local"}}, "owner")
    monkeypatch.setattr(native, "model_channel", lambda lease_id: "trusted-runtime-channel")
    _, key_auth = make_key(client, "external")
    if mode == "browser":
        monkeypatch.setattr(settings, "session_secret", "native-api-test-session")
        client.cookies.set("studio_native", sign_session({
            "uid": "owner", "aud": settings.native_comfy_url, "purpose": "native-session", "exp": 2**31,
        }), domain="comfy.example.test")
        auth = {"Origin": settings.native_comfy_url}
    else:
        auth = key_auth

    requested = {"prompt": RENDERED, "client_id": "browser-ws"}
    expected = requested if mode == "browser" else {"prompt": RENDERED}
    received = []
    model_result = {"category": "checkpoints", "name": "x.safetensors",
                    "state": "ready" if download_ok else "failed"}
    if not download_ok:
        model_result["error"] = "missing model"
    success = {"prompt_id": "real-prompt", "number": 12, "node_errors": {}}

    class JsonStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield json.dumps(success).encode()

    async def upstream(req):
        path = req.url.path
        received.append(path)
        assert "authorization" not in req.headers
        if path == "/studio_native/models/prepare":
            assert req.headers["X-Studio-Model-Channel"] == "trusted-runtime-channel"
            assert json.loads(await req.aread())["workflow"] == expected
            assert native.active_writes(repo.get("leases", lease["id"], "owner"), native.time.time())
            return httpx.Response(202, json={"id": "preparation"})
        if path == "/studio_native/models/prepare/preparation":
            return httpx.Response(200, json={"state": "ready" if download_ok else "failed", "models": [model_result]})
        if path == "/history/real-prompt":
            return httpx.Response(200, json={"real-prompt": {"outputs": {}}})
        assert path in {"/prompt", "/api/prompt"} and download_ok
        assert req.headers["X-Studio-Model-Prepared"] == "preparation"
        assert json.loads(await req.aread()) == expected
        if mode == "api":
            return httpx.Response(200, json=success)
        return httpx.Response(200, stream=JsonStream(), headers={"Content-Type": "application/json"})

    original = httpx.AsyncClient
    monkeypatch.setattr(native_proxy.httpx, "AsyncClient", lambda **kw: original(transport=httpx.MockTransport(upstream), **kw))
    result = client.post(f"{settings.native_comfy_url}/api/prompt",
                         headers={**auth, "X-Studio-Model-Prepared": "spoof"}, json=requested)
    assert received[:2] == ["/studio_native/models/prepare", "/studio_native/models/prepare/preparation"]
    if download_ok:
        assert result.status_code == 200 and result.json() == success
        if mode == "api":
            assert client.get(f"{settings.native_comfy_url}/history/real-prompt", headers=auth).json() == {
                "real-prompt": {"outputs": {}},
            }
    else:
        assert result.status_code == 422 and result.json()["models"] == [model_result]
        assert len(received) == 2
    assert not repo.get("leases", lease["id"], "owner").get("native_writes")
    assert client.get(f"{settings.native_comfy_url}/api/studio_native/models/prepare/preparation", headers=auth).status_code == 404


def test_manual_preparation_uses_same_runtime_and_does_not_submit_prompt(server, monkeypatch):
    client, repo, lease = server
    repo.update("leases", lease["id"], {"native_model_storage": {"model_storage_mode": "local"}}, "owner")
    monkeypatch.setattr(native, "model_channel", lambda lease_id: "trusted-runtime-channel")
    ident = "2" * 32
    declared = [{"category": "loras", "name": "styles/custom.safetensors"}]
    workflow = {"nodes": []}

    async def upstream(req):
        assert req.headers["X-Studio-Model-Channel"] == "trusted-runtime-channel"
        if req.method == "POST":
            assert req.url.path == "/studio_native/models/prepare"
            submitted = json.loads(await req.aread())
            assert submitted["workflow"] == {"workflow": workflow}
            assert submitted["dependencies"] == declared and submitted["visual"] is True
            return httpx.Response(202, json={"id": ident})
        assert req.url.path == "/studio_native/models/prepare/" + ident
        return httpx.Response(200, json={"state": "preparing", "models": [{"bytes": 12, "total": 28}]})

    original = httpx.AsyncClient
    monkeypatch.setattr(native_proxy.httpx, "AsyncClient", lambda **kw: original(transport=httpx.MockTransport(upstream), **kw))
    route = "/api/gpu/native/model-preparation"
    result = client.post(route, headers=ADMIN, json={"workflow": workflow, "dependencies": declared})
    assert result.status_code == 202 and result.json()["data"]["id"] == ident
    assert client.get(route + "/" + ident, headers=ADMIN).json()["data"]["models"][0]["bytes"] == 12
    assert client.get(route + "/" + ident).status_code == 401


def test_drive_workflows_routes_and_ref_preparation_control_plane(server, monkeypatch):
    client, repo, lease = server
    repo.update("leases", lease["id"], {"native_model_storage": {"model_storage_mode": "local"}}, "owner")
    monkeypatch.setattr(native, "model_channel", lambda lease_id: "trusted-runtime-channel")
    ident = "3" * 32
    observed = []

    async def upstream(req):
        assert req.headers["X-Studio-Model-Channel"] == "trusted-runtime-channel"
        observed.append((req.method, req.url.path, req.url.query.decode()))
        if req.method == "GET" and req.url.path == "/studio_native/models/workflows":
            return httpx.Response(200, json={
                "workflows": [{"source": "comfy", "path": "anima/base.json", "size": 512, "modified_at": 1700000000.0}]
            })
        if req.method == "POST" and req.url.path == "/studio_native/models/workflows":
            body = json.loads(await req.aread())
            if not body.get("overwrite"):
                return httpx.Response(409, json={
                    "error": {"code": "NATIVE_WORKFLOW_EXISTS", "message": "同名工作流已存在"}
                })
            return httpx.Response(201, json={
                "workflow": {"source": "warmup", "path": "tavern.json", "size": 256, "modified_at": 1700000005.0}
            })
        if req.method == "POST" and req.url.path == "/studio_native/models/workflows/delete":
            body = json.loads(await req.aread())
            assert body == {"source": "warmup", "path": "tavern.json"}
            return httpx.Response(200, json={"deleted": True, "source": "warmup", "path": "tavern.json"})
        if req.method == "POST" and req.url.path == "/studio_native/models/prepare":
            body = json.loads(await req.aread())
            assert body["workflow_ref"] == {"source": "comfy", "path": "anima/base.json"}
            assert "workflow" not in body and body["visual"] is True
            return httpx.Response(202, json={"id": ident})
        raise AssertionError(f"unexpected request {req.method} {req.url}")

    original = httpx.AsyncClient
    monkeypatch.setattr(native_proxy.httpx, "AsyncClient", lambda **kw: original(transport=httpx.MockTransport(upstream), **kw))

    listed = client.get("/api/gpu/native/workflows?refresh=true", headers=ADMIN)
    assert listed.status_code == 200
    assert listed.json()["data"]["workflows"][0]["path"] == "anima/base.json"
    assert ("GET", "/studio_native/models/workflows", "refresh=1") in observed

    conflict = client.post("/api/gpu/native/workflows", headers=ADMIN, json={
        "name": "tavern.json",
        "workflow": {"1": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}}},
    })
    assert conflict.status_code == 409 and conflict.json()["error"] == "NATIVE_WORKFLOW_EXISTS"

    saved = client.post("/api/gpu/native/workflows", headers=ADMIN, json={
        "name": "tavern.json",
        "workflow": {"1": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}}},
        "overwrite": True,
    })
    assert saved.status_code == 201 and saved.json()["data"]["workflow"]["source"] == "warmup"

    prep = client.post("/api/gpu/native/model-preparation", headers=ADMIN, json={
        "workflow_ref": {"source": "comfy", "path": "anima/base.json"},
    })
    assert prep.status_code == 202 and prep.json()["data"]["id"] == ident

    # Reject ambiguous payload containing both workflow and workflow_ref.
    both = client.post("/api/gpu/native/model-preparation", headers=ADMIN, json={
        "workflow": {"nodes": []},
        "workflow_ref": {"source": "comfy", "path": "anima/base.json"},
    })
    assert both.status_code == 422

    # Reject deleting comfy workflows from Studio API.
    forbid = client.delete("/api/gpu/native/workflows?source=comfy&path=anima/base.json", headers=ADMIN)
    assert forbid.status_code == 422

    deleted = client.delete("/api/gpu/native/workflows?source=warmup&path=tavern.json", headers=ADMIN)
    assert deleted.status_code == 200 and deleted.json()["data"]["deleted"] is True
    assert not repo.get("leases", lease["id"], "owner").get("native_writes")
