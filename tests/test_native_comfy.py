"""Credential-free native data-plane and lifecycle regression tests."""
import asyncio
import time
from contextlib import asynccontextmanager, contextmanager

import httpx
import pytest
from app import native_proxy
from app.config import settings
from app.gpu import native
from app.gpu.native_runtime import EXTENSION, model_paths, prepare_paths
from app.native_proxy import NativeHostMiddleware
from app.security import sign_session
from fastapi import FastAPI
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect


class BytesStream(httpx.AsyncByteStream):
    def __init__(self, content):
        self.content = content

    async def __aiter__(self):
        # Multiple chunks catch accidental buffering and preserve encoded media bytes.
        for offset in range(0, len(self.content), 4):
            yield self.content[offset:offset + 4]


class Repo:
    def __init__(self):
        self.store = self
        self.lease = {"id": "lease", "owner_id": "owner", "instance_id": "instance", "native_comfy": True,
                      "native_ready": True, "state": "idle", "native_last_busy": 100}
        self.instance = {}
        self.job = {}

    @contextmanager
    def dispatch_lock(self):
        yield True

    def get(self, kind, ident, owner):
        if kind == "jobs":
            return self.job
        return self.lease if kind == "leases" else self.instance

    def update(self, kind, ident, patch, owner):
        self.lease.update(patch)
        return self.lease

    def list(self, kind, owner):
        return [self.lease] if kind == "leases" else []


@pytest.fixture
def proxy(monkeypatch):
    monkeypatch.setattr(settings, "native_comfy_url", "https://comfy.example.test")
    monkeypatch.setattr(settings, "site_url", "https://studio.example.test")
    monkeypatch.setattr(settings, "studio_api_token", "api-test-token")
    monkeypatch.setattr(settings, "session_secret", "test-session-secret")
    monkeypatch.setattr(native, "backend", lambda: ("https://internal.example.test", {"CF-Access-Client-Id": "service-id", "CF-Access-Client-Secret": "internal-secret"}))
    app = FastAPI()
    app.state.repo = Repo()
    app.add_middleware(NativeHostMiddleware, studio_app=app)
    return app


@pytest.mark.parametrize("path", ["/", "/assets/app.js", "/extensions/example/main.js", "/object_info", "/system_stats", "/history", "/history/123", "/view?filename=video.mp4&type=output", "/queue", "/api/userdata/workflows%2Ftest.json", "/unknown-extension/route"])
def test_complete_http_proxy_keeps_raw_paths_and_streamed_bytes(proxy, monkeypatch, path):
    seen = []

    async def handler(req):
        seen.append(req)
        assert req.headers["origin"] == "https://internal.example.test"
        assert req.headers["host"] == "internal.example.test"
        assert "authorization" not in req.headers and "cookie" not in req.headers
        assert req.headers["CF-Access-Client-Secret"] == "internal-secret"
        return httpx.Response(206, stream=BytesStream(b"raw-media"), headers={"Content-Type": "video/mp4", "Content-Range": "bytes 0-8/99", "ETag": '"v1"', "Set-Cookie": "CF_Authorization=private"})

    original = httpx.AsyncClient
    monkeypatch.setattr(native_proxy.httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))
    with TestClient(proxy, base_url=settings.native_comfy_url) as c:
        response = c.get(path, headers={"Authorization": "Bearer api-test-token", "Cookie": "private=browser"})
    assert response.status_code == 206
    assert response.content == b"raw-media"
    assert response.headers["content-range"] == "bytes 0-8/99"
    assert "set-cookie" not in response.headers
    assert seen[0].url.raw_path == path.encode()


@pytest.mark.parametrize("path,method", [("/prompt", "POST"), ("/interrupt", "POST"), ("/upload/image", "POST"), ("/upload/mask", "POST"), ("/api/userdata/workflows%2Ftest.json", "PUT"), ("/extension/route", "DELETE")])
def test_upload_and_arbitrary_methods(proxy, monkeypatch, path, method):
    content = b"binary\x00payload" * 200000

    async def handler(req):
        assert await req.aread() == content
        assert req.headers["content-type"] == "application/octet-stream"
        return httpx.Response(200, stream=BytesStream(b'{"ok":true}'), headers={"Content-Type": "application/json"})
    original = httpx.AsyncClient
    monkeypatch.setattr(native_proxy.httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))
    with TestClient(proxy, base_url=settings.native_comfy_url) as c:
        r = c.request(method, path, content=content, headers={"Authorization": "Bearer api-test-token", "Content-Type": "application/octet-stream"})
    assert r.status_code == 200
    assert proxy.state.repo.lease.get("native_last_write")


@pytest.mark.parametrize('method,body,persistent,network,expected_calls,expected_status', [
    ('GET', b'', False, False, 2, 200), ('HEAD', b'', False, False, 2, 200),
    ('GET', b'', True, False, 3, 502), ('POST', b'write', False, False, 1, 502),
    ('GET', b'body', False, False, 1, 502), ('GET', b'', False, True, 2, 200),
    ('GET', b'', True, True, 3, 502),
])
def test_transient_upstream_retry_only_replays_empty_safe_reads(
    proxy, monkeypatch, method, body, persistent, network, expected_calls, expected_status,
):
    calls, responses = [], []

    async def handler(req):
        calls.append(await req.aread())
        if len(calls) == 1 or persistent:
            if network:
                raise httpx.ConnectError('temporary connection failure', request=req)
            status = 502
        else:
            status = 200
        response = httpx.Response(status, stream=BytesStream(b'upstream'))
        responses.append(response)
        return response

    original = httpx.AsyncClient
    monkeypatch.setattr(native_proxy.httpx, 'AsyncClient', lambda **kwargs: original(
        transport=httpx.MockTransport(handler), **kwargs))
    with TestClient(proxy, base_url=settings.native_comfy_url) as client:
        response = client.request(method, '/assets/module.js', content=body,
                                  headers={'Authorization': 'Bearer api-test-token'})
        assert response.status_code == expected_status
        assert len(calls) == expected_calls
        assert all(content == body for content in calls)
        assert all(item.is_closed for item in responses)


def test_browser_login_cookie_writes_and_csrf(proxy, monkeypatch):
    now = int(time.time())
    ticket = sign_session({"uid": "owner", "aud": settings.native_comfy_url, "purpose": "native-handoff", "iat": now, "exp": now + 60})
    original = httpx.AsyncClient
    monkeypatch.setattr(native_proxy.httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(lambda req: httpx.Response(200, stream=BytesStream(b'{"ok":true}'), headers={"Content-Type": "application/json"})), **kwargs))
    with TestClient(proxy, base_url=settings.native_comfy_url) as c:
        assert c.get("/prompt").status_code == 401
        r = c.post("/_studio/session", data={"ticket": ticket}, headers={"Origin": settings.site_url}, follow_redirects=False)
        assert r.status_code == 303 and r.headers["location"] == "/"
        assert "HttpOnly" in r.headers["set-cookie"] and "Secure" in r.headers["set-cookie"]
        assert "internal-secret" not in r.text
        assert c.post("/prompt", json={}, headers={"Origin": settings.native_comfy_url}).status_code == 200
        assert c.post("/prompt", json={}, headers={"Origin": "https://evil.example"}).status_code == 403
        assert c.post("/_studio/session", data={"ticket": ticket}, headers={"Origin": "https://evil.example"}).status_code == 403
        assert c.post("/_studio/session", data={"ticket": sign_session({"uid": "owner", "iat": now, "exp": now + 60})}, headers={"Origin": settings.site_url}).status_code == 401


def test_stop_and_deadline_reject_new_work(proxy):
    with TestClient(proxy, base_url=settings.native_comfy_url) as c:
        proxy.state.repo.instance["manual_stop_requested"] = True
        assert c.post("/prompt", json={}, headers={"Authorization": "Bearer api-test-token"}).status_code == 409
        proxy.state.repo.instance.clear()
        proxy.state.repo.lease["deadline_epoch"] = time.time() - 1
        assert c.post("/prompt", json={}, headers={"Authorization": "Bearer api-test-token"}).status_code == 409


def test_ws_binary_text_and_cancellation(proxy, monkeypatch):
    closed = []

    class Upstream:
        subprotocol = None

        def __init__(self):
            self.messages = asyncio.Queue()

        async def send(self, message):
            await self.messages.put(message)

        def __aiter__(self):
            return self

        async def __anext__(self):
            return await self.messages.get()

    @asynccontextmanager
    async def connect(url, **kwargs):
        assert url == "wss://internal.example.test/ws?clientId=abc"
        assert kwargs["origin"] == "https://internal.example.test"
        assert not any(k.lower().startswith("sec-websocket") for k, v in kwargs["additional_headers"])
        try:
            yield Upstream()
        finally:
            closed.append(True)
    monkeypatch.setattr(native_proxy, "connect", connect)
    with TestClient(proxy, base_url=settings.native_comfy_url) as c:
        with c.websocket_connect("wss://comfy.example.test/ws?clientId=abc", headers={"Authorization": "Bearer api-test-token", "Origin": settings.native_comfy_url}) as ws:
            ws.send_text('{"type":"status"}')
            assert ws.receive_text() == '{"type":"status"}'
            ws.send_bytes(b"\x00\x01preview")
            assert ws.receive_bytes() == b"\x00\x01preview"
        with pytest.raises(WebSocketDisconnect):
            with c.websocket_connect("wss://comfy.example.test/ws"):
                pass
    assert closed


@pytest.mark.parametrize("queue,persistence,result", [({"queue_running": [["video"]], "queue_pending": []}, False, True), ({"queue_running": [], "queue_pending": [["image"]]}, False, True), ({"queue_running": [], "queue_pending": []}, True, True), ({"queue_running": [], "queue_pending": []}, False, False), ({}, False, True)])
def test_activity_probes_queue_and_persistence(proxy, queue, persistence, result):
    def handler(req):
        return httpx.Response(200, json=queue if req.url.path == "/queue" else {"busy": persistence})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert native.busy(client) is result


def test_unknown_activity_never_releases(proxy, monkeypatch):
    def handler(req):
        raise httpx.ConnectError("offline", request=req)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert native.busy(client)
    monkeypatch.setattr(native, "busy", lambda: True)
    assert native.protect_lease(proxy.state.repo, proxy.state.repo.lease, clock=lambda: 10000)
    assert proxy.state.repo.lease["native_last_busy"] == 10000
    monkeypatch.setattr(native, "busy", lambda: False)
    assert native.protect_lease(proxy.state.repo, proxy.state.repo.lease, clock=lambda: 10001)
    assert native.protect_lease(proxy.state.repo, proxy.state.repo.lease, clock=lambda: 10000 + settings.native_comfy_idle_minutes * 60 - 1)
    assert not native.protect_lease(proxy.state.repo, proxy.state.repo.lease, clock=lambda: 10000 + settings.native_comfy_idle_minutes * 60)


def test_ready_requires_protected_tunnel_and_native_extension(proxy):
    def handler(req):
        if "CF-Access-Client-Secret" not in req.headers:
            return httpx.Response(403)
        return httpx.Response(200, json={"system": {}} if req.url.path == "/system_stats" else {"busy": False})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert native.ready(client)
    with httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"system": {}}))) as client:
        assert not native.ready(client)


def test_persistent_directories_restore_after_checkout_recreation(tmp_path):
    mount = tmp_path / "drive"
    mount.mkdir()
    for name in ("first", "after-restart"):
        comfy = tmp_path / name
        comfy.mkdir()
        (comfy / "models").mkdir()
        (comfy / "models" / "example.txt").write_text("official placeholder")
        data = prepare_paths(comfy, mount)
        if name == "first":
            (data / "user" / "workflow.json").write_text('{"nodes": []}')
            (comfy / "models" / "checkpoints" / "model.safetensors").write_bytes(b"model")
        else:
            assert (data / "user" / "workflow.json").read_text() == '{"nodes": []}'
            assert (comfy / "models" / "checkpoints" / "model.safetensors").read_bytes() == b"model"
        assert (comfy / "models.checkout" / "example.txt").is_file()
    paths = model_paths(mount)["studio_drive"]
    assert "models/unet" in paths["diffusion_models"]
    assert "models/clip" in paths["text_encoders"]
    compile(EXTENSION, "native_extension", "exec")


@pytest.mark.parametrize("timeouts", [{}, {"retain_idle_minutes": 7, "max_runtime_minutes": 20},
                                     {"retain_idle_minutes": 90, "max_runtime_minutes": 120}])
def test_native_admission_idempotent_across_restart_and_gpu_switch(proxy, monkeypatch, timeouts):
    from app.common import APIError
    from app.db import Store
    from app.gpu.compute import GpuConfig
    from app.gpu.repository import Repository

    store = Store("sqlite:///:memory:")
    repo = Repository(store)
    repo.create("accounts", "owner", {"account_id": "account", "enabled": True, "allow_allocation": True, "connected": True}, record_id="account")
    monkeypatch.setattr(native.GpuConfig, "from_env", lambda: GpuConfig(real_enabled=True, unit_rate=0.01))
    monkeypatch.setattr(native, "secret_file", lambda name: "test-secret")
    monkeypatch.setenv("NATIVE_COMFY_COMMIT", "a" * 40)
    body = {"account_id": "account", "spec": {"key": {"variant": "VARIANT_GPU", "accelerator": "T4", "shape": "SHAPE_STANDARD"}}, **timeouts}
    job = native.admit(repo, "owner", body, "native-start")
    assert job["execution_policy"]["retain_idle_minutes"] == timeouts.get("retain_idle_minutes", settings.native_comfy_idle_minutes)
    assert job["max_runtime_minutes"] == job["execution_policy"]["max_runtime_minutes"] == timeouts.get("max_runtime_minutes", settings.native_comfy_max_runtime_minutes)
    assert job["estimated_units"] == job["max_runtime_minutes"] * 0.01
    assert 0 <= job["native_session_deadline_epoch"] - time.time() <= job["max_runtime_minutes"] * 60
    assert native.admit(Repository(store), "owner", body, "native-start")["id"] == job["id"]
    assert native.admit(Repository(store), "owner", body, "another-browser")["id"] == job["id"]
    assert len(repo.list("jobs", "owner")) == 1
    with pytest.raises(APIError) as error:
        native.admit(repo, "owner", {**body, "spec": {"key": {"accelerator": "A100"}}}, "switch-gpu")
    assert error.value.status == 409
    with pytest.raises(APIError) as error:
        native.admit(repo, "owner", {**body, "retain_idle_minutes": 3}, "change-timeouts")
    assert error.value.status == 409
    with pytest.raises(APIError) as error:
        native.admit(repo, "owner", {**body, "max_runtime_minutes": 5}, "native-start")
    assert error.value.code == "IDEMPOTENCY_CONFLICT"
    assert "secret" not in str(job)


@pytest.mark.parametrize("field,value", [
    ("retain_idle_minutes", 0), ("retain_idle_minutes", -1),
    ("max_runtime_minutes", -1), ("max_runtime_minutes", 0),
    ("retain_idle_minutes", True), ("max_runtime_minutes", 1.5),
    ("retain_idle_minutes", "5"), ("max_runtime_minutes", None),
])
def test_native_admission_rejects_invalid_timeouts(monkeypatch, field, value):
    from app.common import APIError
    from app.gpu.compute import GpuConfig

    monkeypatch.setattr(native.GpuConfig, "from_env", lambda: GpuConfig(max_runtime_minutes=30))
    with pytest.raises(APIError) as error:
        native.admit(Repo(), "owner", {field: value}, "invalid-timeout")
    assert error.value.code == "NATIVE_INPUT"
    assert error.value.status == 422


@pytest.mark.parametrize("idle_minutes,deadline,now,protected", [
    (2, 10000, 219, True), (2, 10000, 220, False),
    (20, 10000, 1000, True), (20, 10000, 1300, False),
    (20, 500, 500, False),
    (90, 10000, 5499, True), (90, 10000, 5500, False),
])
def test_native_reclamation_uses_session_timeouts(proxy, monkeypatch, idle_minutes, deadline, now, protected):
    repo = proxy.state.repo
    repo.lease.update(owner_job_id="job", deadline_epoch=deadline)
    repo.job = {"execution_policy": {"retain_idle_minutes": idle_minutes}}
    monkeypatch.setattr(native, "busy", lambda: False)
    assert native.protect_lease(repo, repo.lease, clock=lambda: now) is protected
    assert bool(repo.lease.get("native_draining")) is (not protected)


def test_native_idle_timeout_survives_studio_handoff(proxy, monkeypatch):
    repo = proxy.state.repo
    repo.lease.update(owner_job_id="studio-job", native_idle_minutes=2, deadline_epoch=10000)
    repo.job = {"execution_policy": {"retain_idle_minutes": 30}}
    monkeypatch.setattr(native, "busy", lambda: False)
    assert not native.protect_lease(repo, repo.lease, clock=lambda: 220)


def test_native_status_exposes_timeout_defaults_and_limits(proxy, monkeypatch):
    from app.native_routes import router

    assert settings.native_comfy_idle_minutes == 120
    assert settings.native_comfy_max_runtime_minutes == 720
    proxy.include_router(router)
    proxy.state.repo.lease["native_ready"] = False
    monkeypatch.setattr(settings, "native_comfy_idle_minutes", 8)
    monkeypatch.setattr(settings, "native_comfy_max_runtime_minutes", 60)
    with TestClient(proxy) as client:
        response = client.get("/api/gpu/native/status", headers={"Authorization": "Bearer api-test-token"})
    assert response.status_code == 200
    assert response.json()["data"]["timeouts"] == {
        "defaults": {"retain_idle_minutes": 8, "max_runtime_minutes": 60},
        "limits": {"retain_idle_minutes": None, "max_runtime_minutes": None},
    }


def test_native_shared_studio_does_not_spawn_or_kill_comfy(monkeypatch, tmp_path):
    import json

    from app.gpu import runtime_worker as worker

    root = tmp_path / "studio-job"
    root.mkdir()
    comfy = tmp_path / "native" / "ComfyUI"
    comfy.mkdir(parents=True)
    marker = tmp_path / "native" / "state.json"
    marker.write_text(json.dumps({"commit": "a" * 40, "comfy": str(comfy)}))
    monkeypatch.setattr(worker, "NATIVE_SERVICE_MARKER", marker)
    monkeypatch.setattr(worker, "_verify_checkout", lambda *args: None)
    monkeypatch.setattr(worker, "_restore_weights", lambda *args, **kwargs: [])
    monkeypatch.setattr(worker, "_warm_put", lambda *args: pytest.fail("native process must remain in its own directory"))
    monkeypatch.setattr(worker.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("must use the resident ComfyUI process"))
    calls = []

    def api(url, method="GET", body=None, timeout=30):
        calls.append((url, method))
        if url.endswith("/prompt"):
            return {"prompt_id": "native-and-studio"}
        if "/history/" in url:
            return {"native-and-studio": {"status": {"completed": True}, "outputs": {}}}
        return {"system": {}}
    monkeypatch.setattr(worker, "_http_json", api)
    manifest = {"comfy_commit": "a" * 40, "node_allowlist": ["TestNode"], "weight_files": [{"repo_id": "test/model", "filename": "model.safetensors", "target": "checkpoints/model.safetensors", "revision": "b" * 40, "sha256": "c" * 64}]}
    assert worker._run_comfy(root, {"snapshot": {}}, manifest, {"1": {"class_type": "TestNode", "inputs": {}}}, worker._Stages(root)) == []
    assert any(url.endswith("/prompt") for url, method in calls)
    assert marker.exists()


def test_shutdown_fence_and_inflight_uploads(proxy, monkeypatch):
    from app.common import APIError
    repo = proxy.state.repo
    monkeypatch.setattr(native, "busy", lambda: False)
    repo.lease["native_http_heartbeat"] = 9999
    assert native.protect_lease(repo, repo.lease, clock=lambda: 10000)
    repo.lease["native_http_heartbeat"] = 0
    assert not native.protect_lease(repo, repo.lease, clock=lambda: 10000)
    assert repo.lease["native_draining"]
    with pytest.raises(APIError):
        native.begin_write(repo, repo.lease)


def test_runtime_probe_and_watchdog_compile():
    from app.gpu.native_runtime import EXTENSION, WATCHDOG
    probe = EXTENSION[EXTENSION.index("def open_writes():"):EXTENSION.index("@PromptServer")]
    compile("import os, json\nfrom pathlib import Path\nfrom urllib.request import Request, urlopen\n" + probe, "probe.py", "exec")
    compile(WATCHDOG, "watchdog.py", "exec")


@pytest.mark.parametrize("fail_start", [False, True])
@pytest.mark.parametrize("runtime_minutes", [10, 120])
def test_native_boot_job_retains_one_lease_and_watchdog_waits_for_video(monkeypatch, fail_start, runtime_minutes):
    from app.db import Store
    from app.gpu.compute import GpuConfig, MockColabProvider, MockGPUExecutor
    from app.gpu.repository import Repository
    from app.gpu.scheduler import Scheduler

    store = Store("sqlite:///:memory:")
    repo = Repository(store)
    provider = MockColabProvider()
    executor = MockGPUExecutor(fail_on_start=fail_start)
    now = [1000.0]
    monkeypatch.setattr(native, "ready", lambda: True)
    monkeypatch.setattr(native, "busy", lambda: False)
    spec = provider.list_specs()[0]
    job, _ = repo.create_job("owner", {"project_id": "native", "execution_policy": {"provider": "mock"}}, "boot-native")
    job = repo.update("jobs", job["id"], {"provider": "mock", "project_id": "native", "lifecycle": "retain",
        "execution_policy": {"provider": "mock", "lifecycle": "retain", "retain_idle_minutes": 90,
                             "max_runtime_minutes": runtime_minutes},
        "native_session_deadline_epoch": 1000 + runtime_minutes * 60,
        "snapshot": {"native_comfy": True, "manifest": {}, "bound_graph": {}}, "requested_spec": spec}, "owner")
    config = GpuConfig(max_runtime_minutes=30, unit_rate=0.01)

    def scheduler():
        return Scheduler(store, config=config, clock=lambda: now[0], provider_factory=lambda job: provider, executor_factory=lambda job: executor)
    for _ in range(30):
        scheduler().tick()
        job = repo.get("jobs", job["id"], "owner")
        if job["state"] in native.TERMINAL:
            break
    if fail_start:
        assert job["state"] == "failed", job
        assert repo.list("leases", "owner")[0]["state"] == "deleted"
        assert repo.list("instances", "owner")[0]["state"] == "deleted"
        return
    assert job["state"] == "completed", job
    leases = repo.list("leases", "owner")
    assert len(leases) == 1
    assert leases[0]["native_ready"] and leases[0]["state"] == "idle"
    assert leases[0]["native_idle_minutes"] == 90
    assert job["max_runtime_minutes"] == runtime_minutes
    assert leases[0]["deadline_epoch"] == 1000 + runtime_minutes * 60
    assert repo.list("instances", "owner")[0]["hard_deadline_at"] == 1000 + runtime_minutes * 60
    assert repo.list("instances", "owner")[0]["retain_idle_minutes"] == 90
    assert len(repo.list("instances", "owner")) == 1
    now[0] += max(6000, runtime_minutes * 60 + 1000)  # Busy work survives both deadlines.
    monkeypatch.setattr(native, "busy", lambda: True)
    scheduler().watchdog()
    assert repo.get("leases", leases[0]["id"], "owner")["state"] != "deleted"
    monkeypatch.setattr(native, "busy", lambda: False)
    now[0] += 1000
    scheduler().watchdog()
    assert repo.get("leases", leases[0]["id"], "owner")["state"] == "deleted"


@pytest.mark.parametrize("processing,uploads,errors,current_errors,expected", [
    (True, 0, 0, 0, True), (False, 1, 0, 0, True), (False, 0, 0, 0, False),
    (False, 0, 2, 0, False), (False, 0, 2, 1, True),
])
def test_native_probe_includes_manager_download_queue(processing, uploads, errors, current_errors, expected):
    import io
    import json

    from app.gpu.native_runtime import EXTENSION

    namespace = {}
    probe = EXTENSION[EXTENSION.index("def open_writes():"):EXTENSION.index("@PromptServer")]
    exec("import os, json\nfrom pathlib import Path\nfrom urllib.request import Request, urlopen\n" + probe, namespace)

    def local_api(request, **kwargs):
        url = request if isinstance(request, str) else request.full_url
        if url.endswith("/vfs/stats"):
            data = {"inUse": 1, "diskCache": {"uploadsQueued": uploads, "erroredFiles": current_errors}}
        elif url.endswith("/core/stats"):
            data = {"errors": errors}
        elif url.endswith("/manager/queue/status"):
            data = {"is_processing": processing, "total_count": 1 if processing else 0, "done_count": 0}
        else:
            pytest.fail(url)
        return io.BytesIO(json.dumps(data).encode())
    namespace["urlopen"] = local_api
    namespace["open_writes"] = lambda: False
    assert namespace["probe"]()["busy"] is expected


@pytest.mark.parametrize('target,expected', [
    ('native-comfy/user/comfyui_8188.log', False),
    ('native-comfy/user/comfyui.log', False),
    ('native-comfy/output/result.png', True),
    ('native-comfy/user/default/workflows/current.json', True),
])
def test_manager_diagnostic_log_does_not_hold_gpu_but_user_writes_do(tmp_path, target, expected):
    from pathlib import Path

    from app.gpu.native_runtime import EXTENSION

    proc = tmp_path / 'proc'
    (proc / '123/fd').mkdir(parents=True)
    (proc / '123/fdinfo').mkdir()
    (proc / '123/fd/4').symlink_to('/content/ai-studio/native/drive/' + target)
    (proc / '123/fdinfo/4').write_text('flags:\t0100001\n')
    source = EXTENSION[EXTENSION.index('def open_writes():'):EXTENSION.index('def manager_busy():')]
    namespace = {'os': __import__('os'), 'Path': lambda value: proc if value == '/proc' else Path(value)}
    exec(source, namespace)
    assert namespace['open_writes']() is expected


def test_rclone_install_checks_checksum_and_publishes_atomically(monkeypatch, tmp_path):
    import hashlib
    import io
    import zipfile

    from app.gpu import native_runtime as runtime
    from app.gpu import runtime_worker as worker

    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr(f"rclone-{runtime.RCLONE_VERSION}-linux-amd64/rclone", b"test-rclone-executable")
    content = data.getvalue()
    monkeypatch.setattr(runtime, "urlopen", lambda *args, **kwargs: io.BytesIO(content))
    with pytest.raises(RuntimeError, match="checksum"):
        runtime.install_rclone(tmp_path, worker)
    assert not (tmp_path / "rclone").exists()
    assert not (tmp_path / "rclone.zip").exists()
    monkeypatch.setattr(runtime, "RCLONE_SHA256", hashlib.sha256(content).hexdigest())
    binary = runtime.install_rclone(tmp_path, worker)
    assert binary.read_bytes() == b"test-rclone-executable"
    assert binary.stat().st_mode & 0o777 == 0o700


def _mount_fixture(monkeypatch, tmp_path, *, exit_code=None, mounted=False, timeout=False, slow_stop=False):
    import subprocess

    from app.gpu import native_runtime as runtime

    secret = {'access_token': 'fake-access', 'refresh_token': 'fake-refresh',
              'client_id': 'fake-client', 'client_secret': 'fake-secret',
              'owner_folder_id': 'fake-folder', 'cache_gb': 20}
    seen = []

    class Process:
        stopped = False
        killed = False

        def poll(self):
            return exit_code if not self.stopped else 0

        def terminate(self):
            self.stopped = True

        def wait(self, timeout):
            if slow_stop and not self.killed:
                raise subprocess.TimeoutExpired('fake-rclone', timeout)
            return 0

        def kill(self):
            self.killed = True

    process = Process()

    def launch(args, **kwargs):
        seen.append(args)
        assert '--daemon' not in args and '--daemon-wait' not in args
        assert args[args.index('--rc-addr') + 1] == '127.0.0.1:5572'
        assert kwargs['start_new_session'] is True
        assert (tmp_path / 'rclone.conf').stat().st_mode & 0o777 == 0o600
        assert (tmp_path / 'rclone.log').stat().st_mode & 0o777 == 0o600
        assert all(value not in args for value in ['fake-access', 'fake-refresh', 'fake-secret'])
        kwargs['stdout'].write(b'Cannot bind RC port; fake-access fake-refresh fake-secret access_token=fresh-token\n')
        kwargs['stdout'].flush()
        return process

    monkeypatch.setattr(runtime.subprocess, 'Popen', launch)
    checks = iter([False, mounted])
    monkeypatch.setattr(runtime.os.path, 'ismount', lambda path: next(checks, mounted))
    clock = iter([0, 61] if timeout else [0])
    monkeypatch.setattr(runtime.time, 'monotonic', lambda: next(clock, 0))
    monkeypatch.setattr(runtime.time, 'sleep', lambda _: None)
    return runtime, secret, process, seen


def test_drive_mount_uses_one_rc_process_and_waits_for_actual_fuse_mount(monkeypatch, tmp_path):
    runtime, secret, process, seen = _mount_fixture(monkeypatch, tmp_path, mounted=True)
    assert runtime.mount_drive(tmp_path, secret) == tmp_path / 'drive'
    assert len(seen) == 1 and not process.stopped


def test_drive_mount_early_exit_reports_sanitized_child_error(monkeypatch, tmp_path):
    runtime, secret, _, _ = _mount_fixture(monkeypatch, tmp_path, exit_code=1)
    with pytest.raises(RuntimeError, match='Cannot bind RC port') as caught:
        runtime.mount_drive(tmp_path, secret)
    assert not any(value in str(caught.value) for value in ['fake-access', 'fake-refresh', 'fake-secret', 'fresh-token'])


@pytest.mark.parametrize('slow_stop', [False, True])
def test_drive_mount_timeout_reaps_its_own_process(monkeypatch, tmp_path, slow_stop):
    runtime, secret, process, _ = _mount_fixture(monkeypatch, tmp_path, timeout=True, slow_stop=slow_stop)
    with pytest.raises(RuntimeError, match='挂载就绪超时'):
        runtime.mount_drive(tmp_path, secret)
    assert process.stopped and process.killed is slow_stop


def test_drive_mount_reuses_existing_mount_without_rewriting_credentials(monkeypatch, tmp_path):
    from app.gpu import native_runtime as runtime

    monkeypatch.setattr(runtime.os.path, 'ismount', lambda _: True)
    monkeypatch.setattr(runtime.subprocess, 'Popen', lambda *a, **kw: pytest.fail('must not start another RC server'))
    assert runtime.mount_drive(tmp_path, {}) == tmp_path / 'drive'
    assert not (tmp_path / 'rclone.conf').exists()


@pytest.mark.parametrize('content_type,expected', [('text/html; charset=utf-8', 'strict-origin-when-cross-origin'), ('application/json', 'same-origin')])
def test_studio_document_policy_preserves_handoff_origin(content_type, expected):
    from app.main import security_context
    from starlette.requests import Request
    from starlette.responses import Response

    async def response(request):
        return Response('test', headers={'content-type': content_type})

    request = Request({'type': 'http', 'method': 'GET', 'path': '/', 'headers': [], 'query_string': b''})
    result = asyncio.run(security_context(request, response))
    assert result.headers['Referrer-Policy'] == expected


@pytest.mark.parametrize('origin', ['null', '', 'https://evil.example'])
def test_handoff_still_rejects_missing_opaque_and_foreign_origins(proxy, origin):
    now = int(time.time())
    ticket = sign_session({'uid': 'owner', 'aud': settings.native_comfy_url, 'purpose': 'native-handoff', 'iat': now, 'exp': now + 60})
    with TestClient(proxy, base_url=settings.native_comfy_url) as c:
        assert c.post('/_studio/session', data={'ticket': ticket}, headers={'Origin': origin}).status_code == 403


@pytest.mark.parametrize('deleted', [True, False])
def test_offline_native_tunnel_only_yields_to_confirmed_remote_deletion(proxy, monkeypatch, deleted):
    repo = proxy.state.repo
    monkeypatch.setattr(native, 'busy', lambda: True)
    monkeypatch.setattr(native, '_runtime_deleted', lambda repo, lease: deleted)
    assert native.protect_lease(repo, repo.lease, clock=lambda: 20000) is not deleted
    if deleted:
        assert repo.lease['native_ready'] is False
        assert repo.lease['native_draining'] is True
        assert repo.lease['native_remote_deleted_verified_at'] == 20000
    else:
        assert repo.lease['native_ready'] is True


def test_remote_deletion_evidence_does_not_apply_after_lease_handoff(proxy, monkeypatch):
    repo = proxy.state.repo
    lease = {**repo.lease, 'owner_job_id': 'old-job'}
    repo.lease = lease
    monkeypatch.setattr(native, 'busy', lambda: True)

    def inspect(repo, original):
        repo.lease = {**original, 'owner_job_id': 'new-job'}
        return True

    monkeypatch.setattr(native, '_runtime_deleted', inspect)
    assert native.protect_lease(repo, lease, clock=lambda: 20000) is True
    assert repo.lease['native_ready'] is True


@pytest.mark.parametrize('status', [{'deleted': True}, {'deleted': False}, {}, {'deleted': 'true'}, None])
def test_remote_deletion_probe_uses_exact_account_and_requires_boolean_confirmation(monkeypatch, status):
    from app.gpu import colab

    class Provider:
        def __init__(self, repo, owner, *, account_id, client):
            assert owner == 'owner' and account_id == 'colab-account'

        def inspect(self, handle):
            assert handle['owner_job_id'] == handle['job_id'] == 'current-job'
            if status is None:
                raise httpx.ConnectError('unavailable')
            return status

    monkeypatch.setattr(colab, 'GoogleColabProvider', Provider)
    lease = {'provider': 'colab', 'account_id': 'colab-account', 'owner_id': 'owner',
             'owner_job_id': 'current-job', 'handle': {'owner_job_id': 'old-job'}}
    assert native._runtime_deleted(object(), lease) is bool(status and status.get('deleted') is True)
    lease['account_id'] = None
    assert native._runtime_deleted(object(), lease) is False


def test_http_pool_reuses_connections_closes_and_never_replays_cookies(proxy, monkeypatch):
    clients = []
    seen = []

    async def handler(req):
        seen.append(req)
        assert 'cookie' not in req.headers
        return httpx.Response(200, stream=BytesStream(b'ok'), headers={
            'Set-Cookie': 'private_backend_cookie=never-replay; Path=/',
        })

    original = httpx.AsyncClient

    def make_client(**kwargs):
        client = original(transport=httpx.MockTransport(handler), **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(native_proxy.httpx, 'AsyncClient', make_client)
    with TestClient(proxy, base_url=settings.native_comfy_url) as c:
        for path in ['/assets/first.js', '/assets/second.js', '/api/object_info']:
            r = c.get(path, headers={'Authorization': 'Bearer api-test-token'})
            assert r.content == b'ok'
            assert 'set-cookie' not in r.headers
        assert len(clients) == 1
        assert not clients[0].is_closed
    assert len(seen) == 3
    assert clients[0].is_closed


@pytest.mark.parametrize('path', [
    '/customnode/install', '/manager/queue/install', '/v2/manager/queue/install_model',
    '/v2/customnode/install', '/api/v2/manager/queue/install_model',
])
def test_manager_install_requires_login_and_same_origin(proxy, monkeypatch, path):
    seen = []

    async def handler(req):
        seen.append(req)
        return httpx.Response(200, stream=BytesStream(b'{}'))

    original = httpx.AsyncClient
    monkeypatch.setattr(native_proxy.httpx, 'AsyncClient', lambda **kwargs: original(
        transport=httpx.MockTransport(handler), **kwargs))
    now = int(time.time())
    cookie = sign_session({'uid': 'owner', 'aud': settings.native_comfy_url, 'purpose': 'native-session',
                           'iat': now, 'exp': now + 60})
    with TestClient(proxy, base_url=settings.native_comfy_url) as c:
        assert c.post(path, json={}).status_code == 401
        assert c.post(path, json={}, headers={'Authorization': 'Bearer invalid'}).status_code == 401
        c.cookies.set('studio_native', cookie)
        assert c.post(path, json={}, headers={'Origin': 'https://evil.example'}).status_code == 403
        assert not seen
        assert c.post(path, json={}, headers={'Origin': settings.native_comfy_url}).status_code == 200
    assert len(seen) == 1


def test_ready_rejects_manager_access_bypass(proxy):
    def handler(req):
        if 'CF-Access-Client-Secret' not in req.headers and req.url.path == '/system_stats':
            return httpx.Response(403)
        return httpx.Response(200, json={'system': {}, 'busy': False})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert not native.ready(client)


@pytest.mark.parametrize('failure', [None, 'requirements', 'log', 'api', 'stale-log', 'node-import'])
def test_native_bootstrap_installs_and_verifies_manager_once(monkeypatch, tmp_path, failure):
    import json
    import sys
    from types import SimpleNamespace

    from app.gpu import native_runtime as runtime
    from app.gpu import runtime_worker as worker

    monkeypatch.setitem(sys.modules, 'runtime_worker', worker)
    from app.gpu.node_persistence import Remote
    monkeypatch.setattr(Remote, 'get', lambda *args: False)
    monkeypatch.setenv('PIP_CONSTRAINT', '')
    monkeypatch.setenv('UV_CONSTRAINT', '')
    service = tmp_path / 'native'
    comfy = service / 'ComfyUI'
    comfy.mkdir(parents=True)
    (comfy / 'requirements.txt').touch()
    if failure != 'requirements':
        (comfy / 'manager_requirements.txt').touch()
    job = tmp_path / 'job'
    job.mkdir()
    if failure == 'stale-log':
        (job / 'comfy.log').write_text('[START] ComfyUI-Manager\n')
    drive = tmp_path / 'drive'
    drive.mkdir()
    monkeypatch.setattr(runtime, 'ROOT', service)
    monkeypatch.setattr(runtime, 'install_rclone', lambda *args: None)
    monkeypatch.setattr(runtime.shutil, 'which', lambda name: '/usr/bin/' + name)
    monkeypatch.setattr(runtime, '_index_models', lambda *args: None)
    monkeypatch.setattr(runtime, 'mount_drive', lambda *args: drive)
    monkeypatch.setattr(worker, '_verify_checkout', lambda *args: None)
    monkeypatch.setattr(worker, '_process_metadata', lambda *args: {'start_id': '1'})
    commands, launches = [], []
    monkeypatch.setattr(worker, '_run', lambda args, **kwargs: commands.append(args))

    def http(url, **kwargs):
        if url.endswith('/system_stats'):
            return {'system': {}}
        if failure == 'api':
            raise worker.WorkerError('Manager missing')
        return {'is_processing': False}
    monkeypatch.setattr(worker, '_http_json', http)
    process = SimpleNamespace(pid=123, stopped=False, killed=False)
    process.terminate = lambda: setattr(process, 'stopped', True)
    process.wait = lambda **kwargs: 0
    process.kill = lambda: setattr(process, 'killed', True)

    def launch(args, **kwargs):
        launches.append((args, kwargs))
        if 'main.py' in args and failure not in {'log', 'stale-log'}:
            kwargs['stdout'].write(b'[START] ComfyUI-Manager\n')
            if failure == 'node-import':
                kwargs['stdout'].write(b'1.0 seconds (IMPORT FAILED): /custom_nodes/broken\n')
            kwargs['stdout'].flush()
        return process
    monkeypatch.setattr(runtime.subprocess, 'Popen', launch)
    stages = SimpleNamespace(start=lambda name: None, end=lambda: None, timings={})
    payload = {'snapshot': {'manifest': {'comfy_commit': 'a' * 40}},
               'native_secrets': {'max_upload_mb': 2048, 'tunnel_token': 'fake-token'}}
    if failure:
        with pytest.raises(RuntimeError, match='Manager|Custom node load failed'):
            runtime.run(job, payload, stages)
        assert not (service / 'state.json').exists()
        assert len(launches) == (0 if failure == 'requirements' else 1)
        assert process.stopped is (failure != 'requirements')
        return

    runtime.run(job, payload, stages)
    assert [sys.executable, '-m', 'pip', 'install', '-q', 'jedi>=0.16,<0.20'] in commands
    manager_pip = [args for args in commands if str(comfy / 'manager_requirements.txt') in args]
    assert manager_pip == [[sys.executable, '-m', 'pip', 'install', '-q', '-r',
                            str(comfy / 'manager_requirements.txt')]]
    args, kwargs = launches[0]
    data = drive / 'native-comfy'
    assert args == [sys.executable, 'main.py', '--listen', '127.0.0.1', '--port', '8188', '--enable-manager',
                    '--user-directory', str(data / 'user'), '--input-directory', str(data / 'input'),
                    '--output-directory', str(data / 'output'), '--temp-directory', str(service / 'temp'),
                    '--max-upload-size', '2048', '--extra-model-paths-config', str(service / 'extra_model_paths.yaml'),
                    '--preview-method', 'auto', '--disable-auto-launch']
    assert kwargs['cwd'] == comfy
    marker = json.loads((service / 'state.json').read_text())
    assert marker['comfy_log'] == str(job / 'comfy.log')
    # A different bootstrap job reuses the existing process and its original log.
    second_job = tmp_path / 'second-job'
    second_job.mkdir()
    runtime.run(second_job, payload, stages)
    assert len(launches) == 3  # ComfyUI, tunnel and watchdog; no further launches.
    assert len([args for args in commands if str(comfy / 'manager_requirements.txt') in args]) == 1
    # A reclaimed VM has no native state: a fresh bootstrap reinstalls dependencies.
    (service / 'state.json').unlink()
    runtime.run(second_job, payload, stages)
    assert len([args for args in commands if str(comfy / 'manager_requirements.txt') in args]) == 2
    assert sum('main.py' in args for args, _ in launches) == 2


def test_manager_probe_supports_legacy_queue_path(monkeypatch, tmp_path):
    from app.gpu import native_runtime as runtime
    from app.gpu import runtime_worker as worker

    log = tmp_path / 'comfy.log'
    log.write_text('[START] ComfyUI-Manager\n')
    seen = []

    def http(url, **kwargs):
        seen.append(url)
        if '/v2/' in url:
            raise worker.WorkerError('404')
        return {'is_processing': False}
    monkeypatch.setattr(worker, '_http_json', http)
    runtime.verify_manager(worker, log)
    assert len(seen) == 2


@pytest.mark.parametrize('path', ['/api/v2/manager/queue/task', '/v2/manager/queue/task', '/upload/image'])
@pytest.mark.parametrize('outcome', ['ok', 'busy', 'deleted', 'draining', 'identity', 'stopped', 'expired', 'not_ready'])
def test_write_admission_waits_and_revalidates(proxy, monkeypatch, path, outcome):
    repo = proxy.state.repo
    attempts, calls = [], []
    original_begin = native.begin_write
    monkeypatch.setattr(native, 'begin_write', lambda *args, **kwargs: original_begin(*args, timeout=0.05, **kwargs))

    @contextmanager
    def lock():
        attempts.append(1)
        if len(attempts) == 1:
            # Replace, rather than mutate, the snapshot selected before waiting.
            fresh = dict(repo.lease)
            if outcome == 'deleted':
                fresh = None
            elif outcome == 'identity':
                fresh['runtime_id'] = 'replacement'
            elif outcome == 'draining':
                fresh['native_draining'] = True
            elif outcome == 'stopped':
                repo.instance['manual_stop_requested'] = True
            elif outcome == 'expired':
                fresh['deadline_epoch'] = time.time() - 1
            elif outcome == 'not_ready':
                fresh['native_ready'] = False
            repo.lease = fresh
        yield len(attempts) > 1 and outcome != 'busy'

    monkeypatch.setattr(repo, 'dispatch_lock', lock)

    async def handler(req):
        calls.append(await req.aread())
        return httpx.Response(200, stream=BytesStream(b'accepted'))

    original = httpx.AsyncClient
    monkeypatch.setattr(native_proxy.httpx, 'AsyncClient', lambda **kwargs: original(
        transport=httpx.MockTransport(handler), **kwargs))
    started = time.monotonic()
    with TestClient(proxy, base_url=settings.native_comfy_url) as client:
        response = client.post(path, content=b'ordered\x00upload', headers={'Authorization': 'Bearer api-test-token'})
    assert time.monotonic() - started < 1
    assert len(attempts) >= 2
    assert response.status_code == (200 if outcome == 'ok' else 503 if outcome == 'busy' else 409)
    assert calls == ([b'ordered\x00upload'] if outcome == 'ok' else [])
    if outcome != 'ok' and repo.lease:
        assert 'native_http_heartbeat' not in repo.lease
    if outcome == 'busy':
        assert response.json() == {'error': 'SCHEDULER_BUSY', 'message': '会话状态繁忙，请稍后重试'}


@pytest.mark.parametrize('path', ['/interrupt', '/api/interrupt', '/queue', '/api/queue'])
@pytest.mark.parametrize('change', ['stop', 'deadline'])
def test_stop_operations_remain_allowed_after_wait(proxy, monkeypatch, path, change):
    repo = proxy.state.repo
    attempts = []

    @contextmanager
    def lock():
        attempts.append(1)
        if change == 'stop':
            repo.instance['manual_stop_requested'] = True
        else:
            repo.lease = {**repo.lease, 'deadline_epoch': time.time() - 1}
        yield len(attempts) > 1

    monkeypatch.setattr(repo, 'dispatch_lock', lock)
    original = httpx.AsyncClient
    monkeypatch.setattr(native_proxy.httpx, 'AsyncClient', lambda **kwargs: original(
        transport=httpx.MockTransport(lambda req: httpx.Response(200, stream=BytesStream(b'ok'))), **kwargs))
    with TestClient(proxy, base_url=settings.native_comfy_url) as client:
        assert client.post(path, headers={'Authorization': 'Bearer api-test-token'}).status_code == 200
    assert repo.lease['native_http_heartbeat']


@pytest.mark.parametrize('network,status', [(False, 502), (False, 503), (True, 502)])
def test_admission_retry_never_replays_manager_post(proxy, monkeypatch, network, status):
    attempts, calls = [], []

    @contextmanager
    def lock():
        attempts.append(1)
        yield len(attempts) > 1

    monkeypatch.setattr(proxy.state.repo, 'dispatch_lock', lock)

    async def handler(req):
        calls.append(await req.aread())
        if network:
            raise httpx.ReadError('outcome unknown', request=req)
        return httpx.Response(status, stream=BytesStream(b'unknown'))

    original = httpx.AsyncClient
    monkeypatch.setattr(native_proxy.httpx, 'AsyncClient', lambda **kwargs: original(
        transport=httpx.MockTransport(handler), **kwargs))
    with TestClient(proxy, base_url=settings.native_comfy_url) as client:
        assert client.post('/api/v2/manager/queue/task', content=b'install',
                           headers={'Authorization': 'Bearer api-test-token'}).status_code == status
    assert calls == [b'install']


@pytest.mark.parametrize('admitted', [False, True])
def test_admission_does_not_consume_upload_until_lock(proxy, monkeypatch, admitted):
    """Drive ASGI directly so receive calls prove the upload stays unbuffered."""
    chunks, consumed, calls, attempts = [b'first\x00', b'second', b'last'], [], [], []
    original_begin = native.begin_write
    monkeypatch.setattr(native, 'begin_write', lambda *args, **kwargs: original_begin(*args, timeout=0.05, **kwargs))

    @contextmanager
    def lock():
        attempts.append(1)
        assert not consumed
        yield admitted and len(attempts) > 1

    monkeypatch.setattr(proxy.state.repo, 'dispatch_lock', lock)

    async def handler(req):
        calls.append(await req.aread())
        return httpx.Response(200, stream=BytesStream(b'ok'))

    original = httpx.AsyncClient
    monkeypatch.setattr(native_proxy.httpx, 'AsyncClient', lambda **kwargs: original(
        transport=httpx.MockTransport(handler), **kwargs))

    async def run():
        messages = []

        async def receive():
            if len(consumed) < len(chunks):
                chunk = chunks[len(consumed)]
                consumed.append(chunk)
                return {'type': 'http.request', 'body': chunk, 'more_body': len(consumed) < len(chunks)}
            await asyncio.Event().wait()

        async def send(message):
            messages.append(message)

        scope = {'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.3'},
                 'http_version': '1.1', 'method': 'POST', 'scheme': 'https', 'path': '/upload/image',
                 'raw_path': b'/upload/image', 'query_string': b'',
                 'headers': [(b'host', b'comfy.example.test'), (b'authorization', b'Bearer api-test-token')],
                 'server': ('comfy.example.test', 443), 'client': ('127.0.0.1', 1)}
        middleware = NativeHostMiddleware(proxy, proxy)
        try:
            await middleware(scope, receive, send)
        finally:
            if middleware._http_client:
                await middleware._http_client.aclose()
        assert messages[0]['status'] == (200 if admitted else 503)

    asyncio.run(run())
    assert calls == ([b''.join(chunks)] if admitted else [])
    assert consumed == (chunks if admitted else [])


def test_begin_write_waits_for_real_sqlite_dispatch_lock(tmp_path):
    import threading

    from app.db import Store
    from app.gpu.repository import Repository

    url = f'sqlite:///{tmp_path / "studio.db"}'
    holder, contender = Store(url), Store(url)
    repo = Repository(contender)
    lease = repo.create('leases', 'owner', Repo().lease)
    entered, done = threading.Event(), threading.Event()
    errors = []

    def admit():
        entered.set()
        try:
            native.begin_write(repo, lease)
        except Exception as exc:
            errors.append(exc)
        finally:
            done.set()

    with holder.dispatch_lock() as acquired:
        assert acquired
        thread = threading.Thread(target=admit)
        thread.start()
        assert entered.wait(1)
        assert not done.wait(0.08)
        assert not repo.get('leases', lease['id'], 'owner').get('native_http_heartbeat')
    thread.join(2)
    assert done.is_set() and not errors
    assert repo.get('leases', lease['id'], 'owner')['native_http_heartbeat']
    holder.engine.dispose()
    contender.engine.dispose()


def test_manual_stop_bypasses_busy_and_heartbeat_protection(proxy, monkeypatch):
    repo = proxy.state.repo
    repo.instance['manual_stop_requested'] = True
    repo.lease['native_http_heartbeat'] = 10000
    monkeypatch.setattr(native, 'busy', lambda: True)
    monkeypatch.setattr(native, '_runtime_deleted', lambda repo, lease: False)
    assert not native.protect_lease(repo, repo.lease, clock=lambda: 10005)
    assert repo.lease['native_draining'] is True


def test_native_cold_start_can_outlast_three_minutes(monkeypatch):
    from types import SimpleNamespace

    from app.gpu import native_runtime as runtime

    class NotReady(Exception):
        pass
    clock = [0]
    def health(*args, **kwargs):
        if clock[0] < 240:
            raise NotReady
        return {'system': {}}
    monkeypatch.setattr(runtime.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(runtime.time, 'sleep', lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    worker = SimpleNamespace(_http_json=health, WorkerError=NotReady)
    runtime.wait_for_comfy(worker, SimpleNamespace(poll=lambda: None))
    assert clock[0] == 240


@pytest.mark.parametrize('exit_code,timeout,error', [(1, 2100, 'exited'), (None, 2, 'timed out')])
def test_native_startup_still_fails_on_exit_or_deadline(monkeypatch, exit_code, timeout, error):
    from types import SimpleNamespace

    from app.gpu import native_runtime as runtime

    class NotReady(Exception):
        pass
    clock = [0]
    def health(*args, **kwargs):
        raise NotReady
    monkeypatch.setattr(runtime.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(runtime.time, 'sleep', lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    worker = SimpleNamespace(_http_json=health, WorkerError=NotReady)
    with pytest.raises(RuntimeError, match=error):
        runtime.wait_for_comfy(worker, SimpleNamespace(poll=lambda: exit_code), timeout)
    assert clock[0] == (0 if exit_code is not None else timeout)


def test_colab_cli_dual_layer_keepalive_refreshes_proxy_recreates_kernel_and_executes_ws(monkeypatch):
    import json

    import httpx
    from app.db import Store
    from app.gpu.colab import GoogleColabProvider
    from app.gpu.mcp import MCPGPUExecutor
    from app.gpu.repository import Repository
    from app.security import decrypt_secret, encrypt_secret
    from cryptography.fernet import Fernet

    monkeypatch.setattr(settings, "encryption_key", Fernet.generate_key().decode())

    store = Store("sqlite:///:memory:")
    repo = Repository(store)
    seen = []
    ws_calls = []
    state = {"tfe_timeout": False}

    def transport(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, str(request.url.host), request.url.path, dict(request.headers)))
        if request.url.path == "/tun/m/m-s-test123/keep-alive/":
            assert request.headers.get("X-Colab-Tunnel") == "Google"
            assert request.headers.get("X-Colab-Client-Agent") == "colab-cli"
            assert request.headers.get("Authorization") == "Bearer colab-token"
            if state["tfe_timeout"]:
                raise httpx.ReadTimeout("stream held open")
            return httpx.Response(200, json={"ok": True})
        if request.url.path == "/tun/m/assignments":
            return httpx.Response(200, json={
                "assignments": [{
                    "endpoint": "m-s-test123",
                    "runtimeProxyInfo": {
                        "url": "https://8080-test.colab.dev/",
                        "token": "fresh-proxy-token",
                        "tokenExpiresInSeconds": 3600,
                    },
                }]
            })
        if request.url.host == "8080-test.colab.dev":
            tok = request.headers.get("X-Colab-Runtime-Proxy-Token")
            if tok != "fresh-proxy-token":
                return httpx.Response(401, json={"error": "expired"})
            if request.url.path == "/api/kernels/dead-kernel":
                return httpx.Response(404, json={"message": "not found"})
            if request.url.path == "/api/kernels" and request.method == "GET":
                return httpx.Response(200, json=[])
            if request.url.path == "/api/kernels" and request.method == "POST":
                assert json.loads(request.content.decode()) == {"name": "python3"}
                return httpx.Response(201, json={"id": "new-kernel-99"})
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    class FakeWS:
        def __init__(self, url, **kwargs):
            self.url = url
            self.kwargs = kwargs
            self.replies = []

        def __enter__(self):
            ws_calls.append((self.url, self.kwargs.get("additional_headers")))
            return self

        def __exit__(self, *args):
            return False

        def send(self, frame):
            msg = MCPGPUExecutor._ws_unpack_default(frame)
            assert msg["content"]["code"] == "1 + 1"
            assert msg["content"]["store_history"] is False
            mid = msg["header"]["msg_id"]
            self.replies = [
                MCPGPUExecutor._ws_pack_default({
                    "header": {"msg_type": "execute_reply"},
                    "parent_header": {"msg_id": mid},
                    "content": {"status": "ok"},
                }),
                MCPGPUExecutor._ws_pack_default({
                    "header": {"msg_type": "status"},
                    "parent_header": {"msg_id": mid},
                    "content": {"execution_state": "idle"},
                }),
            ]

        def recv(self, timeout=None):
            return self.replies.pop(0)

    class FakeGoogle:
        def access_token(self, scope):
            assert scope == "colab"
            return "colab-token"

    lease = {
        "id": "lease-1",
        "owner_id": "owner",
        "tunnel_endpoint": "m-s-test123",
        "proxy_url": "https://8080-test.colab.dev",
        "encrypted_proxy_token": encrypt_secret("stale-proxy-token"),
        "proxy_token_expires_at": time.time() + 1000,
        "kernel_id": "dead-kernel",
    }
    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        provider = GoogleColabProvider(repo, "owner", account_id="acc", client=client, google=FakeGoogle())
        res = provider.keep_alive(lease, run_kernel_exec=True, ws_connect_fn=FakeWS)
        assert res["ok"] is True
        assert res["tfe"] == "ok:200"
        assert res["proxy"] == "ok"
        assert res["kernel_exec"] == "ok"
        assert res["lease_patch"]["kernel_id"] == "new-kernel-99"
        assert decrypt_secret(res["lease_patch"]["encrypted_proxy_token"]) == "fresh-proxy-token"
        assert len(ws_calls) == 1
        assert "new-kernel-99" in ws_calls[0][0]
        assert ws_calls[0][1]["X-Colab-Runtime-Proxy-Token"] == "fresh-proxy-token"

        # ReadTimeout on TFE stream is also accepted as signal ack
        state["tfe_timeout"] = True
        lease_updated = {**lease, **res["lease_patch"]}
        seen_known = []

        def transport_steady(request: httpx.Request) -> httpx.Response:
            seen_known.append(request.url.path)
            if request.url.path == "/tun/m/m-s-test123/keep-alive/":
                raise httpx.ReadTimeout("stream held open")
            if request.url.path == "/api/kernels/new-kernel-99":
                return httpx.Response(200, json={"id": "new-kernel-99"})
            raise AssertionError(f"Unexpected: {request.url}")

        with httpx.Client(transport=httpx.MockTransport(transport_steady)) as client2:
            provider2 = GoogleColabProvider(repo, "owner", account_id="acc", client=client2, google=FakeGoogle())
            res2 = provider2.keep_alive(lease_updated, run_kernel_exec=False, ws_connect_fn=FakeWS)
            assert res2["ok"] is True
            assert res2["tfe"] == "read_timeout_ack"
            assert res2["proxy"] == "ok"
            assert res2["kernel_exec"] == "skipped"


def test_protect_lease_gates_and_throttles_keepalive(monkeypatch):
    from app.db import Store
    from app.gpu.compute import GpuConfig
    from app.gpu.repository import Repository

    store = Store("sqlite:///:memory:")
    repo = Repository(store)
    now = [1000.0]
    repo.create("instances", "owner", {
        "state": "retained_idle", "provider": "colab", "account_id": "acc", "lease_id": "lease-ka",
        "manual_stop_requested": False,
    }, record_id="inst-ka")
    repo.create("jobs", "owner", {
        "state": "completed", "provider": "colab", "snapshot": {"native_comfy": True},
        "execution_policy": {"lifecycle": "retain", "retain_idle_minutes": 120, "max_runtime_minutes": 720},
    }, record_id="job-ka")
    repo.create("leases", "owner", {
        "state": "idle", "provider": "colab", "owner_job_id": "job-ka", "account_id": "acc",
        "instance_id": "inst-ka", "native_comfy": True, "native_ready": True,
        "native_last_busy": 1000.0, "native_idle_minutes": 120, "deadline_epoch": 1000.0 + 720 * 60,
        "runtime_id": "studio-ka", "request_id": "req-ka", "tunnel_endpoint": "m-s-ka",
        "handle": {"runtime_name": "runtimes/studio-ka"},
    }, record_id="lease-ka")

    calls = []

    class FakeProvider:
        def keep_alive(self, lease, *, run_kernel_exec=False):
            calls.append((now[0], run_kernel_exec))
            return {
                "ok": True,
                "tfe": "ok:200",
                "proxy": "ok",
                "kernel_exec": "ok" if run_kernel_exec else "skipped",
                "lease_patch": {"kernel_id": "k-1"},
            }

    monkeypatch.setattr(native, "busy", lambda: False)
    monkeypatch.setattr(native.GpuConfig, "from_env", lambda: GpuConfig(real_enabled=False))

    # t=1000: first check fires both HTTP keep-alive and WS kernel exec
    assert native.protect_lease(repo, repo.get("leases", "lease-ka", "owner"), clock=lambda: now[0], provider_factory=lambda job: FakeProvider())
    assert calls == [(1000.0, True)]
    saved = repo.get("leases", "lease-ka", "owner")
    assert saved["kernel_id"] == "k-1"
    assert saved["native_last_keepalive_at"] == 1000.0
    assert saved["native_last_kernel_exec_at"] == 1000.0

    # t=1030 (<60s): throttled, no provider call
    now[0] = 1030.0
    assert native.protect_lease(repo, repo.get("leases", "lease-ka", "owner"), clock=lambda: now[0], provider_factory=lambda job: FakeProvider())
    assert len(calls) == 1

    # t=1065 (>=60s, <240s): HTTP keep-alive only, no WS kernel exec
    now[0] = 1065.0
    assert native.protect_lease(repo, repo.get("leases", "lease-ka", "owner"), clock=lambda: now[0], provider_factory=lambda job: FakeProvider())
    assert calls[-1] == (1065.0, False)

    # t=1245 (>=240s since last WS exec): both HTTP keep-alive and WS kernel exec
    now[0] = 1245.0
    assert native.protect_lease(repo, repo.get("leases", "lease-ka", "owner"), clock=lambda: now[0], provider_factory=lambda job: FakeProvider())
    assert calls[-1] == (1245.0, True)

    # Deadline reached even if upstream_busy=True: keep-alive stops immediately (no infinite CU burn)
    monkeypatch.setattr(native, "busy", lambda: True)
    monkeypatch.setattr(native, "_runtime_deleted", lambda r, lease: False)
    now[0] = 1000.0 + 720 * 60
    count_before = len(calls)
    assert native.protect_lease(repo, repo.get("leases", "lease-ka", "owner"), clock=lambda: now[0], provider_factory=lambda job: FakeProvider())
    assert len(calls) == count_before

    # Manual stop: drains immediately and never sends keep-alive
    repo.update("instances", "inst-ka", {"manual_stop_requested": True}, "owner")
    now[0] = 1500.0
    assert not native.protect_lease(repo, repo.get("leases", "lease-ka", "owner"), clock=lambda: now[0], provider_factory=lambda job: FakeProvider())
    assert len(calls) == count_before
