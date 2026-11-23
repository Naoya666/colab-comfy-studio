import time

import httpx
import pytest
from app.common import APIError
from app.db import Store
from app.gpu import native
from app.gpu import native_restart as restart
from app.gpu.repository import Repository

URL = "https://native.test"


@pytest.fixture
def session(monkeypatch):
    repo = Repository(Store("sqlite:///:memory:"))
    lease = repo.create("leases", "owner", {
        "instance_id": "instance", "runtime_id": "runtime", "state": "idle",
        "deadline_epoch": time.time() + 3600, "native_comfy": True, "native_ready": True,
    }, record_id="lease")
    repo.create("instances", "owner", {"lease_id": lease["id"], "state": "retained_idle"}, record_id="instance")
    monkeypatch.setattr(native, "backend", lambda: (URL, {}))
    return repo, lease


class Runtime:
    def __init__(self, monkeypatch):
        self.boot = "old-boot"
        self.calls = 0
        self.reboot_status = 502
        self.drop = False
        self.busy_reason = None
        self.running = False
        self.missing = []
        self.fail_index = False
        client = httpx.Client
        monkeypatch.setattr(restart.httpx, "Client", lambda **kwargs: client(
            transport=httpx.MockTransport(self.handle), **kwargs
        ))
        monkeypatch.setattr(restart, "_sync_models",
                            lambda *args: [{"category": "diffusion_models", "alias": "model.safetensors"}])

    def handle(self, request):
        if request.url.path == "/queue":
            return httpx.Response(200, json={"queue_running": [1] if self.running else [], "queue_pending": []})
        if request.url.path == "/studio_native/state":
            return httpx.Response(200, json={"boot_id": self.boot, "lease_id": "lease",
                                            "busy": bool(self.busy_reason), "busy_reason": self.busy_reason})
        if request.url.path == "/system_stats":
            return httpx.Response(200, json={"system": {}})
        if request.url.path == "/studio_native/index":
            return httpx.Response(503) if self.fail_index else httpx.Response(200, json={"missing": self.missing})
        assert request.url.path == "/v2/manager/reboot" and request.method == "POST"
        assert request.headers["x-studio-restart-boot"] == "old-boot"
        assert request.headers["x-studio-restart-lease"] == "lease"
        self.calls += 1
        if self.drop:
            raise httpx.ReadError("connection closed")
        return httpx.Response(self.reboot_status)


def phase(repo):
    return repo.get("leases", "lease", "owner")["native_restart"]["phase"]


@pytest.mark.parametrize("drop", [False, True])
def test_restarts_once_and_requires_new_boot(session, monkeypatch, drop):
    repo, lease = session
    runtime = Runtime(monkeypatch)
    runtime.drop = drop
    restart_id = restart.submit(repo, "owner", "instance", "one")
    assert restart.submit(repo, "owner", "instance", "one") == restart_id
    assert restart.submit(repo, "owner", "instance", "other-tab") == restart_id
    with pytest.raises(APIError, match="正在重启"):
        native.begin_write(repo, lease)
    # Three-pool isolation: native ComfyUI restart must NOT block creation/ephemeral generation jobs.
    native.guard_generation(repo, "owner", {"provider": "colab"})

    restart.tick(repo)
    assert phase(repo) == "recovering"
    restart.tick(repo)
    assert phase(repo) == "recovering" and runtime.calls == 1
    runtime.boot = "new-boot"
    restart.tick(repo)
    fresh = repo.get("leases", lease["id"], "owner")
    assert restart.public_restart(fresh)["phase"] == "succeeded"
    for key in ("id", "instance_id", "runtime_id", "deadline_epoch"):
        assert fresh[key] == lease[key]
    native.begin_write(repo, fresh)
    assert restart.submit(repo, "owner", "instance", "other-tab") == restart_id
    assert runtime.calls == 1


@pytest.mark.parametrize("reason,message", [("manager", "Manager 安装"), ("plugins", "插件归档"), ("drive", "Drive 写入"), ("queue", "ComfyUI 生成")])
def test_busy_revokes_fence(session, monkeypatch, reason, message):
    repo, lease = session
    runtime = Runtime(monkeypatch)
    runtime.running = reason == "queue"
    runtime.busy_reason = reason
    with pytest.raises(APIError, match=message):
        restart.submit(repo, "owner", "instance", "key")
    assert phase(repo) == "failed" and runtime.calls == 0
    native.begin_write(repo, lease)


@pytest.mark.parametrize("failure", ["index", "timeout", "reclaimed", "stopped", "missing"])
def test_failure_keeps_stage_and_never_resends(session, monkeypatch, failure):
    repo, lease = session
    runtime = Runtime(monkeypatch)
    runtime.fail_index = failure == "index"
    if failure in {"timeout", "missing"}:
        monkeypatch.setattr(restart, "RECOVERY_SECONDS", 1)
    restart.submit(repo, "owner", "instance", "key")
    restart.tick(repo)
    if failure != "index":
        if failure == "reclaimed":
            repo.update("leases", "lease", {"state": "deleted"}, "owner")
        elif failure == "stopped":
            repo.update("instances", "instance", {"manual_stop_requested": True}, "owner")
        else:
            fresh = repo.get("leases", "lease", "owner")
            repo.update("leases", "lease", {"native_restart": {**fresh["native_restart"], "recover_until": 0}}, "owner")
            if failure == "missing":
                runtime.boot = "new"
                runtime.missing = ["model.safetensors"]
        restart.tick(repo)
    state = restart.public_restart(repo.get("leases", "lease", "owner"))
    assert state["phase"] == "failed" and state["error"]
    assert state["failed_phase"] == ("syncing" if failure == "index" else "recovering")
    assert runtime.calls == (0 if failure == "index" else 1)


def test_auth_and_expired_instance(session, monkeypatch):
    repo, _ = session
    runtime = Runtime(monkeypatch)
    with pytest.raises(APIError, match="已释放或已切换"):
        restart.submit(repo, "other-owner", "instance", "key")
    repo.update("leases", "lease", {"deadline_epoch": time.time() - 1}, "owner")
    with pytest.raises(APIError, match="已到期"):
        restart.submit(repo, "owner", "instance", "key")
    assert runtime.calls == 0


def test_releasing_lease_reports_stop_reason(session, monkeypatch):
    repo, _ = session
    runtime = Runtime(monkeypatch)
    repo.update("leases", "lease", {"state": "releasing"}, "owner")
    with pytest.raises(APIError, match="正在停止"):
        restart.submit(repo, "owner", "instance", "key")
    assert runtime.calls == 0


def test_crashed_worker_does_not_resend_manager_request(session, monkeypatch):
    repo, _ = session
    runtime = Runtime(monkeypatch)
    restart.submit(repo, "owner", "instance", "key")
    original = restart._save

    def killed(*args, **changes):
        if changes.get("phase") == "recovering":
            raise SystemExit
        return original(*args, **changes)

    with monkeypatch.context() as patch:
        patch.setattr(restart, "_save", killed)
        with pytest.raises(SystemExit):
            restart.tick(repo)
    assert phase(repo) == "rebooting" and runtime.calls == 1
    runtime.boot = "new-boot"
    restart.tick(repo)
    assert phase(repo) == "succeeded" and runtime.calls == 1


def test_drive_shortcut_target_verified_before_reuse():
    from types import SimpleNamespace

    from app.gpu.native_runtime import _index_models

    class Cache:
        folder_id = "models"
        _q = staticmethod(lambda value: value)

        def __init__(self):
            self.created = []
            self.items = []

        def _folder(self, name, parent, create):
            assert create
            return parent + "/" + name

        def _json(self, method, url, body=None):
            if method == "GET":
                return {"files": self.items}
            self.created.append(body)
            self.items.append(body)
            return {"id": "shortcut"}

    cache = Cache()
    worker = SimpleNamespace(DriveCache=lambda *_: cache, DRIVE_API="https://drive.test")
    secret = {"access_token": "token", "models_folder_id": "models",
              "models": [{"filename": "x.safetensors", "category": "vae", "file_id": "id"}]}
    expected = [{"category": "vae", "alias": "id-x.safetensors"}]
    assert _index_models(secret, worker) == expected
    assert _index_models(secret, worker) == expected and len(cache.created) == 1
    cache.items[0]["shortcutDetails"]["targetId"] = "different"
    with pytest.raises(ValueError, match="目标冲突"):
        _index_models(secret, worker)
    assert len(cache.created) == 1


def test_restart_records_phase_timings_and_live_in_progress_elapsed(session, monkeypatch):
    repo, _lease = session
    runtime = Runtime(monkeypatch)
    clock = {"now": 1000.0}
    monkeypatch.setattr(restart.time, "time", lambda: clock["now"])

    # Submit transitions checking -> syncing immediately after idle probe.
    restart.submit(repo, "owner", "instance", "timed-restart")
    fresh = repo.get("leases", "lease", "owner")
    clock["now"] = 1004.5
    pub = restart.public_restart(fresh, now=clock["now"])
    assert pub["phase"] == "syncing"
    assert pub["timing_details"]["restart_checking"] == {"seconds": 0.0, "source": "events", "status": "completed"}
    assert pub["timing_details"]["restart_syncing"] == {"seconds": 4.5, "source": "events", "status": "in_progress"}

    # Advance through syncing -> rebooting -> recovering.
    clock["now"] = 1006.0
    restart.tick(repo)
    fresh = repo.get("leases", "lease", "owner")
    pub = restart.public_restart(fresh, now=1018.3)
    assert pub["phase"] == "recovering"
    assert pub["timing_details"]["restart_syncing"]["status"] == "completed"
    assert pub["timing_details"]["restart_syncing"]["seconds"] == 6.0
    assert pub["timing_details"]["restart_rebooting"]["status"] == "completed"
    assert pub["timing_details"]["restart_recovering"] == {"seconds": 12.3, "source": "events", "status": "in_progress"}

    # Complete recovery on new boot_id.
    clock["now"] = 1020.4
    runtime.boot = "new-boot-2"
    restart.tick(repo)
    fresh = repo.get("leases", "lease", "owner")
    pub = restart.public_restart(fresh, now=1050.0)
    assert pub["phase"] == "succeeded"
    assert pub["timing_details"]["restart_recovering"] == {"seconds": 14.4, "source": "events", "status": "completed"}
    assert pub["completed_at"] == 1020.4
