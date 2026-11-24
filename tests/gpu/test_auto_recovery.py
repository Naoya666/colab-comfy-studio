"""Comprehensive unit & integration tests for GPU Keepalive & Auto-Recovery (docs/specs/GPU_AUTO_RECOVERY.md §10.1)."""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from app.config import settings
from app.db import Store
from app.gpu import ccu, native, native_workspace, recovery
from app.gpu.colab import (
    TUNNEL_RECONCILE_WINDOW_SECONDS,
    ColabAllocationError,
    ColabError,
    GoogleColabProvider,
)
from app.gpu.compute import GpuConfig
from app.gpu.instances import POOL_CREATION, POOL_EPHEMERAL, POOL_NATIVE
from app.gpu.native_models import DiskSpaceError, LocalModels
from app.gpu.native_runtime import WATCHDOG
from app.gpu.repository import Repository
from app.gpu.scheduler import Scheduler
from app.main import app as studio_app
from app.native_proxy import NativeGateway
from starlette.testclient import TestClient

ACC1 = "11111111-1111-4111-8111-111111111111"
ACC2 = "22222222-2222-4222-8222-222222222222"
SPEC_L4 = {"variant": "VARIANT_GPU", "accelerator": "L4", "shape": "SHAPE_STANDARD"}
SPEC_A100 = {"variant": "VARIANT_GPU", "accelerator": "A100", "shape": "SHAPE_STANDARD"}
VALID_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


class FakeClock:
    def __init__(self, start: float = 1_700_000_000.0):
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += float(seconds)
        return self.now


class MockColab:
    """Controllable Colab provider for deterministic L0/L2/L3 tests."""

    def __init__(self, *, specs=None, inspect_map=None, create_plan=None, delete_ok=True):
        self.specs = specs or [
            {"key": SPEC_L4, "eligible": True, "gpu": "L4", "vram_gb": 24},
            {"key": SPEC_A100, "eligible": True, "gpu": "A100", "vram_gb": 40},
        ]
        self.inspect_map = dict(inspect_map or {})
        self.create_plan = list(create_plan or [])
        self.delete_ok = delete_ok
        self.create_calls: list[dict] = []
        self.inspect_calls: list[dict] = []
        self.delete_calls: list[dict] = []
        self.keepalive_calls: list[dict] = []

    def list_specs(self):
        return list(self.specs)

    def create(self, spec, request_id, job_id, runtime_id):
        self.create_calls.append({
            "spec": spec,
            "request_id": request_id,
            "job_id": job_id,
            "runtime_id": runtime_id,
        })
        if self.create_plan:
            step = self.create_plan.pop(0)
            if isinstance(step, Exception):
                raise step
            if callable(step):
                return step(spec, request_id, job_id, runtime_id)
            return step
        return {
            "runtime_name": f"runtimes/{runtime_id}",
            "endpoint": f"ep-{runtime_id[-6:]}",
            "runtime_spec": spec,
        }

    def inspect(self, handle):
        self.inspect_calls.append(dict(handle))
        key = handle.get("runtime_id") or handle.get("runtime_name")
        res = self.inspect_map.get(key)
        if callable(res):
            return res(handle)
        if isinstance(res, dict):
            return dict(res)
        return {"ready": True, "deleted": False, "state": "RUNNING", "runtime_spec": {"key": SPEC_L4}}

    def delete(self, handle):
        self.delete_calls.append(dict(handle))
        if callable(self.delete_ok):
            return self.delete_ok(handle)
        return bool(self.delete_ok)

    def keep_alive(self, lease, *, run_kernel_exec: bool = False, smart: bool = False, tfe_only: bool = False, **kw):
        self.keepalive_calls.append({"lease": dict(lease), "run_kernel_exec": run_kernel_exec, "smart": smart, "tfe_only": tfe_only})
        return {"ok": True, "tfe": "ok:200", "proxy": "skipped", "kernel_exec": "skipped", "lease_patch": {}}

    def close(self):
        pass


class MockExecutor:
    def __init__(self, *, poll_plan=None, health_plan=None):
        self.poll_plan = list(poll_plan or [])
        self.health_plan = list(health_plan or [])
        self.start_calls: list[dict] = []
        self.poll_calls: list[dict] = []

    def start(self, handle, job, execution_id):
        self.start_calls.append({"handle": dict(handle), "job_id": job["id"], "execution_id": execution_id})
        return {"execution_id": execution_id}

    def poll(self, handle, job):
        self.poll_calls.append({"handle": dict(handle), "job_id": job["id"], "execution_id": job.get("execution_id")})
        if self.poll_plan:
            step = self.poll_plan.pop(0)
            if isinstance(step, Exception):
                raise step
            if callable(step):
                return step(handle, job)
            return step
        return {
            "state": "completed",
            "execution_id": job.get("execution_id"),
            "outputs": [{"filename": "out.png", "content": VALID_PNG_1X1, "mime": "image/png"}],
            "creation_runtime": {
                "instance_id": job.get("instance_id"),
                "process": "pid-1234",
                "environment": {"py": "3.12"},
                "workflow_sha256": (job.get("snapshot") or {}).get("workflow_sha256"),
                "warmup_completed": True,
            },
        }

    def creation_process_health(self, handle, job, runtime):
        if self.health_plan:
            return self.health_plan.pop(0)
        return True


def _make_repo(tmp_path: Path) -> Repository:
    settings.data_dir = tmp_path / "data"
    store = Store(f"sqlite:///{tmp_path / 'studio.db'}")
    return Repository(store)


def _seed_account(repo: Repository, owner: str = "owner-1", account_id: str = ACC1, alias: str = "主号") -> dict:
    from app.security import encrypt_secret

    acc = repo.create(
        "accounts",
        owner,
        {
            "account_id": account_id,
            "alias": alias,
            "provider": "colab",
            "connected": True,
            "auth_status": "connected",
            "enabled": True,
            "allow_allocation": True,
            "drive_account_id": account_id,
        },
        record_id=account_id,
    )
    repo.create(
        "credentials",
        owner,
        {
            "provider": "google",
            "purpose": "colab",
            "account_id": account_id,
            "state": "connected",
            "auth_channel": "colab_cli",
            "encrypted_refresh_token": encrypt_secret("fake-refresh-token"),
            "scopes": ["https://www.googleapis.com/auth/colaboratory"],
        },
    )
    return acc


def _seed_published_workflow(repo: Repository, owner: str = "owner-1", wf_id: str = "wf-1", key: str = "txt2img", version: int = 1) -> dict:
    if not any(m.get("model_id") == "flux-dev" for m in repo.list("models", owner)):
        repo.create(
            "models",
            owner,
            {"model_id": "flux-dev", "display_name": "Flux Dev", "enabled": True, "is_mock": False},
            record_id="model-flux-dev",
        )
    return repo.create(
        "workflows",
        owner,
        {
            "workflow_key": key,
            "workflow_version": version,
            "model_id": "flux-dev",
            "state": "published",
            "active": True,
            "review_approved": True,
            "smoke_test_kind": "real",
            "sha256": "a" * 64,
            "workflow_sha256": "a" * 64,
            "manifest": {
                "schema_version": "1",
                "model_id": "flux-dev",
                "workflow_key": key,
                "workflow_version": version,
                "title": "Test Workflow",
                "compute_spec": SPEC_L4,
                "parameters": {
                    "filename_prefix": {
                        "type": "string",
                        "default": "out",
                        "binding": {"node": "1", "input": "filename_prefix"},
                    }
                },
                "weight_files": [],
            },
            "api_graph": {"1": {"class_type": "SaveImage", "inputs": {"filename_prefix": "out"}}},
            "object_info": {
                "SaveImage": {
                    "input": {"required": {"filename_prefix": ["STRING", {"default": "out"}]}},
                    "output": [],
                    "output_node": True,
                }
            },
        },
        record_id=wf_id,
    )


def _seed_project(repo: Repository, owner: str = "owner-1", project_id: str = "proj-1") -> dict:
    return repo.store.create("projects", owner, {"title": "测试项目"}, record_id=project_id)


def _seed_running_instance(
    repo: Repository,
    clock: FakeClock,
    *,
    owner: str = "owner-1",
    account_id: str = ACC1,
    pool: str = POOL_NATIVE,
    instance_state: str = "retained_idle",
    lease_state: str = "idle",
    job_state: str = "completed",
    retention_mode: str = "retain",
    hard_deadline_at: float | None = None,
    idle_deadline_at: float | None = None,
    last_present_at: float | None = None,
    spec: dict | None = None,
    model_storage: dict | None = None,
    manual_stop_requested: bool = False,
    creation_runtime: dict | None = None,
    job_extra: dict | None = None,
    instance_extra: dict | None = None,
    lease_extra: dict | None = None,
) -> tuple[dict, dict, dict]:
    now = clock()
    spec = spec or SPEC_L4
    hard = hard_deadline_at if hard_deadline_at is not None else now + 3600.0
    idle = idle_deadline_at if idle_deadline_at is not None else now + 1800.0
    is_native = pool == POOL_NATIVE
    is_ephemeral = pool == POOL_EPHEMERAL

    job_payload = {
        "provider": "colab",
        "state": job_state,
        "account_id": account_id,
        "requested_account_id": account_id,
        "account_alias_snapshot": "主号",
        "instance_pool": pool,
        "requested_spec": {"key": spec},
        "workflow_key": "txt2img" if not is_native else None,
        "workflow_version": 1 if not is_native else None,
        "workflow_version_id": "wf-1" if not is_native else None,
        "model_id": "flux-dev" if not is_native else "native-comfy",
        "creation_fingerprint": "fp-abc" if pool == POOL_CREATION else None,
        "max_runtime_minutes": 60,
        "deadline_epoch": hard,
        "hard_deadline_at": hard,
        "attempt_number": 1,
        "snapshot": {
            "native_comfy": is_native,
            "system_smoke": is_ephemeral,
            "workflow_sha256": "a" * 64,
            "manifest": {"compute_spec": spec, "weight_files": [], "comfy_commit": "deadbeef"},
        },
        "execution_policy": {
            "provider": "colab",
            "account_id": account_id,
            "compute_spec": spec,
            "lifecycle": retention_mode,
            "retain_idle_minutes": 30,
            "max_runtime_minutes": 60,
            **(model_storage or {"model_storage_mode": "drive_mount", "cache_gb": 20, "cache_max_age_minutes": 60}),
        },
        **(job_extra or {}),
    }
    job = repo.create("jobs", owner, job_payload, project_id="proj-1")

    inst_payload = {
        "account_id": account_id,
        "account_alias_snapshot": "主号",
        "provider": "colab",
        "instance_pool": pool,
        "state": instance_state,
        "current_job_id": job["id"] if instance_state == "busy" else None,
        "runtime_spec_requested": {"key": spec},
        "runtime_spec_reported": {"key": spec},
        "retention_mode": retention_mode,
        "retain_idle_minutes": 30.0,
        "hard_deadline_at": hard,
        "idle_deadline_at": idle if instance_state == "retained_idle" else None,
        "manual_stop_requested": manual_stop_requested,
        "runtime_deleted": False,
        "native_comfy": is_native,
        "creation_runtime": creation_runtime,
        "allocation_requested_at": now - 600.0,
        "last_seen_at": now - 30.0,
        **(instance_extra or {}),
    }
    inst = repo.create("instances", owner, inst_payload, project_id="proj-1")

    runtime_id = f"studio-{inst['id'][:8]}"
    lease_payload = {
        "owner_job_id": job["id"],
        "instance_id": inst["id"],
        "account_id": account_id,
        "provider": "colab",
        "instance_pool": pool,
        "state": lease_state,
        "request_id": f"req-{inst['id'][:8]}",
        "runtime_id": runtime_id,
        "tunnel_endpoint": f"ep-{inst['id'][:6]}",
        "handle": {"runtime_name": f"runtimes/{runtime_id}", "runtime_id": runtime_id, "endpoint": f"ep-{inst['id'][:6]}"},
        "created_epoch": now - 600.0,
        "last_seen": now - 30.0,
        "deadline_epoch": hard,
        "hard_deadline_at": hard,
        "idle_deadline_at": idle if lease_state == "idle" else None,
        "native_comfy": is_native,
        "native_ready": is_native,
        "native_idle_minutes": 30,
        "native_last_busy": idle - 1800.0 if is_native else None,
        "native_bootstrap_job_id": job["id"] if is_native else None,
        "native_model_storage": model_storage or ({"model_storage_mode": "drive_mount", "cache_gb": 20, "cache_max_age_minutes": 60} if is_native else None),
        "allocation_confirmed": True,
        "provider_runtime_check": {
            "status": "present",
            "checked_at": now - 60.0,
            "next_check_at": now - 1.0,
            "last_present_at": last_present_at if last_present_at is not None else now - 60.0,
            "missing_first_seen_at": None,
        },
        **(lease_extra or {}),
    }
    lease = repo.create("leases", owner, lease_payload)
    inst = repo.update("instances", inst["id"], {"lease_id": lease["id"]}, owner)
    job = repo.update("jobs", job["id"], {"instance_id": inst["id"], "lease_id": lease["id"]}, owner)
    ccu.open_instance_session(repo, job, lease, now - 600.0)
    return job, inst, lease


def _enable_recovery(monkeypatch, **overrides):
    monkeypatch.setenv("GPU_REAL_ENABLED", "1")
    monkeypatch.setenv("GPU_AUTO_RECOVERY_ENABLED", "1")
    monkeypatch.setenv("GPU_AUTO_RECOVERY_STARTUP_GRACE_SECONDS", str(overrides.get("startup_grace", 0)))
    if "max_rounds" in overrides:
        monkeypatch.setenv("GPU_AUTO_RECOVERY_MAX_ROUNDS", str(overrides["max_rounds"]))
    if "max_job_retries" in overrides:
        monkeypatch.setenv("GPU_AUTO_RECOVERY_MAX_JOB_RETRIES", str(overrides["max_job_retries"]))
    if "round_timeout" in overrides:
        monkeypatch.setenv("GPU_AUTO_RECOVERY_ROUND_TIMEOUT_SECONDS", str(overrides["round_timeout"]))
    if "min_remaining" in overrides:
        monkeypatch.setenv("GPU_AUTO_RECOVERY_MIN_REMAINING_MINUTES", str(overrides["min_remaining"]))
    if "max_vps_outage" in overrides:
        monkeypatch.setenv("GPU_AUTO_RECOVERY_MAX_VPS_OUTAGE_MINUTES", str(overrides["max_vps_outage"]))


def _configure_native_mock(monkeypatch, *, mock_sync: bool = True):
    from app.gpu import native_config

    fake_cfg = SimpleNamespace(
        native_comfy_url="https://comfy.example.test",
        native_comfy_backend_url="https://internal.example.test",
        access_client_id="cf-id",
        access_client_secret="cf-sec",
        tunnel_token="cf-tunnel-tok",
        native_comfy_commit="a" * 40,
        tunnel_configured=True,
        web_ui_configured=True,
        any_configured=True,
        has_db_override=True,
    )
    monkeypatch.setattr(native_config, "resolve", lambda *a, **kw: fake_cfg)
    monkeypatch.setattr(native, "backend_for", lambda *a, **kw: ("https://internal.example.test", {"CF-Access-Client-Id": "cf-id", "CF-Access-Client-Secret": "cf-sec"}))
    monkeypatch.setattr(native, "ready_for", lambda *a, **kw: True)
    if mock_sync:
        monkeypatch.setattr(native, "sync_plugins", lambda *a, **kw: None)


# ---------------------------------------------------------------------------
# 1. Default-off safety (GPU_AUTO_RECOVERY_ENABLED=0)
# ---------------------------------------------------------------------------

def test_default_disabled_zero_side_effects_native_and_creation(tmp_path, monkeypatch):
    monkeypatch.setenv("GPU_REAL_ENABLED", "1")
    monkeypatch.setenv("GPU_AUTO_RECOVERY_ENABLED", "0")
    repo = _make_repo(tmp_path)
    clock = FakeClock()
    _seed_project(repo)
    _seed_account(repo)

    _, inst_native, lease_native = _seed_running_instance(repo, clock, pool=POOL_NATIVE)
    _, inst_cre, lease_cre = _seed_running_instance(
        repo, clock, pool=POOL_CREATION, retention_mode="smart",
        creation_runtime={"workflow_key": "txt2img", "workflow_version": 1, "process": "p1"},
    )
    provider = MockColab(inspect_map={
        lease_native["runtime_id"]: {"deleted": True, "missing": True},
        lease_cre["runtime_id"]: {"deleted": True, "missing": True},
    })
    sched = Scheduler(repo, provider_factory=lambda job: provider, executor_factory=lambda job: MockExecutor(), clock=clock)
    res = sched.watchdog()

    assert res["recovered"] >= 2
    assert repo.get("instances", inst_native["id"], "owner-1")["state"] == "deleted"
    assert repo.get("instances", inst_cre["id"], "owner-1")["state"] == "deleted"
    # Zero recovery plans created when disabled by default
    assert recovery.list_plans(repo, "owner-1") == []


# ---------------------------------------------------------------------------
# 2. L0 Active Keep-Alive (TFE-only, no WS 1+1) & Accelerated Inspection Hint
# ---------------------------------------------------------------------------

def test_l0_active_keepalive_tfe_only_and_ws_not_touched(tmp_path, monkeypatch):
    monkeypatch.setenv("GPU_ACTIVE_KEEPALIVE", "1")
    repo = _make_repo(tmp_path)
    clock = FakeClock()
    _seed_project(repo)
    _seed_account(repo)

    # First verify GoogleColabProvider.keep_alive(..., tfe_only=True) sends ONLY TFE keep-alive (no proxy kernels/WS).
    seen_urls: list[str] = []

    def transport_handler(req: httpx.Request) -> httpx.Response:
        seen_urls.append(str(req.url))
        return httpx.Response(200, text="ok")

    prov = GoogleColabProvider.__new__(GoogleColabProvider)
    prov.google = SimpleNamespace(access_token=lambda purpose="colab": "fake-access-token")
    prov._is_cli_tunnel = lambda lease: True
    prov.client = httpx.Client(transport=httpx.MockTransport(transport_handler))
    prov._resolve_proxy_credentials = lambda *a, **kw: (_ for _ in ()).throw(
        AssertionError("tfe_only must NOT call _resolve_proxy_credentials or WS")
    )
    res = prov.keep_alive({"tunnel_endpoint": "ep-abc"}, tfe_only=True)
    assert res["ok"] is True
    assert res["tfe"].startswith("ok:")
    assert res["proxy"] == "skipped"
    assert res["kernel_exec"] == "skipped"
    assert len(seen_urls) == 1
    assert "/tun/m/ep-abc/keep-alive/" in seen_urls[0]

    # Now verify watchdog triggers keepalive_active_lease every 60s for active Colab leases.
    _, _, lease = _seed_running_instance(
        repo, clock, pool=POOL_CREATION, instance_state="busy", lease_state="active", job_state="running",
        retention_mode="smart",
    )
    mock_prov = MockColab(inspect_map={lease["runtime_id"]: {"ready": True, "deleted": False, "state": "RUNNING"}})
    sched = Scheduler(repo, provider_factory=lambda job: mock_prov, executor_factory=lambda job: MockExecutor(), clock=clock)
    sched.watchdog()
    assert len(mock_prov.keepalive_calls) == 1
    assert mock_prov.keepalive_calls[0]["tfe_only"] is True
    # Within 60s -> throttled
    clock.advance(30)
    sched.watchdog()
    assert len(mock_prov.keepalive_calls) == 1
    # After 60s -> fires again
    clock.advance(35)
    sched.watchdog()
    assert len(mock_prov.keepalive_calls) == 2

    # When GPU_ACTIVE_KEEPALIVE=0 -> skipped
    monkeypatch.setenv("GPU_ACTIVE_KEEPALIVE", "0")
    clock.advance(70)
    sched.watchdog()
    assert len(mock_prov.keepalive_calls) == 2


def test_l0_proxy_failures_trigger_accelerated_runtime_check_hint(tmp_path):
    repo = _make_repo(tmp_path)
    clock = FakeClock()
    _seed_project(repo)
    _seed_account(repo)
    _, _, lease = _seed_running_instance(repo, clock, pool=POOL_NATIVE)

    # First inspection at t=0 caches "present" for 45s (next_check_at = t+45).
    mock_prov = MockColab(inspect_map={lease["runtime_id"]: {"ready": True, "deleted": False, "state": "RUNNING"}})
    sched = Scheduler(repo, provider_factory=lambda job: mock_prov, executor_factory=lambda job: MockExecutor(), clock=clock)
    assert sched._inspect_runtime(repo.get("leases", lease["id"], "owner-1")) == "present"
    assert len(mock_prov.inspect_calls) == 1

    # Simulate 3 consecutive upstream 502 failures within 10s via NativeGateway
    gw = NativeGateway(SimpleNamespace(state=SimpleNamespace(repo=repo)))
    for _ in range(3):
        gw._note_upstream_failure(repo, lease)
    updated_lease = repo.get("leases", lease["id"], "owner-1")
    assert updated_lease.get("provider_runtime_check_hint_at") is not None

    # Advance 12s (< 45s normal throttle, >= 10s hint min interval) -> triggers immediate re-probe!
    clock.advance(12)
    # Ensure hint timestamp > previous checked_at (real clock vs fake clock alignment)
    repo.update("leases", lease["id"], {"provider_runtime_check_hint_at": clock()}, "owner-1")
    sched._inspect_runtime(repo.get("leases", lease["id"], "owner-1"))
    assert len(mock_prov.inspect_calls) == 2


# ---------------------------------------------------------------------------
# 3. L1/L2 In-VM Self-Heal (Native Watchdog + Creation L2)
# ---------------------------------------------------------------------------

def _load_watchdog_module(tmp_path: Path):
    ns: dict = {}
    exec(compile(WATCHDOG, "<studio_watchdog>", "exec"), ns)
    return ns


def test_l1_native_watchdog_comfy_and_rclone_rules(tmp_path, monkeypatch):
    wd = _load_watchdog_module(tmp_path)
    root = tmp_path / "native-root"
    root.mkdir()
    drive = root / "drive"
    drive.mkdir()
    state_file = root / "state.json"

    launches: list[list[str]] = []
    unmounts: list[str] = []

    class FakeProc:
        def __init__(self, pid):
            self.pid = pid

    def fake_popen(args, **kw):
        launches.append(list(args))
        return FakeProc(9000 + len(launches))

    def fake_run(args, **kw):
        unmounts.append(str(args))
        return SimpleNamespace(returncode=0)

    # State: ComfyUI alive, rclone alive, tunnel alive
    state = {
        "tunnel_pid": 3333,
        "tunnel_start_id": "300",
        "tunnel_binary": "/usr/bin/cloudflared",
        "pid": 1111,
        "comfy_start_id": "100",
        "comfy_args": ["python3", "main.py", "--port", "8188"],
        "comfy_cwd": str(root),
        "rclone_pid": 2222,
        "rclone_start_id": "200",
        "rclone_args": ["rclone", "mount", "studio:", str(drive)],
        "mount_path": str(drive),
        "self_heal_enabled": True,
    }
    state_file.write_text(json.dumps(state))

    # Case A: Alive but slow (HTTP 8188 timeout, RC 5572 timeout) -> MUST NOT kill/restart/unmount!
    alive_set = {(1111, "100"), (2222, "200"), (3333, "300")}
    wd["proc_alive"] = lambda pid, sid=None, **kw: (pid, sid) in alive_set
    wd["proc_start_id"] = lambda pid, **kw: "999"

    now_box = {"t": 1000.0}
    stats_ready = {"ok": False}

    def run_cycle():
        return wd["heal_cycle"](
            root,
            clock=lambda: now_box["t"],
            sleep_fn=lambda s: None,
            popen_fn=fake_popen,
            run_fn=fake_run,
            ismount_fn=lambda p: True,
            http_ok_fn=lambda url, **kw: stats_ready["ok"] if "8188" in url else False,
        )
    status = run_cycle()
    assert launches == []
    assert unmounts == []
    assert "comfy" in status["slow_warnings"]
    assert "rclone" in status["slow_warnings"]

    # Case B: reboot-active present while ComfyUI PID dies -> MUST exempt restart!
    (root / "reboot-active").write_text(json.dumps({"ts": 1000.0}))
    alive_set.discard((1111, "100"))
    now_box["t"] = 1015.0
    run_cycle()
    assert launches == []
    assert wd["is_reboot_active"](root, now=1015.0) is True
    (root / "reboot-active").unlink()

    # Case C: Dead ComfyUI (without reboot-active, port 8188 down -> becomes ready after launch) -> restarts cleanly!
    now_box["t"] = 1030.0
    def fake_popen_ready(args, **kw):
        p = fake_popen(args, **kw)
        if args and args[0] == "python3":
            stats_ready["ok"] = True
        return p

    status = wd["heal_cycle"](
        root,
        clock=lambda: now_box["t"],
        sleep_fn=lambda s: None,
        popen_fn=fake_popen_ready,
        run_fn=fake_run,
        ismount_fn=lambda p: True,
        http_ok_fn=lambda url, **kw: stats_ready["ok"] if "8188" in url else False,
    )
    assert len(launches) == 1
    assert launches[0][:2] == ["python3", "main.py"]
    assert status["comfy_restarts"] == 1
    assert (root / "self-heal-status.json").is_file()

    # Case D: >3 ComfyUI crashes in 10 min -> stops restarting & sets unrecoverable=True
    stats_ready["ok"] = False
    saved_status = json.loads((root / "self-heal-status.json").read_text())
    saved_status["history"]["comfy"] = [1000.0, 1020.0, 1040.0]
    (root / "self-heal-status.json").write_text(json.dumps(saved_status))
    now_box["t"] = 1060.0
    status = run_cycle()
    assert len(launches) == 1  # no 4th restart
    assert status["unrecoverable"] is True

    # Case E: Dead rclone -> fusermount3 -uz + remount!
    alive_set.discard((2222, "200"))
    saved_status = json.loads((root / "self-heal-status.json").read_text())
    saved_status["unrecoverable"] = False
    saved_status["history"]["comfy"] = []
    (root / "self-heal-status.json").write_text(json.dumps(saved_status))
    cur_state = json.loads(state_file.read_text())
    cur_state["pid"] = 9001
    cur_state["comfy_start_id"] = "999"
    state_file.write_text(json.dumps(cur_state))
    alive_set.add((9001, "999"))
    now_box["t"] = 1080.0
    wd["heal_cycle"](
        root,
        clock=lambda: now_box["t"],
        sleep_fn=lambda s: None,
        popen_fn=fake_popen,
        run_fn=fake_run,
        ismount_fn=lambda p: True,
        http_ok_fn=lambda url, **kw: "8188" in url,
    )
    assert any("fusermount3" in cmd for cmd in unmounts)
    assert any(cmd[0] == "rclone" for cmd in launches)


def test_l2_creation_in_vm_retry_and_idle_process_probe(tmp_path, monkeypatch):
    monkeypatch.setenv("GPU_REAL_ENABLED", "1")
    monkeypatch.setenv("GPU_IN_VM_SELF_HEAL", "1")
    monkeypatch.setenv("GPU_AUTO_RECOVERY_ENABLED", "0")
    repo = _make_repo(tmp_path)
    clock = FakeClock()
    _seed_project(repo)
    _seed_account(repo)

    # Part 1: Running creation smart job fails with gpu_worker_exited while VM present -> retries once in-VM
    job, inst, lease = _seed_running_instance(
        repo, clock, pool=POOL_CREATION, instance_state="busy", lease_state="active", job_state="running",
        retention_mode="smart",
        creation_runtime={"workflow_key": "txt2img", "workflow_version": 1, "process": "pid-99", "fingerprint": "fp-abc"},
        job_extra={"execution_id": "exec-1"},
    )
    provider = MockColab(inspect_map={lease["runtime_id"]: {"ready": True, "deleted": False, "state": "RUNNING"}})
    executor = MockExecutor(poll_plan=[
        {"state": "failed", "execution_id": "exec-1", "error_code": "gpu_worker_exited", "message": "worker crashed"},
    ])
    sched = Scheduler(repo, provider_factory=lambda j: provider, executor_factory=lambda j: executor, clock=clock)
    updated = sched._run(repo.get("jobs", job["id"], "owner-1"))
    assert updated["state"] == "loading"
    assert updated["auto_retry_count"] == 1
    assert len(updated["attempt_history"]) == 1
    assert repo.get("instances", inst["id"], "owner-1")["creation_runtime"]["process"] is None

    # Deterministic OOM must NEVER retry
    job_oom, _, lease_oom = _seed_running_instance(
        repo, clock, pool=POOL_CREATION, instance_state="busy", lease_state="active", job_state="running",
        retention_mode="smart", job_extra={"execution_id": "exec-2"},
    )
    executor_oom = MockExecutor(poll_plan=[
        {"state": "failed", "execution_id": "exec-2", "error_code": "gpu_worker_exited", "message": "torch.cuda.OutOfMemoryError"},
    ])
    sched_oom = Scheduler(repo, provider_factory=lambda j: provider, executor_factory=lambda j: executor_oom, clock=clock)
    res_oom = sched_oom._run(repo.get("jobs", job_oom["id"], "owner-1"))
    assert res_oom["state"] == "cleaning_up"
    assert int(res_oom.get("auto_retry_count") or 0) == 0

    # Part 2: Idle retained_idle probe -> 2 consecutive failures clears process while keeping workflow metadata
    _, inst_idle, lease_idle = _seed_running_instance(
        repo, clock, pool=POOL_CREATION, instance_state="retained_idle", lease_state="idle", job_state="completed",
        retention_mode="smart",
        creation_runtime={
            "lease_id": None,  # patched below
            "workflow_key": "txt2img",
            "workflow_version": 1,
            "fingerprint": "fp-abc",
            "process": "pid-77",
            "warmup_completed": False,
            "health_checked_at": clock() - 60.0,
        },
    )
    rt = {**inst_idle["creation_runtime"], "lease_id": lease_idle["id"]}
    inst_idle = repo.update("instances", inst_idle["id"], {"creation_runtime": rt}, "owner-1")
    exec_probe = MockExecutor(health_plan=[False, False])
    sched_probe = Scheduler(repo, provider_factory=lambda j: provider, executor_factory=lambda j: exec_probe, clock=clock)
    probe_job = {"id": lease_idle["owner_job_id"], "owner_id": "owner-1", "provider": "colab", "account_id": "acc-1"}

    sched_probe._probe_creation_process(probe_job, inst_idle, lease_idle, observation="present")
    after1 = repo.get("instances", inst_idle["id"], "owner-1")
    assert after1["creation_runtime"]["process"] == "pid-77"
    assert after1["creation_runtime"]["health_failures"] == 1

    clock.advance(50)
    sched_probe._probe_creation_process(probe_job, after1, lease_idle, observation="present")
    after2 = repo.get("instances", inst_idle["id"], "owner-1")
    assert after2["creation_runtime"]["process"] is None
    assert after2["creation_runtime"]["status"] == "process_exited"
    assert after2["creation_runtime"]["workflow_key"] == "txt2img"
    assert after2["creation_runtime"]["fingerprint"] == "fp-abc"


# ---------------------------------------------------------------------------
# 4. L3 Double-Confirm & All Hard Vetoes
# ---------------------------------------------------------------------------

def test_l3_double_confirm_missing_and_native_proxy_guard(tmp_path, monkeypatch):
    _enable_recovery(monkeypatch)
    repo = _make_repo(tmp_path)
    clock = FakeClock()
    _seed_project(repo)
    _seed_account(repo)
    _, inst, lease = _seed_running_instance(repo, clock, pool=POOL_NATIVE)

    # 1st missing -> missing_pending; 2nd present -> stays alive!
    responses = [{"deleted": True, "missing": True}, {"ready": True, "deleted": False, "state": "RUNNING"}]
    prov = MockColab(inspect_map={lease["runtime_id"]: lambda h: responses.pop(0)})
    sched = Scheduler(repo, provider_factory=lambda j: prov, executor_factory=lambda j: MockExecutor(), clock=clock)

    assert sched._inspect_runtime(repo.get("leases", lease["id"], "owner-1")) == "missing_pending"
    assert repo.get("instances", inst["id"], "owner-1")["state"] != "deleted"
    assert recovery.list_plans(repo, "owner-1") == []

    clock.advance(6)
    assert sched._inspect_runtime(repo.get("leases", lease["id"], "owner-1")) == "present"
    assert repo.get("instances", inst["id"], "owner-1")["state"] != "deleted"
    assert repo.get("leases", lease["id"], "owner-1")["provider_runtime_check"].get("missing_first_seen_at") is None

    # Native proxy guard: if proxy still answers 200 on 2nd missing, downgrade to unknown (do NOT kill)
    monkeypatch.setattr(native, "native_proxy_healthy", lambda r, o: True)
    responses[:] = [{"deleted": True, "missing": True}, {"deleted": True, "missing": True}]
    clock.advance(50)
    assert sched._inspect_runtime(repo.get("leases", lease["id"], "owner-1")) == "missing_pending"
    clock.advance(6)
    assert sched._inspect_runtime(repo.get("leases", lease["id"], "owner-1")) == "unknown"
    assert repo.get("instances", inst["id"], "owner-1")["state"] != "deleted"

    # Now proxy is genuinely down -> 2nd missing confirms reclaim & creates scheduled plan!
    monkeypatch.setattr(native, "native_proxy_healthy", lambda r, o: False)
    responses[:] = [{"deleted": True, "missing": True}, {"deleted": True, "missing": True}]
    clock.advance(100)
    assert sched._inspect_runtime(repo.get("leases", lease["id"], "owner-1")) == "missing_pending"
    clock.advance(6)
    assert sched._inspect_runtime(repo.get("leases", lease["id"], "owner-1")) == "missing"
    assert repo.get("instances", inst["id"], "owner-1")["state"] == "deleted"
    plans = recovery.list_plans(repo, "owner-1")
    assert len(plans) == 1
    assert plans[0]["state"] == "scheduled"
    assert plans[0]["pool"] == POOL_NATIVE


def test_l3_hard_vetoes_all_rules(tmp_path, monkeypatch):
    _enable_recovery(monkeypatch, max_rounds=3, min_remaining=15, max_vps_outage=18)
    monkeypatch.setattr(native, "native_proxy_healthy", lambda r, o: False)
    repo = _make_repo(tmp_path)
    clock = FakeClock()
    _seed_project(repo)
    _seed_account(repo)
    _seed_published_workflow(repo)

    def reclaim_two_step(inst_row, lease_row, sched):
        prov_map = {lease_row["runtime_id"]: {"deleted": True, "missing": True}}
        sched.provider_factory = lambda j: MockColab(inspect_map=prov_map)
        r1 = sched._inspect_runtime(repo.get("leases", lease_row["id"], "owner-1"))
        if r1 == "missing_pending":
            clock.advance(6)
            sched._inspect_runtime(repo.get("leases", lease_row["id"], "owner-1"))

    sched = Scheduler(repo, provider_factory=lambda j: MockColab(), executor_factory=lambda j: MockExecutor(), clock=clock)

    # 1) Ephemeral pool -> no plan
    _, inst_e, lease_e = _seed_running_instance(repo, clock, pool=POOL_EPHEMERAL, retention_mode="save")
    reclaim_two_step(inst_e, lease_e, sched)
    assert recovery.list_plans(repo, "owner-1") == []

    # 2) Creation save & batch -> no plan
    for lc in ("save", "batch"):
        _, inst_c, lease_c = _seed_running_instance(repo, clock, pool=POOL_CREATION, retention_mode=lc)
        reclaim_two_step(inst_c, lease_c, sched)
    assert recovery.list_plans(repo, "owner-1") == []

    # 3) Manual stop requested -> no plan
    _, inst_m, lease_m = _seed_running_instance(repo, clock, pool=POOL_NATIVE, manual_stop_requested=True)
    reclaim_two_step(inst_m, lease_m, sched)
    assert recovery.list_plans(repo, "owner-1") == []

    # 4) Remaining lifetime < 15 min -> aborted_fatal
    _, inst_rem, lease_rem = _seed_running_instance(repo, clock, pool=POOL_NATIVE, hard_deadline_at=clock() + 10 * 60)
    reclaim_two_step(inst_rem, lease_rem, sched)
    p_rem = recovery.get_plan(repo, "owner-1", recovery.make_group_id(inst_rem["id"]))
    assert p_rem["state"] == "aborted_fatal"

    # 5) VPS outage >= 18 min -> aborted_vps_outage
    _, inst_vps, lease_vps = _seed_running_instance(
        repo, clock, pool=POOL_NATIVE, last_present_at=clock() - 20 * 60,
    )
    reclaim_two_step(inst_vps, lease_vps, sched)
    p_vps = recovery.get_plan(repo, "owner-1", recovery.make_group_id(inst_vps["id"]))
    assert p_vps["state"] == "aborted_vps_outage"

    # 6) Max rounds (3/3) already used -> exhausted
    gid = "rec-maxrounds01"
    recovery.save_plan(repo, "owner-1", {
        "plan_id": gid, "recovery_group_id": gid, "owner_id": "owner-1", "pool": POOL_NATIVE,
        "state": "recovered", "account_id": ACC1, "requested_spec": SPEC_L4,
        "rounds_used": 3, "max_rounds": 3, "hard_deadline_at": clock() + 3600,
    })
    _, inst_r, lease_r = _seed_running_instance(
        repo, clock, pool=POOL_NATIVE, instance_extra={"recovery_group_id": gid},
    )
    reclaim_two_step(inst_r, lease_r, sched)
    assert recovery.get_plan(repo, "owner-1", gid)["state"] == "exhausted"

    # 7) CU soft fuse tripped -> aborted_fatal
    from app.gpu.cu_guard import CuGuardService

    monkeypatch.setattr(
        CuGuardService,
        "state_in",
        lambda self, session, owner: {"soft_tripped": True, "daily_message": "今日上限已达", "hard_tripped_accounts": []},
    )
    _, inst_cu, lease_cu = _seed_running_instance(repo, clock, pool=POOL_NATIVE)
    reclaim_two_step(inst_cu, lease_cu, sched)
    p_cu = recovery.get_plan(repo, "owner-1", recovery.make_group_id(inst_cu["id"]))
    assert p_cu["state"] == "aborted_fatal"


# ---------------------------------------------------------------------------
# 5 & 6. Single-Round Backoff, Exact Account/Spec, Fatal Fast-Abort & Exact Clock Inheritance
# ---------------------------------------------------------------------------

def test_l3_single_round_backoff_account_spec_and_exact_clock_inheritance(tmp_path, monkeypatch):
    _enable_recovery(monkeypatch, max_rounds=3, round_timeout=600)
    _configure_native_mock(monkeypatch)
    monkeypatch.setattr(native, "native_proxy_healthy", lambda r, o: False)
    repo = _make_repo(tmp_path)
    clock = FakeClock()
    _seed_project(repo)
    _seed_account(repo, account_id=ACC1, alias="主号")
    _seed_account(repo, account_id=ACC2, alias="副号")

    # Exact non-integer hard deadline & idle deadline (stays >15m across 30s + 60s backoff)
    exact_hard = clock() + 27 * 60 + 23.4
    exact_idle = clock() + 20 * 60 + 11.2
    _, inst, lease = _seed_running_instance(
        repo, clock, pool=POOL_NATIVE,
        hard_deadline_at=exact_hard, idle_deadline_at=exact_idle,
        spec=SPEC_L4,
    )
    # Reclaim with double-confirm
    inspect_map = {lease["runtime_id"]: {"deleted": True, "missing": True}}
    create_plan = [
        ColabAllocationError("gpu_capacity_unavailable", "NO_RUNTIMES", error_code="gpu_capacity_unavailable", remote_state_unknown=False),
        ColabAllocationError("gpu_capacity_unavailable", "NO_RUNTIMES", error_code="gpu_capacity_unavailable", remote_state_unknown=False),
        {"runtime_name": "runtimes/succ-1", "endpoint": "ep-succ1", "runtime_spec": {"key": SPEC_L4}},
    ]
    prov = MockColab(inspect_map=inspect_map, create_plan=create_plan)
    exec_mock = MockExecutor(poll_plan=[{"state": "completed", "outputs": [{"filename": "out.png", "content": VALID_PNG_1X1, "mime": "image/png"}]}])
    sched = Scheduler(repo, provider_factory=lambda j: prov, executor_factory=lambda j: exec_mock, clock=clock,
                      config=GpuConfig(enabled=True, real_enabled=True, create_retries=0))

    assert sched._inspect_runtime(repo.get("leases", lease["id"], "owner-1")) == "missing_pending"
    clock.advance(6)
    assert sched._inspect_runtime(repo.get("leases", lease["id"], "owner-1")) == "missing"

    gid = recovery.make_group_id(inst["id"])
    plan = recovery.get_plan(repo, "owner-1", gid)
    assert plan["state"] == "scheduled"
    assert plan["rounds_used"] == 0

    def drive_until_idle(max_steps=20):
        for _ in range(max_steps):
            res = sched.tick()
            if not res.get("advanced"):
                break

    # Attempt 1 -> NO_RUNTIMES -> backoff 30s, rounds_used STILL 0!
    recovery.tick(repo, clock=clock, scheduler=sched, process_started_at=clock() - 100)
    drive_until_idle()
    recovery.tick(repo, clock=clock, scheduler=sched, process_started_at=clock() - 100)
    plan = recovery.get_plan(repo, "owner-1", gid)
    assert plan["state"] == "scheduled"
    assert plan["rounds_used"] == 0
    assert plan["backoff_attempt"] == 1
    assert pytest.approx(plan["next_retry_at"] - clock(), abs=1.0) == 30.0

    # Attempt 2 -> NO_RUNTIMES -> backoff 60s, rounds_used STILL 0!
    clock.advance(31)
    recovery.tick(repo, clock=clock, scheduler=sched, process_started_at=clock() - 100)
    drive_until_idle()
    recovery.tick(repo, clock=clock, scheduler=sched, process_started_at=clock() - 100)
    plan = recovery.get_plan(repo, "owner-1", gid)
    assert plan["state"] == "scheduled"
    assert plan["rounds_used"] == 0
    assert plan["backoff_attempt"] == 2
    assert pytest.approx(plan["next_retry_at"] - clock(), abs=1.0) == 60.0

    # Attempt 3 -> succeeds! Deducts exactly 1 round (1/3), exact original account & spec, exact inherited clocks!
    clock.advance(61)
    recovery.tick(repo, clock=clock, scheduler=sched, process_started_at=clock() - 100)
    drive_until_idle(30)
    recovery.tick(repo, clock=clock, scheduler=sched, process_started_at=clock() - 100)
    plan = recovery.get_plan(repo, "owner-1", gid)
    assert plan["state"] == "recovered"
    assert plan["rounds_used"] == 1
    assert len(prov.create_calls) == 3
    assert all(call["spec"]["key"] == SPEC_L4 for call in prov.create_calls)

    succ_inst = repo.get("instances", plan["successor_instance_id"], "owner-1")
    succ_lease = repo.get("leases", succ_inst["lease_id"], "owner-1")
    assert succ_inst["account_id"] == ACC1
    assert succ_inst["parent_instance_id"] == inst["id"]
    assert succ_inst["recovery_source"] == "auto_recovery"
    assert succ_inst["recovery_generation"] == 1
    assert repo.get("instances", inst["id"], "owner-1")["replaced_by_instance_id"] == succ_inst["id"]
    # Exact float deadline preserved (no ceil expansion!) and idle deadline NOT extended!
    assert succ_inst["hard_deadline_at"] == pytest.approx(exact_hard, abs=1e-3)
    assert succ_lease["deadline_epoch"] == pytest.approx(exact_hard, abs=1e-3)
    assert succ_inst["idle_deadline_at"] == pytest.approx(exact_idle, abs=1e-3)
    assert succ_lease["idle_deadline_at"] == pytest.approx(exact_idle, abs=1e-3)


def test_l3_fatal_error_aborts_immediately_and_round_timeout_exhausts(tmp_path, monkeypatch):
    _enable_recovery(monkeypatch, round_timeout=120)
    _configure_native_mock(monkeypatch)
    monkeypatch.setattr(native, "native_proxy_healthy", lambda r, o: False)
    repo = _make_repo(tmp_path)
    clock = FakeClock()
    _seed_project(repo)
    _seed_account(repo)

    # Fatal DENYLISTED aborts immediately with 0 retries
    _, inst1, lease1 = _seed_running_instance(repo, clock, pool=POOL_NATIVE)
    prov1 = MockColab(
        inspect_map={lease1["runtime_id"]: {"deleted": True, "missing": True}},
        create_plan=[ColabAllocationError("gpu_allocation_failed", "DENYLISTED", error_code="gpu_allocation_failed", outcome="DENYLISTED", remote_state_unknown=False)],
    )
    sched1 = Scheduler(repo, provider_factory=lambda j: prov1, executor_factory=lambda j: MockExecutor(), clock=clock,
                       config=GpuConfig(enabled=True, real_enabled=True, create_retries=0))
    sched1._inspect_runtime(repo.get("leases", lease1["id"], "owner-1"))
    clock.advance(6)
    sched1._inspect_runtime(repo.get("leases", lease1["id"], "owner-1"))
    gid1 = recovery.make_group_id(inst1["id"])
    recovery.tick(repo, clock=clock, scheduler=sched1, process_started_at=clock() - 100)
    for _ in range(15):
        if not sched1.tick().get("advanced"):
            break
    recovery.tick(repo, clock=clock, scheduler=sched1, process_started_at=clock() - 100)
    assert recovery.get_plan(repo, "owner-1", gid1)["state"] == "aborted_fatal"
    assert len(prov1.create_calls) == 1

    # Single-round backoff timeout > round_timeout_seconds -> exhausted
    _, inst2, lease2 = _seed_running_instance(repo, clock, pool=POOL_NATIVE)
    prov2 = MockColab(
        inspect_map={lease2["runtime_id"]: {"deleted": True, "missing": True}},
        create_plan=[ColabAllocationError("gpu_capacity_unavailable", "NO_RUNTIMES", error_code="gpu_capacity_unavailable", remote_state_unknown=False)],
    )
    sched2 = Scheduler(repo, provider_factory=lambda j: prov2, executor_factory=lambda j: MockExecutor(), clock=clock,
                       config=GpuConfig(enabled=True, real_enabled=True, create_retries=0))
    sched2._inspect_runtime(repo.get("leases", lease2["id"], "owner-1"))
    clock.advance(6)
    sched2._inspect_runtime(repo.get("leases", lease2["id"], "owner-1"))
    gid2 = recovery.make_group_id(inst2["id"])
    clock.advance(125)  # exceed 120s round timeout
    recovery.tick(repo, clock=clock, scheduler=sched2, process_started_at=clock() - 100)
    assert recovery.get_plan(repo, "owner-1", gid2)["state"] == "exhausted"


# ---------------------------------------------------------------------------
# 7. Creation Pool Path A (Idle Warmup), Path B (Busy Atomic Takeover), & Uploading
# ---------------------------------------------------------------------------

def test_l3_creation_path_a_idle_warmup_and_separate_cu_segment(tmp_path, monkeypatch):
    _enable_recovery(monkeypatch)
    repo = _make_repo(tmp_path)
    clock = FakeClock()
    _seed_project(repo)
    _seed_account(repo)
    _seed_published_workflow(repo)

    exact_hard = clock() + 3600.0
    exact_idle = clock() + 1800.0
    _, inst, lease = _seed_running_instance(
        repo, clock, pool=POOL_CREATION, instance_state="retained_idle", lease_state="idle", job_state="completed",
        retention_mode="smart", hard_deadline_at=exact_hard, idle_deadline_at=exact_idle,
        creation_runtime={"workflow_id": "wf-1", "workflow_key": "txt2img", "workflow_version": 1, "fingerprint": "fp-abc", "process": "pid-1"},
    )
    prov = MockColab(inspect_map={lease["runtime_id"]: {"deleted": True, "missing": True}})
    sched = Scheduler(repo, provider_factory=lambda j: prov, executor_factory=lambda j: MockExecutor(), clock=clock,
                      config=GpuConfig(enabled=True, real_enabled=True, create_retries=0))

    sched._inspect_runtime(repo.get("leases", lease["id"], "owner-1"))
    clock.advance(6)
    sched._inspect_runtime(repo.get("leases", lease["id"], "owner-1"))
    gid = recovery.make_group_id(inst["id"])
    plan = recovery.get_plan(repo, "owner-1", gid)
    assert plan["state"] == "scheduled"
    assert plan["creation_snapshot"]["mode"] == "warmup"

    recovery.tick(repo, clock=clock, scheduler=sched, process_started_at=clock() - 100)
    for _ in range(25):
        if not sched.tick().get("advanced"):
            break
    recovery.tick(repo, clock=clock, scheduler=sched, process_started_at=clock() - 100)
    plan = recovery.get_plan(repo, "owner-1", gid)
    assert plan["state"] == "recovered"
    assert plan["rounds_used"] == 1
    succ = repo.get("instances", plan["successor_instance_id"], "owner-1")
    assert succ["id"] != inst["id"]
    assert succ["state"] == "retained_idle"
    assert succ["idle_deadline_at"] == pytest.approx(exact_idle, abs=1e-3)
    # Separate CU sessions for old and new physical instances
    summary = ccu.usage_summary(repo, "owner-1")
    sessions = [*(summary.get("active_sessions") or []), *(summary.get("recent_sessions") or [])]
    assert {s["instance_id"] for s in sessions} >= {inst["id"], succ["id"]}
    old_sess = next(s for s in sessions if s["instance_id"] == inst["id"])
    assert old_sess["state"] in {"closed", "reconciling"}


def test_l3_creation_path_b_atomic_takeover_history_and_retry_limit(tmp_path, monkeypatch):
    _enable_recovery(monkeypatch, max_job_retries=1)
    repo = _make_repo(tmp_path)
    clock = FakeClock()
    _seed_project(repo)
    _seed_account(repo)
    _seed_published_workflow(repo)

    job, inst, lease = _seed_running_instance(
        repo, clock, pool=POOL_CREATION, instance_state="busy", lease_state="active", job_state="running",
        retention_mode="smart",
        job_extra={"execution_id": "exec-initial", "attempt_number": 1},
    )
    prov = MockColab(inspect_map={lease["runtime_id"]: {"deleted": True, "missing": True}})
    sched = Scheduler(repo, provider_factory=lambda j: prov, executor_factory=lambda j: MockExecutor(), clock=clock,
                      config=GpuConfig(enabled=True, real_enabled=True, create_retries=0))

    sched._inspect_runtime(repo.get("leases", lease["id"], "owner-1"))
    clock.advance(6)
    sched._inspect_runtime(repo.get("leases", lease["id"], "owner-1"))

    # Immediately after TX commits: job NEVER hit terminal failed -> /history/{prompt_id} stays {}!
    from app import creation_api

    mid_job = repo.get("jobs", job["id"], "owner-1")
    assert mid_job["state"] == "waiting_for_capacity"
    assert mid_job["recovery_hold"] is True
    assert mid_job["auto_retry_count"] == 1
    assert creation_api.is_active(mid_job) is True
    assert mid_job["attempt_number"] == 1  # NOT mutated in TX; incremented cleanly by prepare_execution!

    # Release hold and run on successor VM -> attempt_number becomes 2, job completes!
    recovery.tick(repo, clock=clock, scheduler=sched, process_started_at=clock() - 100)
    for _ in range(25):
        if not sched.tick().get("advanced"):
            break
    recovery.tick(repo, clock=clock, scheduler=sched, process_started_at=clock() - 100)
    done_job = repo.get("jobs", job["id"], "owner-1")
    assert done_job["state"] == "completed"
    assert done_job["attempt_number"] == 2
    assert done_job["instance_id"] != inst["id"]

    # Now test 2nd reclaim when auto_retry_count == max_job_retries (1) -> stops retrying & fails cleanly
    job2, inst2, lease2 = _seed_running_instance(
        repo, clock, pool=POOL_CREATION, instance_state="busy", lease_state="active", job_state="running",
        retention_mode="smart", job_extra={"auto_retry_count": 1},
    )
    prov2 = MockColab(inspect_map={lease2["runtime_id"]: {"deleted": True, "missing": True}})
    sched2 = Scheduler(repo, provider_factory=lambda j: prov2, executor_factory=lambda j: MockExecutor(), clock=clock)
    sched2._inspect_runtime(repo.get("leases", lease2["id"], "owner-1"))
    clock.advance(6)
    sched2._inspect_runtime(repo.get("leases", lease2["id"], "owner-1"))
    failed_job = repo.get("jobs", job2["id"], "owner-1")
    assert failed_job["state"] == "cleaning_up"
    assert failed_job["cleanup_final_state"] == "failed"


def test_l3_creation_uploading_with_persisted_local_outputs_not_rerun(tmp_path, monkeypatch):
    _enable_recovery(monkeypatch)
    repo = _make_repo(tmp_path)
    clock = FakeClock()
    _seed_project(repo)
    _seed_account(repo)

    job, inst, lease = _seed_running_instance(
        repo, clock, pool=POOL_CREATION, instance_state="busy", lease_state="active", job_state="uploading",
        retention_mode="smart",
        job_extra={"outputs": [{"filename": "img.png", "content": "aGVsbG8=", "content_encoding": "base64"}]},
    )
    prov = MockColab(inspect_map={lease["runtime_id"]: {"deleted": True, "missing": True}})
    sched = Scheduler(repo, provider_factory=lambda j: prov, executor_factory=lambda j: MockExecutor(), clock=clock)
    sched._inspect_runtime(repo.get("leases", lease["id"], "owner-1"))
    clock.advance(6)
    sched._inspect_runtime(repo.get("leases", lease["id"], "owner-1"))

    after = repo.get("jobs", job["id"], "owner-1")
    assert after["state"] == "uploading"
    assert recovery.list_plans(repo, "owner-1") == []


# ---------------------------------------------------------------------------
# 8. Native local Model Snapshot Incremental Sync, Background Restore & 5 GiB Limit
# ---------------------------------------------------------------------------

def test_native_local_models_inventory_digest_restore_and_5gib_limit(tmp_path, monkeypatch):
    root = tmp_path / "vm-root"
    root.mkdir()
    model_dir = root / "models"
    (model_dir / "checkpoints").mkdir(parents=True)
    (model_dir / "loras").mkdir(parents=True)

    f1 = model_dir / "checkpoints" / "recent.safetensors"
    f2 = model_dir / "checkpoints" / "older.safetensors"
    f_wild = model_dir / "loras" / "unregistered_wild.safetensors"
    f1.write_bytes(b"MODEL1" * 100)
    f2.write_bytes(b"MODEL2" * 100)
    f_wild.write_bytes(b"WILD" * 50)

    catalog = [
        {
            "model_type": "checkpoints",
            "filename": "recent.safetensors",
            "relative_path": "recent.safetensors",
            "drive_file_id": "drive-file-1",
            "account": "acc-1",
            "size": f1.stat().st_size,
            "source": {"kind": "huggingface", "download_url": "https://huggingface.co/org/repo/resolve/main/recent.safetensors?token=SECRET_SHOULD_STRIP"},
        },
        {
            "model_type": "checkpoints",
            "filename": "older.safetensors",
            "relative_path": "older.safetensors",
            "drive_file_id": "drive-file-2",
            "account": "acc-1",
            "size": f2.stat().st_size,
        },
    ]
    lm = LocalModels(root, model_dir, {"channel": "ch-secret", "catalog": catalog, "drive_auth": {"acc-1": {}}})
    lm.usage_records["checkpoints/recent.safetensors"] = {"category": "checkpoints", "name": "recent.safetensors", "last_used_at": 2000.0}
    lm.usage_records["checkpoints/older.safetensors"] = {"category": "checkpoints", "name": "older.safetensors", "last_used_at": 1000.0}

    inv = lm.scan_inventory()
    assert len(inv["recoverable_models"]) == 2
    assert [m["name"] for m in inv["recoverable_models"]] == ["recent.safetensors", "older.safetensors"]
    assert inv["wild_models"] == ["loras/unregistered_wild.safetensors"]
    # Sanitized source MUST NOT contain secret token query param
    assert "SECRET_SHOULD_STRIP" not in json.dumps(inv)
    assert inv["recoverable_models"][0]["source"]["download_url"] == "https://huggingface.co/org/repo/resolve/main/recent.safetensors"
    assert lm.inventory_digest() == inv["digest"]

    # Incremental sync via native.sync_plugins(): only fetches /inventory when digest changes & verifies active_lease.id
    repo = _make_repo(tmp_path)
    clock = FakeClock()
    _seed_project(repo)
    _seed_account(repo)
    _, inst, lease = _seed_running_instance(
        repo, clock, pool=POOL_NATIVE,
        model_storage={"model_storage_mode": "local", "cache_gb": 20, "cache_max_age_minutes": 60},
    )
    _configure_native_mock(monkeypatch, mock_sync=False)
    monkeypatch.setattr(native, "active_lease", lambda r, o: repo.get("leases", lease["id"], o))
    from app import native_model_api

    inv_calls = []
    monkeypatch.setattr(native_model_api, "headers", lambda *a, **kw: {"X-Studio-Model-Channel": "ch-secret"})

    def sync_transport(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/studio_native/models/inventory"):
            inv_calls.append(req.url.path)
            return httpx.Response(200, json=inv)
        return httpx.Response(200, json={"local_models_digest": inv["digest"]})

    state_client = httpx.Client(transport=httpx.MockTransport(sync_transport))
    native.sync_plugins(repo, lease, state_client)
    assert len(inv_calls) == 1
    saved = repo.store.get("settings", recovery.local_models_key("owner-1"), "owner-1")
    assert saved["by_instance"][inst["id"]]["digest"] == inv["digest"]
    # Second poll with unchanged digest -> ZERO /inventory requests!
    native.sync_plugins(repo, repo.get("leases", lease["id"], "owner-1"), state_client)
    assert len(inv_calls) == 1

    # Background restore in last_used_at desc order hitting 5 GiB DiskSpaceError on 2nd model
    root2 = tmp_path / "vm-root-2"
    root2.mkdir()
    model_dir2 = root2 / "models"
    model_dir2.mkdir()
    lm2 = LocalModels(root2, model_dir2, {"channel": "ch-2", "catalog": catalog, "drive_auth": {"acc-1": {}}})
    order: list[str] = []
    refreshed = []

    def fake_prepare_one(target, plan, status):
        order.append(target.name)
        if target.name == "recent.safetensors":
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"restored-ok")
            status.update(state="ready")
        else:
            status.update(state="failed", error=str(DiskSpaceError()), disk_space_limit=True)

    lm2.prepare_one = fake_prepare_one
    restore_list = [
        {"category": "checkpoints", "name": "older.safetensors", "last_used_at": 1000.0, "drive_file_id": "drive-file-2"},
        {"category": "checkpoints", "name": "recent.safetensors", "last_used_at": 2000.0, "drive_file_id": "drive-file-1"},
        {"category": "checkpoints", "name": "oldest.safetensors", "last_used_at": 500.0, "drive_file_id": "drive-file-3"},
    ]

    async def _run_restore():
        ident, task = lm2.submit_restore(restore_list, refresh_fn=lambda: refreshed.append(1))
        await task
        return lm2.status(ident)

    res = asyncio.run(_run_restore())
    assert order == ["recent.safetensors", "older.safetensors"]  # MRU first; oldest skipped immediately!
    assert len(refreshed) == 1
    assert res["state"] == "partial_disk_limit"
    assert res["partial_disk_limit"] is True
    states_by_name = {m["name"]: m["state"] for m in res["models"]}
    assert states_by_name["recent.safetensors"] == "ready"
    assert states_by_name["older.safetensors"] == "skipped_disk_limit"
    assert states_by_name["oldest.safetensors"] == "skipped_disk_limit"
    assert (model_dir2 / "checkpoints" / "recent.safetensors").is_file()


# ---------------------------------------------------------------------------
# 9. Workspace Bidirectional Mutual Exclusion, Manual Start Override & REST API
# ---------------------------------------------------------------------------

def test_workspace_mutual_exclusion_api_and_manual_start_override(tmp_path, monkeypatch):
    _enable_recovery(monkeypatch)
    _configure_native_mock(monkeypatch)
    repo = _make_repo(tmp_path)
    clock = FakeClock()
    _seed_project(repo)
    _seed_account(repo)

    gid = "rec-nativews01"
    recovery.save_plan(repo, "owner-1", {
        "plan_id": gid, "recovery_group_id": gid, "owner_id": "owner-1", "pool": POOL_NATIVE,
        "state": "scheduled", "account_id": ACC1, "requested_spec": SPEC_L4,
        "retention_mode": "retain", "retain_idle_minutes": 30,
        "hard_deadline_at": clock() + 3600, "orig_idle_deadline_at": clock() + 1800,
        "was_busy": False, "rounds_used": 0, "max_rounds": 3, "backoff_attempt": 0,
        "round_started_at": clock(), "next_retry_at": clock(),
        "native_snapshot": {"model_storage": {"model_storage_mode": "drive_mount", "cache_gb": 20, "cache_max_age_minutes": 60}},
    })

    # Direction 1: Active recovery plan blocks offline workspace write & exposes recovery_plan_id in overview
    writable, reason = native.workspace_offline_status(repo, "owner-1")
    assert writable is False
    assert "自动恢复" in (reason or "")
    overview = native_workspace.get_workspace_overview(repo, "owner-1")
    assert overview["offline_writable"] is False
    assert overview["recovery_plan_id"] == gid

    # Direction 2: Active workspace write lock defers recovery dispatch by 5s WITHOUT burning backoff_attempt!
    # Seed lock row directly (simulating a workspace mutation already holding the lock when plan becomes due)
    repo.store.create(
        "settings",
        "owner-1",
        {"lock_id": "op-1", "reason": "editing", "started_at": clock(), "expires_at": clock() + 90.0},
        record_id=native._workspace_lock_key("owner-1"),
    )
    assert recovery.tick(repo, clock=clock, process_started_at=clock() - 100) == 0
    deferred_plan = recovery.get_plan(repo, "owner-1", gid)
    assert deferred_plan["state"] == "scheduled"
    assert deferred_plan["backoff_attempt"] == 0
    assert pytest.approx(deferred_plan["next_retry_at"] - clock(), abs=0.5) == 5.0
    native.release_workspace_write_lock(repo, "owner-1", "op-1")

    # REST API GET /api/gpu/recovery/status and POST /api/gpu/recovery/{plan_id}/cancel
    monkeypatch.setattr(settings, "studio_api_token", "api-test-token")
    studio_app.state.store = repo.store
    studio_app.state.repo = repo
    from app.common import current_user

    studio_app.dependency_overrides[current_user] = lambda: {"id": "owner-1", "role": "admin"}
    try:
        with TestClient(studio_app) as client:
            r_status = client.get("/api/gpu/recovery/status")
            assert r_status.status_code == 200
            data = r_status.json()["data"]
            assert data["config"]["enabled"] is True
            assert len(data["active_plans"]) == 1
            assert data["active_plans"][0]["plan_id"] == gid

            r_cancel = client.post(f"/api/gpu/recovery/{gid}/cancel")
            assert r_cancel.status_code == 200
            assert r_cancel.json()["data"]["state"] == "cancelled"
    finally:
        studio_app.dependency_overrides.pop(current_user, None)

    # After cancel, offline workspace is immediately unlocked!
    writable_after, _ = native.workspace_offline_status(repo, "owner-1")
    assert writable_after is True
    assert native_workspace.get_workspace_overview(repo, "owner-1")["recovery_plan_id"] is None

    # Manual native.admit() cancels any scheduled native recovery plan in the same TX (only 1 VM!)
    gid2 = "rec-nativews02"
    recovery.save_plan(repo, "owner-1", {
        "plan_id": gid2, "recovery_group_id": gid2, "owner_id": "owner-1", "pool": POOL_NATIVE,
        "state": "scheduled", "account_id": ACC1, "requested_spec": SPEC_L4,
        "hard_deadline_at": clock() + 3600, "rounds_used": 0, "max_rounds": 3,
    })
    manual_job = native.admit(repo, "owner-1", {"account_id": ACC1, "spec": {"key": SPEC_L4}}, "manual-idem-1")
    assert manual_job["id"]
    assert recovery.get_plan(repo, "owner-1", gid2)["state"] == "cancelled"


# ---------------------------------------------------------------------------
# 10. All 4 High-Risk Concurrency Interleavings (§10.1(10) A / B / C / D)
# ---------------------------------------------------------------------------

def test_interleaving_a_assign_timeout_cancel_waits_for_60s_nbh_reconciliation(tmp_path, monkeypatch):
    _enable_recovery(monkeypatch)
    _configure_native_mock(monkeypatch)
    repo = _make_repo(tmp_path)
    clock = FakeClock()
    _seed_project(repo)
    _seed_account(repo)

    gid = "rec-interleave-a"
    recovery.save_plan(repo, "owner-1", {
        "plan_id": gid, "recovery_group_id": gid, "owner_id": "owner-1", "pool": POOL_NATIVE,
        "state": "scheduled", "account_id": ACC1, "requested_spec": SPEC_L4,
        "retention_mode": "retain", "retain_idle_minutes": 30,
        "hard_deadline_at": clock() + 3600, "orig_idle_deadline_at": clock() + 1800,
        "was_busy": False, "rounds_used": 0, "max_rounds": 3, "backoff_attempt": 0,
        "round_started_at": clock(), "next_retry_at": clock(),
        "native_snapshot": {"model_storage": {"model_storage_mode": "drive_mount", "cache_gb": 20, "cache_max_age_minutes": 60}},
    })

    # Assign POST times out leaving tunnel_nbh set (remote_state_unknown=True)
    def timeout_create(spec, request_id, job_id, runtime_id):
        for lrow in repo.list("leases", "owner-1"):
            if lrow.get("owner_job_id") == job_id:
                repo.update("leases", lrow["id"], {"tunnel_nbh": "nbh-unknown-1", "create_started": clock()}, "owner-1")
        raise ColabError("assign POST timed out", error_code="colab_assign_timeout", phase="assign_post", remote_state_unknown=True)

    reconcile_done = {"ok": False}
    prov = MockColab(
        create_plan=[timeout_create],
        inspect_map={},
        delete_ok=lambda h: reconcile_done["ok"],
    )
    prov.inspect = lambda h: {"deleted": False, "ready": False} if not reconcile_done["ok"] else {"deleted": True, "missing": True}
    sched = Scheduler(repo, provider_factory=lambda j: prov, executor_factory=lambda j: MockExecutor(), clock=clock,
                      config=GpuConfig(enabled=True, real_enabled=True, create_retries=0))

    # Dispatch successor -> enters allocating -> hits timeout -> lease stuck in cleanup_pending for 60s nbh window
    recovery.tick(repo, clock=clock, scheduler=sched, process_started_at=clock() - 100)
    for _ in range(10):
        sched.tick()

    plan = recovery.get_plan(repo, "owner-1", gid)
    assert plan["state"] == "dispatching"
    assert plan["active_lease_id"] is not None

    # User clicks Cancel while lease is reconciling -> enters cancelling, workspace STILL locked!
    res = recovery.cancel_plan(repo, "owner-1", gid, scheduler=sched, clock=clock)
    assert res["state"] == "cancelling"
    assert native.workspace_offline_status(repo, "owner-1")[0] is False

    # Before 60s reconciliation completes, plan stays cancelling (never double-allocates, never unlocks prematurely)
    clock.advance(20)
    recovery.tick(repo, clock=clock, scheduler=sched, process_started_at=clock() - 100)
    assert recovery.get_plan(repo, "owner-1", gid)["state"] == "cancelling"

    # After 60s nbh window -> late runtime reconciled & deleted -> plan transitions to cancelled & workspace unlocks!
    clock.advance(TUNNEL_RECONCILE_WINDOW_SECONDS + 5)
    reconcile_done["ok"] = True
    for _ in range(5):
        sched.tick()
    recovery.tick(repo, clock=clock, scheduler=sched, process_started_at=clock() - 100)
    assert recovery.get_plan(repo, "owner-1", gid)["state"] == "cancelled"
    assert native.workspace_offline_status(repo, "owner-1")[0] is True


def test_interleaving_b_process_restart_after_reclaim_tx_commits(tmp_path, monkeypatch):
    _enable_recovery(monkeypatch)
    repo = _make_repo(tmp_path)
    clock = FakeClock()
    _seed_project(repo)
    _seed_account(repo)
    _seed_published_workflow(repo)

    job, inst, lease = _seed_running_instance(
        repo, clock, pool=POOL_CREATION, instance_state="busy", lease_state="active", job_state="running",
        retention_mode="smart",
    )
    prov1 = MockColab(inspect_map={lease["runtime_id"]: {"deleted": True, "missing": True}})
    sched1 = Scheduler(repo, provider_factory=lambda j: prov1, executor_factory=lambda j: MockExecutor(), clock=clock)
    sched1._inspect_runtime(repo.get("leases", lease["id"], "owner-1"))
    clock.advance(6)
    sched1._inspect_runtime(repo.get("leases", lease["id"], "owner-1"))

    # Simulate complete process crash & restart with brand-new Repository, InstanceService, Scheduler on same DB!
    repo2 = Repository(Store(f"sqlite:///{tmp_path / 'studio.db'}"))
    prov2 = MockColab()
    sched2 = Scheduler(
        repo2,
        provider_factory=lambda j: prov2,
        executor_factory=lambda j: MockExecutor(),
        clock=clock,
        config=GpuConfig(enabled=True, real_enabled=True, create_retries=0),
    )
    gid = recovery.make_group_id(inst["id"])
    assert recovery.get_plan(repo2, "owner-1", gid)["state"] == "scheduled"
    assert repo2.get("jobs", job["id"], "owner-1")["state"] == "waiting_for_capacity"

    recovery.tick(repo2, clock=clock, scheduler=sched2, process_started_at=clock() - 100)
    for _ in range(25):
        if not sched2.tick().get("advanced"):
            break
    recovery.tick(repo2, clock=clock, scheduler=sched2, process_started_at=clock() - 100)
    assert repo2.get("jobs", job["id"], "owner-1")["state"] == "completed"
    assert recovery.get_plan(repo2, "owner-1", gid)["state"] == "recovered"


def test_interleaving_c_old_deleted_lease_shares_owner_job_id(tmp_path, monkeypatch):
    _enable_recovery(monkeypatch)
    repo = _make_repo(tmp_path)
    clock = FakeClock()
    _seed_project(repo)
    _seed_account(repo)

    job, _, old_lease = _seed_running_instance(
        repo, clock, pool=POOL_CREATION, instance_state="deleted", lease_state="deleted", job_state="waiting_for_capacity",
        retention_mode="smart",
        job_extra={"instance_id": None, "lease_id": None, "attempt_id": "att-new-2", "attempt_number": 1},
    )
    # Force job back to no lease_id/instance_id while old_lease still has owner_job_id == job["id"] and state == "deleted"
    job = repo.update("jobs", job["id"], {"instance_id": None, "lease_id": None, "attempt_id": "att-new-2"}, "owner-1")
    sched = Scheduler(repo, provider_factory=lambda j: MockColab(), executor_factory=lambda j: MockExecutor(), clock=clock)

    assert sched._lease(job) is None
    new_lease = sched._ensure_lease(job)
    assert new_lease["id"] != old_lease["id"]
    assert new_lease["attempt_id"] == "att-new-2"
    repo.update("jobs", job["id"], {"state": "loading", "lease_id": new_lease["id"]}, "owner-1")
    prepared = repo.prepare_execution(job["id"], "owner-1", new_lease["id"], clock())
    assert prepared["attempt_number"] == 2


def test_interleaving_d_successor_reclaimed_during_restoring_env(tmp_path, monkeypatch):
    _enable_recovery(monkeypatch)
    _configure_native_mock(monkeypatch, mock_sync=False)
    monkeypatch.setattr(native, "native_proxy_healthy", lambda r, o: False)
    repo = _make_repo(tmp_path)
    clock = FakeClock()
    _seed_project(repo)
    _seed_account(repo)

    frozen_7 = [
        {"category": "checkpoints", "name": f"m{i}.safetensors", "last_used_at": 1000.0 + i, "drive_file_id": f"df-{i}"}
        for i in range(7)
    ]
    gid = "rec-interleave-d"
    _, succ_inst, succ_lease = _seed_running_instance(
        repo, clock, pool=POOL_NATIVE,
        model_storage={"model_storage_mode": "local", "cache_gb": 20, "cache_max_age_minutes": 60},
        instance_extra={"recovery_group_id": gid, "recovery_generation": 1, "recovery_source": "auto_recovery"},
    )
    recovery.save_plan(repo, "owner-1", {
        "plan_id": gid, "recovery_group_id": gid, "owner_id": "owner-1", "pool": POOL_NATIVE,
        "state": "restoring_env", "account_id": ACC1, "requested_spec": SPEC_L4,
        "retention_mode": "retain", "retain_idle_minutes": 30,
        "hard_deadline_at": clock() + 3600, "orig_idle_deadline_at": clock() + 1800,
        "was_busy": False, "rounds_used": 1, "max_rounds": 3,
        "round_counted_for_instance_id": succ_inst["id"],
        "source_instance_id": "orig-inst-0",
        "successor_instance_id": succ_inst["id"],
        "active_lease_id": succ_lease["id"],
        "native_snapshot": {
            "model_storage": {"model_storage_mode": "local", "cache_gb": 20, "cache_max_age_minutes": 60},
            "recoverable_models": frozen_7,
            "wild_models": ["loras/wild.safetensors"],
            "model_restore_job_id": "restore-job-1",
            "model_restore_state": {"state": "preparing", "total": 7, "completed": 2, "failed": 0, "skipped": 0, "partial_disk_limit": False},
        },
    })

    # Empty-disk heartbeat arrives on successor during restoring_env -> MUST NOT overwrite frozen 7 models!
    from app import native_model_api

    monkeypatch.setattr(native, "active_lease", lambda r, o: repo.get("leases", succ_lease["id"], o))
    monkeypatch.setattr(native_model_api, "headers", lambda *a, **kw: {"X-Studio-Model-Channel": "ch"})
    monkeypatch.setattr(
        native_model_api,
        "rpc_sync",
        lambda *a, **kw: {"digest": "empty-disk-digest", "recoverable_models": [], "wild_models": []},
    )
    empty_client = httpx.Client(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json={"local_models_digest": "empty-disk-digest"})
    ))
    native.sync_plugins(repo, succ_lease, empty_client)
    mid_plan = recovery.get_plan(repo, "owner-1", gid)
    assert len(mid_plan["native_snapshot"]["recoverable_models"]) == 7

    # Successor is reclaimed during restoring_env -> enters round 2/3 AND preserves all 7 original models!
    prov = MockColab(inspect_map={succ_lease["runtime_id"]: {"deleted": True, "missing": True}})
    sched = Scheduler(repo, provider_factory=lambda j: prov, executor_factory=lambda j: MockExecutor(), clock=clock)
    assert sched._inspect_runtime(repo.get("leases", succ_lease["id"], "owner-1")) == "missing_pending"
    clock.advance(6)
    assert sched._inspect_runtime(repo.get("leases", succ_lease["id"], "owner-1")) == "missing"

    next_plan = recovery.get_plan(repo, "owner-1", gid)
    assert next_plan["state"] == "scheduled"
    assert next_plan["rounds_used"] == 1  # Round 1 already deducted -> next round will be 2/3
    assert len(next_plan["native_snapshot"]["recoverable_models"]) == 7
    assert {m["name"] for m in next_plan["native_snapshot"]["recoverable_models"]} == {f"m{i}.safetensors" for i in range(7)}
