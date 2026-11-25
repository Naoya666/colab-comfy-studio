"""Venv --system-site-packages isolation, VM baseline purity & teardown tests (Spec §9.1(1))."""

from __future__ import annotations

import io
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

from app.gpu import native_runtime, native_workspace, node_persistence, runtime_worker


class _FakeRemote:
    """Minimal node_persistence.Remote stand-in for env-cache v4 upload/restore."""

    def __init__(self, root: Path):
        self.root = root
        self.uploads: list[str] = []

    def available(self) -> bool:
        return True

    def put_env_cache(self, source: Path, name: str, *, verify: bool = True) -> bool:
        self.uploads.append(name)
        dst = self.root / "env-cache" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dst)
        return True

    def get_env_cache(self, name: str, target: Path) -> bool:
        src = self.root / "env-cache" / name
        if not src.exists():
            return False
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
        return True

    def prune_env_cache(self, keep: int = 3) -> None:
        return None


def test_domain_venv_creation_uses_system_site_packages_and_no_user_site(tmp_path, monkeypatch):
    venv_dir = tmp_path / "domain-venv"
    py = runtime_worker.ensure_domain_venv(venv_dir)
    assert py.exists()
    cfg = (venv_dir / "pyvenv.cfg").read_text(encoding="utf-8")
    assert "include-system-site-packages = true" in cfg

    env = runtime_worker.domain_python_env(venv_dir, extra={"CUSTOM_FLAG": "1"})
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["VIRTUAL_ENV"] == str(venv_dir)
    assert env["CUSTOM_FLAG"] == "1"
    assert str(venv_dir / "bin") == env["PATH"].split(os.pathsep)[0]

    # _pip_install must target domain venv and set PYTHONNOUSERSITE=1.
    calls: list[tuple[list[str], dict[str, str]]] = []

    def fake_run(cmd, **kwargs):
        calls.append((list(cmd), dict(kwargs.get("env") or {})))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    comfy_dir = tmp_path / "ComfyUI"
    comfy_dir.mkdir()
    (comfy_dir / "requirements.txt").write_text("einops\n", encoding="utf-8")
    monkeypatch.setattr(runtime_worker.subprocess, "run", fake_run)
    monkeypatch.setenv("CREATION_USE_UV", "0")
    runtime_worker._pip_install(comfy_dir, 60.0, venv_dir=venv_dir)
    pip_calls = [(c, e) for c, e in calls if "-m" in c and "pip" in c]
    assert pip_calls
    cmd, env_used = pip_calls[0]
    assert cmd[0] == str(py)
    assert env_used.get("PYTHONNOUSERSITE") == "1"

    # Native runtime uses /content/ai-studio/native/venv + PYTHONNOUSERSITE=1.
    assert native_runtime.ROOT == Path("/content/ai-studio/native")
    assert "PYTHONNOUSERSITE" in native_runtime.WATCHDOG
    assert "venv" in native_runtime.WATCHDOG


def test_node_persistence_uses_domain_venv_and_no_user_site(tmp_path):
    venv_dir = tmp_path / "native-venv"
    py = runtime_worker.ensure_domain_venv(venv_dir)
    comfy = tmp_path / "ComfyUI"
    (comfy / "custom_nodes" / "ComfyUI-Manager").mkdir(parents=True)
    (comfy / "custom_nodes" / "ComfyUI-Manager" / "cm-cli.py").write_text("print('ok')\n", encoding="utf-8")

    cli_path = node_persistence.manager_cli(comfy, venv_dir=venv_dir)
    assert cli_path is not None
    assert Path(cli_path).name in {"cm-cli.py", "__main__.py"}

    pers = node_persistence.Persistence(tmp_path / "state", comfy, tmp_path / "user", remote=None, venv_dir=venv_dir)
    env = pers._python_env()
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["VIRTUAL_ENV"] == str(venv_dir)
    assert pers._python_bin() == str(py)


def test_vm_baseline_ignores_pycache_and_fails_closed_on_global_or_user_site_pollution(tmp_path, monkeypatch):
    global_site = tmp_path / "global_site"
    global_site.mkdir()
    (global_site / "torch").mkdir()
    (global_site / "torch" / "__init__.py").write_text("VERSION = '2.4'\n", encoding="utf-8")
    (global_site / "torch-2.4.0.dist-info").mkdir()
    (global_site / "torch-2.4.0.dist-info" / "METADATA").write_text("Name: torch\nVersion: 2.4.0\n", encoding="utf-8")

    fake_home = tmp_path / "fake_home"
    fake_home.mkdir()
    monkeypatch.setattr(runtime_worker, "_global_site_dirs", lambda: [global_site])
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))

    studio_root = tmp_path / "ai-studio"
    baseline = runtime_worker.ensure_vm_baseline(studio_root)
    expected_digest = baseline["digest"]
    assert len(expected_digest) == 64

    def check():
        return runtime_worker.verify_vm_purity(
            expected_digest,
            studio_root=studio_root,
            check_ports=False,
            check_gpu=False,
            check_mounts=False,
            home_dir=fake_home,
        )

    # 1. __pycache__ & .pyc noise MUST NOT change digest or fail purity.
    pycache = global_site / "torch" / "__pycache__"
    pycache.mkdir()
    (pycache / "__init__.cpython-311.pyc").write_bytes(b"\x00\x01\x02")
    rep_clean = check()
    assert rep_clean["ok"] is True
    assert rep_clean["site_snapshot_ok"] is True
    assert rep_clean["user_site_ok"] is True

    # 2. Adding a new non-protected package into global site-packages fails closed.
    extra_pkg = global_site / "rogue_node_pkg"
    extra_pkg.mkdir()
    (extra_pkg / "__init__.py").write_text("x = 1\n", encoding="utf-8")
    rep_pkg = check()
    assert rep_pkg["ok"] is False
    assert rep_pkg["site_snapshot_ok"] is False
    shutil.rmtree(extra_pkg)

    # 3. Injecting a .pth file into global site-packages fails closed.
    pth_file = global_site / "injected.pth"
    pth_file.write_text("/tmp/evil\n", encoding="utf-8")
    rep_pth = check()
    assert rep_pth["ok"] is False
    assert rep_pth["site_snapshot_ok"] is False
    pth_file.unlink()

    # 4. Writing into ~/.local/lib/python*/site-packages fails closed.
    user_site = fake_home / ".local" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    user_site.mkdir(parents=True)
    (user_site / "leaked.py").write_text("a = 1\n", encoding="utf-8")
    rep_user = check()
    assert rep_user["ok"] is False
    assert rep_user["user_site_ok"] is False
    shutil.rmtree(fake_home / ".local")

    # 5. Tampering with a protected package's dist-info METADATA fails closed.
    (global_site / "torch-2.4.0.dist-info" / "METADATA").write_text("Name: torch\nVersion: 2.9.9\n", encoding="utf-8")
    rep_tamper = check()
    assert rep_tamper["ok"] is False
    assert rep_tamper["site_snapshot_ok"] is False


def test_native_env_cache_v4_build_restore_and_v3_isolation(tmp_path):
    assert node_persistence._ENV_CACHE_SCHEMA == 4
    assert native_workspace.CURRENT_ENV_CACHE_SCHEMA == 4

    comfy = tmp_path / "ComfyUI"
    (comfy / "custom_nodes").mkdir(parents=True)
    (comfy / "requirements.txt").write_text("einops==0.8.0\n", encoding="utf-8")
    (comfy / "manager_requirements.txt").write_text("GitPython\n", encoding="utf-8")
    key_info = node_persistence.compute_env_cache_key(
        comfy,
        "a" * 40,
        "l4",
        comfy / "custom_nodes",
        [],
        base_packages=("einops",),
        base_pkg_sha="b" * 64,
    )
    assert key_info["key"].startswith("native-env-v4-")
    assert key_info["spec"]["schema"] == 4

    # Build env cache v4 inside isolated venv_src.
    venv_src = tmp_path / "venv_src"
    runtime_worker.ensure_domain_venv(venv_src)
    src_site = runtime_worker.venv_site_dir(venv_src)
    before = node_persistence.site_snapshot(src_site)

    (src_site / "demo_plugin_pkg").mkdir(parents=True)
    (src_site / "demo_plugin_pkg" / "__init__.py").write_text("MARKER = 'v4-isolated'\n", encoding="utf-8")
    (src_site / "demo_plugin_pkg-1.0.dist-info").mkdir()
    (src_site / "demo_plugin_pkg-1.0.dist-info" / "METADATA").write_text("Name: demo_plugin_pkg\nVersion: 1.0\n", encoding="utf-8")
    (src_site / "demo_plugin_pkg-1.0.dist-info" / "RECORD").write_text(
        "demo_plugin_pkg/__init__.py,,\ndemo_plugin_pkg-1.0.dist-info/METADATA,,\n",
        encoding="utf-8",
    )

    remote = _FakeRemote(tmp_path / "drive")
    work_build = tmp_path / "work_build"
    work_build.mkdir()
    plan = node_persistence.prepare_env_cache_plan(work_build, src_site, before, key_info)
    assert plan is not None
    assert node_persistence.upload_env_cache_from_plan(work_build, plan, remote=remote) is True
    assert any(name.startswith("native-env-v4-") for name in remote.uploads)

    # Restore into fresh venv_dst: files land in venv_dst site-packages.
    venv_dst = tmp_path / "venv_dst"
    runtime_worker.ensure_domain_venv(venv_dst)
    dst_site = runtime_worker.venv_site_dir(venv_dst)
    ok = node_persistence.try_restore_env_cache(
        tmp_path / "work_restore",
        dst_site,
        key_info,
        baseline_protected={},
        remote=remote,
        venv_dir=venv_dst,
    )
    assert ok is True
    assert (dst_site / "demo_plugin_pkg" / "__init__.py").read_text(encoding="utf-8") == "MARKER = 'v4-isolated'\n"

    # Mutate metadata to legacy schema=3: v4 restore MUST reject it.
    meta_file = tmp_path / "drive" / "env-cache" / f"{key_info['key']}.meta.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    meta["schema"] = 3
    meta_file.write_text(json.dumps(meta), encoding="utf-8")
    venv_dst2 = tmp_path / "venv_dst2"
    runtime_worker.ensure_domain_venv(venv_dst2)
    dst_site2 = runtime_worker.venv_site_dir(venv_dst2)
    assert (
        node_persistence.try_restore_env_cache(
            tmp_path / "work_restore2",
            dst_site2,
            key_info,
            baseline_protected={},
            remote=remote,
            venv_dir=venv_dst2,
        )
        is False
    )
    assert not (dst_site2 / "demo_plugin_pkg").exists()


def test_workflow_smoke_disables_warm_and_cleans_between_runs(tmp_path, monkeypatch):
    studio_root = tmp_path / "ai-studio"
    studio_root.mkdir()
    old_exec = studio_root / "11111111-1111-1111-1111-111111111111"
    old_exec.mkdir()
    (old_exec / "stale.txt").write_text("stale", encoding="utf-8")

    # Non-UUID directories must remain untouched by smoke history cleanup.
    (studio_root / ".vm-baseline.json").write_text("{}", encoding="utf-8")
    (studio_root / "native").mkdir()

    cur_exec = studio_root / "22222222-2222-2222-2222-222222222222"
    (cur_exec / "ComfyUI" / "output").mkdir(parents=True)

    warm_calls: list[str] = []
    monkeypatch.setattr(runtime_worker, "_warm_take", lambda *a, **kw: warm_calls.append("take") or False)
    monkeypatch.setattr(runtime_worker, "_warm_put", lambda *a, **kw: warm_calls.append("put"))
    monkeypatch.setattr(runtime_worker, "_restore_env", lambda root, *a, **kw: {"source": "fresh"})
    monkeypatch.setattr(runtime_worker, "_restore_weights", lambda *a, **kw: [])
    monkeypatch.setattr(runtime_worker, "_write_input_files", lambda *a, **kw: None)

    out_file = cur_exec / "ComfyUI" / "output" / "smoke.png"
    out_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)

    popen_envs: list[dict[str, str]] = []
    popen_cmds: list[list[str]] = []

    class _DummyProc:
        pid = 0
        _code = None

        def poll(self):
            return self._code

        def terminate(self):
            self._code = 0

        def kill(self):
            self._code = -9

        def wait(self, timeout=None):
            self._code = 0
            return 0

    real_popen = runtime_worker.subprocess.Popen

    def fake_popen(cmd, *args, **kwargs):
        cmd_list = [str(c) for c in cmd] if isinstance(cmd, (list, tuple)) else [str(cmd)]
        if "main.py" not in cmd_list:
            return real_popen(cmd, *args, **kwargs)
        popen_cmds.append(cmd_list)
        popen_envs.append(dict(kwargs.get("env") or {}))
        return _DummyProc()

    def fake_http_json(url, method="GET", body=None, timeout=5):
        if url.endswith("/system_stats"):
            return {"system": {}}
        if url.endswith("/prompt"):
            return {"prompt_id": "p-smoke-1"}
        if "/history/p-smoke-1" in url:
            return {
                "p-smoke-1": {
                    "status": {"completed": True},
                    "outputs": {"1": {"images": [{"filename": "smoke.png", "subfolder": "", "type": "output"}]}},
                }
            }
        return {}

    monkeypatch.setattr(runtime_worker.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(runtime_worker, "_http_json", fake_http_json)
    monkeypatch.setattr(runtime_worker, "urlopen", lambda req, timeout=30: io.BytesIO(out_file.read_bytes()))

    global_site = tmp_path / "global_site"
    global_site.mkdir()
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(runtime_worker, "_global_site_dirs", lambda: [global_site])
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    base = runtime_worker.ensure_vm_baseline(studio_root)

    manifest = {
        "comfy_commit": "a" * 40,
        "node_allowlist": ["CheckpointLoaderSimple"],
        "custom_nodes": [],
        "weight_files": [
            {
                "repo_id": "acme/model",
                "filename": "m.safetensors",
                "revision": "b" * 40,
                "sha256": "c" * 64,
                "target": "checkpoints/m.safetensors",
            }
        ],
    }
    graph = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {}}}
    payload = {
        "snapshot": {"workflow_smoke": True},
        "expected_baseline_digest": base["digest"],
    }

    outputs = runtime_worker._run_comfy(cur_exec, payload, manifest, graph, runtime_worker._Stages(cur_exec))
    assert len(outputs) == 1
    assert warm_calls == []
    assert popen_cmds and str(cur_exec / "venv") in popen_cmds[0][0]
    assert popen_envs and popen_envs[0].get("PYTHONNOUSERSITE") == "1"
    # Previous ephemeral UUID dir cleaned; baseline & native untouched.
    assert not old_exec.exists()
    assert (studio_root / ".vm-baseline.json").exists()
    assert (studio_root / "native").exists()


def test_teardown_native_order_aborts_on_unsynced_and_blocks_rmtree_on_active_mount(tmp_path, monkeypatch):
    studio_root = tmp_path / "ai-studio"
    native_dir = studio_root / "native"
    native_dir.mkdir(parents=True)
    (native_dir / "state.json").write_text(
        json.dumps({"watchdog_pid": 9001, "tunnel_pid": 9002, "pid": 9003, "rclone_pid": 9004}),
        encoding="utf-8",
    )
    (native_dir / "rclone-state.json").write_text(
        json.dumps({"rclone_pid": 9004, "rc_port": 5572}),
        encoding="utf-8",
    )

    global_site = tmp_path / "global_site"
    global_site.mkdir()
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(runtime_worker, "_global_site_dirs", lambda: [global_site])
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    monkeypatch.setattr(runtime_worker, "_scan_matching_pids", lambda pred: [])
    base = runtime_worker.ensure_vm_baseline(studio_root)

    # Case 1: Unsynced VFS upload detected AFTER stopping watchdog & cloudflared -> abort BEFORE killing ComfyUI/rclone.
    order: list[str] = []

    def fake_stop_pid(pid, start_id=None, **kwargs):
        order.append(f"stop:{pid}")
        return True

    def fake_local_sync(_native_root):
        order.append("probe_sync")
        return False, "rclone_vfs_uploads_in_progress:2"

    monkeypatch.setattr(runtime_worker, "_stop_pid_verified", fake_stop_pid)
    monkeypatch.setattr(runtime_worker, "_local_native_sync_check", fake_local_sync)

    exec_root1 = studio_root / "teardown-1"
    exec_root1.mkdir()
    res1 = runtime_worker.run_teardown(
        {
            "from_pool": "native",
            "to_pool": "creation",
            "expected_baseline_digest": base["digest"],
            "allow_mock_gpu": True,
        },
        exec_root1,
        home_dir=fake_home,
    )
    assert res1["state"] == "aborted_unsynced"
    assert res1["cleaned"] is False
    # Watchdog (9001) and cloudflared (9002) stopped BEFORE sync probe; ComfyUI (9003) and rclone (9004) NOT killed!
    assert order == ["stop:9001", "stop:9002", "probe_sync"]
    assert native_dir.exists()

    # Case 2: Synced and rclone drained cleanly, but /proc/self/mountinfo still reports an active mount -> rmtree FORBIDDEN.
    order.clear()
    monkeypatch.setattr(runtime_worker, "_local_native_sync_check", lambda _nr: (True, None))
    monkeypatch.setattr(runtime_worker, "_check_port_free", lambda _p: False)
    monkeypatch.setattr(runtime_worker, "_wait_rclone_drained", lambda *a, **kw: None)
    monkeypatch.setattr(runtime_worker, "_unmount_native_drive", lambda *a, **kw: True)
    monkeypatch.setattr(
        runtime_worker,
        "_active_mounts_under",
        lambda path, **kw: [str(native_dir / "drive" / "nested")] if Path(path) == studio_root else [],
    )
    exec_root2 = studio_root / "teardown-2"
    exec_root2.mkdir()
    res2 = runtime_worker.run_teardown(
        {
            "from_pool": "native",
            "to_pool": "creation",
            "expected_baseline_digest": base["digest"],
            "allow_mock_gpu": True,
        },
        exec_root2,
        home_dir=fake_home,
    )
    assert res2["state"] == "failed"
    assert res2["cleaned"] is False
    assert res2["error_code"] == "mount_still_active"
    assert native_dir.exists()


def test_domain_venv_installs_pip_shims_and_disables_colab_uv_system_python(tmp_path, monkeypatch):
    """Colab sets UV_SYSTEM_PYTHON=true by default; domain venvs must block global installs."""
    monkeypatch.setenv("UV_SYSTEM_PYTHON", "true")
    monkeypatch.setenv("PYTHONPATH", "/tmp/rogue")

    vdir = tmp_path / "native" / "venv"
    py_bin = runtime_worker.ensure_domain_venv(vdir)
    assert py_bin.is_file()
    for shim_name in ("pip", "pip3"):
        shim = vdir / "bin" / shim_name
        assert shim.is_file()
        assert os.access(shim, os.X_OK)
        text = shim.read_text(encoding="utf-8")
        assert "-m pip" in text and str(py_bin) in text

    env = runtime_worker.domain_python_env(vdir)
    assert env["UV_SYSTEM_PYTHON"] == "false"
    assert env["PIP_REQUIRE_VIRTUALENV"] == "true"
    assert env["PYTHONNOUSERSITE"] == "1"
    assert "PYTHONPATH" not in env
    assert env["VIRTUAL_ENV"] == str(vdir)
    assert env["PATH"].split(os.pathsep)[0] == str(vdir / "bin")

    # Persistence._python_env() must enforce the same venv guards for Manager/cm-cli.
    native_root = tmp_path / "native"
    comfy = native_root / "ComfyUI"
    (comfy / "custom_nodes").mkdir(parents=True)
    user_dir = native_root / "user"
    user_dir.mkdir(parents=True)
    pers = node_persistence.Persistence(native_root, comfy, user_dir, venv_dir=vdir)
    penv = pers._python_env()
    assert penv["UV_SYSTEM_PYTHON"] == "false"
    assert penv["PIP_REQUIRE_VIRTUALENV"] == "true"
    assert penv["PYTHONNOUSERSITE"] == "1"
    assert penv["VIRTUAL_ENV"] == str(vdir)

    # verify_vm_purity() reports exact site_diff when global site-packages is mutated.
    fake_site = tmp_path / "global_site"
    fake_site.mkdir()
    (fake_site / "torch").mkdir()
    monkeypatch.setattr(runtime_worker, "_global_site_dirs", lambda: [fake_site])
    studio_root = tmp_path / "studio"
    base = runtime_worker.ensure_vm_baseline(studio_root=studio_root)
    (fake_site / "comfyui_frontend_package").mkdir()
    purity = runtime_worker.verify_vm_purity(
        base["digest"],
        studio_root=studio_root,
        check_ports=False,
        check_gpu=False,
        check_mounts=False,
        allow_mock_gpu=True,
        home_dir=tmp_path / "home",
    )
    assert purity["site_snapshot_ok"] is False
    assert any("comfyui_frontend_package" in p for p in (purity.get("site_diff") or {}).get("added", []))
    assert any("comfyui_frontend_package" in e for e in purity["errors"])


def test_wait_rclone_drained_allows_transient_writeback_and_fails_closed_on_stuck_or_error(tmp_path, monkeypatch):
    """Post-stop rclone check waits briefly for 5s write-back flush of closed logs, but fails closed on errors/timeout."""
    mount_dir = tmp_path / "drive"
    mount_dir.mkdir()
    monkeypatch.setattr(runtime_worker, "_native_open_writes", lambda _m: False)
    monkeypatch.setattr(runtime_worker, "_RCLONE_DRAIN_POLL_INTERVAL_SECONDS", 0.01)

    # 1. Transient upload queue (e.g. comfyui_8188.log closed on SIGTERM) drains cleanly.
    seq = iter(["rclone_uploads_queued:1", "rclone_uploads_in_progress:1", None])
    monkeypatch.setattr(runtime_worker, "_local_rclone_unsynced_reason", lambda: next(seq))
    assert runtime_worker._wait_rclone_drained(mount_dir, timeout=1.0) is None

    # 2. Hard rclone error fails immediately without waiting out timeout.
    calls = 0

    def _err():
        nonlocal calls
        calls += 1
        return "rclone_errored_files:1"

    monkeypatch.setattr(runtime_worker, "_local_rclone_unsynced_reason", _err)
    assert runtime_worker._wait_rclone_drained(mount_dir, timeout=1.0) == "rclone_errored_files:1"
    assert calls == 1

    # 3. Persistent upload queue past budget fails closed.
    monkeypatch.setattr(runtime_worker, "_local_rclone_unsynced_reason", lambda: "rclone_uploads_queued:3")
    assert runtime_worker._wait_rclone_drained(mount_dir, timeout=0.05) == "rclone_uploads_queued:3"


def test_teardown_aborts_unsynced_and_preserves_vfs_cache_when_rclone_exits_during_stop(tmp_path, monkeypatch):
    """If rclone exits before post-stop write-back confirmation, teardown must fail closed and keep vfs-cache."""
    studio_root = tmp_path / "ai-studio"
    native_dir = studio_root / "native"
    cache_file = native_dir / "vfs-cache" / "vfs" / "studio" / "output" / "pending.png"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_bytes(b"unsynced-image-bytes")
    (native_dir / "state.json").write_text(
        json.dumps({"watchdog_pid": 9101, "tunnel_pid": 9102, "pid": 9103, "rclone_pid": 9104}),
        encoding="utf-8",
    )
    (native_dir / "rclone-state.json").write_text(
        json.dumps({"rclone_pid": 9104, "rc_port": 5572}),
        encoding="utf-8",
    )

    global_site = tmp_path / "global_site"
    global_site.mkdir()
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(runtime_worker, "_global_site_dirs", lambda: [global_site])
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    monkeypatch.setattr(runtime_worker, "_scan_matching_pids", lambda pred: [])
    monkeypatch.setattr(runtime_worker, "_stop_pid_verified", lambda *a, **kw: True)
    monkeypatch.setattr(runtime_worker, "_active_mounts_under", lambda *a, **kw: [])
    monkeypatch.setattr(runtime_worker, "_unmount_native_drive", lambda *a, **kw: True)
    monkeypatch.setattr(runtime_worker, "_native_open_writes", lambda *a, **kw: False)
    base = runtime_worker.ensure_vm_baseline(studio_root)

    # 1. Normal native transfer: Step 1c saw rclone healthy, then rclone died during ComfyUI shutdown (port 5572 free).
    monkeypatch.setattr(runtime_worker, "_local_native_sync_check", lambda _nr: (True, None))
    monkeypatch.setattr(runtime_worker, "_check_port_free", lambda _p: True)

    exec_root1 = studio_root / "teardown-rclone-dead"
    exec_root1.mkdir()
    res1 = runtime_worker.run_teardown(
        {
            "from_pool": "native",
            "to_pool": "creation",
            "expected_baseline_digest": base["digest"],
            "allow_mock_gpu": True,
        },
        exec_root1,
        home_dir=fake_home,
    )
    assert res1["state"] == "aborted_unsynced"
    assert res1["cleaned"] is False
    assert res1["error_code"] == "native_unsynced_data"
    assert res1["teardown_report"]["reason"] == "rclone_exited_before_flush_check"
    assert cache_file.is_file()

    # 2. clean_native_only with cached files present and RC port free also fails closed.
    exec_root2 = studio_root / "teardown-clean-only-cache"
    exec_root2.mkdir()
    res2 = runtime_worker.run_teardown(
        {
            "from_pool": "native",
            "to_pool": "native",
            "clean_native_only": True,
            "expected_baseline_digest": base["digest"],
            "allow_mock_gpu": True,
        },
        exec_root2,
        home_dir=fake_home,
    )
    assert res2["state"] == "aborted_unsynced"
    assert res2["cleaned"] is False
    assert res2["teardown_report"]["reason"] == "rclone_rc_unavailable_with_cache"
    assert cache_file.is_file()

    # 3. Normal native transfer where rclone refuses to exit cleanly on SIGTERM also aborts and preserves cache.
    monkeypatch.setattr(runtime_worker, "_check_port_free", lambda _p: False)
    monkeypatch.setattr(runtime_worker, "_wait_rclone_drained", lambda *a, **kw: None)
    monkeypatch.setattr(
        runtime_worker,
        "_stop_pid_verified",
        lambda pid, *a, **kw: False if pid == 9104 else True,
    )
    exec_root3 = studio_root / "teardown-rclone-hang"
    exec_root3.mkdir()
    res3 = runtime_worker.run_teardown(
        {
            "from_pool": "native",
            "to_pool": "creation",
            "expected_baseline_digest": base["digest"],
            "allow_mock_gpu": True,
        },
        exec_root3,
        home_dir=fake_home,
    )
    assert res3["state"] == "aborted_unsynced"
    assert res3["cleaned"] is False
    assert res3["teardown_report"]["reason"] == "rclone_stop_timeout"
    assert cache_file.is_file()

    # 4. clean_native_only when rclone never started (no rclone-state.json, no mount, no vfs-cache files) cleans cleanly.
    shutil.rmtree(native_dir)
    native_dir.mkdir(parents=True)
    (native_dir / "partial-clone.txt").write_text("residual", encoding="utf-8")
    monkeypatch.setattr(runtime_worker, "_stop_pid_verified", lambda *a, **kw: True)
    monkeypatch.setattr(runtime_worker, "_check_port_free", lambda _p: True)
    exec_root4 = studio_root / "teardown-clean-only-early"
    exec_root4.mkdir()
    res4 = runtime_worker.run_teardown(
        {
            "from_pool": "native",
            "to_pool": "native",
            "clean_native_only": True,
            "expected_baseline_digest": base["digest"],
            "allow_mock_gpu": True,
        },
        exec_root4,
        home_dir=fake_home,
    )
    assert res4["state"] == "completed"
    assert res4["cleaned"] is True
    assert not native_dir.exists()
