"""Regression tests for Batch 3 performance optimizations."""

from __future__ import annotations

import hashlib
import sys
import threading
import time
from types import SimpleNamespace

import httpx
from app.db import Store
from app.gpu import drive as drive_mod
from app.gpu import model_library, native, native_models, native_runtime, node_persistence
from app.gpu.repository import Repository


def test_node_persistence_status_cache_avoids_redundant_inventory(tmp_path, monkeypatch):
    root = tmp_path / "native"
    comfy = tmp_path / "ComfyUI"
    nodes = comfy / "custom_nodes"
    plugin = nodes / "demo-plugin"
    plugin.mkdir(parents=True)
    (plugin / "__init__.py").write_text("NODE_CLASS_MAPPINGS = {}\n", encoding="utf-8")

    p = node_persistence.Persistence(root, comfy, tmp_path / "user")
    inv_calls = 0
    orig_inv = node_persistence.inventory

    def counted_inventory(*args, **kwargs):
        nonlocal inv_calls
        inv_calls += 1
        return orig_inv(*args, **kwargs)

    monkeypatch.setattr(node_persistence, "inventory", counted_inventory)
    # tick() runs inventory once and populates the stat cache.
    p.tick(busy=True, now=100.0)
    assert inv_calls == 1
    # Status reuses tick's fingerprint after checking the current tree stamp.
    st1 = p.status(use_cache=True)
    assert inv_calls == 1
    assert st1["durable"] is False
    # External /studio_native/state status(use_cache=True) hits the tree-stat cache.
    p.status(use_cache=True)
    p.status(use_cache=True)
    assert inv_calls == 1
    # Modifying a file invalidates the tree-stat cache immediately.
    (plugin / "extra.py").write_text("X = 1\n", encoding="utf-8")
    p.status(use_cache=True)
    assert inv_calls == 2


def test_protect_lease_fetches_studio_native_state_only_once(monkeypatch):
    store = Store("sqlite:///:memory:")
    repo = Repository(store)
    inst = repo.create("instances", "owner-1", {"state": "retained_idle", "account_id": "acc-1", "native_comfy": True})
    lease = repo.create(
        "leases",
        "owner-1",
        {
            "state": "idle",
            "provider": "colab",
            "account_id": "acc-1",
            "instance_id": inst["id"],
            "native_comfy": True,
            "native_ready": True,
            "deadline_epoch": time.time() + 3600,
        },
    )
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path == "/studio_native/state":
            return httpx.Response(200, json={"busy": True, "plugins": {"durable": True, "error": None}})
        if request.url.path == "/queue":
            return httpx.Response(200, json={"queue_running": [], "queue_pending": []})
        return httpx.Response(404)

    monkeypatch.setenv("GPU_REAL_ENABLED", "1")
    monkeypatch.setattr(native, "backend_for", lambda *a, **kw: ("https://native.example", {"CF-Test": "1"}))
    monkeypatch.setattr(native, "keepalive_lease", lambda *a, **kw: None)
    orig_client = httpx.Client
    monkeypatch.setattr(native.httpx, "Client", lambda **kw: orig_client(transport=httpx.MockTransport(handler), **kw))

    assert native.protect_lease(repo, lease, check_remote=False) is True
    assert paths.count("/studio_native/state") == 1
    assert paths.count("/queue") == 1


def test_native_models_single_pass_hash_and_persisted_hash_cache(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    payload = b"safetensors-demo-weight-bytes" * 1024
    expected_sha = hashlib.sha256(payload).hexdigest()
    expected_md5 = hashlib.md5(payload).hexdigest()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            stream=httpx.ByteStream(payload),
            headers={
                "Content-Length": str(len(payload)),
                "Content-Type": "application/octet-stream",
                "ETag": '"etag-1"',
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    lm = native_models.LocalModels(root, models_dir, {"channel": "c"}, http=client)
    assert lm.executor._max_workers == 4

    # Assert file_hashes is NEVER called during a clean single-pass transfer.
    monkeypatch.setattr(
        lm,
        "file_hashes",
        lambda _path: (_ for _ in ()).throw(AssertionError("file_hashes should not re-read on clean transfer")),
    )
    target = lm.target("checkpoints", "demo.safetensors")
    plan = {"url": "https://huggingface.co/org/repo/resolve/main/demo.safetensors", "sha256": expected_sha, "md5": expected_md5, "size": len(payload), "drive": None}
    lm.transfer(plan["url"], target, plan, lambda **_kw: None)
    assert target.is_file()
    cache_file = root / ".model-hash-cache.json"
    assert cache_file.is_file()
    assert plan["url"] not in cache_file.read_text(encoding="utf-8")

    # A restarted LocalModels instance loads the persisted hash cache and skips re-hashing.
    lm2 = native_models.LocalModels(root, models_dir, {"channel": "c"}, http=client)
    monkeypatch.setattr(
        lm2,
        "file_hashes",
        lambda _path: (_ for _ in ()).throw(AssertionError("restarted LocalModels should hit persisted hash cache")),
    )
    assert lm2.local_hit(target, plan) is True


def test_model_library_single_pass_download_skips_second_hash_file(tmp_path, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    store = Store("sqlite:///:memory:")
    repo = Repository(store)
    content = b"A" * (128 * 1024)
    sha = hashlib.sha256(content).hexdigest()
    md5 = hashlib.md5(content).hexdigest()

    model = repo.create(
        "model_library",
        "owner-1",
        {
            "filename": "test.gguf",
            "model_type": "diffusion_models",
            "source": "huggingface",
            "download_url": "https://huggingface.co/a/b/resolve/main/test.gguf",
            "expected_sha256": sha,
            "expected_size": len(content),
            "state": "pending",
        },
    )
    task = repo.create("model_downloads", "owner-1", {"model_id": model["id"], "state": "pending"})

    class _FakeResp:
        status_code = 200
        headers = {"content-length": str(len(content)), "content-encoding": "identity", "etag": '"e1"'}
        extensions = {"studio_source_resource": "res1"}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def iter_bytes(self, size):
            yield content

    class _FakeHttp:
        def stream(self, *_a, **_kw):
            return _FakeResp()

        def close(self):
            pass

    lib = model_library.ModelLibrary(repo, "owner-1", http=_FakeHttp())
    transfer = model_library.ModelTransfer(lib, task)
    out_path = tmp_path / "w.partial"
    result = transfer.download(out_path)
    assert result is not None
    got_sha, got_md5, got_size = result
    assert got_sha.hexdigest() == sha
    assert got_md5.hexdigest() == md5
    assert got_size == len(content)


def test_drive_verify_uses_server_sha256_checksum_without_redownload():
    store = Store("sqlite:///:memory:")
    repo = Repository(store)
    content = b"verified-payload-bytes"
    sha = hashlib.sha256(content).hexdigest()

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"Should not download media when sha256Checksum is present: {request.url}")

    storage = drive_mod.DriveStorage(
        {"id": "job-1", "owner_id": "owner-1"},
        repo,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    record = {"id": "f-1", "size": len(content), "sha256": sha, "sha256Checksum": sha}
    assert storage.verify(record, content) is True
    bad = {"id": "f-1", "size": len(content), "sha256": sha, "sha256Checksum": "0" * 64}
    assert storage.verify(bad, content) is False


def test_native_runtime_hides_apt_behind_plugins_and_python(tmp_path, monkeypatch):
    # Verify _install_system_apt runs concurrently with _install_base_python rather than blocking before it.
    root = tmp_path / "native"
    comfy = root / "ComfyUI"
    comfy.mkdir(parents=True)
    (comfy / "requirements.txt").write_text("", encoding="utf-8")
    (comfy / "manager_requirements.txt").write_text("", encoding="utf-8")
    drive = tmp_path / "drive"
    drive.mkdir()

    monkeypatch.setattr(native_runtime, "ROOT", root)
    monkeypatch.setattr(native_runtime, "install_rclone", lambda *a: root / "rclone")
    monkeypatch.setattr(native_runtime, "install_cloudflared", lambda *a: "/usr/bin/cloudflared")
    monkeypatch.setattr(native_runtime, "_ensure_comfy_checkout", lambda *a: None)
    monkeypatch.setattr(native_runtime, "mount_drive", lambda *a: drive)
    monkeypatch.setattr(native_runtime, "wait_for_comfy", lambda *a, **kw: None)
    monkeypatch.setattr(native_runtime, "verify_manager", lambda *a, **kw: None)
    monkeypatch.setattr(native_runtime, "wait_for_tunnel", lambda *a, **kw: None)

    apt_started = threading.Event()
    apt_finished = threading.Event()
    py_observed_concurrent_apt = []

    def slow_apt(_worker):
        apt_started.set()
        time.sleep(0.15)
        apt_finished.set()

    def fast_py(*_a, **_kw):
        assert apt_started.wait(timeout=1.0)
        py_observed_concurrent_apt.append(not apt_finished.is_set())

    monkeypatch.setattr(native_runtime, "_install_system_apt", slow_apt)
    monkeypatch.setattr(native_runtime, "_install_base_python", fast_py)
    monkeypatch.setattr(
        node_persistence,
        "Persistence",
        lambda *a, **kw: SimpleNamespace(
            remote=SimpleNamespace(available=lambda: False),
            restore=lambda *_a, **_kw: {},
            dependencies=lambda *_a, **_kw: None,
            try_restore_env_cache=lambda *_a, **_kw: (False, "miss"),
            env_cache_should_upload=lambda *_a, **_kw: False,
            record_env_cache_key=lambda *_a, **_kw: None,
        ),
    )
    monkeypatch.setattr(node_persistence, "protect_environment", lambda *_a, **_kw: None)
    monkeypatch.setitem(sys.modules, "node_persistence", node_persistence)
    monkeypatch.setattr(
        native_runtime.subprocess,
        "Popen",
        lambda args, **kw: SimpleNamespace(pid=42, poll=lambda: None, terminate=lambda: None, wait=lambda **k: 0),
    )

    from app.gpu import runtime_worker as worker

    monkeypatch.setitem(sys.modules, "runtime_worker", worker)
    monkeypatch.setattr(worker, "_process_metadata", lambda *a: {"start_id": "1"})
    monkeypatch.setattr(worker, "_site_dir", lambda: tmp_path / "nosite")

    stages = SimpleNamespace(start=lambda _name: None, end=lambda: None, timings={})
    payload = {
        "snapshot": {"manifest": {"comfy_commit": "a" * 40}},
        "native_secrets": {
            "max_upload_mb": 2048,
            "tunnel_token": "fake-token",
            "model_storage_mode": "local",
            "catalog": [],
            "drive_auth": {},
            "drive_account": "acct",
            "models_folder_id": "models",
            "model_channel": "ch",
            "directories": {},
            "rules": {},
        },
    }
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    t0 = time.monotonic()
    native_runtime.run(job_dir, payload, stages)
    elapsed = time.monotonic() - t0
    assert py_observed_concurrent_apt == [True]
    assert apt_finished.is_set()
    assert elapsed < 2.0
