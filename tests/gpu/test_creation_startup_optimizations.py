"""Unit tests for Creation Mode GPU startup & control-plane/tail optimizations."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import tarfile
import threading
import time
import uuid
from types import SimpleNamespace

import httpx
import pytest
from app.db import Store
from app.gpu import mcp
from app.gpu import runtime_worker as worker
from app.gpu.drive import DriveStorage
from app.gpu.repository import Repository
from app.gpu.scheduler import Scheduler
from app.job_history import job_timings

COMMIT = "a" * 40
FILE_ID = "f" * 32


class _FakeRangeResponse:
    def __init__(self, status: int, body: bytes, headers: dict[str, str] | None = None):
        self.status = status
        self.headers = headers or {}
        self._buf = io.BytesIO(body)

    def read(self, n: int = -1) -> bytes:
        return self._buf.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_pip_install_uses_uv_with_constraints_and_falls_back_to_pip(tmp_path, monkeypatch):
    comfy = tmp_path / "ComfyUI"
    comfy.mkdir()
    (comfy / "requirements.txt").write_text("einops\n", encoding="utf-8")
    monkeypatch.setenv("CREATION_USE_UV", "1")
    monkeypatch.setattr(worker.shutil, "which", lambda name: "/usr/local/bin/uv" if name == "uv" else None)

    calls: list[list[str]] = []

    def fake_run(args, *, cwd=None, timeout=60, capture=True, env=None):
        calls.append(args)
        assert env and "PIP_CONSTRAINT" in env and "UV_CONSTRAINT" in env
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(worker, "_run", fake_run)
    info = worker._pip_install(comfy, 120, root=tmp_path)
    assert info["installer"] == "uv"
    assert len(calls) == 1
    assert calls[0][0] == "/usr/local/bin/uv"
    assert "--constraint" in calls[0]
    assert "huggingface_hub" in calls[0]
    assert str(comfy / "requirements.txt") in calls[0]

    # Automatic fallback to merged pip when uv fails.
    calls.clear()

    def uv_fails(args, *, cwd=None, timeout=60, capture=True, env=None):
        calls.append(args)
        if args[0].endswith("uv"):
            raise worker.WorkerError("uv failed")
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(worker, "_run", uv_fails)
    info2 = worker._pip_install(comfy, 120, root=tmp_path)
    assert info2["installer"] == "pip"
    assert len(calls) == 2
    assert calls[1][1:4] == ["-m", "pip", "install"]
    assert "-c" in calls[1]
    assert "huggingface_hub" in calls[1]


def test_pack_env_v2_excludes_pycache_and_protected_torch_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(worker.shutil, "which", lambda _: None)
    comfy = tmp_path / "ComfyUI"
    comfy.mkdir()
    (comfy / "main.py").write_text("print('ok')\n")
    (comfy / "__pycache__").mkdir()
    (comfy / "__pycache__" / "main.cpython-311.pyc").write_bytes(b"pyc")

    site = tmp_path / "site"
    site.mkdir()
    (site / "torch").mkdir()
    (site / "torch" / "__init__.py").write_text("# preinstalled")
    (site / "torch-2.8.0.dist-info").mkdir()
    (site / "torch-2.8.0.dist-info" / "METADATA").write_text("Name: torch\nVersion: 2.8.0\n")

    before = worker._site_dist_state(site)

    # Simulate importing torch writing __pycache__ inside preinstalled torch dir (must NOT be packed!)
    (site / "torch" / "__pycache__").mkdir()
    (site / "torch" / "__pycache__" / "__init__.cpython-311.pyc").write_bytes(b"bytecode")
    # Simulate newly installed regular package
    (site / "demo_pkg").mkdir()
    (site / "demo_pkg" / "__init__.py").write_text("VALUE = 1\n")
    (site / "demo_pkg" / "__pycache__").mkdir()
    (site / "demo_pkg" / "__pycache__" / "__init__.cpython-311.pyc").write_bytes(b"skip")
    (site / "demo_pkg-1.0.dist-info").mkdir()
    (site / "demo_pkg-1.0.dist-info" / "METADATA").write_text("Name: demo_pkg\nVersion: 1.0\n")
    (site / "demo_pkg-1.0.dist-info" / "RECORD").write_text("demo_pkg/__init__.py,,\n")

    out = tmp_path / "env.tar.gz"
    stats = worker._pack_env(comfy, site, before, "v2-key", out)
    assert stats["site_entries_added"] == 2
    with tarfile.open(out, "r:gz") as tar:
        names = tar.getnames()
    assert not any("torch" in n for n in names)
    assert not any("__pycache__" in n or n.endswith(".pyc") for n in names)
    assert "site/demo_pkg/__init__.py" in names
    assert "site/demo_pkg-1.0.dist-info/METADATA" in names


def test_drive_cache_multistream_range_single_pass_dual_hash_and_retry(tmp_path, monkeypatch):
    monkeypatch.setenv("STUDIO_OFFLINE_TEST", "1")
    monkeypatch.setenv("CREATION_MULTI_STREAM_DRIVE", "1")
    monkeypatch.setenv("CREATION_DRIVE_STREAMS", "4")
    monkeypatch.setattr(worker, "COMFY_SERVICE_ROOT", tmp_path / "services")

    payload = (b"0123456789abcdef" * (256 * 1024)) + b"tail-bytes"  # ~4 MiB + 10B
    expected_sha = hashlib.sha256(payload).hexdigest()
    expected_md5 = hashlib.md5(payload, usedforsecurity=False).hexdigest()

    cache = worker.DriveCache("t" * 32, "f" * 32)
    cache._part_size = 1024 * 1024  # 1 MiB parts -> 5 parts
    cache._min_multistream_bytes = 2 * 1024 * 1024

    lock = threading.Lock()
    seen_ranges: list[str] = []
    failed_once = {"part2": False}

    def fake_open(method, url, *, body=None, headers=None, timeout=120):
        rng = (headers or {}).get("Range", "")
        with lock:
            seen_ranges.append(rng)
            if rng.startswith("bytes=2097152-") and not failed_once["part2"]:
                failed_once["part2"] = True
                raise ConnectionResetError("transient reset on part 2")
        assert rng.startswith("bytes=")
        s_str, e_str = rng[6:].split("-", 1)
        start, end = int(s_str), int(e_str)
        chunk = payload[start : end + 1]
        return _FakeRangeResponse(
            206,
            chunk,
            headers={"Content-Range": f"bytes {start}-{end}/{len(payload)}"},
        )

    monkeypatch.setattr(cache, "_open", fake_open)
    entry = {"id": FILE_ID, "size": str(len(payload)), "md5Checksum": expected_md5}
    dest = tmp_path / "weight.safetensors"

    sha, md5 = cache.fetch(entry, dest, expected_sha256=expected_sha)
    assert sha == expected_sha
    assert md5 == expected_md5
    assert dest.read_bytes() == payload
    assert len(seen_ranges) == 6  # 5 parts + 1 retry on part 2

    # Local stamp hash cache must now hit without calling _sha256 again!
    monkeypatch.setattr(worker, "_sha256", lambda _: pytest.fail("should hit stamp hash cache"))
    assert worker._local_model_hit(dest, len(payload), expected_sha) is True


def test_drive_cache_multistream_falls_back_to_single_on_http_200(tmp_path, monkeypatch):
    monkeypatch.setenv("STUDIO_OFFLINE_TEST", "1")
    monkeypatch.setenv("CREATION_MULTI_STREAM_DRIVE", "1")
    monkeypatch.setattr(worker, "COMFY_SERVICE_ROOT", tmp_path / "services")

    payload = b"z" * (3 * 1024 * 1024)
    expected_sha = hashlib.sha256(payload).hexdigest()
    expected_md5 = hashlib.md5(payload, usedforsecurity=False).hexdigest()

    cache = worker.DriveCache("t" * 32, "f" * 32)
    cache._part_size = 1024 * 1024
    cache._min_multistream_bytes = 2 * 1024 * 1024

    def fake_open(method, url, *, body=None, headers=None, timeout=120):
        # Ignore Range and return full 200 -> triggers seamless fallback to single-stream
        return _FakeRangeResponse(200, payload)

    monkeypatch.setattr(cache, "_open", fake_open)
    dest = tmp_path / "fallback.safetensors"
    sha, md5 = cache.fetch({"id": FILE_ID, "size": str(len(payload)), "md5Checksum": expected_md5}, dest, expected_sha256=expected_sha)
    assert sha == expected_sha and md5 == expected_md5
    assert dest.read_bytes() == payload


def test_prep_skips_drive_weight_download_and_scheduler_skips_all_drive_prep(tmp_path, monkeypatch):
    # 1. Scheduler._await_prep skips CPU prep when every weight is already source=drive
    repo = Repository(Store(f"sqlite:///{tmp_path / 'db.sqlite'}"))
    sched = Scheduler(repo, clock=lambda: 100.0)
    drive_job = {
        "id": "job-drive",
        "owner_id": "owner",
        "provider": "colab",
        "snapshot": {
            "workflow_sha256": "wf-sha",
            "manifest": {
                "comfy_commit": COMMIT,
                "weight_files": [
                    {"source": "drive", "drive_file_id": FILE_ID, "target": "checkpoints/a.safetensors", "size": 10, "sha256": "a" * 64}
                ],
            },
        },
    }
    monkeypatch.setenv("CREATION_SKIP_DRIVE_PREP", "1")
    assert sched._await_prep(drive_job) is False
    assert repo.list("jobs", "owner") == []

    # When disabled, it falls back to creating a prep job
    monkeypatch.setenv("CREATION_SKIP_DRIVE_PREP", "0")
    drive_job_with_meta = repo.create(
        "jobs",
        "owner",
        {**drive_job, "project_id": "p1", "model_id": "m1", "workflow_key": "wf", "workflow_version": 1},
        record_id="job-drive",
    )
    assert sched._await_prep(drive_job_with_meta) is True
    assert len(repo.list("jobs", "owner")) == 2  # original job + newly created prep job

    # 2. _restore_weights(prep=True) for source=drive verifies metadata ONLY, never downloads bytes!
    cache = worker.DriveCache("t" * 32, "f" * 32)
    monkeypatch.setattr(
        cache,
        "_json",
        lambda method, url: {
            "id": FILE_ID, "size": "10", "md5Checksum": "0" * 32,
            "mimeType": "application/octet-stream", "trashed": False,
        },
    )
    monkeypatch.setattr(cache, "fetch", lambda *a, **k: pytest.fail("prep must NOT download existing Drive weight"))
    records = worker._restore_weights(
        tmp_path / "ComfyUI",
        {"weight_files": [{"source": "drive", "drive_file_id": FILE_ID, "target": "checkpoints/a.safetensors", "size": 10, "sha256": "a" * 64}]},
        60,
        cache,
        prep=True,
    )
    assert records[0]["source"] == "drive" and records[0]["cache_stored"] is True


def test_in_place_reuse_on_lora_switch_and_lru_base_weight_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "COMFY_SERVICE_ROOT", tmp_path / "services")
    service_dir = tmp_path / "services" / "inst-1"
    comfy = service_dir / "ComfyUI"
    comfy.mkdir(parents=True)
    (comfy / "main.py").write_text("# comfy\n")

    w1_bytes = b"base-model-one"
    w2_bytes = b"base-model-two"
    lora1_bytes = b"user-lora-one"
    lora2_bytes = b"user-lora-two"
    w1 = {"source": "drive", "drive_file_id": "1" * 32, "target": "checkpoints/w1.safetensors",
          "size": len(w1_bytes), "sha256": hashlib.sha256(w1_bytes).hexdigest()}
    w2 = {"source": "drive", "drive_file_id": "2" * 32, "target": "checkpoints/w2.safetensors",
          "size": len(w2_bytes), "sha256": hashlib.sha256(w2_bytes).hexdigest()}
    lora2 = {"target": "user/l2.safetensors", "id": "4" * 32, "size": len(lora2_bytes),
             "md5Checksum": hashlib.md5(lora2_bytes, usedforsecurity=False).hexdigest()}

    # Pre-seed ComfyUI with w1 + lora1 + env-state matching COMMIT (simulating an already-warmed instance)
    (comfy / "models" / "checkpoints").mkdir(parents=True)
    (comfy / "models" / "checkpoints" / "w1.safetensors").write_bytes(w1_bytes)
    (comfy / "models" / "loras" / "user").mkdir(parents=True)
    (comfy / "models" / "loras" / "user" / "l1.safetensors").write_bytes(lora1_bytes)
    manifest1 = {"comfy_commit": COMMIT, "weight_files": [w1]}
    code_fp = worker._code_env_fingerprint(COMMIT, [], manifest1)
    worker._write_env_state(service_dir, code_fp, COMMIT)

    monkeypatch.setattr(worker, "_verify_checkout", lambda path, commit: None)
    monkeypatch.setattr(worker, "_verify_custom_nodes", lambda path, nodes: None)
    monkeypatch.setattr(worker, "_checkout_comfy", lambda *a, **k: pytest.fail("in-place reuse must skip git checkout"))
    monkeypatch.setattr(worker, "_pip_install", lambda *a, **k: pytest.fail("in-place reuse must skip pip/uv install"))

    restarted: list[str] = []

    def fake_start(sdir, fp, reason, *, deadline=None):
        restarted.append(reason)
        return {"pid": 5001, "process_start_id": "10", "env_fingerprint": fp, "cache_reason": reason}

    monkeypatch.setattr(worker, "_start_comfy_service", fake_start)

    # Pre-place lora2 and w2 download stubs
    def fake_restore_weights(c, manifest, timeout, cache, prep, **kwargs):
        for item in manifest["weight_files"]:
            dst = c / "models" / item["target"]
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(w2_bytes if item["target"].endswith("w2.safetensors") else w1_bytes)
        return [{"target": item["target"], "source": "local"} for item in manifest["weight_files"]]

    def fake_restore_loras(c, loras, cache, **kwargs):
        for item in loras:
            dst = c / "models" / "loras" / item["target"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(lora2_bytes if item["target"].endswith("l2.safetensors") else lora1_bytes)
        return []

    monkeypatch.setattr(worker, "_restore_weights", fake_restore_weights)
    monkeypatch.setattr(worker, "_restore_user_loras", fake_restore_loras)

    # Switch to workflow 2 + lora 2 on the same ComfyUI commit!
    manifest2 = {"comfy_commit": COMMIT, "weight_files": [w2]}
    stages = worker._Stages(tmp_path / "exec2")
    service, env, _ = worker._creation_service(
        service_dir, {"use_drive_cache": False}, manifest2, COMMIT, [], [w2], [lora2], stages, time.time() + 60
    )
    assert env["source"] == "in_place_reuse"
    assert service["cache_reason"] == "in_place_restart"
    assert restarted == ["in_place_restart"]
    # Strict user LoRA isolation: l1 removed, l2 present!
    assert not (comfy / "models/loras/user/l1.safetensors").exists()
    assert (comfy / "models/loras/user/l2.safetensors").is_file()
    # Base weight LRU caching: w1 preserved alongside w2 while disk has room!
    assert (comfy / "models/checkpoints/w1.safetensors").is_file()
    assert (comfy / "models/checkpoints/w2.safetensors").is_file()

    # Now simulate low disk space: LRU must evict older w1 while keeping w2!
    worker._atomic_json(service_dir / worker._MODEL_LRU_FILE, {"checkpoints/w1.safetensors": 100.0, "checkpoints/w2.safetensors": 200.0})
    monkeypatch.setattr(worker.shutil, "disk_usage", lambda _: SimpleNamespace(free=100))
    evicted = worker._evict_unused_weights_if_needed(
        service_dir, comfy / "models", {"checkpoints/w2.safetensors"}, needed_bytes=50, reserve_bytes=60
    )
    assert evicted == ["checkpoints/w1.safetensors"]
    assert not (comfy / "models/checkpoints/w1.safetensors").exists()
    assert (comfy / "models/checkpoints/w2.safetensors").is_file()


def test_drive_storage_caches_subfolder_ids_in_repo(monkeypatch):
    monkeypatch.setenv("CREATION_FAST_CONTROL_PLANE", "1")
    from tests.gpu.test_drive_contract import Repo

    repo = Repo()
    get_queries: list[str] = []
    ids = {"AI-Studio": "root-1", "owner-1": "owner-f", "models": "models-f", "runtime-cache": "rc-f", "loras": "loras-f", "h3": "h3-f"}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/drive/v3/files" and request.method == "GET":
            q = request.url.params["q"]
            get_queries.append(q)
            name = q.split("name = '", 1)[1].split("'", 1)[0]
            return httpx.Response(200, json={"files": [{"id": ids[name], "name": name, "mimeType": "application/vnd.google-apps.folder"}]})
        raise AssertionError(f"unexpected {request.method} {request.url}")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    first = DriveStorage({"id": "j1", "owner_id": "owner-1"}, repo, client=client)
    c1 = first.model_cache()
    l1 = first.lora_folder()
    assert c1["folder_id"] == "models-f" and c1["runtime_folder_id"] == "rc-f" and l1 == "h3-f"
    first_calls = len(get_queries)
    assert first_calls == 6  # root + owner + models + runtime-cache + loras + h3

    # Second DriveStorage instance on the same repo must hit DB cache with ZERO Drive API calls!
    second = DriveStorage({"id": "j2", "owner_id": "owner-1"}, repo, client=client)
    c2 = second.model_cache()
    l2 = second.lora_folder()
    assert c2["folder_id"] == "models-f" and c2["runtime_folder_id"] == "rc-f" and l2 == "h3-f"
    assert len(get_queries) == first_calls


def test_lean_bootstrap_and_large_chunk_code_and_job_timings(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("CREATION_FAST_CONTROL_PLANE", "1")
    eid = str(uuid.uuid4())
    creation_code = mcp.build_bootstrap_code(eid, {"snapshot": {"creation_api": True}}, worker_source="print(1)")
    native_code = mcp.build_bootstrap_code(eid, {"snapshot": {"native_comfy": True}}, worker_source="print(1)")
    assert "node_persistence.py" not in creation_code and "native_runtime.py" not in creation_code
    assert "node_persistence.py" in native_code and "native_runtime.py" in native_code

    # 4 MiB chunk reader reads exact requested chunk_bytes
    root = tmp_path / eid
    root.mkdir()
    blob = b"A" * (3 * 1024 * 1024) + b"END"
    (root / "out.png").write_bytes(blob)
    (root / "meta.json").write_text(json.dumps({"execution_id": eid, "artifacts": [{"path": "out.png"}]}))
    chunk_code = mcp.build_chunk_code(eid, 0, 0, chunk_bytes=4 * 1024 * 1024).replace("/content/ai-studio", str(tmp_path))
    exec(compile(chunk_code, "chunk-4m", "exec"), {})
    out = json.loads(capsys.readouterr().out)
    assert base64.b64decode(out["content"]) == blob

    # job_timings preserves new creation stages
    timings = job_timings(
        {"state": "completed", "worker_timings": {"restore_env": 2.4, "creation_python": 5.1, "restore_weights": 18.0, "restore_loras": 1.2, "creation_warming": 9.5}},
        [],
    )
    assert timings == {
        "restore_env": 2.4,
        "creation_python": 5.1,
        "restore_weights": 18.0,
        "restore_loras": 1.2,
        "creation_warming": 9.5,
    }


def test_parallel_cold_bootstrap_overlaps_uv_and_weights_and_writes_env_state(tmp_path, monkeypatch):
    monkeypatch.setenv("CREATION_PARALLEL_BOOTSTRAP", "1")
    monkeypatch.setenv("CREATION_ENV_CACHE", "auto")
    monkeypatch.setattr(worker, "COMFY_SERVICE_ROOT", tmp_path / "services")
    service_dir = tmp_path / "services" / "inst-par"
    w_bytes = b"parallel-weight-data"
    w = {
        "source": "drive", "drive_file_id": FILE_ID, "target": "checkpoints/p.safetensors",
        "size": len(w_bytes), "sha256": hashlib.sha256(w_bytes).hexdigest(),
    }
    manifest = {"comfy_commit": COMMIT, "weight_files": [w]}

    barrier = threading.Barrier(2, timeout=3.0)
    overlapped = {"ok": False}

    def fake_checkout(comfy, commit, timeout):
        comfy.mkdir(parents=True, exist_ok=True)
        (comfy / "main.py").write_text("# comfy\n")
        (comfy / "requirements.txt").write_text("einops\n")

    def fake_pip(comfy, timeout, *, root=None, custom_nodes=None):
        barrier.wait()
        overlapped["ok"] = True
        return {"installer": "uv", "python_seconds": 0.1}

    def fake_weights(comfy, manifest_arg, timeout, cache, prep, **kwargs):
        barrier.wait()
        dst = comfy / "models" / w["target"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(w_bytes)
        return [{"target": w["target"], "source": "drive"}]

    monkeypatch.setattr(worker, "_checkout_comfy", fake_checkout)
    monkeypatch.setattr(worker, "_verify_checkout", lambda c, cm: None)
    monkeypatch.setattr(worker, "_pip_install", fake_pip)
    monkeypatch.setattr(worker, "_restore_weights", fake_weights)
    monkeypatch.setattr(
        worker,
        "_start_comfy_service",
        lambda sdir, fp, reason, *, deadline=None: {"pid": 6001, "process_start_id": "1", "env_fingerprint": fp, "cache_reason": reason},
    )

    stages = worker._Stages(tmp_path / "exec-par")
    service, env, weights = worker._creation_service(
        service_dir, {"use_drive_cache": False}, manifest, COMMIT, [], [w], [], stages, time.time() + 60
    )
    assert overlapped["ok"] is True
    assert env["installer"] == "uv"
    assert service["cache_reason"] == "cold_start"
    env_state = worker._read_env_state(service_dir)
    assert env_state and env_state["code_env_fingerprint"] == worker._code_env_fingerprint(COMMIT, [], manifest)


def test_tunnel_artifact_streaming_reuses_single_ws_with_4mib_chunks(tmp_path, monkeypatch):
    import websockets.sync.client as ws_sync
    from app.gpu.artifacts import LocalArtifactStore

    eid = str(uuid.uuid4())
    remote = tmp_path / "remote"
    root = remote / eid
    root.mkdir(parents=True)
    blob = b"01234567" * (700 * 1024)  # ~5.6 MiB -> 2 chunks at 4 MiB each
    (root / "video.mp4").write_bytes(blob)
    (root / "meta.json").write_text(json.dumps({"execution_id": eid, "artifacts": [{"path": "video.mp4"}]}))

    ws_opens = 0
    ws_sends = 0

    class FakeWS:
        def __init__(self):
            self._queue: list[bytes] = []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def send(self, raw: bytes):
            nonlocal ws_sends
            ws_sends += 1
            msg = mcp.MCPGPUExecutor._ws_unpack_default(raw)
            msg_id = msg["header"]["msg_id"]
            code = msg["content"]["code"].replace("/content/ai-studio", str(remote))
            buf = io.StringIO()
            import contextlib

            with contextlib.redirect_stdout(buf):
                exec(compile(code, "stream-chunk", "exec"), {})
            self._queue.append(
                mcp.MCPGPUExecutor._ws_pack_default(
                    {"header": {"msg_type": "stream"}, "parent_header": {"msg_id": msg_id}, "content": {"name": "stdout", "text": buf.getvalue()}}
                )
            )
            self._queue.append(
                mcp.MCPGPUExecutor._ws_pack_default(
                    {"header": {"msg_type": "execute_reply"}, "parent_header": {"msg_id": msg_id}, "content": {"status": "ok"}}
                )
            )
            self._queue.append(
                mcp.MCPGPUExecutor._ws_pack_default(
                    {"header": {"msg_type": "status"}, "parent_header": {"msg_id": msg_id}, "content": {"execution_state": "idle"}}
                )
            )

        def recv(self, timeout=None):
            return self._queue.pop(0)

    def fake_connect(url, **kwargs):
        nonlocal ws_opens
        ws_opens += 1
        return FakeWS()

    monkeypatch.setattr(ws_sync, "connect", fake_connect)
    monkeypatch.setattr(mcp, "_connect_ipv4_socket", lambda *a, **k: None)

    executor = object.__new__(mcp.MCPGPUExecutor)
    executor.owner_id = "owner"
    executor.timeout = 30.0
    executor._refresh_tunnel_proxy = lambda lease: ("https://proxy.example", "tok")
    objects = LocalArtifactStore(tmp_path / "artifacts")
    sha = hashlib.sha256(blob).hexdigest()
    item = objects.begin("owner", eid, "video.mp4", "video/mp4", len(blob), sha)

    end_offset, reads = executor._stream_tunnel_artifact_chunks(
        {"kernel_id": "k-1"}, eid, 0, len(blob), objects, item["artifact_id"], 0
    )
    assert end_offset == len(blob)
    assert reads == 2
    assert ws_opens == 1  # single persistent WebSocket for all chunks!
    assert ws_sends == 2
    verified = objects.verify(item["artifact_id"], "owner", eid)
    assert verified["sha256"] == sha

