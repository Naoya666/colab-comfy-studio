import hashlib
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from app.gpu import runtime_worker as worker

DATA = b"reviewed model data"
SHA = hashlib.sha256(DATA).hexdigest()
MD5 = hashlib.md5(DATA, usedforsecurity=False).hexdigest()
FILE_ID = "a" * 32
MANIFEST = {
    "state": "published", "reviewed": True, "comfy_commit": "a" * 40, "node_allowlist": ["TestNode"],
}
HF = {"target": "checkpoints/model.safetensors", "repo_id": "test/model", "filename": "source/model.safetensors",
      "revision": "b" * 40, "sha256": SHA, "size": len(DATA)}
DRIVE = {"target": "vae/drive.safetensors", "source": "drive", "drive_file_id": FILE_ID,
         "sha256": SHA, "size": len(DATA)}
LORA = {"target": "user/user.safetensors", "id": FILE_ID, "md5Checksum": MD5, "size": len(DATA)}


def fake_sources(monkeypatch, *, cached=False, media=DATA, stored_md5=MD5, hf_data=DATA):
    calls = []
    cache = worker.DriveCache("t" * 32, "f" * 32)
    entry = {"id": FILE_ID, "size": str(len(DATA)), "md5Checksum": stored_md5,
             "mimeType": "application/octet-stream", "trashed": False}

    def lookup(target, sha, size):
        calls.append(("lookup", target, sha, size))
        return entry if cached else None

    def metadata(method, url):
        calls.append(("metadata", method, url))
        return entry

    def download(method, url, **kwargs):
        calls.append(("media", method, url))
        return io.BytesIO(media)

    def hf(repo, filename, *, revision, local_dir):
        calls.append(("hf", repo, filename, revision))
        destination = Path(local_dir) / filename
        destination.parent.mkdir(parents=True)
        destination.write_bytes(hf_data)
        return str(destination)

    monkeypatch.setattr(cache, "lookup", lookup)
    monkeypatch.setattr(cache, "_json", metadata)
    monkeypatch.setattr(cache, "_open", download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=hf))
    return cache, calls


@pytest.mark.parametrize("cached", [True, False])
def test_drive_cache_precedes_pinned_hugging_face_and_local_hit(tmp_path, monkeypatch, cached):
    cache, calls = fake_sources(monkeypatch, cached=cached)
    manifest = {"weight_files": [HF]}
    records = worker._restore_weights(tmp_path, manifest, 60, cache, prep=False)
    target = tmp_path / "models" / HF["target"]
    assert target.read_bytes() == DATA and not target.is_symlink()
    assert records[0]["source"] == ("drive" if cached else "huggingface")
    assert calls[0] == ("lookup", HF["target"], SHA, len(DATA))
    assert len(calls) == 2
    if cached:
        assert calls[1][0] == "media" and calls[1][2].endswith("?alt=media")
    else:
        assert calls[1] == ("hf", HF["repo_id"], HF["filename"], HF["revision"])
    calls.clear()
    assert worker._restore_weights(tmp_path, manifest, 60, cache, prep=False)[0]["source"] == "local"
    assert not calls
    next_comfy = tmp_path / "next"
    assert worker._restore_weights(next_comfy, manifest, 60, cache, prep=False,
                                   warm_models=target.parents[1])[0]["source"] == "local"
    assert (next_comfy / "models" / HF["target"]).read_bytes() == DATA
    assert not calls


def test_drive_weights_and_user_loras_use_file_ids(tmp_path, monkeypatch):
    cache, calls = fake_sources(monkeypatch)
    assert worker._restore_weights(tmp_path, {"weight_files": [DRIVE]}, 60, cache, prep=False,
                                   use_drive_cache=False)[0]["source"] == "drive"
    assert worker._restore_user_loras(tmp_path, [LORA], cache)[0]["source"] == "drive"
    assert [call[0] for call in calls] == ["metadata", "media", "media"]
    assert all(FILE_ID in call[2] for call in calls)
    assert (tmp_path / "models" / DRIVE["target"]).read_bytes() == DATA
    assert (tmp_path / "models/loras" / LORA["target"]).read_bytes() == DATA
    calls.clear()
    assert worker._restore_user_loras(tmp_path, [LORA], cache=None)[0]["source"] == "local"
    assert not calls


@pytest.mark.parametrize("failure", ["short", "wrong_sha", "hf_sha", "space", "lora"])
def test_failed_restore_never_starts_comfy(tmp_path, monkeypatch, failure):
    wrong = b"x" * len(DATA)
    cache, calls = fake_sources(monkeypatch, cached=failure in {"short", "wrong_sha"},
                               media=DATA[:3] if failure in {"short", "lora"} else wrong if failure == "wrong_sha" else DATA,
                               stored_md5=hashlib.md5(wrong, usedforsecurity=False).hexdigest() if failure == "wrong_sha" else MD5,
                               hf_data=wrong if failure == "hf_sha" else DATA)
    root = tmp_path / "job"
    root.mkdir()
    monkeypatch.setattr(worker, "NATIVE_SERVICE_MARKER", tmp_path / "missing.json")
    monkeypatch.setattr(worker, "_drive_cache", lambda *args: cache)
    monkeypatch.setattr(worker, "_restore_env", lambda *args, **kwargs: {"source": "fake"})
    monkeypatch.setattr(worker.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Comfy must not start"))
    monkeypatch.setattr(worker, "_probe_gpu", lambda: None)
    if failure == "space":
        monkeypatch.setattr(worker.shutil, "disk_usage", lambda *_: SimpleNamespace(free=0))
    snapshot = {"manifest": {**MANIFEST, "weight_files": [HF]}, "bound_graph": {},
                "lora_files": [LORA] if failure == "lora" else []}
    worker.run_job({"snapshot": snapshot}, root)
    status = json.loads((root / "status.json").read_text())
    assert status["state"] == "failed"
    assert "sampling" not in status["timings"]
    assert ("磁盘空间不足" if failure == "space" else "恢复失败") in status["error_message"]
    assert failure == "lora" or not (root / "ComfyUI/models" / HF["target"]).exists()
    assert not (root / "ComfyUI/models/loras" / LORA["target"]).exists()
    assert failure in {"lora", "hf_sha"} or not any(call[0] == "hf" for call in calls)
    assert not list(root.rglob("*.part"))


def test_drive_fetch_cleans_partial_on_interrupted_read(tmp_path, monkeypatch):
    cache, _ = fake_sources(monkeypatch)

    class BrokenMedia(io.BytesIO):
        def read(self, length):
            if self.tell():
                raise OSError("connection lost")
            return super().read(3)

    monkeypatch.setattr(cache, "_open", lambda *args, **kwargs: BrokenMedia(DATA))
    target = tmp_path / "model.safetensors"
    with pytest.raises(OSError, match="connection lost"):
        cache.fetch(LORA, target)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("layout", ["models_link", "child_link", "extra", "fuse", "nested_mount"])
def test_studio_rejects_mounted_or_redirected_model_paths(tmp_path, monkeypatch, layout):
    comfy = tmp_path / "ComfyUI"
    models = comfy / "models"
    external = tmp_path / "drive"
    comfy.mkdir()
    external.mkdir()
    if layout == "models_link":
        models.symlink_to(external)
    elif layout == "child_link":
        models.mkdir()
        (models / "checkpoints").symlink_to(external)
    elif layout == "extra":
        (comfy / "extra_model_paths.yaml").write_text("drive: {}")
    else:
        read_text = Path.read_text
        mount = comfy if layout == "fuse" else models / "vae"
        kind = "fuse.rclone" if layout == "fuse" else "ext4"

        def mountinfo(path, *args, **kwargs):
            if str(path) == "/proc/self/mountinfo":
                return f"123 12 0:43 / {mount} rw - {kind} remote rw\n"
            return read_text(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", mountinfo)
    with pytest.raises(worker.WorkerError):
        worker._local_models(comfy)


def test_standalone_queue_receives_only_verified_local_weights(tmp_path, monkeypatch):
    cache, _ = fake_sources(monkeypatch, cached=True)
    root = tmp_path / "job"
    root.mkdir()
    comfy = root / "ComfyUI"
    target = comfy / "models" / HF["target"]
    monkeypatch.setattr(worker, "NATIVE_SERVICE_MARKER", tmp_path / "missing.json")
    monkeypatch.setattr(worker, "_drive_cache", lambda *args: cache)
    monkeypatch.setattr(worker, "_restore_env", lambda *args, **kwargs: {"source": "fake"})
    calls = []

    def launch(args, *, cwd, **kwargs):
        assert cwd == comfy and target.read_bytes() == DATA
        assert args[1:5] == ["main.py", "--listen", "127.0.0.1", "--port"]
        assert len(args) == 6 and int(args[-1]) != worker.COMFY_SERVICE_PORT
        return SimpleNamespace(poll=lambda: None, terminate=lambda: None, wait=lambda **_: None)

    def api(url, method="GET", body=None, **kwargs):
        calls.append(url)
        if url.endswith("/prompt"):
            assert method == "POST" and target.resolve() == target
            assert target.read_bytes() == DATA
            return {"prompt_id": "simulated"}
        if url.endswith("/history/simulated"):
            return {"simulated": {"status": {"completed": True}, "outputs": {}}}
        return {}

    monkeypatch.setattr(worker.subprocess, "Popen", launch)
    monkeypatch.setattr(worker, "_http_json", api)
    worker._run_comfy(root, {"snapshot": {}}, {**MANIFEST, "weight_files": [HF]}, {}, worker._Stages(root))
    assert [url.rsplit("/", 1)[-1] for url in calls] == ["system_stats", "prompt", "simulated"]
