"""Trusted Colab worker installed by :mod:`app.gpu.mcp`.

This file is intentionally boring: it is the only code allowed to run inside a
remote Colab runtime.  The worker accepts a JSON snapshot, never a command or
Python string from the caller, and records every outcome in status.json.
"""

from __future__ import annotations

import base64
import contextlib
import csv
import errno
import hashlib
import importlib.metadata
import io
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen

MAX_OUTPUT_BYTES = 100 * 1024 * 1024
COMFY_REPO = "https://github.com/comfyanonymous/ComfyUI"
NATIVE_SERVICE_MARKER = Path("/content/ai-studio/native/state.json")
SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
COMMIT = re.compile(r"^[0-9a-fA-F]{40}$")
DRIVE_API = "https://www.googleapis.com/drive/v3"
DRIVE_UPLOAD = "https://www.googleapis.com/upload/drive/v3"
DRIVE_ID = re.compile(r"^[A-Za-z0-9_-]{10,200}$")
DRIVE_TOKEN = re.compile(r"^[A-Za-z0-9._~+/=-]{20,4096}$")
FOLDER_MIME = "application/vnd.google-apps.folder"
UPLOAD_CHUNK = 64 * 1024 * 1024  # must be a multiple of 256 KiB
# Third-party ComfyUI nodes that were read line by line and pinned. Nothing else may be installed;
# no pip step runs for them (their only dependency, einops, is already in ComfyUI's requirements).
REVIEWED_CUSTOM_NODES = {
    ("https://github.com/xmarre/Comfyui_Minimax_h3_latent_Upscaler-Plus", "1ddcc33e9b674aaabbccc3808fe36e1a47bdd36d"):
        "Comfyui_Minimax_h3_latent_Upscaler",
    # Read-only reviewed fork: pure torch node, no install-time dependency beyond ComfyUI.
    ("https://github.com/huwhitememes/comfyui-krea2-conditioning", "729cda4fade982988a375b01928f515458407a5c"):
        "comfyui-krea2-conditioning",
}
REPO_ID = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")

# Retained Comfy services deliberately have no payload-controlled filesystem or
# network address.  The root is fixed in the worker image; tests may replace the
# module constant with an isolated directory, but callers cannot do so.
COMFY_SERVICE_ROOT = Path("/content/ai-studio/comfy-services")
COMFY_SERVICE_STATE = "state.json"
COMFY_SERVICE_LOCK = "manager.lock"
COMFY_JOB_LOCK = "job.lock"
COMFY_SERVICE_ACTIONS = frozenset({"health", "read_safe_log", "stop_current_job"})
COMFY_SERVICE_PORT = 8188
COMFY_SERVICE_HOST = "127.0.0.1"
MAX_SAFE_LOG_BYTES = 8 * 1024
_SERVICE_HANDLES: dict[int, Any] = {}
_SERVICE_LOG_HANDLES: dict[int, Any] = {}


class WorkerError(RuntimeError):
    pass


class _StopRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


_NO_REDIRECT = build_opener(_StopRedirect)


def _safe_error(exc: BaseException) -> str:
    value = str(exc).replace("Authorization", "[redacted]").replace("Bearer", "[redacted]")
    value = re.sub(r"(?i)(token|secret|password|api[_-]?key)[=: ]+\S+", r"\1=[redacted]", value)
    return value[:1400] or exc.__class__.__name__


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _status(root: Path, state: str, **fields: Any) -> None:
    # Later stage updates keep the measured GPU so the coordinator can still poll it.
    path = root / "status.json"
    if "gpu_diagnostics" not in fields and path.exists():
        previous = json.loads(path.read_text(encoding="utf-8")).get("gpu_diagnostics")
        if previous:
            fields["gpu_diagnostics"] = previous
    _atomic_json(root / "status.json", {"state": state, "updated_at": time.time(), **fields})


def _probe_gpu() -> dict | None:
    """Read the allocated device, independently of the requested Colab spec."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        devices = [
            {"name": name.strip(), "memory_mib": int(memory)}
            for name, memory in csv.reader(result.stdout.splitlines())
        ]
    except (OSError, ValueError, subprocess.SubprocessError):
        # Device reporting is informational and must never stop generation.
        return None
    return {"source": "runtime_nvidia_smi", "captured_at": time.time(), "devices": devices} if devices else None


def _instance_service_name(instance_id: str) -> str:
    """Return a non-path-derived directory name for a validated UUID."""
    try:
        parsed = uuid.UUID(instance_id)
    except (AttributeError, ValueError, TypeError) as exc:
        raise WorkerError("instance_id must be a UUID") from exc
    if str(parsed) != str(instance_id).lower():
        raise WorkerError("instance_id must use canonical UUID form")
    return hashlib.sha256(("comfy-service:" + parsed.hex).encode("ascii")).hexdigest()[:32]


def _comfy_service_dir(instance_id: str) -> Path:
    """Resolve the only directory a retained service may use."""
    name = _instance_service_name(instance_id)
    root = COMFY_SERVICE_ROOT.expanduser().resolve()
    if COMFY_SERVICE_ROOT.exists() and COMFY_SERVICE_ROOT.is_symlink():
        raise WorkerError("Comfy service root must not be a symlink")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    service = root / name
    if service.exists() and service.is_symlink():
        raise WorkerError("Comfy service directory must not be a symlink")
    resolved = service.resolve()
    if root not in resolved.parents:
        raise WorkerError("Comfy service directory escaped fixed root")
    return resolved


@contextlib.contextmanager
def _file_lock(path: Path, *, shared: bool = False):
    """Serialize worker-local service management without trusting a PID lock."""
    # Service management runs in the Linux runtime; coordinators also import
    # this module for the fixed worker protocol on Windows.
    import fcntl

    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with path.open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
        try:
            yield stream
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _service_fingerprint(
    commit: str, custom_nodes: list[dict], weights: list[dict], manifest: dict | None = None
) -> str:
    """Fingerprint the inputs that make a resident process safe to reuse.

    A declared dependency signature is authoritative when present.  For old
    manifests, the installed package set remains a deterministic fallback; it
    is intentionally separate from the warm-disk fingerprint below.
    """
    manifest = manifest or {}
    declared = manifest.get("dependency_signature", manifest.get("dependencies_signature"))
    if declared is None:
        declared = manifest.get("dependencies", manifest.get("requirements_sha256"))
    if declared is None:
        packages = sorted(f"{d.metadata.get('Name', '').lower()}=={d.version}" for d in importlib.metadata.distributions())
        declared = hashlib.sha256("\n".join(packages).encode()).hexdigest()
    body = {
        "version": 1,
        "comfy_commit": commit,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "custom_nodes": sorted(f"{n.get('repo')}@{n.get('commit')}" for n in custom_nodes),
        "weights": sorted(
            (w.get("target"), w.get("sha256"), w.get("size"), w.get("source", "huggingface"))
            for w in weights
        ),
        "dependencies": declared,
    }
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _process_metadata(pid: int) -> dict:
    """Read identity data from Linux procfs; failure means ownership is unknown."""
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise WorkerError("Comfy process ownership is unknown")
    proc = Path("/proc") / str(pid)
    try:
        raw_stat = (proc / "stat").read_text(encoding="utf-8")
        after_comm = raw_stat.rsplit(")", 1)[1].split()
        start_id = after_comm[19]
        raw_cmdline = (proc / "cmdline").read_bytes()
        cmdline = [item.decode("utf-8", "replace") for item in raw_cmdline.split(b"\0") if item]
        cwd = os.readlink(proc / "cwd")
    except (OSError, IndexError, UnicodeError) as exc:
        raise WorkerError("Comfy process ownership is unknown") from exc
    return {"start_id": start_id, "cmdline": cmdline, "cwd": cwd}


def _trusted_process_metadata(service_dir: Path, state: dict) -> dict:
    """Verify PID, start time, command and cwd before any reuse or signal."""
    pid = state.get("pid")
    expected_start = state.get("process_start_id", state.get("start_id"))
    if not isinstance(expected_start, str) or not expected_start:
        raise WorkerError("Comfy process ownership is unknown")
    metadata = _process_metadata(pid)
    if metadata.get("start_id") != expected_start:
        raise WorkerError("Comfy process ownership check failed")
    expected_cwd = (service_dir / "ComfyUI").resolve()
    actual_cwd = Path(str(metadata.get("cwd", ""))).resolve()
    command = metadata.get("cmdline")
    if not isinstance(command, list) or len(command) != 6:
        raise WorkerError("Comfy process ownership check failed")
    if Path(command[0]).name != Path(sys.executable).name or command[1] != "main.py":
        raise WorkerError("Comfy process ownership check failed")
    if command[2:] != ["--listen", COMFY_SERVICE_HOST, "--port", str(COMFY_SERVICE_PORT)]:
        raise WorkerError("Comfy process ownership check failed")
    if actual_cwd != expected_cwd:
        raise WorkerError("Comfy process ownership check failed")
    return metadata


def _read_service_state(service_dir: Path) -> dict | None:
    try:
        value = json.loads((service_dir / COMFY_SERVICE_STATE).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _service_api_url(path: str) -> str:
    if path not in {"/system_stats", "/interrupt"}:
        raise WorkerError("Comfy service endpoint is not allowed")
    return f"http://{COMFY_SERVICE_HOST}:{COMFY_SERVICE_PORT}{path}"


def _stop_service_if_owned(service_dir: Path) -> bool:
    """Stop only the process whose complete identity matches its state file."""
    state = _read_service_state(service_dir)
    if not state:
        return False
    _trusted_process_metadata(service_dir, state)
    pid = state["pid"]
    handle = _SERVICE_HANDLES.pop(pid, None)
    log_handle = _SERVICE_LOG_HANDLES.pop(pid, None)
    try:
        if handle is not None:
            if handle.poll() is None:
                handle.terminate()
                try:
                    handle.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    handle.kill()
            return True
        os.killpg(pid, signal.SIGTERM)
        return True
    except ProcessLookupError:
        return True
    finally:
        if log_handle is not None:
            log_handle.close()
        (service_dir / COMFY_SERVICE_STATE).unlink(missing_ok=True)


def _start_comfy_service(service_dir: Path, fingerprint: str, cache_reason: str, *, deadline=None) -> dict:
    comfy = service_dir / "ComfyUI"
    if not comfy.is_dir() or comfy.is_symlink():
        raise WorkerError("Comfy service checkout is unavailable")
    log_path = service_dir / "comfy.log"
    log_handle = log_path.open("ab", buffering=0)
    args = [sys.executable, "main.py", "--listen", COMFY_SERVICE_HOST, "--port", str(COMFY_SERVICE_PORT)]
    try:
        process = subprocess.Popen(
            args,
            cwd=comfy,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            shell=False,
            start_new_session=True,
        )
        metadata = _process_metadata(process.pid)
        state = {
            "pid": process.pid,
            "process_start_id": metadata["start_id"],
            "start_id": metadata["start_id"],
            "env_fingerprint": fingerprint,
            "cache_reason": cache_reason,
            "cwd": str(comfy.resolve()),
            "argv": args,
            "host": COMFY_SERVICE_HOST,
            "port": COMFY_SERVICE_PORT,
            "updated_at": time.time(),
        }
        _SERVICE_HANDLES[process.pid] = process
        _SERVICE_LOG_HANDLES[process.pid] = log_handle
        _atomic_json(service_dir / COMFY_SERVICE_STATE, state)
        if deadline is None:
            _http_json(_service_api_url("/system_stats"), timeout=5)
        else:
            while time.time() < deadline:
                try:
                    _http_json(_service_api_url("/system_stats"), timeout=5)
                    break
                except WorkerError:
                    if process.poll() is not None:
                        raise WorkerError("Comfy worker exited during startup")
                    time.sleep(1)
            else:
                raise WorkerError("Comfy worker startup timed out")
        return {**state, "system_stats": True, "service_dir": str(service_dir)}
    except Exception:
        try:
            if "process" in locals() and process.poll() is None:
                process.terminate()
        finally:
            log_handle.close()
        raise


def _ensure_comfy_service(
    instance_id: str,
    commit: str,
    custom_nodes: list[dict],
    weights: list[dict],
    *,
    manifest: dict | None = None,
    expected_fingerprint: str | None = None,
) -> dict:
    """Return a real, healthy resident service or fail closed."""
    service_dir = _comfy_service_dir(instance_id)
    fingerprint = expected_fingerprint or _service_fingerprint(commit, custom_nodes, weights, manifest)
    state = _read_service_state(service_dir)
    restart_reason = None
    if state is not None:
        _trusted_process_metadata(service_dir, state)
        if state.get("env_fingerprint") == fingerprint:
            try:
                stats = _http_json(_service_api_url("/system_stats"), timeout=5)
            except WorkerError:
                restart_reason = "restart_health_failed"
            else:
                state.update(cache_reason="hot_reuse", updated_at=time.time())
                _atomic_json(service_dir / COMFY_SERVICE_STATE, state)
                return {**state, "system_stats": stats, "service_dir": str(service_dir)}
        else:
            restart_reason = "restart_fingerprint_mismatch"
        _stop_service_if_owned(service_dir)
    reason = restart_reason or "cold_start"
    return _start_comfy_service(service_dir, fingerprint, reason)


def _validate_workbench_request(request: dict) -> dict:
    if not isinstance(request, dict):
        raise WorkerError("workbench action must be a JSON object")
    action = request.get("action")
    if action not in COMFY_SERVICE_ACTIONS:
        raise WorkerError("workbench action is not allowed")
    allowed = {"action", "instance_id"} | ({"max_bytes"} if action == "read_safe_log" else set())
    if set(request) - allowed:
        raise WorkerError("workbench action contains unsupported fields")
    _instance_service_name(request.get("instance_id"))
    if action == "read_safe_log":
        limit = request.get("max_bytes", MAX_SAFE_LOG_BYTES)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_SAFE_LOG_BYTES:
            raise WorkerError("max_bytes is outside the safe bound")
    return request


def _safe_log(service_dir: Path, limit: int) -> str:
    try:
        raw = (service_dir / "comfy.log").read_bytes()[-limit:]
    except OSError as exc:
        raise WorkerError("safe log is unavailable") from exc
    text = raw.decode("utf-8", "replace")
    text = re.sub(r"(?i)authorization\s*:\s*bearer\s+\S+", "Authorization: Bearer [redacted]", text)
    text = re.sub(r"(?i)\b(token|secret|password|api[_-]?key)\s*[=:]\s*\S+", r"\1=[redacted]", text)
    return text[-limit:]


def _handle_workbench_action(request: dict) -> dict:
    request = _validate_workbench_request(request)
    service_dir = _comfy_service_dir(request["instance_id"])
    state = _read_service_state(service_dir)
    try:
        if state is None:
            raise WorkerError("service state is unavailable")
        _trusted_process_metadata(service_dir, state)
        if request["action"] == "health":
            stats = _http_json(_service_api_url("/system_stats"), timeout=5)
            return {
                "state": "healthy",
                "cache_reason": state.get("cache_reason", "unknown"),
                "pid": state["pid"],
                "process_start_id": state.get("process_start_id"),
                "system_stats": stats,
            }
        if request["action"] == "read_safe_log":
            return {
                "state": "ok",
                "cache_reason": state.get("cache_reason", "unknown"),
                "log": _safe_log(service_dir, request.get("max_bytes", MAX_SAFE_LOG_BYTES)),
            }
        _http_json(_service_api_url("/interrupt"), "POST", {}, timeout=5)
        return {"state": "interrupt_requested", "cache_reason": state.get("cache_reason", "unknown")}
    except Exception as exc:  # safe protocol never reports an unprobed healthy state
        return {"state": "unknown", "reason": _safe_error(exc)[:200]}


def _main_workbench_action(path: str) -> int:
    action_path = Path(path)
    if action_path.is_symlink() or not action_path.is_file():
        return 2
    try:
        request = json.loads(action_path.read_text(encoding="utf-8"))
        result = _handle_workbench_action(request)
    except Exception as exc:  # malformed/unsupported requests fail closed
        result = {"state": "unknown", "reason": _safe_error(exc)[:200]}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


def _run(
    args: list[str], *, cwd: Path | None = None, timeout: float, capture: bool = True
) -> subprocess.CompletedProcess:
    if not isinstance(args, list) or not args or any(not isinstance(item, str) for item in args):
        raise WorkerError("internal command must be an argument list")
    try:
        return subprocess.run(
            args, cwd=cwd, timeout=timeout, check=True, shell=False, capture_output=capture, text=True
        )
    except subprocess.TimeoutExpired as exc:
        raise WorkerError("controlled command timed out") from exc
    except subprocess.CalledProcessError as exc:
        tail = ((exc.stderr or "") + (exc.stdout or "")).strip()[-600:]
        raise WorkerError(
            f"controlled command failed ({Path(args[0]).name} {' '.join(args[1:3])}): {tail}"
        ) from exc
    except OSError as exc:
        raise WorkerError("controlled command failed") from exc


def _snapshot(payload: dict) -> tuple[dict, dict, dict]:
    snapshot = payload.get("snapshot")
    if not isinstance(snapshot, dict):
        raise WorkerError("missing immutable snapshot")
    manifest = snapshot.get("manifest")
    graph = snapshot.get("bound_graph")
    if not isinstance(manifest, dict):
        raise WorkerError("missing workflow manifest")
    if manifest.get("state") != "published" or manifest.get("reviewed") is not True:
        raise WorkerError("workflow is not reviewed and published")
    _reviewed_custom_nodes(manifest.get("custom_nodes"))
    if snapshot.get("custom_nodes"):
        raise WorkerError("custom_nodes belong in the reviewed manifest")
    if not isinstance(graph, dict):
        graph = {}
    return snapshot, manifest, graph


def _reviewed_custom_nodes(items: Any) -> list[dict]:
    """Each entry must be exactly a pinned (repo, commit) from REVIEWED_CUSTOM_NODES."""
    items = items or []
    if not isinstance(items, list):
        raise WorkerError("custom_nodes must be a list")
    for item in items:
        if not isinstance(item, dict) or REVIEWED_CUSTOM_NODES.get((item.get("repo"), item.get("commit"))) != item.get("dir"):
            raise WorkerError("custom node is not a reviewed, pinned repository")
    return items


def _install_custom_nodes(comfy: Path, items: list[dict], timeout: float) -> None:
    for item in items:
        folder = comfy / "custom_nodes" / item["dir"]
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(mode=0o700, parents=True)
        _run(["git", "init", "-q"], cwd=folder, timeout=30)
        _run(["git", "fetch", "-q", "--depth", "1", item["repo"], item["commit"]], cwd=folder, timeout=timeout)
        _run(["git", "checkout", "-q", "--detach", "FETCH_HEAD"], cwd=folder, timeout=timeout)
    _verify_custom_nodes(comfy, items)


def _verify_custom_nodes(comfy: Path, items: list[dict]) -> None:
    for item in items:
        head = _run(["git", "rev-parse", "HEAD"], cwd=comfy / "custom_nodes" / item["dir"], timeout=30).stdout.strip()
        if head.lower() != item["commit"].lower():
            raise WorkerError(f"custom node {item['dir']} is not the reviewed commit")


def _validate_weights(manifest: dict) -> list[dict]:
    weights = manifest.get("weight_files", manifest.get("weights"))
    if not isinstance(weights, list) or not weights:
        raise WorkerError("no reviewed model weights; refusing to run")
    checked = []
    for item in weights:
        if not isinstance(item, dict):
            raise WorkerError("weight manifest is invalid")
        source = item.get("source", "huggingface")
        if source not in {"huggingface", "drive"}:
            raise WorkerError("weight source is unknown")
        sha256 = item.get("sha256")
        if not isinstance(sha256, str) or not SHA256.fullmatch(sha256):
            raise WorkerError("weight sha256 is not locked")
        if source == "drive":
            drive_file_id = item.get("drive_file_id")
            if not isinstance(drive_file_id, str) or not DRIVE_ID.fullmatch(drive_file_id):
                raise WorkerError("Drive file id is invalid")
            size = item.get("size")
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                raise WorkerError("Drive weight size must be a positive integer")
            target = item.get("target")
            if not isinstance(target, str) or not target or target.startswith("/") or ".." in Path(target).parts:
                raise WorkerError("weight target path traversal refused")
            checked.append({"source": "drive", "drive_file_id": drive_file_id, "sha256": sha256.lower(), "target": target, "size": size})
            continue
        repo_id, filename, revision = (item.get(key) for key in ("repo_id", "filename", "revision"))
        if not isinstance(repo_id, str) or not REPO_ID.fullmatch(repo_id):
            raise WorkerError("weight repo_id is not an allowed Hugging Face repository")
        if not isinstance(filename, str) or not filename or filename.startswith("/"):
            raise WorkerError("weight filename is invalid")
        path = Path(filename)
        if ".." in path.parts or path.is_absolute():
            raise WorkerError("weight filename path traversal refused")
        if not isinstance(revision, str) or not COMMIT.fullmatch(revision):
            raise WorkerError("weight revision must be a fixed commit")
        target = item.get("target", filename)
        if not isinstance(target, str) or not target or target.startswith("/") or ".." in Path(target).parts:
            raise WorkerError("weight target path traversal refused")
        size = item.get("size")
        if size is not None and (isinstance(size, bool) or not isinstance(size, int) or size <= 0):
            raise WorkerError("weight size must be a positive integer")
        checked.append({"repo_id": repo_id, "filename": filename, "revision": revision, "sha256": sha256.lower(), "target": target, "size": size})
    return checked


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_local_tree(models: Path) -> None:
    if models.resolve() != models.absolute():
        raise WorkerError("创作模式的 ComfyUI 模型目录必须位于 Colab 本地且不能经过软链接")
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        info, fs = line.split(" - ", 1)
        mount = Path(re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), info.split()[4]))
        kind = fs.split()[0]
        if (mount.is_relative_to(models) or
                (models.is_relative_to(mount) and (kind.startswith("fuse") or kind in {
                    "nfs", "nfs4", "cifs", "smb3", "9p", "autofs", "davfs",
                }))):
            raise WorkerError("创作模式的模型目录不能位于挂载目录或包含其他挂载点")
    if not models.exists():
        return
    if not models.is_dir():
        raise WorkerError("创作模式的本地模型路径不是目录")

    def inspection_failed(exc: OSError) -> None:
        raise WorkerError("无法检查创作模式的本地模型路径") from exc

    for parent, dirs, files in os.walk(models, onerror=inspection_failed):
        if any((Path(parent) / name).is_symlink() for name in (*dirs, *files)):
            raise WorkerError("创作模式的模型目录不能包含软链接")


def _local_models(comfy: Path) -> Path:
    """Validate the model tree independently of Drive-backed input/output directories."""
    extra = comfy / "extra_model_paths.yaml"
    if extra.exists() or extra.is_symlink():
        raise WorkerError("创作模式需要默认本地模型目录，请检查 extra_model_paths.yaml")
    models = comfy / "models"
    _check_local_tree(models)
    return models


def _local_model_hit(path: Path, size: int | None, digest: str, *, md5: bool = False) -> bool:
    if not path.is_file():
        return False
    before = path.stat()
    if before.st_size <= 0 or (size is not None and before.st_size != size):
        return False
    actual = _hash_file(path)[1] if md5 else _sha256(path)
    after = path.stat()
    return actual == digest and (before.st_ino, before.st_ctime_ns) == (after.st_ino, after.st_ctime_ns)


def _check_model_space(directory: Path, needed: int) -> None:
    free = shutil.disk_usage(directory).free
    if free < needed + 256 * 1024**2:
        raise WorkerError(f"Colab 本地磁盘空间不足，模型下载需要 {needed} 字节及 256 MiB 预留，当前剩余 {free} 字节")


def _model_download_error(exc: BaseException) -> str:
    cause = exc
    while cause is not None:
        if isinstance(cause, OSError) and cause.errno == errno.ENOSPC:
            return "Colab 本地磁盘空间不足"
        cause = cause.__cause__
    return _safe_error(exc)[:200]


def run_cuda_smoke(root: Path) -> dict:
    """Run only real CUDA probes; a missing GPU is a hard failure."""
    try:
        nvidia = _run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            timeout=30,
        ).stdout.strip()
        import torch

        available = bool(torch.cuda.is_available())
        count = int(torch.cuda.device_count())
        names = [str(torch.cuda.get_device_name(index)) for index in range(count)] if available else []
    except Exception as exc:
        raise WorkerError("CUDA smoke probe failed") from exc
    if not available or count < 1 or not nvidia:
        raise WorkerError("CUDA GPU is not available")
    payload = {
        "cuda_available": available,
        "device_count": count,
        "device_names": names,
        "nvidia_smi": nvidia[:4000],
    }
    content = json.dumps(payload, sort_keys=True).encode("utf-8")
    (root / "outputs").mkdir(mode=0o700, exist_ok=True)
    output = root / "outputs" / "cuda-smoke.json"
    output.write_bytes(content)
    return {
        "filename": output.name,
        "mime": "application/json",
        "path": str(output.relative_to(root)),
        "size": len(content),
        "metadata": {"cuda_smoke": True},
    }


class _Stages:
    """Wall-clock seconds per worker stage, mirrored into status.json so the server can show them."""

    def __init__(self, root: Path):
        self.root, self.timings, self._name, self._start = root, {}, None, 0.0

    def start(self, name: str) -> None:
        self.end()
        self._name, self._start = name, time.time()
        _status(self.root, "running", stage=name, timings=self.timings, stage_started=self._start)

    def end(self) -> None:
        if self._name:
            self.timings[self._name] = round(time.time() - self._start, 1)
            self._name = None


def _checkout_comfy(comfy: Path, commit: str, timeout: float) -> None:
    if not COMMIT.fullmatch(commit):
        raise WorkerError("ComfyUI commit must be a fixed 40-character commit")
    comfy.mkdir(mode=0o700, parents=True, exist_ok=True)
    _run(["git", "init", "-q"], cwd=comfy, timeout=30)
    _run(["git", "fetch", "-q", "--depth", "1", COMFY_REPO, commit], cwd=comfy, timeout=timeout)
    _run(["git", "checkout", "-q", "--detach", "FETCH_HEAD"], cwd=comfy, timeout=timeout)


def _verify_checkout(comfy: Path, commit: str) -> None:
    current = _run(["git", "rev-parse", "HEAD"], cwd=comfy, timeout=30).stdout.strip()
    if current.lower() != commit.lower():
        raise WorkerError("ComfyUI checkout is not the reviewed commit")
    if not (comfy / "requirements.txt").is_file():
        raise WorkerError("reviewed ComfyUI checkout has no requirements.txt")


def _site_dir() -> Path:
    return Path(sysconfig.get_paths()["purelib"])


def _site_state(site: Path) -> dict[str, int]:
    return {entry.name: entry.stat().st_mtime_ns for entry in site.iterdir()}


def _gpu_cache_tag() -> str:
    """Separate native extensions by GPU architecture and driver, including CPU prep."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        identity = result.stdout.strip()
        if not identity:
            raise ValueError("empty GPU identity")
    except (OSError, ValueError, subprocess.SubprocessError):
        # No shared environment cache hit when device identity cannot be read.
        identity = "unidentified-" + str(os.getpid()) + "-" + str(time.time_ns())
    return hashlib.sha256(identity.encode()).hexdigest()[:16]


def _check_compute_hardware(manifest: dict, diagnostics: dict | None, requested: dict | None = None) -> None:
    gpu = (requested or {}).get("accelerator")
    if gpu in {"T4", "L4", "A100", "H100"}:
        devices = (diagnostics or {}).get("devices") or []
        pattern = rf"(?:^|[^A-Z0-9]){gpu}(?:[^A-Z0-9]|$)"
        if not devices or not all(re.search(pattern, d["name"].upper()) for d in devices):
            raise WorkerError(f"实测设备与请求 {gpu} 不一致或无法确认，已停止运行；不会切换其他 GPU")
    req = manifest.get("compute_requirements") or {}
    minimum = req.get("min_vram_gb")
    if minimum:
        devices = (diagnostics or {}).get("devices") or []
        if not devices or max(d["memory_mib"] for d in devices) * 1024**2 < float(minimum) * 1e9:
            raise WorkerError(f"实际 GPU 显存不满足最低 {minimum} GB 要求，已停止运行")
    if req.get("min_ram_gb"):
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
        if ram < float(req["min_ram_gb"]):
            raise WorkerError("实际系统 RAM 低于工作流最低要求，已停止运行")


def _env_key(commit: str, timeout: float, custom_nodes: list[dict] | None = None) -> str:
    """Cache key: ComfyUI commit + Python + hash of the packages already on the image + requirements hash.

    Hashing the image's package list means the cache can only hit on a runtime whose torch/CUDA/etc.
    are the same as when the environment was built.
    """
    packages = sorted(f"{d.metadata['Name'].lower()}=={d.version}" for d in importlib.metadata.distributions())
    image = hashlib.sha256("\n".join(packages).encode()).hexdigest()[:12]
    url = COMFY_REPO.replace("https://github.com/", "https://raw.githubusercontent.com/") + f"/{commit}/requirements.txt"
    with urlopen(url, timeout=timeout) as response:
        requirements = hashlib.sha256(response.read(1024 * 1024)).hexdigest()[:12]
    py = f"py{sys.version_info.major}{sys.version_info.minor}"
    key = f"comfy-{commit[:12]}-{py}-img{image}-req{requirements}-gpu{_gpu_cache_tag()}"
    if custom_nodes:
        pins = "\n".join(sorted(f"{n['repo']}@{n['commit']}" for n in custom_nodes))
        key += "-nodes" + hashlib.sha256(pins.encode()).hexdigest()[:12]
    return key


def _pip_install(comfy: Path, timeout: float) -> None:
    base = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "-q"]
    _run([*base, "-r", str(comfy / "requirements.txt")], cwd=comfy, timeout=timeout)
    _run([*base, "huggingface_hub"], cwd=comfy, timeout=timeout)


def _pack_env(comfy: Path, site: Path, before: dict[str, int], key: str, out: Path) -> dict:
    """Tar the ComfyUI checkout plus every site-packages entry the install added or changed."""
    after = _site_state(site)
    added = sorted(name for name, mtime in after.items() if before.get(name) != mtime)
    removed = sorted(name for name in before if name not in after)
    with tarfile.open(out, "w:gz", compresslevel=1) as tar:
        tar.add(comfy, arcname="ComfyUI")
        for name in added:
            tar.add(site / name, arcname=f"site/{name}")
        meta = json.dumps({"key": key, "removed": removed}).encode()
        info = tarfile.TarInfo("env.json")
        info.size = len(meta)
        tar.addfile(info, io.BytesIO(meta))
    return {"site_entries_added": len(added), "site_entries_removed": len(removed)}


def _unpack_env(archive: Path, root: Path, site: Path) -> None:
    stage = root / ".env-stage"
    shutil.rmtree(stage, ignore_errors=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(stage, filter="data")
    removed = json.loads((stage / "env.json").read_text())["removed"]
    for name in removed:
        _remove(site / name)
    for entry in (stage / "site").iterdir():
        _remove(site / entry.name)
        shutil.move(str(entry), str(site / entry.name))
    shutil.move(str(stage / "ComfyUI"), str(root / "ComfyUI"))
    shutil.rmtree(stage, ignore_errors=True)


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


WARM_VERSION = 1


def _warm_fingerprint(commit: str, custom_nodes: list[dict], weights: list[dict]) -> str:
    """What a restored ComfyUI tree must match for the next job on this runtime to reuse it as-is."""
    body = {
        "v": WARM_VERSION,
        "commit": commit,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "nodes": sorted(f"{n['repo']}@{n['commit']}" for n in custom_nodes),
        "weights": sorted(
            (w["target"], w["sha256"], w.get("size"), w.get("source", "huggingface")) for w in weights
        ),
    }
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def _warm_take(root: Path, fingerprint: str, custom_nodes: list[dict], weights: list[dict],
               *, user_loras: list[dict] | tuple = ()) -> bool:
    """Move the previous job's restored ComfyUI tree into this job's root when it is the same environment.

    The marker is dropped while the tree is in use, so a job that dies mid-run can never leave a
    marker that points at a tree that has moved or half changed.
    """
    warm = root.parent / ".warm"
    marker = warm / "state.json"
    try:
        state = json.loads(marker.read_text("utf-8"))
    except (OSError, ValueError):
        return False
    tree = warm / "ComfyUI"
    if not isinstance(state, dict) or state.get("fingerprint") != fingerprint or not tree.is_dir():
        return False
    models = _local_models(tree)
    for item in weights:
        if not _local_model_hit(models / item["target"], item.get("size"), item["sha256"]):
            return False
    for node in custom_nodes:
        if not (tree / "custom_nodes" / node["dir"]).is_dir():
            return False
    marker.unlink(missing_ok=True)
    shutil.rmtree(root / "ComfyUI", ignore_errors=True)
    tree.rename(root / "ComfyUI")
    # Only the LoRAs admitted for the new job may remain. Their bytes are checked on restore.
    selected = {Path(item["target"]).name for item in user_loras}
    loras = root / "ComfyUI" / "models" / "loras" / "user"
    if loras.is_dir():
        for file in loras.iterdir():
            if file.name not in selected:
                _remove(file)
    for leftover in ("input", "output", "temp"):
        shutil.rmtree(root / "ComfyUI" / leftover, ignore_errors=True)
    return True


def _warm_put(root: Path, fingerprint: str) -> None:
    """Park the restored tree where the next job on this runtime can pick it up."""
    tree = root / "ComfyUI"
    if not tree.is_dir():
        return
    warm = root.parent / ".warm"
    try:
        warm.mkdir(mode=0o700, exist_ok=True)
        (warm / "state.json").unlink(missing_ok=True)
        shutil.rmtree(warm / "ComfyUI", ignore_errors=True)
        tree.rename(warm / "ComfyUI")
        _atomic_json(warm / "state.json", {"fingerprint": fingerprint, "saved_at": time.time()})
    except OSError:  # the next job simply restores from the cache again
        shutil.rmtree(warm / "ComfyUI", ignore_errors=True)


def _restore_env(
    root: Path, commit: str, timeout: float, cache: "DriveCache | None", build: bool, custom_nodes: list[dict] | None = None
) -> dict:
    """ComfyUI + pip deps: from the Drive env cache when possible, otherwise installed fresh.

    ``build`` (CPU prep jobs only) packs a fresh install and uploads it; GPU jobs never upload,
    so the GPU is never kept alive for cache work.
    """
    comfy = root / "ComfyUI"
    site = _site_dir()
    started = time.time()
    record: dict[str, Any] = {}
    key = target = None
    if cache is not None:
        try:
            key = _env_key(commit, timeout, custom_nodes)
            target = f"env/{key}.tar.gz"
            record["key"] = key
            entry = cache.lookup(target, None, None)
            if entry:
                archive = root / "env.tar.gz"
                cache.fetch(entry, archive)
                if _hash_file(archive)[0] != entry["appProperties"]["sha256"]:
                    raise WorkerError("env cache sha256 mismatch")
                _unpack_env(archive, root, site)
                archive.unlink(missing_ok=True)
                _verify_checkout(comfy, commit)
                _verify_custom_nodes(comfy, custom_nodes or [])
                record.update(source="drive", seconds=round(time.time() - started, 1))
                return record
        except Exception as exc:  # noqa: BLE001 - a broken cache falls back to a normal install
            record["cache_error"] = _safe_error(exc)[:200]
            shutil.rmtree(comfy, ignore_errors=True)
    before = _site_state(site)
    _checkout_comfy(comfy, commit, timeout)
    _verify_checkout(comfy, commit)
    _pip_install(comfy, timeout)
    _install_custom_nodes(comfy, custom_nodes or [], timeout)
    record.update(source="install", seconds=round(time.time() - started, 1))
    if build and cache is not None and key:
        try:
            archive = root / "env.tar.gz"
            record.update(_pack_env(comfy, site, before, key, archive))
            sha256, md5 = _hash_file(archive)
            cache.store(target, archive, sha256, md5, archive.stat().st_size)
            record.update(cache_stored=True, cache_bytes=archive.stat().st_size)
            archive.unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001 - the prep job still succeeded in installing
            record.update(cache_stored=False, cache_error=_safe_error(exc)[:200])
    return record


def _hash_file(path: Path) -> tuple[str, str]:
    """Return (sha256, md5) from one pass; md5 is only for comparing with Drive's checksum."""
    sha, md5 = hashlib.sha256(), hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(block)
            md5.update(block)
    return sha.hexdigest(), md5.hexdigest()


class DriveCache:
    """Model cache in the user's Drive, reached with a short-lived drive.file token.

    drive.file only exposes files this app created, so the cache can never read
    anything else in the user's Drive.  A cached file is trusted when its stored
    sha256/size marker matches the reviewed manifest and the md5 computed while
    downloading matches the checksum Drive reports.  The sha256 marker is only
    written after a full sha256 verification of the Hugging Face download.
    """

    def __init__(self, token: str, folder_id: str):
        if not isinstance(token, str) or not DRIVE_TOKEN.fullmatch(token):
            raise WorkerError("drive cache token is invalid")
        if not isinstance(folder_id, str) or not DRIVE_ID.fullmatch(folder_id):
            raise WorkerError("drive cache folder id is invalid")
        self._token = token
        self.folder_id = folder_id
        self._folders: dict[tuple[str, str], str] = {}

    def _open(
        self, method: str, url: str, *, body: bytes | None = None, headers: dict | None = None, timeout=120
    ):
        merged = {"Authorization": "Bearer " + self._token}
        merged.update(headers or {})
        for _ in range(4):
            host = urlsplit(url).hostname or ""
            if urlsplit(url).scheme != "https" or not (
                host == "www.googleapis.com" or host.endswith(".googleusercontent.com")
            ):
                raise WorkerError("drive cache refuses a non-Google URL")
            try:
                return _NO_REDIRECT.open(
                    Request(url, data=body, method=method, headers=merged), timeout=timeout
                )
            except HTTPError as exc:
                location = exc.headers.get("Location") if exc.code in {301, 302, 303, 307} else None
                if not location:
                    raise
                url = location  # Drive may hand media downloads to a googleusercontent host
        raise WorkerError("drive cache redirected too many times")

    def _json(self, method: str, url: str, body: dict | None = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json; charset=UTF-8"} if data is not None else {}
        with self._open(method, url, body=data, headers=headers, timeout=60) as response:
            raw = response.read(1024 * 1024)
        value = json.loads(raw.decode("utf-8")) if raw else {}
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _q(value: str) -> str:
        return value.replace("\\", "\\\\").replace("'", "\\'")

    def _list(self, name: str, parent: str, folder: bool) -> list[dict]:
        op = "=" if folder else "!="
        query = f"name = '{self._q(name)}' and mimeType {op} '{FOLDER_MIME}' and trashed = false and '{parent}' in parents"
        params = urlencode(
            {"q": query, "pageSize": "20", "fields": "files(id,name,size,md5Checksum,appProperties)"}
        )
        return self._json("GET", f"{DRIVE_API}/files?{params}").get("files") or []

    def _folder(self, name: str, parent: str, create: bool) -> str | None:
        cached = self._folders.get((parent, name))
        if cached:
            return cached
        found = self._list(name, parent, True)
        if found:
            folder_id = found[0]["id"]
        elif create:
            made = self._json(
                "POST",
                f"{DRIVE_API}/files?fields=id",
                {"name": name, "mimeType": FOLDER_MIME, "parents": [parent]},
            )
            folder_id = made.get("id")
            if not isinstance(folder_id, str):
                raise WorkerError("drive cache folder creation failed")
        else:
            return None
        self._folders[(parent, name)] = folder_id
        return folder_id

    def _parent(self, target: str, create: bool) -> tuple[str | None, str]:
        parts = Path(target).parts
        parent = self.folder_id
        for part in parts[:-1]:
            parent = self._folder(part, parent, create)
            if parent is None:
                return None, parts[-1]
        return parent, parts[-1]

    def lookup(self, target: str, sha256: str | None, size: int | None) -> dict | None:
        """Entry whose stored sha256/size marker matches; with sha256=None any marked entry (env archives)."""
        parent, name = self._parent(target, create=False)
        if parent is None:
            return None
        for entry in self._list(name, parent, False):
            props = entry.get("appProperties") or {}
            stored_size = int(entry.get("size") or 0)
            if not props.get("sha256") or (sha256 is not None and props["sha256"] != sha256) or stored_size <= 0:
                continue
            if size is not None and stored_size != size:
                continue
            if props.get("size") not in (None, str(stored_size)):
                continue
            return entry
        return None

    def fetch(self, entry: dict, destination: Path) -> None:
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        digest = hashlib.md5(usedforsecurity=False)
        size = int(entry["size"])
        written = 0
        url = f"{DRIVE_API}/files/{quote(entry['id'])}?alt=media"
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".part", delete=False) as out:
                temporary = Path(out.name)
                with self._open("GET", url, timeout=300) as response:
                    for block in iter(lambda: response.read(8 * 1024 * 1024), b""):
                        written += len(block)
                        if written > size:
                            raise WorkerError("drive cached file exceeds expected size")
                        out.write(block)
                        digest.update(block)
            if written != size or digest.hexdigest() != entry.get("md5Checksum"):
                raise WorkerError("drive cached file failed size/md5 check")
            os.replace(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def store(self, target: str, source: Path, sha256: str, md5: str, size: int) -> None:
        parent, name = self._parent(target, create=True)
        stale = self._list(name, parent, False)
        metadata = {
            "name": name,
            "parents": [parent],
            "appProperties": {"sha256": sha256, "size": str(size)},
        }
        init = self._open(
            "POST",
            f"{DRIVE_UPLOAD}/files?uploadType=resumable&fields=id,size,md5Checksum",
            body=json.dumps(metadata).encode(),
            headers={
                "Content-Type": "application/json; charset=UTF-8",
                "X-Upload-Content-Length": str(size),
                "X-Upload-Content-Type": "application/octet-stream",
            },
            timeout=60,
        )
        with init:
            location = init.headers.get("Location", "")
        if urlsplit(location).hostname != "www.googleapis.com":
            raise WorkerError("drive upload session location is invalid")
        result: dict = {}
        with source.open("rb") as stream:
            offset = 0
            while offset < size:
                chunk = stream.read(min(UPLOAD_CHUNK, size - offset))
                end = offset + len(chunk) - 1
                headers = {
                    "Content-Range": f"bytes {offset}-{end}/{size}",
                    "Content-Type": "application/octet-stream",
                }
                for attempt in range(3):
                    try:
                        with self._open(
                            "PUT", location, body=chunk, headers=headers, timeout=300
                        ) as response:
                            raw = response.read(1024 * 1024)
                        if raw:
                            result = json.loads(raw.decode("utf-8"))
                        break
                    except HTTPError as exc:
                        if exc.code == 308:  # chunk accepted, more expected
                            break
                        if exc.code < 500 or attempt == 2:
                            raise
                    except (URLError, OSError, TimeoutError):
                        if attempt == 2:
                            raise
                    time.sleep(2**attempt)
                offset = end + 1
        if int(result.get("size") or 0) != size or result.get("md5Checksum") != md5:
            raise WorkerError("drive upload verification failed")
        for old in stale:  # replace a stale copy only after the new one is verified
            self._json("PATCH", f"{DRIVE_API}/files/{quote(old['id'])}", {"trashed": True})


def _drive_cache(payload: dict, folder_key: str = "folder_id") -> DriveCache | None:
    """Model cache (``folder_id``) or runtime/env cache (``runtime_folder_id``) from the in-memory payload."""
    config = payload.get("drive_cache")
    if config is None:
        return None
    if not isinstance(config, dict):
        raise WorkerError("drive_cache must be an object")
    return DriveCache(config.get("access_token"), config.get(folder_key))


def _lora_drive_cache(payload: dict) -> DriveCache | None:
    return _drive_cache({"drive_cache": payload.get("lora_drive_cache", payload.get("drive_cache"))})


def _check_loras(graph: dict, manifest: dict, user_files: list[dict]) -> None:
    """Fail before any download if every referenced LoRA is not in this job."""
    weights = manifest.get("weight_files", manifest.get("weights")) or []
    have = {w["target"].removeprefix("loras/") for w in weights if str(w.get("target", "")).startswith("loras/")}
    if {Path(f["target"]) for f in user_files} & {Path(target) for target in have}:
        raise WorkerError("用户 LoRA 与工作流模型绑定在同一本地路径")
    have |= {f.get("target") for f in user_files}
    for node in graph.values():
        if isinstance(node, dict) and node.get("class_type") == "LoraLoaderModelOnly":
            name = node.get("inputs", {}).get("lora_name")
            if name not in have:
                raise WorkerError(f"LoRA 文件没有随任务下发：{name}")


def _publish_model(source: Path, target: Path, *, preserve_existing: bool) -> None:
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if preserve_existing:
        # A native session must keep its own existing weights, including concurrent imports.
        try:
            os.link(source, target)
        except FileExistsError as exc:
            raise WorkerError(f"原生 ComfyUI 本地模型路径已被占用：{target.name}") from exc
    else:
        os.replace(source, target)


def _restore_user_loras(comfy: Path, files: list[dict], cache: DriveCache | None,
                        *, preserve_existing: bool = False, paths: dict[str, Path] | None = None) -> list[dict]:
    """Verify the job's Drive LoRAs before exposing them to ComfyUI."""
    if not files:
        return []
    models = _local_models(comfy)
    pending = {}
    records = []
    for item in files:
        relative = Path(item["target"])
        md5 = item.get("md5Checksum")
        size = item.get("size")
        if (len(relative.parts) != 2 or relative.parts[0] != "user" or relative.suffix.lower() != ".safetensors"
                or not DRIVE_ID.fullmatch(str(item.get("id", "")))
                or type(size) is not int or size <= 0
                or not isinstance(md5, str) or not re.fullmatch(r"[0-9a-f]{32}", md5)):
            raise WorkerError("用户 LoRA 的路径或 Drive 文件身份无效")
        model_name = "loras/" + item["target"]
        target = paths[model_name] if paths is not None else models / model_name
        if not target.is_relative_to(models):
            raise WorkerError("用户 LoRA 路径不在本地模型目录")
        if _local_model_hit(target, size, md5, md5=True):
            records.append({"target": "loras/" + item["target"], "source": "local"})
            continue
        if preserve_existing and target.exists():
            raise WorkerError(f"原生 ComfyUI 本地 LoRA 身份不匹配：{relative.name}")
        if target in pending and pending[target] != item:
            raise WorkerError("同一路径绑定了不同的用户 LoRA")
        pending[target] = item
    if not pending:
        return records
    if cache is None:
        raise WorkerError("自定义 LoRA 需要 Drive 存储授权")
    _check_model_space(comfy, sum(item["size"] for item in pending.values()))
    with tempfile.TemporaryDirectory(prefix=".studio-loras-", dir=comfy) as scratch:
        for target, item in pending.items():
            started = time.time()
            try:
                source = Path(scratch) / target.name
                cache.fetch(item, source)
                _publish_model(source, target, preserve_existing=preserve_existing)
                records.append({"target": "loras/" + item["target"], "source": "drive",
                                "seconds": round(time.time() - started, 1)})
            except Exception as exc:
                raise WorkerError(f"用户 LoRA 恢复失败：{_model_download_error(exc)}") from exc
    return records


def _restore_weights(
    comfy: Path, manifest: dict, timeout: float, cache: DriveCache | None, prep: bool,
    *, preserve_existing: bool = False, warm_models: Path | None = None, use_drive_cache: bool = True,
    paths: dict[str, Path] | None = None,
) -> list[dict]:
    """Bring every reviewed weight file in, four at a time, on the destination filesystem."""
    checked = _validate_weights(manifest)
    models = _local_models(comfy)
    destinations = {item["target"]: paths[item["target"]] if paths is not None else models / item["target"]
                    for item in checked}
    by_target = {}
    for item in checked:
        key = destinations[item["target"]]
        if key == models or not key.is_relative_to(models) or (key in by_target and by_target[key] != item):
            raise WorkerError("同一本地路径绑定了不同的模型，或模型路径无效")
        by_target[key] = item
    models.mkdir(mode=0o700, parents=True, exist_ok=True)
    if prep and cache is not None:
        for item in by_target.values():
            if item.get("source") != "drive":
                cache._parent(item["target"], create=True)  # noqa: SLF001

    def drive_entry(item: dict) -> dict:
        if cache is None:
            raise WorkerError("Drive 模型需要 Drive 存储授权")
        fields = urlencode({"fields": "id,size,md5Checksum,mimeType,trashed"})
        entry = cache._json("GET", f"{DRIVE_API}/files/{quote(item['drive_file_id'])}?{fields}")  # noqa: SLF001
        if (
            entry.get("id") != item["drive_file_id"]
            or entry.get("trashed") is not False
            or not isinstance(entry.get("mimeType"), str)
            or entry.get("mimeType") == FOLDER_MIME
            or int(entry.get("size") or 0) != item["size"]
            or not isinstance(entry.get("md5Checksum"), str)
        ):
            raise WorkerError("Drive weight metadata failed id/trashed/type/size/md5 checks")
        return entry

    def local(item: dict) -> dict:
        started = time.time()
        target = destinations[item["target"]]
        record = {"target": item["target"], "sha256": item["sha256"]}
        if item.get("source") == "drive":
            record["drive_file_id"] = item["drive_file_id"]
        if not prep:
            if _local_model_hit(target, item["size"], item["sha256"]):
                return {**record, "source": "local", "seconds": round(time.time() - started, 1)}
            if target.exists():
                if preserve_existing or not target.is_file():
                    raise WorkerError(f"本地模型路径已被占用或文件身份不匹配：{item['target']}")
                target.unlink()
            if warm_models is not None:
                candidate = warm_models / item["target"]
                if _local_model_hit(candidate, item["size"], item["sha256"]):
                    _publish_model(candidate, target, preserve_existing=True)
                    return {**record, "source": "local", "seconds": round(time.time() - started, 1)}
        return record

    def restore(item: dict, record: dict, scratch: Path) -> dict:
        if record.get("source") == "local":
            return record
        started = time.time()
        target = destinations[item["target"]]
        scratch.mkdir()

        def fetch(entry: dict, source: Path) -> None:
            _check_model_space(scratch, int(entry["size"]))
            cache.fetch(entry, source)
            if not _local_model_hit(source, item["size"], item["sha256"]):
                raise WorkerError("Drive weight sha256 or size mismatch")
            if not prep:
                _publish_model(source, target, preserve_existing=preserve_existing)

        try:
            if item.get("source") == "drive":
                fetch(drive_entry(item), scratch / "drive")
                record.update(source="drive")
                if prep:
                    record["cache_stored"] = True  # The library file is already in Drive.
            else:
                entry = None
                if cache is not None and use_drive_cache:
                    try:
                        entry = cache.lookup(item["target"], item["sha256"], item["size"])
                    except Exception as exc:  # Cache lookup is optional for pinned Hugging Face models.
                        record["cache_error"] = _model_download_error(exc)
                if entry:
                    if not prep:
                        fetch(entry, scratch / "drive")
                    record.update(source="drive")
                else:
                    _check_model_space(scratch, item["size"] or 0)
                    from huggingface_hub import hf_hub_download

                    source = Path(hf_hub_download(
                        item["repo_id"], item["filename"], revision=item["revision"], local_dir=str(scratch)
                    ))
                    if not source.is_file():
                        raise WorkerError("reviewed model download produced no file")
                    sha256, md5 = _hash_file(source)
                    size = source.stat().st_size
                    if not size or sha256 != item["sha256"] or (item["size"] is not None and size != item["size"]):
                        raise WorkerError("downloaded model sha256 or size mismatch")
                    record.update(source="huggingface")
                    if prep and cache is not None:
                        try:
                            cache.store(item["target"], source, sha256, md5, size)
                            record["cache_stored"] = True
                        except Exception as exc:  # The prep report records an incomplete cache fill.
                            record.update(cache_stored=False, cache_error=_model_download_error(exc))
                    elif not prep:
                        _publish_model(source, target, preserve_existing=preserve_existing)
        except Exception as exc:
            raise WorkerError(f"模型 {item['target']} 恢复失败：{_model_download_error(exc)}") from exc
        record["seconds"] = round(time.time() - started, 1)
        return record

    items = list(by_target.values())
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(local, items))
        if not prep and any(not record.get("source") for record in records):
            _check_model_space(models, sum(item["size"] or 0 for item, record in zip(items, records, strict=True)
                                           if not record.get("source")))
        with tempfile.TemporaryDirectory(prefix=".studio-weights-", dir=comfy) as scratch:
            return list(pool.map(
                lambda args: restore(args[1], args[2], Path(scratch) / str(args[0])),
                ((index, item, record) for index, (item, record) in enumerate(zip(items, records, strict=True))),
            ))


def _http_json(url: str, method: str = "GET", body: dict | None = None, timeout: float = 30,
               *, extra_headers=None) -> dict:
    request = Request(
        url, method=method, headers={"Accept": "application/json", "Content-Type": "application/json", **(extra_headers or {})}
    )
    data = json.dumps(body).encode() if body is not None else None
    try:
        with urlopen(request, data=data, timeout=timeout) as response:
            raw = response.read(MAX_OUTPUT_BYTES + 1)
    except (HTTPError, URLError, OSError, TimeoutError) as exc:
        raise WorkerError("Comfy local API request failed") from exc
    if len(raw) > MAX_OUTPUT_BYTES:
        raise WorkerError("Comfy API response exceeds output bound")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise WorkerError("Comfy local API returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise WorkerError("Comfy local API returned invalid object")
    return value


_OUTPUT_MIME = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif",
    ".mp4": "video/mp4", ".webm": "video/webm",
}
_INPUT_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,79}$")
MAX_INPUT_BYTES = 20 * 1024 * 1024


def _write_input_files(comfy: Path, snapshot: dict) -> None:
    """Reference images the job ships with go into ComfyUI/input, where LoadImage reads them."""
    folder = comfy / "input"
    folder.mkdir(exist_ok=True)
    for item in snapshot.get("input_files") or []:
        name = item.get("filename") if isinstance(item, dict) else None
        if not isinstance(name, str) or not _INPUT_NAME.fullmatch(name):
            raise WorkerError("input file name is not allowed")
        data = base64.b64decode(item.get("content_b64", ""), validate=True)
        if not data or len(data) > MAX_INPUT_BYTES:
            raise WorkerError("input file is empty or too large")
        (folder / name).write_bytes(data)


def _stream_output(response, target: Path, limit: int) -> int:
    total = 0
    partial = target.with_suffix(target.suffix + '.partial')
    try:
        with partial.open('wb') as output:
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > limit:
                    raise WorkerError('outputs exceed configured bound')
                output.write(chunk)
        partial.replace(target)
        return total
    finally:
        partial.unlink(missing_ok=True)


def _prepare_native_weights(api, native_dir, graph, weights, loras, payload, deadline, timeout):
    try:
        config = json.loads((native_dir / "local-model-config.json").read_text())
        headers = {"X-Studio-Model-Channel": config["channel"]}
        started = _http_json(api + "/studio_native/models/prepare-studio", "POST", {
            "workflow": {"prompt": graph}, "weights": weights, "loras": loras,
            "drive_cache": payload.get("drive_cache"),
            "lora_drive_cache": payload.get("lora_drive_cache", payload.get("drive_cache")),
            "use_drive_cache": payload.get("use_drive_cache", True),
            "timeout": timeout,
        }, extra_headers=headers)
    except (OSError, ValueError, KeyError, WorkerError) as exc:
        raise WorkerError("原生 ComfyUI 无法确认创作模式的本地模型准备能力，请重启原生本地会话或改用创作实例") from exc
    ident = started.get("id")
    if not isinstance(ident, str) or not re.fullmatch(r"[0-9a-f]{32}", ident):
        raise WorkerError("Native model preparation did not return a valid ID")
    while time.time() < deadline:
        result = _http_json(api + "/studio_native/models/prepare/" + ident, extra_headers=headers)
        if result.get("state") == "ready":
            return result["models"], {**headers, "X-Studio-Model-Prepared": ident}
        if result.get("state") == "failed":
            raise WorkerError((result.get("message") or "Native Studio model preparation failed")[:1400])
        time.sleep(1)
    raise WorkerError("Native model preparation exceeded this Studio job's runtime limit")


def _creation_service(service_dir, payload, manifest, commit, nodes, checked, user_loras, stages, deadline):
    def environment_fingerprint():
        environment = _service_fingerprint(commit, nodes, checked, manifest)
        encoded = json.dumps([environment, user_loras], sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    fingerprint = environment_fingerprint()
    state = _read_service_state(service_dir)
    if state:
        # An absent PID or different start time is a stale marker. Never signal that PID.
        try:
            metadata = _process_metadata(state["pid"])
        except (KeyError, WorkerError):
            metadata = {}
        if metadata.get("start_id") != state.get("process_start_id"):
            (service_dir / COMFY_SERVICE_STATE).unlink(missing_ok=True)
            state = None
        else:
            _trusted_process_metadata(service_dir, state)
    if state and state["env_fingerprint"] == fingerprint:
        try:
            _http_json(_service_api_url("/system_stats"), timeout=5)
        except WorkerError:
            pass
        else:
            stages.start("warm_reuse")
            return {**state, "cache_reason": "hot_reuse"}, {"source": "hot_reuse"}, []
    if state:
        pid = state["pid"]
        _stop_service_if_owned(service_dir)
        stop_deadline = min(deadline, time.time() + 10)
        while time.time() < stop_deadline:
            try:
                metadata = _process_metadata(pid)
            except WorkerError:
                break
            if metadata["start_id"] != state["process_start_id"] or not metadata["cmdline"]:
                break
            # The worker that started this child exited. Wait for the process group to close.
            time.sleep(0.2)
        else:
            raise WorkerError("上一个 ComfyUI 进程尚未退出")
    comfy = service_dir / "ComfyUI"
    parked = service_dir / "previous"
    if comfy.exists() and state and state["env_fingerprint"] != fingerprint:
        shutil.rmtree(parked, ignore_errors=True)
        comfy.rename(parked)
    service_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    stages.start("restore_env")
    timeout = min(float(payload.get("download_timeout_seconds", 600)), 600)
    env_cache = _drive_cache(payload, "runtime_folder_id") if payload.get("use_drive_cache", True) else None
    env = _restore_env(service_dir, commit, timeout, env_cache, build=False, custom_nodes=nodes)
    stages.start("restore_weights")
    weights = _restore_weights(comfy, manifest, timeout, _drive_cache(payload), prep=False,
                               warm_models=_local_models(parked) if parked.is_dir() else None,
                               use_drive_cache=payload.get("use_drive_cache", True))
    shutil.rmtree(parked, ignore_errors=True)
    if user_loras:
        stages.start("restore_loras")
        _restore_user_loras(comfy, user_loras, _lora_drive_cache(payload))
    stages.start("comfy_start")
    service = _start_comfy_service(service_dir, environment_fingerprint(), "cold_start", deadline=deadline)
    return service, env, weights


def _runtime_evidence(service, payload, warmup):
    return {"instance_id": payload["instance_id"],
            "process": f"{service['pid']}:{service['process_start_id']}",
            "environment": service["env_fingerprint"],
            "workflow_sha256": payload["snapshot"]["workflow_sha256"],
            "cache_reason": service["cache_reason"], "warmup_completed": warmup}


def _remove_warm_outputs(comfy, item):
    for result in (item.get("outputs") or {}).values():
        for category in ("images", "gifs", "videos"):
            for entry in result.get(category, []) if isinstance(result, dict) else []:
                folder = entry.get("type") if isinstance(entry, dict) else None
                if (folder not in {"output", "temp"} or not isinstance(entry.get("filename"), str)
                        or entry.get("subfolder") != "studio_warmup"):
                    continue
                if Path(entry["filename"]).name != entry["filename"]:
                    continue
                base = (comfy / folder).resolve()
                path = (base / str(entry.get("subfolder", "")) / entry["filename"]).resolve()
                if path.is_relative_to(base) and path.is_file():
                    path.unlink()


def _run_comfy(root: Path, payload: dict, manifest: dict, graph: dict, stages: _Stages, *, service_dir=None) -> list[dict]:
    commit = manifest.get("comfy_commit", manifest.get("comfy_version"))
    if not isinstance(commit, str) or not COMMIT.fullmatch(commit):
        raise WorkerError("Comfy workflow must lock a 40-character commit")
    allowlist = manifest.get("node_allowlist")
    if not isinstance(allowlist, list) or not allowlist:
        raise WorkerError("Comfy node allowlist is missing")
    for node in graph.values():
        if not isinstance(node, dict) or node.get("class_type") not in allowlist:
            raise WorkerError("bound graph contains an unreviewed node")
    native_marker = NATIVE_SERVICE_MARKER
    native_state = json.loads(native_marker.read_text()) if native_marker.is_file() and not service_dir else None
    shared_native = native_state is not None
    if shared_native and native_state.get("model_storage_mode") != "local":
        raise WorkerError("创作模式需要 Colab 本地模型。当前原生 ComfyUI 使用 Drive 挂载，请改用兼容实例或停止原生会话后重试")
    if shared_native and native_state.get("commit") != commit:
        raise WorkerError("Studio workflow uses a different ComfyUI version; stop native session before switching versions")
    comfy = Path(native_state["comfy"]) if shared_native else (service_dir or root) / "ComfyUI"
    user_loras = (payload.get("snapshot") or {}).get("lora_files") or []
    _check_loras(graph, manifest, user_loras)
    timeout = min(float(payload.get("download_timeout_seconds", 600)), 600)
    nodes = _reviewed_custom_nodes(manifest.get("custom_nodes"))
    checked = _validate_weights(manifest)
    fingerprint = _warm_fingerprint(commit, nodes, checked)
    deadline = min(time.time() + min(float(payload.get("max_runtime_minutes", 10)) * 60, 600 * 60),
                   float(payload.get("deadline_epoch") or float("inf")))
    warmup = bool((payload.get("snapshot") or {}).get("creation_warmup"))
    service = None
    if service_dir:
        service, env, weights = _creation_service(
            service_dir, payload, manifest, commit, nodes, checked, user_loras, stages, deadline)
    elif shared_native:
        _local_models(comfy)
        _verify_checkout(comfy, commit)
        # The shared process already loaded its nodes. Never install packages or restart it under queued native work.
        for node in nodes:
            folder = comfy / "custom_nodes" / node["dir"]
            actual = _run(["git", "rev-parse", "HEAD"], cwd=folder, timeout=30).stdout.strip()
            if actual != node["commit"]:
                raise WorkerError("Native service custom node version differs from reviewed Studio workflow")
        env = {"source": "native_shared"}
    elif _warm_take(root, fingerprint, nodes, checked, user_loras=user_loras):
        # Same runtime, same environment: the previous job left it restored. Nothing to download.
        stages.start("warm_reuse")
        env = {"source": "warm"}
        weights = [{"target": w["target"], "sha256": w["sha256"], "source": "warm", "seconds": 0.0} for w in checked]
    else:
        stages.start("restore_env")
        env_cache = _drive_cache(payload, "runtime_folder_id") if payload.get("use_drive_cache", True) else None
        env = _restore_env(root, commit, timeout, env_cache, build=False, custom_nodes=nodes)
        stages.start("restore_weights")
        parked = root.parent / ".warm" / "ComfyUI"
        warm_models = _local_models(parked) if parked.is_dir() else None
        weights = _restore_weights(comfy, manifest, timeout, _drive_cache(payload), prep=False,
                                   warm_models=warm_models, use_drive_cache=payload.get("use_drive_cache", True))
    try:
        if user_loras and not (shared_native or service):
            stages.start("restore_loras")
            _restore_user_loras(comfy, user_loras, _lora_drive_cache(payload))
        if not shared_native:
            _local_models(comfy)
            _check_local_tree(comfy / "output")  # ComfyUI also registers model folders under its output directory.
            _write_input_files(comfy, payload.get("snapshot") or {})
        if not service:
            stages.start("comfy_start")
        log_path = root / "comfy.log"
        log = log_path.open("ab", buffering=0)
        process = None
        try:
            port = COMFY_SERVICE_PORT
            if not (shared_native or service):
                with socket.socket() as listener:
                    listener.bind((COMFY_SERVICE_HOST, 0))
                    port = listener.getsockname()[1]
            process = None if shared_native or service else subprocess.Popen(
                [sys.executable, "main.py", "--listen", COMFY_SERVICE_HOST, "--port", str(port)],
                cwd=comfy,
                stdout=log,
                stderr=subprocess.STDOUT,
                shell=False,
                start_new_session=True,
            )
            if not service:
                deadline = time.time() + min(float(payload.get("max_runtime_minutes", 10)) * 60, 600 * 60)
            api = f"http://{COMFY_SERVICE_HOST}:{port}"
            while time.time() < deadline:
                try:
                    _http_json(api + "/system_stats", timeout=5)
                    break
                except WorkerError:
                    if process is not None and process.poll() is not None:
                        raise WorkerError("Comfy worker exited during startup")
                    time.sleep(1)
            else:
                raise WorkerError("Comfy worker startup timed out")
            fence = (_file_lock(native_marker.parent / "models-active.lock", shared=True)
                     if shared_native else contextlib.nullcontext())
            with fence:
                submit_args = {}
                if shared_native:
                    stages.start("restore_weights")
                    weights, headers = _prepare_native_weights(
                        api, native_marker.parent, graph, checked, user_loras, payload, deadline, timeout)
                    submit_args["extra_headers"] = headers
                    _write_input_files(comfy, payload.get("snapshot") or {})
                if process is not None and process.poll() is not None:
                    raise WorkerError("Comfy worker exited before queue submission")
                if time.time() >= deadline:
                    raise WorkerError("Comfy execution timed out before submission")
                stages.start("creation_warming" if warmup else "sampling")
                if warmup:
                    # Isolate reviewed outputs that support a prefix. Otherwise keep the private
                    # VM file in case Comfy returns a cached result to the first real request.
                    for node in graph.values():
                        if "filename_prefix" in node["inputs"]:
                            node["inputs"]["filename_prefix"] = "studio_warmup/" + payload["execution_id"]
                try:
                    queued = _http_json(api + "/prompt", "POST", {"prompt": graph}, timeout=30, **submit_args)
                except WorkerError as exc:
                    if shared_native:
                        raise WorkerError("原生 ComfyUI 拒绝创作任务入队，请检查当前模型路径和工作流") from exc
                    raise
            prompt_id = queued.get("prompt_id")
            if not isinstance(prompt_id, str) or not prompt_id:
                raise WorkerError("Comfy queue did not return a prompt id")
            while time.time() < deadline:
                history = _http_json(api + "/history/" + prompt_id, timeout=30)
                item = history.get(prompt_id)
                if not isinstance(item, dict):
                    time.sleep(2)
                    continue
                status = item.get("status") or {}
                if status.get("status_str") in {"error", "failed"}:
                    detail = ""
                    for message in status.get("messages") or []:
                        if isinstance(message, list) and message and message[0] == "execution_error":
                            data = message[1] if len(message) > 1 and isinstance(message[1], dict) else {}
                            detail = f"{data.get('node_type')}: {data.get('exception_message')}"
                    raise WorkerError(f"Comfy workflow failed: {detail}"[:500])
                if status.get("completed") is not True and (service or not item.get("outputs")):
                    time.sleep(2)
                    continue
                stages.end()
                if service:
                    _atomic_json(root / "creation-runtime.json", _runtime_evidence(service, payload, warmup))
                if warmup:
                    _remove_warm_outputs(comfy, item)
                    return []
                outputs = []
                total = 0
                output_dir = root / "outputs"
                output_dir.mkdir(mode=0o700, exist_ok=True)
                for node_id, node_output in (item.get("outputs") or {}).items():
                    if not isinstance(node_output, dict):
                        continue
                    for category in ("images", "gifs", "videos"):
                        for entry in node_output.get(category, []) or []:
                            if not isinstance(entry, dict) or not isinstance(entry.get("filename"), str):
                                continue
                            filename = Path(entry["filename"]).name
                            mime = _OUTPUT_MIME.get(Path(filename).suffix.lower())
                            if mime is None:  # SaveVideo reports videos under "images"; judge by extension
                                continue
                            query = urlencode({'filename':filename,'subfolder':entry.get('subfolder',''),'type':entry.get('type','output')})
                            request = Request(
                                api + "/view?" + query, headers={"Accept": "application/octet-stream"}
                            )
                            stored_name = filename
                            if service:
                                stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", Path(filename).stem)[:100]
                                stored_name = f"{len(outputs)}-{stem}{Path(filename).suffix.lower()}"
                            target = output_dir / stored_name
                            output_limit = payload.get('_local_artifact_limit', MAX_OUTPUT_BYTES)
                            with urlopen(request, timeout=30) as response:
                                size = _stream_output(response, target, output_limit - total)
                            total += size
                            outputs.append(
                                {
                                    "filename": target.name,
                                    "mime": mime,
                                    **({"comfy_node": node_id, "comfy_category": category} if service else {}),
                                    "path": str(target.relative_to(root)),
                                    "size": size,
                                    "metadata": {
                                        "comfy_commit": commit,
                                        "weights": weights,
                                        "env": env,
                                        "stage_seconds": stages.timings,
                                    },
                                }
                            )
                return outputs
            raise WorkerError("Comfy execution timed out")
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
            log.close()
    finally:
        if not (shared_native or service_dir):
            _warm_put(root, fingerprint)  # restore succeeded above, so the tree is reusable even if this job failed later


def _run_prep(root: Path, payload: dict, manifest: dict, stages: _Stages) -> list[dict]:
    """CPU-only cache fill: build the env archive and put every missing weight into Drive. No sampling."""
    commit = manifest.get("comfy_commit", manifest.get("comfy_version"))
    if not isinstance(commit, str) or not COMMIT.fullmatch(commit):
        raise WorkerError("Comfy workflow must lock a 40-character commit")
    if _drive_cache(payload) is None:
        raise WorkerError("prep needs the Drive cache")
    timeout = min(float(payload.get("download_timeout_seconds", 600)), 600)
    stages.start("prep_env")
    env = _restore_env(root, commit, timeout, _drive_cache(payload, "runtime_folder_id"), build=True,
                       custom_nodes=_reviewed_custom_nodes(manifest.get("custom_nodes")))
    stages.start("prep_weights")
    weights = _restore_weights(root / "ComfyUI", manifest, timeout, _drive_cache(payload), prep=True)
    stages.end()
    failed = [item["target"] for item in weights if item.get("cache_stored") is False]
    if env.get("source") != "drive" and env.get("cache_stored") is not True:
        failed.append("env")
    if failed:
        raise WorkerError("cache fill incomplete: " + ", ".join(failed))
    report = {
        "env": env,
        "weights": weights,
        "stage_seconds": stages.timings,
        "python": sys.version.split()[0],
        "torch": importlib.metadata.version("torch"),
    }
    path = root / "outputs" / "prep-report.json"
    _atomic_json(path, report)
    return [
        {
            "filename": "prep-report.json",
            "mime": "application/json",
            "path": str(path.relative_to(root)),
            "size": path.stat().st_size,
            "metadata": report,
        }
    ]


def run_job(payload: dict, root: Path, *, artifact_store=None) -> None:
    stages = _Stages(root)
    try:
        _status(root, "running")
        snapshot, manifest, graph = _snapshot(payload)
        if snapshot.get("prep") is not True:
            diagnostics = _probe_gpu()
            if diagnostics:
                _status(root, "running", gpu_diagnostics=diagnostics)
            _check_compute_hardware(manifest, diagnostics, snapshot.get("runtime_compute_spec"))
        if snapshot.get("native_comfy") is True:
            from native_runtime import run as run_native
            outputs = run_native(root, payload, stages)
        elif snapshot.get("system_smoke") is True:
            if manifest.get("executor") != "cuda-smoke":
                raise WorkerError("system smoke requires the cuda-smoke executor manifest")
            outputs = [run_cuda_smoke(root)]
        elif snapshot.get("prep") is True:
            outputs = _run_prep(root, payload, manifest, stages)
        else:
            local_payload = {**payload, '_local_artifact_limit': min(artifact_store.max_size, 2*1024**3)} if artifact_store is not None else {k:v for k,v in payload.items() if k!='_local_artifact_limit'}
            if snapshot.get("creation_api"):
                service_dir = _comfy_service_dir(payload.get("instance_id"))
                with _file_lock(service_dir.parent / "creation-service.lock"):
                    outputs = _run_comfy(root, local_payload, manifest, graph, stages, service_dir=service_dir)
            else:
                outputs = _run_comfy(root, local_payload, manifest, graph, stages)
        runtime_path = root / "creation-runtime.json"
        evidence = {"creation_runtime": json.loads(runtime_path.read_text())} if snapshot.get("creation_api") else {}
        if artifact_store is not None:
            artifacts = []
            for item in outputs:
                source = root / item['path']
                if source.is_symlink() or root.resolve() not in source.resolve().parents:
                    raise WorkerError('worker output escaped execution directory')
                artifact = artifact_store.publish_file(payload["owner_id"], payload["execution_id"], source, item["mime"])
                artifacts.append({**artifact, **{k: item[k] for k in ("comfy_node", "comfy_category") if k in item}})
            _atomic_json(root / 'meta.json', {'execution_id': payload['execution_id'], 'artifacts': artifacts})
            _status(root, 'completed', artifacts=artifacts, timings=stages.timings, **evidence)
        else:
            _atomic_json(root / "meta.json", {"execution_id": payload.get("execution_id"), "outputs": outputs})
            _status(root, "completed", outputs=outputs, timings=stages.timings, **evidence)
    except Exception as exc:  # noqa: BLE001 - worker must persist every failure
        stages.end()
        message = _safe_error(exc)
        log = root / "comfy.log"
        if not log.is_file() or not log.stat().st_size:
            if (payload.get("snapshot") or {}).get("creation_api") and payload.get("instance_id"):
                with contextlib.suppress(OSError, WorkerError):
                    log = _comfy_service_dir(payload["instance_id"]) / "comfy.log"
        if log.is_file():
            with log.open("rb") as stream:
                stream.seek(max(0, log.stat().st_size - 500))
                message = (message + " | comfy.log: " + stream.read(500).decode("utf-8", "replace"))[:1400]
        _status(root, "failed", error_code="gpu_worker_failed", error_message=message, timings=stages.timings)


class _MCPManifestStore:
    """Worker-side descriptor writer; media stays on disk for bounded MCP reads."""
    max_size = 2 * 1024**3

    def __init__(self, root):
        self.root = root.resolve()

    def publish_file(self, owner, execution, source, mime):
        import hashlib

        if not owner or not execution or source.is_symlink() or self.root not in source.resolve().parents:
            raise WorkerError('invalid artifact identity or path')
        size = source.stat().st_size
        if not 0 < size <= self.max_size:
            raise WorkerError('artifact size invalid')
        digest = hashlib.sha256()
        with source.open('rb') as file:
            while chunk := file.read(1024**2):
                digest.update(chunk)
        return {'filename': source.name, 'path': str(source.relative_to(self.root)),
            'mime': mime, 'size_bytes': size, 'sha256': digest.hexdigest()}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) == 2 and argv[0] == "workbench":
        return _main_workbench_action(argv[1])
    if len(argv) != 1:
        return 2
    payload_path = Path(argv[0]).resolve()
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise WorkerError("payload must be an object")
        root = Path(payload["execution_dir"]).resolve()
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        secret_fields = {"drive_cache", "lora_drive_cache", "native_secrets"}
        if secret_fields & payload.keys():  # keep credentials out of files that outlive the worker's read
            payload_path.chmod(0o600)
            _atomic_json(payload_path, {k: v for k, v in payload.items() if k not in secret_fields})
            payload_path.chmod(0o600)
        artifacts = _MCPManifestStore(root) if payload.get('output_transport') == 'mcp-chunks-v1' else None
        run_job(payload, root, artifact_store=artifacts)
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI boundary is fail-closed
        try:
            root = (
                Path(payload.get("execution_dir", payload_path.parent)).resolve()
                if isinstance(payload, dict)
                else payload_path.parent
            )
            _status(root, "failed", error_code="gpu_worker_bootstrap_failed", error_message=_safe_error(exc))
        except Exception:  # noqa: BLE001 - status persistence must not mask exit code
            return 1
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
