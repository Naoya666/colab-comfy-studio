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
import importlib
import importlib.metadata
import io
import json
import math
import os
import random
import re
import shlex
import shutil
import signal
import site
import socket
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import threading
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
_MULTI_STREAM_MIN_BYTES = 128 * 1024 * 1024
_MULTI_STREAM_STREAM_CHUNK = 4 * 1024 * 1024
_DRIVE_STREAM_SEM = threading.Semaphore(12)
_CONTENT_RANGE_RE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")
_LRU_RESERVE_BYTES = 5 * 1024**3
_PROTECTED_DIST_NAMES = frozenset({"torch", "torchvision", "torchaudio", "triton"})
_PROTECTED_TOP_DIRS = frozenset({"torch", "torchvision", "torchaudio", "triton", "torchgen", "functorch"})
CREATION_FLAG_NAMES = (
    "CREATION_USE_UV",
    "CREATION_PARALLEL_BOOTSTRAP",
    "CREATION_MULTI_STREAM_DRIVE",
    "CREATION_DRIVE_STREAMS",
    "CREATION_LOCAL_HASH_CACHE",
    "CREATION_FAST_CONTROL_PLANE",
    "CREATION_ENV_CACHE",
)
_LOCAL_HASH_LOCK = threading.Lock()
_LOCAL_HASH_CACHE: dict[str, dict[str, Any]] = {}
_LOCAL_HASH_LOADED_PATH: Path | None = None
_RETRYABLE_HTTP_CODES = frozenset({429, 500, 502, 503, 504})


def _creation_flag(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() not in {"0", "false", "off", "no"}


def creation_optimization_flags() -> dict[str, str]:
    defaults = {
        "CREATION_USE_UV": "1",
        "CREATION_PARALLEL_BOOTSTRAP": "1",
        "CREATION_MULTI_STREAM_DRIVE": "0",
        "CREATION_DRIVE_STREAMS": "4",
        "CREATION_LOCAL_HASH_CACHE": "1",
        "CREATION_FAST_CONTROL_PLANE": "1",
        "CREATION_ENV_CACHE": "auto",
    }
    return {name: (os.getenv(name) or default).strip() or default for name, default in defaults.items()}


def _env_cache_mode() -> str:
    raw = (os.getenv("CREATION_ENV_CACHE") or "auto").strip().lower()
    if raw in {"0", "false", "off", "no"}:
        return "off"
    if raw in {"1", "true", "on", "yes", "force"}:
        return "force"
    return "auto"
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
    # Later stage updates keep the measured GPU and VM baseline metadata so the coordinator can still poll them.
    path = root / "status.json"
    if path.exists():
        try:
            previous_doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous_doc = {}
        if isinstance(previous_doc, dict):
            for sticky in ("gpu_diagnostics", "venv_isolated", "vm_baseline_digest", "clean_after_smoke"):
                if sticky not in fields and sticky in previous_doc and previous_doc[sticky] is not None:
                    fields[sticky] = previous_doc[sticky]
    _atomic_json(root / "status.json", {"state": state, "updated_at": time.time(), **fields})


def venv_python_path(venv_dir: Path) -> Path:
    """Return the Python binary path inside a domain virtual environment."""
    return Path(venv_dir) / "bin" / "python"


def venv_site_dir(venv_dir: Path) -> Path:
    """Return the site-packages directory inside a domain virtual environment."""
    vdir = Path(venv_dir)
    py_tag = f"python{sys.version_info.major}.{sys.version_info.minor}"
    preferred = vdir / "lib" / py_tag / "site-packages"
    if preferred.is_dir():
        return preferred
    lib_dir = vdir / "lib"
    if lib_dir.is_dir():
        for cand in sorted(lib_dir.glob("python*/site-packages")):
            if cand.is_dir():
                return cand
        for cand in sorted(lib_dir.glob("python*/dist-packages")):
            if cand.is_dir():
                return cand
    preferred.mkdir(mode=0o755, parents=True, exist_ok=True)
    return preferred


def _venv_has_system_site_packages(venv_dir: Path) -> bool:
    cfg = Path(venv_dir) / "pyvenv.cfg"
    if not cfg.is_file():
        return False
    try:
        for line in cfg.read_text(encoding="utf-8", errors="replace").splitlines():
            key, _, val = line.partition("=")
            if key.strip().lower() == "include-system-site-packages":
                return val.strip().lower() == "true"
    except OSError:
        return False
    return False


def _ensure_venv_pip_shims(venv_dir: Path, py_bin: Path) -> None:
    """Install pip/pip3 shims in venv/bin so bare `pip` invocations stay inside the domain venv."""
    vbin = Path(venv_dir) / "bin"
    if not vbin.is_dir():
        return
    quoted_py = shlex.quote(str(py_bin))
    script = f"#!/bin/sh\nexec {quoted_py} -m pip \"$@\"\n"
    for name in ("pip", "pip3", f"pip{sys.version_info.major}.{sys.version_info.minor}"):
        shim = vbin / name
        if shim.is_symlink():
            continue
        try:
            if shim.is_file() and shim.read_text(encoding="utf-8", errors="replace") == script:
                continue
            shim.write_text(script, encoding="utf-8")
            shim.chmod(0o755)
        except OSError:
            pass


def ensure_domain_venv(venv_dir: Path, *, timeout: float = 120.0) -> Path:
    """Create or reuse a domain-isolated venv inheriting Colab's system site-packages."""
    vdir = Path(venv_dir)
    py_bin = venv_python_path(vdir)
    if py_bin.is_file() and _venv_has_system_site_packages(vdir):
        _ensure_venv_pip_shims(vdir, py_bin)
        return py_bin
    if vdir.exists():
        shutil.rmtree(vdir, ignore_errors=True)
    vdir.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    uv_bin = shutil.which("uv")
    if uv_bin and Path(uv_bin).is_file():
        try:
            uv_env = dict(os.environ)
            uv_env["UV_SYSTEM_PYTHON"] = "false"
            res = subprocess.run(
                [uv_bin, "venv", "--system-site-packages", "--python", sys.executable, str(vdir)],
                env=uv_env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if res.returncode == 0 and py_bin.is_file() and _venv_has_system_site_packages(vdir):
                _ensure_venv_pip_shims(vdir, py_bin)
                return py_bin
        except (OSError, TypeError, subprocess.SubprocessError):
            pass
        shutil.rmtree(vdir, ignore_errors=True)
    try:
        import venv as _venv

        _venv.EnvBuilder(system_site_packages=True, symlinks=True, with_pip=False).create(str(vdir))
    except Exception as exc:  # noqa: BLE001
        raise WorkerError(f"创建域隔离 venv 失败：{_safe_error(exc)}") from exc
    if not py_bin.is_file():
        raise WorkerError("创建域隔离 venv 失败：未生成 venv python 可执行文件")
    _ensure_venv_pip_shims(vdir, py_bin)
    return py_bin


def domain_python_env(
    venv_dir: Path | None,
    *,
    base_env: dict[str, str] | None = None,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """Construct a clean subprocess environment pinned to a domain venv with PYTHONNOUSERSITE=1."""
    env = dict(os.environ if base_env is None else base_env)
    env["PYTHONNOUSERSITE"] = "1"
    # Colab sets UV_SYSTEM_PYTHON=true in the default container environment, which causes
    # `uv pip install` (e.g. from ComfyUI-Manager or plugins) to bypass VIRTUAL_ENV and pollute
    # global dist-packages. Force uv and pip to stay strictly inside the domain venv.
    env["UV_SYSTEM_PYTHON"] = "false"
    env.pop("PYTHONPATH", None)
    if venv_dir is not None:
        vdir = Path(venv_dir)
        env["VIRTUAL_ENV"] = str(vdir)
        env["PIP_REQUIRE_VIRTUALENV"] = "true"
        vbin = str(vdir / "bin")
        old_path = env.get("PATH", "")
        parts = [p for p in old_path.split(os.pathsep) if p and p != vbin]
        env["PATH"] = os.pathsep.join([vbin, *parts])
    if extra:
        for k, v in extra.items():
            if v is None:
                env.pop(k, None)
            else:
                env[k] = str(v)
    return env


def _global_site_dirs() -> list[Path]:
    """Return global system site/dist-packages directories monitored by VM baseline snapshots."""
    default_purelib = Path(sysconfig.get_paths()["purelib"])
    configured = _site_dir()
    if configured != default_purelib:
        return [configured] if configured.is_dir() else []
    candidates: list[Path] = []
    seen: set[str] = set()
    raw_dirs: list[str] = [
        sysconfig.get_paths().get("purelib", ""),
        sysconfig.get_paths().get("platlib", ""),
    ]
    try:
        raw_dirs.extend(site.getsitepackages())
    except Exception:
        pass
    for raw in raw_dirs:
        if not raw:
            continue
        p = Path(raw)
        if not p.is_dir():
            continue
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            candidates.append(p.resolve())
    return sorted(candidates, key=lambda item: str(item))


def _user_site_packages_dirty(home_dir: Path | None = None) -> list[str]:
    """Detect any packages installed under ~/.local/lib/python*/site-packages (or dist-packages)."""
    home = Path(home_dir) if home_dir is not None else Path.home()
    local_lib = home / ".local" / "lib"
    dirty: list[str] = []
    if not local_lib.is_dir():
        return dirty
    for py_dir in sorted(local_lib.glob("python*")):
        if not py_dir.is_dir() or py_dir.is_symlink():
            continue
        for sub_name in ("site-packages", "dist-packages"):
            sub = py_dir / sub_name
            if not sub.is_dir() or sub.is_symlink():
                continue
            for entry in sorted(sub.iterdir(), key=lambda p: p.name):
                if entry.name == "__pycache__":
                    continue
                dirty.append(str(entry.relative_to(home)))
    return dirty


def _hash_small_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def global_site_snapshot(site_dirs: list[Path] | None = None) -> dict[str, Any]:
    """Capture a fast (<50ms) metadata snapshot of global site-packages immune to __pycache__ churn."""
    dirs = _global_site_dirs() if site_dirs is None else [Path(d) for d in site_dirs if Path(d).is_dir()]
    roots_data: dict[str, Any] = {}
    protected: dict[str, str] = {}
    for root_dir in sorted(dirs, key=lambda p: str(p)):
        entries: dict[str, Any] = {}
        dists: dict[str, Any] = {}
        try:
            children = sorted(root_dir.iterdir(), key=lambda p: p.name)
        except OSError:
            children = []
        for entry in children:
            name = entry.name
            if name == "__pycache__" or name.startswith((".studio-creation-", ".studio-native-")):
                continue
            try:
                st = entry.lstat()
            except OSError:
                continue
            if entry.is_symlink():
                try:
                    target = os.readlink(entry)
                except OSError:
                    target = ""
                entries[name] = ["l", target]
            elif entry.is_dir():
                # Record directory name/kind only; importing system packages creates __pycache__
                # inside directories and mutates directory mtime/size without altering packages.
                entries[name] = ["d"]
                if name.endswith((".dist-info", ".egg-info")):
                    meta_hashes: dict[str, Any] = {}
                    dist_name = None
                    dist_ver = None
                    for mf in ("METADATA", "PKG-INFO", "direct_url.json"):
                        mp = entry / mf
                        if mp.is_file() and not mp.is_symlink():
                            raw_bytes = mp.read_bytes()
                            meta_hashes[mf] = hashlib.sha256(raw_bytes).hexdigest()
                            if mf in {"METADATA", "PKG-INFO"} and dist_name is None:
                                for line in raw_bytes.decode("utf-8", errors="replace").splitlines():
                                    low = line.lower()
                                    if low.startswith("name:"):
                                        dist_name = _norm_dist_name(line.split(":", 1)[1])
                                    elif low.startswith("version:"):
                                        dist_ver = line.split(":", 1)[1].strip()
                                    elif not line.strip():
                                        break
                    rec = entry / "RECORD"
                    if rec.is_file() and not rec.is_symlink():
                        try:
                            rst = rec.lstat()
                            meta_hashes["RECORD"] = [rst.st_size, rst.st_mtime_ns]
                        except OSError:
                            pass
                    dists[name] = meta_hashes
                    if dist_name and dist_ver and _is_protected_dist(dist_name):
                        protected[dist_name] = dist_ver
            else:
                if name.endswith((".pth", ".egg-link")):
                    entries[name] = ["pth", _hash_small_file(entry)]
                else:
                    entries[name] = ["f", st.st_size, st.st_mtime_ns]
        roots_data[str(root_dir)] = {"entries": entries, "dists": dists}
    return {"schema": 1, "roots": roots_data, "protected": protected}


def global_site_digest(site_dirs: list[Path] | None = None) -> tuple[str, dict[str, Any]]:
    snap = global_site_snapshot(site_dirs)
    raw = json.dumps(snap, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest(), snap


def _default_studio_root(exec_root: Path | None = None) -> Path:
    default_service_root = Path("/content/ai-studio/comfy-services")
    if COMFY_SERVICE_ROOT != default_service_root:
        return COMFY_SERVICE_ROOT.parent
    if exec_root is not None:
        return Path(exec_root).resolve().parent
    return Path("/content/ai-studio")


def ensure_vm_baseline(studio_root: Path | None = None, *, site_dirs: list[Path] | None = None) -> dict[str, Any]:
    """Capture once and persist .vm-baseline.json before any pip/uv install occurs on this VM."""
    base_root = Path(studio_root) if studio_root is not None else _default_studio_root()
    base_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    baseline_path = base_root / ".vm-baseline.json"
    lock_path = base_root / ".vm-baseline.lock"
    with _file_lock(lock_path):
        if baseline_path.is_file():
            try:
                existing = json.loads(baseline_path.read_text(encoding="utf-8"))
                if (
                    isinstance(existing, dict)
                    and existing.get("schema") == 1
                    and isinstance(existing.get("digest"), str)
                    and SHA256.fullmatch(existing["digest"])
                ):
                    return existing
            except (OSError, ValueError, TypeError):
                pass
        digest_hex, snap = global_site_digest(site_dirs=site_dirs)
        record = {
            "schema": 1,
            "digest": digest_hex,
            "captured_at": time.time(),
            "protected": snap.get("protected") or {},
            "roots": sorted((snap.get("roots") or {}).keys()),
            "snapshot": snap,
        }
        _atomic_json(baseline_path, record)
        return record


def _diff_site_snapshots(expected_snap: dict[str, Any] | None, actual_snap: dict[str, Any] | None, *, limit: int = 20) -> dict[str, list[str]]:
    """Compute a compact diff between two global_site_snapshot() dicts for diagnostics."""
    if not isinstance(expected_snap, dict) or not isinstance(actual_snap, dict):
        return {}
    exp_roots = expected_snap.get("roots") if isinstance(expected_snap.get("roots"), dict) else {}
    act_roots = actual_snap.get("roots") if isinstance(actual_snap.get("roots"), dict) else {}
    added: list[str] = []
    removed: list[str] = []
    changed: list[str] = []
    for root_str in sorted(set(exp_roots) | set(act_roots)):
        e_doc = exp_roots.get(root_str) if isinstance(exp_roots.get(root_str), dict) else {}
        a_doc = act_roots.get(root_str) if isinstance(act_roots.get(root_str), dict) else {}
        e_entries = e_doc.get("entries") if isinstance(e_doc.get("entries"), dict) else {}
        a_entries = a_doc.get("entries") if isinstance(a_doc.get("entries"), dict) else {}
        e_dists = e_doc.get("dists") if isinstance(e_doc.get("dists"), dict) else {}
        a_dists = a_doc.get("dists") if isinstance(a_doc.get("dists"), dict) else {}
        for name in sorted(set(e_entries) | set(a_entries)):
            label = f"{root_str}/{name}"
            if name not in e_entries:
                if len(added) < limit:
                    added.append(label)
            elif name not in a_entries:
                if len(removed) < limit:
                    removed.append(label)
            elif e_entries[name] != a_entries[name] or e_dists.get(name) != a_dists.get(name):
                if len(changed) < limit:
                    changed.append(label)
    out: dict[str, list[str]] = {}
    if added:
        out["added"] = added
    if removed:
        out["removed"] = removed
    if changed:
        out["changed"] = changed
    return out


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
    cmd0 = Path(command[0])
    venv_dir = service_dir / "venv"
    venv_bin = (venv_dir / "bin").resolve()
    if venv_dir.is_dir() or state.get("venv_isolated") is True:
        cmd0_ok = cmd0.parent.resolve() == venv_bin and cmd0.name.startswith("python")
    else:
        cmd0_ok = cmd0.name == Path(sys.executable).name or (
            cmd0.parent.resolve() == venv_bin and cmd0.name.startswith("python")
        )
    if not cmd0_ok or command[1] != "main.py":
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
    if path not in {"/system_stats", "/interrupt", "/queue"}:
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
    venv_dir = service_dir / "venv"
    venv_py = venv_python_path(venv_dir)
    use_venv = venv_py.is_file()
    python_bin = str(venv_py) if use_venv else sys.executable
    proc_env = domain_python_env(venv_dir if use_venv else None)
    log_path = service_dir / "comfy.log"
    log_handle = log_path.open("ab", buffering=0)
    args = [python_bin, "main.py", "--listen", COMFY_SERVICE_HOST, "--port", str(COMFY_SERVICE_PORT)]
    try:
        process = subprocess.Popen(
            args,
            cwd=comfy,
            env=proc_env,
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
            "venv_isolated": use_venv,
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
            poll_interval = 0.3 if _creation_flag("CREATION_FAST_CONTROL_PLANE", True) else 1.0
            while time.time() < deadline:
                try:
                    _http_json(_service_api_url("/system_stats"), timeout=5)
                    break
                except WorkerError:
                    if process.poll() is not None:
                        raise WorkerError("Comfy worker exited during startup")
                    time.sleep(poll_interval)
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
    args: list[str], *, cwd: Path | None = None, timeout: float, capture: bool = True, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess:
    if not isinstance(args, list) or not args or any(not isinstance(item, str) for item in args):
        raise WorkerError("internal command must be an argument list")
    try:
        return subprocess.run(
            args, cwd=cwd, timeout=timeout, check=True, shell=False, capture_output=capture, text=True, env=env
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
    if not items:
        return

    def _clone_one(item: dict) -> None:
        folder = comfy / "custom_nodes" / item["dir"]
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(mode=0o700, parents=True)
        _run(["git", "init", "-q"], cwd=folder, timeout=30)
        _run(["git", "fetch", "-q", "--depth", "1", item["repo"], item["commit"]], cwd=folder, timeout=timeout)
        _run(["git", "checkout", "-q", "--detach", "FETCH_HEAD"], cwd=folder, timeout=timeout)

    if len(items) > 1:
        with ThreadPoolExecutor(max_workers=min(4, len(items))) as pool:
            list(pool.map(_clone_one, items))
    else:
        _clone_one(items[0])
    _verify_custom_nodes(comfy, items)


def _nodes_need_pip(comfy: Path, items: list[dict]) -> bool:
    for item in items or ():
        folder = comfy / "custom_nodes" / item["dir"]
        if (folder / "requirements.txt").is_file() or (folder / "install.py").is_file():
            return True
    return False


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
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _file_stamp(st: os.stat_result) -> list[int]:
    return [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]


def _hash_cache_file() -> Path | None:
    try:
        root = COMFY_SERVICE_ROOT.expanduser()
        if root.exists() and root.is_symlink():
            return None
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        return root / ".model-hash-cache.json"
    except OSError:
        return None


def _load_local_hash_cache_locked() -> None:
    global _LOCAL_HASH_LOADED_PATH
    cache_file = _hash_cache_file()
    if cache_file is None or _LOCAL_HASH_LOADED_PATH == cache_file:
        return
    _LOCAL_HASH_CACHE.clear()
    try:
        if cache_file.is_file() and not cache_file.is_symlink():
            raw = json.loads(cache_file.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                for key, entry in raw.items():
                    if isinstance(key, str) and isinstance(entry, dict) and isinstance(entry.get("stamp"), list):
                        _LOCAL_HASH_CACHE[key] = entry
    except (OSError, ValueError, TypeError):
        pass
    _LOCAL_HASH_LOADED_PATH = cache_file


def _save_local_hash_cache_locked() -> None:
    cache_file = _hash_cache_file()
    if cache_file is None:
        return
    if len(_LOCAL_HASH_CACHE) > 2048:
        for key in list(_LOCAL_HASH_CACHE.keys())[:-1024]:
            _LOCAL_HASH_CACHE.pop(key, None)
    try:
        _atomic_json(cache_file, _LOCAL_HASH_CACHE)
    except OSError:
        pass


def _record_local_hash(path: Path, sha256: str | None = None, md5: str | None = None) -> None:
    if not _creation_flag("CREATION_LOCAL_HASH_CACHE", True) or (not sha256 and not md5):
        return
    try:
        resolved = path.resolve()
        st = resolved.stat()
    except OSError:
        return
    stamp = _file_stamp(st)
    key = str(resolved)
    with _LOCAL_HASH_LOCK:
        _load_local_hash_cache_locked()
        existing = _LOCAL_HASH_CACHE.get(key)
        entry: dict[str, Any] = {"stamp": stamp}
        if isinstance(existing, dict) and existing.get("stamp") == stamp:
            if existing.get("sha256"):
                entry["sha256"] = existing["sha256"]
            if existing.get("md5"):
                entry["md5"] = existing["md5"]
        if sha256:
            entry["sha256"] = sha256.lower()
        if md5:
            entry["md5"] = md5.lower()
        _LOCAL_HASH_CACHE[key] = entry
        _save_local_hash_cache_locked()


def _lookup_local_hash(path: Path, stamp: list[int], *, md5: bool = False) -> str | None:
    if not _creation_flag("CREATION_LOCAL_HASH_CACHE", True):
        return None
    try:
        key = str(path.resolve())
    except OSError:
        return None
    with _LOCAL_HASH_LOCK:
        _load_local_hash_cache_locked()
        entry = _LOCAL_HASH_CACHE.get(key)
        if not isinstance(entry, dict) or entry.get("stamp") != stamp:
            return None
        value = entry.get("md5" if md5 else "sha256")
        return value if isinstance(value, str) and value else None


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
    stamp = _file_stamp(before)
    cached = _lookup_local_hash(path, stamp, md5=md5)
    if cached is not None:
        after = path.stat()
        return cached == digest and (before.st_ino, before.st_ctime_ns) == (after.st_ino, after.st_ctime_ns)
    if md5:
        actual_sha, actual_md5 = _hash_file(path)
        actual = actual_md5
    else:
        actual_sha, actual_md5 = _sha256(path), None
        actual = actual_sha
    after = path.stat()
    matched = actual == digest and (before.st_ino, before.st_ctime_ns) == (after.st_ino, after.st_ctime_ns)
    if matched:
        _record_local_hash(path, sha256=actual_sha, md5=actual_md5)
    return matched


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


def _has_nvidia_gpu() -> bool:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        return bool(result.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return False


def _norm_dist_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name.strip()).lower()


def _is_protected_dist(name: str) -> bool:
    norm = _norm_dist_name(name)
    return norm in _PROTECTED_DIST_NAMES or norm.startswith("nvidia-")


def _is_protected_site_entry(name: str, site: Path | None = None) -> bool:
    if name in _PROTECTED_TOP_DIRS:
        return True
    if name.endswith((".dist-info", ".egg-info")):
        base = name.rsplit(".", 1)[0]
        prefix = base.split("-", 1)[0]
        if _is_protected_dist(prefix):
            return True
        if site is not None:
            meta = site / name / "METADATA"
            if meta.is_file():
                try:
                    for line in meta.read_text(errors="replace").splitlines():
                        if line.lower().startswith("name:"):
                            return _is_protected_dist(line.split(":", 1)[1])
                        if not line.strip():
                            break
                except OSError:
                    pass
    return False


def _protect_creation_environment(root: Path) -> tuple[dict[str, str], Path]:
    """Pin preinstalled PyTorch/CUDA distributions in gpu-constraints.txt."""
    protected: dict[str, str] = {}
    for dist in importlib.metadata.distributions():
        raw_name = dist.metadata.get("Name")
        if not raw_name or not dist.version:
            continue
        norm = _norm_dist_name(raw_name)
        if _is_protected_dist(norm):
            protected[norm] = dist.version
    constraints = root / "gpu-constraints.txt"
    constraints.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    lines = [f"{name}=={ver}" for name, ver in sorted(protected.items())]
    constraints.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return protected, constraints


def _cuda_build_env(constraints: Path, comfy: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PIP_CONSTRAINT"] = str(constraints)
    env["UV_CONSTRAINT"] = str(constraints)
    env["COMFYUI_PATH"] = str(comfy)
    if "TORCH_CUDA_ARCH_LIST" not in env:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                capture_output=True, text=True, check=True, timeout=5,
            ).stdout
            caps = sorted({line.strip() for line in out.splitlines() if re.fullmatch(r"\d+\.\d+", line.strip())})
            if caps:
                env["TORCH_CUDA_ARCH_LIST"] = ";".join(caps)
        except (OSError, subprocess.SubprocessError):
            pass
    env.setdefault("MAX_JOBS", str(min(8, max(2, os.cpu_count() or 4))))
    return env


def _pip_install(
    comfy: Path,
    timeout: float,
    *,
    root: Path | None = None,
    custom_nodes: list[dict] | None = None,
    venv_dir: Path | None = None,
) -> dict[str, Any]:
    """Install ComfyUI base requirements + huggingface_hub (+ optional custom node deps) inside a domain venv."""
    t0 = time.monotonic()
    constraint_root = root or comfy.parent
    target_venv = Path(venv_dir) if venv_dir is not None else (constraint_root / "venv")
    venv_python = ensure_domain_venv(target_venv, timeout=min(120.0, timeout))
    before_protected, constraints = _protect_creation_environment(constraint_root)
    build_env = domain_python_env(target_venv, base_env=_cuda_build_env(constraints, comfy))
    req_args = ["-r", str(comfy / "requirements.txt")]
    node_install_scripts: list[Path] = []
    for item in custom_nodes or ():
        folder = comfy / "custom_nodes" / item["dir"]
        node_req = folder / "requirements.txt"
        if node_req.is_file():
            req_args.extend(["-r", str(node_req)])
        node_install = folder / "install.py"
        if node_install.is_file():
            node_install_scripts.append(node_install)

    installer = "pip"
    uv_bin = shutil.which("uv") if _creation_flag("CREATION_USE_UV", True) else None
    installed_via_uv = False
    if uv_bin:
        cmd = [
            uv_bin, "pip", "install", "--no-system", "--python", str(venv_python),
            "--constraint", str(constraints), "-q", "huggingface_hub", *req_args,
        ]
        try:
            _run(cmd, cwd=comfy, timeout=timeout, env=build_env)
            installed_via_uv = True
            installer = "uv"
        except WorkerError as exc:
            print(f"[studio-creation] uv pip install failed ({exc}); falling back to pip", flush=True)
    if not installed_via_uv:
        base = [
            str(venv_python), "-m", "pip", "install", "--disable-pip-version-check", "-q",
            "-c", str(constraints),
        ]
        _run([*base, "huggingface_hub", *req_args], cwd=comfy, timeout=timeout, env=build_env)

    for script in node_install_scripts:
        _run([str(venv_python), str(script)], cwd=script.parent, timeout=timeout, env=build_env)

    importlib.invalidate_caches()
    after_protected, _ = _protect_creation_environment(constraint_root)
    if before_protected and after_protected != before_protected:
        raise WorkerError("Python 依赖安装修改了受保护的 PyTorch/CUDA 基础包，已中止运行")
    vsite = venv_site_dir(target_venv)
    if vsite.is_dir():
        for entry in vsite.iterdir():
            if _is_protected_site_entry(entry.name, vsite):
                raise WorkerError(f"Python 依赖安装在域 venv 中覆写了受保护的 PyTorch/CUDA 包：{entry.name}")
    elapsed = round(time.monotonic() - t0, 2)
    print(f"[studio-creation] Python deps ready in venv via {installer} in {elapsed:.1f}s", flush=True)
    return {"installer": installer, "python_seconds": elapsed, "venv_dir": str(target_venv)}


def _env_tar_filter(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
    parts = Path(tarinfo.name).parts
    if "__pycache__" in parts or tarinfo.name.endswith((".pyc", ".pyo")):
        return None
    return tarinfo


def _dist_record_top_names(dist_entry: Path) -> set[str]:
    names: set[str] = set()
    record = dist_entry / "RECORD"
    if record.is_file():
        try:
            with record.open("r", encoding="utf-8", errors="replace", newline="") as stream:
                for row in csv.reader(stream):
                    if not row or not row[0].strip():
                        continue
                    raw = row[0].strip().replace("\\", "/")
                    if raw.startswith("/") or raw.startswith("../"):
                        continue
                    top = raw.split("/", 1)[0]
                    if top and top not in {".", "..", "__pycache__"}:
                        names.add(top)
        except (OSError, csv.Error):
            pass
    return names


def _site_dist_state(site: Path) -> dict[str, Any]:
    if not site.is_dir():
        return {"entries": set(), "dists": {}}
    entries: set[str] = set()
    dists: dict[str, str] = {}
    for entry in site.iterdir():
        if entry.name == "__pycache__" or entry.name.startswith(".studio-creation-"):
            continue
        entries.add(entry.name)
        if entry.name.endswith((".dist-info", ".egg-info")):
            meta = entry / "METADATA"
            try:
                fp = _sha256(meta) if meta.is_file() else str(entry.stat().st_mtime_ns)
            except OSError:
                fp = ""
            dists[entry.name] = fp
    return {"entries": entries, "dists": dists}


def _pack_env(comfy: Path, site: Path, before: dict[str, Any], key: str, out: Path) -> dict:
    """Tar ComfyUI plus semantic site-packages delta (excluding __pycache__ and protected PyTorch/CUDA dirs)."""
    if isinstance(before, dict) and "entries" in before and "dists" in before:
        after = _site_dist_state(site)
        before_entries: set[str] = before["entries"]
        before_dists: dict[str, str] = before["dists"]
        added_set = set(after["entries"] - before_entries)
        removed = sorted(name for name in (before_entries - after["entries"]) if not _is_protected_site_entry(name, site))
        for dist_name, fp in after["dists"].items():
            if before_dists.get(dist_name) != fp and not _is_protected_site_entry(dist_name, site):
                added_set.add(dist_name)
                added_set.update(_dist_record_top_names(site / dist_name))
        added = sorted(
            name for name in added_set
            if (site / name).exists() and not _is_protected_site_entry(name, site)
        )
    else:
        after_legacy = _site_state(site)
        added = sorted(
            name for name, mtime in after_legacy.items()
            if before.get(name) != mtime and not _is_protected_site_entry(name, site)
        )
        removed = sorted(
            name for name in before
            if name not in after_legacy and not _is_protected_site_entry(name, site)
        )

    def _write_members(tar: tarfile.TarFile) -> None:
        tar.add(comfy, arcname="ComfyUI", filter=_env_tar_filter)
        for name in added:
            tar.add(site / name, arcname=f"site/{name}", filter=_env_tar_filter)
        meta = json.dumps({"schema": 2, "key": key, "removed": removed}).encode()
        info = tarfile.TarInfo("env.json")
        info.size = len(meta)
        tar.addfile(info, io.BytesIO(meta))

    if shutil.which("pigz"):
        with out.open("wb") as out_stream:
            pigz = subprocess.Popen(["pigz", "-1", "-c"], stdin=subprocess.PIPE, stdout=out_stream, stderr=subprocess.PIPE)
            try:
                assert pigz.stdin is not None
                with tarfile.open(fileobj=pigz.stdin, mode="w|") as tar:
                    _write_members(tar)
                pigz.stdin.close()
                pigz.stdin = None
                _, err = pigz.communicate(timeout=600)
                if pigz.returncode != 0:
                    raise WorkerError("pigz pack failed: " + err.decode(errors="replace")[-200:])
            finally:
                if pigz.poll() is None:
                    pigz.kill()
                    pigz.wait(timeout=5)
    else:
        with tarfile.open(out, "w:gz", compresslevel=1) as tar:
            _write_members(tar)
    return {"site_entries_added": len(added), "site_entries_removed": len(removed)}


def _unpack_env(archive: Path, root: Path, site: Path) -> None:
    same_fs = site.is_dir() and site.parent.is_dir()
    stage = (site.parent / f".studio-creation-env-stage-{os.getpid()}") if same_fs else (root / ".env-stage")
    rollback = (site.parent / f".studio-creation-env-rollback-{os.getpid()}") if same_fs else (root / ".env-rollback")
    shutil.rmtree(stage, ignore_errors=True)
    shutil.rmtree(rollback, ignore_errors=True)
    stage.mkdir(mode=0o700, parents=True, exist_ok=True)
    before_protected, _ = _protect_creation_environment(root)
    moved: list[str] = []
    installed: list[str] = []
    try:
        if shutil.which("pigz"):
            proc = subprocess.Popen(["pigz", "-dc", str(archive)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                assert proc.stdout is not None
                with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
                    tar.extractall(stage, filter="data")
                _, err = proc.communicate(timeout=300)
                if proc.returncode != 0:
                    raise WorkerError("pigz unpack failed: " + err.decode(errors="replace")[-200:])
            except BaseException:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait(timeout=5)
                raise
        else:
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(stage, filter="data")
        removed = [
            x for x in json.loads((stage / "env.json").read_text())["removed"]
            if isinstance(x, str) and "/" not in x and x not in {"", ".", ".."}
        ]
        staged_site = stage / "site"
        incoming = sorted(e.name for e in staged_site.iterdir()) if staged_site.is_dir() else []
        for name in set(removed) | set(incoming):
            if _is_protected_site_entry(name, site):
                raise WorkerError(f"env cache attempted to modify protected package: {name}")
        rollback.mkdir(mode=0o700, parents=True, exist_ok=True)
        for name in sorted(set(removed) | set(incoming)):
            target = site / name
            if target.exists() or target.is_symlink():
                shutil.move(str(target), str(rollback / name))
                moved.append(name)
        for name in incoming:
            shutil.move(str(staged_site / name), str(site / name))
            installed.append(name)
        importlib.invalidate_caches()
        after_protected, _ = _protect_creation_environment(root)
        if before_protected and after_protected != before_protected:
            raise WorkerError("env cache altered protected PyTorch/CUDA packages")
        shutil.move(str(stage / "ComfyUI"), str(root / "ComfyUI"))
    except Exception:
        for name in reversed(installed):
            _remove(site / name)
        for name in reversed(moved):
            src = rollback / name
            if src.exists() or src.is_symlink():
                _remove(site / name)
                shutil.move(str(src), str(site / name))
        importlib.invalidate_caches()
        raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)
        shutil.rmtree(rollback, ignore_errors=True)


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
    """ComfyUI + Python deps inside a domain venv; fixed v3 Drive env-cache when explicitly needed."""
    comfy = root / "ComfyUI"
    venv_dir = root / "venv"
    ensure_domain_venv(venv_dir, timeout=min(120.0, timeout))
    site = venv_site_dir(venv_dir)
    started = time.time()
    record: dict[str, Any] = {}
    key = target = None
    cache_mode = _env_cache_mode()
    # By default ("auto"), skip Drive env tarballs when no custom nodes are present because
    # uv installs base ComfyUI deps in ~5-8s in parallel with weights; only use Drive env-cache
    # when explicitly forced or when custom nodes may carry heavy/compiled pip dependencies
    # on a real GPU runtime (CPU VMs never produce GPU-compatible env keys).
    try_env_cache = (
        cache is not None
        and cache_mode != "off"
        and (cache_mode == "force" or bool(custom_nodes))
        and _has_nvidia_gpu()
    )
    if try_env_cache:
        try:
            key = "v3-" + _env_key(commit, timeout, custom_nodes)
            target = f"env/{key}.tar.gz"
            record["key"] = key
            entry = cache.lookup(target, None, None)
            if entry:
                archive = root / "env.tar.gz"
                sha_actual, _ = cache.fetch(entry, archive, expected_sha256=entry["appProperties"]["sha256"])
                if sha_actual != entry["appProperties"]["sha256"]:
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
    before = _site_dist_state(site) if (build and try_env_cache) else None
    t_co = time.monotonic()
    _checkout_comfy(comfy, commit, timeout)
    _verify_checkout(comfy, commit)
    record["checkout_seconds"] = round(time.monotonic() - t_co, 2)
    t_nodes = time.monotonic()
    _install_custom_nodes(comfy, custom_nodes or [], timeout)
    record["nodes_seconds"] = round(time.monotonic() - t_nodes, 2)
    pip_info = _pip_install(comfy, timeout, root=root, custom_nodes=custom_nodes, venv_dir=venv_dir)
    record.update(pip_info)
    record.update(source="install", seconds=round(time.time() - started, 1))
    should_build_cache = (
        build
        and try_env_cache
        and cache is not None
        and key
        and before is not None
        and (cache_mode == "force" or _nodes_need_pip(comfy, custom_nodes or []))
    )
    if should_build_cache:
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


class _RangeUnsupported(RuntimeError):
    """Raised when an upstream response ignores Range (HTTP 200) so fetch falls back cleanly."""


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
        self._lock = threading.Lock()
        self._http_client: Any = None
        self._min_multistream_bytes = _MULTI_STREAM_MIN_BYTES

    def _is_open_overridden(self) -> bool:
        return type(self)._open is not DriveCache._open or "_open" in self.__dict__

    def _get_http_client(self):
        if self._is_open_overridden():
            return None
        with self._lock:
            if self._http_client is not None:
                return self._http_client
            try:
                import httpx
            except ImportError:
                return None
            self._http_client = httpx.Client(
                timeout=httpx.Timeout(connect=10.0, read=60.0, write=30.0, pool=60.0),
                limits=httpx.Limits(max_connections=16, max_keepalive_connections=16),
                follow_redirects=False,
                trust_env=True,
            )
            return self._http_client

    @staticmethod
    def _check_google_url(url: str) -> None:
        parsed = urlsplit(url)
        host = parsed.hostname or ""
        if parsed.scheme != "https" or not (
            host == "www.googleapis.com" or host.endswith(".googleusercontent.com")
        ):
            raise WorkerError("drive cache refuses a non-Google URL")

    @staticmethod
    def _retry_sleep(attempt: int) -> None:
        if os.getenv("STUDIO_OFFLINE_TEST") == "1":
            return
        time.sleep(min(8.0, (0.5 * (2**attempt)) + random.uniform(0.0, 0.4)))

    def _open(
        self, method: str, url: str, *, body: bytes | None = None, headers: dict | None = None, timeout=120
    ):
        merged = {"Authorization": "Bearer " + self._token}
        merged.update(headers or {})
        for _ in range(4):
            self._check_google_url(url)
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
        with self._lock:
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
        with self._lock:
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

    def _stream_range_segment(self, url: str, fd: int, start: int, end: int, total_size: int) -> None:
        """Stream one large contiguous Range segment directly via os.pwrite, resuming on drop."""
        pos = start
        client = self._get_http_client()
        for attempt in range(4):
            if pos > end:
                return
            range_header = f"bytes={pos}-{end}"
            try:
                if client is not None:
                    cur = url
                    headers = {"Authorization": "Bearer " + self._token, "Range": range_header}
                    # Resolve Google 302 redirect hop(s) once before opening the continuous stream.
                    for hop in range(4):
                        self._check_google_url(cur)
                        with client.stream("GET", cur, headers=headers) as resp:
                            status = resp.status_code
                            if status in {301, 302, 303, 307} and resp.headers.get("location"):
                                cur = resp.headers["location"]
                                if hop == 3:
                                    raise WorkerError("drive cache redirected too many times")
                                continue
                            if status == 200:
                                raise _RangeUnsupported("upstream ignored Range header")
                            if status in _RETRYABLE_HTTP_CODES:
                                raise HTTPError(cur, status, f"HTTP {status}", resp.headers, None)
                            if status != 206:
                                raise WorkerError(f"drive range request returned HTTP {status}")
                            cr = (resp.headers.get("content-range") or "").strip()
                            match = _CONTENT_RANGE_RE.fullmatch(cr)
                            if (
                                not match
                                or int(match[1]) != pos
                                or int(match[2]) != end
                                or int(match[3]) != total_size
                            ):
                                raise _RangeUnsupported(f"unexpected Content-Range: {cr}")
                            for chunk in resp.iter_bytes(chunk_size=_MULTI_STREAM_STREAM_CHUNK):
                                if not chunk:
                                    continue
                                if pos + len(chunk) > end + 1:
                                    raise WorkerError("drive range segment exceeded expected end")
                                written = os.pwrite(fd, chunk, pos)
                                while written < len(chunk):
                                    written += os.pwrite(fd, chunk[written:], pos + written)
                                pos += len(chunk)
                            break
                else:
                    with self._open("GET", url, headers={"Range": range_header}, timeout=120) as response:
                        status = getattr(response, "status", None)
                        if status is None and hasattr(response, "getcode"):
                            status = response.getcode()
                        if status == 200 or status is None:
                            raise _RangeUnsupported("upstream ignored Range header")
                        if status != 206:
                            raise WorkerError(f"drive range request returned HTTP {status}")
                        resp_headers = getattr(response, "headers", None) or {}
                        cr = str(resp_headers.get("Content-Range") or resp_headers.get("content-range") or "").strip()
                        match = _CONTENT_RANGE_RE.fullmatch(cr)
                        if (
                            not match
                            or int(match[1]) != pos
                            or int(match[2]) != end
                            or int(match[3]) != total_size
                        ):
                            raise _RangeUnsupported(f"unexpected Content-Range: {cr}")
                        while chunk := response.read(_MULTI_STREAM_STREAM_CHUNK):
                            if pos + len(chunk) > end + 1:
                                raise WorkerError("drive range segment exceeded expected end")
                            written = os.pwrite(fd, chunk, pos)
                            while written < len(chunk):
                                written += os.pwrite(fd, chunk[written:], pos + written)
                            pos += len(chunk)
                if pos > end:
                    return
                if attempt == 3:
                    raise WorkerError(f"drive range segment incomplete ({pos} <= {end})")
                self._retry_sleep(attempt)
            except _RangeUnsupported:
                raise
            except HTTPError as exc:
                if exc.code in _RETRYABLE_HTTP_CODES and attempt < 3:
                    self._retry_sleep(attempt)
                    continue
                raise
            except WorkerError:
                raise
            except OSError as exc:
                if exc.errno == errno.ENOSPC or attempt == 3:
                    raise
                self._retry_sleep(attempt)
            except Exception:
                if attempt == 3:
                    raise
                self._retry_sleep(attempt)
        raise WorkerError("drive range segment failed after retries")

    def _fetch_multistream(
        self, entry: dict, destination: Path, *, expected_sha256: str | None, streams: int
    ) -> tuple[str, str]:
        size = int(entry["size"])
        url = f"{DRIVE_API}/files/{quote(entry['id'])}?alt=media"
        effective_streams = max(2, min(streams, 8))
        seg_size = math.ceil(size / effective_streams)
        segments = [
            (idx * seg_size, min(size, (idx + 1) * seg_size) - 1)
            for idx in range(effective_streams)
            if idx * seg_size < size
        ]
        temporary = None
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".part", delete=False) as out:
                temporary = Path(out.name)
                os.ftruncate(out.fileno(), size)
                fd = out.fileno()

                def run_seg(bounds: tuple[int, int]) -> None:
                    with _DRIVE_STREAM_SEM:
                        self._stream_range_segment(url, fd, bounds[0], bounds[1], size)

                with ThreadPoolExecutor(max_workers=len(segments)) as pool:
                    futures = [pool.submit(run_seg, bounds) for bounds in segments]
                    for fut in futures:
                        fut.result()
            if temporary.stat().st_size != size:
                raise WorkerError("drive cached file failed size/md5 check")
            actual_sha, actual_md5 = _hash_file(temporary)
            if actual_md5 != entry.get("md5Checksum"):
                raise WorkerError("drive cached file failed size/md5 check")
            if expected_sha256 and actual_sha != expected_sha256.lower():
                raise WorkerError("Drive weight sha256 or size mismatch")
            os.replace(temporary, destination)
            temporary = None
            _record_local_hash(destination, sha256=actual_sha, md5=actual_md5)
            return actual_sha, actual_md5
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _fetch_single(self, entry: dict, destination: Path, *, expected_sha256: str | None) -> tuple[str, str]:
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        size = int(entry["size"])
        url = f"{DRIVE_API}/files/{quote(entry['id'])}?alt=media"
        for attempt in range(3):
            sha = hashlib.sha256()
            md5 = hashlib.md5(usedforsecurity=False)
            written = 0
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
                            sha.update(block)
                            md5.update(block)
                actual_sha = sha.hexdigest()
                actual_md5 = md5.hexdigest()
                if written != size or actual_md5 != entry.get("md5Checksum"):
                    raise WorkerError("drive cached file failed size/md5 check")
                if expected_sha256 and actual_sha != expected_sha256.lower():
                    raise WorkerError("Drive weight sha256 or size mismatch")
                os.replace(temporary, destination)
                temporary = None
                _record_local_hash(destination, sha256=actual_sha, md5=actual_md5)
                return actual_sha, actual_md5
            except HTTPError as exc:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
                    temporary = None
                if exc.code in _RETRYABLE_HTTP_CODES and attempt < 2:
                    self._retry_sleep(attempt)
                    continue
                raise
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        raise WorkerError("drive cached file download failed")

    def fetch(
        self,
        entry: dict,
        destination: Path,
        *,
        expected_sha256: str | None = None,
        label: str | None = None,
    ) -> tuple[str, str]:
        started = time.monotonic()
        size = int(entry["size"])
        try:
            streams = max(1, min(int(os.getenv("CREATION_DRIVE_STREAMS", "4")), 8))
        except ValueError:
            streams = 4
        use_multi = (
            _creation_flag("CREATION_MULTI_STREAM_DRIVE", False)
            and streams > 1
            and size >= self._min_multistream_bytes
        )
        used_streams = 1
        if use_multi:
            try:
                hashes = self._fetch_multistream(
                    entry, destination, expected_sha256=expected_sha256, streams=streams
                )
                used_streams = streams
            except _RangeUnsupported:
                hashes = self._fetch_single(entry, destination, expected_sha256=expected_sha256)
            except OSError as exc:
                if exc.errno == errno.ENOSPC:
                    raise
                print(
                    f"[studio-creation] multi-stream drive fetch fallback to single stream ({type(exc).__name__})",
                    flush=True,
                )
                hashes = self._fetch_single(entry, destination, expected_sha256=expected_sha256)
            except WorkerError as exc:
                if "sha256" in str(exc).lower() or "md5" in str(exc).lower():
                    raise
                print(
                    f"[studio-creation] multi-stream drive fetch fallback to single stream ({exc})",
                    flush=True,
                )
                hashes = self._fetch_single(entry, destination, expected_sha256=expected_sha256)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[studio-creation] multi-stream drive fetch fallback to single stream ({type(exc).__name__})",
                    flush=True,
                )
                hashes = self._fetch_single(entry, destination, expected_sha256=expected_sha256)
        else:
            hashes = self._fetch_single(entry, destination, expected_sha256=expected_sha256)
        elapsed = max(0.001, time.monotonic() - started)
        if size >= 1024 * 1024:
            mib = size / (1024**2)
            print(
                f"[studio-creation] drive fetch {label or destination.name}: {mib:.1f} MiB in {elapsed:.1f}s "
                f"({mib / elapsed:.1f} MiB/s, streams={used_streams})",
                flush=True,
            )
        return hashes

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
        if isinstance(node, dict) and node.get("class_type") in {"LoraLoader", "LoraLoaderModelOnly"}:
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
        def _download_one(pair: tuple[Path, dict]) -> dict:
            target, item = pair
            started = time.time()
            try:
                source = Path(scratch) / f"{uuid.uuid4().hex}-{target.name}"
                sha, md5 = cache.fetch(item, source, label="loras/" + item["target"])
                _publish_model(source, target, preserve_existing=preserve_existing)
                _record_local_hash(target, sha256=sha, md5=md5)
                return {
                    "target": "loras/" + item["target"],
                    "source": "drive",
                    "seconds": round(time.time() - started, 1),
                }
            except Exception as exc:
                raise WorkerError(f"用户 LoRA 恢复失败：{_model_download_error(exc)}") from exc

        pairs = list(pending.items())
        if len(pairs) > 1:
            with ThreadPoolExecutor(max_workers=min(4, len(pairs))) as pool:
                records.extend(pool.map(_download_one, pairs))
        else:
            records.append(_download_one(pairs[0]))
    return records


def _restore_weights(
    comfy: Path, manifest: dict, timeout: float, cache: DriveCache | None, prep: bool,
    *, preserve_existing: bool = False, warm_models: Path | None = None, use_drive_cache: bool = True,
    paths: dict[str, Path] | None = None, env_future: Any = None,
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
                    _record_local_hash(target, sha256=item["sha256"])
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
            fetched = cache.fetch(entry, source, expected_sha256=item["sha256"], label=item["target"])
            if isinstance(fetched, tuple) and len(fetched) == 2:
                actual_sha, actual_md5 = fetched
                if actual_sha != item["sha256"]:
                    raise WorkerError("Drive weight sha256 or size mismatch")
                if item["size"] is not None and source.stat().st_size != item["size"]:
                    raise WorkerError("Drive weight sha256 or size mismatch")
            else:
                if not _local_model_hit(source, item["size"], item["sha256"]):
                    raise WorkerError("Drive weight sha256 or size mismatch")
                actual_sha, actual_md5 = item["sha256"], None
            if not prep:
                _publish_model(source, target, preserve_existing=preserve_existing)
                _record_local_hash(target, sha256=actual_sha, md5=actual_md5)

        try:
            if item.get("source") == "drive":
                entry = drive_entry(item)
                if not prep:
                    fetch(entry, scratch / "drive")
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
                    try:
                        from huggingface_hub import hf_hub_download
                    except ImportError:
                        if env_future is not None:
                            env_future.result()
                            importlib.invalidate_caches()
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
                        _record_local_hash(target, sha256=sha256, md5=md5)
        except Exception as exc:
            raise WorkerError(f"模型 {item['target']} 恢复失败：{_model_download_error(exc)}") from exc
        record["seconds"] = round(time.time() - started, 1)
        return record

    # Schedule largest weights (e.g. UNet/DiT/CLIP) first so they start at t=0
    # and are never tail-delayed behind small LoRAs in the 4-worker pool.
    items = sorted(by_target.values(), key=lambda x: -(int(x.get("size") or 0)))
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


_ENV_STATE_FILE = "env-state.json"
_MODEL_LRU_FILE = ".model-lru.json"


def _code_env_fingerprint(commit: str, custom_nodes: list[dict], manifest: dict | None = None) -> str:
    """Fingerprint only ComfyUI code + Python/custom-node env, decoupled from weights & user LoRAs."""
    manifest = manifest or {}
    declared = manifest.get("dependency_signature", manifest.get("dependencies_signature"))
    if declared is None:
        raw_deps = manifest.get("dependencies")
        if isinstance(raw_deps, dict):
            declared = {k: v for k, v in raw_deps.items() if k not in {"weight_files", "weights"}}
        else:
            declared = manifest.get("requirements_sha256")
    body = {
        "version": 1,
        "comfy_commit": commit,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "custom_nodes": sorted(f"{n.get('repo')}@{n.get('commit')}" for n in custom_nodes),
        "dependencies": declared,
    }
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _read_env_state(service_dir: Path) -> dict | None:
    try:
        value = json.loads((service_dir / _ENV_STATE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _write_env_state(service_dir: Path, code_fp: str, commit: str) -> None:
    _atomic_json(
        service_dir / _ENV_STATE_FILE,
        {
            "code_env_fingerprint": code_fp,
            "commit": commit,
            "venv_isolated": venv_python_path(service_dir / "venv").is_file(),
            "updated_at": time.time(),
        },
    )


def _prune_user_loras(comfy: Path, user_loras: list[dict]) -> None:
    """Strictly keep only user LoRAs admitted for the current job."""
    selected = {Path(item["target"]).name for item in user_loras}
    loras = comfy / "models" / "loras" / "user"
    if loras.is_dir():
        for entry in loras.iterdir():
            if entry.name not in selected:
                _remove(entry)


def _read_model_lru(service_dir: Path) -> dict[str, float]:
    try:
        raw = json.loads((service_dir / _MODEL_LRU_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(k): float(v) for k, v in raw.items() if isinstance(k, str) and isinstance(v, (int, float))}


def _touch_model_lru(service_dir: Path, targets: list[str]) -> None:
    if not targets:
        return
    lru = _read_model_lru(service_dir)
    now = time.time()
    for target in targets:
        lru[str(target)] = now
    with contextlib.suppress(OSError):
        _atomic_json(service_dir / _MODEL_LRU_FILE, lru)


def _evict_unused_weights_if_needed(
    service_dir: Path,
    models: Path,
    keep_targets: set[str],
    needed_bytes: int,
    *,
    reserve_bytes: int = _LRU_RESERVE_BYTES,
) -> list[str]:
    """Keep cached base weights on disk while free space suffices; evict oldest unused ones if tight."""
    if not models.is_dir() or needed_bytes <= 0:
        return []
    free = shutil.disk_usage(models).free
    if free >= needed_bytes + reserve_bytes:
        return []
    lru = _read_model_lru(service_dir)
    candidates: list[tuple[float, int, str, Path]] = []
    for parent, _dirs, files in os.walk(models):
        for name in files:
            path = Path(parent) / name
            if path.is_symlink():
                continue
            rel = str(path.relative_to(models)).replace("\\", "/")
            if rel in keep_targets or rel.startswith("loras/user/"):
                continue
            try:
                st = path.stat()
            except OSError:
                continue
            last_used = lru.get(rel, st.st_mtime)
            candidates.append((last_used, st.st_size, rel, path))
    candidates.sort(key=lambda item: item[0])
    evicted: list[str] = []
    for _ts, file_size, rel, path in candidates:
        if free >= needed_bytes + reserve_bytes:
            break
        try:
            path.unlink()
            free += file_size
            lru.pop(rel, None)
            evicted.append(rel)
        except OSError:
            continue
    if evicted:
        with contextlib.suppress(OSError):
            _atomic_json(service_dir / _MODEL_LRU_FILE, lru)
        print(f"[studio-creation] Evicted {len(evicted)} LRU cached weight(s) to free disk space", flush=True)
    return evicted


def _can_reuse_in_place_env(service_dir: Path, code_fp: str, commit: str, nodes: list[dict]) -> bool:
    env_state = _read_env_state(service_dir)
    if not isinstance(env_state, dict) or env_state.get("code_env_fingerprint") != code_fp:
        return False
    if env_state.get("venv_isolated") is True and not venv_python_path(service_dir / "venv").is_file():
        return False
    comfy = service_dir / "ComfyUI"
    if not (comfy / "main.py").is_file():
        return False
    try:
        _verify_checkout(comfy, commit)
        _verify_custom_nodes(comfy, nodes)
    except WorkerError:
        return False
    return True


def _creation_service(service_dir, payload, manifest, commit, nodes, checked, user_loras, stages, deadline):
    def environment_fingerprint():
        environment = _service_fingerprint(commit, nodes, checked, manifest)
        encoded = json.dumps([environment, user_loras], sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    fingerprint = environment_fingerprint()
    code_fp = _code_env_fingerprint(commit, nodes, manifest)
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
            _touch_model_lru(service_dir, [w["target"] for w in checked])
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
    timeout = min(float(payload.get("download_timeout_seconds", 600)), 600)
    use_drive_cache = payload.get("use_drive_cache", True)
    service_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    in_place_reuse = _can_reuse_in_place_env(service_dir, code_fp, commit, nodes)

    if in_place_reuse:
        # Same ComfyUI commit & custom nodes (e.g. switched user LoRA or another workflow on same commit):
        # keep ComfyUI/ tree and Python packages intact, strictly prune unselected user LoRAs, and LRU-cache base weights.
        for leftover in ("input", "output", "temp"):
            shutil.rmtree(comfy / leftover, ignore_errors=True)
        _prune_user_loras(comfy, user_loras)
        models_dir = _local_models(comfy)
        keep_targets = {w["target"] for w in checked}
        missing_bytes = sum(
            int(w.get("size") or 0)
            for w in checked
            if not (models_dir / w["target"]).is_file()
        ) + sum(
            int(u.get("size") or 0)
            for u in user_loras
            if not (models_dir / "loras" / u["target"]).is_file()
        )
        _evict_unused_weights_if_needed(service_dir, models_dir, keep_targets, missing_bytes)
        env = {"source": "in_place_reuse", "seconds": 0.0}
        stages.start("restore_weights")
        if user_loras and _creation_flag("CREATION_PARALLEL_BOOTSTRAP", True):
            with ThreadPoolExecutor(max_workers=2) as pool:
                weights_future = pool.submit(
                    _restore_weights,
                    comfy, manifest, timeout, _drive_cache(payload), False,
                    use_drive_cache=use_drive_cache,
                )
                loras_future = pool.submit(
                    _restore_user_loras,
                    comfy, user_loras, _lora_drive_cache(payload),
                )
                weights = weights_future.result()
                loras_future.result()
        else:
            weights = _restore_weights(comfy, manifest, timeout, _drive_cache(payload), prep=False,
                                       use_drive_cache=use_drive_cache)
            if user_loras:
                stages.start("restore_loras")
                _restore_user_loras(comfy, user_loras, _lora_drive_cache(payload))
        _touch_model_lru(service_dir, [w["target"] for w in checked])
        stages.start("comfy_start")
        service = _start_comfy_service(service_dir, environment_fingerprint(), "in_place_restart", deadline=deadline)
        return service, env, weights

    parked = service_dir / "previous"
    if comfy.exists():
        shutil.rmtree(parked, ignore_errors=True)
        comfy.rename(parked)
    warm_models = _local_models(parked) if parked.is_dir() else None
    env_cache = _drive_cache(payload, "runtime_folder_id") if use_drive_cache else None
    parallel_boot = (
        _creation_flag("CREATION_PARALLEL_BOOTSTRAP", True)
        and _env_cache_mode() != "force"
        and _restore_env.__name__ == "_restore_env"
    )

    if parallel_boot:
        stages.start("restore_env")
        t_env0 = time.monotonic()
        _checkout_comfy(comfy, commit, timeout)
        _verify_checkout(comfy, commit)
        checkout_s = round(time.monotonic() - t_env0, 2)
        t_nodes = time.monotonic()
        _install_custom_nodes(comfy, nodes, timeout)
        nodes_s = round(time.monotonic() - t_nodes, 2)

        def _finish_python() -> dict[str, Any]:
            pip_info = _pip_install(comfy, timeout, root=service_dir, custom_nodes=nodes)
            return {
                **pip_info,
                "checkout_seconds": checkout_s,
                "nodes_seconds": nodes_s,
                "source": "install",
                "seconds": round(time.monotonic() - t_env0, 1),
            }

        with ThreadPoolExecutor(max_workers=3) as boot_pool:
            env_future = boot_pool.submit(_finish_python)
            stages.start("restore_weights")
            loras_future = (
                boot_pool.submit(_restore_user_loras, comfy, user_loras, _lora_drive_cache(payload))
                if user_loras
                else None
            )
            weights = _restore_weights(
                comfy, manifest, timeout, _drive_cache(payload), prep=False,
                warm_models=warm_models, use_drive_cache=use_drive_cache, env_future=env_future,
            )
            if loras_future is not None:
                if not loras_future.done():
                    stages.start("restore_loras")
                loras_future.result()
            if not env_future.done():
                stages.start("creation_python")
            env = env_future.result()
    else:
        stages.start("restore_env")
        env = _restore_env(service_dir, commit, timeout, env_cache, build=False, custom_nodes=nodes)
        stages.start("restore_weights")
        weights = _restore_weights(comfy, manifest, timeout, _drive_cache(payload), prep=False,
                                   warm_models=warm_models, use_drive_cache=use_drive_cache)
        if user_loras:
            stages.start("restore_loras")
            _restore_user_loras(comfy, user_loras, _lora_drive_cache(payload))

    _write_env_state(service_dir, code_fp, commit)
    _touch_model_lru(service_dir, [w["target"] for w in checked])
    shutil.rmtree(parked, ignore_errors=True)
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


def _clean_ephemeral_history(root: Path) -> None:
    """Remove leftover .warm and sibling ephemeral execution directories before a clean smoke run."""
    parent = root.parent
    if not parent.is_dir():
        return
    shutil.rmtree(parent / ".warm", ignore_errors=True)
    for entry in list(parent.iterdir()):
        if entry.name in {root.name, ".vm-baseline.json", ".vm-baseline.lock", "native", "comfy-services"}:
            continue
        try:
            uuid.UUID(entry.name)
        except ValueError:
            continue
        _remove(entry)


def _terminate_process_group(process: subprocess.Popen | None, *, grace_seconds: float = 10.0) -> bool:
    if process is None:
        return True
    if process.poll() is not None:
        return True
    pid = getattr(process, "pid", None)
    if isinstance(pid, int) and pid > 0:
        try:
            os.killpg(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            with contextlib.suppress(OSError):
                process.terminate()
    else:
        with contextlib.suppress(OSError):
            process.terminate()
    try:
        process.wait(timeout=grace_seconds)
        return True
    except subprocess.TimeoutExpired:
        if isinstance(pid, int) and pid > 0:
            try:
                os.killpg(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                with contextlib.suppress(OSError):
                    process.kill()
        else:
            with contextlib.suppress(OSError):
                process.kill()
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=5)
        return process.poll() is not None


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
    snapshot = payload.get("snapshot") or {}
    is_workflow_smoke = bool(snapshot.get("workflow_smoke")) and service_dir is None
    studio_root = _default_studio_root(root)
    expected_baseline = payload.get("expected_baseline_digest")
    if is_workflow_smoke:
        _clean_ephemeral_history(root)
        if isinstance(expected_baseline, str) and SHA256.fullmatch(expected_baseline):
            actual_digest, _ = global_site_digest()
            if actual_digest != expected_baseline or _user_site_packages_dirty():
                raise WorkerError("冒烟前全局 Python 环境基线校验未通过，已中止运行")
    comfy = (service_dir or root) / "ComfyUI"
    user_loras = snapshot.get("lora_files") or []
    _check_loras(graph, manifest, user_loras)
    timeout = min(float(payload.get("download_timeout_seconds", 600)), 600)
    nodes = _reviewed_custom_nodes(manifest.get("custom_nodes"))
    checked = _validate_weights(manifest)
    fingerprint = _warm_fingerprint(commit, nodes, checked)
    runtime_seconds = float(payload.get("max_runtime_minutes", 10)) * 60
    if not payload.get("generation_timeouts"):
        runtime_seconds = min(runtime_seconds, 600 * 60)
    deadline = min(time.time() + runtime_seconds, float(payload.get("deadline_epoch") or float("inf")))
    warmup = bool(snapshot.get("creation_warmup"))
    service = None
    process = None
    if service_dir:
        service, env, weights = _creation_service(
            service_dir, payload, manifest, commit, nodes, checked, user_loras, stages, deadline)
    elif not is_workflow_smoke and _warm_take(root, fingerprint, nodes, checked, user_loras=user_loras):
        # Same runtime, same environment: the previous job left it restored. Nothing to download.
        stages.start("warm_reuse")
        env = {"source": "warm"}
        weights = [{"target": w["target"], "sha256": w["sha256"], "source": "warm", "seconds": 0.0} for w in checked]
    else:
        stages.start("restore_env")
        env_cache = _drive_cache(payload, "runtime_folder_id") if payload.get("use_drive_cache", True) else None
        env = _restore_env(root, commit, timeout, env_cache, build=False, custom_nodes=nodes)
        stages.start("restore_weights")
        warm_models = None
        if not is_workflow_smoke:
            parked = root.parent / ".warm" / "ComfyUI"
            warm_models = _local_models(parked) if parked.is_dir() else None
        weights = _restore_weights(comfy, manifest, timeout, _drive_cache(payload), prep=False,
                                   warm_models=warm_models, use_drive_cache=payload.get("use_drive_cache", True))
    try:
        if user_loras and not service:
            stages.start("restore_loras")
            _restore_user_loras(comfy, user_loras, _lora_drive_cache(payload))
        _local_models(comfy)
        _check_local_tree(comfy / "output")  # ComfyUI also registers model folders under its output directory.
        _write_input_files(comfy, snapshot)
        if not service:
            stages.start("comfy_start")
        log_path = root / "comfy.log"
        log = log_path.open("ab", buffering=0)
        try:
            port = COMFY_SERVICE_PORT
            if not service:
                with socket.socket() as listener:
                    listener.bind((COMFY_SERVICE_HOST, 0))
                    port = listener.getsockname()[1]
                run_venv = root / "venv"
                run_py = venv_python_path(run_venv)
                if not run_py.is_file() and is_workflow_smoke:
                    run_py = ensure_domain_venv(run_venv, timeout=min(120.0, timeout))
                comfy_python = str(run_py) if run_py.is_file() else sys.executable
                comfy_env = domain_python_env(run_venv if run_py.is_file() else None)
                process = subprocess.Popen(
                    [comfy_python, "main.py", "--listen", COMFY_SERVICE_HOST, "--port", str(port)],
                    cwd=comfy,
                    env=comfy_env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    shell=False,
                    start_new_session=True,
                )
            else:
                process = None
            if not service and not payload.get("generation_timeouts"):
                deadline = time.time() + runtime_seconds
            api = f"http://{COMFY_SERVICE_HOST}:{port}"
            fast_poll = _creation_flag("CREATION_FAST_CONTROL_PLANE", True)
            ready_poll_sleep = 0.3 if fast_poll else 1.0
            history_poll_sleep = 0.5 if fast_poll else 2.0
            while time.time() < deadline:
                try:
                    _http_json(api + "/system_stats", timeout=5)
                    break
                except WorkerError:
                    if process is not None and process.poll() is not None:
                        raise WorkerError("Comfy worker exited during startup")
                    time.sleep(ready_poll_sleep)
            else:
                raise WorkerError("Comfy worker startup timed out")
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
            queued = _http_json(api + "/prompt", "POST", {"prompt": graph}, timeout=30)
            prompt_id = queued.get("prompt_id")
            if not isinstance(prompt_id, str) or not prompt_id:
                raise WorkerError("Comfy queue did not return a prompt id")
            while time.time() < deadline:
                history = _http_json(api + "/history/" + prompt_id, timeout=30)
                item = history.get(prompt_id)
                if not isinstance(item, dict):
                    time.sleep(history_poll_sleep)
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
                    time.sleep(history_poll_sleep)
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
            _terminate_process_group(process, grace_seconds=10.0)
            log.close()
    finally:
        if is_workflow_smoke:
            proc_dead = _terminate_process_group(process, grace_seconds=5.0)
            shutil.rmtree(root / "ComfyUI", ignore_errors=True)
            shutil.rmtree(root / "venv", ignore_errors=True)
            shutil.rmtree(root.parent / ".warm", ignore_errors=True)
            clean_ok = False
            try:
                base_expected = expected_baseline
                if not (isinstance(base_expected, str) and SHA256.fullmatch(base_expected)):
                    base_expected = ensure_vm_baseline(studio_root=studio_root)["digest"]
                actual_digest, _ = global_site_digest()
                clean_ok = bool(
                    proc_dead
                    and not (root / "ComfyUI").exists()
                    and not (root / "venv").exists()
                    and actual_digest == base_expected
                    and not _user_site_packages_dirty()
                )
            except Exception:
                clean_ok = False
            _status(root, "running", clean_after_smoke=clean_ok)
        elif not service_dir:
            _warm_put(root, fingerprint)  # restore succeeded above, so the tree is reusable even if this job failed later


def _run_prep(root: Path, payload: dict, manifest: dict, stages: _Stages) -> list[dict]:
    """CPU-only cache fill: put every missing Hugging Face weight into Drive. No sampling."""
    commit = manifest.get("comfy_commit", manifest.get("comfy_version"))
    if not isinstance(commit, str) or not COMMIT.fullmatch(commit):
        raise WorkerError("Comfy workflow must lock a 40-character commit")
    if _drive_cache(payload) is None:
        raise WorkerError("prep needs the Drive cache")
    timeout = min(float(payload.get("download_timeout_seconds", 600)), 600)
    if _env_cache_mode() == "force" and _has_nvidia_gpu():
        stages.start("prep_env")
        env = _restore_env(root, commit, timeout, _drive_cache(payload, "runtime_folder_id"), build=True,
                           custom_nodes=_reviewed_custom_nodes(manifest.get("custom_nodes")))
    else:
        # CPU VMs cannot build GPU-compatible env caches and base deps use fast parallel uv on GPU boot.
        (root / "ComfyUI").mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            import huggingface_hub  # noqa: F401
        except ImportError:
            uv_bin = shutil.which("uv") if _creation_flag("CREATION_USE_UV", True) else None
            if uv_bin:
                _run([uv_bin, "pip", "install", "--system", "--python", sys.executable, "-q", "huggingface_hub"], timeout=timeout)
            else:
                _run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "-q", "huggingface_hub"], timeout=timeout)
        env = {"source": "uv_direct", "skipped_env_archive": True, "seconds": 0.0}
    stages.start("prep_weights")
    weights = _restore_weights(root / "ComfyUI", manifest, timeout, _drive_cache(payload), prep=True)
    stages.end()
    failed = [item["target"] for item in weights if item.get("cache_stored") is False]
    if not env.get("skipped_env_archive") and env.get("source") != "drive" and env.get("cache_stored") is not True:
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
    for flag_name, flag_val in (payload.get("creation_flags") or {}).items():
        if flag_name in CREATION_FLAG_NAMES:
            os.environ[flag_name] = str(flag_val)
    stages = _Stages(root)
    try:
        _status(root, "running")
        snapshot, manifest, graph = _snapshot(payload)
        if snapshot.get("prep") is not True:
            baseline = ensure_vm_baseline(studio_root=_default_studio_root(root))
            diagnostics = _probe_gpu()
            status_fields: dict[str, Any] = {
                "venv_isolated": True,
                "vm_baseline_digest": baseline["digest"],
            }
            if diagnostics:
                status_fields["gpu_diagnostics"] = diagnostics
            _status(root, "running", **status_fields)
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
            use_resident = bool(snapshot.get("creation_api") or payload.get("use_resident_service")) and bool(payload.get("instance_id"))
            if use_resident:
                service_dir = _comfy_service_dir(payload.get("instance_id"))
                with _file_lock(service_dir.parent / "creation-service.lock"):
                    outputs = _run_comfy(root, local_payload, manifest, graph, stages, service_dir=service_dir)
            else:
                outputs = _run_comfy(root, local_payload, manifest, graph, stages)
        runtime_path = root / "creation-runtime.json"
        use_resident_evidence = bool(snapshot.get("creation_api") or payload.get("use_resident_service")) and runtime_path.is_file()
        evidence = {"creation_runtime": json.loads(runtime_path.read_text())} if use_resident_evidence else {}
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
            if ((payload.get("snapshot") or {}).get("creation_api") or payload.get("use_resident_service")) and payload.get("instance_id"):
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


_MOUNTINFO_PATH = Path("/proc/self/mountinfo")
_PROC_ROOT = Path("/proc")


def _unescape_mount_path(raw: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), raw)


def _active_mounts_under(target_root: Path, *, mountinfo_path: Path | None = None) -> list[str]:
    """Return any mount points mounted at or beneath target_root from /proc/self/mountinfo."""
    mi = mountinfo_path or _MOUNTINFO_PATH
    resolved_root = Path(target_root).resolve()
    if not mi.is_file():
        return []
    found: list[str] = []
    try:
        lines = mi.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        raise WorkerError(f"无法读取挂载表：{exc}") from exc
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        mount_point = Path(_unescape_mount_path(parts[4]))
        try:
            mp_resolved = mount_point.resolve()
        except OSError:
            mp_resolved = mount_point
        if mp_resolved == resolved_root or resolved_root in mp_resolved.parents:
            found.append(str(mount_point))
    return sorted(set(found))


def _proc_pid_alive(pid: int, expected_start_id: str | None = None) -> bool:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False
    stat_path = _PROC_ROOT / str(pid) / "stat"
    try:
        raw = stat_path.read_text(encoding="utf-8")
        rest = raw.rsplit(")", 1)[1].split()
        if rest[0] == "Z":
            return False
        if expected_start_id and rest[19] != str(expected_start_id):
            return False
        return True
    except (OSError, IndexError, ValueError):
        return False


def _stop_pid_verified(
    pid: int | None,
    expected_start_id: str | None = None,
    *,
    grace_seconds: float = 10.0,
    kill_group: bool = True,
    allow_sigkill: bool = True,
) -> bool:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0 or pid == os.getpid():
        return True
    if not _proc_pid_alive(pid, expected_start_id):
        return True
    try:
        if kill_group:
            try:
                pgid = os.getpgid(pid)
                if pgid == os.getpgid(os.getpid()):
                    os.kill(pid, signal.SIGTERM)
                else:
                    os.killpg(pgid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                os.kill(pid, signal.SIGTERM)
        else:
            os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except OSError:
        return not _proc_pid_alive(pid, expected_start_id)
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if not _proc_pid_alive(pid, expected_start_id):
            return True
        time.sleep(0.2)
    if allow_sigkill and _proc_pid_alive(pid, expected_start_id):
        try:
            if kill_group:
                try:
                    pgid = os.getpgid(pid)
                    if pgid == os.getpgid(os.getpid()):
                        os.kill(pid, signal.SIGKILL)
                    else:
                        os.killpg(pgid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    os.kill(pid, signal.SIGKILL)
            else:
                os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            return True
        except OSError:
            pass
        kill_deadline = time.monotonic() + 3.0
        while time.monotonic() < kill_deadline:
            if not _proc_pid_alive(pid, expected_start_id):
                return True
            time.sleep(0.1)
    return not _proc_pid_alive(pid, expected_start_id)


def _my_ancestor_pids() -> set[int]:
    pids = {os.getpid(), os.getppid()}
    cur = os.getppid()
    for _ in range(16):
        if cur <= 1:
            break
        try:
            raw = (_PROC_ROOT / str(cur) / "stat").read_text(encoding="utf-8")
            ppid = int(raw.rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            break
        if ppid in pids or ppid <= 1:
            break
        pids.add(ppid)
        cur = ppid
    return pids


def _scan_matching_pids(predicate) -> list[tuple[int, str]]:
    """Scan /proc for live processes matching predicate(pid, cmdline, cwd)."""
    skip = _my_ancestor_pids()
    matches: list[tuple[int, str]] = []
    if not _PROC_ROOT.is_dir():
        return matches
    for entry in _PROC_ROOT.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in skip:
            continue
        try:
            raw_stat = (entry / "stat").read_text(encoding="utf-8")
            rest = raw_stat.rsplit(")", 1)[1].split()
            if rest[0] == "Z":
                continue
            start_id = rest[19]
            raw_cmd = (entry / "cmdline").read_bytes()
            cmdline = [item.decode("utf-8", "replace") for item in raw_cmd.split(b"\0") if item]
            try:
                cwd = os.readlink(entry / "cwd")
            except OSError:
                cwd = ""
            if predicate(pid, cmdline, cwd):
                matches.append((pid, start_id))
        except (OSError, IndexError, ValueError):
            continue
    return matches


def _native_open_writes(mount_dir: Path) -> bool:
    prefix = str(mount_dir).rstrip("/") + "/"
    ignored = {
        prefix + "native-comfy/user/comfyui.log",
        prefix + "native-comfy/user/comfyui_8188.log",
    }
    if not _PROC_ROOT.is_dir():
        return False
    for proc in _PROC_ROOT.glob("[0-9]*"):
        try:
            for fd in (proc / "fd").iterdir():
                try:
                    target = os.readlink(fd)
                    if not target.startswith(prefix) or target in ignored:
                        continue
                    info = (proc / "fdinfo" / fd.name).read_text(encoding="utf-8")
                    flags = next(line.split()[1] for line in info.splitlines() if line.startswith("flags:"))
                    if int(flags, 8) & 3:
                        return True
                except (OSError, StopIteration, ValueError):
                    continue
        except OSError:
            continue
    return False


def _local_rclone_unsynced_reason() -> str | None:
    """Return a non-empty reason if local rclone RC (127.0.0.1:5572) reports pending uploads/writes or errors."""
    def _rpc(path: str) -> dict:
        req = Request(
            f"http://127.0.0.1:5572/{path}",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=5) as resp:
            return json.loads(resp.read().decode("utf-8"))

    v = _rpc("vfs/stats")
    d = v.get("diskCache") if isinstance(v, dict) else None
    d = d if isinstance(d, dict) else {}
    io_stats = _rpc("core/stats")
    io_stats = io_stats if isinstance(io_stats, dict) else {}
    if d.get("uploadsQueued", 0):
        return f"rclone_uploads_queued:{d.get('uploadsQueued')}"
    if d.get("uploadsInProgress", 0):
        return f"rclone_uploads_in_progress:{d.get('uploadsInProgress')}"
    if d.get("erroredFiles", 0):
        return f"rclone_errored_files:{d.get('erroredFiles')}"
    if d.get("outOfSpace"):
        return "rclone_out_of_space"
    if io_stats.get("transferring"):
        return "rclone_transferring"
    return None


def _local_native_sync_check(native_root: Path) -> tuple[bool, str | None]:
    """Local loopback verification after stopping cloudflared & watchdog, before touching ComfyUI/rclone."""
    mount_dir = native_root / "drive"
    if _native_open_writes(mount_dir):
        return False, "open_writes_on_drive"
    try:
        queue = _http_json("http://127.0.0.1:8188/queue", timeout=5)
        if not isinstance(queue, dict):
            return False, "queue_malformed"
        if queue.get("queue_running") or queue.get("queue_pending"):
            return False, "comfy_queue_busy"
    except Exception as exc:
        return False, f"queue_probe_failed:{type(exc).__name__}"
    try:
        state = _http_json("http://127.0.0.1:8188/studio_native/state?fresh=1", timeout=8)
        if not isinstance(state, dict):
            return False, "native_state_malformed"
        if state.get("busy") is not False:
            return False, f"native_state_busy:{state.get('busy_reason') or 'unknown'}"
        plugins = state.get("plugins")
        if isinstance(plugins, dict):
            if plugins.get("durable") is not True or plugins.get("error"):
                return False, f"plugins_not_durable:{plugins.get('error') or 'pending'}"
    except Exception as exc:
        return False, f"native_state_probe_failed:{type(exc).__name__}"
    try:
        rc_reason = _local_rclone_unsynced_reason()
        if rc_reason:
            return False, rc_reason
    except Exception as exc:
        return False, f"rclone_rc_probe_failed:{type(exc).__name__}"
    return True, None


def _check_port_free(port: int, host: str = "127.0.0.1") -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _gpu_compute_pids() -> list[int] | None:
    """Return active CUDA compute PIDs from nvidia-smi, or None if nvidia-smi fails."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    pids: list[int] = []
    for line in res.stdout.splitlines():
        token = line.strip()
        if token.isdigit():
            pids.append(int(token))
    return pids


def verify_vm_purity(
    expected_baseline_digest: str,
    *,
    studio_root: Path | None = None,
    check_ports: bool = True,
    check_gpu: bool = True,
    check_mounts: bool = True,
    allow_mock_gpu: bool = False,
    home_dir: Path | None = None,
    mountinfo_path: Path | None = None,
) -> dict[str, Any]:
    """Fail-closed purity verification required before switching instance_pool."""
    base_root = Path(studio_root) if studio_root is not None else _default_studio_root()
    report: dict[str, Any] = {
        "mounts_ok": True,
        "ports_ok": True,
        "gpu_compute_ok": True,
        "gpu_alive_ok": True,
        "site_snapshot_ok": False,
        "user_site_ok": True,
        "errors": [],
    }
    if check_mounts:
        mounts = _active_mounts_under(base_root, mountinfo_path=mountinfo_path)
        if mounts:
            report["mounts_ok"] = False
            report["active_mounts"] = mounts
            report["errors"].append("mount_still_active:" + ",".join(mounts))
    if check_ports:
        busy_ports = [p for p in (8188, 5572) if not _check_port_free(p)]
        if busy_ports:
            report["ports_ok"] = False
            report["busy_ports"] = busy_ports
            report["errors"].append("ports_in_use:" + ",".join(str(p) for p in busy_ports))
    if check_gpu and not allow_mock_gpu:
        pids = _gpu_compute_pids()
        if pids is None:
            report["gpu_compute_ok"] = False
            report["errors"].append("nvidia_smi_compute_probe_failed")
        elif pids:
            report["gpu_compute_ok"] = False
            report["gpu_compute_pids"] = pids
            report["errors"].append("gpu_compute_processes_alive:" + ",".join(str(p) for p in pids))
        diag = _probe_gpu()
        if not diag or not diag.get("devices"):
            report["gpu_alive_ok"] = False
            report["errors"].append("gpu_hardware_unavailable")
        else:
            report["gpu_diagnostics"] = diag
    actual_digest, actual_snap = global_site_digest()
    report["actual_baseline_digest"] = actual_digest
    report["expected_baseline_digest"] = expected_baseline_digest
    if (
        isinstance(expected_baseline_digest, str)
        and SHA256.fullmatch(expected_baseline_digest)
        and actual_digest == expected_baseline_digest
    ):
        report["site_snapshot_ok"] = True
    else:
        report["site_snapshot_ok"] = False
        base_file = base_root / ".vm-baseline.json"
        site_diff: dict[str, list[str]] = {}
        if base_file.is_file():
            with contextlib.suppress(OSError, ValueError, TypeError):
                bdoc = json.loads(base_file.read_text(encoding="utf-8"))
                if isinstance(bdoc, dict) and isinstance(bdoc.get("snapshot"), dict):
                    site_diff = _diff_site_snapshots(bdoc.get("snapshot"), actual_snap)
        if site_diff:
            report["site_diff"] = site_diff
            sample = [Path(p).name for vals in site_diff.values() for p in vals][:5]
            report["errors"].append("global_site_snapshot_mismatch:" + ",".join(sample))
        else:
            report["errors"].append("global_site_snapshot_mismatch")
    dirty_user = _user_site_packages_dirty(home_dir=home_dir)
    if dirty_user:
        report["user_site_ok"] = False
        report["dirty_user_site"] = dirty_user[:20]
        report["errors"].append("user_site_packages_present")
    report["ok"] = (
        report["mounts_ok"]
        and report["ports_ok"]
        and report["gpu_compute_ok"]
        and report["gpu_alive_ok"]
        and report["site_snapshot_ok"]
        and report["user_site_ok"]
    )
    return report


def _unmount_native_drive(mount_dir: Path, *, timeout: float = 10.0, mountinfo_path: Path | None = None) -> bool:
    if not _active_mounts_under(mount_dir, mountinfo_path=mountinfo_path):
        return True
    unmounter = shutil.which("fusermount3") or shutil.which("fusermount")
    if unmounter:
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            subprocess.run([unmounter, "-u", str(mount_dir)], capture_output=True, text=True, timeout=10)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _active_mounts_under(mount_dir, mountinfo_path=mountinfo_path):
            return True
        if unmounter:
            with contextlib.suppress(OSError, subprocess.SubprocessError):
                subprocess.run([unmounter, "-u", str(mount_dir)], capture_output=True, text=True, timeout=5)
        time.sleep(0.4)
    return not _active_mounts_under(mount_dir, mountinfo_path=mountinfo_path)


def run_teardown(payload: dict, root: Path, *, home_dir: Path | None = None, mountinfo_path: Path | None = None) -> dict[str, Any]:
    """Execute controlled, fail-closed domain teardown & purity check inside the Colab VM."""
    root = Path(root).resolve()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    studio_root = _default_studio_root(root)
    native_root = studio_root / "native"
    from_pool = str(payload.get("from_pool") or "creation")
    native_bootstrap_failed = bool(payload.get("native_bootstrap_failed") or payload.get("skip_native_stop"))
    clean_native_only = bool(payload.get("clean_native_only"))
    expected_digest = str(payload.get("expected_baseline_digest") or "")
    allow_mock_gpu = bool(payload.get("allow_mock_gpu") or os.getenv("STUDIO_OFFLINE_TEST") == "1")
    steps_log: list[str] = []

    def _fail(code: str, msg: str, *, state: str = "failed", report: dict | None = None) -> dict[str, Any]:
        out = {
            "state": state,
            "cleaned": False,
            "error_code": code,
            "error_message": msg[:1400],
            "teardown_report": {"steps": steps_log, **(report or {})},
        }
        _status(root, state, **{k: v for k, v in out.items() if k != "state"})
        return out

    _status(root, "running", stage="teardown_stop_processes")
    try:
        # Step 1: Domain-specific ordered shutdown & data-protection re-verification
        if (from_pool == "native" and not native_bootstrap_failed) or clean_native_only:
            state_doc: dict[str, Any] = {}
            state_file = native_root / "state.json"
            if state_file.is_file():
                with contextlib.suppress(OSError, ValueError):
                    parsed = json.loads(state_file.read_text(encoding="utf-8"))
                    if isinstance(parsed, dict):
                        state_doc = parsed
            rclone_doc: dict[str, Any] = {}
            rclone_file = native_root / "rclone-state.json"
            if rclone_file.is_file():
                with contextlib.suppress(OSError, ValueError):
                    parsed_rc = json.loads(rclone_file.read_text(encoding="utf-8"))
                    if isinstance(parsed_rc, dict):
                        rclone_doc = parsed_rc

            # 1a) Stop watchdog.py FIRST so it cannot restart cloudflared/ComfyUI/rclone or unassign VM
            if state_doc.get("watchdog_pid"):
                _stop_pid_verified(state_doc.get("watchdog_pid"), state_doc.get("watchdog_start_id"), grace_seconds=5.0)
            for pid, sid in _scan_matching_pids(
                lambda _p, cmd, _c: any("watchdog.py" in arg and "native" in arg for arg in cmd)
            ):
                _stop_pid_verified(pid, sid, grace_seconds=5.0)
            steps_log.append("stopped_watchdog")

            # 1b) Immediately stop cloudflared to sever ALL existing external WS & HTTP sessions at the edge
            if state_doc.get("tunnel_pid"):
                _stop_pid_verified(state_doc.get("tunnel_pid"), state_doc.get("tunnel_start_id"), grace_seconds=5.0)
            for pid, sid in _scan_matching_pids(
                lambda _p, cmd, _c: any("cloudflared" in arg for arg in cmd)
                and any("tunnel.token" in arg for arg in cmd)
            ):
                _stop_pid_verified(pid, sid, grace_seconds=5.0)
            steps_log.append("stopped_cloudflared")

            # 1c) Local loopback sync & queue re-check BEFORE stopping ComfyUI or rclone!
            if clean_native_only:
                mount_dir = native_root / "drive"
                if _native_open_writes(mount_dir):
                    return _fail(
                        "native_unsynced_data",
                        "原生启动失败清理前检测到 Drive 仍有活跃写入，已中止清理以保护数据",
                        state="aborted_unsynced",
                        report={"reason": "open_writes_on_drive"},
                    )
                if not _check_port_free(5572):
                    try:
                        rc_reason = _local_rclone_unsynced_reason()
                    except Exception as exc:
                        rc_reason = f"rclone_rc_probe_failed:{type(exc).__name__}"
                    if rc_reason:
                        return _fail(
                            "native_unsynced_data",
                            f"原生启动失败清理前检测到 rclone 仍有未同步数据（{rc_reason}），已中止清理以保护数据",
                            state="aborted_unsynced",
                            report={"reason": rc_reason},
                        )
            else:
                sync_ok, sync_reason = _local_native_sync_check(native_root)
                if not sync_ok:
                    return _fail(
                        "native_unsynced_data",
                        f"断流后本机复核发现原生 ComfyUI 或 rclone 仍有未完成任务/未上传缓存（{sync_reason}），已中止转移并保护实例数据",
                        state="aborted_unsynced",
                        report={"reason": sync_reason},
                    )
            steps_log.append("verified_local_sync")

            # 1d) Stop background env-cache uploader
            if state_doc.get("env_cache_uploader_pid"):
                _stop_pid_verified(state_doc.get("env_cache_uploader_pid"), grace_seconds=5.0)
            for pid, sid in _scan_matching_pids(
                lambda _p, cmd, _c: any("node_persistence.py" in arg for arg in cmd) and "upload-env-cache" in cmd
            ):
                _stop_pid_verified(pid, sid, grace_seconds=5.0)
            steps_log.append("stopped_env_uploader")

            # 1e) Stop native ComfyUI (check comfy-runtime.json for Manager reboot PID as well as state.json)
            comfy_rt = native_root / "comfy-runtime.json"
            if comfy_rt.is_file():
                with contextlib.suppress(OSError, ValueError):
                    rt_doc = json.loads(comfy_rt.read_text(encoding="utf-8"))
                    if isinstance(rt_doc, dict) and rt_doc.get("pid"):
                        _stop_pid_verified(rt_doc.get("pid"), rt_doc.get("comfy_start_id"), grace_seconds=10.0)
            if state_doc.get("pid"):
                _stop_pid_verified(state_doc.get("pid"), state_doc.get("comfy_start_id"), grace_seconds=10.0)
            native_comfy_str = str((native_root / "ComfyUI").resolve())
            for pid, sid in _scan_matching_pids(
                lambda _p, cmd, cwd: cwd == native_comfy_str or any(native_comfy_str in arg for arg in cmd)
            ):
                _stop_pid_verified(pid, sid, grace_seconds=10.0)
            steps_log.append("stopped_native_comfy")

            # 1f) Re-check rclone upload queue right before stopping rclone
            if not _check_port_free(5572):
                try:
                    rc_reason = _local_rclone_unsynced_reason()
                except Exception as exc:
                    rc_reason = f"rclone_rc_probe_failed:{type(exc).__name__}"
                if rc_reason:
                    return _fail(
                        "native_unsynced_data",
                        f"停机后复核发现 rclone 仍有待上传缓存（{rc_reason}），已中止清理以保护数据",
                        state="aborted_unsynced",
                        report={"reason": rc_reason},
                    )
            rclone_pid = state_doc.get("rclone_pid") or rclone_doc.get("rclone_pid")
            rclone_sid = state_doc.get("rclone_start_id") or rclone_doc.get("rclone_start_id")
            if rclone_pid:
                _stop_pid_verified(rclone_pid, rclone_sid, grace_seconds=10.0, allow_sigkill=False)
            for pid, sid in _scan_matching_pids(
                lambda _p, cmd, _c: any("rclone" in arg for arg in cmd) and str(native_root / "drive") in cmd
            ):
                _stop_pid_verified(pid, sid, grace_seconds=10.0, allow_sigkill=False)
            steps_log.append("stopped_rclone")

            # 1g) Unmount FUSE drive and confirm unmounted
            mount_dir = native_root / "drive"
            if not _unmount_native_drive(mount_dir, timeout=10.0, mountinfo_path=mountinfo_path):
                return _fail(
                    "mount_still_active",
                    "无法安全卸载 Google Drive FUSE 挂载点，已严禁删除目录并中止转移",
                )
            steps_log.append("unmounted_fuse")

        elif from_pool == "creation":
            svc_root = studio_root / "comfy-services"
            if svc_root.is_dir():
                for child in sorted(svc_root.iterdir()):
                    if not child.is_dir() or child.is_symlink():
                        continue
                    sdoc = _read_service_state(child)
                    if sdoc and sdoc.get("pid"):
                        try:
                            _trusted_process_metadata(child, sdoc)
                            _stop_pid_verified(sdoc.get("pid"), sdoc.get("process_start_id"), grace_seconds=10.0)
                        except WorkerError:
                            pass
            svc_prefix = str(svc_root.resolve())
            warm_prefix = str((studio_root / ".warm").resolve())
            for pid, sid in _scan_matching_pids(
                lambda _p, cmd, cwd: (
                    cwd.startswith(svc_prefix)
                    or cwd.startswith(warm_prefix)
                    or any(svc_prefix in arg or warm_prefix in arg for arg in cmd)
                )
            ):
                _stop_pid_verified(pid, sid, grace_seconds=10.0)
            steps_log.append("stopped_creation_services")

        elif from_pool == "ephemeral":
            studio_prefix = str(studio_root.resolve())
            cur_dir_str = str(root)
            for pid, sid in _scan_matching_pids(
                lambda _p, cmd, cwd: (
                    (cwd.startswith(studio_prefix) and not cwd.startswith(cur_dir_str))
                    or any(studio_prefix in arg and cur_dir_str not in arg for arg in cmd)
                )
            ):
                _stop_pid_verified(pid, sid, grace_seconds=10.0)
            steps_log.append("stopped_ephemeral_processes")

        # Step 2: Full-tree mount check (REDLINE: NEVER rmtree if ANY mount exists under studio_root!)
        _status(root, "running", stage="teardown_clean_files")
        active_mounts = _active_mounts_under(studio_root, mountinfo_path=mountinfo_path)
        if active_mounts:
            return _fail(
                "mount_still_active",
                f"检测到 {studio_root} 目录树下仍存在活跃挂载点（{', '.join(active_mounts)}），严禁执行删除",
                report={"active_mounts": active_mounts},
            )
        steps_log.append("verified_no_mounts")

        # Strict removal of old domain directories, weights, credentials, venvs, and ~/.local user packages
        if native_root.exists() or native_root.is_symlink():
            _remove(native_root)
        if not clean_native_only:
            for target_name in ("comfy-services", ".warm"):
                target_path = studio_root / target_name
                if target_path.exists() or target_path.is_symlink():
                    _remove(target_path)
            if studio_root.is_dir():
                for entry in list(studio_root.iterdir()):
                    if entry.name in {root.name, ".vm-baseline.json", ".vm-baseline.lock"}:
                        continue
                    _remove(entry)
        home = Path(home_dir) if home_dir is not None else Path.home()
        local_lib = home / ".local" / "lib"
        if local_lib.is_dir():
            for py_dir in list(local_lib.glob("python*")):
                if py_dir.is_dir() and not py_dir.is_symlink():
                    shutil.rmtree(py_dir, ignore_errors=True)
        steps_log.append("removed_old_domain_trees")

        # Step 3: Fail-closed purity verification
        _status(root, "running", stage="teardown_verify_purity")
        purity = verify_vm_purity(
            expected_digest,
            studio_root=studio_root,
            check_ports=True,
            check_gpu=True,
            check_mounts=True,
            allow_mock_gpu=allow_mock_gpu,
            home_dir=home,
            mountinfo_path=mountinfo_path,
        )
        if not purity.get("ok"):
            err_code = "mount_still_active" if not purity.get("mounts_ok") else (
                "global_site_polluted" if not purity.get("site_snapshot_ok") or not purity.get("user_site_ok") else "teardown_purity_failed"
            )
            return _fail(
                err_code,
                "域清理纯净度校验未通过：" + "; ".join(purity.get("errors") or ["unknown"]),
                report=purity,
            )
        steps_log.append("purity_verified")
        result: dict[str, Any] = {
            "state": "completed",
            "cleaned": True,
            "venv_isolated": True,
            "vm_baseline_digest": purity.get("actual_baseline_digest"),
            "teardown_report": {"steps": steps_log, **purity},
        }
        if purity.get("gpu_diagnostics"):
            result["gpu_diagnostics"] = purity["gpu_diagnostics"]
        _atomic_json(root / "meta.json", {"execution_id": payload.get("execution_id"), "outputs": []})
        _status(root, "completed", **{k: v for k, v in result.items() if k != "state"})
        return result
    except Exception as exc:  # noqa: BLE001
        return _fail("teardown_exception", _safe_error(exc))


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) == 2 and argv[0] == "workbench":
        return _main_workbench_action(argv[1])
    if len(argv) == 2 and argv[0] == "teardown":
        payload_path = Path(argv[1]).resolve()
        payload: Any = None
        try:
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise WorkerError("teardown payload must be an object")
            root = Path(payload["execution_dir"]).resolve()
            res = run_teardown(payload, root)
            return 0 if res.get("cleaned") is True else 1
        except Exception as exc:  # noqa: BLE001
            with contextlib.suppress(Exception):
                root = (
                    Path(payload.get("execution_dir", payload_path.parent)).resolve()
                    if isinstance(payload, dict)
                    else payload_path.parent
                )
                _status(root, "failed", cleaned=False, error_code="teardown_bootstrap_failed", error_message=_safe_error(exc))
            return 1
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
