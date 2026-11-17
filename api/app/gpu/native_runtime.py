"""Colab native service bootstrap. Shipped beside the existing trusted worker.

Uses native ComfyUI directories and rclone's bounded on-demand VFS cache; it
never copies the whole Drive, restores a Python environment, or preloads VRAM.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from urllib.request import urlopen

ROOT = Path("/content/ai-studio/native")
RCLONE_VERSION = "v1.70.3"
RCLONE_SHA256 = "7d69057e69385f6514a9684c7eaa424d972096b130284bb34dd967c4ed4f9dad"
CATEGORIES = ("checkpoints", "diffusion_models", "unet", "text_encoders", "clip", "vae", "loras", "controlnet",
              "embeddings", "upscale_models", "clip_vision", "style_models", "vae_approx", "diffusers",
              "gligen", "hypernetworks", "model_patches", "audio_encoders")

# A small native extension supplies a persistence/busy probe. It is not a UI extension.
EXTENSION = '''import asyncio
import json
from urllib.request import Request, urlopen
from aiohttp import web
from server import PromptServer
import os
from pathlib import Path
import time

def open_writes():
    for process in Path("/proc").glob("[0-9]*"):
        try:
            for fd in (process / "fd").iterdir():
                try:
                    target = os.readlink(fd)
                    if not target.startswith("/content/ai-studio/native/drive/"):
                        continue
                    info = (process / "fdinfo" / fd.name).read_text()
                    flags = next(line.split()[1] for line in info.splitlines() if line.startswith("flags:"))
                    if int(flags, 8) & 3:
                        return True
                except (OSError, StopIteration):
                    continue
        except OSError:
            continue
    return False

def manager_busy():
    from urllib.error import HTTPError
    for path in ("/manager/queue/status", "/v2/manager/queue/status"):
        try:
            with urlopen("http://127.0.0.1:8188" + path, timeout=5) as response:
                state = json.load(response)
            if not isinstance(state.get("is_processing"), bool):
                return True
            return bool(state["is_processing"] or state.get("in_progress_count", 0) or state.get("pending_count", 0) or state.get("total_count", 0) > state.get("done_count", 0))
        except HTTPError as exc:
            if exc.code != 404:
                raise
    return True

def probe():
    try:
        config = Path("/content/ai-studio/native/plugins-config.json")
        if config.exists():
            import sys
            sys.path.insert(0, "/content/ai-studio/native")
            from node_persistence import inventory, digest, read
            root = config.parent
            pending = read(root / "plugins-pending.json", {})
            status = read(root / "plugins-status.json", {})
            nodes = Path(read(config)["comfy"]) / "custom_nodes"
            if status.get("error") or pending.get("source") != digest(inventory(nodes)):
                return {"busy": True}
        def rpc(path):
            with urlopen(Request("http://127.0.0.1:5572/" + path, data=b"{}", headers={"Content-Type":"application/json"}), timeout=5) as r:
                return json.load(r)
        v = rpc("vfs/stats")
        d = v.get("diskCache", {})
        io = rpc("core/stats")
        return {"busy": bool(manager_busy() or open_writes() or d.get("uploadsQueued", 0) or d.get("uploadsInProgress", 0) or d.get("erroredFiles", 0) or d.get("outOfSpace", False) or io.get("transferring") or io.get("errors", 0))}
    except Exception:
        return {"busy": True}

@PromptServer.instance.routes.get("/studio_native/state")
async def state(request):
    Path("/content/ai-studio/native/heartbeat").write_text(str(time.time()))
    result = await asyncio.to_thread(probe)
    status_file = Path("/content/ai-studio/native/plugins-status.json")
    if status_file.exists():
        result["plugins"] = await asyncio.to_thread(PERSISTENCE.status)
        result["busy"] = bool(result["busy"] or not result["plugins"].get("durable") or result["plugins"].get("error"))
    return web.json_response(result)

import sys
sys.path.insert(0, "/content/ai-studio/native")
from node_persistence import install
PERSISTENCE = install(PromptServer.instance, Path("/content/ai-studio/native"), manager_busy)
NODE_CLASS_MAPPINGS = {}
'''


WATCHDOG = r"""import os
import subprocess
import sys
import json
import time
from pathlib import Path
from urllib.request import urlopen

root = Path("/content/ai-studio/native")
# Load only the probe definition, without importing PromptServer or registering routes.
source = (root / "probe.py").read_text()
namespace = {}
exec(compile(source, "native_probe", "exec"), namespace)
while True:
    time.sleep(30)
    try:
        state = json.loads((root / "state.json").read_text())
        try:
            pid = state["tunnel_pid"]
            actual = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[19]
            alive = actual == state.get("tunnel_start_id")
        except (OSError, KeyError, IndexError):
            alive = False
        if not alive:
            binary = state["tunnel_binary"]
            with (root / "tunnel.log").open("ab") as log:
                tunnel = subprocess.Popen([binary, "tunnel", "--no-autoupdate", "--protocol", "http2", "run", "--token-file", str(root / "tunnel.token")], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            state["tunnel_pid"] = tunnel.pid
            state["tunnel_start_id"] = (Path("/proc") / str(tunnel.pid) / "stat").read_text().rsplit(")", 1)[1].split()[19]
            temp = root / "state.tmp"
            temp.write_text(json.dumps(state))
            temp.replace(root / "state.json")
    except Exception:
        pass
    try:
        heartbeat = float((root / "heartbeat").read_text())
        if time.time() - heartbeat < 1200:
            continue
        with urlopen("http://127.0.0.1:8188/queue", timeout=8) as response:
            queue = json.load(response)
        if queue.get("queue_running") or queue.get("queue_pending") or namespace["probe"]()["busy"]:
            continue
        # Two clear probes avoid releasing during the handoff from queued work to persistence.
        time.sleep(10)
        with urlopen("http://127.0.0.1:8188/queue", timeout=8) as response:
            queue = json.load(response)
        if time.time() - float((root / "heartbeat").read_text()) < 1200:
            continue
        if queue.get("queue_running") or queue.get("queue_pending") or namespace["probe"]()["busy"]:
            continue
        from google.colab import runtime
        runtime.unassign()
        break
    except Exception:
        # Unknown state must not interrupt generation or lose pending Drive writes.
        continue
"""


def model_paths(mount):
    # JSON is valid YAML and accepted by ComfyUI's official extra-model-paths loader.
    paths = {"base_path": str(mount), "is_default": True}
    for category in CATEGORIES:
        paths[category] = f"models/{category}\nmodels/native-index/{category}"
    paths["diffusion_models"] += "\nmodels/unet\nmodels/native-index/unet"
    paths["text_encoders"] += "\nmodels/clip\nmodels/native-index/clip"
    return {"studio_drive": paths}


def _write_secret(path, content):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as output:
        output.write(content)
    path.chmod(0o600)


def install_rclone(root, worker):
    # Ubuntu's apt package can predate --vfs-cache-min-free-space. Pin a release
    # with that feature, verified against the upstream release SHA256SUMS.
    binary = root / "rclone"
    if binary.is_file():
        return binary
    archive = root / "rclone.zip"
    release = f"rclone-{RCLONE_VERSION}-linux-amd64"
    url = f"https://github.com/rclone/rclone/releases/download/{RCLONE_VERSION}/{release}.zip"
    try:
        with urlopen(url, timeout=60) as response:
            worker._stream_output(response, archive, 128 * 1024**2)
        with archive.open("rb") as source:
            actual = hashlib.file_digest(source, "sha256").hexdigest()
        if actual != RCLONE_SHA256:
            raise RuntimeError("rclone release checksum mismatch")
        with zipfile.ZipFile(archive) as source, (root / "rclone.partial").open("wb") as output:
            with source.open(release + "/rclone") as executable:
                shutil.copyfileobj(executable, output)
        (root / "rclone.partial").chmod(0o700)
        (root / "rclone.partial").replace(binary)
        return binary
    finally:
        archive.unlink(missing_ok=True)
        (root / "rclone.partial").unlink(missing_ok=True)


def _mount_error(log, secret):
    text = log.read_text(errors="replace")[-8000:]
    for value in secret.values():
        if isinstance(value, str) and value:
            text = text.replace(value, "[redacted]")
    text = re.sub(r'(?i)(authorization|(?:access|refresh)[_-]?token|client[_-]?secret|password|api[_-]?key)["\s]*[:=]["\s]*[^\s,}"]+', r'\1=[redacted]', text)
    return text.strip()[-1200:] or "rclone 未提供错误详情"


def mount_drive(root, secret):
    mount = root / "drive"
    mount.mkdir(exist_ok=True)
    if os.path.ismount(mount):
        return mount
    config = root / "rclone.conf"
    token = json.dumps({"access_token": secret["access_token"], "refresh_token": secret["refresh_token"],
                        "token_type": "Bearer", "expiry": "2000-01-01T00:00:00Z"})
    # The root is the existing AI-Studio/<owner> folder, including models and its existing cache layout.
    _write_secret(config, "[studio]\ntype = drive\nscope = drive.file\n" +
                  f"client_id = {secret['client_id']}\nclient_secret = {secret['client_secret']}\n" +
                  f"root_folder_id = {secret['owner_folder_id']}\ntoken = {token}\n")
    args = [str(root / "rclone"), "mount", "studio:", str(mount), "--config", str(config),
         "--vfs-cache-mode", "full", "--vfs-write-back", "0s",
         "--vfs-cache-max-size", f"{int(secret['cache_gb'])}G", "--vfs-cache-min-free-space", "5G",
         "--cache-dir", str(root / "vfs-cache"), "--dir-cache-time", "30s", "--poll-interval", "15s",
         "--rc", "--rc-addr", "127.0.0.1:5572", "--rc-no-auth", "--log-level", "ERROR"]
    # rclone 1.70.3 starts RC in both the --daemon parent and child. The parent
    # holds 5572 while waiting, so the child cannot bind and the mount fails.
    # Launch one foreground rclone in its own session and supervise readiness.
    log = root / "rclone.log"
    _write_secret(log, "")
    with log.open("ab") as output:
        process = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    try:
        deadline = time.monotonic() + 60
        while True:
            if process.poll() is not None:
                raise RuntimeError("Google Drive 挂载进程退出：" + _mount_error(log, secret))
            if os.path.ismount(mount):
                return mount
            if time.monotonic() >= deadline:
                raise RuntimeError("Google Drive 挂载就绪超时：" + _mount_error(log, secret))
            time.sleep(0.2)
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        raise


def prepare_paths(comfy, mount):
    data = mount / "native-comfy"
    for name in ("user", "input", "output", "custom_nodes"):
        (data / name).mkdir(parents=True, exist_ok=True)
    # Keep Manager-created models under the existing persistent models directory.
    for category in CATEGORIES:
        (mount / "models" / category).mkdir(parents=True, exist_ok=True)
    for name, target in (("models", mount / "models"), ("input", data / "input")):
        link = comfy / name
        if link.is_symlink():
            if link.resolve() != target.resolve():
                raise RuntimeError("Existing native directory points elsewhere")
            continue
        if link.exists():
            # A fresh official checkout has examples; preserve them locally, never remove user data.
            backup = comfy / (name + ".checkout")
            if backup.exists():
                raise RuntimeError("Existing native directory must be inspected")
            link.rename(backup)
        link.symlink_to(target, target_is_directory=True)
    # Plugins execute from local SSD; Drive holds archives rather than live code.
    nodes = comfy / "custom_nodes"
    if nodes.is_symlink():
        if nodes.resolve() != (data / "custom_nodes").resolve():
            raise RuntimeError("Unexpected custom_nodes symlink")
        nodes.unlink()
    nodes.mkdir(exist_ok=True)
    # One-time import of the old Drive layout. Preserve the original as a backup.
    legacy = data / "custom_nodes"
    if legacy.exists() and not (comfy.parent / "plugins-legacy-imported").exists():
        shutil.copytree(legacy, nodes, dirs_exist_ok=True, symlinks=True)
        (comfy.parent / "plugins-legacy-imported").touch()
    return data


def _index_models(secret, worker):
    cache = worker.DriveCache(secret["access_token"], secret["models_folder_id"])
    index = cache._folder("native-index", cache.folder_id, True)
    for model in secret.get("models", []):
        category, name = model.get("category"), model.get("filename")
        if category not in CATEGORIES or not name or Path(name).name != name:
            continue
        parent = cache._folder(category, index, True)
        # ID-based aliases avoid moving, copying, or overwriting existing model files.
        alias = model["file_id"] + "-" + name
        if not cache._list(alias, parent, False):
            cache._json("POST", worker.DRIVE_API + "/files", {
                "name": alias, "mimeType": "application/vnd.google-apps.shortcut", "parents": [parent],
                "shortcutDetails": {"targetId": model["file_id"]}})


def verify_manager(worker, log_path, log_offset=0):
    """Require this process's startup log and a live Manager API, not just Comfy health."""
    try:
        with log_path.open("rb") as log:
            log.seek(log_offset)
            loaded = any(b"[START] ComfyUI-Manager" in line for line in log)
    except OSError as exc:
        raise RuntimeError("Cannot verify native Manager startup; inspect comfy.log") from exc
    if not loaded:
        raise RuntimeError("Native Manager startup marker missing; inspect comfy.log")
    for path in ("/v2/manager/queue/status", "/manager/queue/status"):
        try:
            state = worker._http_json("http://127.0.0.1:8188" + path, timeout=5)
        except worker.WorkerError:
            continue
        if isinstance(state.get("is_processing"), bool):
            print(f"[studio-native] Manager loaded: {path}; startup log: {log_path}", flush=True)
            return
    raise RuntimeError("Native Manager API unavailable; inspect comfy.log")


def run(root, payload, stages):
    import runtime_worker as worker
    secret = payload["native_secrets"]
    ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    with worker._file_lock(ROOT / ".bootstrap.lock"):
        commit = payload["snapshot"]["manifest"]["comfy_commit"]
        marker = ROOT / "state.json"
        if marker.exists():
            state = json.loads(marker.read_text())
            if state.get("commit") != commit:
                raise RuntimeError("Native ComfyUI version differs; stop the old session first")
            worker._http_json("http://127.0.0.1:8188/system_stats", timeout=5)
            if not state.get("comfy_log"):
                raise RuntimeError("Native Manager verification metadata missing; stop the old session first")
            verify_manager(worker, Path(state["comfy_log"]), state.get("comfy_log_offset", 0))
        else:
            stages.start("native_environment")
            worker._run(["apt-get", "update", "-qq"], timeout=180)
            worker._run(["apt-get", "install", "-y", "-qq", "fuse3"], timeout=180)
            install_rclone(ROOT, worker)
            if not shutil.which("cloudflared"):
                binary = ROOT / "cloudflared"
                with urlopen("https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", timeout=60) as response:
                    worker._stream_output(response, binary, 100 * 1024**2)
                binary.chmod(0o700)
            stages.start("native_drive")
            _index_models(secret, worker)
            mount = mount_drive(ROOT, secret)
            stages.start("native_models")
            comfy = ROOT / "ComfyUI"
            if not comfy.exists():
                comfy.mkdir()
                worker._checkout_comfy(comfy, commit, 180)
            worker._verify_checkout(comfy, commit)
            data = prepare_paths(comfy, mount)
            extra = ROOT / "extra_model_paths.yaml"
            extra.write_text(json.dumps(model_paths(mount)))
            stages.start("native_environment")
            try:
                from node_persistence import protect_environment
            except ImportError:
                from .node_persistence import protect_environment
            protect_environment(ROOT)
            worker._run([sys.executable, "-m", "pip", "install", "-q", "-r", str(comfy / "requirements.txt")], timeout=600)
            manager = comfy / "manager_requirements.txt"
            if not manager.is_file():
                raise RuntimeError("Pinned ComfyUI must support the official Manager (--enable-manager)")
            print(f"[studio-native] Installing Manager requirements: {manager}; Python: {sys.executable}", flush=True)
            worker._run([sys.executable, "-m", "pip", "install", "-q", "-r", str(manager)], timeout=300)
            try:
                from node_persistence import Persistence, protect_environment
            except ImportError:
                from .node_persistence import Persistence, protect_environment
            shutil.copyfile(Path(__file__).with_name("node_persistence.py"), ROOT / "node_persistence.py")
            persistence = Persistence(ROOT, comfy, data / "user")
            persistence.restore(payload.get("native_plugin_generation"))
            protect_environment(ROOT)
            persistence.dependencies()
            (comfy / "custom_nodes" / "studio_native_state.py").write_text(EXTENSION)
            guard = comfy / "custom_nodes" / "studio_native_guard"
            guard.mkdir(exist_ok=True)
            (guard / "__init__.py").write_text("NODE_CLASS_MAPPINGS = {}\n")
            (guard / "prestartup_script.py").write_text(
                "import sys, os\nfrom pathlib import Path\n"
                "sys.path.insert(0, '/content/ai-studio/native')\n"
                "from node_persistence import read, atomic, protect_environment, Persistence\n"
                "root = Path('/content/ai-studio/native')\n"
                "config = read(root / 'plugins-config.json')\n"
                "if config and config.get('pid') != os.getpid():\n"
                "    config['log_offset'] = Path(config['log']).stat().st_size\n"
                "    config['pid'] = os.getpid()\n"
                "    atomic(root / 'plugins-config.json', config)\n"
                "protect_environment(root)\n"
                "try:\n"
                "    Persistence(root, Path(config['comfy']), Path(config['user'])).dependencies()\n"
                "except Exception as exc:\n"
                "    atomic(root / 'plugin-errors.json', {'error': str(exc)})\n"
                "    raise\n")
            probe = EXTENSION[EXTENSION.index("def open_writes():"):EXTENSION.index("@PromptServer")]
            (ROOT / "probe.py").write_text("import os, json\nfrom pathlib import Path\nfrom urllib.request import Request, urlopen\n" + probe)
            stages.start("comfy_start")
            args = [sys.executable, "main.py", "--listen", "127.0.0.1", "--port", "8188", "--enable-manager",
                    "--user-directory", str(data / "user"), "--input-directory", str(data / "input"),
                    "--output-directory", str(data / "output"), "--temp-directory", str(ROOT / "temp"),
                    "--max-upload-size", str(secret["max_upload_mb"]),
                    "--extra-model-paths-config", str(extra), "--preview-method", "auto", "--disable-auto-launch"]
            comfy_log = root / "comfy.log"
            with comfy_log.open("ab") as log:
                log_offset = log.tell()
                worker._atomic_json(ROOT / "plugins-config.json", {"comfy": str(comfy), "user": str(data / "user"),
                                                                  "log": str(comfy_log), "log_offset": log_offset})
                process = subprocess.Popen(args, cwd=comfy, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            for _ in range(180):
                try:
                    worker._http_json("http://127.0.0.1:8188/system_stats", timeout=3)
                    break
                except worker.WorkerError:
                    if process.poll() is not None:
                        raise RuntimeError("Native ComfyUI exited during startup; inspect comfy.log")
                    time.sleep(1)
            else:
                raise RuntimeError("Native ComfyUI health check timed out")
            try:
                verify_manager(worker, comfy_log, log_offset)
                with comfy_log.open("rb") as source:
                    source.seek(log_offset)
                    failed = [line.decode(errors="replace").strip() for line in source
                              if b"(IMPORT FAILED)" in line or b"Cannot import " in line]
                if failed:
                    raise RuntimeError("Custom node load failed: " + "; ".join(failed)[-2000:])
            except BaseException:
                # A retry must not leave a second, unverified ComfyUI on the same port.
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                raise
            stages.start("native_tunnel")
            tunnel_file = ROOT / "tunnel.token"
            _write_secret(tunnel_file, secret["tunnel_token"])
            binary = shutil.which("cloudflared") or str(ROOT / "cloudflared")
            with (ROOT / "tunnel.log").open("ab") as log:
                tunnel = subprocess.Popen([binary, "tunnel", "--no-autoupdate", "--protocol", "http2", "run", "--token-file", str(tunnel_file)],
                                          stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            (ROOT / "heartbeat").write_text(str(time.time()))
            (ROOT / "watchdog.py").write_text(WATCHDOG)
            with (ROOT / "watchdog.log").open("ab") as log:
                subprocess.Popen([sys.executable, str(ROOT / "watchdog.py")], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            worker._atomic_json(marker, {"commit": commit, "pid": process.pid, "tunnel_pid": tunnel.pid,
                                         "tunnel_start_id": worker._process_metadata(tunnel.pid)["start_id"], "tunnel_binary": binary,
                                         "comfy": str(comfy), "comfy_log": str(comfy_log),
                                         "comfy_log_offset": log_offset, "started": time.time()})
        stages.end()
        report = root / "outputs" / "native-ready.json"
        report.parent.mkdir(exist_ok=True)
        report.write_text(json.dumps({"native_comfy": True, "commit": commit, "stage_seconds": stages.timings}))
        return [{"filename": report.name, "path": str(report.relative_to(root)), "mime": "application/json", "size": report.stat().st_size}]
