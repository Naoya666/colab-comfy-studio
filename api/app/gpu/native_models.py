"""Local model preparation in the ComfyUI process. All file and network I/O runs off its event loop."""
from __future__ import annotations

import asyncio
import errno
import fcntl
import hashlib
import hmac
import json
import logging
import os
import re
import shutil
import sys
import threading
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import parse_qsl, quote, urlencode, urlsplit

import httpx

try:
    from . import model_dependencies as deps
    from . import runtime_worker as studio_worker
    from .model_sources import (
        SHA,
        ModelError,
        SourceHTTP,
        filename,
        https_url,
        is_controlled_upscale_extension,
        relative_path,
        validate_controlled_upscale_source,
        validate_weight_file,
    )
except ImportError:  # The bootstrap ships standalone modules to Colab.
    import model_dependencies as deps
    import runtime_worker as studio_worker
    from model_sources import (
        SHA,
        ModelError,
        SourceHTTP,
        filename,
        https_url,
        is_controlled_upscale_extension,
        relative_path,
        validate_controlled_upscale_source,
        validate_weight_file,
    )

RESERVE = 5 * 1024**3
CHUNK = 1024**2
MAX_WORKFLOW_FILE_BYTES = 16 * 1024**2
MAX_WORKFLOW_LIST_ITEMS = 500
WORKFLOW_SOURCES = {
    "comfy": ("native-comfy", "user", "default", "workflows"),
    "warmup": ("native-comfy", "warmup"),
}
MD5 = re.compile(r"[0-9a-fA-F]{32}")
FILE_ID = re.compile(r"[A-Za-z0-9_-]{1,200}")
RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+)")
ALIASES = {"unet": "diffusion_models", "clip": "text_encoders", "t2i_adapter": "controlnet"}
LOG = logging.getLogger("studio-native.models")


class WorkflowConflictError(ModelError):
    pass


class WorkflowNotFoundError(ModelError):
    pass


def _is_api_node_map(data):
    return isinstance(data, dict) and bool(data) and all(
        isinstance(node, dict) and isinstance(node.get("class_type"), str) and isinstance(node.get("inputs"), dict)
        for node in data.values()
    )


def wrap_visual_workflow(workflow):
    if not isinstance(workflow, dict) or not workflow:
        raise ModelError("请提供有效的 ComfyUI 工作流 JSON 对象")
    if "prompt" in workflow or "workflow" in workflow:
        if "prompt" in workflow and not _is_api_node_map(workflow.get("prompt")):
            raise ModelError("工作流中的 prompt 字段必须是有效的 API 节点字典")
        if "workflow" in workflow and not isinstance(workflow.get("workflow"), dict):
            raise ModelError("工作流中的 workflow 字段必须是有效的 JSON 对象")
        return workflow
    if isinstance(workflow.get("nodes"), list):
        return {"workflow": workflow}
    if _is_api_node_map(workflow):
        return {"prompt": workflow}
    raise ModelError("无法识别的工作流 JSON：需包含原生 nodes 数组或 API 节点图")


def workflow_relpath(value, *, ensure_json_ext=False):
    if not isinstance(value, str):
        raise ModelError("工作流路径无效")
    cleaned = value.strip()
    if ensure_json_ext and cleaned and not cleaned.lower().endswith(".json"):
        cleaned += ".json"
    try:
        normalized = relative_path(cleaned)
    except ModelError:
        raise ModelError("工作流路径无效，不能包含绝对路径、特殊字符或目录穿越") from None
    parts = normalized.split("/")
    if (len(normalized) > 255 or any(part.startswith(".") or len(part) > 120 for part in parts)
            or not parts[-1].lower().endswith(".json") or len(parts[-1]) <= 5):
        raise ModelError("工作流文件名需以 .json 结尾且不能为隐藏文件")
    return normalized


def category_for(category):
    return ALIASES.get(category, category)


def registered_paths(row):
    category = category_for(row["model_type"])
    name = relative_path(row.get("relative_path") or row["filename"])
    legacy = row.get("legacy_target")
    if legacy:
        directory, _, path = relative_path(legacy).partition("/")
        if category_for(directory) == category and path:
            return category, {path, name} if not row.get("relative_path") else {name}
    return category, {name}


def runnable_graph(graph, targets, node_types):
    if node_types is None:
        return graph
    roots = []
    for ident, node in graph.items():
        if not isinstance(node, dict) or not isinstance(node.get("class_type"), str):
            return {}  # Let ComfyUI return its own graph validation errors.
        node_type = node_types.get(node["class_type"])
        if node_type is None:
            return {}
        if getattr(node_type, "OUTPUT_NODE", False) is True and (targets is None or ident in targets):
            roots.append(ident)
    reachable = {}
    while roots:
        ident = roots.pop()
        if ident in reachable:
            continue
        node = graph[ident]
        reachable[ident] = node
        inputs = node.get("inputs")
        if isinstance(inputs, dict):
            roots.extend(value[0] for value in inputs.values() if (
                isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and value[0] in graph
            ))
    return {ident: node for ident, node in reachable.items()
            if not getattr(node_types[node["class_type"]], "API_NODE", False)}


def dependencies(body, explicit, *, visual=False, trusted=False, node_types=None):
    graph = body.get("prompt", {})
    extra = body.get("extra_data") or {}
    extra_png = extra.get("extra_pnginfo") or {} if isinstance(extra, dict) else {}
    workflow = extra_png.get("workflow") if isinstance(extra_png, dict) else None
    if visual:
        workflow = body.get("workflow") or workflow
    if not isinstance(graph, dict) or ("prompt" not in body and not isinstance(workflow, dict)):
        raise ModelError("请提供原生 ComfyUI API 工作流或工作流 JSON")
    if not isinstance(workflow, dict):
        workflow = {}
    # Only known file selectors and admin-configured rules. Unknown and linked
    # inputs are handled through explicit declarations or the native validator.
    selected = graph if visual else runnable_graph(graph, body.get("partial_execution_targets"), node_types)
    # Empty reachable API graphs should not fall through to the visual widget scanner.
    visual_metadata = workflow if selected or not graph else {}
    for dep in deps.scan({"api_graph": selected, "editable_graph": visual_metadata}, node_types=node_types):
        if dep.get("bypassed") or dep["status"] == "manual_review":
            continue
        if (dep.get("enabled_field") or dep.get("disabled_by_switch")) and not dep.get("required", True):
            continue
        if dep["node_type"] == "VAELoader" and dep["reference"] in {
            "pixel_space", "taesd", "taesdxl", "taesd3", "taef1", "taef2", "taeqi2_1",
        }:
            continue
        yield {"category": category_for(dep["model_type"]), "name": dep["reference"],
               "url": dep.get("source_hint"), "sha256": dep.get("expected_sha256"), "size": dep.get("expected_size"),
               "conflict": dep.get("source_conflict") or dep.get("source_error"),
               "embedding": dep.get("embedding", False), "match_stem": dep.get("match_stem", False)}
    if not isinstance(explicit, list) or len(explicit) > 256:
        raise ModelError("显式依赖必须为至多 256 项的数组")
    for item in explicit:
        if not isinstance(item, dict) or item.get("category") not in deps.directories():
            raise ModelError("显式依赖的模型类别无效")
        name = relative_path(item.get("name"))
        url = https_url(item["url"]) if item.get("url") else None
        digest, size = item.get("sha256"), item.get("size")
        if digest is not None and (not isinstance(digest, str) or not SHA.fullmatch(digest)):
            raise ModelError("显式依赖的 SHA-256 无效")
        if size is not None and (type(size) is not int or size <= 0):
            raise ModelError("显式依赖大小必须为正整数字节")
        drive_id = item.get("drive_file_id") if trusted else None
        md5 = item.get("md5") if trusted else None
        if drive_id and (not isinstance(drive_id, str) or not FILE_ID.fullmatch(drive_id)):
            raise ModelError("Drive 文件 ID 无效")
        if md5 and (not isinstance(md5, str) or not MD5.fullmatch(md5)):
            raise ModelError("Drive 文件 MD5 无效")
        yield {"category": category_for(item["category"]), "name": name, "url": url,
               "sha256": digest.lower() if digest else None, "size": size, "md5": md5.lower() if md5 else None, "drive_file_id": drive_id}


def model_problem(message, *, code="model_preparation_failed"):
    return {"error": {"type": code, "code": code, "message": message, "details": message,
                      "extra_info": {}}, "node_errors": {}}


class DiskSpaceError(ModelError):
    def __init__(self):
        super().__init__("Colab 本地磁盘空间不足（需额外保留 5 GiB）；请释放闲置文件或使用 Drive 挂载模式")


class LocalModels:
    def __init__(self, root, model_dir, config, *, http=None, node_types=None):
        self.root, self.model_dir = root, model_dir
        self.config = config
        self.node_types = node_types
        self.catalog = config.get("catalog", [])
        self.tokens = config.get("tokens", {})
        self.drive_auth = config.get("drive_auth", {})
        self.client = http or httpx.Client(timeout=45, trust_env=False, follow_redirects=False)
        self._state_lock = threading.Lock()
        self.drive_tokens = {}
        self.partials = root / "model-transfers"
        self.partials.mkdir(mode=0o700, exist_ok=True)
        self.inventory_path = root / "local-models-inventory.json"
        self.usage_records = self._load_usage_records()
        self.hashes = self._load_hash_cache()
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="studio-models")
        self.jobs = {}
        self.current_rows = {}
        self.inflight = {}
        self._reserved_bytes: dict[Path, int] = {}
        self._rw_loop = None
        self._rw_state = None
        self.handoffs = 0
        self.lora_generation = 0

    def _rw_lock_state(self):
        loop = asyncio.get_running_loop()
        if self._rw_loop is not loop or self._rw_state is None:
            self._rw_loop = loop
            self._rw_state = {
                "readers": 0,
                "writer": False,
                "writers_waiting": 0,
                "waiters": [],
            }
        return self._rw_state

    @staticmethod
    def _notify_rw_waiters(state):
        waiters = state["waiters"]
        if waiters:
            state["waiters"] = []
            for waiter in waiters:
                if not waiter.done():
                    waiter.set_result(None)

    @asynccontextmanager
    async def _activity_lock(self, *, exclusive: bool = False):
        """Serialize studio vs native prepares asynchronously without blocking the event loop,
        while holding models-active.lock for cross-process probe visibility.

        Yields a refcounted lock handle whose ``track(future)`` method extends both the
        in-process rw-lock and cross-process flock until tracked background executor futures
        finish, even if the awaiting coroutine is cancelled mid-download.
        """
        loop = asyncio.get_running_loop()
        state = self._rw_lock_state()
        if exclusive:
            state["writers_waiting"] += 1
            try:
                while state["writer"] or state["readers"] > 0:
                    waiter = loop.create_future()
                    state["waiters"].append(waiter)
                    try:
                        await waiter
                    finally:
                        try:
                            state["waiters"].remove(waiter)
                        except ValueError:
                            pass
                state["writer"] = True
            except BaseException:
                state["writers_waiting"] -= 1
                self._notify_rw_waiters(state)
                raise
            else:
                state["writers_waiting"] -= 1
        else:
            while state["writer"] or state["writers_waiting"] > 0:
                waiter = loop.create_future()
                state["waiters"].append(waiter)
                try:
                    await waiter
                finally:
                    try:
                        state["waiters"].remove(waiter)
                    except ValueError:
                        pass
            state["readers"] += 1

        lock_file = None
        try:
            lock_file = (self.root / "models-active.lock").open("a")
            mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            while True:
                try:
                    fcntl.flock(lock_file, mode | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    await asyncio.sleep(0.05)
        except BaseException:
            if lock_file is not None:
                try:
                    lock_file.close()
                except OSError:
                    pass
            if exclusive:
                state["writer"] = False
            else:
                state["readers"] -= 1
            self._notify_rw_waiters(state)
            raise

        notify = self._notify_rw_waiters

        class _ActivityLockHold:
            def __init__(self):
                self._refs = 1
                self._released = False

            def track(self, fut: asyncio.Future) -> None:
                if self._released or fut.done():
                    return
                self._refs += 1
                fut.add_done_callback(self._on_done)

            def _on_done(self, fut) -> None:
                if not fut.cancelled():
                    try:
                        fut.exception()
                    except Exception:
                        pass
                self.release()

            def release(self) -> None:
                if self._released:
                    return
                self._refs -= 1
                if self._refs > 0:
                    return
                self._released = True
                try:
                    lock_file.close()
                except OSError:
                    pass
                if exclusive:
                    state["writer"] = False
                else:
                    state["readers"] -= 1
                notify(state)

        hold = _ActivityLockHold()
        try:
            yield hold
        finally:
            hold.release()

    _SENSITIVE_QUERY_KEYS = frozenset({
        "token", "access_token", "auth", "key", "api_key", "apikey",
        "sig", "signature", "x-amz-signature", "x-amz-credential",
        "x-amz-security-token", "x-goog-signature", "x-goog-credential",
        "expires", "policy", "key-pair-id", "password", "secret",
    })

    @classmethod
    def _sanitize_source(cls, url, source_kind=None):
        if not isinstance(url, str) or not url.strip():
            return None
        parsed = urlsplit(url.strip())
        if parsed.scheme.lower() != "https" or not parsed.hostname:
            return None
        host = parsed.hostname.lower()
        if host.endswith("googleapis.com") or host.endswith("googleusercontent.com"):
            return None
        port = f":{parsed.port}" if parsed.port else ""
        safe_params = [
            (k, v)
            for k, v in parse_qsl(parsed.query, keep_blank_values=False)
            if k.lower() not in cls._SENSITIVE_QUERY_KEYS
            and not any(tok in k.lower() for tok in ("token", "secret", "sig", "auth", "key"))
        ]
        query = ("?" + urlencode(safe_params)) if safe_params else ""
        clean = f"https://{host}{port}{parsed.path}{query}"
        out = {"download_url": clean}
        if isinstance(source_kind, str) and source_kind:
            out["kind"] = source_kind
        return out

    def _load_usage_records(self) -> dict:
        try:
            if not self.inventory_path.is_file():
                return {}
            raw = json.loads(self.inventory_path.read_text())
            if not isinstance(raw, dict) or not isinstance(raw.get("usage"), dict):
                return {}
            return {str(k): dict(v) for k, v in raw["usage"].items() if isinstance(k, str) and isinstance(v, dict)}
        except (OSError, ValueError, TypeError):
            return {}

    def _write_inventory_doc(self, doc: dict) -> None:
        tmp = self.inventory_path.with_name(f".local-models-inventory.{uuid.uuid4().hex}.tmp")
        try:
            tmp.write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")))
            tmp.replace(self.inventory_path)
        except OSError:
            tmp.unlink(missing_ok=True)

    def _record_model_used(self, target: Path, plan: dict) -> None:
        try:
            rel_parts = target.relative_to(self.model_dir).parts
            if len(rel_parts) < 2:
                return
            category = category_for(rel_parts[0])
            name = "/".join(rel_parts[1:])
            rel_key = f"{category}/{name}"
            drive = plan.get("drive") if isinstance(plan.get("drive"), dict) else {}
            sanitized = self._sanitize_source(plan.get("url"), plan.get("source_kind"))
            size = plan.get("size")
            if not size:
                try:
                    size = target.stat().st_size
                except OSError:
                    size = None
            now = round(time.time(), 3)
            record = {
                "category": category,
                "name": name,
                "last_used_at": now,
                "sha256": plan.get("sha256") or drive.get("sha256"),
                "md5": plan.get("md5") or drive.get("md5"),
                "size": size,
                "drive_file_id": drive.get("drive_file_id"),
                "account": drive.get("account") or (self.config.get("drive_account") if drive.get("drive_file_id") else None),
                "source": sanitized,
            }
            with self._state_lock:
                prev = self.usage_records.get(rel_key) or {}
                merged = {**prev, **{k: v for k, v in record.items() if v is not None}}
                merged["last_used_at"] = now
                self.usage_records[rel_key] = merged
                usage_copy = dict(self.usage_records)
            try:
                existing = json.loads(self.inventory_path.read_text()) if self.inventory_path.is_file() else {}
                if not isinstance(existing, dict):
                    existing = {}
            except (OSError, ValueError):
                existing = {}
            existing["usage"] = usage_copy
            existing["updated_at"] = now
            self._write_inventory_doc(existing)
        except Exception:
            pass

    def scan_inventory(self) -> dict:
        """Scan local SSD models into catalog/source-backed recoverable_models vs wild_models."""
        catalog_index: dict[tuple[str, str], dict] = {}
        for row in self.catalog:
            if not isinstance(row, dict):
                continue
            try:
                row_cat, paths = registered_paths(row)
            except (KeyError, ModelError, TypeError):
                continue
            for p in paths:
                catalog_index.setdefault((row_cat, p), row)
            if row.get("drive_file_id") and row.get("filename"):
                catalog_index.setdefault((row_cat, f"{row['drive_file_id']}-{row['filename']}"), row)

        with self._state_lock:
            usage_copy = dict(self.usage_records)

        recoverable: list[dict] = []
        wild: list[str] = []
        valid_categories = set(deps.directories().keys())
        if self.model_dir.is_dir() and not self.model_dir.is_symlink():
            for dirpath, dirnames, filenames in os.walk(self.model_dir, followlinks=False):
                cur_dir = Path(dirpath)
                dirnames[:] = sorted(
                    d for d in dirnames
                    if not d.startswith(".") and not (cur_dir / d).is_symlink()
                )
                for fname in sorted(filenames):
                    if fname.startswith(".") or fname.endswith((".partial", ".tmp")):
                        continue
                    fpath = cur_dir / fname
                    if fpath.is_symlink():
                        continue
                    try:
                        rel_parts = fpath.relative_to(self.model_dir).parts
                        if len(rel_parts) < 2:
                            continue
                        category = category_for(rel_parts[0])
                        if category not in valid_categories:
                            continue
                        name = "/".join(rel_parts[1:])
                        stat = fpath.stat()
                    except (OSError, ValueError):
                        continue
                    if stat.st_size <= 0:
                        continue
                    rel_key = f"{category}/{name}"
                    cat_row = catalog_index.get((category, name))
                    usage = usage_copy.get(rel_key)
                    drive_file_id = None
                    account = None
                    sanitized_source = None
                    sha256 = None
                    md5 = None
                    size = stat.st_size
                    if cat_row is not None:
                        drive_file_id = cat_row.get("drive_file_id")
                        account = cat_row.get("account")
                        src = cat_row.get("source") if isinstance(cat_row.get("source"), dict) else None
                        if src:
                            sanitized_source = self._sanitize_source(src.get("download_url"), src.get("kind"))
                        sha256 = cat_row.get("sha256") or (usage or {}).get("sha256")
                        md5 = cat_row.get("md5") or (usage or {}).get("md5")
                        size = cat_row.get("size") or stat.st_size
                    if not drive_file_id and not sanitized_source and usage is not None:
                        drive_file_id = usage.get("drive_file_id")
                        account = usage.get("account")
                        usrc = usage.get("source") if isinstance(usage.get("source"), dict) else None
                        if usrc:
                            sanitized_source = self._sanitize_source(usrc.get("download_url"), usrc.get("kind"))
                        sha256 = sha256 or usage.get("sha256")
                        md5 = md5 or usage.get("md5")
                        size = usage.get("size") or size
                    if not drive_file_id and not sanitized_source:
                        wild.append(rel_key)
                        continue
                    last_used = float((usage or {}).get("last_used_at") or round(stat.st_mtime, 3))
                    entry = {
                        "category": category,
                        "name": name,
                        "last_used_at": round(last_used, 3),
                        "size": int(size),
                    }
                    if sha256:
                        entry["sha256"] = sha256
                    if md5:
                        entry["md5"] = md5
                    if drive_file_id:
                        entry["drive_file_id"] = drive_file_id
                    if account:
                        entry["account"] = account
                    if sanitized_source:
                        entry["source"] = sanitized_source
                    recoverable.append(entry)

        recoverable.sort(key=lambda item: (-float(item.get("last_used_at") or 0.0), item["category"], item["name"]))
        wild = sorted(set(wild))
        canonical = json.dumps(
            {"recoverable_models": recoverable, "wild_models": wild},
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        inv_digest = hashlib.sha256(canonical).hexdigest()
        now = round(time.time(), 3)
        with self._state_lock:
            latest_usage = dict(self.usage_records)
        self._write_inventory_doc({
            "digest": inv_digest,
            "recoverable_models": recoverable,
            "wild_models": wild,
            "usage": latest_usage,
            "updated_at": now,
        })
        return {
            "digest": inv_digest,
            "recoverable_models": recoverable,
            "wild_models": wild,
            "updated_at": now,
        }

    def inventory_digest(self) -> str:
        return self.scan_inventory()["digest"]

    def _hash_cache_file(self) -> Path:
        return self.root / ".model-hash-cache.json"

    def _load_hash_cache(self) -> dict:
        path = self._hash_cache_file()
        try:
            if not path.is_file():
                return {}
            raw = json.loads(path.read_text())
            if not isinstance(raw, dict):
                return {}
            loaded = {}
            for key, value in raw.items():
                if (
                    isinstance(key, str)
                    and isinstance(value, dict)
                    and isinstance(value.get("stamp"), list)
                    and len(value["stamp"]) == 5
                ):
                    loaded[Path(key)] = (
                        tuple(value["stamp"]),
                        value.get("sha256"),
                        None,
                        value.get("md5"),
                    )
            return loaded
        except (OSError, ValueError, TypeError):
            return {}

    def _save_hash_cache(self) -> None:
        # Never persist external download URLs in plaintext; URL provenance is
        # stored separately as an HMAC digest in model-transfers/*.json.
        with self._state_lock:
            payload = {
                str(path): {"stamp": list(entry[0]), "sha256": entry[1], "md5": entry[3]}
                for path, entry in self.hashes.items()
            }
        cache_file = self._hash_cache_file()
        tmp = cache_file.with_name(f".model-hash-cache.{uuid.uuid4().hex}.tmp")
        try:
            tmp.write_text(json.dumps(payload, separators=(",", ":")))
            tmp.replace(cache_file)
        except OSError:
            tmp.unlink(missing_ok=True)

    def signed(self, request):
        return hmac.compare_digest(request.headers.get("X-Studio-Model-Channel", ""), self.config["channel"])

    def target(self, category, name, *, directory=None):
        if category not in deps.directories():
            raise ModelError("模型目录未登记，请补充节点映射")
        path = self.model_dir / relative_path(directory or deps.directories()[category]) / relative_path(name)
        if path.resolve() != path or not path.is_relative_to(self.model_dir) or self.model_dir.is_symlink():
            raise ModelError("本地模型路径含软链接或越界，请整理模型目录")
        return path

    def drive_token(self, account):
        with self._state_lock:
            if account not in self.drive_auth:
                raise ModelError("Drive 副本所属账号未授权，请检查模型库存储账号")
            auth = dict(self.drive_auth[account])
            identity = graph_key(auth)
            current = self.drive_tokens.get(account)
            if current and current[1] > time.monotonic() and current[2] == identity:
                return current[0]
        response = self.client.post("https://oauth2.googleapis.com/token", data={
            "grant_type": "refresh_token", **auth,
        })
        if not response.is_success:
            raise ModelError(f"Drive 授权刷新失败（HTTP {response.status_code}）")
        data = response.json()
        token = data.get("access_token")
        if not isinstance(token, str) or not token:
            raise ModelError("Drive 未返回有效授权")
        with self._state_lock:
            self.drive_tokens[account] = token, time.monotonic() + max(0, int(data.get("expires_in", 0)) - 60), identity
        return token

    @staticmethod
    def drive_identity(item):
        if item.get("trashed") or item.get("mimeType", "").startswith("application/vnd.google-apps."):
            return None
        size = int(item.get("size") or 0)
        sha256 = item.get("sha256Checksum") or (item.get("appProperties") or {}).get("sha256")
        md5 = item.get("md5Checksum")
        sha256 = sha256.lower() if isinstance(sha256, str) and SHA.fullmatch(sha256) else None
        md5 = md5.lower() if isinstance(md5, str) and MD5.fullmatch(md5) else None
        if size > 0 and (sha256 or md5):
            return {"drive_file_id": item["id"], "size": size, "sha256": sha256, "md5": md5}
        return None

    def drive_metadata(self, file_id, token):
        params = urlencode({"fields": "id,trashed,mimeType,size,sha256Checksum,md5Checksum,appProperties"})
        url = f"https://www.googleapis.com/drive/v3/files/{quote(file_id)}?{params}"
        response = self.client.get(url, headers={"Authorization": "Bearer " + token})
        if not response.is_success:
            raise ModelError(f"Drive 模型元数据读取失败（HTTP {response.status_code}）")
        item = self.drive_identity(response.json())
        if not item or item["drive_file_id"] != file_id:
            raise ModelError("Drive 副本缺少完整性校验信息")
        return item

    def legacy_copy(self, target, plan, *, token):
        folder_id = self.config.get("models_folder")
        if not folder_id:
            return None
        if not FILE_ID.fullmatch(folder_id):
            raise ModelError("Drive 模型根目录 ID 无效")
        parts = target.relative_to(self.model_dir).parts
        folder_mime = "application/vnd.google-apps.folder"

        def exact(name, parent, folder):
            name = name.replace("\\", "\\\\").replace("'", "\\'")
            op = "=" if folder else "!="
            query = f"name = '{name}' and '{parent}' in parents and trashed = false and mimeType {op} '{folder_mime}'"
            params = urlencode({"q": query, "pageSize": 100,
                                "fields": "nextPageToken,files(id,mimeType,size,sha256Checksum,md5Checksum,appProperties)"})
            url = "https://www.googleapis.com/drive/v3/files?" + params
            response = self.client.get(url, headers={"Authorization": "Bearer " + token})
            if not response.is_success:
                raise ModelError(f"读取 Drive 模型目录失败（HTTP {response.status_code}）")
            data = response.json()
            if data.get("nextPageToken"):
                raise ModelError("Drive 上存在过多同路径条目，请在模型库绑定具体文件")
            return data.get("files", [])

        dirs = [(parts[0], *parts[1:-1])]
        dirs.extend((alias, *parts[1:-1]) for alias, canonical in ALIASES.items() if canonical == parts[0])
        matches = {}
        for directory in dirs:
            parent = folder_id
            for component in directory:
                folders = exact(component, parent, True)
                if len(folders) > 1:
                    raise ModelError("Drive 上存在同名目录，请在模型库指定文件 ID")
                if not folders:
                    break
                parent = folders[0]["id"]
                if not FILE_ID.fullmatch(parent):
                    raise ModelError("Drive 父目录 ID 无效")
            else:
                for item in exact(parts[-1], parent, False):
                    identity = self.drive_identity(item)
                    if not identity or (plan["size"] and identity["size"] != plan["size"]):
                        continue
                    if plan["sha256"] and identity["sha256"] != plan["sha256"]:
                        continue
                    matches[identity["drive_file_id"]] = identity
        if len(matches) > 1:
            raise ModelError("Drive 中同类别和相对路径有多个文件，请在模型库绑定具体文件 ID")
        return next(iter(matches.values()), None)

    def plan(self, entries, catalog):
        dep = entries[0]
        category, name = dep["category"], relative_path(dep["name"])
        hashes = {item["sha256"] for item in entries if item.get("sha256")}
        sizes = {item["size"] for item in entries if item.get("size")}
        urls = {item["url"] for item in entries if item.get("url")}
        drive_ids = {item["drive_file_id"] for item in entries if item.get("drive_file_id")}
        md5s = {item["md5"] for item in entries if item.get("md5")}
        if len(hashes) > 1 or len(sizes) > 1 or len(drive_ids) > 1 or len(md5s) > 1:
            raise ModelError("同一路径声明了不同模型身份，请指定唯一版本")
        digest, size = next(iter(hashes), None), next(iter(sizes), None)
        stem_match = any(item.get("embedding") or item.get("match_stem") for item in entries)
        match_stem_only = any(item.get("match_stem") for item in entries)
        name_stem = deps._strip_weight_ext(name) if stem_match else None
        matches = []
        for row in catalog:
            row_category, paths = registered_paths(row)
            if row_category != category:
                continue
            alias = row.get("drive_file_id") and row["drive_file_id"] + "-" + row["filename"]
            if (
                name in paths
                or (alias and name == alias)
                or (
                    stem_match
                    and any(
                        deps._strip_weight_ext(p) == name_stem
                        or (match_stem_only and "/" not in name_stem and deps._strip_weight_ext(Path(p).name) == name_stem)
                        for p in paths
                    )
                )
            ):
                matches.append(row)
        candidates = [row for row in matches if (not digest or row.get("sha256") in {digest, None})
                      and (not size or row.get("size") in {size, None})]
        if matches and not candidates and not drive_ids and not (digest and urls):
            raise ModelError("工作流声明的身份与已登记版本冲突，请核对 SHA-256、大小和路径")
        verified = [row for row in candidates if (row.get("source") or {}).get("status") == "verified"]
        if len(candidates) > 1 and digest:
            exact = [row for row in candidates if row.get("sha256") == digest]
            if exact:
                candidates = exact
                verified = [row for row in exact if (row.get("source") or {}).get("status") == "verified"]
        if len(candidates) > 1:
            identities = {row.get("sha256") for row in candidates}
            if len(identities) != 1 or None in identities:
                raise ModelError("同类别和路径存在多个版本，请补充 SHA-256 或使用不同相对路径")
        bindings = [row["source"] for row in verified or candidates if row.get("source")]
        sources = {binding.get("download_url") for binding in bindings} - {None}
        if len(sources) > 1 or (not verified and (len(urls | sources) > 1 or any(e.get("conflict") for e in entries))):
            raise ModelError("工作流与模型库的来源冲突，请核对并绑定已验证的来源")
        url = next(iter(sources or urls), None)
        source_kind = next((binding.get("kind") for binding in bindings
                            if url and binding.get("download_url") == url and binding.get("kind")), None)
        drive = next((r for r in candidates if r.get("drive_file_id") and r.get("account") in self.drive_auth), None)
        if not drive:
            drive = next((r for r in candidates if r.get("drive_file_id")), None)
        if drive_ids:
            fid = next(iter(drive_ids))
            matched_drive = next((r for r in candidates if r.get("drive_file_id") == fid), None)
            entry_account = next((e.get("account") for e in entries if e.get("account")), None)
            drive = {
                "drive_file_id": fid,
                "sha256": digest,
                "account": (matched_drive or {}).get("account") or entry_account or self.config.get("drive_account"),
            }
        row = next(iter(verified or candidates), {})
        digest = digest or row.get("sha256") or (drive or {}).get("sha256")
        size = size or row.get("size") or (drive or {}).get("size")
        if stem_match and row:
            _, row_paths = registered_paths(row)
            if name not in row_paths:
                exact_stem = next((p for p in sorted(row_paths) if deps._strip_weight_ext(p) == name_stem), None)
                if exact_stem is None and match_stem_only and "/" not in name_stem:
                    exact_stem = next(
                        (p for p in sorted(row_paths) if deps._strip_weight_ext(Path(p).name) == name_stem),
                        None,
                    )
                if exact_stem is not None:
                    name = exact_stem
                elif not name.lower().endswith(deps._STEM_WEIGHT_EXTS):
                    name += Path(row["filename"]).suffix
        target = self.target(category, name)
        directories = [deps.directories()[category], *(alias for alias, canonical in ALIASES.items() if canonical == category)]
        names = [name]
        if stem_match and not name.lower().endswith(deps._STEM_WEIGHT_EXTS):
            names += [name + suffix for suffix in (".safetensors", ".pt", ".bin", ".ckpt")]
        available = {self.target(category, candidate, directory=directory)
                     for directory in directories for candidate in names
                     if (self.model_dir / directory / candidate).is_file()}
        if len(available) > 1:
            raise ModelError("本地有多个模型匹配该引用，请核对目录别名和版本")
        target = next(iter(available), target)
        md5 = next(iter(md5s), None) or (drive or {}).get("md5")
        return target, {"url": url, "source_kind": source_kind, "drive": drive, "sha256": digest, "size": size, "md5": md5}

    def stamp(self, target):
        stat = target.stat()
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)

    def local_hit(self, target, plan):
        if not target.exists():
            return False
        if not target.is_file():
            raise ModelError("本地目标不是常规模型文件")
        stamp = self.stamp(target)
        size = stamp[2]
        with self._state_lock:
            known = self.hashes.get(target)
        if not known or known[0] != stamp:
            if plan["sha256"] or plan["md5"]:
                sha, md5 = self.file_hashes(target)
                if self.stamp(target) != stamp:
                    raise ModelError("本地文件仍在写入，请完成后重试")
                known = stamp, sha, None, md5
                with self._state_lock:
                    self.hashes[target] = known
                self._save_hash_cache()
            else:
                known = stamp, None, None, None
        provenance_matches = known[2] == plan["url"]
        if plan["url"] and not plan["sha256"] and not provenance_matches:
            key = hashlib.sha256(str(target).encode()).hexdigest()
            try:
                record = json.loads((self.partials / (key + ".json")).read_text())
                provenance_matches = (record["stamp"] == list(stamp)
                                      and record["digest"] == hashlib.sha256(plan["url"].encode()).hexdigest())
            except (OSError, ValueError, KeyError, TypeError):
                pass
        if (size <= 0 or (plan["size"] and size != plan["size"])
                or (plan["sha256"] and known[1] != plan["sha256"])
                or (plan["md5"] and known[3] != plan["md5"])
                or (plan["url"] and not plan["sha256"] and not provenance_matches)):
            raise ModelError("本地同路径文件身份无法匹配，请核对版本或为所需模型使用不同路径")
        if not known[1] and target.suffix.lower() == ".safetensors":
            self.check_unpinned_format(target, ".safetensors", size)
            if self.stamp(target) != stamp:
                raise ModelError("本地文件仍在写入，请完成后重试")
        return True

    @staticmethod
    def file_hashes(path):
        sha = hashlib.sha256()
        md5 = hashlib.md5(usedforsecurity=False)
        with path.open("rb") as source:
            for block in iter(lambda: source.read(CHUNK), b""):
                sha.update(block)
                md5.update(block)
        return sha.hexdigest(), md5.hexdigest()

    def ensure_space(self, needed, *, scratch: Path | None = None):
        with self._state_lock:
            other_reserved = sum(
                amount for path, amount in self._reserved_bytes.items() if path != scratch
            )
            free = shutil.disk_usage(self.partials).free
            if free - other_reserved < RESERVE + needed:
                raise DiskSpaceError()
            if scratch is not None:
                self._reserved_bytes[scratch] = max(0, int(needed))

    def _release_reserved(self, scratch: Path) -> None:
        with self._state_lock:
            self._reserved_bytes.pop(scratch, None)

    def transfer(self, url, target, plan, update, *, drive_token=None):
        key = hashlib.sha256(str(target).encode()).hexdigest()
        scratch = self.partials / (key + ".partial")
        # Restart only within a transfer unless we can prove the partial's identity.
        scratch.unlink(missing_ok=True)
        source = SourceHTTP(self.tokens, client=self.client)
        etag = resource = None
        written = 0
        total = plan["size"]
        sha_hasher = hashlib.sha256()
        md5_hasher = hashlib.md5(usedforsecurity=False)
        hasher_aligned = True
        try:
            self.ensure_space(total or CHUNK, scratch=scratch)
            deadline = time.monotonic() + 3600
            for attempt in range(3):
                offset = written if etag else 0
                if offset == 0:
                    sha_hasher = hashlib.sha256()
                    md5_hasher = hashlib.md5(usedforsecurity=False)
                    hasher_aligned = True
                try:
                    with source.stream(url, offset=offset, validator=etag, drive_token=drive_token) as response:
                        if response.headers.get("content-type", "").split(";")[0] in {"text/html", "application/json", "text/plain"}:
                            raise ModelError("来源未返回模型文件，请填写具体文件下载地址")
                        if response.headers.get("content-encoding", "identity").lower() != "identity":
                            raise ModelError("来源返回了不受支持的压缩传输")
                        current_etag = response.headers.get("etag")
                        current_resource = response.extensions.get("studio_source_resource")
                        if response.status_code == 206:
                            bounds = RANGE.fullmatch(response.headers.get("content-range", ""))
                            if (not bounds or int(bounds[1]) != offset or int(bounds[2]) + 1 != int(bounds[3])
                                    or current_etag != etag or current_resource != resource):
                                etag = None
                                raise ModelError("来源续传身份已变化，重新下载")
                            declared = int(bounds[3])
                        elif response.status_code == 200:
                            offset = written = 0
                            sha_hasher = hashlib.sha256()
                            md5_hasher = hashlib.md5(usedforsecurity=False)
                            hasher_aligned = True
                            declared = int(response.headers.get("content-length") or 0)
                        else:
                            etag = None
                            raise ModelError("来源不接受续传，重新下载")
                        expected = plan["size"]
                        if expected and declared and expected != declared:
                            raise ModelError("下载文件大小与登记值不符")
                        total = expected or declared
                        if not total or offset >= total:
                            raise ModelError("来源未提供可校验的完整文件大小")
                        etag = current_etag if current_etag and current_etag.startswith('"') else None
                        resource = current_resource
                        if offset == 0:
                            scratch.unlink(missing_ok=True)
                        self.ensure_space(total - offset, scratch=scratch)
                        next_space_check = written + 32 * CHUNK
                        with scratch.open("ab" if offset else "wb") as output:
                            for chunk in response.iter_raw(CHUNK):
                                if time.monotonic() >= deadline:
                                    raise ModelError("单个模型下载超过 60 分钟，请检查来源或改用 Drive 挂载")
                                if written + len(chunk) > total:
                                    raise ModelError("下载内容超过声明的文件大小")
                                if written >= next_space_check:
                                    self.ensure_space(max(len(chunk), total - written), scratch=scratch)
                                    next_space_check = written + 32 * CHUNK
                                output.write(chunk)
                                if hasher_aligned:
                                    sha_hasher.update(chunk)
                                    md5_hasher.update(chunk)
                                written += len(chunk)
                                update(bytes=written, total=total, attempt=attempt + 1)
                            output.flush()
                            os.fsync(output.fileno())
                    if written == total:
                        break
                    raise ModelError("模型下载中断")
                except DiskSpaceError:
                    raise
                except (ModelError, httpx.HTTPError, ValueError) as exc:
                    if attempt == 2:
                        raise ModelError(str(exc) if isinstance(exc, ModelError) else "模型下载连接中断") from None
                    if written == total:
                        etag = None
                    time.sleep(1 + attempt)
            if hasher_aligned and written == total:
                digest, md5 = sha_hasher.hexdigest(), md5_hasher.hexdigest()
            else:
                digest, md5 = self.file_hashes(scratch)
            if plan["sha256"] and digest != plan["sha256"]:
                raise ModelError("下载文件 SHA-256 不匹配，请核对版本和来源")
            target_category = category_for(target.relative_to(self.model_dir).parts[0])
            if is_controlled_upscale_extension(target.name):
                validate_weight_file(scratch, target.name, written, model_type=target_category)
            elif not plan["sha256"] and not plan["md5"]:
                self.check_unpinned_format(scratch, target.suffix.lower(), written)
            if plan["md5"] and md5 != plan["md5"]:
                raise ModelError("Drive 模型 MD5 不匹配，请检查文件是否已变更")
            # Atomic publication with no replacement of imported/loaded weights.
            try:
                self.publish(scratch, target)
            except FileExistsError:
                if not self.local_hit(target, plan):
                    raise ModelError("同路径已被其他写入占用，请核对模型文件") from None
            else:
                scratch.unlink()
                target_stamp = self.stamp(target)
                with self._state_lock:
                    self.hashes[target] = target_stamp, digest, plan["url"], md5
                self._save_hash_cache()
                if plan["url"] and not plan["sha256"]:
                    # Only a URL fingerprint is persisted. This lets a ComfyUI process
                    # restart reuse a complete URL-only file on the same instance.
                    provenance = self.partials / (key + ".json")
                    try:
                        with provenance.with_suffix(".tmp").open("w") as record:
                            json.dump({"stamp": target_stamp, "digest": hashlib.sha256(plan["url"].encode()).hexdigest()}, record)
                        provenance.with_suffix(".tmp").replace(provenance)
                    except OSError:
                        LOG.warning("Could not save model provenance for process restart")
        finally:
            self._release_reserved(scratch)
            scratch.unlink(missing_ok=True)

    def publish(self, scratch, target):
        # Dirfds prevent a symlink swap between path validation and publication.
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        self.model_dir.mkdir(exist_ok=True)
        parent = os.open(self.model_dir, flags)
        try:
            for component in target.relative_to(self.model_dir).parts[:-1]:
                try:
                    os.mkdir(component, dir_fd=parent)
                except FileExistsError:
                    pass
                child = os.open(component, flags, dir_fd=parent)
                os.close(parent)
                parent = child
            os.link(scratch, target.name, dst_dir_fd=parent, follow_symlinks=False)
        finally:
            os.close(parent)
        try:
            rel_parts = target.relative_to(self.model_dir).parts
            if rel_parts and category_for(rel_parts[0]) == "loras":
                with self._state_lock:
                    self.lora_generation += 1
        except ValueError:
            pass

    @staticmethod
    def check_unpinned_format(path, suffix, size):
        with path.open("rb") as source:
            prefix = source.read(8)
            if suffix == ".gguf" and prefix[:4] == b"GGUF":
                return
            if suffix == ".safetensors":
                try:
                    length = int.from_bytes(prefix, "little")
                    if not 0 < length <= min(size - 8, 100 * 1024**2):
                        raise ValueError
                    header = json.loads(source.read(length))
                    end = max(tensor["data_offsets"][1] for name, tensor in header.items() if name != "__metadata__")
                    if end == size - length - 8:
                        return
                except (ValueError, KeyError, IndexError, AttributeError, TypeError):
                    pass
        raise ModelError("无法校验来源文件格式；请检查下载地址或提供经验证的 SHA-256")

    def prepare_one(self, target, plan, status):
        started = time.monotonic()
        update = status.update
        try:
            if self.local_hit(target, plan):
                update(state="ready", source="local", cache_hit=True)
                self._record_model_used(target, plan)
                return
            url, drive = plan["url"], plan["drive"]
            token = None
            account = self.config.get("drive_account")
            target_category = category_for(target.relative_to(self.model_dir).parts[0])
            if url or drive:
                filename(target.name, model_type=target_category)
            if url:
                if is_controlled_upscale_extension(target.name):
                    validate_controlled_upscale_source(url, source=plan.get("source_kind"))
                update(state="downloading", source="external", host=urlsplit(url).hostname)
                try:
                    self.transfer(https_url(url), target, plan, update)
                except DiskSpaceError:
                    raise
                except ModelError as exc:
                    if not plan["sha256"]:
                        raise
                    if not drive or drive.get("sha256") != plan["sha256"]:
                        if not self.config.get("models_folder"):
                            raise
                        token = self.drive_token(account)
                        drive = self.legacy_copy(target, plan, token=token)
                        if not drive:
                            raise
                    update(fallback=str(exc), bytes=0)
                else:
                    update(state="ready", integrity="sha256" if plan["sha256"] else "transport")
                    self._record_model_used(target, plan)
                    return
            if not drive and self.config.get("models_folder"):
                token = self.drive_token(account)
                drive = self.legacy_copy(target, plan, token=token)
            if not drive:
                raise ModelError("找不到明确来源或 Drive 副本，请在模型库登记准确类别和相对路径")
            filename(target.name, model_type=target_category)
            file_id = drive["drive_file_id"]
            if not isinstance(file_id, str) or not FILE_ID.fullmatch(file_id):
                raise ModelError("Drive 文件 ID 无效")
            update(state="downloading", source="drive", host=None)
            token = token or self.drive_token(drive.get("account", account))
            if not (plan["sha256"] or plan["md5"] or drive.get("sha256") or drive.get("md5")):
                meta = self.drive_metadata(file_id, token)
                drive = {**drive, **meta}
                if plan["size"] and drive["size"] != plan["size"]:
                    raise ModelError("Drive 文件大小与记录不符，请重新校验模型库")
            plan = {**plan, "drive": drive, **{key: plan[key] or drive.get(key) for key in ("sha256", "size", "md5")}}
            self.transfer(f"https://www.googleapis.com/drive/v3/files/{quote(file_id)}?alt=media",
                          target, plan, update, drive_token=token)
            update(state="ready", integrity="sha256" if plan["sha256"] else "drive_md5")
            self._record_model_used(target, plan)
        except DiskSpaceError as exc:
            update(state="failed", error=str(exc), disk_space_limit=True)
        except ModelError as exc:
            update(state="failed", error=str(exc))
        except OSError as exc:
            if exc.errno == errno.ENOSPC:
                update(state="failed", error=str(DiskSpaceError()), disk_space_limit=True)
            else:
                update(state="failed", error="无法写入 Colab 本地模型目录")
        except Exception:
            # Underlying exceptions can contain signed URLs, tokens or provider responses.
            update(state="failed", error="准备模型失败，请检查网络、Drive 授权和 Colab 本地磁盘")
        finally:
            update(seconds=round(time.monotonic() - started, 2))
            LOG.info("model preparation %s", json.dumps(status, ensure_ascii=False))

    async def prepare(self, body, explicit, catalog, *, visual=False, trusted=False):
        # A separate process can test the lock even when the gateway is disconnected.
        async with self._activity_lock(exclusive=False) as lock_hold:
            rows = {}
            tasks = []
            grouped = defaultdict(list)
            try:
                for dep in dependencies(body, explicit, visual=visual, trusted=trusted, node_types=self.node_types):
                    name = relative_path(dep["name"])
                    grouped[(dep["category"], name)].append(dep)
                    if len(grouped) > 256:
                        raise ModelError("每个工作流最多检查 256 个模型路径")
            except (ModelError, TypeError, ValueError, KeyError, AttributeError):
                raise ModelError("工作流模型声明或文件路径无效") from None
            planned: list[tuple[dict, Path, dict, tuple]] = []
            for (category, name), entries in grouped.items():
                status = rows[(category, name)] = {"category": category, "name": name, "state": "pending"}
                try:
                    target, plan = self.plan(entries, catalog)
                    identity = (plan["sha256"], plan["md5"], plan["size"], None if plan["sha256"] else plan["url"],
                                (plan["drive"] or {}).get("drive_file_id") if not plan["sha256"] else None)
                    planned.append((status, target, plan, identity))
                except ModelError as exc:
                    status.update(state="failed", error=str(exc))
            # Start largest model weights first so multi-GB downloads overlap with smaller CLIP/VAE weights.
            planned.sort(key=lambda item: -(int(item[2].get("size") or 0)))
            for status, target, plan, identity in planned:
                try:
                    current = self.inflight.get(target)
                    if current and current[0] != identity:
                        raise ModelError("该路径正在准备另一版本，请等候并核对模型身份")
                    if not current:
                        task = asyncio.get_running_loop().run_in_executor(self.executor, self.prepare_one, target, plan, status)
                        self.inflight[target] = (identity, task, status)
                        task.add_done_callback(lambda _, path=target: self.inflight.pop(path, None))
                        lock_hold.track(task)
                        current = self.inflight[target]
                    else:
                        lock_hold.track(current[1])
                    tasks.append((current[1], status, current[2]))
                except ModelError as exc:
                    status.update(state="failed", error=str(exc))
            self.current_rows[asyncio.current_task()] = rows, tasks
            try:
                await asyncio.gather(*(asyncio.shield(task) for task, _, _ in tasks))
                return self.progress(rows, tasks)
            finally:
                self.current_rows.pop(asyncio.current_task(), None)


    @staticmethod
    def progress(rows, tasks):
        copied = {}
        for _, status, shared in tasks:
            if status.get("state") == "skipped_disk_limit":
                copied[id(status)] = dict(status)
            else:
                copied[id(status)] = dict(shared, category=status["category"], name=status["name"])
        return [copied.get(id(status), dict(status)) for status in rows.values()]

    async def prepare_studio(self, weights, loras, cache_payload, verify_paths, paths, timeout):
        # Keep the existing native activity fence and serialize with native prepares off the event loop.
        async with self._activity_lock(exclusive=True) as lock_hold:
            rows = {}
            for target in [*(item["target"] for item in weights), *("loras/" + item["target"] for item in loras)]:
                category, _, name = target.partition("/")
                rows[(category, name)] = {"category": category, "name": name, "state": "preparing"}
            task = asyncio.current_task()
            self.current_rows[task] = rows, []

            def restore():
                try:
                    paths.update(verify_paths(rows, require_files=False))
                    cache = studio_worker._drive_cache(cache_payload)
                    models = studio_worker._restore_weights(
                        self.model_dir.parent, {"weight_files": weights}, timeout, cache, prep=False,
                        preserve_existing=True, use_drive_cache=cache_payload["use_drive_cache"], paths=paths,
                    )
                    models.extend(studio_worker._restore_user_loras(
                        self.model_dir.parent, loras, studio_worker._lora_drive_cache(cache_payload),
                        preserve_existing=True, paths=paths,
                    ))
                    if verify_paths(rows, require_files=True) != paths:
                        raise ModelError("原生 ComfyUI 的模型搜索路径在准备期间发生变化")
                except studio_worker.WorkerError as exc:
                    raise ModelError(str(exc)) from exc
                for record in models:
                    category, _, name = record["target"].partition("/")
                    status = {**record, "category": category, "name": name, "state": "ready"}
                    rows[(category, name)] = status
                    LOG.info("Studio model preparation %s", json.dumps(status, ensure_ascii=False))
                return list(rows.values())

            try:
                future = asyncio.get_running_loop().run_in_executor(self.executor, restore)
                lock_hold.track(future)
                return await asyncio.shield(future)
            finally:
                self.current_rows.pop(task, None)

    async def restore(self, models_list, catalog, *, refresh_fn=None):
        """Background full restore for native local mode in last_used_at desc order (§6.1(4))."""
        if not isinstance(models_list, list) or len(models_list) > 256:
            raise ModelError("模型恢复列表必须是至多 256 项的数组")
        ordered = sorted(
            [m for m in models_list if isinstance(m, dict) and m.get("category") and m.get("name")],
            key=lambda item: (
                -float(item.get("last_used_at") or 0.0),
                str(item.get("category") or ""),
                str(item.get("name") or ""),
            ),
        )
        async with self._activity_lock(exclusive=False) as lock_hold:
            rows = {}
            tasks = []
            items: list[tuple[str, str, dict, dict]] = []
            for item in ordered:
                try:
                    cat = category_for(str(item["category"]))
                    name = relative_path(str(item["name"]))
                except (ModelError, TypeError, ValueError):
                    continue
                if (cat, name) in rows:
                    continue
                status = rows[(cat, name)] = {"category": cat, "name": name, "state": "pending"}
                items.append((cat, name, item, status))
            self.current_rows[asyncio.current_task()] = rows, tasks
            disk_limit_hit = False
            try:
                for cat, name, item, status in items:
                    if disk_limit_hit:
                        status.update(state="skipped_disk_limit", error=str(DiskSpaceError()), disk_space_limit=True)
                        continue
                    try:
                        src = item.get("source") if isinstance(item.get("source"), dict) else {}
                        raw_url = src.get("download_url") or item.get("url")
                        has_catalog_source = False
                        for crow in catalog:
                            if not isinstance(crow, dict):
                                continue
                            try:
                                crow_cat, crow_paths = registered_paths(crow)
                            except (KeyError, ModelError, TypeError):
                                continue
                            if crow_cat == cat and name in crow_paths and isinstance(crow.get("source"), dict) and crow["source"].get("download_url"):
                                has_catalog_source = True
                                break
                        entry = {
                            "category": cat,
                            "name": name,
                            "url": https_url(raw_url) if (raw_url and not has_catalog_source) else None,
                            "sha256": item.get("sha256"),
                            "size": item.get("size"),
                            "md5": item.get("md5"),
                            "drive_file_id": item.get("drive_file_id"),
                            "account": item.get("account"),
                        }
                        target, plan = self.plan([entry], catalog)
                        if not plan.get("source_kind") and isinstance(src.get("kind"), str) and src.get("kind"):
                            plan["source_kind"] = src["kind"]
                        if item.get("account") and isinstance(plan.get("drive"), dict) and not plan["drive"].get("account"):
                            plan["drive"] = {**plan["drive"], "account": item["account"]}
                        identity = (
                            plan["sha256"],
                            plan["md5"],
                            plan["size"],
                            None if plan["sha256"] else plan["url"],
                            (plan["drive"] or {}).get("drive_file_id") if not plan["sha256"] else None,
                        )
                        current = self.inflight.get(target)
                        if current and current[0] != identity:
                            raise ModelError("该路径正在准备另一版本，请等候并核对模型身份")
                        if not current:
                            task = asyncio.get_running_loop().run_in_executor(
                                self.executor, self.prepare_one, target, plan, status
                            )
                            self.inflight[target] = (identity, task, status)
                            task.add_done_callback(lambda _, path=target: self.inflight.pop(path, None))
                            lock_hold.track(task)
                            current = self.inflight[target]
                        else:
                            lock_hold.track(current[1])
                        tasks.append((current[1], status, current[2]))
                        await asyncio.shield(current[1])
                        shared = current[2]
                        if shared.get("disk_space_limit") or "5 GiB" in str(shared.get("error") or ""):
                            disk_limit_hit = True
                            status.update(state="skipped_disk_limit", error=shared.get("error") or str(DiskSpaceError()), disk_space_limit=True)
                        elif shared.get("state") == "ready" and refresh_fn is not None:
                            try:
                                refresh_fn()
                            except Exception:
                                pass
                    except DiskSpaceError as exc:
                        disk_limit_hit = True
                        status.update(state="skipped_disk_limit", error=str(exc), disk_space_limit=True)
                    except ModelError as exc:
                        status.update(state="failed", error=str(exc))
                return self.progress(rows, tasks)
            finally:
                self.current_rows.pop(asyncio.current_task(), None)

    def submit_restore(self, models_list, *, refresh_fn=None):
        expiry = time.monotonic() - 3600
        self.jobs = {k: v for k, v in self.jobs.items() if not v[0].done() or v[1] > expiry}
        if len(self.jobs) >= 256:
            oldest = next((key for key, (task, _) in self.jobs.items() if task.done()), None)
            if oldest:
                del self.jobs[oldest]
        if sum(not task.done() for task, _ in self.jobs.values()) >= 16:
            raise ModelError("模型准备请求过多，请等待当前下载完成")
        ident = uuid.uuid4().hex
        task = asyncio.create_task(self.restore(models_list, self.catalog, refresh_fn=refresh_fn))
        task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        self.jobs[ident] = task, time.monotonic()
        return ident, task

    def submit(self, body, explicit=None, *, visual=False, trusted=False, studio=None):
        expiry = time.monotonic() - 3600
        self.jobs = {k: v for k, v in self.jobs.items() if not v[0].done() or v[1] > expiry}
        if len(self.jobs) >= 256:
            oldest = next((key for key, (task, _) in self.jobs.items() if task.done()), None)
            if oldest:
                del self.jobs[oldest]
        if sum(not task.done() for task, _ in self.jobs.values()) >= 16:
            raise ModelError("模型准备请求过多，请等待当前下载完成")
        ident = uuid.uuid4().hex
        work = (self.prepare_studio(*studio) if studio is not None else
                self.prepare(body, [] if explicit is None else explicit, self.catalog, visual=visual, trusted=trusted))
        task = asyncio.create_task(work)
        # Retrieve exceptions even if the client disappears.
        task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        self.jobs[ident] = task, time.monotonic()
        return ident, task

    def status(self, ident):
        existing = self.jobs.get(ident)
        if not existing:
            return None
        task, _ = existing
        if task.done():
            try:
                models = task.result()
                partial_disk = any(
                    m.get("state") == "skipped_disk_limit" or m.get("disk_space_limit")
                    for m in models
                )
                if partial_disk:
                    state = "partial_disk_limit"
                elif any(m["state"] == "failed" for m in models):
                    state = "failed"
                else:
                    state = "ready"
                return {"state": state, "partial_disk_limit": partial_disk, "models": models}
            except ModelError as exc:
                return {"state": "failed", "partial_disk_limit": False, "message": str(exc), "models": []}
        models = self.progress(*self.current_rows.get(task, ({}, [])))
        partial_disk = any(m.get("state") == "skipped_disk_limit" or m.get("disk_space_limit") for m in models)
        return {"state": "preparing", "partial_disk_limit": partial_disk, "models": models}

    def names(self, category, extensions=None):
        names = set()
        for row in self.catalog:
            row_category, paths = registered_paths(row)
            if row_category != category_for(category):
                continue
            if row.get("source") or (row.get("drive_file_id") and row.get("account") in self.drive_auth):
                names.update(paths)
                if row.get("drive_file_id"):
                    names.add(row["drive_file_id"] + "-" + row["filename"])
        return sorted(name for name in names if not extensions or Path(name).suffix.lower() in extensions)

    def _workflow_base(self, source):
        if source not in WORKFLOW_SOURCES:
            raise ModelError("工作流来源类型无效")
        drive_root = self.root / "drive"
        current = drive_root
        if current.is_symlink():
            raise ModelError("Drive 挂载根目录异常")
        for segment in WORKFLOW_SOURCES[source]:
            current = current / segment
            if current.is_symlink():
                raise ModelError("工作流目录含软链接，拒绝访问")
        return current

    def _resolve_workflow_file(self, source, raw_path, *, ensure_json_ext=False, create_parents=False):
        rel = workflow_relpath(raw_path, ensure_json_ext=ensure_json_ext)
        base = self._workflow_base(source)
        expected_base = self.root.resolve() / "drive" / Path(*WORKFLOW_SOURCES[source])
        if create_parents:
            base.mkdir(parents=True, exist_ok=True)
        if base.exists() and base.resolve() != expected_base:
            raise ModelError("工作流目录路径校验失败")
        current = base
        parts = rel.split("/")
        for folder in parts[:-1]:
            current = current / folder
            if current.is_symlink():
                raise ModelError("工作流子目录含软链接，拒绝访问")
            if create_parents:
                current.mkdir(exist_ok=True)
        target = current / parts[-1]
        if target.is_symlink():
            raise ModelError("工作流文件为软链接，拒绝访问")
        if base.exists() and not target.resolve().is_relative_to(expected_base):
            raise ModelError("工作流路径越界")
        return rel, target

    def _refresh_workflows_vfs(self):
        if not (self.root / "rclone.conf").is_file():
            return
        from urllib.request import Request, urlopen

        for parts in WORKFLOW_SOURCES.values():
            rel_dir = "/".join(parts)
            try:
                payload = json.dumps({"dir": rel_dir, "recursive": "true"}).encode()
                req = Request("http://127.0.0.1:5572/vfs/refresh", data=payload,
                              headers={"Content-Type": "application/json"})
                with urlopen(req, timeout=10) as resp:
                    resp.read()
            except Exception:
                # An uncreated directory on Drive does not prevent local directory scans.
                continue

    def list_workflows(self, *, refresh=False):
        if refresh:
            self._refresh_workflows_vfs()
        items = []
        for source in ("comfy", "warmup"):
            base = self._workflow_base(source)
            if not base.is_dir():
                continue
            expected_base = self.root.resolve() / "drive" / Path(*WORKFLOW_SOURCES[source])
            if base.resolve() != expected_base:
                continue
            for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
                current_dir = Path(dirpath)
                dirnames[:] = sorted(
                    d for d in dirnames
                    if not d.startswith(".") and not (current_dir / d).is_symlink()
                )
                for fname in sorted(filenames):
                    if fname.startswith(".") or not fname.lower().endswith(".json"):
                        continue
                    fpath = current_dir / fname
                    if fpath.is_symlink():
                        continue
                    try:
                        stat = fpath.stat()
                        rel = fpath.relative_to(base).as_posix()
                        workflow_relpath(rel)
                    except (OSError, ValueError, ModelError):
                        continue
                    if not (0 < stat.st_size <= MAX_WORKFLOW_FILE_BYTES):
                        continue
                    items.append({
                        "source": source,
                        "path": rel,
                        "size": stat.st_size,
                        "modified_at": round(stat.st_mtime, 3),
                    })
                    if len(items) >= MAX_WORKFLOW_LIST_ITEMS:
                        break
                if len(items) >= MAX_WORKFLOW_LIST_ITEMS:
                    break
        items.sort(key=lambda item: (-item["modified_at"], item["source"] != "comfy", item["path"]))
        return items

    def read_workflow(self, source, raw_path):
        _, target = self._resolve_workflow_file(source, raw_path)
        if not target.is_file():
            raise WorkflowNotFoundError(f"Drive 中找不到工作流：{raw_path}")
        try:
            size = target.stat().st_size
            if size <= 0 or size > MAX_WORKFLOW_FILE_BYTES:
                raise ModelError("工作流文件大小超出允许范围（最大 16 MiB）")
            parsed = json.loads(target.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            raise ModelError("工作流文件不是有效的 UTF-8 文本") from None
        except json.JSONDecodeError:
            raise ModelError("工作流文件内容不是有效的 JSON") from None
        except OSError:
            raise ModelError("无法读取 Drive 工作流文件，请稍后重试") from None
        return wrap_visual_workflow(parsed)

    def save_workflow(self, raw_name, workflow, *, overwrite=False):
        wrapped = wrap_visual_workflow(workflow)
        visual = wrapped.get("workflow")
        if visual is not None and not isinstance(visual.get("nodes"), list):
            raise ModelError("原生工作流必须包含 nodes 数组")
        # The sidebar reads canvas JSON directly. Keep API graphs and mixed
        # bundles intact for warmup, where their prompt can drive model detection.
        source = "comfy" if visual is not None and "prompt" not in wrapped else "warmup"
        document = visual if source == "comfy" else workflow
        try:
            raw = json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
        except (ValueError, UnicodeError, RecursionError):
            raise ModelError("工作流包含无法保存的 JSON 值") from None
        if len(raw) > MAX_WORKFLOW_FILE_BYTES:
            raise ModelError("工作流文件不能超过 16 MiB")
        try:
            # Serialize Studio writers on the local disk. The Drive mount
            # provides rename for publishing the complete file.
            with (self.root / "workflow-save.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                rel, target = self._resolve_workflow_file(source, raw_name, ensure_json_ext=True, create_parents=True)
                if target.exists() and not overwrite:
                    raise WorkflowConflictError(f"同名工作流「{rel}」已存在，请确认覆盖或更换名称")
                scratch = target.parent / f".studio-workflow-{uuid.uuid4().hex}.tmp"
                try:
                    with scratch.open("xb") as handle:
                        handle.write(raw)
                        handle.flush()
                        os.fsync(handle.fileno())
                    scratch.replace(target)
                    stat = target.stat()
                finally:
                    scratch.unlink(missing_ok=True)
        except OSError:
            raise ModelError("无法写入 Drive 工作流目录，请检查磁盘与挂载状态") from None
        return {"source": source, "path": rel, "size": stat.st_size, "modified_at": round(stat.st_mtime, 3)}

    def delete_workflow(self, source, raw_path):
        if source != "warmup":
            raise ModelError("仅允许在 Studio 移除自定义保存的 API 预热工作流；原生工作流请在 ComfyUI 中管理")
        rel, target = self._resolve_workflow_file(source, raw_path)
        if not target.is_file():
            raise WorkflowNotFoundError(f"工作流不存在或已移除：{rel}")
        try:
            target.unlink()
        except OSError:
            raise ModelError("无法删除 Drive 工作流文件，请稍后重试") from None
        return {"deleted": True, "source": source, "path": rel}


def graph_key(body):
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()


def install(server, root, folder_paths, node_types):
    from aiohttp import web

    config = json.loads((root / "local-model-config.json").read_text())
    for setting, key in (("MODEL_LIBRARY_DIRECTORIES", "directories"), ("MODEL_DEPENDENCY_RULES", "rules")):
        if key in config:
            os.environ[setting] = json.dumps(config[key])
    local = LocalModels(root, Path(folder_paths.models_dir), config, node_types=node_types)
    for category, directory in deps.directories().items():
        category = category_for(category)
        path = local.model_dir / directory
        if path.resolve() != path:
            raise ModelError("模型目录配置含软链接")
        if category in folder_paths.folder_names_and_paths:
            folder_paths.add_model_folder_path(category, str(path), is_default=True)
    completed = {}

    def verify_studio_paths(references, *, require_files):
        models = studio_worker._local_models(local.model_dir.parent)
        if models != local.model_dir:
            raise ModelError("原生 ComfyUI 的模型根目录与创作模式不兼容")
        # ComfyUI adds these output paths for its model-saving nodes. Preserve
        # native I/O and allow them only behind every Studio model's local path.
        output = Path(folder_paths.get_output_directory())
        output_folders = {"checkpoints": "checkpoints", "text_encoders": "clip", "vae": "vae",
                          "diffusion_models": "diffusion_models", "loras": "loras"}
        for category, (paths, _) in folder_paths.folder_names_and_paths.items():
            if category in {"custom_nodes", "datasets"}:
                continue
            for folder in map(Path, paths):
                if category in output_folders and folder == output / output_folders[category]:
                    continue
                if not folder.is_relative_to(models) or folder.resolve() != folder:
                    raise ModelError(f"原生 ComfyUI 的 {category} 搜索路径不在 Colab 本地模型目录内")
        selected = {}
        for category, name in references:
            mapped = category_for(category)
            mapped = getattr(folder_paths, "map_legacy", category_for)(mapped)
            directory = models / category
            target = directory / name
            paths = folder_paths.folder_names_and_paths.get(mapped, ([], set()))[0]
            if str(directory) not in paths or not name or target.resolve() != target or not target.is_relative_to(directory):
                raise ModelError(f"原生 ComfyUI 的搜索路径与创作模型位置不兼容：{category}/{name}")
            for folder in map(Path, paths[:paths.index(str(directory))]):
                if not folder.is_relative_to(models):
                    raise ModelError(f"原生 ComfyUI 有优先于创作本地模型的挂载搜索路径：{category}/{name}")
                if (folder / name).is_file():
                    target = folder / name
                    break
            if require_files and (not target.is_file() or folder_paths.get_full_path(mapped, name) != str(target)):
                raise ModelError(f"原生 ComfyUI 未解析到创作模式的本地文件：{category}/{name}")
            selected[f"{category}/{name}"] = target
        if len(set(selected.values())) != len(selected):
            raise ModelError("多个创作模型绑定到了原生 ComfyUI 的同一文件")
        return selected

    synced_lora_generation = 0

    def refresh():
        folder_paths.filename_list_cache.clear()
        helper = getattr(folder_paths, "cache_helper", None)
        if helper:
            helper.clear()

    async def refresh_lora_manager():
        nonlocal synced_lora_generation
        target_generation = local.lora_generation
        if target_generation == synced_lora_generation:
            return
        for name, mod in sys.modules.copy().items():
            if mod is None or not name.casefold().endswith("comfyui-lora-manager.py.services.service_registry"):
                continue
            try:
                registry = vars(mod).get("ServiceRegistry")
                getter = getattr(registry, "get_lora_scanner", None)
                if callable(getter):
                    scanner = await getter()
                    if scanner is not None and callable(getattr(scanner, "get_cached_data", None)):
                        await scanner.get_cached_data(force_refresh=True)
                        synced_lora_generation = target_generation
            except Exception:
                LOG.warning("Failed to refresh LoRA Manager cache after model preparation", exc_info=True)
            return
        synced_lora_generation = target_generation

    async def json_body(request):
        try:
            body = await request.json()
            if isinstance(body, dict):
                return body
        except (ValueError, UnicodeError):
            pass
        raise ModelError("工作流请求必须是有效的 JSON 对象")

    def sync(body):
        with local._state_lock:
            local.catalog = body["catalog"]
            if "tokens" in body:
                local.tokens = body["tokens"]
            if "drive_auth" in body:
                local.drive_auth = body["drive_auth"]

    @server.routes.post("/studio_native/models/catalog")
    async def model_catalog(request):
        if not local.signed(request):
            raise web.HTTPNotFound()
        try:
            sync(await json_body(request))
        except (ModelError, KeyError, TypeError) as exc:
            return web.json_response(model_problem(str(exc)), status=422)
        return web.json_response({})

    @server.routes.get("/studio_native/models/inventory")
    async def model_inventory(request):
        if not local.signed(request):
            raise web.HTTPNotFound()
        try:
            inv = await asyncio.to_thread(local.scan_inventory)
        except ModelError as exc:
            return web.json_response(model_problem(str(exc)), status=422)
        return web.json_response(inv)

    @server.routes.post("/studio_native/models/restore")
    async def start_restore(request):
        if not local.signed(request):
            raise web.HTTPNotFound()
        try:
            data = await json_body(request)
            if "catalog" in data:
                sync(data)
            models_list = data.get("models")
            if not isinstance(models_list, list):
                raise ModelError("模型恢复请求缺少有效的 models 数组")
            ident, _ = local.submit_restore(models_list, refresh_fn=refresh)
            await asyncio.sleep(0)
        except ModelError as exc:
            return web.json_response(model_problem(str(exc)), status=422)
        return web.json_response({"id": ident}, status=202)

    @server.routes.get("/studio_native/models/restore/{ident}")
    async def restore_status(request):
        if not local.signed(request):
            raise web.HTTPNotFound()
        ident = request.match_info["ident"]
        if ident == "latest":
            ident = next(reversed(local.jobs), None)
        result = local.status(ident)
        if result is None:
            raise web.HTTPNotFound()
        return web.json_response({"id": ident, **result})

    @server.routes.get("/studio_native/models/workflows")
    async def list_drive_workflows(request):
        if not local.signed(request):
            raise web.HTTPNotFound()
        query = getattr(request, "query", None) or {}
        refresh_flag = query.get("refresh") in {"1", "true"}
        try:
            workflows = await asyncio.to_thread(local.list_workflows, refresh=refresh_flag)
        except ModelError as exc:
            return web.json_response(model_problem(str(exc)), status=422)
        return web.json_response({"workflows": workflows})

    @server.routes.post("/studio_native/models/workflows")
    async def save_drive_workflow(request):
        if not local.signed(request):
            raise web.HTTPNotFound()
        try:
            data = await json_body(request)
            saved = await asyncio.to_thread(
                local.save_workflow,
                data.get("name"),
                data.get("workflow"),
                overwrite=data.get("overwrite") is True,
            )
        except WorkflowConflictError as exc:
            return web.json_response(model_problem(str(exc), code="NATIVE_WORKFLOW_EXISTS"), status=409)
        except ModelError as exc:
            return web.json_response(model_problem(str(exc)), status=422)
        return web.json_response({"workflow": saved}, status=201)

    @server.routes.post("/studio_native/models/workflows/delete")
    async def delete_drive_workflow(request):
        if not local.signed(request):
            raise web.HTTPNotFound()
        try:
            data = await json_body(request)
            result = await asyncio.to_thread(local.delete_workflow, data.get("source"), data.get("path"))
        except WorkflowNotFoundError as exc:
            return web.json_response(model_problem(str(exc), code="NATIVE_WORKFLOW_NOT_FOUND"), status=404)
        except (ModelError, KeyError, TypeError) as exc:
            return web.json_response(model_problem(str(exc)), status=422)
        return web.json_response(result)

    @server.routes.post("/studio_native/models/prepare")
    async def start(request):
        if not local.signed(request):
            raise web.HTTPNotFound()
        try:
            data = await json_body(request)
            if "catalog" in data:
                sync(data)
            ref = data.get("workflow_ref")
            if ref is not None:
                if not isinstance(ref, dict):
                    raise ModelError("工作流引用格式无效")
                body = await asyncio.to_thread(local.read_workflow, ref.get("source"), ref.get("path"))
            else:
                body = data["workflow"]
            ident, _ = local.submit(body, data.get("dependencies", []), visual=data.get("visual") is True,
                                    trusted=data.get("reviewed") is True)
            await asyncio.sleep(0)  # Acquire the download fence before acknowledging the gateway.
            if not data.get("visual"):
                completed[ident] = (graph_key(body), None)
            # Bound the handoff map with the same lifetime and count as progress records.
            for old in completed.keys() - local.jobs.keys():
                del completed[old]
        except WorkflowNotFoundError as exc:
            return web.json_response(model_problem(str(exc), code="NATIVE_WORKFLOW_NOT_FOUND"), status=404)
        except (ModelError, KeyError, TypeError) as exc:
            return web.json_response(model_problem(str(exc)), status=422)
        return web.json_response({"id": ident}, status=202)

    @server.routes.post("/studio_native/models/prepare-studio")
    async def start_studio(request):
        if not local.signed(request):
            raise web.HTTPNotFound()
        try:
            data = await json_body(request)
            body, loras = data["workflow"], data["loras"]
            weights = studio_worker._validate_weights({"weight_files": data["weights"]})
            references = [tuple(relative_path(item["target"]).split("/", 1)) for item in weights]
            references.extend(("loras", relative_path(item["target"])) for item in loras)
            if any(len(reference) != 2 for reference in references):
                raise ModelError("创作模型需要明确的类别和文件路径")
            cache_payload = {"drive_cache": data.get("drive_cache"), "lora_drive_cache": data.get("lora_drive_cache"),
                             "use_drive_cache": data.get("use_drive_cache", True)}
            paths = {}
            ident, _ = local.submit(body, studio=(weights, loras, cache_payload, verify_studio_paths, paths, data["timeout"]))
            await asyncio.sleep(0)
            completed[ident] = (graph_key(body), (references, paths))
            for old in completed.keys() - local.jobs.keys():
                del completed[old]
        except (ModelError, studio_worker.WorkerError, KeyError, TypeError) as exc:
            return web.json_response(model_problem(str(exc)), status=422)
        return web.json_response({"id": ident}, status=202)

    @server.routes.get("/studio_native/models/prepare/{ident}")
    async def status(request):
        if not local.signed(request):
            raise web.HTTPNotFound()
        ident = request.match_info["ident"]
        if ident == "latest":
            ident = next(reversed(local.jobs), None)
        result = local.status(ident)
        if result is None:
            raise web.HTTPNotFound()
        return web.json_response({"id": ident, **result})

    @web.middleware
    async def prepare_prompt(request, handler):
        path = request.path.removeprefix("/api").rstrip("/")
        if request.method == "POST" and path == "/prompt":
            try:
                body = await json_body(request)
            except ModelError as exc:
                return web.json_response(model_problem(str(exc)), status=400)
            if "prompt" not in body:
                return await handler(request)
            signed = local.signed(request)
            if "X-Studio-Model-Channel" in request.headers and not signed:
                return web.json_response(model_problem("原生会话已切换，请重新准备模型"), status=409)
            ident = request.headers.get("X-Studio-Model-Prepared", "") if signed else ""
            if ident:
                key, studio = completed.get(ident, (None, None))
                if key != graph_key(body) or (local.status(ident) or {}).get("state") != "ready":
                    return web.json_response(model_problem("模型准备记录已过期或与当前工作流不符，请重新提交"), status=409)
                # One handoff authorizes one validation of the exact prepared prompt.
                del completed[ident]
                if studio is not None:
                    references, paths = studio
                    try:
                        resolved = await asyncio.to_thread(verify_studio_paths, references, require_files=True)
                        if resolved != paths:
                            raise ModelError("原生 ComfyUI 的模型搜索路径在排队前发生变化")
                    except (ModelError, studio_worker.WorkerError) as exc:
                        return web.json_response(model_problem(str(exc)), status=422)
            else:
                try:
                    _, task = local.submit(body, body.get("studio_model_dependencies", []) if signed else [])
                    rows = await asyncio.shield(task)
                except ModelError as exc:
                    return web.json_response(model_problem(str(exc)), status=422)
                failed = [row for row in rows if row.get("state") != "ready"]
                if failed:
                    problem = model_problem("、".join(f"{r['category']}/{r['name']} · {r.get('error') or r.get('state')}" for r in failed)[:3000])
                    return web.json_response({**problem, "models": rows}, status=422)
            refresh()
            await refresh_lora_manager()
            local.handoffs += 1
            try:
                return await handler(request)
            finally:
                local.handoffs -= 1
        response = await handler(request)
        if request.method != "GET" or response.status != 200:
            return response
        if path.startswith("/models/") and response.content_type == "application/json":
            category = path.removeprefix("/models/")
            extensions = folder_paths.folder_names_and_paths.get(category_for(category), ([], set()))[1]
            names = local.names(category, extensions)
            if names:
                return web.json_response(sorted(set(json.loads(response.body)) | set(names)))
        if (path == "/object_info" or path.startswith("/object_info/")) and response.content_type == "application/json":
            data = json.loads(response.body)
            mapping = deps.rules()
            available = {}
            for node, definition in data.items():
                if definition.get("api_node"):
                    continue
                for section in (definition.get("input") or {}).values():
                    if not isinstance(section, dict):
                        continue
                    for field, category in mapping.get(node, {}).items():
                        item = section.get(field)
                        if isinstance(item, list) and item:
                            options = item[0] if isinstance(item[0], list) else (
                                item[1].get("options") if item[0] == "COMBO" and len(item) > 1 and isinstance(item[1], dict) else None
                            )
                            if isinstance(options, list):
                                gguf = node.endswith("GGUF")
                                key = category, gguf
                                if key not in available:
                                    extensions = {".gguf"} if gguf else folder_paths.folder_names_and_paths.get(
                                        category_for(category), ([], set())
                                    )[1]
                                    available[key] = local.names(category, extensions)
                                existing = set(options)
                                options.extend(name for name in available[key] if name not in existing)
            return web.json_response(data)
        return response

    server.app.middlewares.append(prepare_prompt)
    return local
