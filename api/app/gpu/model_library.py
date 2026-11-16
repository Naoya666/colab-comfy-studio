"""Durable model catalog and a bounded, CPU-only transfer queue.

Large files live in temporary scratch only during a transfer. One dedicated
worker owns the queue lock, enabling safe restart/retry across API processes.
"""
from __future__ import annotations

import copy
import hashlib
import os
import re
import shutil
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote, urlsplit

from app.config import settings
from app.filelocks import file_lock
from app.security import decrypt_secret, encrypt_secret

from . import model_dependencies as deps
from .drive import DriveError, DriveStorage
from .model_sources import SHA, ModelError, SourceHTTP, Sources, filename, https_url, relative_path

ACTIVE = {"pending", "downloading", "verifying", "uploading"}
FIELDS = "id,name,size,mimeType,trashed,md5Checksum,sha256Checksum,appProperties"


class ModelLibrary:
    def __init__(self, repo, owner, *, drive=None, http=None):
        self.repo, self.owner = repo, owner
        self._drive, self._http = drive, http

    @property
    def drive(self):
        if self._drive is None:
            self._drive = DriveStorage({"id": "model-library", "owner_id": self.owner}, self.repo,
                                       account_id=self.storage_settings().get("account_id"))
        return self._drive

    @property
    def http(self):
        if self._http is None:
            self._http = SourceHTTP(self.tokens())
        return self._http

    def close(self):
        if self._http:
            self._http.close()
        if self._drive and self._drive._owns_client:
            self._drive.client.close()

    def models(self):
        return self.repo.list("model_library", self.owner)

    def get(self, ident):
        value = self.repo.get("model_library", ident, self.owner)
        if not value:
            raise ModelError("模型不存在或不属于当前用户")
        return value

    def update(self, ident, **changes):
        return self.repo.update("model_library", ident, changes, self.owner)

    def storage_settings(self):
        ident = hashlib.sha256(f"model-storage:{self.owner}".encode()).hexdigest()
        return self.repo.get("files", ident, self.owner) or {"account_id": None}

    def save_storage(self, account_id):
        from .accounts import AccountService
        if account_id is not None and not self.repo.get("accounts", account_id, self.owner):
            raise ModelError("Drive 存储账号不存在或不属于当前用户")
        resolved = AccountService(self.repo, self.owner).resolve_drive_account_id(account_id)
        old = self.storage_settings()
        if self.models() and old.get("account_id") != resolved:
            raise ModelError("已有模型索引，不能直接切换存储账号；请先迁移文件与索引")
        ident = hashlib.sha256(f"model-storage:{self.owner}".encode()).hexdigest()
        data = {"kind": "model_library_storage", "account_id": resolved}
        if old.get("id"):
            return self.repo.update("files", ident, data, self.owner)
        return self.repo.create("files", self.owner, data, record_id=ident)

    def tokens(self):
        result = {}
        for row in self.repo.list("credentials", self.owner):
            if row.get("kind") == "model_source":
                result[row["source"]] = decrypt_secret(row["encrypted_secret"])
        return result

    def token_status(self):
        rows = self.repo.list("credentials", self.owner)
        return {s: any(r.get("kind") == "model_source" and r.get("source") == s for r in rows)
                for s in ("huggingface", "civitai")}

    def save_token(self, source, value):
        if source not in {"huggingface", "civitai"} or not isinstance(value, str) or len(value) > 4096:
            raise ModelError("下载源或凭据格式无效")
        ident = hashlib.sha256(f"model-token:{self.owner}:{source}".encode()).hexdigest()
        old = self.repo.get("credentials", ident, self.owner)
        if not value:
            if old:
                self.repo.store.delete("gpu_credentials", ident, self.owner)
        elif old:
            self.repo.update("credentials", ident, {"encrypted_secret": encrypt_secret(value)}, self.owner)
        else:
            self.repo.create("credentials", self.owner, {"kind": "model_source", "source": source,
                                                       "encrypted_secret": encrypt_secret(value)}, record_id=ident)
        return self.token_status()

    def resolve(self, body):
        return Sources(self.http).resolve(body)

    def add(self, data):
        name = filename(data.get("filename"))
        category = data.get("model_type")
        if category not in deps.directories():
            raise ModelError("请选择模型类型（可通过 MODEL_LIBRARY_DIRECTORIES 扩展）")
        expected = data.get("expected_sha256")
        if expected is not None and (not isinstance(expected, str) or not SHA.fullmatch(expected)):
            raise ModelError("预期 SHA-256 必须为完整 64 位十六进制值")
        size = data.get("expected_size")
        if size is not None and (isinstance(size, bool) or not isinstance(size, int) or size <= 0):
            raise ModelError("预期大小必须为正整数字节数")
        source = data.get("source") or "https"
        if source not in {"https", "huggingface", "civitai", "drive", "upload", "legacy_cache"}:
            raise ModelError("不支持的模型来源")
        # All source fields are selected explicitly: never persist arbitrary client metadata/tokens.
        row = {"display_name": str(data.get("display_name") or name)[:300], "filename": name,
               "model_type": category, "architecture": str(data.get("architecture") or "unknown")[:120],
               "expected_sha256": expected.lower() if expected else None, "sha256": None,
               "expected_size": size, "size": None, "source": source,
               "source_url": https_url(data["source_url"]) if data.get("source_url") else None,
               "download_url": https_url(data["download_url"]) if data.get("download_url") else None,
               "source_identity": {}, "state": "pending", "error": None,
               "version_name": str(data.get("version_name") or "")[:200],
               "trigger_words": [str(x)[:200] for x in (data.get("trigger_words") or [])[:100]],
               "tags": [], "last_used_at": None, "managed_file": False,
               "legacy_target": relative_path(data["legacy_target"]) if data.get("legacy_target") else None}
        identity = data.get("source_identity") or {}
        if not isinstance(identity, dict):
            raise ModelError("来源身份格式无效")
        keys = {"huggingface": ("repo_id", "revision", "filename"), "civitai": ("model_id", "version_id", "file_id")}.get(source, ())
        row["source_identity"] = {k: identity[k] for k in keys if k in identity}
        if source in {"drive", "upload", "legacy_cache"}:
            file_id = data.get("drive_file_id")
            if not isinstance(file_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,200}", file_id):
                raise ModelError("Drive 文件 ID 无效；drive.file 授权只能访问当前应用可见文件")
            row["drive_file_id"] = file_id
            row["state"] = "unverified"
        elif not row["download_url"]:
            raise ModelError("缺少下载来源，请先解析并选择文件")
        identity_key = None
        if expected:
            for existing in self.models():
                if existing["model_type"] == category and expected.lower() in {existing.get("expected_sha256"), existing.get("sha256")}:
                    return existing
            identity_key = f"sha256:{expected.lower()}"
        elif row["source_identity"] and source in {"huggingface", "civitai"}:
            from .manifest import digest
            identity_key = source + ":" + digest(row["source_identity"])
        record_id = hashlib.sha256(f"library:{self.owner}:{category}:{identity_key}".encode()).hexdigest() if identity_key else None
        try:
            return self.repo.create("model_library", self.owner, row, record_id=record_id)
        except ValueError:
            existing = self.repo.get("model_library", record_id, self.owner) if record_id else None
            if existing:
                return existing
            raise

    def references(self, ident):
        return [{"workflow_id": row["workflow_id"], "nodes": [d["node_id"] for d in row["dependencies"] if d.get("model_id") == ident]}
                for row in self.repo.list("model_dependencies", self.owner)
                if any(d.get("model_id") == ident for d in row["dependencies"])]

    def enqueue(self, ident, action="prepare"):
        self.get(ident)
        if action not in {"prepare", "verify"}:
            raise ModelError("未知模型任务操作")
        with self.repo.store.admission_lock() as acquired:
            if not acquired:
                raise ModelError("模型队列繁忙，请重试")
            existing = next((t for t in self.repo.list("model_downloads", self.owner)
                             if t["model_id"] == ident and t.get("state") in ACTIVE), None)
            if existing:
                return existing
            task = self.repo.create("model_downloads", self.owner,
                                    {"model_id": ident, "action": action, "state": "pending", "bytes": 0,
                                     "total": None, "speed": None, "error": None, "attempt": 0, "cancel_requested": False})
            self.update(ident, state="pending", error=None)
            return task

    def cancel(self, task_id):
        task = self.repo.get("model_downloads", task_id, self.owner)
        if not task:
            raise ModelError("任务不存在")
        if task.get("state") not in ACTIVE:
            return task
        # Worker observes cancellation at every chunk; pending tasks cancel immediately.
        patch = {"cancel_requested": True}
        if task["state"] == "pending":
            patch["state"] = "cancelled"
            self.update(task["model_id"], state="failed", error="模型准备已取消，可重试")
        return self.repo.update("model_downloads", task_id, patch, self.owner)

    def metadata(self, file_id):
        response = self.drive._request("GET", self.drive.base_url + "/files/" + quote(file_id, safe=""), params={"fields": FIELDS})
        if response.status_code == 404:
            raise ModelError("Drive 文件不存在，请重新导入或下载")
        meta = self.drive._ensure_ok(response, "模型元数据读取")
        if meta.get("trashed") or meta.get("mimeType", "").startswith("application/vnd.google-apps.") or int(meta.get("size") or 0) <= 0:
            raise ModelError("Drive 模型文件为空、已删除或是非权重文件")
        return meta

    def check_remote(self, ident):
        row = self.get(ident)
        if not row.get("drive_file_id"):
            raise ModelError("模型尚未写入 Drive，请先准备模型")
        try:
            meta = self.metadata(row["drive_file_id"])
        except ModelError:
            self.update(ident, state="missing", error="Drive 文件缺失，请重新下载或导入")
            raise
        if row.get("size") != int(meta["size"]) or (row.get("drive_md5") and row["drive_md5"] != meta.get("md5Checksum")):
            return self.update(ident, state="unverified", error="Drive 文件已变化，请重新校验")
        return row

    def upload_session(self, name, size, origin):
        filename(name)
        if not isinstance(size, int) or isinstance(size, bool) or not 0 < size <= max_file_bytes():
            raise ModelError("文件大小超过模型上传限制")
        folder = self.drive.model_cache()["folder_id"]
        parent = self.drive._folder("library-uploads", folder)
        response = self.drive._request("POST", self.drive.upload_url + "/files",
                                       params={"uploadType": "resumable", "fields": "id,name,size"},
                                       json_body={"name": name, "parents": [parent]},
                                       headers={"X-Upload-Content-Type": "application/octet-stream",
                                                "X-Upload-Content-Length": str(size), "Origin": origin})
        location = response.headers.get("location") or ""
        if response.status_code >= 300 or urlsplit(location).hostname not in {"www.googleapis.com", "content.googleapis.com"} or urlsplit(location).scheme != "https":
            raise ModelError("Drive 分块上传会话创建失败，请重新授权 Drive 后重试")
        return {"upload_url": location, "chunk_size": 8 * 1024 * 1024}

    def dependency_record(self, workflow_id, *, rescan=False):
        workflow = self.repo.get("workflows", workflow_id, self.owner)
        if not workflow:
            raise ModelError("工作流不存在")
        ident = hashlib.sha256(f"model-deps:{self.owner}:{workflow_id}".encode()).hexdigest()
        old = self.repo.get("model_dependencies", ident, self.owner)
        scanned = deps.scan(workflow) if rescan or not old else copy.deepcopy(old["dependencies"])
        matched = deps.match(scanned, self.models(), (old or {}).get("dependencies") or [])
        complete = all(not d.get("required") or d.get("disabled") or d["status"] == "matched" for d in matched)
        data = {"workflow_id": workflow_id, "dependencies": matched, "models_ready": complete,
                "workflow_state": workflow.get("state"), "runtime_validated": workflow.get("smoke_test_kind") == "real"}
        if old:
            return self.repo.update("model_dependencies", ident, data, self.owner)
        return self.repo.create("model_dependencies", self.owner, data, record_id=ident)

    def bind(self, workflow_id, dep_id, model_id, **options):
        record = self.dependency_record(workflow_id)
        model = self.get(model_id)
        found = False
        for index, dependency in enumerate(record["dependencies"]):
            if dependency["id"] == dep_id:
                record["dependencies"][index] = deps.bind(dependency, model, **options)
                found = True
        if not found:
            raise ModelError("依赖不存在，请重新扫描")
        self.repo.update("model_dependencies", record["id"], {"dependencies": record["dependencies"]}, self.owner)
        return self.dependency_record(workflow_id)

    def prepare_dependencies(self, workflow_id):
        record = self.dependency_record(workflow_id)
        tasks, unresolved = [], []
        for dep in record["dependencies"]:
            if not dep.get("required") or dep.get("disabled"):
                continue
            model_id = dep.get("model_id")
            if not model_id and dep.get("weight"):
                weight = dep["weight"]
                target = weight.get("target") or weight.get("filename") or ""
                if weight.get("source") == "drive":
                    data = {"source": "drive", "drive_file_id": weight.get("drive_file_id")}
                elif all(weight.get(k) for k in ("repo_id", "revision", "filename")):
                    url = f"https://huggingface.co/{weight['repo_id']}/resolve/{weight['revision']}/{quote(weight['filename'], safe='/')}"
                    data = {"source": "huggingface", "download_url": url, "source_url": url, "source_identity": weight}
                else:
                    unresolved.append(dep["id"])
                    continue
                model = self.add({**data, "filename": Path(target).name, "model_type": dep["model_type"],
                                  "expected_sha256": weight.get("sha256"), "expected_size": weight.get("size"), "legacy_target": target})
                model_id = model["id"]
                dep.update(model_id=model_id, confirmed=True)
            if not model_id and (dep.get("source_hint") or dep.get("expected_sha256")):
                try:
                    candidates = self.resolve({"url": dep["source_hint"]} if dep.get("source_hint") else {"sha256": dep["expected_sha256"]})
                    exact = [f for f in candidates if not dep.get("expected_sha256") or f.get("expected_sha256") == dep["expected_sha256"]]
                    if len(exact) == 1:
                        selected = exact[0]
                        model = self.add({**selected, "model_type": dep["model_type"],
                                          "expected_sha256": dep.get("expected_sha256") or selected.get("expected_sha256")})
                        model_id = model["id"]
                        dep.update(model_id=model_id, confirmed=True)
                    else:
                        dep["source_error"] = "来源包含多个或不匹配的文件，请选择具体文件"
                except ModelError as exc:
                    dep["source_error"] = str(exc)
            if model_id:
                if self.get(model_id).get("state") != "ready":
                    tasks.append(self.enqueue(model_id))
            else:
                unresolved.append(dep["id"])
        self.repo.update("model_dependencies", record["id"], {"dependencies": record["dependencies"]}, self.owner)
        return {"tasks": list({t["id"]: t for t in tasks}.values()), "unresolved": unresolved}

    def delete(self, ident, *, confirm_references=False, delete_file=False):
        row = self.get(ident)
        refs = self.references(ident)
        if refs and not confirm_references:
            raise ModelError(f"模型仍被 {len(refs)} 个工作流引用，请明确确认删除")
        if any(t["model_id"] == ident and t.get("state") in ACTIVE for t in self.repo.list("model_downloads", self.owner)):
            raise ModelError("请先取消模型准备任务并等待停止")
        if delete_file and row.get("drive_file_id"):
            if not row.get("managed_file"):
                raise ModelError("外部导入/历史缓存文件仅允许移除索引，请在 Drive 手动管理原文件")
            if any(m["id"] != ident and m.get("drive_file_id") == row["drive_file_id"] for m in self.models()):
                raise ModelError("该物理文件仍被其他模型索引共享，不能删除")
            response = self.drive._request("PATCH", self.drive.base_url + "/files/" + quote(row["drive_file_id"]), json_body={"trashed": True})
            self.drive._ensure_ok(response, "模型删除")
        self.repo.store.delete("gpu_model_library", ident, self.owner)
        for ref in refs:
            self.dependency_record(ref["workflow_id"])
        return {"removed": ident, "references": len(refs)}

    def adopt_cache(self, workflow_id):
        """Index reviewed target/hash entries from the existing Drive cache, without copying bytes."""
        from .runtime_worker import DriveCache
        workflow = self.repo.get("workflows", workflow_id, self.owner)
        if not workflow:
            raise ModelError("工作流不存在")
        cache_config = self.drive.model_cache()
        cache = DriveCache(cache_config["access_token"], cache_config["folder_id"])
        adopted = []
        for weight in deps.weights_of(workflow):
            target = weight.get("target") or ""
            parts = target.split("/", 1)
            category = parts[0]
            if len(parts) != 2 or category not in deps.directories():
                continue
            if any(m.get("sha256") == weight.get("sha256") and m.get("state") == "ready" for m in self.models()):
                continue
            entry = cache.lookup(target, weight.get("sha256"), weight.get("size"))
            if entry:
                model = self.add({"source": "legacy_cache", "drive_file_id": entry["id"], "filename": Path(target).name,
                                  "model_type": category, "expected_sha256": weight.get("sha256"),
                                  "expected_size": weight.get("size")})
                adopted.append(self.enqueue(model["id"], "verify"))
        return adopted

    def register_workflow_model(self, ident, data):
        """Create a disabled registry family; normal license review and real smoke still apply."""
        row = self.get(ident)
        if row.get("state") != "ready":
            raise ModelError("请先完成模型 Drive 准备与完整性校验")
        key = data.get("model_key")
        if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", key):
            raise ModelError("工作流模型标识仅支持字母、数字、点、下划线和短横线")
        if any(m.get("model_id") == key for m in self.repo.list("models", self.owner)):
            raise ModelError("工作流模型标识已存在")
        return self.repo.create("models", self.owner, {
            "model_id": key, "display_name": row["display_name"], "enabled": False, "is_mock": False,
            "library_model_id": ident, "source_kind": row["source"], "source": row.get("source_url"),
            "source_identity": row.get("source_identity"), "drive_file_id": row.get("drive_file_id"),
            "model_sha256": row["sha256"], "revision": (row.get("source_identity") or {}).get("revision"),
            "license": str(data.get("license") or "")[:200], "resource_profile": data.get("resource_profile") or {},
            "review_approved": False, "license_accepted": False,
            "disabled_reason": "等待管理员来源、许可审查及真实 GPU 冒烟验证"})

    def export_bound_draft(self, workflow_id):
        """Construct a reviewable new artifact with explicit Drive weights and a copied graph."""
        from .registry import Registry
        original = self.repo.get("workflows", workflow_id, self.owner)
        if not original:
            raise ModelError("工作流不存在")
        bundle = {k: copy.deepcopy(original.get(k) or {}) for k in ("manifest", "api_graph", "object_info", "editable_graph")}
        bundle.update(workflow_key=original["workflow_key"], model_id=original["model_id"])
        bundle["manifest"].pop("version", None)
        weights = deps.weights_of(original)
        weights = copy.deepcopy(weights)
        for d in self.dependency_record(workflow_id)["dependencies"]:
            if not d.get("model_id"):
                continue
            m = self.get(d["model_id"])
            if m["state"] != "ready":
                raise ModelError("请先准备全部绑定模型")
            target = deps.target_for(d, m)
            weights = [w for w in weights if w.get("target") != target]
            weights.append({"source": "drive", "drive_file_id": m["drive_file_id"], "sha256": m["sha256"], "size": m["size"], "target": target})
            if d.get("strengths"):
                for field, value in d["strengths"].items():
                    deps.set_input(bundle["api_graph"][d["node_id"]]["inputs"], field, 0 if d.get("disabled") else value)
        bundle["manifest"].setdefault("dependencies", {})["weight_files"] = weights
        bundle["manifest"].pop("weight_files", None)
        return Registry(self.repo).import_workflow(self.owner, bundle)


def max_file_bytes():
    return int(os.getenv("MODEL_LIBRARY_MAX_FILE_BYTES", str(150 * 1024**3)))


@contextmanager
def transfer_lock():
    directory = settings.data_dir / "locks"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "model-library.lock").open("a+b") as handle, file_lock(handle, blocking=False) as acquired:
        yield acquired


class ModelTransfer:
    def __init__(self, library, task):
        self.lib, self.task = library, task
        self.model = library.get(task["model_id"])
        self.started = time.monotonic()
        self.last_progress = 0.0
        self.phase = None
        self.phase_started = self.started

    def progress(self, phase, written=0, total=None, *, force=False):
        current = self.lib.repo.get("model_downloads", self.task["id"], self.lib.owner)
        if current.get("cancel_requested"):
            raise ModelError("模型准备已取消，可重试")
        now = time.monotonic()
        if self.phase != phase:
            self.phase = phase
            self.phase_started = now
        if force or now - self.last_progress >= 1:
            self.lib.repo.update("model_downloads", self.task["id"], {"state": phase, "bytes": written, "total": total,
                                 "speed": round(written / max(now - self.phase_started, 0.01)), "heartbeat": time.time()}, self.lib.owner)
            self.lib.update(self.model["id"], state=phase)
            self.last_progress = now

    def read_drive(self, path):
        meta = self.lib.metadata(self.model["drive_file_id"])
        total = int(meta["size"])
        self.ensure_space(path, total)
        token = self.lib.drive.google.access_token("drive")
        # Reuse the runtime's Google-only authenticated media transport; no token enters catalog.
        from .runtime_worker import DriveCache
        cache = DriveCache(token, self.lib.drive.model_cache()["folder_id"])
        written = 0
        with cache._open("GET", f"https://www.googleapis.com/drive/v3/files/{quote(meta['id'])}?alt=media") as response, path.open("wb") as output:
            for block in iter(lambda: response.read(8 * 1024 * 1024), b""):
                written += len(block)
                if written > total:
                    raise ModelError("Drive 文件大小变化，请重试")
                output.write(block)
                self.progress("verifying", written, total)
        if written != total:
            raise ModelError("Drive 文件读取不完整，请重试")
        return meta

    def ensure_space(self, path, size):
        if size > max_file_bytes():
            raise ModelError("模型文件超过管理员设置的单文件大小限制")
        if shutil.disk_usage(path.parent).free < size + 512 * 1024**2:
            raise ModelError("模型临时下载磁盘不足，请扩容 MODEL_LIBRARY_SCRATCH 或导入已有 Drive 文件")

    def download(self, path):
        url = self.model["download_url"]
        written = 0
        with self.lib.http.stream(url) as response:
            raw_size = response.headers.get("content-length")
            total = int(raw_size) if raw_size and raw_size.isdigit() else self.model.get("expected_size")
            self.ensure_space(path, total or 512 * 1024**2)
            self.progress("downloading", 0, total, force=True)
            with path.open("wb") as output:
                for block in response.iter_bytes(8 * 1024 * 1024):
                    written += len(block)
                    self.ensure_space(path, len(block))
                    if written > max_file_bytes() or (total and written > total):
                        raise ModelError("下载文件大小超过预期/限制")
                    output.write(block)
                    self.progress("downloading", written, total)
        if total and written != total:
            raise ModelError("下载中断或文件不完整，请重试")

    def upload(self, path, sha, md5, size):
        drive = self.lib.drive
        root = drive.model_cache()["folder_id"]
        content = drive._folder("by-sha256", root)
        parent = drive._folder(sha[:2], content)
        # Content-addressed physical identity, independent of filename/category/GPU.
        query = f"'{drive._q(parent)}' in parents and trashed = false and name = '{sha}'"
        response = drive._request("GET", drive.base_url + "/files", params={"q": query, "fields": f"files({FIELDS})"})
        for entry in drive._ensure_ok(response, "模型去重查询").get("files") or []:
            if int(entry.get("size") or 0) == size and entry.get("md5Checksum") == md5:
                return entry
        response = drive._request("POST", drive.upload_url + "/files",
                                  params={"uploadType": "resumable", "fields": FIELDS},
                                  json_body={"name": sha, "parents": [parent],
                                             "appProperties": {"sha256": sha, "size": str(size)}},
                                  headers={"X-Upload-Content-Length": str(size), "X-Upload-Content-Type": "application/octet-stream"})
        location = response.headers.get("location") or ""
        parsed = urlsplit(location)
        if response.status_code >= 300 or parsed.scheme != "https" or parsed.hostname not in {"www.googleapis.com", "content.googleapis.com"}:
            raise ModelError("Drive 模型上传会话创建失败，请重新授权后重试")
        # Persist encrypted upload URL for diagnostics/reconciliation; never expose its bearer capability.
        self.lib.repo.update("model_downloads", self.task["id"], {"encrypted_upload_session": encrypt_secret(location)}, self.lib.owner)
        offset = 0
        with path.open("rb") as stream:
            while offset < size:
                block = stream.read(8 * 1024 * 1024)
                end = offset + len(block) - 1
                response = drive._request("PUT", location, content=block,
                                          headers={"Content-Type": "application/octet-stream", "Content-Range": f"bytes {offset}-{end}/{size}"},
                                          retryable=False)
                offset = end + 1
                if offset < size and response.status_code != 308:
                    raise ModelError(f"Drive 分块上传失败（HTTP {response.status_code}），请重试")
                self.progress("uploading", offset, size)
        entry = drive._ensure_ok(response, "模型上传")
        if int(entry.get("size") or 0) != size or entry.get("md5Checksum") != md5:
            raise ModelError("Drive 上传结果大小或校验和不匹配，未提交 ready")
        return entry

    def run(self):
        scratch = Path(os.getenv("MODEL_LIBRARY_SCRATCH", str(settings.data_dir / "model-scratch")))
        scratch.mkdir(parents=True, exist_ok=True)
        # Any previous process holding the global lock has exited. Its scratch can be safely removed.
        for old in scratch.glob("transfer-*"):
            if old.is_dir():
                shutil.rmtree(old)
        self.lib.repo.update("model_downloads", self.task["id"], {"attempt": self.task.get("attempt", 0) + 1}, self.lib.owner)
        try:
            self.progress("verifying", force=True)
            # Reuse verified catalog content before any download. Check actual Drive metadata first.
            expected = self.model.get("expected_sha256") or self.model.get("sha256")
            if expected and self.task.get("action") != "verify":
                for other in self.lib.models():
                    if other["id"] != self.model["id"] and other.get("sha256") == expected and other.get("state") == "ready":
                        other = self.lib.check_remote(other["id"])
                        if other["state"] == "ready":
                            self.finish(other["drive_file_id"], expected, other["size"], other.get("drive_md5"), other.get("managed_file", False))
                            return
            verify_existing = self.model.get("source") in {"drive", "upload", "legacy_cache"} or self.task.get("action") == "verify"
            if verify_existing and self.model.get("drive_file_id"):
                remote = self.lib.metadata(self.model["drive_file_id"])
                native_hash = remote.get("sha256Checksum")
                if isinstance(native_hash, str) and SHA.fullmatch(native_hash) and remote.get("md5Checksum"):
                    native_hash = native_hash.lower()
                    size = int(remote["size"])
                    if (expected and native_hash != expected) or (self.model.get("expected_size") and size != self.model["expected_size"]):
                        raise ModelError("Drive 原生 SHA-256 或大小不匹配，未提交 ready")
                    self.finish(remote["id"], native_hash, size, remote["md5Checksum"], self.model.get("managed_file", False))
                    return
            with tempfile.TemporaryDirectory(prefix="transfer-", dir=scratch) as directory:
                path = Path(directory) / "weight.partial"
                legacy = False
                if self.model.get("legacy_target") and self.model.get("expected_sha256") and not self.model.get("drive_file_id"):
                    from .runtime_worker import DriveCache
                    config = self.lib.drive.model_cache()
                    cache = DriveCache(config["access_token"], config["folder_id"])
                    entry = cache.lookup(self.model["legacy_target"], self.model["expected_sha256"], self.model.get("expected_size"))
                    if entry:
                        self.model["drive_file_id"] = entry["id"]
                        legacy = True
                imported = legacy or self.model.get("source") in {"drive", "upload", "legacy_cache"} or (self.task.get("action") == "verify" and bool(self.model.get("drive_file_id")))
                meta = self.read_drive(path) if imported else None
                if not imported:
                    self.download(path)
                self.progress("verifying", force=True)
                sha, md5 = hashlib.sha256(), hashlib.md5(usedforsecurity=False)
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                        sha.update(block)
                        md5.update(block)
                        self.progress("verifying", stream.tell(), path.stat().st_size)
                sha, md5, size = sha.hexdigest(), md5.hexdigest(), path.stat().st_size
                if not size or (expected and sha != expected) or (self.model.get("expected_size") and size != self.model["expected_size"]):
                    raise ModelError("模型 SHA-256 或文件大小不匹配，未提交 ready；请核对来源和预期值")
                if imported and meta.get("md5Checksum") and md5 != meta["md5Checksum"]:
                    raise ModelError("Drive 文件传输校验失败，请重试")
                if not imported:
                    self.progress("uploading", 0, size, force=True)
                    meta = self.upload(path, sha, md5, size)
                self.progress("verifying", size, size, force=True)
                # Confirm remote commit, not just the final upload response.
                remote = self.lib.metadata(meta["id"])
                if int(remote["size"]) != size or remote.get("md5Checksum") != md5:
                    raise ModelError("Drive 最终完整性检查失败，未提交 ready")
                self.finish(meta["id"], sha, size, md5, self.model.get("managed_file", False) if imported else True)
        except Exception as exc:
            # Provider exception messages/responses may contain signed URLs or credentials.
            error = str(exc) if isinstance(exc, ModelError) else (
                str(exc) if isinstance(exc, DriveError) and "HTTP " in str(exc) else "模型准备失败，请检查 Drive 授权、网络、配额和临时磁盘后重试")
            cancelled = (self.lib.repo.get("model_downloads", self.task["id"], self.lib.owner) or {}).get("cancel_requested")
            self.lib.update(self.model["id"], state="failed", error=error)
            self.lib.repo.update("model_downloads", self.task["id"], {"state": "cancelled" if cancelled else "failed", "error": error}, self.lib.owner)

    def finish(self, file_id, sha, size, md5, managed):
        from .accounts import AccountService
        storage_account = getattr(self.lib.drive, "account_id", None) or AccountService.default_account_id(self.lib.owner)
        self.lib.update(self.model["id"], drive_account_id=storage_account, state="ready", drive_file_id=file_id, sha256=sha, size=size,
                        drive_md5=md5, managed_file=managed, error=None,
                        integrity="expected_hash_verified" if self.model.get("expected_sha256") else "computed_hash_only")
        self.lib.repo.update("model_downloads", self.task["id"], {"state": "ready", "bytes": size, "total": size,
                             "error": None, "encrypted_upload_session": None}, self.lib.owner)


def transfer_tick(repo):
    """Run in a dedicated worker; one active transfer per shared data directory."""
    with transfer_lock() as acquired:
        if not acquired:
            return {"advanced": False}
        tasks = [t for t in repo.list("model_downloads") if t.get("state") in ACTIVE]
        if not tasks:
            return {"advanced": False}
        task = sorted(tasks, key=lambda t: str(t.get("created_at")))[0]
        lib = ModelLibrary(repo, task["owner_id"])
        try:
            # Interrupted active tasks are safely retried; no ready state is inferred.
            ModelTransfer(lib, task).run()
        finally:
            lib.close()
        return {"advanced": True, "job_id": task["id"], "state": "model_transfer"}
