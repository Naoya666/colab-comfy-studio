"""Durable model catalog and a bounded, CPU-only transfer queue.

Task-owned scratch preserves download checkpoints across worker restarts. One dedicated
worker owns the queue lock, enabling safe restart/retry across API processes.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import time
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from urllib.parse import quote, urlsplit

from app.config import settings
from app.filelocks import file_lock
from app.security import decrypt_secret, encrypt_secret

from . import model_dependencies as deps
from .drive import _RETRYABLE_STATUSES, DriveError, DriveStorage, _DriveSessionExpired, _DriveTransportError
from .model_sources import SHA, ModelError, SourceHTTP, Sources, filename, https_url, relative_path

ACTIVE = {"pending", "downloading", "verifying", "uploading"}
BUFFER = 1024 ** 2
CHECKPOINT_BYTES = 32 * BUFFER
UPLOAD_CHUNK = 8 * BUFFER
FIELDS = "id,name,size,mimeType,trashed,md5Checksum,sha256Checksum,appProperties"


def external_source(row):
    if "external_source" in row:
        return row["external_source"]
    if row.get("download_url"):
        return {"url": row.get("original_source_url") or row.get("source_url") or row["download_url"],
                "download_url": row["download_url"],
                "status": row.get("source_verification") or "unverified"}
    return None


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

    def storage_account(self):
        from .accounts import AccountService

        accounts = AccountService(self.repo, self.owner)
        account = (getattr(self._drive, "account_id", None) if self._drive
                   else accounts.resolve_drive_account_id(self.storage_settings().get("account_id")))
        return account or accounts.default_account_id(self.owner)

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
        files = Sources(self.http).resolve(body)
        if body.get("url") and not body.get("sha256") and not body.get("repo_id"):
            return [{**item, "original_source_url": https_url(body["url"])} for item in files]
        return files

    def bind_external_source(self, row, url):
        if url is None or url == "":
            return None
        original = https_url(url)
        choices = self.resolve({"url": original, "filename": row["filename"]})
        known_hash = row.get("sha256") or row.get("expected_sha256")
        known_size = row.get("size") or row.get("expected_size")
        compatible = [
            item for item in choices
            if (not known_hash or item.get("expected_sha256") in {None, known_hash})
            and (not known_size or item.get("expected_size") in {None, known_size})
        ]
        exact = [item for item in compatible if known_hash and item.get("expected_sha256") == known_hash]
        selected = exact or [item for item in compatible if item["filename"] == row["filename"]]
        if len(selected) != 1:
            raise ModelError("来源文件不匹配或版本有歧义，请提供具体文件链接并核对 SHA-256 和大小")
        return {"url": original, "download_url": selected[0]["download_url"],
                "status": "verified" if exact and row.get("state") == "ready" and row.get("sha256") else "unverified"}

    def add(self, data):
        name = filename(data.get("filename"))
        legacy_target = relative_path(data["legacy_target"]) if data.get("legacy_target") else None
        model_path = relative_path(data.get("relative_path") or (legacy_target.split("/", 1)[-1] if legacy_target else name))
        if PurePosixPath(model_path).name != name:
            raise ModelError("模型相对路径必须以所登记的文件名结尾")
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
        account = self.storage_account()
        # All source fields are selected explicitly: never persist arbitrary client metadata/tokens.
        row = {"display_name": str(data.get("display_name") or name)[:300], "filename": name,
               "model_type": category, "relative_path": model_path, "drive_account_id": account, "architecture": str(data.get("architecture") or "unknown")[:120],
               "expected_sha256": expected.lower() if expected else None, "sha256": None,
               "expected_size": size, "size": None, "source": source,
               "source_url": https_url(data["source_url"]) if data.get("source_url") else None,
               "original_source_url": https_url(data["original_source_url"]) if data.get("original_source_url") else None,
               "download_url": https_url(data["download_url"]) if data.get("download_url") else None,
               "source_identity": {}, "state": "pending", "error": None,
               "version_name": str(data.get("version_name") or "")[:200],
               "trigger_words": [str(x)[:200] for x in (data.get("trigger_words") or [])[:100]],
               "tags": [], "last_used_at": None, "managed_file": False,
               "legacy_target": legacy_target}
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
            identity_key = f"sha256:{expected.lower()}"
        elif row["source_identity"] and source in {"huggingface", "civitai"}:
            from .manifest import digest
            identity_key = source + ":" + digest(row["source_identity"])
        elif row.get("drive_file_id"):
            identity_key = "drive:" + row["drive_file_id"]
        row_origin = row.get("original_source_url") or row.get("source_url") or row.get("download_url")
        record_id = hashlib.sha256(
            json.dumps([self.owner, account, category, model_path, identity_key, row_origin, row.get("drive_file_id")]).encode()
        ).hexdigest() if identity_key else None
        if expected:
            from .accounts import AccountService

            legacy_account = AccountService.default_account_id(self.owner)
            for existing in self.models():
                if (existing["model_type"] == category and (existing.get("relative_path") or existing["filename"]) == model_path
                        and (existing.get("drive_account_id") or legacy_account) == account
                        and (existing.get("original_source_url") or existing.get("source_url") or existing.get("download_url")) == row_origin
                        and expected.lower() in {existing.get("expected_sha256"), existing.get("sha256")}
                        and (not row.get("drive_file_id") or row["drive_file_id"] == existing.get("drive_file_id"))):
                    return existing
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
        model = self.get(ident)
        if action not in {"prepare", "verify"}:
            raise ModelError("未知模型任务操作")
        with self.repo.store.admission_lock() as acquired:
            if not acquired:
                raise ModelError("模型队列繁忙，请重试")
            existing = next((t for t in self.repo.list("model_downloads", self.owner)
                             if t["model_id"] == ident and t.get("state") in ACTIVE), None)
            if existing:
                return existing
            failed = next((t for t in self.repo.list("model_downloads", self.owner)
                           if t["model_id"] == ident and t.get("state") == "failed" and t.get("action") == action), None)
            if failed:
                task = self.repo.update("model_downloads", failed["id"],
                                        {"state": "pending", "error": None, "cancel_requested": False}, self.owner)
                self.update(ident, state="pending", error=None)
                return task
            task = self.repo.create("model_downloads", self.owner,
                                    {"model_id": ident, "action": action, "state": "pending", "bytes": 0,
                                     "original_source_url": model.get("original_source_url") or model.get("source_url") or model.get("download_url"),
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


def transfer_directory(task_id):
    scratch = Path(os.getenv("MODEL_LIBRARY_SCRATCH", str(settings.data_dir / "model-scratch")))
    return scratch / ("transfer-" + hashlib.sha256(task_id.encode()).hexdigest())


@contextmanager
def transfer_lock():
    directory = settings.data_dir / "locks"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "model-library.lock").open("a+b") as handle, file_lock(handle, blocking=False) as acquired:
        yield acquired


class TransferPaused(Exception):
    """The worker is shutting down; leave the task active for its replacement."""


class ModelTransfer:
    def __init__(self, library, task, *, stopping=lambda: False):
        self.lib, self.task = library, task
        self.stopping = stopping
        self.model = library.get(task["model_id"])
        self.started = time.monotonic()
        self.last_progress = 0.0
        self.phase = None
        self.phase_started = self.started
        self.phase_offset = 0

    def progress(self, phase, written=0, total=None, *, force=False):
        if self.stopping():
            raise TransferPaused
        current = self.lib.repo.get("model_downloads", self.task["id"], self.lib.owner)
        if current.get("cancel_requested"):
            raise ModelError("模型准备已取消，可重试")
        now = time.monotonic()
        if self.phase != phase:
            self.phase = phase
            self.phase_started = now
            self.phase_offset = written
        if force or now - self.last_progress >= 1:
            self.lib.repo.update("model_downloads", self.task["id"], {"state": phase, "bytes": written, "total": total,
                                 "speed": round(max(0, written - self.phase_offset) / max(now - self.phase_started, 0.01)), "heartbeat": time.time()}, self.lib.owner)
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
            for block in iter(lambda: response.read(BUFFER), b""):
                written += len(block)
                if written > total:
                    raise ModelError("Drive 文件大小变化，请重试")
                output.write(block)
                if written % CHECKPOINT_BYTES == 0:
                    self.flush_data(output)
                self.progress("verifying", written, total)
            self.flush_data(output)
        if written != total:
            raise ModelError("Drive 文件读取不完整，请重试")
        return meta

    def ensure_space(self, path, size):
        if size > max_file_bytes():
            raise ModelError("模型文件超过管理员设置的单文件大小限制")
        if shutil.disk_usage(path.parent).free < size + 512 * 1024**2:
            raise ModelError("模型临时下载磁盘不足，请扩容 MODEL_LIBRARY_SCRATCH 或导入已有 Drive 文件")

    def task_update(self, **patch):
        self.lib.repo.update("model_downloads", self.task["id"], patch, self.lib.owner)

    @staticmethod
    def discard_cache(stream, offset=0, length=0):
        if hasattr(os, "posix_fadvise"):
            os.posix_fadvise(stream.fileno(), offset, length, os.POSIX_FADV_DONTNEED)

    @classmethod
    def flush_data(cls, stream):
        stream.flush()
        os.fsync(stream.fileno())
        cls.discard_cache(stream)

    def source_key(self):
        from .manifest import digest

        return digest({key: self.model.get(key) for key in
                       ("download_url", "source_identity", "expected_size", "expected_sha256")})

    def hash_file(self, path, limit=None):
        sha, md5 = hashlib.sha256(), hashlib.md5(usedforsecurity=False)
        total = path.stat().st_size if limit is None else limit
        with path.open("rb", buffering=0) as stream:
            while stream.tell() < total:
                block = stream.read(min(BUFFER, total - stream.tell()))
                if not block:
                    raise ModelError("模型暂存文件读取不完整")
                sha.update(block)
                md5.update(block)
                position = stream.tell()
                if position % CHECKPOINT_BYTES == 0:
                    self.discard_cache(stream, position - CHECKPOINT_BYTES, CHECKPOINT_BYTES)
                self.progress("verifying", position, total)
            self.discard_cache(stream)
        return sha, md5, total

    def checkpoint(self, path, stream, identity, sha, total, *, complete=False):
        self.flush_data(stream)
        marker = path.with_suffix(".json")
        tmp = marker.with_suffix(".tmp")
        with tmp.open("w") as output:
            json.dump({**identity, "bytes": stream.tell(), "sha256": sha.hexdigest(),
                       "total": total, "complete": complete}, output)
            output.flush()
            os.fsync(output.fileno())
        tmp.replace(marker)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def read_checkpoint(self, path):
        try:
            data = json.loads(path.with_suffix(".json").read_text())
            size = path.stat().st_size
            if (data["source_key"] != self.source_key() or type(data["bytes"]) is not int
                    or not 0 < data["bytes"] <= min(size, max_file_bytes())):
                return None, None
            sha, md5, _ = self.hash_file(path, data["bytes"])
            if sha.hexdigest() == data["sha256"]:
                return data, (sha, md5)
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            pass
        return None, None

    @staticmethod
    def response_etag(response):
        etag = response.headers.get("etag", "")
        return etag if re.fullmatch(r'"[^"\r\n]+"', etag) else None

    def response_total(self, response, checkpoint):
        if response.headers.get("content-encoding", "identity").lower() != "identity":
            if checkpoint:
                return None
            raise ModelError("下载来源返回了压缩传输，无法验证文件字节范围")
        length = response.headers.get("content-length")
        expected = self.model.get("expected_size")
        if checkpoint:
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("content-range", ""))
            if response.status_code != 206 or not match:
                return None
            start, end, total = map(int, match.groups())
            if (start != checkpoint["bytes"] or end != total - 1 or start >= total
                    or checkpoint.get("total") not in {None, total} or expected not in {None, total}
                    or (length and length != str(total - start))
                    or (checkpoint.get("etag") and (self.response_etag(response) != checkpoint["etag"]
                        or response.extensions.get("studio_source_resource") != checkpoint.get("resource")))):
                return None
            return total
        if length and length.isdecimal():
            total = int(length)
            if expected is not None and total != expected:
                raise ModelError("来源声明的文件大小与预期不匹配")
            return total
        return expected

    def consume_download(self, path, response, checkpoint, sha, total):
        written = checkpoint["bytes"] if checkpoint else 0
        if total is not None and total > max_file_bytes():
            raise ModelError("模型文件超过管理员设置的单文件大小限制")
        self.ensure_space(path, (total - written) if total is not None else BUFFER)
        self.task_update(resume_offset=written, resume_phase="downloading")
        self.progress("downloading", written, total, force=True)
        identity = {"source_key": self.source_key(), "etag": self.response_etag(response),
                    "resource": response.extensions.get("studio_source_resource")}
        with path.open("r+b" if checkpoint else "wb", buffering=0) as output:
            output.seek(written)
            output.truncate()
            checkpoint_at = written
            for block in response.iter_bytes(BUFFER):
                written += len(block)
                if written > max_file_bytes() or (total is not None and written > total):
                    raise ModelError("下载文件大小超过预期或限制")
                output.write(block)
                sha.update(block)
                if written - checkpoint_at >= CHECKPOINT_BYTES:
                    self.ensure_space(path, 0)
                    self.checkpoint(path, output, identity, sha, total)
                    checkpoint_at = written
                if self.stopping():
                    self.checkpoint(path, output, identity, sha, total)
                self.progress("downloading", written, total)
            if not written or (total is not None and written != total):
                raise ModelError("下载中断或文件不完整，请重试")
            self.checkpoint(path, output, identity, sha, written, complete=True)

    def download(self, path):
        prior = path.exists()
        checkpoint, hashes = self.read_checkpoint(path)
        if checkpoint:
            sha, md5 = hashes
            offset = checkpoint["bytes"]
            expected_hash = self.model.get("expected_sha256")
            if checkpoint.get("complete") or (
                expected_hash and offset == checkpoint.get("total") and sha.hexdigest() == expected_hash
            ):
                with path.open("r+b") as stream:
                    stream.truncate(offset)
                self.task_update(resume_offset=offset, resume_phase="verifying", restart_reason=None)
                return sha, md5, offset
            # A pinned upstream object with a known digest can resume without a strong ETag.
            pinned = (self.model.get("source") in {"huggingface", "civitai"}
                      and self.model.get("source_identity") and expected_hash)
            if checkpoint.get("etag") or pinned:
                with self.lib.http.stream(self.model["download_url"], offset=offset, validator=checkpoint.get("etag")) as response:
                    total = self.response_total(response, checkpoint)
                    if total is not None:
                        self.task_update(restart_reason=None)
                        self.consume_download(path, response, checkpoint, sha, total)
                        return None
                    if response.status_code == 200:
                        self.task_update(restart_reason="来源版本变化或忽略了 HTTP Range，已从头下载")
                        path.with_suffix(".json").unlink(missing_ok=True)
                        self.consume_download(path, response, None, hashlib.sha256(), self.response_total(response, None))
                        return None
                reason = "来源版本变化或未返回匹配的 HTTP Range，已从头下载"
            else:
                reason = "无法确认下载来源版本，已从头下载"
        else:
            reason = "下载检查点缺失或校验失败，已从头下载"
        if prior or self.task.get("state") in {"downloading", "verifying", "uploading"}:
            self.task_update(restart_reason=reason)
        path.with_suffix(".json").unlink(missing_ok=True)
        with self.lib.http.stream(self.model["download_url"]) as response:
            self.consume_download(path, response, None, hashlib.sha256(), self.response_total(response, None))
        return None

    @staticmethod
    def matches(entry, sha, md5, size):
        return (not entry.get("trashed") and str(entry.get("size")) == str(size)
                and entry.get("md5Checksum") == md5 and entry.get("sha256Checksum") in {None, sha})

    def find_uploaded(self, drive, parent, sha, md5, size):
        query = f"'{drive._q(parent)}' in parents and trashed = false and name = '{sha}'"
        params = {"q": query, "fields": f"nextPageToken,files({FIELDS})"}
        while True:
            response = drive._request("GET", drive.base_url + "/files", params=params)
            data = drive._ensure_ok(response, "模型去重查询")
            for entry in data.get("files") or []:
                if self.matches(entry, sha, md5, size):
                    return entry
            if not data.get("nextPageToken"):
                return None
            params["pageToken"] = data["nextPageToken"]

    def uploaded_by_id(self, drive, file_id, sha, md5, size):
        if not file_id:
            return None
        response = drive._request("GET", drive.base_url + "/files/" + quote(file_id), params={"fields": FIELDS})
        if response.status_code == 404:
            return None
        entry = drive._ensure_ok(response, "模型完整性核对")
        if not self.matches(entry, sha, md5, size):
            raise ModelError("Drive 已存在的上传对象大小或校验值不匹配")
        return entry

    def upload(self, path, sha, md5, size):
        drive = self.lib.drive
        root = drive.model_cache()["folder_id"]
        parent = drive._folder(sha[:2], drive._folder("by-sha256", root))
        mime = "application/octet-stream"
        identity = [root, sha, size]
        file_id = self.task.get("upload_file_id") if self.task.get("upload_identity") == identity else None
        location = None
        if self.task.get("encrypted_upload_session") and self.task.get("upload_identity") in (None, identity):
            location = decrypt_secret(self.task["encrypted_upload_session"])
            if not DriveStorage._controlled_upload_location(location):
                raise ModelError("Drive 模型上传会话地址无效")
        offset = 0
        expired = False
        for _ in range(3):
            try:
                if location:
                    offset, result = drive._file_resume_status(location, size, mime)
                    self.task_update(resume_phase="uploading", resume_offset=offset)
                    if result:
                        return result
                else:
                    existing = self.uploaded_by_id(drive, file_id, sha, md5, size) or self.find_uploaded(drive, parent, sha, md5, size)
                    if existing:
                        return existing
                    if expired:
                        self.task_update(resume_phase="uploading", resume_offset=0,
                                         restart_reason="Drive 上传会话已过期，已用校验后的文件重新上传")
                    if not file_id:
                        response = drive._request("GET", drive.base_url + "/files/generateIds", params={"count": 1})
                        ids = drive._ensure_ok(response, "模型上传文件 ID 分配").get("ids") or []
                        if len(ids) != 1 or not re.fullmatch(r"[A-Za-z0-9_-]+", ids[0]):
                            raise ModelError("Drive 未返回有效的上传文件 ID")
                        file_id = ids[0]
                        self.task_update(upload_file_id=file_id, upload_identity=identity, encrypted_upload_session=None)
                    response = drive._request("POST", drive.upload_url + "/files",
                                              params={"uploadType": "resumable", "fields": FIELDS},
                                              json_body={"id": file_id, "name": sha, "parents": [parent],
                                                         "appProperties": {"sha256": sha, "size": str(size)}},
                                              headers={"X-Upload-Content-Length": str(size), "X-Upload-Content-Type": mime})
                    if response.status_code == 409:
                        existing = self.uploaded_by_id(drive, file_id, sha, md5, size)
                        if existing:
                            return existing
                    location = response.headers.get("location") or ""
                    if response.status_code not in {200, 201} or not DriveStorage._controlled_upload_location(location):
                        raise ModelError(f"Drive 模型上传会话创建失败（HTTP {response.status_code}）")
                    self.task_update(encrypted_upload_session=encrypt_secret(location), upload_identity=identity)
                self.progress("uploading", offset, size, force=True)
                return self.upload_chunks(drive, location, path, offset, size, mime)
            except _DriveSessionExpired:
                location, offset = None, 0
                expired = True
            except _DriveTransportError:
                existing = self.uploaded_by_id(drive, file_id, sha, md5, size) or self.find_uploaded(drive, parent, sha, md5, size)
                if existing:
                    return existing
                raise
        raise ModelError("Drive 上传会话反复失效，请稍后重试")

    def upload_chunks(self, drive, location, path, offset, size, mime):
        stalled = 0
        with path.open("rb", buffering=0) as stream:
            def chunks(length):
                while length > 0:
                    data = stream.read(min(BUFFER, length))
                    if not data:
                        raise ModelError("模型暂存文件在上传时发生变化")
                    yield data
                    length -= len(data)

            while offset < size:
                stream.seek(offset)
                previous = offset
                length = min(UPLOAD_CHUNK, size - offset)
                end = offset + length - 1
                try:
                    response = drive._request("PUT", location, content=chunks(length), retryable=False,
                                              headers={"Content-Type": mime, "Content-Length": str(length),
                                                       "Content-Range": f"bytes {offset}-{end}/{size}"})
                except _DriveTransportError:
                    response = None
                if response is None or response.status_code in _RETRYABLE_STATUSES:
                    offset, result = drive._file_resume_status(location, size, mime)
                    if result:
                        return result
                elif response.status_code in {404, 410}:
                    raise _DriveSessionExpired("Drive 模型上传会话已过期")
                elif response.status_code in {200, 201}:
                    return drive._ensure_ok(response, "模型上传")
                elif response.status_code == 308:
                    offset = DriveStorage._file_upload_offset(response, size)
                    if not previous <= offset <= end + 1:
                        raise ModelError("Drive 返回的模型上传偏移无效")
                else:
                    raise ModelError(f"Drive 分块上传失败（HTTP {response.status_code}）")
                self.discard_cache(stream, previous, length)
                stalled = stalled + 1 if offset <= previous else 0
                if stalled >= 3:
                    raise ModelError("Drive 模型上传未继续推进，请重试")
                self.progress("uploading", offset, size)
        _, result = drive._file_resume_status(location, size, mime)
        if not result:
            raise ModelError("Drive 尚未确认模型已完整上传，请重试")
        return result

    def run(self):
        directory = transfer_directory(self.task["id"])
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        path = directory / "weight.partial"
        self.task_update(attempt=self.task.get("attempt", 0) + 1)
        try:
            self.progress("verifying", force=True)
            # Reuse verified catalog content before any download. Check actual Drive metadata first.
            expected = self.model.get("expected_sha256") or self.model.get("sha256")
            if expected and self.task.get("action") != "verify":
                from .accounts import AccountService

                account = self.lib.storage_account()
                legacy_account = AccountService.default_account_id(self.lib.owner)
                for other in self.lib.models():
                    if (other["id"] != self.model["id"] and other.get("sha256") == expected
                            and other.get("state") == "ready" and (other.get("drive_account_id") or legacy_account) == account):
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
            verified = None if imported else self.download(path)
            self.progress("verifying", force=True)
            sha, md5, size = verified or self.hash_file(path)
            sha, md5 = sha.hexdigest(), md5.hexdigest()
            if not size or (expected and sha != expected) or (self.model.get("expected_size") and size != self.model["expected_size"]):
                path.unlink(missing_ok=True)
                path.with_suffix(".json").unlink(missing_ok=True)
                raise ModelError("模型 SHA-256 或文件大小不匹配，未提交 ready；请核对来源和预期值")
            if imported and meta.get("md5Checksum") and md5 != meta["md5Checksum"]:
                raise ModelError("Drive 文件传输校验失败，请重试")
            if not imported:
                meta = self.upload(path, sha, md5, size)
            self.progress("verifying", size, size, force=True)
            # Confirm remote commit, not just the final upload response.
            remote = self.lib.metadata(meta["id"])
            if not self.matches(remote, sha, md5, size):
                raise ModelError("Drive 最终完整性检查失败，未提交 ready")
            self.finish(meta["id"], sha, size, md5, self.model.get("managed_file", False) if imported else True,
                        source_verified=not imported)
        except TransferPaused:
            return
        except Exception as exc:
            # Provider exception messages/responses may contain signed URLs or credentials.
            error = str(exc) if isinstance(exc, ModelError) else (
                str(exc) if isinstance(exc, DriveError) and "HTTP " in str(exc) else "模型准备失败，请检查 Drive 授权、网络、配额和临时磁盘后重试")
            cancelled = (self.lib.repo.get("model_downloads", self.task["id"], self.lib.owner) or {}).get("cancel_requested")
            self.lib.update(self.model["id"], state="failed", error=error)
            self.lib.repo.update("model_downloads", self.task["id"], {"state": "cancelled" if cancelled else "failed", "error": error}, self.lib.owner)

        finally:
            current = self.lib.repo.get("model_downloads", self.task["id"], self.lib.owner) or {}
            if current.get("state") in {"ready", "cancelled"} or not path.exists():
                shutil.rmtree(directory, ignore_errors=True)

    def finish(self, file_id, sha, size, md5, managed, *, source_verified=False):
        from .accounts import AccountService

        current = self.lib.repo.get("model_downloads", self.task["id"], self.lib.owner)
        if current.get("cancel_requested"):
            raise ModelError("模型准备已取消，可重试")
        storage_account = getattr(self.lib.drive, "account_id", None) or AccountService.default_account_id(self.lib.owner)
        self.lib.update(self.model["id"], drive_account_id=storage_account, state="ready", drive_file_id=file_id, sha256=sha, size=size,
                        drive_md5=md5, managed_file=managed, error=None,
                        **({"source_verification": "verified"} if source_verified else {}),
                        integrity="expected_hash_verified" if self.model.get("expected_sha256") else "computed_hash_only")
        self.lib.repo.update("model_downloads", self.task["id"], {"state": "ready", "bytes": size, "total": size, "drive_file_id": file_id,
                             "error": None, "encrypted_upload_session": None, "upload_identity": None,
                             "upload_file_id": None}, self.lib.owner)


def transfer_tick(repo, *, stopping=lambda: False):
    """Run in a dedicated worker; one active transfer per shared data directory."""
    with transfer_lock() as acquired:
        if not acquired or stopping():
            return {"advanced": False}
        tasks = []
        for task in repo.list("model_downloads"):
            if task.get("state") in ACTIVE:
                tasks.append(task)
            elif task.get("state") in {"ready", "cancelled"}:
                directory = transfer_directory(task["id"])
                if directory.is_dir():
                    shutil.rmtree(directory)
        if not tasks:
            return {"advanced": False}
        task = sorted(tasks, key=lambda t: str(t.get("created_at")))[0]
        lib = ModelLibrary(repo, task["owner_id"])
        try:
            # Interrupted active tasks are safely retried; no ready state is inferred.
            ModelTransfer(lib, task, stopping=stopping).run()
        finally:
            lib.close()
        return {"advanced": True, "job_id": task["id"], "state": "model_transfer"}
