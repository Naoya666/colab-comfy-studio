"""Settings and isolated HTTP access for native ComfyUI integrations."""

from __future__ import annotations

import hashlib
import re
import secrets
import time
import uuid
from urllib.parse import parse_qsl

from .api_access import AUTH_MODES, normalize_origins, validate_listen_port
from .common import APIError
from .db import utc_isoformat, utcnow
from .gpu import native
from .gpu.manifest import digest
from .gpu.specs import spec_key

KEYS = "native_api_keys"
PROMPTS = "native_api_prompts"
KEY_PATTERN = re.compile(r"nca_([0-9a-f]{32})\.[A-Za-z0-9_-]{43}")
PROMPT_ID = re.compile(r"[a-zA-Z0-9_-]{1,64}")
CLIENT_ID = re.compile(r"[a-zA-Z0-9_.:-]{1,128}")
MAX_WORKFLOW_BYTES = 1024 * 1024
START_WAIT_SECONDS = 25
RATE_PER_MINUTE = 20
MAX_OUTSTANDING = 4

TEXT_MARKERS = {"%prompt%", "%negative_prompt%", "%nl%"}
NUMBER_MARKERS = {"%seed%": (0, 2**53 - 1), "%width%": (64, 2048), "%height%": (64, 2048)}


def policy(store, owner):
    try:
        row = store.get("settings", f"native-api:{owner}", owner) or {}
    except TypeError:
        row = store.get("settings", f"native-api:{owner}") or {}
    saved = row.get("policy", {}) if isinstance(row, dict) and isinstance(row.get("policy"), dict) else {}
    return {
        "auto_start": False,
        "account_id": None,
        "gpu": None,
        "auth_mode": "strict",
        "allow_lan": False,
        "listen_port": None,
        "weak_workflow_mode": "trusted",
        "weak_workflow": None,
        "allowed_origins": [],
        **native.timeout_options()["defaults"],
        **native.model_storage_defaults(),
        **saved,
    }


def validate_policy(repo, owner, body):
    previous = policy(repo.store, owner)
    if set(body) - set(previous):
        raise APIError("NATIVE_API_INPUT", "API 启动设置无效")
    merged = {**previous, **body}
    if type(merged.get("auto_start")) is not bool:
        raise APIError("NATIVE_API_INPUT", "API 启动设置无效")
    if merged.get("auth_mode") not in AUTH_MODES:
        raise APIError("NATIVE_API_INPUT", "鉴权模式无效")
    if type(merged.get("allow_lan")) is not bool:
        raise APIError("NATIVE_API_INPUT", "局域网开关必须为布尔值")
    listen_port = validate_listen_port(merged.get("listen_port"), kind="native", store=repo.store, owner=owner)
    allowed_origins = normalize_origins(merged.get("allowed_origins"))
    weak_mode = merged.get("weak_workflow_mode")
    weak_graph = merged.get("weak_workflow")
    if weak_mode not in ("trusted", "approved"):
        raise APIError("NATIVE_API_INPUT", "弱鉴权工作流权限模式无效")
    if weak_mode == "approved":
        if not is_graph(weak_graph):
            raise APIError("NATIVE_API_INPUT", "限定工作流模式下必须提供有效的 ComfyUI API 格式模板")
    else:
        weak_graph = None

    timeouts = native.timeout_options()["defaults"]
    for field in timeouts:
        value = merged.get(field)
        if type(value) is not int or not 1 <= value <= 2**31 - 1:
            raise APIError("NATIVE_API_INPUT", f"{field} 必须为正整数（分钟）")
        timeouts[field] = value
    account, gpu = merged.get("account_id"), merged.get("gpu")
    if isinstance(account, str):
        account = account.strip() or None
    elif account is not None:
        raise APIError("NATIVE_API_INPUT", "Google 账号参数无效")
    if merged["auto_start"]:
        from .compute_routes import _require_account

        if account:
            _require_account(repo, owner, account)
        else:
            from .admission import _account_is_usable, _google_status

            all_accounts = repo.list("accounts", owner)
            usable = [row.get("account_id") or row["id"] for row in all_accounts if _account_is_usable(row)]
            if not usable:
                raise APIError("NATIVE_API_INPUT", "没有已授权且允许分配的 Google 账号，无法启用自动启动")
            creds = repo.list("credentials", owner)
            if any(c.get("purpose") == "colab" for c in creds):
                connected = [aid for aid in usable if _google_status(repo, owner, aid)["colab"]["connected"]]
                if not connected:
                    raise APIError("NATIVE_API_INPUT", "请先授权至少一个 Google Colab 账号后再启用自动启动")
        try:
            gpu = spec_key(gpu)
            if gpu["variant"] != "VARIANT_GPU":
                raise ValueError("请选择 GPU 计算规格")
        except (TypeError, ValueError) as exc:
            raise APIError("NATIVE_API_INPUT", str(exc)) from exc
    else:
        account, gpu = previous["account_id"], previous["gpu"]
    return {
        "auto_start": merged["auto_start"],
        "account_id": account or None,
        "gpu": gpu or None,
        "auth_mode": merged["auth_mode"],
        "allow_lan": merged["allow_lan"],
        "listen_port": listen_port,
        "weak_workflow_mode": weak_mode,
        "weak_workflow": weak_graph,
        "allowed_origins": allowed_origins,
        **timeouts,
        **native.model_storage(merged, validate=True),
    }


def persist_policy(store, owner, values):
    record_id = f"native-api:{owner}"
    with store.admission_lock() as acquired:
        if not acquired:
            raise APIError("NATIVE_API_BUSY", "API 设置正在保存，请重试", 503)
        existing = store.get("settings", record_id, owner)
        patch = {"policy": values}
        if values["auth_mode"] in {"url_path", "local_open"} and not (existing and existing.get("path_code")):
            patch["path_code"] = secrets.token_urlsafe(18)
        if existing:
            store.update("settings", record_id, patch, owner)
        else:
            store.create("settings", owner, patch, record_id=record_id)
    return values


def save_policy(repo, owner, body):
    values = validate_policy(repo, owner, body)
    return persist_policy(repo.store, owner, values)


def public_key(row):
    return {name: row.get(name) for name in (
        "id", "name", "auto_start", "mode", "created_at", "revoked_at", "last_used_at", "submitted",
    )}


def create_key(store, owner, body):
    name = body.get("name")
    mode = body.get("mode", "approved")
    graph = body.get("workflow")
    if (set(body) - {"name", "mode", "auto_start", "workflow"}
            or not isinstance(name, str) or not 1 <= len(name.strip()) <= 80
            or type(body.get("auto_start", False)) is not bool or mode not in ("trusted", "approved")):
        raise APIError("NATIVE_API_INPUT", "请填写有效的接入名称、模式和开机权限")
    if mode == "approved":
        if not is_graph(graph):
            raise APIError("NATIVE_API_INPUT", "请填写获准的 ComfyUI API 格式工作流")
    elif graph is not None:
        raise APIError("NATIVE_API_INPUT", "可信客户端模式无需工作流模板")
    ident = uuid.uuid4().hex
    token = f"nca_{ident}.{secrets.token_urlsafe(32)}"
    row = store.create(KEYS, owner, {
        "name": name.strip(), "token_digest": token_digest(token), "auto_start": body.get("auto_start", False),
        "mode": mode, "workflow": graph, "submitted": 0,
    }, record_id=ident)
    return {**public_key(row), "token": token}


def revoke_key(store, owner, key_id):
    with store.admission_lock() as acquired:
        if not acquired:
            raise APIError("NATIVE_API_BUSY", "暂无法撤销接入 Key，请重试", 503)
        row = store.get(KEYS, key_id, owner)
        if not row:
            raise APIError("NATIVE_API_NOT_FOUND", "接入 Key 不存在", 404)
        if not row.get("revoked_at"):
            row = store.update(KEYS, key_id, {"revoked_at": utc_isoformat(utcnow())}, owner)
    return public_key(row)


def token_digest(token):
    return hashlib.sha256(token.encode()).hexdigest()


def authenticate(store, token):
    match = KEY_PATTERN.fullmatch(token)
    row = store.get(KEYS, match[1]) if match else None
    if not row or row.get("revoked_at") or not secrets.compare_digest(row["token_digest"], token_digest(token)):
        raise APIError("UNAUTHENTICATED", "ComfyUI 接入 Key 无效或已撤销", 401)
    return row


def live_key(store, key):
    row = store.get(KEYS, key["id"], key["owner_id"])
    if not row or row.get("revoked_at"):
        raise APIError("UNAUTHENTICATED", "ComfyUI 接入 Key 已撤销", 401)
    return row


def touch(store, key):
    stamp = utc_isoformat(utcnow())
    if (key.get("last_used_at") or "")[:16] != stamp[:16]:
        with store.admission_lock(timeout=0) as acquired:
            if acquired:
                live_key(store, key)
                store.update(KEYS, key["id"], {"last_used_at": stamp}, key["owner_id"])


def is_graph(graph):
    return isinstance(graph, dict) and bool(graph) and all(
        isinstance(node, dict) and isinstance(node.get("class_type"), str)
        and isinstance(node.get("inputs"), dict) for node in graph.values()
    )


def _matches(template, value, *, allow_markers=False):
    if allow_markers and isinstance(template, str):
        if template in TEXT_MARKERS:
            return isinstance(value, str) and len(value) <= 12000
        if template in NUMBER_MARKERS:
            minimum, maximum = NUMBER_MARKERS[template]
            return type(value) is int and minimum <= value <= maximum
    if type(template) is not type(value):
        return False
    if isinstance(template, dict):
        return template.keys() == value.keys() and all(
            _matches(item, value[name], allow_markers=allow_markers) for name, item in template.items()
        )
    if isinstance(template, list):
        return len(template) == len(value) and all(
            _matches(a, b, allow_markers=allow_markers) for a, b in zip(template, value)
        )
    return template == value


def submission(key, body):
    graph = body.get("prompt") if isinstance(body, dict) else None
    if not is_graph(graph):
        raise APIError("NATIVE_API_INPUT", "请提交包含 prompt 的 ComfyUI API 工作流")
    if key["mode"] == "approved":
        template = key["workflow"]
        if template.keys() != graph.keys() or not all(
            node.keys() == graph[ident].keys()
            and all(_matches(value, graph[ident][field], allow_markers=field == "inputs")
                    for field, value in node.items())
            for ident, node in template.items()
        ):
            raise APIError("NATIVE_WORKFLOW_DENIED", "提交的工作流与此 Key 的获准模板不匹配", 403)
    # Queue priority and extra_data are reserved for native UI and admin clients.
    return {"prompt": graph}


def weak_submission(pol, body):
    """Validate prompt payload for weak/no-auth mode while preserving optional client_id for /ws tracking."""
    result = submission(
        {"mode": pol.get("weak_workflow_mode", "trusted"), "workflow": pol.get("weak_workflow")},
        body,
    )
    client_id = body.get("client_id") if isinstance(body, dict) else None
    if isinstance(client_id, str) and CLIENT_ID.fullmatch(client_id):
        result["client_id"] = client_id
    return result


def scope_path(path, method, query):
    if path.startswith("/api/"):
        path = path.removeprefix("/api")
    if method == "POST" and path == "/prompt" and not query:
        return path
    if method == "GET" and path == "/view":
        return path
    if method == "GET" and not query and (
        path in {"/system_stats", "/object_info", "/_studio/api/status"}
        or re.fullmatch(r"/(?:history|object_info)/[a-zA-Z0-9_. -]{1,128}", path)
        or re.fullmatch(r"/models/[a-zA-Z0-9_]{1,64}", path)
    ):
        if path.split("/")[-1] not in {".", ".."}:
            return path
    raise APIError("NATIVE_API_FORBIDDEN", "此接入 Key 无权调用该接口", 403)


def prompt_record_id(key, lease, prompt_id):
    return digest({"key": key["id"], "lease": lease["id"], "prompt": prompt_id})


def _file_tuple(item):
    values = [item.get(field, default) for field, default in (("filename", ""), ("subfolder", ""), ("type", "output"))]
    if not all(isinstance(part, str) for part in values):
        return None
    filename, subfolder, kind = values
    if (not filename or len(filename) > 255 or len(subfolder) > 512
            or filename in {".", ".."} or "/" in filename or "\\" in filename or "\0" in filename
            or "\\" in subfolder or "\0" in subfolder or subfolder.startswith("/")
            or any(part in {".", ".."} for part in subfolder.split("/"))
            or kind not in {"output", "temp"}):
        return None
    return values


def view_tuple(query):
    fields = parse_qsl(query, keep_blank_values=True)
    values = dict(fields)
    result = _file_tuple(values)
    if len(fields) != len(values) or set(values) - {"filename", "subfolder", "type"} or not result:
        raise APIError("NATIVE_API_INPUT", "图片参数无效", 400)
    return result


def outputs(item):
    nodes = item.get("outputs") if isinstance(item, dict) else None
    if not isinstance(nodes, dict):
        return []
    return [
        result for node in nodes.values() if isinstance(node, dict)
        for field in ("images", "gifs") if isinstance(node.get(field), list)
        for image in node[field] if isinstance(image, dict)
        if (result := _file_tuple(image)) is not None
    ]


def view_id(key, lease, file):
    return digest({"key": key["id"], "lease": lease["id"], "file": file})


def reserve(store, key, lease, observed=None):
    """Limit submissions across gateway processes and account for an in-flight POST."""
    with store.admission_lock() as acquired:
        if not acquired:
            raise APIError("NATIVE_API_BUSY", "API 请求较多，请稍后重试", 503)
        row = live_key(store, key)
        now = time.time()
        active = {ident: entry for ident, entry in (row.get("active") or {}).items()
                  if entry["lease"] == lease["id"] and (entry.get("prompt") or now - entry["time"] < 120)
                  and (observed is None or observed.get(ident) != entry.get("prompt") or not entry.get("prompt"))}
        if len(active) >= MAX_OUTSTANDING:
            return None
        window = [sent for sent in row.get("window", []) if now - sent < 60]
        if len(window) >= RATE_PER_MINUTE:
            raise APIError("NATIVE_API_RATE", "已达到此 Key 的每分钟提交上限", 429)
        ident = uuid.uuid4().hex
        active[ident] = {"lease": lease["id"], "time": now}
        store.update(KEYS, key["id"], {"active": active, "window": [*window, now]}, key["owner_id"])
        return ident


def refresh_reservation(store, key, lease, reservation):
    """Extend an in-flight prompt reservation while long model preparation is still running."""
    if not reservation:
        return
    with store.admission_lock(timeout=2) as acquired:
        if not acquired:
            return
        row = store.get(KEYS, key["id"], key["owner_id"])
        if not row or row.get("revoked_at"):
            return
        active = dict(row.get("active") or {})
        if reservation in active and not active[reservation].get("prompt"):
            active[reservation] = {**active[reservation], "time": time.time()}
            store.update(KEYS, key["id"], {"active": active}, key["owner_id"])


def finish_submission(store, key, lease, reservation, prompt_id=None):
    with store.admission_lock(timeout=5) as acquired:
        if not acquired:
            raise APIError("NATIVE_API_BUSY", "无法记录提交状态", 503)
        row = store.get(KEYS, key["id"], key["owner_id"])
        if not row:
            return
        active = dict(row.get("active") or {})
        patch = {"active": active}
        if prompt_id:
            store.create(PROMPTS, key["owner_id"], {"prompt_id": prompt_id, "lease_id": lease["id"]},
                         project_id=key["id"], record_id=prompt_record_id(key, lease, prompt_id))
            entry = active.get(reservation) or {"lease": lease["id"], "time": time.time()}
            active[reservation] = {**entry, "lease": lease["id"], "prompt": prompt_id}
            patch["submitted"] = row.get("submitted", 0) + 1
        else:
            active.pop(reservation, None)
        store.update(KEYS, key["id"], patch, key["owner_id"])


def register_outputs(store, key, lease, item):
    files = outputs(item)
    if not files:
        return
    with store.admission_lock(timeout=3) as acquired:
        if not acquired:
            raise APIError("NATIVE_API_BUSY", "暂无法记录图片权限，请重试", 503)
        live_key(store, key)
        for file in files:
            ident = view_id(key, lease, file)
            if not store.get(PROMPTS, ident, key["owner_id"]):
                store.create(PROMPTS, key["owner_id"], {"lease_id": lease["id"]}, project_id=key["id"], record_id=ident)


def request_start(repo, owner, key, *, api_key_label=None):
    if key is not None:
        key = live_key(repo.store, key)
        if not key.get("auto_start"):
            return None
    values = policy(repo.store, owner)
    if not values["auto_start"]:
        return None
    body = {"account_id": values["account_id"], "spec": {"key": values["gpu"]},
            "retain_idle_minutes": values["retain_idle_minutes"], "max_runtime_minutes": values["max_runtime_minutes"], **native.model_storage(values)}
    return native.admit(repo, owner, body, f"native-api-start:{uuid.uuid4().hex}",
                        api_key_id=key["id"] if key else (api_key_label or "admin"))
