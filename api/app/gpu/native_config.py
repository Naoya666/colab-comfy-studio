"""Dynamic encrypted Native ComfyUI setup with .env/file fallback and zero-GPU diagnostics.

Database configuration (stored in the existing ``settings`` table with Fernet-encrypted
secrets via ``ENCRYPTION_KEY``) takes precedence over ``.env`` and ``./secrets/*`` files
so all processes (api, worker, watchdog) observe changes immediately without restarting,
while existing deployments configured via environment variables continue working untouched.
"""

from __future__ import annotations

import base64
import ipaddress
import json
import os
import re
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit

import httpx

from app import offline
from app.common import APIError
from app.config import settings
from app.db import utc_isoformat, utcnow
from app.security import decrypt_secret, encrypt_secret

RECOMMENDED_COMFY_COMMIT = "423cbd20511becb1f16122ab518e92e58bfa03d9"
SETTING_PREFIX = "native-setup:"
RECORD_KIND = "native_setup"

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_GITHUB_COMMIT_RE = re.compile(r"github\.com/Comfy-Org/ComfyUI/commit/([0-9a-fA-F]{40})(?:[/?#].*)?$")
_TOKEN_FLAG_RE = re.compile(r"(?:--token(?:\s+|=)|service\s+install\s+)['\"]?([A-Za-z0-9+/=_-]{16,4096})['\"]?")
_SAFE_TOKEN_RE = re.compile(r"^[A-Za-z0-9+/=._:-]{16,4096}$")
_SAFE_HEADER_RE = re.compile(r"^[\x21-\x7e]{4,512}$")

_CACHE_TTL_SECONDS = 2.0
_CACHE_LOCK = threading.Lock()
_ROW_CACHE: dict[tuple[int, str], tuple[float, dict | None]] = {}


@dataclass(frozen=True)
class ResolvedNativeConfig:
    native_comfy_url: str
    native_comfy_url_source: str
    native_comfy_backend_url: str
    native_comfy_backend_url_source: str
    native_comfy_commit: str
    native_comfy_commit_source: str
    access_client_id: str
    access_client_id_source: str
    access_client_secret: str
    access_client_secret_source: str
    tunnel_token: str
    tunnel_token_source: str
    has_db_override: bool
    updated_at: str | None

    @property
    def tunnel_configured(self) -> bool:
        return bool(
            self.native_comfy_backend_url
            and self.access_client_id
            and self.access_client_secret
            and self.tunnel_token
            and _COMMIT_RE.fullmatch(self.native_comfy_commit)
        )

    @property
    def web_ui_configured(self) -> bool:
        return bool(self.native_comfy_url)

    @property
    def any_configured(self) -> bool:
        return bool(self.native_comfy_url or self.native_comfy_backend_url)


def _unwrap_store(store_or_repo):
    if store_or_repo is None:
        return None
    return getattr(store_or_repo, "store", store_or_repo)


def _cache_allowed() -> bool:
    if offline.enabled():
        return False
    if os.getenv("PYTEST_CURRENT_TEST"):
        return False
    if str(getattr(settings, "environment", "")).lower() == "offline-test":
        return False
    return True


def invalidate_cache(store_or_repo=None, owner: str | None = None) -> None:
    store = _unwrap_store(store_or_repo)
    with _CACHE_LOCK:
        if store is None and owner is None:
            _ROW_CACHE.clear()
            return
        store_id = id(store) if store is not None else None
        for key in list(_ROW_CACHE):
            if (store_id is None or key[0] == store_id) and (owner is None or key[1] == owner):
                _ROW_CACHE.pop(key, None)


def _load_db_row(store_or_repo, owner: str = "owner", *, use_cache: bool = True) -> dict | None:
    store = _unwrap_store(store_or_repo)
    if store is None or not hasattr(store, "get"):
        return None
    key = (id(store), owner)
    now = time.monotonic()
    can_cache = use_cache and _cache_allowed()
    if can_cache:
        with _CACHE_LOCK:
            cached = _ROW_CACHE.get(key)
            if cached is not None and now - cached[0] < _CACHE_TTL_SECONDS:
                return cached[1]
    record_id = SETTING_PREFIX + owner
    try:
        try:
            row = store.get("settings", record_id, owner)
        except TypeError:
            row = store.get("settings", record_id)
    except Exception:
        row = None
    valid = row if isinstance(row, dict) and row.get("kind") == RECORD_KIND else None
    if can_cache:
        with _CACHE_LOCK:
            _ROW_CACHE[key] = (now, valid)
    return valid


def _read_secret_file_optional(env_name: str, file_reader: Callable[[str], str] | None = None) -> str:
    if file_reader is not None:
        try:
            return str(file_reader(env_name)).strip()
        except Exception:
            return ""
    path = os.getenv(env_name, "").strip()
    if not path:
        return ""
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _read_env_access(file_reader: Callable[[str], str] | None = None) -> tuple[str, str]:
    raw = _read_secret_file_optional("NATIVE_COMFY_ACCESS_FILE", file_reader)
    if not raw:
        return "", ""
    try:
        parsed = json.loads(raw)
    except ValueError:
        return "", ""
    if not isinstance(parsed, dict):
        return "", ""
    cid = parsed.get("client_id")
    csec = parsed.get("client_secret")
    return (
        cid.strip() if isinstance(cid, str) else "",
        csec.strip() if isinstance(csec, str) else "",
    )


def _decrypt_optional(ciphertext: str | None, field_label: str) -> str:
    if not ciphertext:
        return ""
    try:
        return decrypt_secret(ciphertext).strip()
    except (ValueError, RuntimeError) as exc:
        raise APIError("NATIVE_CONFIG", f"已保存的{field_label}无法解密，请重新保存或检查 ENCRYPTION_KEY", 503) from exc


def resolve(
    store_or_repo=None,
    owner: str = "owner",
    *,
    use_cache: bool = True,
    file_reader: Callable[[str], str] | None = None,
    strict_decrypt: bool = True,
) -> ResolvedNativeConfig:
    row = _load_db_row(store_or_repo, owner, use_cache=use_cache)
    has_db = row is not None

    env_url = (settings.native_comfy_url or "").strip().rstrip("/")
    env_backend = (settings.native_comfy_backend_url or "").strip().rstrip("/")
    env_commit = os.getenv("NATIVE_COMFY_COMMIT", "").strip().lower()
    env_cid, env_csec = _read_env_access(file_reader)
    env_tunnel = _read_secret_file_optional("NATIVE_COMFY_TUNNEL_TOKEN_FILE", file_reader)

    if has_db and "native_comfy_url" in row and isinstance(row.get("native_comfy_url"), str):
        url = row["native_comfy_url"].strip().rstrip("/")
        url_source = "encrypted_database" if url else "unconfigured"
    else:
        url = env_url
        url_source = "environment" if url else "unconfigured"

    if has_db and isinstance(row.get("native_comfy_backend_url"), str) and row["native_comfy_backend_url"].strip():
        backend_url = row["native_comfy_backend_url"].strip().rstrip("/")
        backend_source = "encrypted_database"
    else:
        backend_url = env_backend
        backend_source = "environment" if backend_url else "unconfigured"

    if has_db and isinstance(row.get("native_comfy_commit"), str) and row["native_comfy_commit"].strip():
        commit = row["native_comfy_commit"].strip().lower()
        commit_source = "encrypted_database"
    else:
        commit = env_commit
        commit_source = "environment" if commit else "unconfigured"

    if has_db and isinstance(row.get("access_client_id"), str) and row["access_client_id"].strip():
        cid = row["access_client_id"].strip()
        cid_source = "encrypted_database"
    else:
        cid = env_cid
        cid_source = "environment" if cid else "unconfigured"

    db_csec = row.get("encrypted_access_client_secret") if has_db else None
    if db_csec:
        try:
            csec = _decrypt_optional(db_csec, "Cloudflare Access Secret")
            csec_source = "encrypted_database" if csec else "unconfigured"
        except APIError:
            if strict_decrypt:
                raise
            csec = ""
            csec_source = "error"
    else:
        csec = env_csec
        csec_source = "environment" if csec else "unconfigured"

    db_tunnel = row.get("encrypted_tunnel_token") if has_db else None
    if db_tunnel:
        try:
            tunnel = _decrypt_optional(db_tunnel, "Cloudflare Tunnel Token")
            tunnel_source = "encrypted_database" if tunnel else "unconfigured"
        except APIError:
            if strict_decrypt:
                raise
            tunnel = ""
            tunnel_source = "error"
    else:
        tunnel = env_tunnel
        tunnel_source = "environment" if tunnel else "unconfigured"

    return ResolvedNativeConfig(
        native_comfy_url=url,
        native_comfy_url_source=url_source,
        native_comfy_backend_url=backend_url,
        native_comfy_backend_url_source=backend_source,
        native_comfy_commit=commit,
        native_comfy_commit_source=commit_source,
        access_client_id=cid,
        access_client_id_source=cid_source,
        access_client_secret=csec,
        access_client_secret_source=csec_source,
        tunnel_token=tunnel,
        tunnel_token_source=tunnel_source,
        has_db_override=has_db,
        updated_at=(row or {}).get("updated_at"),
    )


def public_url(store_or_repo=None, owner: str = "owner", *, use_cache: bool = True) -> str:
    row = _load_db_row(store_or_repo, owner, use_cache=use_cache)
    if row is not None and "native_comfy_url" in row and isinstance(row.get("native_comfy_url"), str):
        return row["native_comfy_url"].strip().rstrip("/")
    return (settings.native_comfy_url or "").strip().rstrip("/")


def extract_tunnel_token(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""
    match = _TOKEN_FLAG_RE.search(text)
    if match:
        token = match.group(1).strip()
    else:
        token = text.strip("'\"")
    if not _SAFE_TOKEN_RE.fullmatch(token):
        raise APIError(
            "NATIVE_SETUP_INPUT",
            "Cloudflare Tunnel Token 格式无效；请直接粘贴 token 字符串，或粘贴完整的 cloudflared 安装/运行命令",
            422,
        )
    return token


def inspect_tunnel_token(token: str) -> dict:
    if not token:
        return {"valid_cf_format": False, "tunnel_id": None, "account_id_masked": None}
    padded = token + "=" * (-len(token) % 4)
    for decoder in (base64.b64decode, base64.urlsafe_b64decode):
        try:
            decoded = decoder(padded.encode("ascii"))
            payload = json.loads(decoded.decode("utf-8"))
        except Exception:
            continue
        if (
            isinstance(payload, dict)
            and isinstance(payload.get("a"), str)
            and isinstance(payload.get("t"), str)
            and isinstance(payload.get("s"), str)
            and payload["a"]
            and payload["t"]
            and payload["s"]
        ):
            acc = payload["a"]
            masked = f"{acc[:6]}…{acc[-4:]}" if len(acc) > 12 else acc[:4] + "…"
            return {"valid_cf_format": True, "tunnel_id": payload["t"], "account_id_masked": masked}
    return {"valid_cf_format": False, "tunnel_id": None, "account_id_masked": None}


def extract_comfy_commit(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""
    match = _GITHUB_COMMIT_RE.search(text)
    if match:
        text = match.group(1)
    commit = text.lower()
    if not _COMMIT_RE.fullmatch(commit):
        raise APIError(
            "NATIVE_SETUP_INPUT",
            "ComfyUI 版本必须为 40 位十六进制 Commit SHA（或直接粘贴 Comfy-Org/ComfyUI 的 commit 链接）",
            422,
        )
    return commit


def _is_local_hostname(host: str) -> bool:
    lower = host.lower().strip(".")
    if lower in {"localhost", "127.0.0.1", "::1"} or lower.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(lower).is_loopback
    except ValueError:
        return False


def _is_forbidden_backend_host(host: str) -> bool:
    lower = host.lower().strip(".")
    if not lower or _is_local_hostname(lower) or lower.endswith((".local", ".internal")):
        return True
    try:
        addr = ipaddress.ip_address(lower)
    except ValueError:
        return False
    return not addr.is_global


def validate_native_comfy_url(raw: str) -> str:
    text = (raw or "").strip().rstrip("/")
    if not text:
        return ""
    parsed = urlsplit(text)
    host = (parsed.hostname or "").lower()
    is_local_http = parsed.scheme == "http" and _is_local_hostname(host)
    if (
        (parsed.scheme != "https" and not is_local_http)
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.path
        or parsed.query
        or parsed.fragment
    ):
        raise APIError(
            "NATIVE_SETUP_INPUT",
            "原生网页域名必须为独立 HTTPS 源地址（如 https://comfy.example.com，或本机 http://comfy.localhost:8788），不能带路径或查询参数",
            422,
        )
    site_netloc = urlsplit((settings.site_url or "").strip().rstrip("/")).netloc.lower()
    if site_netloc and parsed.netloc.lower() == site_netloc:
        raise APIError("NATIVE_SETUP_INPUT", "原生网页域名不能与 Studio 控制台域名（SITE_URL）相同", 422)
    return f"{parsed.scheme}://{parsed.netloc.lower()}"


def validate_backend_url(raw: str, *, native_url: str = "") -> str:
    text = (raw or "").strip().rstrip("/")
    if not text:
        raise APIError("NATIVE_SETUP_INPUT", "请填写 Cloudflare 内部隧道域名（如 https://colab-backend.example.com）", 422)
    parsed = urlsplit(text)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.path
        or parsed.query
        or parsed.fragment
    ):
        raise APIError(
            "NATIVE_SETUP_INPUT",
            "内部隧道域名必须为独立 HTTPS 源地址（如 https://colab-backend.example.com），不能带路径或查询参数",
            422,
        )
    if _is_forbidden_backend_host(parsed.hostname):
        raise APIError("NATIVE_SETUP_INPUT", "内部隧道域名必须为公网 Cloudflare Tunnel 域名，不能使用本机或内网地址", 422)
    netloc = parsed.netloc.lower()
    site_netloc = urlsplit((settings.site_url or "").strip().rstrip("/")).netloc.lower()
    native_netloc = urlsplit(native_url.rstrip("/")).netloc.lower() if native_url else ""
    if netloc in {site_netloc, native_netloc} - {""}:
        raise APIError("NATIVE_SETUP_INPUT", "内部隧道域名必须与 Studio 域名和原生网页域名互不相同", 422)
    return f"https://{netloc}"


def _parse_access_input(body: dict) -> tuple[str | None, str | None]:
    if "access_json" in body and body["access_json"] not in (None, ""):
        raw = body["access_json"]
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except ValueError as exc:
                raise APIError("NATIVE_SETUP_INPUT", "Access JSON 格式无效，应为 {\"client_id\":\"...\",\"client_secret\":\"...\"}", 422) from exc
        elif isinstance(raw, dict):
            parsed = raw
        else:
            raise APIError("NATIVE_SETUP_INPUT", "Access JSON 格式无效", 422)
        if not isinstance(parsed, dict):
            raise APIError("NATIVE_SETUP_INPUT", "Access JSON 必须为 JSON 对象", 422)
        cid = parsed.get("client_id")
        csec = parsed.get("client_secret")
        if not isinstance(cid, str) or not cid.strip() or not isinstance(csec, str) or not csec.strip():
            raise APIError("NATIVE_SETUP_INPUT", "Access JSON 必须同时包含非空的 client_id 和 client_secret", 422)
        return cid.strip(), csec.strip()
    cid_raw = body.get("access_client_id")
    csec_raw = body.get("access_client_secret")
    cid = cid_raw.strip() if isinstance(cid_raw, str) else None
    csec = csec_raw.strip() if isinstance(csec_raw, str) else None
    return cid, csec


def _has_active_native_session(repo, owner: str) -> bool:
    from . import native

    if native.active_lease(repo, owner):
        return True
    try:
        jobs = repo.list("jobs", owner)
    except Exception:
        return False
    return any(
        (job.get("snapshot") or {}).get("native_comfy") and job.get("state") not in native.TERMINAL
        for job in jobs
    )


def _prerequisites(repo, owner: str) -> dict:
    from .accounts import AccountService
    from .compute import GpuConfig
    from .google import GoogleService

    cfg = GpuConfig.from_env()
    try:
        accounts = AccountService(repo, owner).list()
    except Exception:
        accounts = []
    oauth_configured = False
    try:
        status = GoogleService(repo, owner).status()
        oauth_configured = bool(
            (status.get("colab") or {}).get("configured")
            or (status.get("drive") or {}).get("configured")
        )
    except Exception:
        oauth_configured = False
    if not oauth_configured:
        oauth_configured = any(
            bool((a.get("authorization") or {}).get("colab", {}).get("configured"))
            or bool((a.get("authorization") or {}).get("drive", {}).get("configured"))
            for a in accounts
        )
    eligible = [a for a in accounts if a.get("enabled", True) is not False and a.get("allow_allocation") is not False]
    colab_ok = [a for a in eligible if (a.get("authorization") or {}).get("colab", {}).get("connected")]
    ready_accounts = [
        a for a in colab_ok if (a.get("authorization") or {}).get("drive", {}).get("connected")
    ]
    return {
        "real_gpu_enabled": bool(cfg.real_enabled),
        "oauth_configured": oauth_configured,
        "total_accounts": len(accounts),
        "eligible_accounts": len(eligible),
        "colab_connected_accounts": len(colab_ok),
        "ready_accounts": len(ready_accounts),
    }


def public_setup(repo, owner: str) -> dict:
    cfg = resolve(repo, owner, use_cache=False, strict_decrypt=False)
    bind_port_raw = os.getenv("BIND_PORT", "8788").strip()
    bind_port = int(bind_port_raw) if bind_port_raw.isdigit() else 8788
    tunnel_info = inspect_tunnel_token(cfg.tunnel_token) if cfg.tunnel_token else {
        "valid_cf_format": False,
        "tunnel_id": None,
        "account_id_masked": None,
    }
    return {
        "configured": cfg.tunnel_configured,
        "web_ui_configured": cfg.web_ui_configured,
        "has_db_override": cfg.has_db_override,
        "updated_at": cfg.updated_at,
        "recommended_commit": RECOMMENDED_COMFY_COMMIT,
        "site_url": (settings.site_url or "").strip().rstrip("/"),
        "bind_port": bind_port,
        "active_session_locked": _has_active_native_session(repo, owner),
        "values": {
            "native_comfy_url": cfg.native_comfy_url,
            "native_comfy_backend_url": cfg.native_comfy_backend_url,
            "native_comfy_commit": cfg.native_comfy_commit,
            "access_client_id": cfg.access_client_id,
        },
        "secrets": {
            "access_client_secret_configured": bool(cfg.access_client_secret),
            "tunnel_token_configured": bool(cfg.tunnel_token),
            "tunnel_info": tunnel_info,
        },
        "sources": {
            "native_comfy_url": cfg.native_comfy_url_source,
            "native_comfy_backend_url": cfg.native_comfy_backend_url_source,
            "native_comfy_commit": cfg.native_comfy_commit_source,
            "access_client_id": cfg.access_client_id_source,
            "access_client_secret": cfg.access_client_secret_source,
            "tunnel_token": cfg.tunnel_token_source,
        },
        "prerequisites": _prerequisites(repo, owner),
    }


def save_setup(repo, owner: str, body: dict) -> dict:
    if not isinstance(body, dict):
        raise APIError("NATIVE_SETUP_INPUT", "配置参数必须是对象", 422)
    allowed_fields = {
        "native_comfy_url",
        "native_comfy_backend_url",
        "native_comfy_commit",
        "access_client_id",
        "access_client_secret",
        "access_json",
        "tunnel_token",
    }
    unknown = set(body) - allowed_fields
    if unknown:
        raise APIError("NATIVE_SETUP_INPUT", f"不支持的配置字段：{', '.join(sorted(unknown))}", 422)

    store = _unwrap_store(repo)
    with store.dispatch_lock() as acquired:
        if not acquired:
            raise APIError("NATIVE_SETUP_BUSY", "原生配置正在保存，请稍后重试", 503, True)

        current = resolve(repo, owner, use_cache=False, strict_decrypt=False)
        existing_row = _load_db_row(repo, owner, use_cache=False)

        raw_url = body["native_comfy_url"] if "native_comfy_url" in body else current.native_comfy_url
        if not isinstance(raw_url, str):
            raise APIError("NATIVE_SETUP_INPUT", "原生网页域名必须为字符串", 422)
        native_url = validate_native_comfy_url(raw_url)

        raw_backend = body.get("native_comfy_backend_url", current.native_comfy_backend_url)
        if not isinstance(raw_backend, str):
            raise APIError("NATIVE_SETUP_INPUT", "内部隧道域名必须为字符串", 422)
        backend_url = validate_backend_url(raw_backend, native_url=native_url)

        raw_commit = body.get("native_comfy_commit")
        if raw_commit is None or (isinstance(raw_commit, str) and not raw_commit.strip()):
            commit = current.native_comfy_commit or RECOMMENDED_COMFY_COMMIT
        elif isinstance(raw_commit, str):
            commit = extract_comfy_commit(raw_commit)
        else:
            raise APIError("NATIVE_SETUP_INPUT", "ComfyUI 版本必须为字符串", 422)
        if not _COMMIT_RE.fullmatch(commit):
            commit = extract_comfy_commit(commit)

        parsed_cid, parsed_csec = _parse_access_input(body)
        cid = parsed_cid if parsed_cid is not None else current.access_client_id
        if not cid or not _SAFE_HEADER_RE.fullmatch(cid):
            raise APIError("NATIVE_SETUP_INPUT", "请填写有效的 Cloudflare Access Client ID", 422)

        cid_changed = cid != current.access_client_id
        if parsed_csec:
            if not _SAFE_HEADER_RE.fullmatch(parsed_csec):
                raise APIError("NATIVE_SETUP_INPUT", "Cloudflare Access Client Secret 格式无效", 422)
            effective_csec = parsed_csec
            encrypted_csec = encrypt_secret(parsed_csec)
        else:
            if cid_changed:
                raise APIError("NATIVE_SETUP_INPUT", "更换 Client ID 时必须同时提供对应的 Client Secret", 422)
            if not current.access_client_secret:
                raise APIError("NATIVE_SETUP_INPUT", "请填写 Cloudflare Access Client Secret", 422)
            effective_csec = current.access_client_secret
            encrypted_csec = (existing_row or {}).get("encrypted_access_client_secret")

        raw_tunnel = body.get("tunnel_token")
        if raw_tunnel is not None and not isinstance(raw_tunnel, str):
            raise APIError("NATIVE_SETUP_INPUT", "Tunnel Token 必须为字符串", 422)
        if isinstance(raw_tunnel, str) and raw_tunnel.strip():
            extracted_tunnel = extract_tunnel_token(raw_tunnel)
            effective_tunnel = extracted_tunnel
            encrypted_tunnel = encrypt_secret(extracted_tunnel)
        else:
            if not current.tunnel_token:
                raise APIError("NATIVE_SETUP_INPUT", "请填写 Cloudflare Tunnel Token", 422)
            effective_tunnel = current.tunnel_token
            encrypted_tunnel = (existing_row or {}).get("encrypted_tunnel_token")

        connection_changed = (
            backend_url != current.native_comfy_backend_url
            or cid != current.access_client_id
            or effective_csec != current.access_client_secret
            or effective_tunnel != current.tunnel_token
        )
        if connection_changed and _has_active_native_session(repo, owner):
            raise APIError(
                "NATIVE_SETUP_LOCKED",
                "当前有运行中或启动中的原生 ComfyUI 实例，请先停止旧实例后再修改隧道或 Access 凭据",
                409,
            )

        record_id = SETTING_PREFIX + owner
        payload = {
            "kind": RECORD_KIND,
            "native_comfy_url": native_url,
            "native_comfy_backend_url": backend_url,
            "native_comfy_commit": commit,
            "access_client_id": cid,
            "encrypted_access_client_secret": encrypted_csec,
            "encrypted_tunnel_token": encrypted_tunnel,
            "saved_at": utc_isoformat(utcnow()),
        }
        existing_any = None
        try:
            existing_any = store.get("settings", record_id, owner)
        except TypeError:
            existing_any = store.get("settings", record_id)
        if existing_any:
            store.update("settings", record_id, payload, owner)
        else:
            store.create("settings", owner, payload, record_id=record_id)
        invalidate_cache(store, owner)
    return public_setup(repo, owner)


def clear_setup(repo, owner: str) -> dict:
    store = _unwrap_store(repo)
    with store.dispatch_lock() as acquired:
        if not acquired:
            raise APIError("NATIVE_SETUP_BUSY", "原生配置正在更新，请稍后重试", 503, True)
        existing_row = _load_db_row(repo, owner, use_cache=False)
        if existing_row is not None:
            current = resolve(repo, owner, use_cache=False, strict_decrypt=False)
            fallback = resolve(None, owner, use_cache=False, strict_decrypt=False)
            connection_changed = (
                current.native_comfy_backend_url != fallback.native_comfy_backend_url
                or current.access_client_id != fallback.access_client_id
                or current.access_client_secret != fallback.access_client_secret
                or current.tunnel_token != fallback.tunnel_token
            )
            if connection_changed and _has_active_native_session(repo, owner):
                raise APIError(
                    "NATIVE_SETUP_LOCKED",
                    "当前有运行中或启动中的原生 ComfyUI 实例，请先停止旧实例后再清除网页配置",
                    409,
                )
            record_id = SETTING_PREFIX + owner
            try:
                store.delete("settings", record_id, owner)
            except KeyError:
                pass
            invalidate_cache(store, owner)
    return public_setup(repo, owner)


def _assert_public_dns(url: str) -> None:
    host = (urlsplit(url).hostname or "").strip()
    if _is_forbidden_backend_host(host):
        raise APIError("NATIVE_DIAG_SSRF", "拒绝向本机或内网地址发起探测请求", 422)
    try:
        infos = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise APIError("NATIVE_DIAG_DNS", f"域名 DNS 解析失败：{host}", 422) from exc
    for info in infos:
        raw_ip = info[4][0].split("%", 1)[0]
        addr = ipaddress.ip_address(raw_ip)
        if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped:
            addr = addr.ipv4_mapped
        if not addr.is_global:
            raise APIError("NATIVE_DIAG_SSRF", f"域名 {host} 解析到了非公网地址，已拒绝探测", 422)


def diagnose_setup(repo, owner: str, *, client: httpx.Client | None = None) -> dict:
    from .native import UNAUTHENTICATED_TUNNEL_PROBES

    cfg = resolve(repo, owner, use_cache=False, strict_decrypt=False)
    prereq = _prerequisites(repo, owner)
    checks: list[dict] = []

    # 1. Real GPU switch
    if prereq["real_gpu_enabled"]:
        checks.append({
            "id": "real_gpu",
            "label": "真实 GPU 开关",
            "status": "pass",
            "detail": "GPU_REAL_ENABLED=1 已开启",
        })
    else:
        checks.append({
            "id": "real_gpu",
            "label": "真实 GPU 开关",
            "status": "fail",
            "detail": "当前仍为离线/模拟模式（GPU_REAL_ENABLED=0）",
            "hint": "请在服务器 .env 中设置 GPU_REAL_ENABLED=1 后重启服务。",
        })

    # 2. Google accounts (Colab + Drive)
    if prereq["ready_accounts"] > 0:
        checks.append({
            "id": "google_accounts",
            "label": "Google Colab 与 Drive 账号",
            "status": "pass",
            "detail": f"已有 {prereq['ready_accounts']} 个账号同时就绪（启用、允许分配 GPU，且已连接 Colab 与 Drive）",
        })
    else:
        if not prereq["oauth_configured"]:
            detail = "尚未配置 Google OAuth 应用客户端"
            hint = "请前往「设置 → Google 账号」导入 Web 应用客户端 JSON，再授权 Colab 与 Drive。"
        elif prereq["eligible_accounts"] == 0:
            detail = "没有已启用且允许分配 GPU 的 Google 账号"
            hint = "请前往「设置 → Google 账号」添加账号，并确保开启「允许分配 GPU」。"
        elif prereq["colab_connected_accounts"] == 0:
            detail = "已有可用账号，但尚未完成 Google Colab 授权"
            hint = "请前往「设置 → Google 账号」点击「连接 Google Colab」。"
        else:
            detail = "已连接 Colab，但尚未连接或绑定 Google Drive"
            hint = "原生 ComfyUI 依赖 Drive 持久化工作流、插件与缓存；请前往「设置 → Google 账号」连接 Google Drive 或绑定共享 Drive 账号。"
        checks.append({
            "id": "google_accounts",
            "label": "Google Colab 与 Drive 账号",
            "status": "fail",
            "detail": detail,
            "hint": hint,
        })

    # 3. ComfyUI commit
    if cfg.native_comfy_commit and _COMMIT_RE.fullmatch(cfg.native_comfy_commit):
        rec_note = "（已验证推荐版本）" if cfg.native_comfy_commit == RECOMMENDED_COMFY_COMMIT else "（自定义版本）"
        checks.append({
            "id": "comfy_commit",
            "label": "ComfyUI 锁定版本",
            "status": "pass",
            "detail": f"已锁定提交 {cfg.native_comfy_commit[:12]} {rec_note}",
        })
    else:
        checks.append({
            "id": "comfy_commit",
            "label": "ComfyUI 锁定版本",
            "status": "fail",
            "detail": "尚未设置有效的 40 位官方 ComfyUI Commit SHA",
            "hint": f"建议直接使用内置已验证版本 {RECOMMENDED_COMFY_COMMIT[:12]}。",
        })

    # 4. Tunnel token format
    if not cfg.tunnel_token:
        checks.append({
            "id": "tunnel_token",
            "label": "Cloudflare Tunnel Token",
            "status": "fail",
            "detail": "尚未填写 Cloudflare Named Tunnel Token",
            "hint": "在 Cloudflare Zero Trust 创建 Named Tunnel 后，复制连接命令中的 Token 粘贴到配置中。",
        })
    else:
        info = inspect_tunnel_token(cfg.tunnel_token)
        if info["valid_cf_format"]:
            checks.append({
                "id": "tunnel_token",
                "label": "Cloudflare Tunnel Token",
                "status": "pass",
                "detail": f"格式有效 · Tunnel ID: {info['tunnel_id']}（账号 {info['account_id_masked']}）",
            })
        else:
            checks.append({
                "id": "tunnel_token",
                "label": "Cloudflare Tunnel Token",
                "status": "warn",
                "detail": "已配置 Token，但未能解析为标准 Cloudflare Named Tunnel Base64 JSON 结构",
                "hint": "请核对粘贴的是否为完整的 cloudflared tunnel connector token（通常以 eyJ 开头）。",
            })

    # 5. Cloudflare Access policy + Service Token
    if not cfg.native_comfy_backend_url or not cfg.access_client_id or not cfg.access_client_secret:
        missing = []
        if not cfg.native_comfy_backend_url:
            missing.append("内部隧道域名")
        if not cfg.access_client_id:
            missing.append("Access Client ID")
        if not cfg.access_client_secret:
            missing.append("Access Client Secret")
        checks.append({
            "id": "backend_access",
            "label": "Cloudflare Access 防护与凭据",
            "status": "fail",
            "detail": f"缺少必填项：{'、'.join(missing)}",
            "hint": "请先补齐内部隧道域名和 Access Service Token 并保存。",
        })
    elif offline.enabled() and client is None:
        checks.append({
            "id": "backend_access",
            "label": "Cloudflare Access 防护与凭据",
            "status": "skip",
            "detail": "当前处于离线测试模式，已跳过外部 Cloudflare 网络探测",
        })
    else:
        try:
            if client is None:
                _assert_public_dns(cfg.native_comfy_backend_url)
            headers = {
                "CF-Access-Client-Id": cfg.access_client_id,
                "CF-Access-Client-Secret": cfg.access_client_secret,
                "User-Agent": "colab-gpu-studio-proxy/1.0",
            }
            probe_ua = {"User-Agent": "colab-gpu-studio-proxy/1.0"}
            owns = client is None
            http = httpx.Client(timeout=8, follow_redirects=False) if owns else client
            try:
                with ThreadPoolExecutor(max_workers=6) as pool:
                    unauth_futures = [
                        (path, pool.submit(http.get, cfg.native_comfy_backend_url + path, headers=probe_ua))
                        for path in UNAUTHENTICATED_TUNNEL_PROBES
                    ]
                    auth_future = pool.submit(
                        http.get, cfg.native_comfy_backend_url + "/system_stats", headers=headers
                    )
                    leaked = []
                    for path, fut in unauth_futures:
                        resp = fut.result()
                        if resp.status_code not in {401, 403, 302}:
                            leaked.append(f"{path} (HTTP {resp.status_code})")
                    auth_resp = auth_future.result()
            finally:
                if owns:
                    http.close()

            if leaked:
                checks.append({
                    "id": "backend_access",
                    "label": "Cloudflare Access 防护与凭据",
                    "status": "fail",
                    "detail": f"检测到未鉴权路径未被 Access 拦截：{', '.join(leaked)}",
                    "hint": "请确认已为内部隧道域名整个主机配置 Self-hosted Access Application，策略为 Service Auth，且未配置任何 Bypass 规则。",
                })
            elif auth_resp.status_code in {401, 403, 302}:
                checks.append({
                    "id": "backend_access",
                    "label": "Cloudflare Access 防护与凭据",
                    "status": "fail",
                    "detail": f"Access 拦截已生效，但 Service Token 被拒绝（HTTP {auth_resp.status_code}）",
                    "hint": "请核对 Client ID / Client Secret 是否粘贴完整，且 Access Application 的 Service Auth 策略已勾选该 Service Token。",
                })
            elif auth_resp.status_code == 200:
                checks.append({
                    "id": "backend_access",
                    "label": "Cloudflare Access 防护与凭据",
                    "status": "pass",
                    "detail": "5 条未鉴权路径全部被拦截，Service Token 鉴权通过，且当前隧道已在线（HTTP 200）",
                })
            elif auth_resp.status_code in {502, 503, 504, 530}:
                checks.append({
                    "id": "backend_access",
                    "label": "Cloudflare Access 防护与凭据",
                    "status": "pass",
                    "detail": f"5 条未鉴权路径全部被拦截，Service Token 已成功穿透 Access 到达隧道入口（当前 Colab 未开机返回 HTTP {auth_resp.status_code}，属预期表现）",
                })
            else:
                checks.append({
                    "id": "backend_access",
                    "label": "Cloudflare Access 防护与凭据",
                    "status": "warn",
                    "detail": f"Access 拦截正常，携带 Service Token 探测返回 HTTP {auth_resp.status_code}",
                    "hint": "请检查 Cloudflare Tunnel 的 Public Hostname 路由是否指向 http://127.0.0.1:8188。",
                })
        except APIError as exc:
            checks.append({
                "id": "backend_access",
                "label": "Cloudflare Access 防护与凭据",
                "status": "fail",
                "detail": exc.message,
                "hint": "请检查内部隧道域名是否已在 Cloudflare Tunnel 的 Published application routes / Public Hostname 中添加并生效。",
            })
        except httpx.HTTPError as exc:
            checks.append({
                "id": "backend_access",
                "label": "Cloudflare Access 防护与凭据",
                "status": "fail",
                "detail": f"无法连接内部隧道域名（{type(exc).__name__}）",
                "hint": "请确认域名 DNS 解析与 Cloudflare 代理已生效，且 VPS 可正常访问 Cloudflare 边缘网络。",
            })

    # 6. Public Web UI URL (optional for API-only mode)
    if not cfg.native_comfy_url:
        checks.append({
            "id": "public_web_url",
            "label": "原生网页反代域名（可选）",
            "status": "skip",
            "detail": "当前未配置原生网页域名（仅 API 模式）：可直接在「API 设置」使用子路径短码或独立端口接入酒馆",
            "hint": "如需在浏览器中打开官方 ComfyUI 节点编辑器界面，请补充此域名并配置反向代理。",
        })
    else:
        parsed_web = urlsplit(cfg.native_comfy_url)
        if _is_local_hostname(parsed_web.hostname or ""):
            checks.append({
                "id": "public_web_url",
                "label": "原生网页反代域名",
                "status": "pass",
                "detail": f"已配置本机调试地址 {cfg.native_comfy_url}（跳过公网回环探测）",
            })
        elif offline.enabled() and client is None:
            checks.append({
                "id": "public_web_url",
                "label": "原生网页反代域名",
                "status": "skip",
                "detail": f"已配置 {cfg.native_comfy_url}（离线测试模式跳过公网探测）",
            })
        else:
            try:
                if client is None:
                    _assert_public_dns(cfg.native_comfy_url)
                owns = client is None
                http = httpx.Client(timeout=8, follow_redirects=False) if owns else client
                try:
                    probe_headers = {"User-Agent": "colab-gpu-studio-proxy/1.0"}
                    if settings.studio_api_token:
                        probe_headers["Authorization"] = f"Bearer {settings.studio_api_token}"
                    resp = http.get(cfg.native_comfy_url + "/_studio/api/status", headers=probe_headers)
                finally:
                    if owns:
                        http.close()

                ctype = resp.headers.get("content-type", "").lower()
                is_json = "application/json" in ctype
                body_data = None
                if is_json:
                    try:
                        body_data = resp.json()
                    except ValueError:
                        body_data = None

                if (
                    resp.status_code == 200
                    and isinstance(body_data, dict)
                    and isinstance(body_data.get("ready"), bool)
                ) or (
                    resp.status_code == 401
                    and isinstance(body_data, dict)
                    and body_data.get("error") == "UNAUTHENTICATED"
                ):
                    checks.append({
                        "id": "public_web_url",
                        "label": "原生网页反代域名",
                        "status": "pass",
                        "detail": f"{cfg.native_comfy_url} DNS、HTTPS 与反向代理 Host 透传验证通过",
                    })
                elif resp.status_code in {401, 403, 302, 307}:
                    checks.append({
                        "id": "public_web_url",
                        "label": "原生网页反代域名",
                        "status": "warn",
                        "detail": f"DNS 与 HTTPS 正常，但服务端本机回连 {cfg.native_comfy_url} 返回 HTTP {resp.status_code}（可能被反代 IP 白名单、WAF 或前置认证网关拦截）",
                        "hint": "若你在反代前配置了 IP 白名单、防火墙或外部登录网关，此提示属正常现象；请在你的受信浏览器中直接访问验证。若未配置前置限制，请核对 Nginx/Caddy 规则。",
                    })
                else:
                    checks.append({
                        "id": "public_web_url",
                        "label": "原生网页反代域名",
                        "status": "fail",
                        "detail": f"探测 {cfg.native_comfy_url}/_studio/api/status 返回 HTTP {resp.status_code}，未命中本站原生网关",
                        "hint": "请检查 Nginx/Caddy 是否将该域名转发到 Studio API 端口，并保留原始 Host 头（proxy_set_header Host $http_host;）。",
                    })
            except APIError as exc:
                checks.append({
                    "id": "public_web_url",
                    "label": "原生网页反代域名",
                    "status": "warn",
                    "detail": f"{exc.message}（若服务器处于内网/回环限制环境可忽略，请用浏览器实测）",
                })
            except httpx.HTTPError as exc:
                checks.append({
                    "id": "public_web_url",
                    "label": "原生网页反代域名",
                    "status": "warn",
                    "detail": f"服务端回连 {cfg.native_comfy_url} 失败（{type(exc).__name__}）",
                    "hint": "可能是反向代理/TLS 尚未就绪，或云主机存在 NAT 回环限制；请先确认 Nginx/Caddy 配置已重载。",
                })

    ok_all = all(item["status"] != "fail" for item in checks)
    has_warnings = any(item["status"] == "warn" for item in checks)
    return {
        "ok": ok_all,
        "has_warnings": has_warnings,
        "checks": checks,
        "checked_at": utc_isoformat(utcnow()),
    }
