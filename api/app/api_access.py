"""Shared three-tier access control, URL token routing, and local/CORS guards for external ComfyUI APIs."""

from __future__ import annotations

import ipaddress
import os
import re
import secrets
from contextlib import nullcontext
from pathlib import Path
from urllib.parse import parse_qsl, quote, unquote, urlencode, urlsplit

from .common import APIError
from .config import settings

AUTH_MODES = ("strict", "url_path", "local_open")
PATH_CODE_RE = re.compile(r"[A-Za-z0-9_-]{16,64}")
DEFAULT_WEAK_KEY_ID = "__weak_default__"

_CREATION_TOKEN_PREFIXES = ("/comfy/c/", "/comfy/creation/t/")
_CREATION_OPEN_PREFIX = "/comfy/creation"
_NATIVE_TOKEN_PREFIXES = ("/comfy/n/", "/comfy/native/t/")
_NATIVE_OPEN_PREFIX = "/comfy/native"
_DEDICATED_TOKEN_PREFIXES = ("/t/", "/c/", "/n/")

_FORWARDED_HEADERS = (
    "x-forwarded-for",
    "x-real-ip",
    "cf-connecting-ip",
    "true-client-ip",
    "x-client-ip",
    "x-cluster-client-ip",
    "fastly-client-ip",
)
_CORS_ALLOW_METHODS = "GET, HEAD, POST, PUT, DELETE, OPTIONS"
_CORS_ALLOW_HEADERS = "Authorization, Content-Type, Idempotency-Key, Accept, X-Requested-With"

_NATIVE_BLOCKED_PREFIXES = (
    "/_studio",
    "/studio_native",
    "/manager",
    "/v2/manager",
    "/customnode",
    "/v2/customnode",
    "/model/install",
    "/v2/model/install",
    "/externalmodel",
    "/v2/externalmodel",
    "/snapshot",
    "/v2/snapshot",
    "/comfyui_manager",
    "/v2/comfyui_manager",
    "/component",
    "/v2/component",
    "/userdata",
    "/settings",
    "/free",
    "/internal",
    "/reboot",
)

_NATIVE_WEAK_GET_EXACT = frozenset(
    {
        "/system_stats",
        "/object_info",
        "/models",
        "/embeddings",
        "/extensions",
        "/queue",
        "/history",
        "/prompt",
        "/view",
        "/view_metadata",
        "/features",
    }
)
_NATIVE_WEAK_GET_PREFIXES = (
    "/object_info/",
    "/models/",
    "/history/",
    "/view_metadata/",
)
_NATIVE_WEAK_POST_EXACT = frozenset(
    {
        "/prompt",
        "/upload/image",
        "/upload/mask",
        "/interrupt",
        "/queue",
    }
)


def _lock(store):
    lock_fn = getattr(store, "admission_lock", None)
    return lock_fn() if callable(lock_fn) else nullcontext(True)


def peek_path_code(store, setting_id: str, owner: str = "owner") -> str | None:
    try:
        row = store.get("settings", setting_id, owner) or {}
    except TypeError:
        row = store.get("settings", setting_id) or {}
    code = row.get("path_code") if isinstance(row, dict) else None
    return code if isinstance(code, str) and PATH_CODE_RE.fullmatch(code) else None


def get_or_create_path_code(store, setting_id: str, owner: str = "owner") -> str:
    existing = peek_path_code(store, setting_id, owner)
    if existing:
        return existing
    return reset_path_code(store, setting_id, owner, preserve_existing=True)


def reset_path_code(store, setting_id: str, owner: str = "owner", *, preserve_existing: bool = False) -> str:
    code = secrets.token_urlsafe(18)
    with _lock(store) as acquired:
        if not acquired:
            raise APIError("SCHEDULER_BUSY", "正在更新接入短码，请重试", 503, True)
        row = store.get("settings", setting_id, owner)
        if preserve_existing and isinstance(row, dict):
            current = row.get("path_code")
            if isinstance(current, str) and PATH_CODE_RE.fullmatch(current):
                return current
        if row:
            store.update("settings", setting_id, {"path_code": code}, owner)
        else:
            store.create("settings", owner, {"path_code": code}, record_id=setting_id)
    return code


def normalize_origins(origins) -> list[str]:
    if origins is None:
        return []
    if not isinstance(origins, list) or len(origins) > 20:
        raise APIError("VALIDATION_ERROR", "允许的 Origin 列表最多包含 20 个有效源地址", 422)
    cleaned: list[str] = []
    for item in origins:
        if not isinstance(item, str):
            raise APIError("VALIDATION_ERROR", "Origin 必须为字符串", 422)
        raw = item.strip().rstrip("/")
        if not raw:
            continue
        parsed = urlsplit(raw)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.path or parsed.query or parsed.fragment:
            raise APIError("VALIDATION_ERROR", f"无效的 Origin 源地址：{raw[:120]}（示例：https://st.example.com）", 422)
        canonical = f"{parsed.scheme}://{parsed.netloc.lower()}"
        if canonical not in cleaned:
            cleaned.append(canonical)
    return cleaned


def validate_listen_port(port, *, kind: str, store, owner: str = "owner") -> int | None:
    if port is None or port == "":
        return None
    if type(port) is not int or not (1024 <= port <= 65535):
        raise APIError("VALIDATION_ERROR", "独立监听端口必须为 1024 至 65535 之间的整数", 422)
    site_port = urlsplit(settings.site_url).port
    bind_port_raw = os.getenv("BIND_PORT", "8788").strip()
    bind_port = int(bind_port_raw) if bind_port_raw.isdigit() else 8788
    if port in {site_port, bind_port}:
        raise APIError("PORT_CONFLICT", f"端口 {port} 与 Studio 主服务端口冲突，请更换其他端口", 409)
    other_setting = f"native-api:{owner}" if kind == "creation" else f"creation-api:{owner}"
    other_label = "原生 ComfyUI API" if kind == "creation" else "创作 API"
    other_row = store.get("settings", other_setting, owner) or {}
    other_port = (other_row.get("policy") or {}).get("listen_port")
    if other_port == port:
        raise APIError("PORT_CONFLICT", f"端口 {port} 已被{other_label}使用，请分别设置不同端口", 409)
    return port


def _strip_prefix(path: str, raw_path: bytes, prefix: str) -> tuple[str, bytes]:
    raw_prefix = prefix.encode("ascii")
    rest_path = path[len(prefix):]
    if raw_path.startswith(raw_prefix):
        rest_raw = raw_path[len(raw_prefix):]
    else:
        rest_raw = quote(rest_path, safe="/").encode("ascii")
    if not rest_path:
        return "/", b"/"
    if not rest_path.startswith("/"):
        return "/" + rest_path, b"/" + rest_raw
    return rest_path, rest_raw if rest_raw.startswith(b"/") else b"/" + rest_raw


def _extract_token_prefix(path: str, raw_path: bytes, prefixes: tuple[str, ...]):
    for prefix in prefixes:
        if path.startswith(prefix):
            remainder = path[len(prefix):]
            code, sep, _ = remainder.partition("/")
            if PATH_CODE_RE.fullmatch(code):
                full_prefix = prefix + code
                if not sep and len(remainder) != len(code):
                    continue
                stripped_path, stripped_raw = _strip_prefix(path, raw_path, full_prefix)
                return code, stripped_path, stripped_raw
    return None


def _extract_query_token(query_bytes: bytes) -> tuple[str | None, bytes]:
    if not query_bytes:
        return None, query_bytes
    try:
        query_str = query_bytes.decode("ascii")
    except UnicodeDecodeError:
        return None, query_bytes
    if "code=" not in query_str and "token=" not in query_str and "api_key=" not in query_str:
        return None, query_bytes
    pairs = parse_qsl(query_str, keep_blank_values=True)
    found: str | None = None
    kept: list[tuple[str, str]] = []
    for k, v in pairs:
        if found is None and k in {"code", "token", "api_key"} and PATH_CODE_RE.fullmatch(v):
            found = v
        else:
            kept.append((k, v))
    if found is None:
        return None, query_bytes
    return found, urlencode(kept).encode("ascii")


def match_gateway_route(
    scope: dict,
    *,
    kind: str,
    dedicated_host_url: str = "",
    dedicated_port: bool = False,
) -> dict | None:
    """Match incoming ASGI scope for creation or native gateway and return rewritten scope metadata."""
    path: str = scope.get("path", "/")
    raw_path: bytes = scope.get("raw_path") or quote(path, safe="/").encode("ascii")
    query_bytes: bytes = scope.get("query_string", b"")

    token_prefixes = _CREATION_TOKEN_PREFIXES if kind == "creation" else _NATIVE_TOKEN_PREFIXES
    open_prefix = _CREATION_OPEN_PREFIX if kind == "creation" else _NATIVE_OPEN_PREFIX

    # 1. Subpath with embedded URL token on any host (e.g. /comfy/c/<token>/... or /comfy/n/<token>/...)
    matched = _extract_token_prefix(path, raw_path, token_prefixes)
    if matched is not None:
        route_token, stripped_path, stripped_raw = matched
        return {
            "entry": "subpath_token",
            "route_token": route_token,
            "scope": _rewritten_scope(scope, stripped_path, stripped_raw, query_bytes),
        }

    # 2. Subpath without embedded URL token on any host (e.g. /comfy/creation/... or /comfy/native/...)
    if path == open_prefix or path.startswith(open_prefix + "/"):
        stripped_path, stripped_raw = _strip_prefix(path, raw_path, open_prefix)
        q_token, clean_query = _extract_query_token(query_bytes)
        return {
            "entry": "subpath",
            "route_token": q_token,
            "scope": _rewritten_scope(scope, stripped_path, stripped_raw, clean_query),
        }

    # 3. Dedicated port or dedicated Host domain
    host_matched = False
    if not dedicated_port and dedicated_host_url:
        parsed_target = urlsplit(dedicated_host_url)
        target_host = (parsed_target.hostname or "").lower()
        target_port = parsed_target.port or (443 if parsed_target.scheme == "https" else 80 if parsed_target.scheme == "http" else None)
        raw_host = next((v.decode("latin1").strip().lower() for k, v in scope.get("headers", []) if k == b"host"), "")
        parsed_req_host = urlsplit(f"{parsed_target.scheme or 'http'}://{raw_host}") if raw_host else None
        req_host = (parsed_req_host.hostname or "").lower() if parsed_req_host else ""
        req_port = (
            parsed_req_host.port
            if parsed_req_host and parsed_req_host.port is not None
            else (443 if parsed_target.scheme == "https" else 80 if parsed_target.scheme == "http" else None)
        )
        host_matched = bool(target_host and req_host == target_host and req_port == target_port)

    if dedicated_port or host_matched:
        matched = _extract_token_prefix(path, raw_path, _DEDICATED_TOKEN_PREFIXES)
        if matched is not None:
            route_token, stripped_path, stripped_raw = matched
            return {
                "entry": "port_token" if dedicated_port else "host_token",
                "route_token": route_token,
                "scope": _rewritten_scope(scope, stripped_path, stripped_raw, query_bytes),
            }
        q_token, clean_query = _extract_query_token(query_bytes)
        return {
            "entry": "port" if dedicated_port else "host",
            "route_token": q_token,
            "scope": _rewritten_scope(scope, path, raw_path, clean_query),
        }

    return None


def _rewritten_scope(scope: dict, path: str, raw_path: bytes, query_string: bytes) -> dict:
    if scope.get("path") == path and scope.get("raw_path") == raw_path and scope.get("query_string") == query_string:
        return scope
    updated = dict(scope)
    updated["path"] = path
    updated["raw_path"] = raw_path
    updated["query_string"] = query_string
    return updated


def _in_container() -> bool:
    return Path("/.dockerenv").exists() or os.getenv("STUDIO_CONTAINER") == "1"


def _is_test_runtime() -> bool:
    return bool(
        os.getenv("PYTEST_CURRENT_TEST")
        or os.getenv("STUDIO_OFFLINE_TEST") == "1"
        or settings.environment == "offline-test"
    )


def _ip_allowed(raw_ip: str, *, allow_lan: bool, allow_docker_loopback_nat: bool, is_peer: bool = False) -> bool:
    value = raw_ip.strip()
    if not value:
        return False
    if is_peer and value == "testclient" and _is_test_runtime():
        return True
    if is_peer and value == "localhost":
        return True
    # Strip IPv6 brackets and optional port (e.g. "[::1]:1234" or "127.0.0.1:1234" from RFC 7239 Forwarded)
    if value.startswith("["):
        end = value.find("]")
        if end == -1:
            return False
        value = value[1:end]
    elif value.count(":") == 1 and "." in value:
        value = value.split(":", 1)[0]
    # Strip IPv6 zone index if present
    if "%" in value:
        value = value.split("%", 1)[0]
    try:
        addr = ipaddress.ip_address(value)
    except ValueError:
        return False
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped:
        addr = addr.ipv4_mapped
    if addr.is_loopback:
        return True
    if addr.is_private or addr.is_link_local:
        return bool(allow_lan or allow_docker_loopback_nat)
    return False


def _header_values(scope: dict, headers, name: str) -> list[str]:
    """Return ALL values for a header name (never truncating duplicate Starlette headers)."""
    target = name.lower().encode("latin1")
    raw_headers = scope.get("headers") if isinstance(scope, dict) else None
    if raw_headers is not None:
        return [v.decode("latin1") for k, v in raw_headers if k.lower() == target]
    getlist = getattr(headers, "getlist", None) or getattr(headers, "get_list", None)
    if callable(getlist):
        return list(getlist(name))
    val = headers.get(name)
    return [val] if val else []


def _parse_rfc7239_forwarded(values: list[str]) -> tuple[bool, list[str]]:
    """Extract all `for=` values from RFC 7239 Forwarded headers; return (ok, ips)."""
    ips: list[str] = []
    for raw in values:
        for element in raw.split(","):
            found_for = False
            for param in element.split(";"):
                k, sep, v = param.strip().partition("=")
                if sep and k.strip().lower() == "for":
                    found_for = True
                    cleaned = v.strip().strip('"').strip()
                    if not cleaned or cleaned.lower() == "unknown" or cleaned.startswith("_"):
                        return False, []
                    ips.append(cleaned)
            if element.strip() and not found_for:
                return False, []
    return True, ips


def is_trusted_local_client(scope: dict, headers, *, allow_lan: bool = False) -> bool:
    """Verify request originates from loopback (or private LAN when allow_lan=True), resisting XFF spoofing and Docker NAT pitfalls."""
    client = scope.get("client")
    peer_ip = client[0] if isinstance(client, (list, tuple)) and client else ""

    forwarded_ips: list[str] = []
    for name in _FORWARDED_HEADERS:
        for raw in _header_values(scope, headers, name):
            if raw:
                forwarded_ips.extend(part.strip() for part in raw.split(",") if part.strip())

    rfc_forwarded = _header_values(scope, headers, "forwarded")
    if rfc_forwarded:
        ok_fwd, rfc_ips = _parse_rfc7239_forwarded(rfc_forwarded)
        if not ok_fwd:
            return False
        forwarded_ips.extend(rfc_ips)

    docker_loopback_nat = (
        not forwarded_ips
        and _in_container()
        and os.getenv("STUDIO_TRUST_DOCKER_BRIDGE_LOOPBACK") == "1"
    )

    if not _ip_allowed(peer_ip, allow_lan=allow_lan, allow_docker_loopback_nat=docker_loopback_nat, is_peer=True):
        return False
    for fwd in forwarded_ips:
        if not _ip_allowed(fwd, allow_lan=allow_lan, allow_docker_loopback_nat=False, is_peer=False):
            return False
    return True


def _is_loopback_host(hostname: str | None) -> bool:
    if not hostname:
        return False
    lower = hostname.lower()
    if lower in {"localhost", "127.0.0.1", "::1"} or lower.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(lower).is_loopback
    except ValueError:
        return False


def is_public_site_url(url: str | None = None) -> bool:
    target = url or settings.site_url
    hostname = urlsplit(target).hostname
    if not hostname or _is_loopback_host(hostname):
        return False
    try:
        addr = ipaddress.ip_address(hostname)
        return not (addr.is_loopback or addr.is_private or addr.is_link_local)
    except ValueError:
        return True


def cors_headers_for(origin: str | None, *, wildcard_fallback: bool = True) -> dict[str, str]:
    if origin and origin != "null":
        return {
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Methods": _CORS_ALLOW_METHODS,
            "Access-Control-Allow-Headers": _CORS_ALLOW_HEADERS,
            "Access-Control-Max-Age": "600",
            "Vary": "Origin",
        }
    if wildcard_fallback:
        return {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": _CORS_ALLOW_METHODS,
            "Access-Control-Allow-Headers": _CORS_ALLOW_HEADERS,
            "Access-Control-Max-Age": "600",
        }
    return {}


def check_weak_origin(
    headers,
    *,
    via_token: bool,
    allowed_origins: list[str] | None = None,
    extra_trusted_urls: tuple[str, ...] = (),
) -> dict[str, str]:
    """Validate browser Origin/Sec-Fetch-Site for weak/local API calls and return applicable CORS headers."""
    origin = headers.get("origin")
    if via_token:
        return cors_headers_for(origin, wildcard_fallback=True)

    sec_site = headers.get("sec-fetch-site", "").lower()
    if not origin:
        if sec_site == "cross-site":
            raise APIError("CSRF_ORIGIN", "跨站浏览器请求被拒绝", 403)
        return {}

    if origin == "null":
        raise APIError("CSRF_ORIGIN", "不信任匿名源（Origin: null）的免鉴权请求", 403)

    clean_origin = origin.rstrip("/")
    parsed = urlsplit(clean_origin)
    canonical = f"{parsed.scheme}://{parsed.netloc.lower()}" if parsed.scheme and parsed.netloc else ""

    trusted = {
        f"{p.scheme}://{p.netloc.lower()}"
        for u in (settings.site_url, settings.creation_comfy_url, settings.native_comfy_url, *extra_trusted_urls)
        if u and (p := urlsplit(u.rstrip("/"))).scheme and p.netloc
    }
    if allowed_origins:
        trusted.update(o.lower() for o in allowed_origins)

    if _is_loopback_host(parsed.hostname) or (canonical and canonical in trusted):
        return cors_headers_for(clean_origin, wildcard_fallback=False)

    raise APIError(
        "CSRF_ORIGIN",
        "免鉴权模式拒绝未知前端跨域直连；请改用「URL 路径短 Token」模式，或将该前端 Origin 加入白名单",
        403,
    )


def is_native_blocked_path(path: str, raw_path: bytes | None = None) -> bool:
    """Reject ambiguous paths before applying weak-access route restrictions."""
    if not path.startswith("/") or "//" in path or "\\" in path or re.search(r"[\x00-\x1f\x7f]", path):
        return True
    if {".", ".."}.intersection(path.split("/")) or re.search(r"%[0-9a-fA-F]{2}", path):
        return True
    if raw_path is not None:
        try:
            raw = raw_path.decode("ascii")
            if re.search(r"[?#\x00-\x20\x7f]|%(?![0-9a-fA-F]{2})|%2f", raw, re.I):
                return True
            if unquote(raw, errors="strict") != path:
                return True
        except UnicodeError:
            return True

    normalized = path.lower().rstrip("/")
    internal = normalized.removeprefix("/api") if normalized.startswith("/api/") else normalized
    return any(
        candidate == blocked or candidate.startswith(blocked + "/")
        for candidate in (normalized, internal)
        for blocked in _NATIVE_BLOCKED_PREFIXES
    )


def is_native_weak_allowed(
    path: str,
    raw_path: bytes | None = None,
    *,
    method: str = "GET",
    websocket: bool = False,
) -> bool:
    """Strict method+path allowlist for Tier-2 (url_path) and Tier-3 (local_open) native ComfyUI access."""
    if is_native_blocked_path(path, raw_path):
        return False
    normalized = path.lower().rstrip("/") or "/"
    internal = normalized.removeprefix("/api") if normalized.startswith("/api/") else normalized
    if websocket:
        return internal == "/ws"
    verb = method.upper()
    if verb in {"GET", "HEAD"}:
        return internal in _NATIVE_WEAK_GET_EXACT or internal.startswith(_NATIVE_WEAK_GET_PREFIXES)
    if verb == "POST":
        return internal in _NATIVE_WEAK_POST_EXACT
    return False


def build_access_urls(
    *,
    kind: str,
    auth_mode: str,
    path_code: str | None,
    listen_port: int | None,
    dedicated_url: str = "",
    allow_lan: bool = False,
) -> dict:
    site = settings.site_url.rstrip("/")
    dedicated = dedicated_url.rstrip("/") if dedicated_url else ""
    short = "c" if kind == "creation" else "n"
    open_subpath = f"{site}/comfy/{kind}"
    code_subpath = f"{site}/comfy/{short}/{path_code}" if path_code else ""
    dedicated_code = f"{dedicated}/t/{path_code}" if dedicated and path_code else ""

    port_base = f"http://127.0.0.1:{listen_port}" if listen_port else ""
    port_code = f"{port_base}/t/{path_code}" if port_base and path_code else ""

    if auth_mode == "url_path":
        recommended = dedicated_code or code_subpath
    elif auth_mode == "local_open":
        recommended = port_base or dedicated or open_subpath
    else:
        recommended = dedicated or open_subpath

    return {
        "recommended": recommended,
        "subpath_url": open_subpath,
        "subpath_code_url": code_subpath,
        "dedicated_url": dedicated,
        "dedicated_code_url": dedicated_code,
        "port_url": port_base,
        "port_code_url": port_code,
        "public_site_warning": is_public_site_url(site),
        "allow_lan": allow_lan,
    }
