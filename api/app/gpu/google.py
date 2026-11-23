"""Google OAuth 2.0 credentials for the independent Colab GPU Studio API.

The adapter deliberately stores only encrypted token values.  OAuth callback
state is also persisted as a short-lived credential record so a callback can
resolve its owner without trusting a browser cookie or a caller supplied owner.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import threading
import time
from contextlib import contextmanager
from importlib import resources
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from app.config import settings
from app.security import decrypt_secret, encrypt_secret

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
REVOKE_URL = "https://oauth2.googleapis.com/revoke"
COLAB_CLI_CLIENT_ID_PREFIX = "764086051850-"
COLAB_CLI_REDIRECT_URI = "https://sdk.cloud.google.com/applicationdefaultauthcode.html"
COLAB_CLI_SCOPES = (
    "openid",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/colaboratory",
)
COLAB_CLI_STATE_TTL_SECONDS = 1800


class GoogleError(RuntimeError):
    """Safe adapter error; upstream bodies are never included."""


class GoogleConfigurationError(GoogleError):
    pass


class GoogleOAuthError(GoogleError):
    pass


_PURPOSES = {
    "colab": "https://www.googleapis.com/auth/colaboratory",
    "drive": "https://www.googleapis.com/auth/drive.file",
}
_STATE_CONSUME_LOCK = threading.Lock()
_REFRESH_FALLBACK_LOCKS: dict[str, threading.Lock] = {}
_REFRESH_FALLBACK_GUARD = threading.Lock()
_ORIGINAL_HTTPX_CLIENT = httpx.Client
_SHARED_CLIENT_LOCK = threading.Lock()
_SHARED_HTTPX_CLIENT: httpx.Client | None = None
_CLI_AVAIL_CACHE: dict[tuple[str, str, str], bool] = {}
_TOKEN_CACHE_LOCK = threading.Lock()
_TOKEN_CACHE: dict[tuple[int, str, str | None, str], tuple[str, float]] = {}


def _shared_httpx_client() -> httpx.Client:
    global _SHARED_HTTPX_CLIENT
    client = _SHARED_HTTPX_CLIENT
    if client is not None and not client.is_closed:
        return client
    with _SHARED_CLIENT_LOCK:
        if _SHARED_HTTPX_CLIENT is None or _SHARED_HTTPX_CLIENT.is_closed:
            _SHARED_HTTPX_CLIENT = httpx.Client(
                timeout=httpx.Timeout(10.0),
                follow_redirects=False,
                limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
            )
        return _SHARED_HTTPX_CLIENT


def _token_cache_allowed(clock) -> bool:
    return clock is _now and not os.getenv("PYTEST_CURRENT_TEST") and os.getenv("STUDIO_OFFLINE_TEST") != "1"


def invalidate_token_cache(store=None, owner_id: str | None = None, account_id: str | None = None, purpose: str | None = None) -> None:
    store_id = id(store) if store is not None else None
    with _TOKEN_CACHE_LOCK:
        for key in list(_TOKEN_CACHE):
            if (
                (store_id is None or key[0] == store_id)
                and (owner_id is None or key[1] == owner_id)
                and (account_id is None or key[2] == account_id)
                and (purpose is None or key[3] == purpose)
            ):
                _TOKEN_CACHE.pop(key, None)


def _safe_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _now() -> float:
    return time.time()


def _is_cli_credential(credential: dict | None) -> bool:
    if not isinstance(credential, dict):
        return False
    if credential.get("auth_channel") == "colab_cli":
        return True
    cid = str(credential.get("client_id") or "").strip()
    return cid.startswith(COLAB_CLI_CLIENT_ID_PREFIX)


def load_colab_cli_oauth_client() -> tuple[str, str, str]:
    """Load the public Colab CLI OAuth client dynamically without hardcoding secrets."""
    env_id = os.getenv("GPU_COLAB_CLI_CLIENT_ID", "").strip()
    env_secret = os.getenv("GPU_COLAB_CLI_CLIENT_SECRET", "").strip()
    env_redirect = os.getenv("GPU_COLAB_CLI_REDIRECT_URI", "").strip() or COLAB_CLI_REDIRECT_URI
    if env_id or env_secret:
        if not env_id or not env_secret:
            raise GoogleConfigurationError("GPU_COLAB_CLI_CLIENT_ID 与 GPU_COLAB_CLI_CLIENT_SECRET 必须同时配置")
        return env_id, env_secret, env_redirect
    try:
        config_text = resources.files("colab_cli").joinpath("oauth_config.json").read_text(encoding="utf-8")
        raw = json.loads(config_text)
        installed = raw.get("installed") if isinstance(raw, dict) else None
        client_id = str(installed.get("client_id") or "").strip() if isinstance(installed, dict) else ""
        client_secret = str(installed.get("client_secret") or "").strip() if isinstance(installed, dict) else ""
    except Exception as exc:
        raise GoogleConfigurationError(
            "当前运行环境未安装 google-colab-cli（需 Python 3.12+ 或官方 Docker 镜像），无法生成 Colab CLI 授权链接"
        ) from exc
    if not client_id or not client_secret:
        raise GoogleConfigurationError("google-colab-cli 内置 OAuth 配置无效")
    return client_id, client_secret, COLAB_CLI_REDIRECT_URI


def colab_cli_oauth_available() -> bool:
    key = (
        os.getenv("GPU_COLAB_CLI_CLIENT_ID", "").strip(),
        os.getenv("GPU_COLAB_CLI_CLIENT_SECRET", "").strip(),
        os.getenv("GPU_COLAB_CLI_REDIRECT_URI", "").strip(),
    )
    if key in _CLI_AVAIL_CACHE and not os.getenv("PYTEST_CURRENT_TEST"):
        return _CLI_AVAIL_CACHE[key]
    try:
        load_colab_cli_oauth_client()
        available = True
    except GoogleConfigurationError:
        available = False
    if not os.getenv("PYTEST_CURRENT_TEST"):
        _CLI_AVAIL_CACHE[key] = available
    return available


def _extract_authorization_code(raw: Any) -> str:
    if not isinstance(raw, str):
        raise GoogleOAuthError("Colab CLI 授权码格式无效，请粘贴完整的授权码")
    text = raw.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1].strip()
    if text.startswith(("http://", "https://")):
        query = parse_qs(urlparse(text).query)
        extracted = (query.get("code") or [""])[0].strip()
        text = extracted
    elif "code=" in text:
        query = parse_qs(text.lstrip("?"))
        extracted = (query.get("code") or [""])[0].strip()
        if extracted:
            text = extracted
    if not text or len(text) > 8192 or any(ch.isspace() for ch in text):
        raise GoogleOAuthError("Colab CLI 授权码格式无效，请粘贴完整的授权码")
    return text


class GoogleService:
    """Owner-bound Google OAuth service.

    ``store`` is the independent project's Repository and must expose the
    repository CRUD surface.  Credentials are records with ``owner_id`` and
    ``purpose``.  No access or refresh token is returned from any public method.
    """

    def __init__(
        self,
        store,
        owner_id: str | None,
        *,
        account_id: str | None = None,
        client: httpx.Client | None = None,
        clock=_now,
        state_ttl_seconds: int = 600,
    ):
        self.store = store
        self.owner_id = owner_id
        # None is the legacy single-account slot. Explicit account ids never
        # fall back to that slot or to another account.
        self.account_id = account_id
        self._client = client
        self._owns_client = False
        self.clock = clock
        self.state_ttl_seconds = state_ttl_seconds

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            if httpx.Client is _ORIGINAL_HTTPX_CLIENT:
                self._client = _shared_httpx_client()
                self._owns_client = False
            else:
                self._client = httpx.Client(timeout=httpx.Timeout(10.0), follow_redirects=False)
                self._owns_client = True
        return self._client

    @client.setter
    def client(self, value: httpx.Client | None) -> None:
        self._client = value
        self._owns_client = False

    @staticmethod
    def callback_uri() -> str:
        return str(getattr(settings, "site_url", "")).rstrip("/") + "/api/gpu/auth/google/callback"

    def _account_matches(self, row: dict, account_id: str | None = None) -> bool:
        account = self.account_id if account_id is None else account_id
        row_account = row.get("account_id")
        owner = row.get("owner_id") or self.owner_id
        legacy_default = owner and self._legacy_default_id(owner)
        if account is None:
            return row_account in (None, "", legacy_default)
        if row_account == account:
            return True
        # The deterministic migrated default account is the same identity as
        # the old unscoped slot.  Other explicit accounts never use it.
        return account == legacy_default and row_account in (None, "")

    def _legacy_default_config(self, row: dict, owner_id: str) -> bool:
        """Return whether a migrated default app may be shared owner-wide.

        The migration temporarily attached the old owner-level OAuth app to the
        deterministic legacy account.  That row is safe to reuse for a newly
        created account only when the account directory marks that identity as
        the trusted legacy default.  A normal account-specific OAuth app never
        becomes owner-wide through this path.
        """
        default_id = self._legacy_default_id(owner_id)
        if row.get("account_id") != default_id:
            return False
        from app.oauth_config_routes import config_id

        if row.get("id") == config_id(owner_id, default_id):
            return False  # Newly saved dedicated apps are never legacy-shared.
        default = self.store.get("accounts", default_id, owner_id)
        return bool(default and (default.get("legacy_default") is True or default.get("legacy_trusted") is True))

    def _client_config(
        self, owner_id: str | None = None, account_id: str | None = None, *, credentials: list[dict] | None = None
    ) -> dict | None:
        owner = self.owner_id if owner_id is None else owner_id
        if not owner:
            return None
        source_rows = credentials if credentials is not None else self._list("credentials", owner)
        rows = [
            row for row in source_rows
            if row.get("provider") == "google"
            and row.get("purpose") == "oauth_client_config"
            and not row.get("revoked")
        ]
        # Web setup saves an owner-wide OAuth application, not an account token.
        # Prefer an account-specific application, then the same owner's shared
        # application. Never borrow another account's app or another owner's app.
        candidates = rows
        exact = [row for row in candidates if account_id is not None and row.get("account_id") == account_id]
        scoped = exact or [row for row in candidates if row.get("account_id") in (None, "")]
        if scoped:
            from app.oauth_config_routes import config_id

            canonical = config_id(owner, account_id) if account_id is not None else config_id(owner)
            rows = sorted(scoped, key=lambda row: row.get("id") != canonical)
        elif account_id is not None:
            # First preserve the original owner-wide application config. If
            # migration attached that same app to the trusted legacy default,
            # permit that exact compatibility case as the final fallback.
            rows = [row for row in rows if row.get("account_id") in (None, "")]
            if not rows:
                # Never borrow a regular account's app or any account's bearer
                # token through this compatibility path.
                rows = [row for row in candidates if self._legacy_default_config(row, owner)]
        else:
            rows = [row for row in candidates if self._legacy_default_config(row, owner)]
        if not rows:
            return None
        row = rows[0]
        try:
            secret = decrypt_secret(row["encrypted_client_secret"])
        except (KeyError, TypeError, ValueError) as exc:
            raise GoogleConfigurationError("Google OAuth配置不可用") from exc
        return {"client_id": str(row.get("client_id", "")).strip(), "client_secret": secret, "id": row.get("id")}

    @staticmethod
    def _legacy_default_id(owner_id: str) -> str:
        from .accounts import AccountService

        return AccountService.default_account_id(owner_id)

    def _config(
        self, owner_id: str | None = None, account_id: str | None = None, *, credentials: list[dict] | None = None
    ) -> tuple[str, str, str]:
        effective_owner = self.owner_id if owner_id is None else owner_id
        effective_account = self.account_id if account_id is None else account_id
        database = self._client_config(effective_owner, effective_account, credentials=credentials)
        if database:
            client_id, client_secret = database["client_id"], database["client_secret"]
        else:
            client_id = os.getenv("GPU_GOOGLE_CLIENT_ID", "").strip()
            client_secret = (
                os.getenv("GPU_GOOGLE_SECRET") or os.getenv("GPU_GOOGLE_CLIENT_SECRET") or ""
            ).strip()
        redirect = self.callback_uri()
        if not client_id or not client_secret or not redirect:
            raise GoogleConfigurationError("Google OAuth未配置，真实GPU能力已关闭")
        if not redirect.startswith("https://") and str(getattr(settings, "environment", "development")).lower() in {
            "prod",
            "production",
        }:
            raise GoogleConfigurationError("生产环境Google OAuth回调必须使用HTTPS")
        return client_id, client_secret, redirect

    def _list(self, kind: str, owner_id: str | None = None) -> list[dict]:
        return self.store.list(kind, owner_id)

    def _create(self, kind: str, owner_id: str, data: dict) -> dict:
        return self.store.create(kind, owner_id, data)

    def _update(self, kind: str, record_id: str, patch: dict, owner_id: str | None = None) -> dict:
        return self.store.update(kind, record_id, patch, owner_id)


    def _credential(
        self,
        purpose: str,
        owner_id: str | None = None,
        account_id: str | None = None,
        *,
        credentials: list[dict] | None = None,
    ) -> dict | None:
        owner = self.owner_id if owner_id is None else owner_id
        if not owner:
            return None
        source_rows = credentials if credentials is not None else self._list("credentials", owner)
        rows = [
            r
            for r in source_rows
            if r.get("purpose") == purpose
            and r.get("provider", "google") == "google"
            and not r.get("revoked")
            and self._account_matches(r, account_id)
        ]
        return rows[0] if rows else None

    def status(self, *, credentials: list[dict] | None = None) -> dict:
        creds = credentials if credentials is not None else (self._list("credentials", self.owner_id) if self.owner_id else [])
        configured = all(self._config_value(credentials=creds))
        cli_available = colab_cli_oauth_available()
        result: dict[str, dict[str, Any]] = {}
        for purpose in _PURPOSES:
            credential = self._credential(purpose, account_id=self.account_id, credentials=creds)
            scope_ok = bool(credential and self._scope_sufficient(purpose, credential.get("scope")))
            connected = bool(credential and credential.get("encrypted_refresh_token") and scope_ok)
            cred_configured = bool(
                credential
                and str(credential.get("client_id", "")).strip()
                and credential.get("encrypted_client_secret")
            )
            auth_channel = (
                ("colab_cli" if _is_cli_credential(credential) else "web_oauth")
                if connected
                else None
            )
            entry: dict[str, Any] = {
                "connected": connected,
                "configured": configured or cred_configured,
                "expires_at": credential.get("expires_at") if credential else None,
                "auth_channel": auth_channel,
            }
            if purpose == "colab":
                entry["cli_available"] = cli_available
            if credential and not scope_ok:
                entry["message"] = "Google权限不足，请重新授权"
            result[purpose] = entry
        return result

    @staticmethod
    def _scope_sufficient(purpose: str, scope: Any) -> bool:
        # Older records did not persist Google's scope response.  Explicitly
        # insufficient scopes fail closed; missing legacy values remain usable.
        if not scope:
            return True
        if isinstance(scope, (list, tuple, set)):
            values = {str(item) for item in scope}
        else:
            values = set(str(scope).split())
        return _PURPOSES[purpose] in values

    def _config_value(self, *, credentials: list[dict] | None = None) -> tuple[str, str, str]:
        try:
            return self._config(self.owner_id, self.account_id, credentials=credentials)
        except GoogleConfigurationError:
            return "", "", self.callback_uri()

    def invalidate_oauth_states(self, owner_id: str | None = None, *, account_id: str | None = None) -> None:
        owner = self.owner_id if owner_id is None else owner_id
        if not owner:
            return
        for row in self._list("credentials", owner):
            if account_id is not None and not self._account_matches(row, account_id):
                continue
            if row.get("purpose") == "_oauth_state" and not row.get("used_at"):
                self._update("credentials", row["id"], {"used_at": self.clock()}, owner)

    def start(self, purpose: str) -> dict:
        if purpose not in _PURPOSES:
            raise ValueError("Google OAuth purpose无效")
        if not self.owner_id:
            raise GoogleOAuthError("当前登录用户无效")
        client_id, _secret, redirect = self._config()
        state = secrets.token_urlsafe(32)
        verifier = secrets.token_urlsafe(64)
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
        self._create(
            "credentials",
            self.owner_id,
            {
                "provider": "google",
                "purpose": "_oauth_state",
                "oauth_purpose": purpose,
                **({"account_id": self.account_id} if self.account_id is not None else {}),
                "state_hash": _safe_id(state),
                "encrypted_code_verifier": encrypt_secret(verifier),
                "expires_at": self.clock() + self.state_ttl_seconds,
                "used_at": None,
            },
        )
        query = urlencode(
            {
                "client_id": client_id,
                "redirect_uri": redirect,
                "response_type": "code",
                "scope": _PURPOSES[purpose],
                "access_type": "offline",
                "prompt": "consent",
                "state": state,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
            }
        )
        return {"authorization_url": AUTH_URL + "?" + query, "state": state}

    def _take_state(self, state: str) -> dict:
        if not isinstance(state, str) or not state or len(state) > 512:
            raise GoogleOAuthError("OAuth state无效或已过期")
        # Lock only the read-and-consume transition.  The network token
        # exchange happens after this context is released.
        with self._state_consume_guard():
            rows = [
                r
                for r in self._list("credentials")
                if r.get("purpose") == "_oauth_state" and r.get("state_hash") == _safe_id(state)
            ]
            if len(rows) != 1:
                raise GoogleOAuthError("OAuth state无效或已过期")
            row = rows[0]
            if row.get("used_at") or row.get("used") or float(row.get("expires_at", 0)) <= self.clock():
                raise GoogleOAuthError("OAuth state无效或已过期")
            # Repository.dispatch_lock is the cross-process SQLite lock.  The
            # process lock keeps the intentionally minimal memory repo safe.
            self._update("credentials", row["id"], {"used_at": self.clock()}, row.get("owner_id"))
            return row

    @contextmanager
    def _state_consume_guard(self):
        dispatch = getattr(self.store, "dispatch_lock", None)
        with _STATE_CONSUME_LOCK:
            if callable(dispatch):
                with dispatch() as acquired:
                    if not acquired:
                        raise GoogleOAuthError("OAuth state无效或已过期")
                    yield
                return
            yield

    def callback(self, code: str, state: str) -> dict:
        if not isinstance(code, str) or not code or len(code) > 8192:
            raise GoogleOAuthError("Google OAuth code无效")
        row = self._take_state(state)
        # The callback starts without an owner.  The consumed state is the
        # only authority for selecting the owner's encrypted client config.
        client_id, client_secret, redirect = self._config(row.get("owner_id"), row.get("account_id"))
        try:
            verifier = decrypt_secret(row["encrypted_code_verifier"])
            response = self.client.post(
                TOKEN_URL,
                data={
                    "code": code,
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "redirect_uri": redirect,
                    "grant_type": "authorization_code",
                    "code_verifier": verifier,
                },
            )
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            raise GoogleOAuthError("Google OAuth token交换失败") from exc
        if response.status_code < 200 or response.status_code >= 300:
            raise GoogleOAuthError("Google OAuth token交换失败")
        try:
            payload = response.json()
            access = payload.get("access_token")
            refresh = payload.get("refresh_token")
            expires_in = int(payload.get("expires_in", 3600))
            if not isinstance(access, str) or not access:
                raise ValueError
        except (ValueError, TypeError, AttributeError):
            raise GoogleOAuthError("Google OAuth响应无效")
        old = self._credential(row.get("oauth_purpose"), row.get("owner_id"), row.get("account_id"))
        if not refresh and old:
            try:
                refresh = decrypt_secret(old["encrypted_refresh_token"])
            except (KeyError, TypeError, ValueError):
                refresh = None
        if not refresh:
            raise GoogleOAuthError("Google OAuth未返回可持久化刷新凭证")
        data = {
            "provider": "google",
            "purpose": row["oauth_purpose"],
            "auth_channel": "web_oauth",
            "client_id": None,
            "encrypted_client_secret": None,
            "encrypted_access_token": encrypt_secret(access),
            "encrypted_refresh_token": encrypt_secret(refresh),
            "expires_at": self.clock() + max(0, expires_in - 60),
            "scope": payload.get("scope", _PURPOSES[row["oauth_purpose"]]),
            "token_type": payload.get("token_type", "Bearer"),
            "revoked": False,
            **({"account_id": row.get("account_id")} if row.get("account_id") is not None else {}),
        }
        if old:
            self._update("credentials", old["id"], data, row.get("owner_id"))
        else:
            self._create("credentials", row["owner_id"], data)
        invalidate_token_cache(self.store, row.get("owner_id"), row.get("account_id"), row["oauth_purpose"])
        if row["oauth_purpose"] == "drive" and row.get("owner_id"):
            from .drive_usage import invalidate_drive_usage_cache
            from .native_outputs import invalidate_native_outputs_cache

            invalidate_native_outputs_cache(row["owner_id"])
            invalidate_drive_usage_cache(row["owner_id"])
        connected = self._scope_sufficient(row["oauth_purpose"], data["scope"])
        result = {"purpose": row["oauth_purpose"], "connected": connected, "owner_id": row["owner_id"]}
        if row.get("account_id") is not None:
            result["account_id"] = row["account_id"]
        if not connected:
            result["message"] = "Google权限不足，请重新授权"
        return result

    def start_colab_cli(self) -> dict:
        if not self.owner_id:
            raise GoogleOAuthError("当前登录用户无效")
        client_id, _secret, redirect = load_colab_cli_oauth_client()
        state = secrets.token_urlsafe(32)
        verifier = secrets.token_urlsafe(64)
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
        expires_at = self.clock() + COLAB_CLI_STATE_TTL_SECONDS
        self._create(
            "credentials",
            self.owner_id,
            {
                "provider": "google",
                "purpose": "_oauth_state",
                "oauth_purpose": "colab",
                "auth_channel": "colab_cli",
                **({"account_id": self.account_id} if self.account_id is not None else {}),
                "state_hash": _safe_id(state),
                "encrypted_code_verifier": encrypt_secret(verifier),
                "expires_at": expires_at,
                "used_at": None,
                "in_flight_until": None,
            },
        )
        query = urlencode(
            {
                "client_id": client_id,
                "redirect_uri": redirect,
                "response_type": "code",
                "scope": " ".join(COLAB_CLI_SCOPES),
                "access_type": "offline",
                "prompt": "consent",
                "token_usage": "remote",
                "state": state,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
            }
        )
        return {
            "authorization_url": AUTH_URL + "?" + query,
            "state": state,
            "expires_at": expires_at,
            "expires_in": COLAB_CLI_STATE_TTL_SECONDS,
            "auth_channel": "colab_cli",
            "scopes": list(COLAB_CLI_SCOPES),
        }

    def _begin_cli_state(self, state: str) -> dict:
        if not isinstance(state, str) or not state or len(state) > 512:
            raise GoogleOAuthError("Colab CLI 授权会话无效或已过期，请重新生成授权链接")
        with self._state_consume_guard():
            rows = [
                r
                for r in self._list("credentials", self.owner_id)
                if r.get("purpose") == "_oauth_state"
                and r.get("auth_channel") == "colab_cli"
                and r.get("oauth_purpose") == "colab"
                and r.get("state_hash") == _safe_id(state)
            ]
            if len(rows) != 1:
                raise GoogleOAuthError("Colab CLI 授权会话无效或已过期，请重新生成授权链接")
            row = rows[0]
            if self.account_id is not None and row.get("account_id") != self.account_id:
                raise GoogleOAuthError("Colab CLI 授权会话与当前账号不匹配，请重新生成授权链接")
            now = self.clock()
            if row.get("used_at") or row.get("used") or float(row.get("expires_at", 0)) <= now:
                raise GoogleOAuthError("Colab CLI 授权会话无效或已过期，请重新生成授权链接")
            if float(row.get("in_flight_until") or 0) > now:
                raise GoogleOAuthError("授权码正在验证中，请稍候片刻")
            self._update("credentials", row["id"], {"in_flight_until": now + 30}, self.owner_id)
            return row

    def _finish_cli_state(self, row: dict, *, consumed: bool) -> None:
        patch: dict[str, Any] = {"in_flight_until": None}
        if consumed:
            patch["used_at"] = self.clock()
        with self._state_consume_guard():
            self._update("credentials", row["id"], patch, self.owner_id)

    def _save_cli_credential(
        self,
        *,
        client_id: str,
        client_secret: str,
        access: str,
        refresh: str,
        expires_in: int,
        scope: str,
        token_type: str = "Bearer",
    ) -> dict:
        old = self._credential("colab", self.owner_id, self.account_id)
        data = {
            "provider": "google",
            "purpose": "colab",
            "auth_channel": "colab_cli",
            "client_id": client_id,
            "encrypted_client_secret": encrypt_secret(client_secret),
            "encrypted_access_token": encrypt_secret(access),
            "encrypted_refresh_token": encrypt_secret(refresh),
            "expires_at": self.clock() + max(0, expires_in - 60),
            "scope": scope,
            "token_type": token_type,
            "revoked": False,
            **({"account_id": self.account_id} if self.account_id is not None else {}),
        }
        if old:
            self._update("credentials", old["id"], data, self.owner_id)
        else:
            self._create("credentials", self.owner_id, data)
        invalidate_token_cache(self.store, self.owner_id, self.account_id, "colab")
        result: dict[str, Any] = {
            "purpose": "colab",
            "auth_channel": "colab_cli",
            "connected": True,
            "owner_id": self.owner_id,
        }
        if self.account_id is not None:
            result["account_id"] = self.account_id
        return result

    def complete_colab_cli(self, code: str, state: str) -> dict:
        if not self.owner_id:
            raise GoogleOAuthError("当前登录用户无效")
        cleaned_code = _extract_authorization_code(code)
        client_id, client_secret, redirect = load_colab_cli_oauth_client()
        row = self._begin_cli_state(state)
        consumed = False
        try:
            try:
                verifier = decrypt_secret(row["encrypted_code_verifier"])
                response = self.client.post(
                    TOKEN_URL,
                    data={
                        "code": cleaned_code,
                        "client_id": client_id,
                        "client_secret": client_secret,
                        "redirect_uri": redirect,
                        "grant_type": "authorization_code",
                        "code_verifier": verifier,
                    },
                )
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                raise GoogleOAuthError("Colab CLI 授权码交换失败，请检查网络连接后重试") from exc
            if response.status_code < 200 or response.status_code >= 300:
                err_code = None
                try:
                    err_body = response.json()
                    if isinstance(err_body, dict):
                        err_code = err_body.get("error")
                except ValueError:
                    err_code = None
                if err_code == "invalid_grant":
                    raise GoogleOAuthError(
                        "授权码无效、已过期或与当前链接不匹配，请核对后重试或重新生成授权链接"
                    )
                if err_code == "invalid_client":
                    raise GoogleConfigurationError("Colab CLI 客户端配置不可用")
                raise GoogleOAuthError("Colab CLI 授权码交换失败，请核对授权码或重新生成链接")
            consumed = True
            try:
                payload = response.json()
                access = payload.get("access_token")
                refresh = payload.get("refresh_token")
                expires_in = int(payload.get("expires_in", 3600))
                if not isinstance(access, str) or not access:
                    raise ValueError
            except (ValueError, TypeError, AttributeError) as exc:
                raise GoogleOAuthError("Google OAuth响应无效") from exc
            granted_scope = str(payload.get("scope") or " ".join(COLAB_CLI_SCOPES)).strip()
            if not self._scope_sufficient("colab", granted_scope):
                raise GoogleOAuthError("授权缺少 Google Colab 权限，请重新生成链接并在授权页保留 Colab 权限")
            old = self._credential("colab", self.owner_id, self.account_id)
            if not refresh and old and _is_cli_credential(old):
                try:
                    refresh = decrypt_secret(old["encrypted_refresh_token"])
                except (KeyError, TypeError, ValueError):
                    refresh = None
            if not refresh:
                raise GoogleOAuthError("Google OAuth 未返回可持久化刷新凭证，请重新生成链接并确认授权")
            return self._save_cli_credential(
                client_id=client_id,
                client_secret=client_secret,
                access=access,
                refresh=refresh,
                expires_in=expires_in,
                scope=granted_scope,
                token_type=str(payload.get("token_type") or "Bearer"),
            )
        finally:
            self._finish_cli_state(row, consumed=consumed)

    def import_colab_cli(self, payload: dict) -> dict:
        if not self.owner_id:
            raise GoogleOAuthError("当前登录用户无效")
        if not isinstance(payload, dict):
            raise GoogleOAuthError("Colab CLI 凭据参数必须是对象")
        source = payload.get("credentials_json")
        if isinstance(source, str):
            if len(source) > 65536:
                raise GoogleOAuthError("Colab CLI 凭据 JSON 过大")
            try:
                source = json.loads(source)
            except ValueError as exc:
                raise GoogleOAuthError("Colab CLI 凭据 JSON 格式无效") from exc
        if source is not None and not isinstance(source, dict):
            raise GoogleOAuthError("Colab CLI 凭据 JSON 必须是对象")
        raw = source if isinstance(source, dict) else payload
        if "web" in raw and not raw.get("refresh_token"):
            raise GoogleOAuthError(
                "这是 Web OAuth 客户端配置文件，不是已授权的 Colab CLI 凭据；请改用上方两步授权或下方 Google 应用配置"
            )
        if "installed" in raw and not raw.get("refresh_token"):
            raise GoogleOAuthError(
                "这是 OAuth 客户端定义文件，缺少 refresh_token；请使用上方两步 Colab CLI 授权流程"
            )
        refresh = str(raw.get("refresh_token") or "").strip()
        if not refresh or len(refresh) > 8192 or any(ch.isspace() for ch in refresh):
            raise GoogleOAuthError("缺少有效的 refresh_token")
        raw_cid = str(raw.get("client_id") or "").strip()
        raw_secret = str(raw.get("client_secret") or "").strip()
        if bool(raw_cid) != bool(raw_secret):
            raise GoogleOAuthError("client_id 与 client_secret 必须同时提供，或全部留空使用内置 Colab CLI 客户端")
        if raw_cid and raw_secret:
            if not raw_cid.startswith(COLAB_CLI_CLIENT_ID_PREFIX):
                raise GoogleOAuthError("仅支持 Colab CLI 公共客户端凭据（764086051850-...）")
            client_id, client_secret = raw_cid, raw_secret
        else:
            client_id, client_secret, _redirect = load_colab_cli_oauth_client()
        declared_scope = raw.get("scopes") if raw.get("scopes") is not None else raw.get("scope")
        if declared_scope and not self._scope_sufficient("colab", declared_scope):
            raise GoogleOAuthError("导入的凭据缺少 Google Colab 权限（colaboratory）")
        try:
            response = self.client.post(
                TOKEN_URL,
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "refresh_token": refresh,
                    "grant_type": "refresh_token",
                },
            )
        except httpx.HTTPError as exc:
            raise GoogleOAuthError("验证 Colab CLI 凭据时网络异常，请稍后重试") from exc
        if response.status_code < 200 or response.status_code >= 300:
            raise GoogleOAuthError("Colab CLI 凭据校验失败（refresh_token 已失效或与客户端不匹配）")
        try:
            body = response.json()
            access = body.get("access_token")
            expires_in = int(body.get("expires_in", 3600))
            if not isinstance(access, str) or not access:
                raise ValueError
        except (ValueError, TypeError, AttributeError) as exc:
            raise GoogleOAuthError("Google OAuth 刷新响应无效") from exc
        returned_scope = body.get("scope")
        if returned_scope and not self._scope_sufficient("colab", returned_scope):
            raise GoogleOAuthError("导入的凭据缺少 Google Colab 权限（colaboratory）")
        if isinstance(declared_scope, (list, tuple, set)):
            fallback_scope = " ".join(str(item) for item in declared_scope)
        elif isinstance(declared_scope, str) and declared_scope.strip():
            fallback_scope = declared_scope.strip()
        else:
            fallback_scope = " ".join(COLAB_CLI_SCOPES)
        effective_scope = str(returned_scope).strip() if isinstance(returned_scope, str) and returned_scope.strip() else fallback_scope
        effective_refresh = str(body.get("refresh_token") or refresh).strip()
        return self._save_cli_credential(
            client_id=client_id,
            client_secret=client_secret,
            access=access,
            refresh=effective_refresh,
            expires_in=expires_in,
            scope=effective_scope,
            token_type=str(body.get("token_type") or "Bearer"),
        )

    def access_token(self, purpose: str) -> str:
        if purpose not in _PURPOSES:
            raise ValueError("Google OAuth purpose无效")
        cache_enabled = _token_cache_allowed(self.clock) and bool(self.owner_id)
        cache_key = (id(self.store), str(self.owner_id or ""), self.account_id, purpose)
        if cache_enabled:
            with _TOKEN_CACHE_LOCK:
                cached = _TOKEN_CACHE.get(cache_key)
                if cached and cached[1] > self.clock() + 60:
                    return cached[0]
        credential = self._credential(purpose, account_id=self.account_id)
        if not credential:
            raise GoogleOAuthError("Google OAuth未连接，真实GPU能力已关闭")
        if not self._scope_sufficient(purpose, credential.get("scope")):
            raise GoogleOAuthError("Google OAuth权限不足，请重新授权")
        try:
            access = decrypt_secret(credential["encrypted_access_token"])
            refresh = decrypt_secret(credential["encrypted_refresh_token"])
        except (KeyError, ValueError) as exc:
            raise GoogleOAuthError("Google OAuth凭证不可用") from exc
        expires_at = float(credential.get("expires_at", 0))
        if expires_at > self.clock() + 60:
            if cache_enabled:
                with _TOKEN_CACHE_LOCK:
                    _TOKEN_CACHE[cache_key] = (access, min(expires_at, self.clock() + 300))
            return access
        cred_client_id = str(credential.get("client_id", "")).strip()
        if cred_client_id and credential.get("encrypted_client_secret"):
            try:
                client_id = cred_client_id
                client_secret = decrypt_secret(credential["encrypted_client_secret"])
            except (KeyError, TypeError, ValueError) as exc:
                raise GoogleConfigurationError("Google OAuth配置不可用") from exc
        else:
            client_id, client_secret, _redirect = self._config()
        try:
            response = self.client.post(
                TOKEN_URL,
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "refresh_token": refresh,
                    "grant_type": "refresh_token",
                },
            )
        except httpx.HTTPError as exc:
            raise GoogleOAuthError("Google OAuth刷新失败") from exc
        if response.status_code < 200 or response.status_code >= 300:
            raise GoogleOAuthError("Google OAuth刷新失败")
        try:
            payload = response.json()
            new_access = payload.get("access_token")
            expires_in = int(payload.get("expires_in", 3600))
            if not isinstance(new_access, str) or not new_access:
                raise ValueError
        except (ValueError, TypeError, AttributeError):
            raise GoogleOAuthError("Google OAuth刷新响应无效")
        new_refresh = payload.get("refresh_token")
        new_expires_at = self.clock() + max(0, expires_in - 60)
        self._update(
            "credentials",
            credential["id"],
            {
                "encrypted_access_token": encrypt_secret(new_access),
                "encrypted_refresh_token": encrypt_secret(new_refresh)
                if new_refresh
                else credential["encrypted_refresh_token"],
                "expires_at": new_expires_at,
            },
            self.owner_id,
        )
        if cache_enabled:
            with _TOKEN_CACHE_LOCK:
                _TOKEN_CACHE[cache_key] = (new_access, min(new_expires_at, self.clock() + 300))
        return new_access

    def disconnect(self, purpose: str, *, revoke_remote: bool = True) -> bool:
        if purpose not in _PURPOSES:
            raise ValueError("Google OAuth purpose无效")
        credential = self._credential(purpose, account_id=self.account_id)
        if not credential:
            return False
        remote_confirmed = False
        if revoke_remote:
            try:
                refresh = decrypt_secret(credential.get("encrypted_refresh_token", ""))
                token = refresh or decrypt_secret(credential.get("encrypted_access_token", ""))
                response = self.client.post(REVOKE_URL, data={"token": token})
                remote_confirmed = 200 <= response.status_code < 300
            except (httpx.HTTPError, KeyError, TypeError, ValueError):
                # Do not include the token or upstream response body in any error.
                remote_confirmed = False
            disconnect_status = "confirmed" if remote_confirmed else "remote_unconfirmed"
        else:
            disconnect_status = "local_only"
        # Remove local bearer material and keep a sanitized tombstone that records
        # whether Google confirmed the revoke.
        self._update(
            "credentials",
            credential["id"],
            {
                "revoked": True,
                "encrypted_access_token": None,
                "encrypted_refresh_token": None,
                "encrypted_client_secret": None,
                "expires_at": 0,
                "disconnect_remote_confirmed": remote_confirmed,
                "disconnect_status": disconnect_status,
            },
            self.owner_id,
        )
        invalidate_token_cache(self.store, self.owner_id, self.account_id, purpose)
        if purpose == "drive" and self.owner_id:
            from .drive_usage import invalidate_drive_usage_cache
            from .native_outputs import invalidate_native_outputs_cache

            invalidate_native_outputs_cache(self.owner_id)
            invalidate_drive_usage_cache(self.owner_id)
        return True

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None


__all__ = [
    "COLAB_CLI_CLIENT_ID_PREFIX",
    "COLAB_CLI_REDIRECT_URI",
    "COLAB_CLI_SCOPES",
    "GoogleConfigurationError",
    "GoogleError",
    "GoogleOAuthError",
    "GoogleService",
    "colab_cli_oauth_available",
    "load_colab_cli_oauth_client",
]
