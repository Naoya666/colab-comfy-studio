from __future__ import annotations

import json
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from app.account_routes import get_current_user_for_tests
from app.account_routes import router as account_router
from app.db import Store
from app.gpu.accounts import AccountService
from app.gpu.colab import GoogleColabProvider
from app.gpu.google import (
    COLAB_CLI_CLIENT_ID_PREFIX,
    COLAB_CLI_REDIRECT_URI,
    COLAB_CLI_SCOPES,
    GoogleOAuthError,
    GoogleService,
    colab_cli_oauth_available,
    load_colab_cli_oauth_client,
)
from app.gpu.repository import Repository
from app.oauth_config_routes import router as oauth_config_router
from app.security import decrypt_secret
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _make_app(st: Store, *, google_factory=None, probe_fn=None):
    from app import account_routes
    from app.common import current_user, get_store

    app = FastAPI()
    app.state.store = st
    app.include_router(account_router)
    app.include_router(oauth_config_router)
    app.dependency_overrides[get_current_user_for_tests] = lambda: {"id": "owner-1", "role": "admin"}
    app.dependency_overrides[current_user] = lambda: {"id": "owner-1", "role": "admin"}
    app.dependency_overrides[get_store] = lambda: st
    return app, account_routes


def test_colab_cli_start_uses_installed_package_config_minimal_scopes_and_pkce(tmp_path, monkeypatch):
    monkeypatch.delenv("GPU_COLAB_CLI_CLIENT_ID", raising=False)
    monkeypatch.delenv("GPU_COLAB_CLI_CLIENT_SECRET", raising=False)
    assert colab_cli_oauth_available() is True
    cid, csecret, redirect = load_colab_cli_oauth_client()
    assert cid.startswith(COLAB_CLI_CLIENT_ID_PREFIX)
    assert len(csecret) >= 8
    assert redirect == COLAB_CLI_REDIRECT_URI

    repo = Repository(Store("sqlite:///" + str(tmp_path / "cli_start.db")))
    service = GoogleService(repo, "owner-1", account_id="acc-1")
    started = service.start_colab_cli()
    assert started["auth_channel"] == "colab_cli"
    assert started["expires_in"] == 1800

    parsed = urlparse(started["authorization_url"])
    params = parse_qs(parsed.query)
    assert params["client_id"] == [cid]
    assert params["redirect_uri"] == [COLAB_CLI_REDIRECT_URI]
    assert params["response_type"] == ["code"]
    assert params["access_type"] == ["offline"]
    assert params["prompt"] == ["consent"]
    assert params["token_usage"] == ["remote"]
    assert params["code_challenge_method"] == ["S256"]
    assert params["state"] == [started["state"]]

    scopes = set(params["scope"][0].split())
    assert scopes == set(COLAB_CLI_SCOPES)
    assert "https://www.googleapis.com/auth/drive.file" not in scopes
    assert "https://www.googleapis.com/auth/cloud-platform" not in scopes


def test_colab_cli_complete_retries_invalid_grant_extracts_url_code_and_stores_encrypted(tmp_path, monkeypatch):
    monkeypatch.setenv("GPU_COLAB_CLI_CLIENT_ID", "764086051850-testcli.apps.googleusercontent.com")
    monkeypatch.setenv("GPU_COLAB_CLI_CLIENT_SECRET", "test-cli-secret-value")
    repo = Repository(Store("sqlite:///" + str(tmp_path / "cli_complete.db")))
    attempts: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        form = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
        attempts.append(form)
        if form["code"] == "wrong-code":
            return httpx.Response(400, json={"error": "invalid_grant", "error_description": "raw secret detail"})
        assert form["code"] == "4/0AValidCode123"
        assert form["client_id"] == "764086051850-testcli.apps.googleusercontent.com"
        assert form["client_secret"] == "test-cli-secret-value"
        assert form["redirect_uri"] == COLAB_CLI_REDIRECT_URI
        assert form["grant_type"] == "authorization_code"
        assert len(form["code_verifier"]) > 20
        return httpx.Response(
            200,
            json={
                "access_token": "cli-access-secret",
                "refresh_token": "cli-refresh-secret",
                "expires_in": 3600,
                # Reordered subset (still has colaboratory) must succeed with zero Scope-has-changed errors.
                "scope": "https://www.googleapis.com/auth/colaboratory openid https://www.googleapis.com/auth/userinfo.email",
                "token_type": "Bearer",
            },
        )

    service = GoogleService(
        repo,
        "owner-1",
        account_id="acc-cli-1",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    started = service.start_colab_cli()

    # Wrong code fails cleanly without exposing upstream description and keeps state reusable.
    with pytest.raises(GoogleOAuthError, match="授权码无效"):
        service.complete_colab_cli("wrong-code", started["state"])

    # Pasting a full callback URL extracts the code and completes authorization.
    pasted = "https://sdk.cloud.google.com/applicationdefaultauthcode.html?code=4/0AValidCode123&scope=openid"
    completed = service.complete_colab_cli(pasted, started["state"])
    assert completed == {
        "purpose": "colab",
        "auth_channel": "colab_cli",
        "connected": True,
        "owner_id": "owner-1",
        "account_id": "acc-cli-1",
    }
    assert "cli-access-secret" not in str(completed)
    assert "cli-refresh-secret" not in str(completed)

    # State is permanently consumed after success.
    with pytest.raises(GoogleOAuthError, match="无效或已过期"):
        service.complete_colab_cli("4/0AValidCode123", started["state"])

    status = service.status()
    assert status["colab"]["connected"] is True
    assert status["colab"]["auth_channel"] == "colab_cli"
    assert status["colab"]["cli_available"] is True

    cred = service._credential("colab", account_id="acc-cli-1")
    assert cred is not None
    assert decrypt_secret(cred["encrypted_access_token"]) == "cli-access-secret"
    assert decrypt_secret(cred["encrypted_refresh_token"]) == "cli-refresh-secret"
    assert decrypt_secret(cred["encrypted_client_secret"]) == "test-cli-secret-value"

    provider = GoogleColabProvider(repo, "owner-1", account_id="acc-cli-1", google=service)
    assert provider._is_cli_tunnel() is True
    assert len(attempts) == 2


def test_colab_cli_complete_rejects_cross_account_state_and_missing_colab_scope(tmp_path, monkeypatch):
    monkeypatch.setenv("GPU_COLAB_CLI_CLIENT_ID", "764086051850-testcli.apps.googleusercontent.com")
    monkeypatch.setenv("GPU_COLAB_CLI_CLIENT_SECRET", "test-cli-secret-value")
    repo = Repository(Store("sqlite:///" + str(tmp_path / "cli_scope.db")))

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "access_token": "tok",
                "refresh_token": "ref",
                "expires_in": 3600,
                "scope": "openid https://www.googleapis.com/auth/userinfo.email",
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    s_a = GoogleService(repo, "owner-1", account_id="acc-a", client=client)
    s_b = GoogleService(repo, "owner-1", account_id="acc-b", client=client)

    started = s_a.start_colab_cli()
    with pytest.raises(GoogleOAuthError, match="不匹配"):
        s_b.complete_colab_cli("4/0ACode", started["state"])

    with pytest.raises(GoogleOAuthError, match="缺少 Google Colab 权限"):
        s_a.complete_colab_cli("4/0ACode", started["state"])
    assert s_a.status()["colab"]["connected"] is False


def test_colab_cli_import_validates_refresh_token_and_rejects_web_json(tmp_path, monkeypatch):
    monkeypatch.setenv("GPU_COLAB_CLI_CLIENT_ID", "764086051850-testcli.apps.googleusercontent.com")
    monkeypatch.setenv("GPU_COLAB_CLI_CLIENT_SECRET", "test-cli-secret-value")
    repo = Repository(Store("sqlite:///" + str(tmp_path / "cli_import.db")))
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        form = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
        seen.append(form)
        return httpx.Response(
            200,
            json={
                "access_token": "fresh-imported-access",
                "expires_in": 3600,
                "scope": "openid https://www.googleapis.com/auth/colaboratory",
            },
        )

    service = GoogleService(
        repo,
        "owner-1",
        account_id="acc-import",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(GoogleOAuthError, match="Web OAuth"):
        service.import_colab_cli({"credentials_json": {"web": {"client_id": "123-abc.apps.googleusercontent.com"}}})

    with pytest.raises(GoogleOAuthError, match="764086051850"):
        service.import_colab_cli(
            {
                "refresh_token": "1//tok",
                "client_id": "999999-other.apps.googleusercontent.com",
                "client_secret": "other-secret",
            }
        )

    result = service.import_colab_cli(
        {
            "credentials_json": json.dumps(
                {
                    "refresh_token": "1//cli-refresh-imported",
                    "scopes": ["openid", "https://www.googleapis.com/auth/colaboratory"],
                }
            )
        }
    )
    assert result["connected"] is True
    assert result["auth_channel"] == "colab_cli"
    assert seen[0]["client_id"] == "764086051850-testcli.apps.googleusercontent.com"
    assert seen[0]["refresh_token"] == "1//cli-refresh-imported"
    assert service.status()["colab"]["auth_channel"] == "colab_cli"


def test_cli_credential_does_not_block_web_oauth_config_and_switching_clears_cli_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("GPU_COLAB_CLI_CLIENT_ID", "764086051850-testcli.apps.googleusercontent.com")
    monkeypatch.setenv("GPU_COLAB_CLI_CLIENT_SECRET", "test-cli-secret-value")
    st = Store("sqlite:///" + str(tmp_path / "cli_switch.db"))
    repo = Repository(st)
    accounts = AccountService(repo, "owner-1")
    acc = accounts.create({"alias": "混合号"}, "create-mix")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "access_token": "tok-1",
                "refresh_token": "ref-1",
                "expires_in": 3600,
                "scope": "https://www.googleapis.com/auth/colaboratory",
            },
        )

    google = GoogleService(
        repo,
        "owner-1",
        account_id=acc["account_id"],
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    started = google.start_colab_cli()
    google.complete_colab_cli("4/0ACode", started["state"])
    assert google.status()["colab"]["auth_channel"] == "colab_cli"

    # Saving/changing the shared Web OAuth app (needed for Drive) must NOT be blocked by a CLI colab credential.
    app, _ = _make_app(st)
    client = TestClient(app)
    saved = client.post(
        "/api/gpu/auth/google/config",
        json={
            "client_id": "123456-webapp.apps.googleusercontent.com",
            "client_secret": "web-app-secret-123",
            "confirmed": True,
        },
    )
    assert saved.status_code == 200, saved.text

    # Re-authorizing the same account via Web OAuth clears old CLI client_id/secret fields.
    web_started = google.start("colab")
    google.callback("web-code", web_started["state"])
    cred = google._credential("colab", account_id=acc["account_id"])
    assert cred is not None
    assert cred.get("client_id") is None
    assert cred.get("encrypted_client_secret") is None
    assert cred.get("auth_channel") == "web_oauth"
    assert google.status()["colab"]["auth_channel"] == "web_oauth"
    assert GoogleColabProvider(repo, "owner-1", account_id=acc["account_id"], google=google)._is_cli_tunnel() is False


def test_account_cli_routes_idempotency_probe_and_local_only_disconnect_delete(tmp_path, monkeypatch):
    monkeypatch.setenv("GPU_COLAB_CLI_CLIENT_ID", "764086051850-testcli.apps.googleusercontent.com")
    monkeypatch.setenv("GPU_COLAB_CLI_CLIENT_SECRET", "test-cli-secret-value")
    monkeypatch.delenv("STUDIO_OFFLINE_TEST", raising=False)
    st = Store("sqlite:///" + str(tmp_path / "cli_routes.db"))
    repo = Repository(st)
    app, account_routes = _make_app(st)
    revoked_calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if "revoke" in str(request.url):
            revoked_calls.append(request.content.decode())
            return httpx.Response(200)
        return httpx.Response(
            200,
            json={
                "access_token": "route-access",
                "refresh_token": "route-refresh",
                "expires_in": 3600,
                "scope": "openid https://www.googleapis.com/auth/colaboratory",
            },
        )

    mock_http = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(
        account_routes,
        "service_google",
        lambda req, owner_id, account_id: GoogleService(repo, owner_id, account_id=account_id, client=mock_http),
    )
    monkeypatch.setattr(
        account_routes,
        "probe_colab_cli_account",
        lambda req, owner_id, account_id: {"balance": 512.5, "eligible_gpus": ["T4", "L4", "A100"], "probe_error": None},
    )

    client = TestClient(app)
    acc = client.post("/api/gpu/accounts", headers={"Idempotency-Key": "mk-1"}, json={"alias": "CLI号"}).json()["data"]
    acc_id = acc["account_id"]

    started = client.post(
        f"/api/gpu/accounts/{acc_id}/auth/colab-cli/start",
        headers={"Idempotency-Key": "cli-start-1"},
        json={},
    )
    assert started.status_code == 200
    start_data = started.json()["data"]
    assert start_data["auth_channel"] == "colab_cli"
    # Idempotent replay returns identical state and URL.
    replayed = client.post(
        f"/api/gpu/accounts/{acc_id}/auth/colab-cli/start",
        headers={"Idempotency-Key": "cli-start-1"},
        json={},
    )
    assert replayed.json()["data"] == start_data

    completed = client.post(
        f"/api/gpu/accounts/{acc_id}/auth/colab-cli/complete",
        headers={"Idempotency-Key": "cli-done-1"},
        json={"state": start_data["state"], "code": "4/0ARouteCode"},
    )
    assert completed.status_code == 200
    done_data = completed.json()["data"]
    assert done_data["connected"] is True
    assert done_data["auth_channel"] == "colab_cli"
    assert done_data["balance"] == 512.5
    assert done_data["eligible_gpus"] == ["T4", "L4", "A100"]

    listed = client.get("/api/gpu/accounts").json()["data"]
    assert listed[0]["authorization"]["colab"]["connected"] is True
    assert listed[0]["authorization"]["colab"]["auth_channel"] == "colab_cli"

    # Local-only disconnect must NOT call Google REVOKE_URL.
    from app.gpu import google as google_mod

    orig_init = GoogleService.__init__

    def patched_init(self, *args, **kwargs):
        kwargs.setdefault("client", mock_http)
        orig_init(self, *args, **kwargs)

    monkeypatch.setattr(google_mod.GoogleService, "__init__", patched_init)

    disc = client.post(
        f"/api/gpu/accounts/{acc_id}/disconnect",
        headers={"Idempotency-Key": "disc-local-1"},
        json={"revoke_remote": False},
    )
    assert disc.status_code == 202
    assert disc.json()["data"]["status"] == "local_only"
    assert disc.json()["data"]["disconnected"] is True
    assert revoked_calls == []

    # Re-import and test delete with revoke_remote=false.
    imp = client.post(
        f"/api/gpu/accounts/{acc_id}/auth/colab-cli/import",
        headers={"Idempotency-Key": "cli-imp-1"},
        json={"refresh_token": "1//reimported"},
    )
    assert imp.status_code == 200
    assert imp.json()["data"]["connected"] is True

    deleted = client.delete(
        f"/api/gpu/accounts/{acc_id}?revoke_remote=false",
        headers={"Idempotency-Key": "del-local-1"},
    )
    assert deleted.status_code == 200
    assert deleted.json()["data"]["deleted"] is True
    assert revoked_calls == []
