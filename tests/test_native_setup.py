"""Tests for dynamic encrypted Native ComfyUI setup, fallback, and zero-GPU diagnostics."""

from __future__ import annotations

import base64
import json

import httpx
import pytest
from app import offline
from app.config import settings
from app.db import Store
from app.gpu import native, native_config
from app.gpu.accounts import AccountService
from app.gpu.compute import GpuConfig
from app.gpu.repository import Repository
from app.main import app
from app.security import encrypt_secret, sign_session
from cryptography.fernet import Fernet
from starlette.testclient import TestClient


def _cf_token(account="1234567890abcdef1234", tunnel="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", secret="super-secret-value"):
    raw = json.dumps({"a": account, "t": tunnel, "s": secret}).encode("utf-8")
    return base64.b64encode(raw).decode("ascii")


@pytest.fixture
def env_app(tmp_path, monkeypatch):
    db_path = tmp_path / "studio.db"
    store = Store(f"sqlite:///{db_path}")
    repo = Repository(store)
    fernet_key = Fernet.generate_key().decode()
    monkeypatch.setattr(settings, "encryption_key", fernet_key)
    monkeypatch.setattr(settings, "session_secret", "test-native-setup-session-secret")
    monkeypatch.setattr(settings, "studio_api_token", "test-admin-api-token")
    monkeypatch.setattr(settings, "site_url", "https://studio.example.test")
    monkeypatch.setattr(settings, "native_comfy_url", "")
    monkeypatch.setattr(settings, "native_comfy_backend_url", "")
    monkeypatch.delenv("NATIVE_COMFY_COMMIT", raising=False)
    monkeypatch.delenv("NATIVE_COMFY_ACCESS_FILE", raising=False)
    monkeypatch.delenv("NATIVE_COMFY_TUNNEL_TOKEN_FILE", raising=False)
    monkeypatch.setenv("GPU_REAL_ENABLED", "1")
    native_config.invalidate_cache()

    old_store = getattr(app.state, "store", None)
    old_repo = getattr(app.state, "repo", None)
    app.state.store = store
    app.state.repo = repo
    try:
        yield app, repo, tmp_path
    finally:
        native_config.invalidate_cache()
        app.state.store = old_store
        app.state.repo = old_repo


def _auth():
    return {"Authorization": "Bearer test-admin-api-token"}


def _seed_ready_account(repo, owner="owner"):
    acct = AccountService(repo, owner).create({"alias": "主账号", "allow_allocation": True}, "seed-acct-1")
    aid = acct["account_id"]
    repo.create("credentials", owner, {
        "provider": "google", "purpose": "oauth_client_config",
        "client_id": "123456-abc.apps.googleusercontent.com",
        "encrypted_client_secret": encrypt_secret("oauth-client-secret"),
    })
    for purpose, scope in (
        ("colab", "https://www.googleapis.com/auth/colaboratory"),
        ("drive", "https://www.googleapis.com/auth/drive.file"),
    ):
        repo.create("credentials", owner, {
            "provider": "google", "purpose": purpose, "account_id": aid,
            "encrypted_access_token": encrypt_secret("access-token"),
            "encrypted_refresh_token": encrypt_secret("refresh-token"),
            "expires_at": 2**31, "scope": scope,
        })
    return aid


def test_env_fallback_and_db_override_precedence_and_clear(env_app, monkeypatch):
    fastapi_app, repo, tmp_path = env_app
    access_file = tmp_path / "comfy-access.json"
    access_file.write_text(json.dumps({"client_id": "env-cid.access", "client_secret": "env-csec-secret"}))
    tunnel_file = tmp_path / "comfy-tunnel.token"
    env_tok = _cf_token(tunnel="11111111-1111-1111-1111-111111111111")
    tunnel_file.write_text(env_tok)

    monkeypatch.setattr(settings, "native_comfy_url", "https://env-comfy.example.test")
    monkeypatch.setattr(settings, "native_comfy_backend_url", "https://env-tunnel.example.test")
    monkeypatch.setenv("NATIVE_COMFY_COMMIT", "1" * 40)
    monkeypatch.setenv("NATIVE_COMFY_ACCESS_FILE", str(access_file))
    monkeypatch.setenv("NATIVE_COMFY_TUNNEL_TOKEN_FILE", str(tunnel_file))

    with TestClient(fastapi_app) as client:
        r = client.get("/api/gpu/native/setup", headers=_auth())
        assert r.status_code == 200
        data = r.json()["data"]
        assert data["configured"] is True
        assert data["web_ui_configured"] is True
        assert data["has_db_override"] is False
        assert data["recommended_commit"] == "423cbd20511becb1f16122ab518e92e58bfa03d9"
        assert data["values"]["native_comfy_url"] == "https://env-comfy.example.test"
        assert data["values"]["native_comfy_backend_url"] == "https://env-tunnel.example.test"
        assert data["values"]["native_comfy_commit"] == "1" * 40
        assert data["values"]["access_client_id"] == "env-cid.access"
        assert data["secrets"]["access_client_secret_configured"] is True
        assert data["secrets"]["tunnel_token_configured"] is True
        assert data["secrets"]["tunnel_info"]["tunnel_id"] == "11111111-1111-1111-1111-111111111111"
        assert data["sources"]["tunnel_token"] == "environment"
        # Secrets must never appear in GET response
        assert "env-csec-secret" not in r.text
        assert env_tok not in r.text

        # Save DB override with pasted cloudflared command + GitHub commit URL
        db_tok = _cf_token(tunnel="22222222-2222-2222-2222-222222222222")
        put = client.put("/api/gpu/native/setup", headers=_auth(), json={
            "native_comfy_url": "https://db-comfy.example.test/",
            "native_comfy_backend_url": "https://db-tunnel.example.test/",
            "native_comfy_commit": "https://github.com/Comfy-Org/ComfyUI/commit/423cbd20511becb1f16122ab518e92e58bfa03d9",
            "access_json": json.dumps({"client_id": "db-cid.access", "client_secret": "db-csec-secret"}),
            "tunnel_token": f"sudo cloudflared service install {db_tok}",
        })
        assert put.status_code == 200
        saved = put.json()["data"]
        assert saved["has_db_override"] is True
        assert saved["values"]["native_comfy_url"] == "https://db-comfy.example.test"
        assert saved["values"]["native_comfy_backend_url"] == "https://db-tunnel.example.test"
        assert saved["values"]["native_comfy_commit"] == native_config.RECOMMENDED_COMFY_COMMIT
        assert saved["values"]["access_client_id"] == "db-cid.access"
        assert saved["sources"]["tunnel_token"] == "encrypted_database"
        assert saved["secrets"]["tunnel_info"]["tunnel_id"] == "22222222-2222-2222-2222-222222222222"
        assert "db-csec-secret" not in put.text
        assert db_tok not in put.text

        # DB row stores encrypted secrets, not plaintext
        raw_row = repo.store.get("settings", "native-setup:owner", "owner")
        assert raw_row["encrypted_access_client_secret"] != "db-csec-secret"
        assert raw_row["encrypted_tunnel_token"] != db_tok

        # Runtime helpers immediately resolve DB credentials
        url, headers = native.backend_for(repo, "owner")
        assert url == "https://db-tunnel.example.test"
        assert headers == {
            "CF-Access-Client-Id": "db-cid.access",
            "CF-Access-Client-Secret": "db-csec-secret",
        }
        assert native.model_channel_for("lease-1", repo=repo, owner="owner") == native.model_channel("lease-1", db_tok)

        # Updating only native_comfy_url preserves existing encrypted secrets when blank
        put2 = client.put("/api/gpu/native/setup", headers=_auth(), json={
            "native_comfy_url": "",
            "native_comfy_backend_url": "https://db-tunnel.example.test",
            "native_comfy_commit": native_config.RECOMMENDED_COMFY_COMMIT,
            "access_client_id": "db-cid.access",
        })
        assert put2.status_code == 200
        saved2 = put2.json()["data"]
        assert saved2["web_ui_configured"] is False
        assert saved2["configured"] is True
        assert saved2["values"]["native_comfy_url"] == ""
        assert saved2["secrets"]["access_client_secret_configured"] is True
        assert saved2["secrets"]["tunnel_token_configured"] is True
        # Status reflects API-only mode (configured=True, web_ui_configured=False)
        st = client.get("/api/gpu/native/status", headers=_auth()).json()["data"]
        assert st["configured"] is True
        assert st["tunnel_configured"] is True
        assert st["web_ui_configured"] is False
        assert st["url"] == ""
        # Ticket rejects when web_ui_configured is False
        assert client.post("/api/gpu/native/ticket", headers=_auth()).status_code == 503

        # Clearing DB override falls back to .env/files cleanly
        deleted = client.delete("/api/gpu/native/setup", headers=_auth())
        assert deleted.status_code == 200
        restored = deleted.json()["data"]
        assert restored["has_db_override"] is False
        assert restored["values"]["native_comfy_url"] == "https://env-comfy.example.test"
        assert restored["sources"]["tunnel_token"] == "environment"
        url_after, hdrs_after = native.backend_for(repo, "owner")
        assert url_after == "https://env-tunnel.example.test"
        assert hdrs_after["CF-Access-Client-Id"] == "env-cid.access"


def test_dynamic_public_url_activates_proxy_ticket_and_api_settings_without_restart(env_app):
    fastapi_app, repo, _ = env_app
    tok = _cf_token()
    with TestClient(fastapi_app) as client:
        res = client.put("/api/gpu/native/setup", headers=_auth(), json={
            "native_comfy_url": "https://dynamic-comfy.example.test",
            "native_comfy_backend_url": "https://dynamic-backend.example.test",
            "native_comfy_commit": native_config.RECOMMENDED_COMFY_COMMIT,
            "access_client_id": "cid-dyn.access",
            "access_client_secret": "csec-dyn-secret",
            "tunnel_token": tok,
        })
        assert res.status_code == 200

        # Ticket immediately targets dynamic URL
        tk = client.post("/api/gpu/native/ticket", headers=_auth())
        assert tk.status_code == 200
        assert tk.json()["data"]["url"] == "https://dynamic-comfy.example.test/_studio/session"

        # API settings immediately exposes dynamic dedicated URL
        api_set = client.get("/api/gpu/native/api-settings", headers=_auth())
        assert api_set.status_code == 200
        assert api_set.json()["data"]["url"] == "https://dynamic-comfy.example.test"

    # NativeHostMiddleware immediately routes Host: dynamic-comfy.example.test
    with TestClient(fastapi_app, base_url="https://dynamic-comfy.example.test") as comfy_client:
        probe = comfy_client.get("/_studio/api/status", headers=_auth())
        assert probe.status_code == 200
        assert probe.json() == {"ready": False}
        unauth = comfy_client.get("/_studio/api/status")
        assert unauth.status_code == 401
        assert unauth.json()["error"] == "UNAUTHENTICATED"


def test_active_session_locks_tunnel_changes_but_allows_web_url_and_commit(env_app):
    fastapi_app, repo, _ = env_app
    tok = _cf_token()
    with TestClient(fastapi_app) as client:
        assert client.put("/api/gpu/native/setup", headers=_auth(), json={
            "native_comfy_url": "",
            "native_comfy_backend_url": "https://live-backend.example.test",
            "native_comfy_commit": native_config.RECOMMENDED_COMFY_COMMIT,
            "access_client_id": "live-cid.access",
            "access_client_secret": "live-csec-secret",
            "tunnel_token": tok,
        }).status_code == 200

        # Simulate an active native lease
        repo.create("leases", "owner", {
            "provider": "colab", "native_comfy": True, "native_ready": True, "state": "idle",
            "instance_id": "inst-1",
        }, record_id="lease-active")

        setup_view = client.get("/api/gpu/native/setup", headers=_auth()).json()["data"]
        assert setup_view["active_session_locked"] is True

        # Changing backend_url while live is rejected (409)
        blocked = client.put("/api/gpu/native/setup", headers=_auth(), json={
            "native_comfy_url": "",
            "native_comfy_backend_url": "https://other-backend.example.test",
            "native_comfy_commit": native_config.RECOMMENDED_COMFY_COMMIT,
            "access_client_id": "live-cid.access",
        })
        assert blocked.status_code == 409

        # Clearing DB override while live is also rejected (409)
        blocked_del = client.delete("/api/gpu/native/setup", headers=_auth())
        assert blocked_del.status_code == 409

        # Adding native_comfy_url or updating next-boot commit IS allowed while live
        allowed = client.put("/api/gpu/native/setup", headers=_auth(), json={
            "native_comfy_url": "https://added-web.example.test",
            "native_comfy_backend_url": "https://live-backend.example.test",
            "native_comfy_commit": "b" * 40,
            "access_client_id": "live-cid.access",
        })
        assert allowed.status_code == 200
        assert allowed.json()["data"]["values"]["native_comfy_url"] == "https://added-web.example.test"


def test_validation_guards_and_admin_only(env_app):
    fastapi_app, _, _ = env_app
    tok = _cf_token()
    with TestClient(fastapi_app) as client:
        # Reject loopback/private backend host (SSRF)
        r = client.put("/api/gpu/native/setup", headers=_auth(), json={
            "native_comfy_backend_url": "https://127.0.0.1",
            "access_client_id": "cid.access",
            "access_client_secret": "csec-1234",
            "tunnel_token": tok,
        })
        assert r.status_code == 422

        # Reject native_comfy_url matching SITE_URL
        r2 = client.put("/api/gpu/native/setup", headers=_auth(), json={
            "native_comfy_url": "https://studio.example.test",
            "native_comfy_backend_url": "https://backend.example.test",
            "access_client_id": "cid.access",
            "access_client_secret": "csec-1234",
            "tunnel_token": tok,
        })
        assert r2.status_code == 422

        # Reject changing client_id without supplying client_secret
        assert client.put("/api/gpu/native/setup", headers=_auth(), json={
            "native_comfy_backend_url": "https://backend.example.test",
            "access_client_id": "cid1.access",
            "access_client_secret": "csec-1111",
            "tunnel_token": tok,
        }).status_code == 200
        r3 = client.put("/api/gpu/native/setup", headers=_auth(), json={
            "native_comfy_backend_url": "https://backend.example.test",
            "access_client_id": "cid2.access",
        })
        assert r3.status_code == 422

        # Non-admin session rejected (403)
        user_cookie = sign_session({"uid": "guest", "role": "user", "exp": 2**31})
        with TestClient(fastapi_app, cookies={"studio_session": user_cookie}) as guest_client:
            r4 = guest_client.get("/api/gpu/native/setup")
        assert r4.status_code == 403


def test_zero_gpu_diagnose_detects_pass_leak_bad_token_and_proxy(env_app, monkeypatch):
    _, repo, _ = env_app
    _seed_ready_account(repo, "owner")
    monkeypatch.setattr(offline, "enabled", lambda: False)
    monkeypatch.setattr(GpuConfig, "from_env", classmethod(lambda cls: GpuConfig(real_enabled=True)))

    tok = _cf_token()
    native_config.save_setup(repo, "owner", {
        "native_comfy_url": "https://comfy.example.test",
        "native_comfy_backend_url": "https://backend.example.test",
        "native_comfy_commit": native_config.RECOMMENDED_COMFY_COMMIT,
        "access_client_id": "cid.access",
        "access_client_secret": "csec-valid",
        "tunnel_token": tok,
    })

    # Case 1: Ideal offline-Colab pre-flight: Access blocks unauth (302), passes valid token to offline tunnel (530),
    # and public web URL hits NativeHostMiddleware (200 {"ready": False}).
    def ok_handler(req: httpx.Request) -> httpx.Response:
        if req.url.host == "backend.example.test":
            if req.headers.get("CF-Access-Client-Secret") == "csec-valid":
                return httpx.Response(530, text="Tunnel offline")
            return httpx.Response(302, headers={"Location": "https://team.cloudflareaccess.com/login"})
        if req.url.host == "comfy.example.test" and req.url.path == "/_studio/api/status":
            return httpx.Response(200, json={"ready": False})
        return httpx.Response(404)

    with httpx.Client(transport=httpx.MockTransport(ok_handler)) as mock_client:
        res = native_config.diagnose_setup(repo, "owner", client=mock_client)
    assert res["ok"] is True
    by_id = {c["id"]: c for c in res["checks"]}
    assert by_id["real_gpu"]["status"] == "pass"
    assert by_id["google_accounts"]["status"] == "pass"
    assert by_id["comfy_commit"]["status"] == "pass"
    assert by_id["tunnel_token"]["status"] == "pass"
    assert by_id["backend_access"]["status"] == "pass"
    assert by_id["public_web_url"]["status"] == "pass"

    # Case 2: Access Bypass leak on /customnode/install
    def leak_handler(req: httpx.Request) -> httpx.Response:
        if req.url.host == "backend.example.test":
            if req.url.path == "/customnode/install":
                return httpx.Response(530)
            return httpx.Response(302)
        return httpx.Response(401, json={"error": "UNAUTHENTICATED"})

    with httpx.Client(transport=httpx.MockTransport(leak_handler)) as mock_client:
        res2 = native_config.diagnose_setup(repo, "owner", client=mock_client)
    assert res2["ok"] is False
    by_id2 = {c["id"]: c for c in res2["checks"]}
    assert by_id2["backend_access"]["status"] == "fail"
    assert "/customnode/install" in by_id2["backend_access"]["detail"]
    assert by_id2["public_web_url"]["status"] == "pass"

    # Case 3: Bad Service Token (returns 403 even with headers) + misconfigured Nginx (returns HTML)
    def bad_token_handler(req: httpx.Request) -> httpx.Response:
        if req.url.host == "backend.example.test":
            return httpx.Response(403)
        return httpx.Response(200, text="<html>Welcome to nginx!</html>", headers={"Content-Type": "text/html"})

    with httpx.Client(transport=httpx.MockTransport(bad_token_handler)) as mock_client:
        res3 = native_config.diagnose_setup(repo, "owner", client=mock_client)
    assert res3["ok"] is False
    by_id3 = {c["id"]: c for c in res3["checks"]}
    assert by_id3["backend_access"]["status"] == "fail"
    assert "Service Token" in by_id3["backend_access"]["detail"]
    assert by_id3["public_web_url"]["status"] == "fail"

    # Case 4: Nginx IP whitelist returns 403 HTML to VPS loopback probe -> warning (not hard failure)
    def ip_whitelist_handler(req: httpx.Request) -> httpx.Response:
        if req.url.host == "backend.example.test":
            if req.headers.get("CF-Access-Client-Secret") == "csec-valid":
                return httpx.Response(530)
            return httpx.Response(302)
        return httpx.Response(403, text="<html>403 Forbidden</html>", headers={"Content-Type": "text/html"})

    with httpx.Client(transport=httpx.MockTransport(ip_whitelist_handler)) as mock_client:
        res4 = native_config.diagnose_setup(repo, "owner", client=mock_client)
    assert res4["ok"] is True
    assert res4["has_warnings"] is True
    by_id4 = {c["id"]: c for c in res4["checks"]}
    assert by_id4["public_web_url"]["status"] == "warn"


def test_admit_succeeds_with_db_only_credentials(env_app):
    _, repo, _ = env_app
    aid = _seed_ready_account(repo, "owner")
    tok = _cf_token()
    native_config.save_setup(repo, "owner", {
        "native_comfy_url": "",
        "native_comfy_backend_url": "https://db-only.example.test",
        "native_comfy_commit": native_config.RECOMMENDED_COMFY_COMMIT,
        "access_client_id": "db-only.access",
        "access_client_secret": "db-only-secret",
        "tunnel_token": tok,
    })
    job = native.admit(
        repo,
        "owner",
        {"account_id": aid, "spec": {"key": {"variant": "VARIANT_GPU", "accelerator": "L4", "shape": "SHAPE_STANDARD"}}},
        "idem-db-only-1",
    )
    assert job["snapshot"]["manifest"]["comfy_commit"] == native_config.RECOMMENDED_COMFY_COMMIT
    assert job["snapshot"]["native_comfy"] is True
