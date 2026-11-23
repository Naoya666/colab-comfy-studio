"""Owner-scoped v0.2 Google account directory API.

This module intentionally contains no OAuth callback trust logic: the callback
uses the persisted state record in :mod:`app.gpu.google`.  These routes only
select an already authenticated owner and an account owned by that owner.
"""

from __future__ import annotations

from typing import Callable

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from .common import APIError, current_user, envelope, ok
from .gpu.accounts import AccountService
from .gpu.google import GoogleError, GoogleOAuthError
from .gpu.repository import Repository

router = APIRouter(prefix="/api/gpu/accounts", tags=["gpu-accounts"])

# Kept as a named dependency so offline contract tests can override auth without
# importing the application composition root.
get_current_user_for_tests = current_user


def _repo(request: Request):
    candidate = getattr(request.app.state, "repo", None)
    if candidate is not None:
        return candidate
    store = getattr(request.app.state, "store", None)
    if store is None:
        raise APIError("ACCOUNT_SERVICE_UNAVAILABLE", "账号存储尚未接入", 503, True)
    # Test doubles expose the Repository CRUD surface directly.  The production
    # Store needs the GPU Repository adapter so account rows use the v0.2 schema.
    if hasattr(store, "engine") and hasattr(store, "db_lock"):
        return Repository(store)
    return store


def _service(request: Request, user: dict) -> AccountService:
    return AccountService(_repo(request), user["id"])


def _idempotency(request: Request) -> str:
    key = request.headers.get("Idempotency-Key", "").strip()
    if not key or len(key) > 200:
        raise APIError("IDEMPOTENCY_REQUIRED", "写操作必须提供固定幂等键，重试沿用原值", 422)
    return key


def _error_response(request: Request, exc: APIError) -> JSONResponse:
    return JSONResponse(
        envelope(
            error={"code": exc.code, "message": exc.message, "retryable": exc.retryable},
            request_id=getattr(getattr(request, "state", None), "request_id", None),
        ),
        status_code=exc.status,
    )


def _invoke(request: Request, action: Callable):
    """Keep this router usable in bare FastAPI contract tests as well as main."""
    try:
        return action()
    except APIError as exc:
        return _error_response(request, exc)
    except GoogleOAuthError as exc:
        return _error_response(request, APIError("GOOGLE_OAUTH_ERROR", str(exc), 422))
    except GoogleError as exc:
        return _error_response(request, APIError("GOOGLE_ACCOUNT_ERROR", str(exc), 503, True))


@router.get("")
def list_accounts(request: Request, user=Depends(get_current_user_for_tests)):
    return _invoke(request, lambda: ok(_service(request, user).list(), request))


@router.post("", status_code=201)
def create_account(
    body: dict,
    request: Request,
    user=Depends(get_current_user_for_tests),
):
    def action():
        key = _idempotency(request)
        return ok(_service(request, user).create(body, key), request, 201)

    return _invoke(request, action)


@router.patch("/{account_id}")
def update_account(
    account_id: str,
    body: dict,
    request: Request,
    user=Depends(get_current_user_for_tests),
):
    def action():
        key = _idempotency(request)
        return ok(_service(request, user).update(account_id, body, key), request)

    return _invoke(request, action)


@router.post("/{account_id}/auth/colab-cli/start")
def start_account_colab_cli_auth(
    account_id: str,
    body: dict | None = None,
    request: Request = None,
    user=Depends(get_current_user_for_tests),
):
    def action():
        key = _idempotency(request)
        if body is not None and not isinstance(body, dict):
            raise APIError("VALIDATION_ERROR", "授权参数必须是对象", 422)
        service = _service(request, user)
        account = service.require(account_id)
        if account.get("enabled", True) is False:
            raise APIError("ACCOUNT_DISABLED", "该账号已停用，不能发起授权", 409)
        operation_body = {"account_id": account_id, "purpose": "colab", "auth_channel": "colab_cli"}
        previous = service._operation(key, operation_body)
        if previous is not None:
            return ok(previous, request)
        google = service_google(request, user["id"], account_id)
        try:
            result = google.start_colab_cli()
        finally:
            google.close()
        return ok(service._save_operation(key, operation_body, result, operation="colab_cli_start"), request)

    return _invoke(request, action)


@router.post("/{account_id}/auth/colab-cli/complete")
def complete_account_colab_cli_auth(
    account_id: str,
    body: dict,
    request: Request = None,
    user=Depends(get_current_user_for_tests),
):
    def action():
        key = _idempotency(request)
        if not isinstance(body, dict):
            raise APIError("VALIDATION_ERROR", "授权参数必须是对象", 422)
        state = body.get("state")
        code = body.get("code")
        if not isinstance(state, str) or not state.strip():
            raise APIError("VALIDATION_ERROR", "缺少授权会话标识 state", 422)
        if not isinstance(code, str) or not code.strip():
            raise APIError("VALIDATION_ERROR", "请粘贴 Google 页面显示的授权码", 422)
        service = _service(request, user)
        account = service.require(account_id)
        if account.get("enabled", True) is False:
            raise APIError("ACCOUNT_DISABLED", "该账号已停用，不能完成授权", 409)
        operation_body = {
            "account_id": account_id,
            "purpose": "colab",
            "auth_channel": "colab_cli",
            "state": state.strip(),
            "code_hash": service._hash(code.strip()),
        }
        previous = service._operation(key, operation_body)
        if previous is not None:
            return ok(previous, request)
        google = service_google(request, user["id"], account_id)
        try:
            result = google.complete_colab_cli(code=code, state=state.strip())
        finally:
            google.close()
        probe = probe_colab_cli_account(request, user["id"], account_id)
        payload = {
            "account_id": account_id,
            "purpose": result["purpose"],
            "auth_channel": result["auth_channel"],
            "connected": result["connected"],
            **probe,
        }
        return ok(service._save_operation(key, operation_body, payload, operation="colab_cli_complete"), request)

    return _invoke(request, action)


@router.post("/{account_id}/auth/colab-cli/import")
def import_account_colab_cli_auth(
    account_id: str,
    body: dict,
    request: Request = None,
    user=Depends(get_current_user_for_tests),
):
    def action():
        key = _idempotency(request)
        if not isinstance(body, dict) or not body:
            raise APIError("VALIDATION_ERROR", "导入参数必须是对象", 422)
        service = _service(request, user)
        account = service.require(account_id)
        if account.get("enabled", True) is False:
            raise APIError("ACCOUNT_DISABLED", "该账号已停用，不能导入授权", 409)
        operation_body = {
            "account_id": account_id,
            "purpose": "colab",
            "auth_channel": "colab_cli",
            "import_hash": service._hash(body),
        }
        previous = service._operation(key, operation_body)
        if previous is not None:
            return ok(previous, request)
        google = service_google(request, user["id"], account_id)
        try:
            result = google.import_colab_cli(body)
        finally:
            google.close()
        probe = probe_colab_cli_account(request, user["id"], account_id)
        payload = {
            "account_id": account_id,
            "purpose": result["purpose"],
            "auth_channel": result["auth_channel"],
            "connected": result["connected"],
            **probe,
        }
        return ok(service._save_operation(key, operation_body, payload, operation="colab_cli_import"), request)

    return _invoke(request, action)


@router.post("/{account_id}/auth/{purpose}/start")
def start_account_auth(
    account_id: str,
    purpose: str,
    body: dict | None = None,
    request: Request = None,
    user=Depends(get_current_user_for_tests),
):
    def action():
        if purpose not in {"colab", "drive"}:
            raise APIError("GOOGLE_PURPOSE_INVALID", "授权用途只能是 colab 或 drive", 422)
        key = _idempotency(request)
        body_value = body or {}
        if not isinstance(body_value, dict):
            raise APIError("VALIDATION_ERROR", "授权参数必须是对象", 422)
        requested = body_value.get("purpose")
        if requested is not None and requested != purpose:
            raise APIError("GOOGLE_PURPOSE_INVALID", "请求用途与路径不一致", 422)
        service = _service(request, user)
        account = service.require(account_id)
        if account.get("enabled", True) is False:
            raise APIError("ACCOUNT_DISABLED", "该账号已停用，不能发起授权", 409)
        operation_body = {"account_id": account_id, "purpose": purpose}
        previous = service._operation(key, operation_body)
        if previous is not None:
            return ok(previous, request)
        result = service_google(request, user["id"], account_id).start(purpose)
        return ok(service._save_operation(key, operation_body, result, operation="oauth_start"), request)

    return _invoke(request, action)


def probe_colab_cli_account(request: Request, owner_id: str, account_id: str) -> dict:
    """Best-effort read-only CCU & eligible GPU probe after CLI authorization."""
    from .gpu import ccu

    if not ccu.sampling_enabled():
        return {"balance": None, "eligible_gpus": [], "probe_error": None}
    repo = _repo(request)
    provider = None
    try:
        from .gpu.colab import GoogleColabProvider

        provider = GoogleColabProvider(repo, owner_id, account_id=account_id, timeout=8.0)
        info = provider.subscription()
        normalized = ccu.normalize_subscription(info)
        if normalized.get("balance") is not None and hasattr(repo, "store"):
            try:
                ccu.sample_and_record(
                    repo,
                    owner_id,
                    account_id,
                    "cli_auth",
                    sampler=lambda *_: normalized,
                )
            except Exception:
                pass
        gpus = [str(g).upper() for g in (info.get("eligibleGpus") or []) if isinstance(g, str) and g]
        return {
            "balance": normalized.get("balance"),
            "eligible_gpus": gpus,
            "probe_error": None,
        }
    except Exception:
        return {
            "balance": None,
            "eligible_gpus": [],
            "probe_error": "凭据已保存，但实时余额探测暂未响应（可稍后在算力页检查）",
        }
    finally:
        if provider is not None:
            try:
                provider.close()
            except Exception:
                pass


def service_google(request: Request, owner_id: str, account_id: str):
    # Imported lazily to keep importing the route free of network/client setup.
    from .gpu.google import GoogleService

    return GoogleService(_repo(request), owner_id, account_id=account_id)


@router.post("/{account_id}/disconnect", status_code=202)
def disconnect_account(
    account_id: str,
    body: dict | None = None,
    request: Request = None,
    user=Depends(get_current_user_for_tests),
):
    def action():
        key = _idempotency(request)
        return ok(_service(request, user).disconnect(account_id, key, body or {}), request, 202)

    return _invoke(request, action)


@router.delete("/{account_id}")
def delete_account(
    account_id: str,
    revoke_remote: bool = True,
    request: Request = None,
    user=Depends(get_current_user_for_tests),
):
    def action():
        key = _idempotency(request)
        return ok(_service(request, user).delete(account_id, key, revoke_remote=revoke_remote), request)

    return _invoke(request, action)


@router.get("/{account_id}/runtimes")
def account_runtimes(account_id: str, request: Request, user=Depends(get_current_user_for_tests)):
    return _invoke(request, lambda: ok(_service(request, user).runtimes(account_id), request))


__all__ = ["router", "get_current_user_for_tests"]
