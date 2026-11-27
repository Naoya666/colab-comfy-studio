"""Settings API endpoints for Telegram Bot push notifications."""

from __future__ import annotations

from fastapi import APIRouter, Body, Depends, Query, Request

from . import telegram_notify
from .common import APIError, admin, current_user, get_store, ok

router = APIRouter(prefix="/api/settings/telegram")


@router.get("")
def get_telegram_settings(request: Request, user=Depends(current_user), st=Depends(get_store)):
    return ok(telegram_notify.get_public_config(st, user["id"]), request)


@router.put("")
def put_telegram_settings(body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)):
    admin(user)
    try:
        saved = telegram_notify.save_config(st, user["id"], body)
    except ValueError as exc:
        raise APIError("INVALID_TELEGRAM_SETTINGS", str(exc), 422) from exc
    return ok(saved, request)


@router.post("/test")
def test_telegram_settings(
    request: Request,
    body: dict | None = Body(default=None),
    user=Depends(current_user),
    st=Depends(get_store),
):
    admin(user)
    try:
        result = telegram_notify.send_test_message(st, user["id"], body or {})
    except ValueError as exc:
        raise APIError("INVALID_TELEGRAM_SETTINGS", str(exc), 422) from exc
    except RuntimeError as exc:
        raise APIError("TELEGRAM_DELIVERY_FAILED", str(exc), 502) from exc
    return ok(result, request)


@router.get("/deliveries")
def list_telegram_deliveries(
    request: Request,
    limit: int = Query(20, ge=1, le=50),
    user=Depends(current_user),
    st=Depends(get_store),
):
    return ok({"items": telegram_notify.list_recent_deliveries(st, user["id"], limit=limit)}, request)
