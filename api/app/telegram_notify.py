"""Telegram Bot notification configuration, transactional outbox, and async dispatcher.

All GPU/scheduler/watchdog/model-worker paths enqueue into ``telegram_notifications``
without performing network I/O under dispatch or admission locks. The worker process
drains pending notifications asynchronously with retries, deduplication, and strict
credential redaction.
"""

from __future__ import annotations

import copy
import html
import logging
import os
import re
import time
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

import httpx
from sqlalchemy import delete as sql_delete
from sqlalchemy import func, select
from sqlalchemy import update as sql_update

from .config import settings
from .db import SettingRow, TelegramNotificationRow, utc_isoformat, utcnow
from .security import decrypt_secret, encrypt_secret

log = logging.getLogger("studio.telegram")

SETTINGS_PREFIX = "telegram-notify:"
DEFAULT_API_BASE_URL = "https://api.telegram.org"
MAX_MESSAGE_LENGTH = 3800
MAX_DELIVERY_ATTEMPTS = 5
RETRY_BACKOFF_SECONDS = (15, 30, 60, 180, 300)
RETENTION_DAYS = 7
MIN_KEEP_RECORDS = 200
DRIVE_LOW_QUOTA_MIN_FREE_BYTES = 5 * 1024 * 1024 * 1024  # 5 GiB

# Ordered catalog exposed to the Settings UI and used for defaults/validation.
EVENT_CATALOG: tuple[dict[str, Any], ...] = (
    # P0 — 默认勾选：高危止损与必须人工介入
    {
        "key": "cu_fuse",
        "priority": "P0",
        "default": True,
        "label": "CU 熔断与余额采样异常",
        "description": "触发余额底线硬熔断、今日 CU 上限温和熔断，或账号 CU 余额连续采样失败。",
    },
    {
        "key": "instance_quarantined",
        "priority": "P0",
        "default": True,
        "label": "实例进入数据保护隔离",
        "description": "检测到未同步数据或清理被拒而隔离实例（不会自动销毁，存在持续计费风险）。",
    },
    {
        "key": "cleanup_failed",
        "priority": "P0",
        "default": True,
        "label": "实例清理反复失败 / 疑似泄漏运行时",
        "description": "远端释放多次未确认（cleanup_pending），或检测到云端仍有活跃分配但本地无对应租约。",
    },
    {
        "key": "account_fatal",
        "priority": "P0",
        "default": True,
        "label": "Google / Colab 账号鉴权或配额致命异常",
        "description": "OAuth 刷新失败、Colab 鉴权失效、配额耗尽、账号受限/封禁或在抢卡中被踢出。",
    },
    {
        "key": "native_health_alert",
        "priority": "P0",
        "default": True,
        "label": "原生 ComfyUI 插件同步报错 / 隧道长时间失联",
        "description": "自定义节点持久化同步失败，或隧道持续 10 分钟不可达导致实例保守滞留。",
    },
    # P1 — 默认勾选：长耗时运维与实例生命周期
    {
        "key": "native_lifecycle",
        "priority": "P1",
        "default": True,
        "label": "原生 ComfyUI 启动就绪 / 启动失败",
        "description": "原生 ComfyUI 初始化完成可访问（含访问链接与规格），或启动失败时通知。",
    },
    {
        "key": "allocation_retry",
        "priority": "P1",
        "default": True,
        "label": "自动排队抢卡结果",
        "description": "后台重试抢卡最终成功、达到最大重试次数耗尽，或因限流/熔断/无可用账号中止。",
    },
    {
        "key": "auto_recovery",
        "priority": "P1",
        "default": True,
        "label": "L3 跨机自动恢复进展",
        "description": "检测到实例被 Colab 回收并启动自动恢复，以及恢复成功（含模型回拉状态）或中止。",
    },
    {
        "key": "transfer_and_restart",
        "priority": "P1",
        "default": True,
        "label": "GPU 跨域转移与原生同机重启结果",
        "description": "创作池 / 原生域 / 冒烟域转移完成或失败，以及原生 ComfyUI 同机重启完成或失败。",
    },
    {
        "key": "model_library",
        "priority": "P1",
        "default": True,
        "label": "模型库后台下载 / 校验 / 上传 Drive 结果",
        "description": "从 Hugging Face / Civitai / 外链下载并校验上传 Drive 完成或失败。",
    },
    # P2 — 默认不勾选：任务产出与定时摘要
    {
        "key": "batch_summary",
        "priority": "P2",
        "default": False,
        "label": "批次（Batch）整体完成汇总",
        "description": "批次全部任务结束后汇总推送成功 / 失败 / 取消数量，避免单张刷屏。",
    },
    {
        "key": "single_job",
        "priority": "P2",
        "default": False,
        "label": "独立创作任务完成 / 失败",
        "description": "非批次的创作任务完成或失败通知（默认过滤高频 Creation API / 酒馆成功消息）。",
    },
    {
        "key": "instance_expiry_billing",
        "priority": "P2",
        "default": False,
        "label": "实例到期前提醒与释放 CU 结算",
        "description": "实例距离空闲回收或硬截止前提醒，以及实例释放后推送时长与实际 CU 结算账单。",
    },
    {
        "key": "daily_report",
        "priority": "P2",
        "default": False,
        "label": "每日 CU 消费简报与 Drive 配额预警",
        "description": "按设定时间汇总当日 CU 消耗与各账号余额，并在 Drive 剩余空间偏低时提醒。",
    },
)

EVENT_META: dict[str, dict[str, Any]] = {item["key"]: item for item in EVENT_CATALOG}
DEFAULT_EVENT_FLAGS: dict[str, bool] = {item["key"]: bool(item["default"]) for item in EVENT_CATALOG}

_PRIORITY_BADGE = {
    "P0": "🚨 [紧急]",
    "P1": "🔔 [通知]",
    "P2": "📋 [摘要]",
}

_BOT_TOKEN_RE = re.compile(r"^\d{5,16}:[A-Za-z0-9_-]{20,128}$")
_CHAT_ID_RE = re.compile(r"^(?:-?\d{5,20}|@[A-Za-z][A-Za-z0-9_]{3,63})$")
_SECRET_QUERY_RE = re.compile(
    r"([?&](?:access_token|refresh_token|token|auth|key|client_secret|sig|signature|X-Goog-Signature)=)[^&\s\"'<>]+",
    re.IGNORECASE,
)
_BOT_URL_RE = re.compile(r"(bot)\d{5,16}:[A-Za-z0-9_-]{10,}", re.IGNORECASE)
_BEARER_RE = re.compile(r"(Bearer\s+)[A-Za-z0-9._~+/=-]{8,}", re.IGNORECASE)


def _base_store(store_or_repo):
    return store_or_repo.store if hasattr(store_or_repo, "store") else store_or_repo


def _settings_id(owner_id: str) -> str:
    return f"{SETTINGS_PREFIX}{owner_id}"


def _mask_token(token: str | None) -> str | None:
    if not isinstance(token, str) or not token:
        return None
    if ":" in token:
        prefix, secret = token.split(":", 1)
        tail = secret[-4:] if len(secret) >= 4 else "****"
        return f"{prefix}:****{tail}"
    return f"****{token[-4:]}" if len(token) >= 4 else "****"


def sanitize_text(value: Any, *, max_len: int = 1200) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = _BOT_URL_RE.sub(r"\1[REDACTED]", text)
    text = _SECRET_QUERY_RE.sub(r"\1[REDACTED]", text)
    text = _BEARER_RE.sub(r"\1[REDACTED]", text)
    if len(text) > max_len:
        text = text[: max_len - 1] + "…"
    return text


def _normalize_api_base_url(raw: Any) -> str:
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return DEFAULT_API_BASE_URL
    if not isinstance(raw, str):
        raise ValueError("Telegram API 地址必须是字符串")
    value = raw.strip().rstrip("/")
    parsed = urlsplit(value)
    if parsed.scheme.lower() not in {"https", "http"} or not parsed.netloc:
        raise ValueError("Telegram API 地址格式无效，例如 https://api.telegram.org")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Telegram API 地址不能包含凭据或查询参数")
    if parsed.scheme.lower() == "http" and parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("非本机 Telegram API 地址必须使用 HTTPS")
    return value


def _normalize_events(raw: Any) -> dict[str, bool]:
    merged = dict(DEFAULT_EVENT_FLAGS)
    if isinstance(raw, Mapping):
        for key in DEFAULT_EVENT_FLAGS:
            if key in raw:
                merged[key] = bool(raw[key])
    return merged


def _raw_config_in_tx(session, owner_id: str) -> dict[str, Any]:
    row = session.get(SettingRow, _settings_id(owner_id))
    if row is None or row.owner_id != owner_id or not isinstance(row.data, Mapping):
        return {}
    return dict(row.data)


def _merge_config(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    data = dict(raw or {})
    events = _normalize_events(data.get("events"))
    try:
        warn_min = int(data.get("expiry_warn_minutes", 15))
    except (TypeError, ValueError):
        warn_min = 15
    warn_min = max(5, min(60, warn_min))
    try:
        report_hour = int(data.get("daily_report_hour", 23))
    except (TypeError, ValueError):
        report_hour = 23
    report_hour = max(0, min(23, report_hour))
    try:
        drive_pct = int(data.get("drive_low_quota_percent", 10))
    except (TypeError, ValueError):
        drive_pct = 10
    drive_pct = max(1, min(50, drive_pct))
    raw_thread = data.get("message_thread_id")
    thread_id = int(raw_thread) if isinstance(raw_thread, int) and not isinstance(raw_thread, bool) and raw_thread > 0 else None
    try:
        api_base = _normalize_api_base_url(data.get("api_base_url"))
    except ValueError:
        api_base = DEFAULT_API_BASE_URL
    return {
        "enabled": bool(data.get("enabled", False)),
        "encrypted_bot_token": data.get("encrypted_bot_token"),
        "bot_token_hint": data.get("bot_token_hint"),
        "chat_id": str(data.get("chat_id") or "").strip(),
        "message_thread_id": thread_id,
        "api_base_url": api_base,
        "events": events,
        "include_creation_api_success": bool(data.get("include_creation_api_success", False)),
        "expiry_warn_minutes": warn_min,
        "daily_report_hour": report_hour,
        "drive_low_quota_percent": drive_pct,
        "updated_at": data.get("updated_at"),
    }


def load_config_in_tx(session, owner_id: str) -> dict[str, Any]:
    return _merge_config(_raw_config_in_tx(session, owner_id))


def load_config(store_or_repo, owner_id: str) -> dict[str, Any]:
    store = _base_store(store_or_repo)
    with store.db_lock(), store.Session() as session:
        return load_config_in_tx(session, owner_id)


def public_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "enabled": bool(cfg.get("enabled", False)),
        "bot_token_configured": bool(cfg.get("encrypted_bot_token")),
        "bot_token_hint": cfg.get("bot_token_hint"),
        "chat_id": str(cfg.get("chat_id") or ""),
        "message_thread_id": cfg.get("message_thread_id"),
        "api_base_url": cfg.get("api_base_url") or DEFAULT_API_BASE_URL,
        "events": dict(cfg.get("events") or DEFAULT_EVENT_FLAGS),
        "include_creation_api_success": bool(cfg.get("include_creation_api_success", False)),
        "expiry_warn_minutes": int(cfg.get("expiry_warn_minutes", 15)),
        "daily_report_hour": int(cfg.get("daily_report_hour", 23)),
        "drive_low_quota_percent": int(cfg.get("drive_low_quota_percent", 10)),
        "updated_at": cfg.get("updated_at"),
        "catalog": [dict(item) for item in EVENT_CATALOG],
        "timezone": settings.studio_timezone,
    }


def get_public_config(store_or_repo, owner_id: str) -> dict[str, Any]:
    return public_config(load_config(store_or_repo, owner_id))


def save_config(store_or_repo, owner_id: str, body: Mapping[str, Any], *, now: float | None = None) -> dict[str, Any]:
    if not isinstance(body, Mapping):
        raise ValueError("配置参数必须是对象")
    allowed_fields = {
        "enabled",
        "bot_token",
        "clear_bot_token",
        "chat_id",
        "message_thread_id",
        "api_base_url",
        "events",
        "include_creation_api_success",
        "expiry_warn_minutes",
        "daily_report_hour",
        "drive_low_quota_percent",
    }
    unknown = set(body.keys()) - allowed_fields
    if unknown:
        raise ValueError(f"包含不支持的配置字段: {', '.join(sorted(unknown))}")

    store = _base_store(store_or_repo)
    current = load_config(store, owner_id)

    enabled = body.get("enabled", current["enabled"])
    if type(enabled) is not bool:
        raise ValueError("enabled 必须为布尔值")

    clear_token = body.get("clear_bot_token", False)
    if type(clear_token) is not bool:
        raise ValueError("clear_bot_token 必须为布尔值")

    encrypted_token = current.get("encrypted_bot_token")
    token_hint = current.get("bot_token_hint")

    if clear_token:
        encrypted_token = None
        token_hint = None
    elif "bot_token" in body and body["bot_token"] is not None:
        raw_token = str(body["bot_token"]).strip()
        if raw_token:
            if not _BOT_TOKEN_RE.fullmatch(raw_token):
                raise ValueError("Telegram Bot Token 格式无效（应为 123456789:ABCdef...）")
            encrypted_token = encrypt_secret(raw_token)
            token_hint = _mask_token(raw_token)

    chat_id = current["chat_id"]
    if "chat_id" in body:
        chat_id = str(body.get("chat_id") or "").strip()
        if chat_id and not _CHAT_ID_RE.fullmatch(chat_id):
            raise ValueError("Chat ID 格式无效（请输入数字 ID、群组负号 ID 或 @频道名）")

    thread_id = current["message_thread_id"]
    if "message_thread_id" in body:
        raw_thread = body["message_thread_id"]
        if raw_thread in (None, ""):
            thread_id = None
        elif isinstance(raw_thread, bool):
            raise ValueError("话题 ID（message_thread_id）必须为正整数")
        else:
            try:
                parsed_thread = int(str(raw_thread).strip())
            except (TypeError, ValueError) as exc:
                raise ValueError("话题 ID（message_thread_id）必须为正整数") from exc
            if parsed_thread <= 0:
                raise ValueError("话题 ID（message_thread_id）必须为正整数")
            thread_id = parsed_thread

    api_base_url = _normalize_api_base_url(body.get("api_base_url", current["api_base_url"]))

    events = dict(current["events"])
    if "events" in body:
        raw_events = body["events"]
        if not isinstance(raw_events, Mapping):
            raise ValueError("events 必须是对象")
        unknown_events = set(raw_events.keys()) - set(DEFAULT_EVENT_FLAGS.keys())
        if unknown_events:
            raise ValueError(f"包含不支持的推送事件项: {', '.join(sorted(unknown_events))}")
        for key, val in raw_events.items():
            if type(val) is not bool:
                raise ValueError(f"事件开关 {key} 必须为布尔值")
            events[key] = val

    include_creation_api = body.get("include_creation_api_success", current["include_creation_api_success"])
    if type(include_creation_api) is not bool:
        raise ValueError("include_creation_api_success 必须为布尔值")

    expiry_warn = body.get("expiry_warn_minutes", current["expiry_warn_minutes"])
    if type(expiry_warn) is not int or not (5 <= expiry_warn <= 60):
        raise ValueError("到期前提醒分钟数必须在 5 ~ 60 之间")

    daily_hour = body.get("daily_report_hour", current["daily_report_hour"])
    if type(daily_hour) is not int or not (0 <= daily_hour <= 23):
        raise ValueError("每日简报发送小时必须在 0 ~ 23 之间")

    drive_pct = body.get("drive_low_quota_percent", current["drive_low_quota_percent"])
    if type(drive_pct) is not int or not (1 <= drive_pct <= 50):
        raise ValueError("Drive 空间预警阈值必须在 1% ~ 50% 之间")

    if enabled and (not encrypted_token or not chat_id):
        raise ValueError("启用 Telegram 推送前请先配置 Bot Token 和 Chat ID")

    ts = int(time.time() if now is None else now)
    stored_payload = {
        "enabled": enabled,
        "encrypted_bot_token": encrypted_token,
        "bot_token_hint": token_hint,
        "chat_id": chat_id,
        "message_thread_id": thread_id,
        "api_base_url": api_base_url,
        "events": events,
        "include_creation_api_success": include_creation_api,
        "expiry_warn_minutes": expiry_warn,
        "daily_report_hour": daily_hour,
        "drive_low_quota_percent": drive_pct,
        "updated_at": ts,
    }
    ident = _settings_id(owner_id)
    with store.db_lock(), store.Session.begin() as session:
        store.begin_immediate(session)
        row = session.get(SettingRow, ident)
        if row is None:
            session.add(
                SettingRow(
                    id=ident,
                    owner_id=owner_id,
                    state="active",
                    version=0,
                    data=stored_payload,
                    created_at=utcnow(),
                    updated_at=utcnow(),
                )
            )
        else:
            row.owner_id = owner_id
            row.data = stored_payload
            row.version = int(row.version or 0) + 1
            row.updated_at = utcnow()
        session.flush()
    return public_config(stored_payload)


def _format_local_time(epoch: float | None) -> str:
    if epoch is None:
        return "—"
    try:
        dt = datetime.fromtimestamp(float(epoch), ZoneInfo(settings.studio_timezone))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "—"


def format_message(
    event_key: str,
    title: str,
    lines: Sequence[tuple[str, Any] | str],
    *,
    link_url: str | None = None,
    link_label: str | None = None,
    now: float | None = None,
) -> str:
    meta = EVENT_META.get(event_key, {"priority": "P1"})
    badge = _PRIORITY_BADGE.get(meta["priority"], "🔔 [通知]")
    safe_title = html.escape(sanitize_text(title, max_len=160))
    parts = [f"<b>{badge} {safe_title}</b>"]
    for item in lines:
        if isinstance(item, tuple) and len(item) == 2:
            label, raw_val = item
            val_str = sanitize_text(raw_val, max_len=600)
            if not val_str:
                continue
            parts.append(f"• <b>{html.escape(str(label))}</b>：{html.escape(val_str)}")
        else:
            val_str = sanitize_text(item, max_len=600)
            if val_str:
                parts.append(html.escape(val_str))
    if link_url:
        clean_url = sanitize_text(link_url, max_len=500)
        parsed = urlsplit(clean_url)
        if parsed.scheme in {"http", "https"} and parsed.netloc:
            label = html.escape(sanitize_text(link_label or "打开工作台", max_len=60))
            parts.append(f"🔗 <a href=\"{html.escape(clean_url, quote=True)}\">{label}</a>")
    parts.append(f"<i>Colab GPU Studio · {_format_local_time(time.time() if now is None else now)}</i>")
    rendered = "\n".join(parts)
    if len(rendered) > MAX_MESSAGE_LENGTH:
        rendered = rendered[: MAX_MESSAGE_LENGTH - 20] + "\n…"
    return rendered


def _should_skip_dedupe_in_tx(
    session,
    owner_id: str,
    dedupe_key: str | None,
    cooldown_seconds: float,
    now_dt: datetime,
) -> bool:
    if not dedupe_key:
        return False
    stmt = (
        select(TelegramNotificationRow)
        .where(
            TelegramNotificationRow.owner_id == owner_id,
            TelegramNotificationRow.dedupe_key == dedupe_key,
        )
        .order_by(TelegramNotificationRow.created_at.desc())
        .limit(1)
    )
    existing = session.scalar(stmt)
    if existing is None:
        return False
    if existing.state in {"pending", "sending"}:
        return True
    if cooldown_seconds <= 0:
        return False
    created = existing.created_at
    if created is None:
        return False
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return (now_dt - created).total_seconds() < cooldown_seconds


def enqueue_in_tx(
    session,
    owner_id: str | None,
    event_key: str,
    title: str,
    lines: Sequence[tuple[str, Any] | str],
    *,
    dedupe_key: str | None = None,
    cooldown_seconds: float = 0.0,
    project_id: str | None = None,
    link_url: str | None = None,
    link_label: str | None = None,
    now: float | None = None,
) -> dict[str, Any] | None:
    """Enqueue a Telegram notification inside an active SQLAlchemy session (zero network I/O)."""
    if not owner_id or event_key not in EVENT_META:
        return None
    # Flush prior business writes before opening a SAVEPOINT so a notification
    # failure rolls back only the notification write and never poisons the outer TX.
    session.flush()
    try:
        with session.begin_nested():
            cfg = load_config_in_tx(session, owner_id)
            if not cfg["enabled"] or not cfg.get("encrypted_bot_token") or not cfg.get("chat_id"):
                return None
            if not cfg["events"].get(event_key, DEFAULT_EVENT_FLAGS.get(event_key, False)):
                return None

            ts = float(time.time() if now is None else now)
            now_dt = datetime.fromtimestamp(ts, timezone.utc)
            norm_dedupe = str(dedupe_key)[:190] if dedupe_key else None
            if _should_skip_dedupe_in_tx(session, owner_id, norm_dedupe, cooldown_seconds, now_dt):
                return None

            priority = EVENT_META[event_key]["priority"]
            text = format_message(
                event_key,
                title,
                lines,
                link_url=link_url,
                link_label=link_label,
                now=ts,
            )
            row_id = uuid.uuid4().hex
            payload = {
                "event_key": event_key,
                "priority": priority,
                "title": sanitize_text(title, max_len=160),
                "text": text,
                "dedupe_key": norm_dedupe,
                "attempts": 0,
                "max_attempts": MAX_DELIVERY_ATTEMPTS,
                "next_attempt_at": ts,
                "last_error": None,
                "sent_at": None,
                "telegram_message_id": None,
                "disable_notification": priority == "P2",
            }
            row = TelegramNotificationRow(
                id=row_id,
                owner_id=owner_id,
                project_id=project_id,
                state="pending",
                dedupe_key=norm_dedupe,
                version=0,
                data=payload,
                created_at=now_dt,
                updated_at=now_dt,
            )
            session.add(row)
            session.flush()
            return {"id": row_id, "state": "pending", **payload}
    except Exception as exc:  # noqa: BLE001 - notification must never break business TX
        log.warning("telegram enqueue_in_tx failed event=%s: %s", event_key, type(exc).__name__)
        return None


def enqueue(
    store_or_repo,
    owner_id: str | None,
    event_key: str,
    title: str,
    lines: Sequence[tuple[str, Any] | str],
    *,
    dedupe_key: str | None = None,
    cooldown_seconds: float = 0.0,
    project_id: str | None = None,
    link_url: str | None = None,
    link_label: str | None = None,
    now: float | None = None,
) -> dict[str, Any] | None:
    """Open a short DB transaction to enqueue a notification safely outside locks."""
    if not owner_id or event_key not in EVENT_META or store_or_repo is None:
        return None
    try:
        store = _base_store(store_or_repo)
        if not (hasattr(store, "db_lock") and hasattr(store, "Session")):
            return None
        with store.db_lock(), store.Session.begin() as session:
            store.begin_immediate(session)
            return enqueue_in_tx(
                session,
                owner_id,
                event_key,
                title,
                lines,
                dedupe_key=dedupe_key,
                cooldown_seconds=cooldown_seconds,
                project_id=project_id,
                link_url=link_url,
                link_label=link_label,
                now=now,
            )
    except Exception as exc:  # noqa: BLE001 - notification must never break callers
        log.warning("telegram enqueue failed event=%s: %s", event_key, type(exc).__name__)
        return None


def _outbound_allowed(*, has_custom_client: bool = False) -> bool:
    if has_custom_client:
        return True
    if os.getenv("STUDIO_OFFLINE_TEST") == "1":
        return False
    if os.getenv("PYTEST_CURRENT_TEST"):
        return False
    return True


def _send_telegram_http(
    *,
    bot_token: str,
    chat_id: str,
    text: str,
    api_base_url: str = DEFAULT_API_BASE_URL,
    message_thread_id: int | None = None,
    disable_notification: bool = False,
    client: httpx.Client | None = None,
    timeout: float = 12.0,
) -> tuple[bool, dict[str, Any]]:
    """Send one message via Telegram Bot API. Never leaks bot_token in returned errors."""
    if not _outbound_allowed(has_custom_client=client is not None):
        return False, {"error": "当前处于离线/测试模式，已拦截 Telegram 外网发送", "retryable": False}

    url = f"{api_base_url.rstrip('/')}/bot{bot_token}/sendMessage"
    body: dict[str, Any] = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
        "disable_notification": bool(disable_notification),
    }
    if message_thread_id:
        body["message_thread_id"] = int(message_thread_id)

    owns_client = client is None
    http_client = client or httpx.Client(timeout=timeout)
    try:
        response = http_client.post(url, json=body)
        payload = {}
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        if response.status_code == 200 and payload.get("ok") is True:
            result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
            return True, {"message_id": result.get("message_id")}
        desc = sanitize_text(payload.get("description") or f"HTTP {response.status_code}", max_len=240)
        retry_after = None
        params = payload.get("parameters") if isinstance(payload.get("parameters"), dict) else {}
        if isinstance(params.get("retry_after"), (int, float)):
            retry_after = max(1.0, float(params["retry_after"]))
        retryable = response.status_code == 429 or response.status_code >= 500
        return False, {
            "error": f"Telegram API 返回错误（HTTP {response.status_code}）：{desc}",
            "http_status": response.status_code,
            "retryable": retryable,
            "retry_after": retry_after,
        }
    except httpx.TimeoutException:
        return False, {"error": "连接 Telegram API 超时", "retryable": True}
    except httpx.HTTPError as exc:
        return False, {"error": f"连接 Telegram API 失败：{type(exc).__name__}", "retryable": True}
    except Exception as exc:  # noqa: BLE001
        return False, {"error": f"发送 Telegram 消息异常：{type(exc).__name__}", "retryable": False}
    finally:
        if owns_client:
            http_client.close()


def send_test_message(
    store_or_repo,
    owner_id: str,
    override: Mapping[str, Any] | None = None,
    *,
    client: httpx.Client | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Immediately send a test message using saved config (optionally overridden by unsaved draft fields)."""
    store = _base_store(store_or_repo)
    saved = load_config(store, owner_id)
    draft = override if isinstance(override, Mapping) else {}

    raw_token = str(draft.get("bot_token") or "").strip()
    if raw_token:
        if not _BOT_TOKEN_RE.fullmatch(raw_token):
            raise ValueError("Telegram Bot Token 格式无效（应为 123456789:ABCdef...）")
        bot_token = raw_token
    else:
        encrypted = saved.get("encrypted_bot_token")
        if not encrypted:
            raise ValueError("请先填写并保存 Telegram Bot Token")
        bot_token = decrypt_secret(encrypted)

    chat_id = str(draft.get("chat_id") if draft.get("chat_id") not in (None, "") else saved.get("chat_id") or "").strip()
    if not chat_id or not _CHAT_ID_RE.fullmatch(chat_id):
        raise ValueError("请先填写有效的 Chat ID")

    thread_id = saved.get("message_thread_id")
    if "message_thread_id" in draft:
        raw_t = draft.get("message_thread_id")
        if raw_t in (None, ""):
            thread_id = None
        else:
            try:
                thread_id = int(str(raw_t).strip())
                if thread_id <= 0:
                    raise ValueError
            except (TypeError, ValueError) as exc:
                raise ValueError("话题 ID 必须为正整数") from exc

    api_base_url = _normalize_api_base_url(draft.get("api_base_url") or saved.get("api_base_url"))
    ts = float(time.time() if now is None else now)
    events = _normalize_events(draft.get("events") if isinstance(draft.get("events"), Mapping) else saved.get("events"))
    enabled_groups = []
    for prio in ("P0", "P1", "P2"):
        keys = [item["key"] for item in EVENT_CATALOG if item["priority"] == prio]
        active_cnt = sum(1 for k in keys if events.get(k))
        enabled_groups.append(f"{prio} {active_cnt}/{len(keys)}")

    text = format_message(
        "native_lifecycle",
        "Telegram 推送测试成功",
        [
            ("状态", "Bot Token 与 Chat ID 连通性验证通过"),
            ("已勾选事件", " · ".join(enabled_groups)),
            ("站点", settings.site_url),
        ],
        link_url=settings.site_url,
        link_label="打开 Colab GPU Studio",
        now=ts,
    )
    ok_sent, detail = _send_telegram_http(
        bot_token=bot_token,
        chat_id=chat_id,
        text=text,
        api_base_url=api_base_url,
        message_thread_id=thread_id,
        disable_notification=False,
        client=client,
    )

    now_dt = datetime.fromtimestamp(ts, timezone.utc)
    record_state = "sent" if ok_sent else "failed"
    with store.db_lock(), store.Session.begin() as session:
        store.begin_immediate(session)
        session.add(
            TelegramNotificationRow(
                id=uuid.uuid4().hex,
                owner_id=owner_id,
                state=record_state,
                dedupe_key=None,
                version=0,
                data={
                    "event_key": "test",
                    "priority": "TEST",
                    "title": "Telegram 推送测试",
                    "text": text,
                    "attempts": 1,
                    "max_attempts": 1,
                    "sent_at": ts if ok_sent else None,
                    "telegram_message_id": detail.get("message_id"),
                    "last_error": None if ok_sent else detail.get("error"),
                },
                created_at=now_dt,
                updated_at=now_dt,
            )
        )
    if not ok_sent:
        raise RuntimeError(str(detail.get("error") or "Telegram 测试发送失败"))
    return {
        "ok": True,
        "message_id": detail.get("message_id"),
        "sent_at": ts,
    }


def dispatch_pending(
    store_or_repo,
    *,
    max_batch: int = 20,
    client: httpx.Client | None = None,
    now: float | None = None,
) -> dict[str, int]:
    """Drain pending outbox notifications and send via Telegram Bot API."""
    stats = {"claimed": 0, "sent": 0, "retried": 0, "failed": 0, "skipped": 0}
    if not _outbound_allowed(has_custom_client=client is not None):
        return stats
    store = _base_store(store_or_repo)
    ts = float(time.time() if now is None else now)

    # Reclaim stuck 'sending' rows BEFORE claiming so an otherwise idle queue
    # never leaves crashed deliveries permanently wedged or blocking dedupe keys.
    _reclaim_stuck_sending(store, ts)

    due_expr = func.coalesce(TelegramNotificationRow.data["next_attempt_at"].as_float(), 0.0)
    now_dt = datetime.fromtimestamp(ts, timezone.utc)
    claimed: list[dict[str, Any]] = []
    with store.db_lock(), store.Session.begin() as session:
        store.begin_immediate(session)
        stmt = (
            select(TelegramNotificationRow)
            .where(
                TelegramNotificationRow.state == "pending",
                due_expr <= ts,
            )
            .order_by(TelegramNotificationRow.created_at.asc())
            .limit(max_batch)
        )
        if store.engine.dialect.name != "sqlite":
            stmt = stmt.with_for_update(skip_locked=True)
        rows = session.scalars(stmt).all()
        for row in rows:
            data = copy.deepcopy(row.data or {})
            next_at = float(data.get("next_attempt_at") or 0.0)
            if next_at > ts:
                continue
            cur_version = int(row.version or 0)
            data["state"] = "sending"
            data["last_attempt_at"] = ts
            res = session.execute(
                sql_update(TelegramNotificationRow)
                .where(
                    TelegramNotificationRow.id == row.id,
                    TelegramNotificationRow.state == "pending",
                    TelegramNotificationRow.version == cur_version,
                )
                .values(
                    state="sending",
                    data=data,
                    version=cur_version + 1,
                    updated_at=now_dt,
                )
            )
            if res.rowcount != 1:
                continue
            claimed.append({"id": row.id, "owner_id": row.owner_id, "data": data})
            if len(claimed) >= max_batch:
                break

    if not claimed:
        return stats

    stats["claimed"] = len(claimed)
    owns_client = client is None
    http_client = client or httpx.Client(timeout=12.0)
    cfg_cache: dict[str, dict[str, Any]] = {}
    token_cache: dict[str, str | None] = {}
    try:
        for item in claimed:
            owner_id = str(item["owner_id"] or "")
            data = item["data"]
            if owner_id not in cfg_cache:
                cfg_cache[owner_id] = load_config(store, owner_id)
                enc = cfg_cache[owner_id].get("encrypted_bot_token")
                try:
                    token_cache[owner_id] = decrypt_secret(enc) if enc else None
                except ValueError:
                    token_cache[owner_id] = None
            cfg = cfg_cache[owner_id]
            bot_token = token_cache[owner_id]

            if not cfg.get("enabled") or not bot_token or not cfg.get("chat_id"):
                _finalize_delivery(
                    store,
                    item["id"],
                    owner_id,
                    state="skipped",
                    patch={"last_error": "Telegram 推送已关闭或凭据不完整"},
                    ts=ts,
                )
                stats["skipped"] += 1
                continue

            ok_sent, detail = _send_telegram_http(
                bot_token=bot_token,
                chat_id=cfg["chat_id"],
                text=str(data.get("text") or ""),
                api_base_url=cfg.get("api_base_url") or DEFAULT_API_BASE_URL,
                message_thread_id=cfg.get("message_thread_id"),
                disable_notification=bool(data.get("disable_notification", False)),
                client=http_client,
            )
            attempts = int(data.get("attempts") or 0) + 1
            max_attempts = int(data.get("max_attempts") or MAX_DELIVERY_ATTEMPTS)
            if ok_sent:
                _finalize_delivery(
                    store,
                    item["id"],
                    owner_id,
                    state="sent",
                    patch={
                        "attempts": attempts,
                        "sent_at": ts,
                        "telegram_message_id": detail.get("message_id"),
                        "last_error": None,
                    },
                    ts=ts,
                )
                stats["sent"] += 1
            elif detail.get("retryable") and attempts < max_attempts:
                delay = detail.get("retry_after")
                if delay is None:
                    delay = RETRY_BACKOFF_SECONDS[min(attempts - 1, len(RETRY_BACKOFF_SECONDS) - 1)]
                _finalize_delivery(
                    store,
                    item["id"],
                    owner_id,
                    state="pending",
                    patch={
                        "attempts": attempts,
                        "next_attempt_at": ts + float(delay),
                        "last_error": detail.get("error"),
                    },
                    ts=ts,
                )
                stats["retried"] += 1
            else:
                _finalize_delivery(
                    store,
                    item["id"],
                    owner_id,
                    state="failed",
                    patch={
                        "attempts": attempts,
                        "last_error": detail.get("error"),
                    },
                    ts=ts,
                )
                stats["failed"] += 1
    finally:
        if owns_client:
            http_client.close()

    return stats


def _finalize_delivery(store, row_id: str, owner_id: str, *, state: str, patch: dict[str, Any], ts: float) -> None:
    with store.db_lock(), store.Session.begin() as session:
        store.begin_immediate(session)
        row = session.get(TelegramNotificationRow, row_id)
        if row is None or row.owner_id != owner_id:
            return
        merged = {**(row.data or {}), **patch, "state": state}
        row.state = state
        row.data = merged
        row.version = int(row.version or 0) + 1
        row.updated_at = datetime.fromtimestamp(ts, timezone.utc)


def _reclaim_stuck_sending(store, ts: float) -> None:
    cutoff = datetime.fromtimestamp(ts - 120.0, timezone.utc)
    now_dt = datetime.fromtimestamp(ts, timezone.utc)
    try:
        with store.db_lock(), store.Session.begin() as session:
            store.begin_immediate(session)
            stmt = select(TelegramNotificationRow).where(
                TelegramNotificationRow.state == "sending",
                TelegramNotificationRow.updated_at < cutoff,
            )
            if store.engine.dialect.name != "sqlite":
                stmt = stmt.with_for_update(skip_locked=True)
            stuck = session.scalars(stmt).all()
            for row in stuck:
                data = copy.deepcopy(row.data or {})
                cur_version = int(row.version or 0)
                attempts = int(data.get("attempts") or 0) + 1
                max_attempts = int(data.get("max_attempts") or MAX_DELIVERY_ATTEMPTS)
                if attempts < max_attempts:
                    next_state = "pending"
                    data.update(state="pending", attempts=attempts, next_attempt_at=ts)
                else:
                    next_state = "failed"
                    data.update(state="failed", attempts=attempts, last_error="发送进程中断超时")
                session.execute(
                    sql_update(TelegramNotificationRow)
                    .where(
                        TelegramNotificationRow.id == row.id,
                        TelegramNotificationRow.state == "sending",
                        TelegramNotificationRow.version == cur_version,
                    )
                    .values(
                        state=next_state,
                        data=data,
                        version=cur_version + 1,
                        updated_at=now_dt,
                    )
                )
    except Exception:  # noqa: BLE001
        pass


def prune_old_deliveries(store_or_repo, *, now: float | None = None) -> int:
    store = _base_store(store_or_repo)
    ts = float(time.time() if now is None else now)
    cutoff = datetime.fromtimestamp(ts - RETENTION_DAYS * 86400.0, timezone.utc)
    try:
        with store.db_lock(), store.Session.begin() as session:
            store.begin_immediate(session)
            result = session.execute(
                sql_delete(TelegramNotificationRow).where(
                    TelegramNotificationRow.state.in_(("sent", "failed", "skipped")),
                    TelegramNotificationRow.created_at < cutoff,
                )
            )
            return int(result.rowcount or 0)
    except Exception:  # noqa: BLE001
        return 0


def list_recent_deliveries(store_or_repo, owner_id: str, *, limit: int = 20) -> list[dict[str, Any]]:
    store = _base_store(store_or_repo)
    bounded = max(1, min(50, int(limit)))
    with store.db_lock(), store.Session() as session:
        rows = session.scalars(
            select(TelegramNotificationRow)
            .where(TelegramNotificationRow.owner_id == owner_id)
            .order_by(TelegramNotificationRow.created_at.desc())
            .limit(bounded)
        ).all()
        items: list[dict[str, Any]] = []
        for row in rows:
            data = row.data or {}
            items.append(
                {
                    "id": row.id,
                    "event_key": data.get("event_key"),
                    "priority": data.get("priority"),
                    "title": data.get("title"),
                    "state": row.state,
                    "attempts": int(data.get("attempts") or 0),
                    "last_error": data.get("last_error"),
                    "sent_at": data.get("sent_at"),
                    "created_at": utc_isoformat(row.created_at),
                }
            )
        return items


# ---------------------------------------------------------------------------
# High-level semantic helpers (P0 / P1 / P2)
# ---------------------------------------------------------------------------

def _site_link(path: str = "") -> str | None:
    base = str(settings.site_url or "").strip().rstrip("/")
    if not base:
        return None
    return f"{base}/{path.lstrip('/')}" if path else base


def _format_bytes(size: int | float | None) -> str:
    if size is None or size < 0:
        return "—"
    val = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if val < 1024 or unit == "TB":
            return f"{val:.2f} {unit}" if unit in {"GB", "TB"} else f"{val:.1f} {unit}" if unit != "B" else f"{int(val)} B"
        val /= 1024.0
    return f"{val:.2f} TB"


def _format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "—"
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours} 小时 {minutes} 分"
    if minutes > 0:
        return f"{minutes} 分 {secs} 秒"
    return f"{secs} 秒"


def _spec_text(spec: Any) -> str:
    if not spec:
        return "—"
    if isinstance(spec, str):
        return spec
    if isinstance(spec, Mapping):
        inner = spec.get("key") if isinstance(spec.get("key"), Mapping) else spec
        acc = str(inner.get("accelerator") or inner.get("gpu") or "").upper()
        shape = str(inner.get("shape") or "").upper()
        if not acc:
            return "—"
        return f"{acc}（高内存）" if "HIGHMEM" in shape else acc
    return str(spec)


def _pool_label(pool: str | None) -> str:
    return {
        "native": "原生 ComfyUI",
        "creation": "创作池",
        "ephemeral": "冒烟域",
    }.get(str(pool or ""), str(pool or "GPU"))


def notify_cu_fuse_in_tx(
    session,
    owner_id: str,
    *,
    level: str,
    message: str,
    instance_id: str | None = None,
    account_label: str | None = None,
    today_cu: float | None = None,
    daily_limit: float | None = None,
    balance: float | None = None,
    min_balance: float | None = None,
    native: bool = False,
    dedupe_key: str | None = None,
    now: float | None = None,
) -> dict[str, Any] | None:
    is_hard = level == "hard_balance_floor"
    title = "CU 余额底线硬熔断（立即停机止损）" if is_hard else "今日 CU 上限温和熔断"
    lines: list[tuple[str, Any]] = [("说明", message)]
    if account_label:
        lines.append(("账号", account_label))
    if is_hard and balance is not None:
        floor_txt = f" / 底线 {min_balance:g} CU" if min_balance is not None else ""
        lines.append(("账号余额", f"{balance:g} CU{floor_txt}"))
    if not is_hard and today_cu is not None:
        lim_txt = f" / 上限 {daily_limit:g} CU" if daily_limit is not None else ""
        lines.append(("今日实耗", f"{today_cu:g} CU{lim_txt}"))
    if instance_id:
        lines.append(("实例", f"{instance_id[:12]}（{'原生 ComfyUI' if native else '创作实例'}）"))
    return enqueue_in_tx(
        session,
        owner_id,
        "cu_fuse",
        title,
        lines,
        dedupe_key=dedupe_key or f"cu_fuse:{level}:{instance_id or account_label or 'owner'}",
        cooldown_seconds=1800,
        link_url=_site_link("compute"),
        link_label="查看算力与 CU 熔断",
        now=now,
    )


def notify_cu_sample_failed_in_tx(
    session,
    owner_id: str,
    *,
    account_id: str,
    account_label: str | None,
    consecutive_failures: int,
    now: float | None = None,
) -> dict[str, Any] | None:
    return enqueue_in_tx(
        session,
        owner_id,
        "cu_fuse",
        "CU 余额连续采样失败",
        [
            ("账号", account_label or account_id),
            ("连续失败", f"{consecutive_failures} 次"),
            ("风险提示", "无法获取最新 CU 余额可能导致余额底线熔断存在盲区，请检查账号授权或网络"),
        ],
        dedupe_key=f"cu_sample_failed:{account_id}",
        cooldown_seconds=3600,
        link_url=_site_link("settings"),
        link_label="检查账号设置",
        now=now,
    )


def notify_instance_quarantined_in_tx(
    session,
    owner_id: str,
    *,
    instance_id: str | None,
    lease_id: str | None,
    reason: str | None,
    message: str,
    pool: str | None = None,
    project_id: str | None = None,
    now: float | None = None,
) -> dict[str, Any] | None:
    return enqueue_in_tx(
        session,
        owner_id,
        "instance_quarantined",
        "GPU 实例已进入数据保护隔离",
        [
            ("实例", (instance_id or lease_id or "—")[:16]),
            ("所属域", _pool_label(pool) if pool else "GPU"),
            ("原因", reason or "unsynced_data"),
            ("详情", message),
            ("操作提醒", "系统未自动销毁该 GPU（仍在持续计费），请尽快到实例中心确认后手动停止"),
        ],
        dedupe_key=f"quarantined:{instance_id or lease_id}",
        cooldown_seconds=1800,
        project_id=project_id,
        link_url=_site_link("compute"),
        link_label="前往实例中心处理",
        now=now,
    )


def notify_cleanup_failed(
    store_or_repo,
    owner_id: str,
    *,
    lease_id: str,
    instance_id: str | None = None,
    account_id: str | None = None,
    attempts: int = 0,
    error: str | None = None,
    now: float | None = None,
) -> dict[str, Any] | None:
    return enqueue(
        store_or_repo,
        owner_id,
        "cleanup_failed",
        "GPU 实例释放多次未确认（存在空烧 CU 风险）",
        [
            ("实例 / 租约", f"{(instance_id or '—')[:12]} / {lease_id[:12]}"),
            ("账号", account_id or "—"),
            ("重试次数", f"已尝试 {attempts} 次"),
            ("最近错误", error or "Colab 未确认删除"),
            ("建议", "请检查 Colab 运行时页面或实例中心，确认远端机器是否已释放"),
        ],
        dedupe_key=f"cleanup_failed:{lease_id}",
        cooldown_seconds=1800,
        link_url=_site_link("compute"),
        link_label="查看算力与实例中心",
        now=now,
    )


def notify_leaked_runtime_in_tx(
    session,
    owner_id: str,
    *,
    account_id: str,
    account_label: str | None,
    assignments: int,
    balance: float | None = None,
    rate_hourly: float | None = None,
    now: float | None = None,
) -> dict[str, Any] | None:
    lines: list[tuple[str, Any]] = [
        ("账号", account_label or account_id),
        ("云端活跃分配", f"{assignments} 台（本地无对应活跃任务或租约）"),
    ]
    if balance is not None:
        lines.append(("当前余额", f"{balance:g} CU"))
    if rate_hourly is not None:
        lines.append(("当前消耗速率", f"{rate_hourly:g} CU/小时"))
    lines.append(("建议", "可能存在残留 Colab 运行时正在计费，请立即登录 Colab 检查并断开"))
    return enqueue_in_tx(
        session,
        owner_id,
        "cleanup_failed",
        "检测到疑似泄漏的 Colab 运行时",
        lines,
        dedupe_key=f"leaked_runtime:{account_id}",
        cooldown_seconds=1800,
        link_url=_site_link("compute"),
        link_label="查看算力监控",
        now=now,
    )


def notify_account_fatal(
    store_or_repo,
    owner_id: str,
    *,
    account_id: str | None,
    account_label: str | None = None,
    error_code: str | None = None,
    message: str | None = None,
    context: str | None = None,
    session=None,
    now: float | None = None,
) -> dict[str, Any] | None:
    lines: list[tuple[str, Any]] = [
        ("账号", account_label or account_id or "默认账号"),
        ("错误码", error_code or "account_fatal"),
        ("原因", message or "账号鉴权失效、配额耗尽或已被限制"),
    ]
    if context:
        lines.append(("影响", context))
    kwargs = dict(
        dedupe_key=f"account_fatal:{account_id or 'default'}:{error_code or 'fatal'}",
        cooldown_seconds=1800,
        link_url=_site_link("settings"),
        link_label="前往账号设置重新授权",
        now=now,
    )
    if session is not None:
        return enqueue_in_tx(session, owner_id, "account_fatal", "Google / Colab 账号异常", lines, **kwargs)
    return enqueue(store_or_repo, owner_id, "account_fatal", "Google / Colab 账号异常", lines, **kwargs)


def notify_native_health_alert(
    store_or_repo,
    owner_id: str,
    *,
    lease_id: str,
    instance_id: str | None,
    kind: str,
    detail: str,
    now: float | None = None,
) -> dict[str, Any] | None:
    title = "原生 ComfyUI 插件持久化同步异常" if kind == "plugin_sync" else "原生 ComfyUI 隧道长时间失联"
    lines: list[tuple[str, Any]] = [
        ("实例", (instance_id or lease_id)[:16]),
        ("详情", detail),
    ]
    if kind == "tunnel_unknown":
        lines.append(("计费提示", "网关无法确认队列状态，系统已保守保留实例，请尽快检查或手动停止以防空烧 CU"))
    return enqueue(
        store_or_repo,
        owner_id,
        "native_health_alert",
        title,
        lines,
        dedupe_key=f"native_health:{kind}:{lease_id}",
        cooldown_seconds=1800,
        link_url=_site_link("native"),
        link_label="查看原生 ComfyUI 状态",
        now=now,
    )


def notify_native_lifecycle(
    store_or_repo,
    owner_id: str,
    *,
    ready: bool,
    job: Mapping[str, Any],
    instance: Mapping[str, Any] | None = None,
    native_url: str | None = None,
    deadline_epoch: float | None = None,
    idle_minutes: int | float | None = None,
    now: float | None = None,
) -> dict[str, Any] | None:
    spec = (
        (instance or {}).get("runtime_spec_reported")
        or job.get("runtime_spec_reported")
        or job.get("requested_spec")
    )
    acct = job.get("account_alias_snapshot") or job.get("account_id") or "—"
    source_note = None
    if job.get("recovery_source") == "auto_recovery":
        source_note = "L3 跨机自动恢复"
    elif job.get("transferred_native_bootstrap"):
        source_note = "GPU 跨域转移至原生域"
    elif (job.get("execution_policy") or {}).get("auto_started") or job.get("native_api_auto_start"):
        source_note = "API 自动开机"

    if ready:
        title = "原生 ComfyUI 已就绪"
        lines: list[tuple[str, Any]] = [
            ("GPU 规格", _spec_text(spec)),
            ("账号", acct),
        ]
        if source_note:
            lines.append(("来源", source_note))
        if idle_minutes is not None:
            lines.append(("空闲回收", f"{int(round(float(idle_minutes)))} 分钟"))
        if deadline_epoch is not None:
            lines.append(("会话上限", _format_local_time(deadline_epoch)))
        target_link = native_url or _site_link("native")
        return enqueue(
            store_or_repo,
            owner_id,
            "native_lifecycle",
            title,
            lines,
            dedupe_key=f"native_ready:{job.get('id')}",
            cooldown_seconds=3600,
            project_id=job.get("project_id"),
            link_url=target_link,
            link_label="打开原生 ComfyUI" if native_url else "打开原生管理页",
            now=now,
        )

    title = "原生 ComfyUI 启动失败"
    lines = [
        ("GPU 规格", _spec_text(spec)),
        ("账号", acct),
        ("阶段", job.get("stage") or "bootstrap"),
        ("错误码", job.get("error_code") or "native_bootstrap_failed"),
        ("原因", job.get("error_message") or "启动未能完成，请查看事件日志"),
    ]
    return enqueue(
        store_or_repo,
        owner_id,
        "native_lifecycle",
        title,
        lines,
        dedupe_key=f"native_failed:{job.get('id')}",
        cooldown_seconds=3600,
        project_id=job.get("project_id"),
        link_url=_site_link("native"),
        link_label="查看启动详情",
        now=now,
    )


def notify_allocation_retry(
    store_or_repo,
    owner_id: str,
    *,
    succeeded: bool,
    job_id: str,
    pool: str | None = None,
    spec: Any = None,
    account_label: str | None = None,
    retries_used: int = 0,
    max_retries: int | None = None,
    reason: str | None = None,
    project_id: str | None = None,
    session=None,
    now: float | None = None,
) -> dict[str, Any] | None:
    if succeeded:
        title = "自动排队抢卡成功"
        lines: list[tuple[str, Any]] = [
            ("目标域", _pool_label(pool)),
            ("分配显卡", _spec_text(spec)),
            ("分配账号", account_label or "—"),
            ("重试统计", f"第 {retries_used} 次重试成功" + (f"（上限 {max_retries}）" if max_retries else "")),
        ]
        dedupe = f"alloc_retry_ok:{job_id}"
    else:
        title = "自动排队抢卡已停止"
        lines = [
            ("目标域", _pool_label(pool)),
            ("已重试", f"{retries_used}" + (f"/{max_retries} 次" if max_retries else " 次")),
            ("停止原因", reason or "抢卡未能获得可用 GPU"),
        ]
        dedupe = f"alloc_retry_stop:{job_id}"
    kwargs = dict(
        dedupe_key=dedupe,
        cooldown_seconds=3600,
        project_id=project_id,
        link_url=_site_link("native" if pool == "native" else "jobs"),
        link_label="查看任务状态",
        now=now,
    )
    if session is not None:
        return enqueue_in_tx(session, owner_id, "allocation_retry", title, lines, **kwargs)
    return enqueue(store_or_repo, owner_id, "allocation_retry", title, lines, **kwargs)


def notify_auto_recovery_in_tx(
    session,
    owner_id: str,
    *,
    plan: Mapping[str, Any],
    state: str,
    now: float | None = None,
) -> dict[str, Any] | None:
    plan_id = str(plan.get("plan_id") or plan.get("recovery_group_id") or "")
    pool = _pool_label(plan.get("pool"))
    spec = _spec_text(plan.get("spec_key"))
    round_num = int(plan.get("round_num") or 1)
    max_rounds = int(plan.get("max_rounds") or 3)
    if state == "scheduled":
        title = "检测到实例被 Colab 回收，已启动自动恢复"
        lines: list[tuple[str, Any]] = [
            ("所属域", pool),
            ("原规格", spec),
            ("恢复轮次", f"第 {round_num}/{max_rounds} 轮"),
            ("说明", "系统正在按原账号与原规格申请继任实例"),
        ]
        dedupe = f"recovery_start:{plan_id}:{round_num}"
    elif state == "recovered":
        title = "L3 跨机自动恢复成功"
        ns = plan.get("native_snapshot") if isinstance(plan.get("native_snapshot"), Mapping) else {}
        mr = ns.get("model_restore_state") if isinstance(ns.get("model_restore_state"), Mapping) else {}
        lines = [
            ("所属域", pool),
            ("规格", spec),
            ("继任实例", str(plan.get("successor_instance_id") or "—")[:16]),
        ]
        if mr.get("partial_disk_limit"):
            lines.append(("模型回拉提示", "受本地磁盘空间限制，部分缓存模型未能全部回拉，请按需在页面补拉"))
        elif mr.get("state") == "completed":
            lines.append(("模型回拉", "本地缓存模型已全部自动回拉完成"))
        dedupe = f"recovery_done:{plan_id}"
    else:
        title_map = {
            "exhausted": "L3 自动恢复已耗尽重试",
            "aborted_fatal": "L3 自动恢复因致命错误中止",
            "aborted_vps_outage": "L3 自动恢复因 VPS 长时间离线中止",
        }
        title = title_map.get(state, "L3 自动恢复未能完成")
        lines = [
            ("所属域", pool),
            ("规格", spec),
            ("中止状态", state),
            ("原因", plan.get("last_error_message") or plan.get("last_error_code") or "请检查实例中心与事件日志"),
        ]
        dedupe = f"recovery_fail:{plan_id}:{state}"
    return enqueue_in_tx(
        session,
        owner_id,
        "auto_recovery",
        title,
        lines,
        dedupe_key=dedupe,
        cooldown_seconds=3600,
        link_url=_site_link("compute"),
        link_label="查看实例中心",
        now=now,
    )


def notify_transfer_result(
    store_or_repo,
    owner_id: str,
    *,
    succeeded: bool,
    transfer_id: str,
    instance_id: str | None,
    from_pool: str | None,
    to_pool: str | None,
    reason: str | None = None,
    session=None,
    now: float | None = None,
) -> dict[str, Any] | None:
    title = "GPU 实例跨域转移完成" if succeeded else "GPU 实例跨域转移失败"
    lines: list[tuple[str, Any]] = [
        ("实例", str(instance_id or "—")[:16]),
        ("方向", f"{_pool_label(from_pool)} → {_pool_label(to_pool)}"),
    ]
    if not succeeded and reason:
        lines.append(("原因", reason))
    kwargs = dict(
        dedupe_key=f"transfer:{'ok' if succeeded else 'fail'}:{transfer_id}",
        cooldown_seconds=3600,
        link_url=_site_link("native" if to_pool == "native" and succeeded else "compute"),
        link_label="查看实例状态",
        now=now,
    )
    if session is not None:
        return enqueue_in_tx(session, owner_id, "transfer_and_restart", title, lines, **kwargs)
    return enqueue(store_or_repo, owner_id, "transfer_and_restart", title, lines, **kwargs)


def notify_native_restart_result(
    store_or_repo,
    owner_id: str,
    *,
    succeeded: bool,
    operation_id: str,
    instance_id: str | None,
    error: str | None = None,
    uncertain: bool = False,
    now: float | None = None,
) -> dict[str, Any] | None:
    title = "原生 ComfyUI 同机重启完成" if succeeded else "原生 ComfyUI 同机重启失败"
    lines: list[tuple[str, Any]] = [
        ("实例", str(instance_id or "—")[:16]),
    ]
    if succeeded:
        lines.append(("状态", "模型索引与 ComfyUI 服务均已恢复健康"))
    else:
        lines.append(("原因", error or "重启超时或校验未通过"))
        if uncertain:
            lines.append(("提示", "远端进程状态尚未确定，系统正在后台自动核对结果"))
    return enqueue(
        store_or_repo,
        owner_id,
        "transfer_and_restart",
        title,
        lines,
        dedupe_key=f"native_restart:{'ok' if succeeded else 'fail'}:{operation_id}",
        cooldown_seconds=3600,
        link_url=_site_link("native"),
        link_label="查看原生 ComfyUI",
        now=now,
    )


def notify_model_library_result(
    store_or_repo,
    owner_id: str,
    *,
    succeeded: bool,
    task_id: str,
    filename: str | None,
    model_type: str | None = None,
    size_bytes: int | None = None,
    source: str | None = None,
    error: str | None = None,
    now: float | None = None,
) -> dict[str, Any] | None:
    title = "模型库文件准备完成" if succeeded else "模型库文件准备失败"
    lines: list[tuple[str, Any]] = [
        ("文件", filename or task_id[:16]),
    ]
    if model_type:
        lines.append(("类别", model_type))
    if source:
        lines.append(("来源", source))
    if succeeded:
        if size_bytes is not None:
            lines.append(("大小", _format_bytes(size_bytes)))
        lines.append(("状态", "已完成校验并同步至 Google Drive，可供使用"))
    else:
        lines.append(("失败原因", error or "下载、校验或上传 Drive 失败"))
    return enqueue(
        store_or_repo,
        owner_id,
        "model_library",
        title,
        lines,
        dedupe_key=f"model_transfer:{'ok' if succeeded else 'fail'}:{task_id}",
        cooldown_seconds=3600,
        link_url=_site_link("models"),
        link_label="打开模型库",
        now=now,
    )


def notify_batch_summary_in_tx(
    session,
    owner_id: str,
    *,
    batch_id: str,
    batch_state: str,
    total: int,
    counts: Mapping[str, int],
    project_id: str | None = None,
    workflow_key: str | None = None,
    elapsed_seconds: float | None = None,
    now: float | None = None,
) -> dict[str, Any] | None:
    state_label = {
        "completed": "全部完成",
        "partial": "部分完成（含失败/取消）",
        "failed": "全部失败",
        "cancelled": "已取消",
    }.get(batch_state, batch_state)
    lines: list[tuple[str, Any]] = [
        ("批次结果", f"{state_label}（共 {total} 项）"),
        ("统计", f"成功 {counts.get('completed', 0)} · 失败 {counts.get('failed', 0)} · 取消 {counts.get('cancelled', 0)}"),
    ]
    if workflow_key:
        lines.append(("工作流", workflow_key))
    if elapsed_seconds is not None:
        lines.append(("总耗时", _format_duration(elapsed_seconds)))
    return enqueue_in_tx(
        session,
        owner_id,
        "batch_summary",
        "创作批次已完成",
        lines,
        dedupe_key=f"batch_done:{batch_id}",
        cooldown_seconds=86400,
        project_id=project_id,
        link_url=_site_link("gallery" if counts.get("completed", 0) > 0 else "jobs"),
        link_label="前往图库查看" if counts.get("completed", 0) > 0 else "查看任务详情",
        now=now,
    )


def notify_single_job_result(
    store_or_repo,
    owner_id: str,
    *,
    job: Mapping[str, Any],
    state: str,
    now: float | None = None,
) -> dict[str, Any] | None:
    if state not in {"completed", "failed"}:
        return None
    if job.get("batch_id") or job.get("warmup") or job.get("is_transfer_job"):
        return None
    snapshot = job.get("snapshot") if isinstance(job.get("snapshot"), Mapping) else {}
    if snapshot.get("native_comfy"):
        return None
    is_creation_api = bool(snapshot.get("creation_api"))
    if state == "completed" and is_creation_api:
        cfg = load_config(store_or_repo, owner_id)
        if not cfg.get("include_creation_api_success"):
            return None

    workflow = str(job.get("workflow_key") or job.get("model_id") or "创作任务")
    spec = _spec_text(job.get("runtime_spec_reported") or job.get("requested_spec"))
    started = job.get("started_epoch")
    elapsed = (float(time.time() if now is None else now) - float(started)) if isinstance(started, (int, float)) else None
    if state == "completed":
        outputs = job.get("output_asset_ids") or job.get("asset_ids") or []
        title = "创作任务已完成"
        lines: list[tuple[str, Any]] = [
            ("工作流", workflow),
            ("显卡", spec),
            ("产出数", f"{len(outputs)} 项" if isinstance(outputs, list) else "已完成"),
        ]
        if elapsed is not None:
            lines.append(("耗时", _format_duration(elapsed)))
        link_path, link_label = "gallery", "前往图库查看作品"
    else:
        title = "创作任务失败"
        lines = [
            ("工作流", workflow),
            ("显卡", spec),
            ("错误码", job.get("error_code") or "gpu_execution_failed"),
            ("原因", job.get("error_message") or "任务未能完成"),
        ]
        link_path, link_label = "jobs", "查看任务日志"
    return enqueue(
        store_or_repo,
        owner_id,
        "single_job",
        title,
        lines,
        dedupe_key=f"job_done:{job.get('id')}:{state}",
        cooldown_seconds=86400,
        project_id=job.get("project_id"),
        link_url=_site_link(link_path),
        link_label=link_label,
        now=now,
    )


def notify_instance_bill_in_tx(
    session,
    owner_id: str,
    *,
    instance_id: str,
    account_id: str | None,
    gpu_label: str | None,
    session_kind: str | None,
    duration_minutes: float | None,
    actual_cu: float | None,
    estimated_cu: float | None,
    remaining_balance: float | None,
    now: float | None = None,
) -> dict[str, Any] | None:
    kind_label = {
        "native_comfy": "原生 ComfyUI",
        "single_job": "单任务实例",
        "smart_reuse": "复用实例",
        "multi_domain": "跨域复用实例",
        "prep": "环境准备",
        "smoke": "冒烟测试",
    }.get(str(session_kind or ""), str(session_kind or "GPU 会话"))
    lines: list[tuple[str, Any]] = [
        ("实例", f"{instance_id[:12]}（{kind_label}）"),
        ("规格", gpu_label or "—"),
    ]
    if duration_minutes is not None:
        lines.append(("运行时长", _format_duration(float(duration_minutes) * 60.0)))
    cu_parts = []
    if actual_cu is not None:
        cu_parts.append(f"实耗 {actual_cu:.3f} CU")
    if estimated_cu is not None:
        cu_parts.append(f"估算 {estimated_cu:.3f} CU")
    if cu_parts:
        lines.append(("本次消耗", " / ".join(cu_parts)))
    if remaining_balance is not None:
        lines.append(("账号剩余", f"{remaining_balance:g} CU"))
    return enqueue_in_tx(
        session,
        owner_id,
        "instance_expiry_billing",
        "GPU 实例已释放并完成 CU 结算",
        lines,
        dedupe_key=f"instance_bill:{instance_id}",
        cooldown_seconds=86400,
        link_url=_site_link("compute"),
        link_label="查看 CU 账单明细",
        now=now,
    )


# ---------------------------------------------------------------------------
# Periodic checks for P2 (expiry warnings + daily CU & Drive quota report)
# ---------------------------------------------------------------------------

def _check_expiry_warnings(store, owner_id: str, cfg: Mapping[str, Any], ts: float) -> int:
    if not cfg["events"].get("instance_expiry_billing"):
        return 0
    from .gpu.repository import GpuInstanceRow

    warn_seconds = float(cfg.get("expiry_warn_minutes", 15)) * 60.0
    sent = 0
    with store.db_lock(), store.Session.begin() as session:
        store.begin_immediate(session)
        rows = session.scalars(
            select(GpuInstanceRow).where(
                GpuInstanceRow.owner_id == owner_id,
                GpuInstanceRow.state.in_(("ready", "busy", "retained_idle")),
            )
        ).all()
        for inst in rows:
            data = inst.data or {}
            if data.get("provider") != "colab":
                continue
            if data.get("manual_stop_requested") or data.get("native_draining") or data.get("force_stop"):
                continue
            pool = _pool_label(data.get("instance_pool") or ("native" if data.get("native_comfy") else "creation"))
            spec = _spec_text(data.get("runtime_spec_reported") or data.get("runtime_spec_requested"))
            checks: list[tuple[str, str, float | None]] = [
                ("hard", "会话上限截止", data.get("hard_deadline_at")),
                ("idle", "空闲自动回收", data.get("idle_deadline_at")),
            ]
            for kind, label, raw_dl in checks:
                if not isinstance(raw_dl, (int, float)) or isinstance(raw_dl, bool):
                    continue
                remaining = float(raw_dl) - ts
                if 30.0 < remaining <= warn_seconds:
                    bucket = int(float(raw_dl) // 300)
                    res = enqueue_in_tx(
                        session,
                        owner_id,
                        "instance_expiry_billing",
                        f"GPU 实例即将到达{label}",
                        [
                            ("实例", f"{inst.id[:12]}（{pool} · {spec}）"),
                            ("剩余时间", f"约 {max(1, int(round(remaining / 60.0)))} 分钟"),
                            ("到期时间", _format_local_time(float(raw_dl))),
                        ],
                        dedupe_key=f"expiry_warn:{inst.id}:{kind}:{bucket}",
                        cooldown_seconds=3600,
                        project_id=inst.project_id,
                        link_url=_site_link("native" if data.get("native_comfy") else "compute"),
                        link_label="查看或续期实例",
                        now=ts,
                    )
                    if res is not None:
                        sent += 1
                    break  # Don't double-warn same instance for both deadlines simultaneously.
    return sent


def _check_daily_report_and_drive(store, owner_id: str, cfg: Mapping[str, Any], ts: float) -> int:
    if not cfg["events"].get("daily_report"):
        return 0
    from .gpu import ccu
    from .gpu.repository import GpuAccountRow, GpuInstanceRow, GpuJobRow, Repository

    tz = ZoneInfo(settings.studio_timezone)
    local_now = datetime.fromtimestamp(ts, tz)
    today_str = local_now.date().isoformat()
    target_hour = int(cfg.get("daily_report_hour", 23))
    queued = 0

    if local_now.hour >= target_hour:
        with store.db_lock(), store.Session.begin() as session:
            store.begin_immediate(session)
            dedupe = f"daily_report:{today_str}"
            if not _should_skip_dedupe_in_tx(session, owner_id, dedupe, 86400, datetime.fromtimestamp(ts, timezone.utc)):
                today_cu = ccu.daily_spend(session, owner_id, today_str)
                accts = session.scalars(select(GpuAccountRow).where(GpuAccountRow.owner_id == owner_id)).all()
                acct_lines: list[str] = []
                for acct in accts:
                    adata = acct.data or {}
                    key = adata.get("account_id") or acct.id
                    alias = adata.get("alias") or adata.get("label") or key[:10]
                    snap = ccu.setting(session, ccu._snapshot_key(owner_id, key))
                    bal = snap.get("balance")
                    if bal is not None:
                        acct_lines.append(f"{alias}: {bal:g} CU")
                start_of_day = datetime.combine(local_now.date(), datetime.min.time(), tzinfo=tz).astimezone(timezone.utc)
                jobs_today = session.scalars(
                    select(GpuJobRow).where(
                        GpuJobRow.owner_id == owner_id,
                        GpuJobRow.created_at >= start_of_day,
                    )
                ).all()
                ok_jobs = sum(1 for j in jobs_today if j.state == "completed")
                fail_jobs = sum(1 for j in jobs_today if j.state == "failed")
                active_insts = session.scalars(
                    select(GpuInstanceRow).where(
                        GpuInstanceRow.owner_id == owner_id,
                        GpuInstanceRow.state.in_(("ready", "busy", "retained_idle", "quarantined")),
                    )
                ).all()
                lines: list[tuple[str, Any]] = [
                    ("日期", today_str),
                    ("今日实耗 CU", f"{today_cu:.3f} CU"),
                    ("今日任务", f"完成 {ok_jobs} · 失败 {fail_jobs} · 总计 {len(jobs_today)}"),
                    ("当前活跃实例", f"{len(active_insts)} 台"),
                ]
                if acct_lines:
                    lines.append(("各账号余额", " / ".join(acct_lines)))
                res = enqueue_in_tx(
                    session,
                    owner_id,
                    "daily_report",
                    f"每日 CU 消费简报（{today_str}）",
                    lines,
                    dedupe_key=dedupe,
                    cooldown_seconds=86400,
                    link_url=_site_link("compute"),
                    link_label="查看 CU 统计页",
                    now=ts,
                )
                if res is not None:
                    queued += 1

    # Check Drive storage quota using cached usage (never spam Drive API every minute).
    try:
        from .gpu import drive_usage

        repo = Repository(store)
        usage_data = drive_usage.get_drive_usage(repo, owner_id, refresh=False)
        quota = usage_data.get("quota") if isinstance(usage_data, Mapping) else None
        limit_b = quota.get("limit") if isinstance(quota, Mapping) else None
        used_b = quota.get("usage") if isinstance(quota, Mapping) else None
        if isinstance(limit_b, int) and limit_b > 0 and isinstance(used_b, int) and used_b >= 0:
            free_b = max(0, limit_b - used_b)
            free_pct = (free_b / limit_b) * 100.0
            threshold_pct = float(cfg.get("drive_low_quota_percent", 10))
            if free_pct <= threshold_pct or free_b < DRIVE_LOW_QUOTA_MIN_FREE_BYTES:
                res = enqueue(
                    store,
                    owner_id,
                    "daily_report",
                    "Google Drive 剩余空间预警",
                    [
                        ("已用 / 总计", f"{_format_bytes(used_b)} / {_format_bytes(limit_b)}"),
                        ("剩余空间", f"{_format_bytes(free_b)}（{free_pct:.1f}%）"),
                        ("建议", "空间不足可能导致模型上传或原生输出持久化失败，请及时清理"),
                    ],
                    dedupe_key=f"drive_low_quota:{today_str}",
                    cooldown_seconds=21600,
                    link_url=_site_link("models"),
                    link_label="查看模型与 Drive 存储",
                    now=ts,
                )
                if res is not None:
                    queued += 1
    except Exception:  # noqa: BLE001 - best-effort check only
        pass
    return queued


def periodic_tick(store_or_repo, *, now: float | None = None) -> dict[str, int]:
    """Run P2 periodic notification checks (expiry warnings + daily CU/Drive report) and retention cleanup."""
    store = _base_store(store_or_repo)
    ts = float(time.time() if now is None else now)
    summary = {"expiry_warnings": 0, "daily_reports": 0, "pruned": 0}
    with store.db_lock(), store.Session() as session:
        rows = session.scalars(
            select(SettingRow).where(SettingRow.id.like(f"{SETTINGS_PREFIX}%"))
        ).all()
        owners = [(str(row.owner_id), _merge_config(row.data)) for row in rows if row.owner_id]
    for owner_id, cfg in owners:
        if not cfg.get("enabled") or not cfg.get("encrypted_bot_token") or not cfg.get("chat_id"):
            continue
        try:
            summary["expiry_warnings"] += _check_expiry_warnings(store, owner_id, cfg, ts)
        except Exception as exc:  # noqa: BLE001
            log.warning("telegram expiry check failed owner=%s: %s", owner_id, type(exc).__name__)
        try:
            summary["daily_reports"] += _check_daily_report_and_drive(store, owner_id, cfg, ts)
        except Exception as exc:  # noqa: BLE001
            log.warning("telegram daily report check failed owner=%s: %s", owner_id, type(exc).__name__)
    summary["pruned"] = prune_old_deliveries(store, now=ts)
    return summary
