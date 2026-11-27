"""Tests for Telegram notification settings, outbox, deduplication, dispatcher, and P0/P1/P2 hooks."""

from __future__ import annotations

import json

import httpx
import pytest
from app import telegram_notify
from app.config import settings as app_settings
from app.db import Store, TelegramNotificationRow
from app.gpu import allocation_retry, ccu, recovery
from app.gpu.cu_guard import CuGuardService
from app.gpu.repository import GpuAccountRow, GpuInstanceRow, GpuInstanceUsageSegmentRow, Repository
from app.main import app
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient


@pytest.fixture
def tg_env(monkeypatch, tmp_path):
    monkeypatch.setattr(app_settings, "encryption_key", Fernet.generate_key().decode())
    monkeypatch.setattr(app_settings, "session_secret", "test-session-secret-0123456789")
    monkeypatch.setattr(app_settings, "studio_password", "test-pass")
    monkeypatch.setattr(app_settings, "site_url", "https://studio.example.com")
    monkeypatch.setattr(app_settings, "studio_timezone", "Asia/Shanghai")
    db_path = tmp_path / "tg-test.db"
    store = Store(f"sqlite:///{db_path}")
    repo = Repository(store)
    return store, repo


def _enable_tg(store: Store, owner: str = "owner", *, events: dict[str, bool] | None = None, **extra):
    body = {
        "enabled": True,
        "bot_token": "123456789:AABBCCDDEEFFGGHHIIJJKKLLMMNNOOPP",
        "chat_id": "987654321",
        "events": events or {},
        **extra,
    }
    return telegram_notify.save_config(store, owner, body)


def _mock_tg_client(captured: list[dict], *, status_code: int = 200, response_json: dict | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured.append({"url": str(request.url), "body": body})
        payload = response_json if response_json is not None else {"ok": True, "result": {"message_id": 42 + len(captured)}}
        return httpx.Response(status_code, json=payload)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_default_event_flags_p0_p1_true_p2_false(tg_env):
    store, _ = tg_env
    cfg = telegram_notify.get_public_config(store, "owner")
    assert cfg["enabled"] is False
    assert cfg["bot_token_configured"] is False
    for item in cfg["catalog"]:
        key = item["key"]
        if item["priority"] in {"P0", "P1"}:
            assert item["default"] is True
            assert cfg["events"][key] is True
        else:
            assert item["priority"] == "P2"
            assert item["default"] is False
            assert cfg["events"][key] is False


def test_settings_encryption_and_validation(tg_env):
    store, _ = tg_env
    with pytest.raises(ValueError, match="Bot Token"):
        telegram_notify.save_config(store, "owner", {"enabled": True, "chat_id": "987654321"})

    with pytest.raises(ValueError, match="Bot Token 格式无效"):
        telegram_notify.save_config(store, "owner", {"enabled": True, "bot_token": "bad", "chat_id": "987654321"})

    saved = _enable_tg(store, events={"batch_summary": True})
    assert saved["enabled"] is True
    assert saved["bot_token_configured"] is True
    assert saved["bot_token_hint"] == "123456789:****OOPP"
    assert "bot_token" not in saved
    assert "encrypted_bot_token" not in saved
    assert saved["events"]["cu_fuse"] is True
    assert saved["events"]["batch_summary"] is True
    assert saved["events"]["daily_report"] is False

    # Preserves token when omitted on subsequent update
    updated = telegram_notify.save_config(store, "owner", {"enabled": True, "chat_id": "-1001234567890"})
    assert updated["bot_token_configured"] is True
    assert updated["chat_id"] == "-1001234567890"


def test_outbox_dedupe_dispatch_retry_and_redaction(tg_env):
    store, _ = tg_env
    _enable_tg(store)
    t0 = 1_700_000_000.0

    # Enqueue with sensitive text -> must be redacted
    r1 = telegram_notify.enqueue(
        store,
        "owner",
        "cu_fuse",
        "测试熔断",
        [("详情", "url=https://colab.test?access_token=SECRET123&foo=bar Bearer SECRET456 bot123456:ABCDEF1234567890")],
        dedupe_key="fuse:1",
        cooldown_seconds=1800,
        now=t0,
    )
    assert r1 is not None
    assert "SECRET123" not in r1["text"]
    assert "SECRET456" not in r1["text"]
    assert "ABCDEF1234567890" not in r1["text"]

    # Same dedupe key within cooldown -> skipped
    r2 = telegram_notify.enqueue(
        store, "owner", "cu_fuse", "重复熔断", [("详情", "dup")], dedupe_key="fuse:1", cooldown_seconds=1800, now=t0 + 60
    )
    assert r2 is None

    # P2 event disabled by default -> skipped
    r_p2 = telegram_notify.enqueue(store, "owner", "batch_summary", "批次完成", [("结果", "ok")], now=t0)
    assert r_p2 is None

    # Dispatch with transient 429 retry_after -> rescheduled
    captured_fail: list[dict] = []
    with _mock_tg_client(
        captured_fail,
        status_code=429,
        response_json={"ok": False, "description": "Too Many Requests", "parameters": {"retry_after": 25}},
    ) as client:
        st1 = telegram_notify.dispatch_pending(store, client=client, now=t0 + 1)
    assert st1["retried"] == 1
    assert len(captured_fail) == 1

    # Not yet due at t0 + 10
    captured_ok: list[dict] = []
    with _mock_tg_client(captured_ok) as client:
        st2 = telegram_notify.dispatch_pending(store, client=client, now=t0 + 10)
    assert st2["claimed"] == 0

    # Due at t0 + 30 -> delivered
    with _mock_tg_client(captured_ok) as client:
        st3 = telegram_notify.dispatch_pending(store, client=client, now=t0 + 30)
    assert st3["sent"] == 1
    assert len(captured_ok) == 1
    assert captured_ok[0]["body"]["chat_id"] == "987654321"
    assert captured_ok[0]["body"]["parse_mode"] == "HTML"

    # Still within cooldown after send
    r3 = telegram_notify.enqueue(
        store, "owner", "cu_fuse", "再次熔断", [("详情", "dup2")], dedupe_key="fuse:1", cooldown_seconds=1800, now=t0 + 600
    )
    assert r3 is None
    # After cooldown expires -> allowed
    r4 = telegram_notify.enqueue(
        store, "owner", "cu_fuse", "冷却后熔断", [("详情", "new")], dedupe_key="fuse:1", cooldown_seconds=1800, now=t0 + 1900
    )
    assert r4 is not None


def test_p0_cu_fuse_and_sample_failed_hooks(tg_env):
    store, repo = tg_env
    _enable_tg(store)
    now = 1_700_000_000.0
    repo.create("accounts", "owner", {"account_id": "acc1", "alias": "主账号", "provider": "colab", "connected": True}, record_id="acc1")
    with ccu.transaction(repo, "owner") as session:
        ccu.setting(session, ccu._snapshot_key("owner", "acc1"), {"balance": 5.0, "sampled_at": now, "error": None}, "owner")

    guard = CuGuardService(repo, clock=lambda: now)
    guard.save("owner", {"daily_limit_cu": 20.0, "min_balance_cu": 10.0})
    guard.enforce("owner")

    deliveries = telegram_notify.list_recent_deliveries(store, "owner")
    assert any(d["event_key"] == "cu_fuse" and "硬熔断" in (d["title"] or "") for d in deliveries)

    # Consecutive 3 CU balance sampling failures -> triggers P0 cu_fuse sample failure alert
    for i in range(3):
        ccu.sample_and_record(repo, "owner", "acc1", "tick", sampler=lambda *_: None, now=now + 60 * (i + 1))
    deliveries = telegram_notify.list_recent_deliveries(store, "owner")
    assert any(d["event_key"] == "cu_fuse" and "连续采样失败" in (d["title"] or "") for d in deliveries)


def test_p1_recovery_and_allocation_retry_hooks(tg_env):
    store, repo = tg_env
    _enable_tg(store)
    now = 1_700_000_000.0

    # L3 recovery start + recovered
    plan = {
        "plan_id": "plan-1",
        "recovery_group_id": "rec-abc",
        "pool": "native",
        "spec_key": {"accelerator": "L4", "shape": "SHAPE_STANDARD"},
        "state": "scheduled",
        "round_num": 1,
        "max_rounds": 3,
        "created_at": now,
        "updated_at": now,
    }
    with store.db_lock(), store.Session.begin() as session:
        recovery.save_plan_in_tx(session, "owner", plan)
    plan["state"] = "recovered"
    plan["successor_instance_id"] = "inst-new-123"
    plan["updated_at"] = now + 60
    with store.db_lock(), store.Session.begin() as session:
        recovery.save_plan_in_tx(session, "owner", plan)

    deliveries = telegram_notify.list_recent_deliveries(store, "owner")
    titles = [d["title"] for d in deliveries if d["event_key"] == "auto_recovery"]
    assert any("启动自动恢复" in (t or "") for t in titles)
    assert any("自动恢复成功" in (t or "") for t in titles)

    # Allocation retry succeeded after 2 retries
    job = repo.create(
        "jobs",
        "owner",
        {
            "provider": "colab",
            "account_id": "acc1",
            "account_alias_snapshot": "主账号",
            "allocation_retry": {
                "enabled": True,
                "active": True,
                "phase": "allocating",
                "attempt": 3,
                "retries_used": 2,
                "max_retries": 20,
                "current_spec": {"accelerator": "A100", "shape": "SHAPE_HIGHMEM"},
            },
        },
        record_id="job-retry-1",
    )
    allocation_retry.mark_succeeded(repo, job, now=now + 120)
    deliveries = telegram_notify.list_recent_deliveries(store, "owner")
    assert any(d["event_key"] == "allocation_retry" and "抢卡成功" in (d["title"] or "") for d in deliveries)


def test_p2_batch_expiry_billing_and_daily_report(tg_env):
    store, repo = tg_env
    now = 1_700_060_400.0  # 2023-11-15 23:00:00+08:00 (Asia/Shanghai)
    _enable_tg(
        store,
        events={
            "batch_summary": True,
            "instance_expiry_billing": True,
            "daily_report": True,
        },
        expiry_warn_minutes=15,
        daily_report_hour=23,
    )

    # 1. Batch summary on terminal transition
    batch, _ = repo.create_batch(
        "owner",
        [
            {"project_id": "p1", "workflow_key": "wf/test", "workflow_version": 1, "execution_policy": {"provider": "mock"}},
            {"project_id": "p1", "workflow_key": "wf/test", "workflow_version": 1, "execution_policy": {"provider": "mock"}},
        ],
        "batch-idem-1",
        {"mode": "batch"},
    )
    for jid in batch["job_ids"]:
        repo.transition_job(jid, "owner", "completed", "done", {}, now)

    # 2. Instance nearing hard deadline + closed usage segment bill
    with store.db_lock(), store.Session.begin() as session:
        session.add(
            GpuInstanceRow(
                id="inst-exp-1",
                owner_id="owner",
                project_id="p1",
                account_id="acc1",
                state="retained_idle",
                data={
                    "provider": "colab",
                    "instance_pool": "native",
                    "native_comfy": True,
                    "runtime_spec_reported": {"accelerator": "L4", "shape": "SHAPE_STANDARD"},
                    "hard_deadline_at": now + 600,
                },
            )
        )
        seg = GpuInstanceUsageSegmentRow(
            id="seg-1",
            owner_id="owner",
            project_id="p1",
            instance_id="inst-closed-1",
            account_id="acc1",
            state="reconciling",
            data={
                "session_kind": "native_comfy",
                "started_at": now - 1800,
                "ended_at": now - 180,
                "reconcile_until": now - 60,
                "gpu_label": "L4",
                "unit_rate": 0.03,
                "actual_cu": 0.912,
                "estimated_cu": 0.81,
                "ccu_after": 42.5,
                "jobs": [],
            },
        )
        session.add(seg)
        session.add(GpuAccountRow(id="acc1", owner_id="owner", state="connected", data={"account_id": "acc1", "alias": "主账号"}))
        session.flush()
        ccu._seal(session, [seg], now)

    periodic_res = telegram_notify.periodic_tick(store, now=now)
    assert periodic_res["expiry_warnings"] == 1
    assert periodic_res["daily_reports"] == 1

    deliveries = telegram_notify.list_recent_deliveries(store, "owner", limit=20)
    by_key = {d["event_key"]: [x["title"] for x in deliveries if x["event_key"] == d["event_key"]] for d in deliveries}
    assert "batch_summary" in by_key
    assert any("即将到达" in (t or "") for t in by_key.get("instance_expiry_billing", []))
    assert any("CU 结算" in (t or "") for t in by_key.get("instance_expiry_billing", []))
    assert any("每日 CU 消费简报" in (t or "") for t in by_key.get("daily_report", []))


def test_telegram_api_routes_and_test_endpoint(tg_env, monkeypatch):
    store, _ = tg_env
    app.state.store = store
    captured: list[dict] = []

    def fake_send_http(**kwargs):
        captured.append(kwargs)
        return True, {"message_id": 777}

    monkeypatch.setattr(telegram_notify, "_send_telegram_http", fake_send_http)

    client = TestClient(app, base_url="https://studio.example.com")
    login = client.post("/api/auth/login", json={"password": "test-pass"})
    assert login.status_code == 200
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

    res_get = client.get("/api/settings/telegram", headers=headers)
    assert res_get.status_code == 200
    data = res_get.json()["data"]
    assert data["enabled"] is False
    assert data["events"]["cu_fuse"] is True
    assert data["events"]["batch_summary"] is False

    res_put = client.put(
        "/api/settings/telegram",
        headers=headers,
        json={
            "enabled": True,
            "bot_token": "123456789:AABBCCDDEEFFGGHHIIJJKKLLMMNNOOPP",
            "chat_id": "987654321",
            "events": {**data["events"], "batch_summary": True},
        },
    )
    assert res_put.status_code == 200
    saved = res_put.json()["data"]
    assert saved["enabled"] is True
    assert saved["bot_token_configured"] is True
    assert saved["events"]["batch_summary"] is True

    res_test = client.post("/api/settings/telegram/test", headers=headers, json={})
    assert res_test.status_code == 200
    assert res_test.json()["data"]["message_id"] == 777
    assert len(captured) == 1

    res_logs = client.get("/api/settings/telegram/deliveries", headers=headers)
    assert res_logs.status_code == 200
    items = res_logs.json()["data"]["items"]
    assert len(items) == 1
    assert items[0]["state"] == "sent"

    # Confirm table exists in Store
    with store.db_lock(), store.Session() as session:
        rows = session.query(TelegramNotificationRow).all()
        assert len(rows) == 1


def test_single_job_creation_api_filter_and_model_library_helpers(tg_env):
    store, _ = tg_env
    now = 1_700_000_000.0
    _enable_tg(store, events={"single_job": True}, include_creation_api_success=False)

    # Creation API success is filtered by default even when single_job is enabled
    res_skip = telegram_notify.notify_single_job_result(
        store,
        "owner",
        job={"id": "j-api-ok", "workflow_key": "wf/api", "snapshot": {"creation_api": True}},
        state="completed",
        now=now,
    )
    assert res_skip is None

    # Creation API failure is STILL sent when single_job is enabled
    res_fail = telegram_notify.notify_single_job_result(
        store,
        "owner",
        job={"id": "j-api-fail", "workflow_key": "wf/api", "snapshot": {"creation_api": True}, "error_code": "OOM", "error_message": "CUDA OOM"},
        state="failed",
        now=now,
    )
    assert res_fail is not None

    # Regular studio creation job success IS sent
    res_ok = telegram_notify.notify_single_job_result(
        store,
        "owner",
        job={"id": "j-web-ok", "workflow_key": "wf/krea", "output_asset_ids": ["a1", "a2"], "started_epoch": now - 45},
        state="completed",
        now=now,
    )
    assert res_ok is not None

    # Model library P1 notification works out of the box
    res_model = telegram_notify.notify_model_library_result(
        store,
        "owner",
        succeeded=True,
        task_id="mdl-task-1",
        filename="flux.safetensors",
        model_type="checkpoints",
        size_bytes=2 * 1024**3,
        source="huggingface",
        now=now,
    )
    assert res_model is not None
    assert "2.00 GB" in res_model["text"]

