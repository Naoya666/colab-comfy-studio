"""CU reconciliation: balance sampling, attribution and idle-runtime detection."""

import pytest
from app.db import Store
from app.gpu import ccu
from app.gpu.repository import Repository

OWNER = "owner-1"


def make_repo(tmp_path, name="ccu.db"):
    return Repository(Store("sqlite:///" + str(tmp_path / name)))


def test_normalize_subscription_supports_tunnel_and_rest_shapes():
    tunnel = {
        "currentBalance": 2500.0,
        "consumptionRateHourly": 1.57,
        "assignmentsCount": 0,
        "eligibleGpus": ["T4", "L4"],
    }
    rest = {
        "subscription": {
            "paidComputeUnitsBalance": "12.5",
            "consumptionRateHourly": 0.5,
            "assignmentsCount": 2,
        }
    }
    assert ccu.normalize_subscription(tunnel) == {"balance": 2500.0, "rate": 1.57, "assignments": 0}
    assert ccu.normalize_subscription(rest) == {"balance": 12.5, "rate": 0.5, "assignments": 2}
    assert ccu.normalize_subscription({}) == {"balance": None, "rate": None, "assignments": None}


def sample(repo, balance, at, account="acc-1", source="tick", **fields):
    return ccu.sample_and_record(repo, OWNER, account, source,
                                 sampler=lambda *_: {"balance": balance, **fields}, now=at)


def open_session(repo, ident, kind, start, *, account="acc-1", accelerator="L4"):
    repo.create("instances", OWNER, {
        "provider": "colab", "account_id": account, "state": "busy", "lease_id": f"lease-{ident}",
    }, record_id=ident)
    policy = {"lifecycle": "smart" if kind == "smart_reuse" else "save"}
    snapshot = {"native_comfy": kind == "native_comfy", "system_smoke": kind == "smoke"}
    job = repo.create("jobs", OWNER, {
        "provider": "colab", "account_id": account, "instance_id": ident, "state": "running",
        "execution_policy": policy, "snapshot": snapshot, "prep": kind == "prep",
    }, record_id=f"job-{ident}")
    lease = repo.create("leases", OWNER, {
        "provider": "colab", "account_id": account, "instance_id": ident, "state": "active",
        "owner_job_id": job["id"], "handle": {"runtime_name": f"runtime-{ident}"},
        "selected_spec": {"accelerator": accelerator, "variant": "VARIANT_CPU" if accelerator == "NONE" else "VARIANT_GPU"},
    }, record_id=f"lease-{ident}")
    ccu.open_instance_session(repo, job, lease, start)
    return job


def test_owner_deltas_topup_midnight_and_failed_sample(tmp_path, monkeypatch):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from app.config import settings
    from app.gpu.cu_guard import CuGuardService

    repo = make_repo(tmp_path)
    monkeypatch.setattr(settings, "studio_timezone", "Asia/Shanghai")
    midnight = datetime(2026, 11, 21, tzinfo=ZoneInfo("Asia/Shanghai")).timestamp()
    repo.create("accounts", OWNER, {"enabled": True}, record_id="acc-1")
    guard = CuGuardService(repo, lambda: midnight - 1)
    guard.save(OWNER, {"daily_limit_cu": 2, "min_balance_cu": 10})
    sample(repo, 12, midnight - 60)
    sample(repo, 10, midnight - 1)
    assert guard.state(OWNER)["status"] == "hard_balance_floor"
    sample(repo, None, midnight - 0.5)
    assert guard.state(OWNER)["status"] == "soft_daily_limit"
    assert ccu.snapshots(repo, OWNER)["acc-1"]["error"] == "sample_failed"
    guard = CuGuardService(repo, lambda: midnight)
    assert guard.state(OWNER)["status"] == "ok"
    sample(repo, 40, midnight)
    sample(repo, 39, midnight + 1)
    sample(repo, 39, midnight + 2)
    rows = sorted(repo.list("ccu_ledger", OWNER), key=lambda row: row["sampled_at"])
    assert [r["state"] for r in rows] == ["baseline", "spend", "topup", "spend"]
    assert [r["delta_spent"] for r in rows] == [0, 2, 0, 1]
    assert [r["local_date"] for r in rows] == ["2026-11-20"] * 2 + ["2026-11-21"] * 2
    assert guard.state(OWNER)["today_actual_cu"] == 1
    assert ccu.usage_summary(repo, OWNER)["history_actual_cu"] == 3
    assert repo.list("ccu_ledger", "other-owner") == []


@pytest.mark.parametrize("kind", ["single_job", "native_comfy", "prep", "smoke"])
def test_session_attributes_live_and_delayed_charges_then_seals(tmp_path, monkeypatch, kind):
    repo = make_repo(tmp_path)
    repo.create("accounts", OWNER, {"enabled": True}, record_id="acc-1")
    job = open_session(repo, "first", kind, 1010)
    sample(repo, 20, 1000, source="alloc_before")
    sample(repo, 19.8, 1070, rate_hourly=1.5, assignments=1)
    assert repo.get("jobs", job["id"], OWNER)["actual_units"] == pytest.approx(0.2)
    row = repo.list("instance_usage_segments", OWNER)[0]
    assert row["estimated_cu"] == pytest.approx(0.025)
    monkeypatch.setattr(ccu, "sampling_enabled", lambda: True)
    monkeypatch.setattr(ccu, "sample_account_balance", lambda *_: {"balance": 19.6, "at": 1101})
    ccu.close_instance_session(repo, OWNER, "first", 1100, "confirmed_delete")
    row = repo.list("instance_usage_segments", OWNER)[0]
    assert row["state"] == "reconciling"
    assert row["session_kind"] == kind
    assert row["ccu_after"] == 19.6
    assert row["actual_cu"] == pytest.approx(0.4)
    sample(repo, 19.1, 1160)
    ccu.tick_ccu(repo, OWNER, now=1220, sampler=lambda *_: {"balance": 18.9})
    row = repo.list("instance_usage_segments", OWNER)[0]
    assert row["state"] == "closed" and row["actual_cu"] == pytest.approx(0.9)
    assert repo.get("jobs", job["id"], OWNER)["actual_units"] == pytest.approx(0.9)
    assert ccu.usage_summary(repo, OWNER)["history_actual_cu"] == pytest.approx(1.1)


def test_reuse_job_and_idle_attribution_and_new_instance_seals_previous(tmp_path):
    repo = make_repo(tmp_path)
    job = open_session(repo, "first", "smart_reuse", 1000)
    repo.update("jobs", job["id"], {"state": "completed", "execution_seconds": 100}, OWNER)
    repo.create("jobs", OWNER, {"instance_id": "first", "state": "completed", "execution_seconds": 200}, record_id="second-job")
    sample(repo, 20, 1000)
    sample(repo, 16, 1400)
    ccu.close_instance_session(repo, OWNER, "first", 1400, "watchdog")
    sample(repo, 15, 1460, source="alloc_before")
    open_session(repo, "next", "single_job", 1461)
    sample(repo, 14, 1470)
    rows = {row["instance_id"]: row for row in repo.list("instance_usage_segments", OWNER)}
    first = rows["first"]
    assert first["state"] == "closed" and first["actual_cu"] == 5
    assert [row["actual_cu"] for row in first["jobs"]] == [1.25, 2.5]
    assert first["idle_cu"] == 1.25
    assert rows["next"]["actual_cu"] == 1
    assert ccu.sessions(repo, OWNER, date=ccu.local_date(1000), kind="smart_reuse", limit=1)[1] == 1


def test_multiple_online_instances_share_debit_by_spec_rate(tmp_path):
    repo = make_repo(tmp_path)
    open_session(repo, "gpu", "single_job", 1000)
    open_session(repo, "cpu", "prep", 1000, accelerator="NONE")
    sample(repo, 20, 1000)
    sample(repo, 18.29, 1060, assignments=2, rate_hourly=100)
    rows = {row["instance_id"]: row for row in repo.list("instance_usage_segments", OWNER)}
    assert rows["gpu"]["actual_cu"] == pytest.approx(1.64)
    assert rows["cpu"]["actual_cu"] == pytest.approx(0.07)
    assert all(row["attribution"] == "prorated" for row in rows.values())


def test_tick_ccu_flags_dedupes_and_resolves_idle_runtime(tmp_path, monkeypatch):
    repo = make_repo(tmp_path)
    repo.create(
        "accounts",
        OWNER,
        {"account_id": "acc-1", "alias": "算力号", "enabled": True, "provider": "colab"},
        record_id="acc-1",
    )
    samples = {"acc-1": {"balance": 50.0, "rate": 1.0, "assignments": 1, "at": 10.0}}
    sampler = lambda *args, **kwargs: samples["acc-1"]  # noqa: E731

    first = ccu.tick_ccu(repo, OWNER, sampler=sampler, now=1000.0)
    anomaly = first["anomalies"][0]
    assert anomaly["account_id"] == "acc-1"
    assert anomaly["assignments"] == 1
    assert anomaly["first_seen"] == 1000.0
    assert anomaly["resolved_at"] is None
    assert first["snapshots"]["acc-1"]["balance"] == 50.0
    assert first["snapshots"]["acc-1"]["active_jobs"] == 0

    second = ccu.tick_ccu(repo, OWNER, sampler=sampler, now=1060.0)
    assert second["anomalies"][0]["first_seen"] == 1000.0
    assert second["anomalies"][0]["last_seen"] == 1060.0

    samples["acc-1"] = {"balance": 50.0, "rate": 0.0, "assignments": 0, "at": 20.0}
    third = ccu.tick_ccu(repo, OWNER, sampler=sampler, now=1120.0)
    assert third["anomalies"][0]["resolved_at"] == 1120.0


def test_tick_ccu_does_not_flag_when_a_job_owns_the_assignment(tmp_path, monkeypatch):
    repo = make_repo(tmp_path)
    repo.create(
        "accounts",
        OWNER,
        {"account_id": "acc-1", "alias": "算力号", "enabled": True, "provider": "colab"},
        record_id="acc-1",
    )
    repo.create(
        "jobs",
        OWNER,
        {"id": "job-live", "account_id": "acc-1", "provider": "colab", "state": "running"},
        record_id="job-live",
    )
    state = ccu.tick_ccu(
        repo,
        OWNER,
        sampler=lambda *args, **kwargs: {"balance": 40.0, "rate": 1.0, "assignments": 1, "at": 30.0},
        now=2000.0,
    )
    assert state["anomalies"] == []
    assert state["snapshots"]["acc-1"]["active_jobs"] == 1


def test_usage_reports_measured_versus_estimated(tmp_path, monkeypatch):
    from app import main

    repo = make_repo(tmp_path)
    cutover = repo.store.get("settings", "cu-ledger-cutover")["epoch"]
    now = cutover - 1
    repo.create(
        "usage",
        OWNER,
        {
            "job_id": "job-measured",
            "provider": "colab",
            "units": 0.26,
            "units_estimate": 0.26,
            "units_actual": 0.175,
            "created_epoch": now,
        },
        record_id="usage-measured",
    )
    repo.create(
        "usage",
        OWNER,
        {
            "job_id": "job-unmeasured",
            "provider": "colab",
            "units": 0.1,
            "units_estimate": 0.1,
            "units_actual": None,
            "created_epoch": now,
        },
        record_id="usage-unmeasured",
    )

    repo.create("usage", OWNER, {"provider": "colab", "units_actual": 1.0, "created_epoch": cutover + 1})
    result = main.usage(repo.store, OWNER)
    reconciliation = result["reconciliation"]
    assert reconciliation["measured_jobs"] == 1
    assert reconciliation["unmeasured_jobs"] == 1
    assert reconciliation["measured_actual_units"] == pytest.approx(0.175)
    assert reconciliation["variance_units"] == pytest.approx(0.175 - 0.26)
    assert result["actual_units"] == pytest.approx(0.175)
    assert result["today_actual_cu"] == 0
    assert result["daily_history"][0]["legacy_actual_cu"] == pytest.approx(0.175)
    assert sum(day["actual_cu"] for day in result["daily_history"]) == result["history_actual_cu"]


def test_manual_balance_queries_owner_accounts_and_preserves_zero(tmp_path, monkeypatch):
    from app.common import current_user, get_store
    from app.main import app
    from fastapi.testclient import TestClient

    repo = make_repo(tmp_path)
    for account_id, owner_id, fields in [
        ("zero", OWNER, {"allow_allocation": False}),
        ("paid", OWNER, {}),
        ("unavailable", OWNER, {}),
        ("disabled", OWNER, {"enabled": False}),
        ("mock", OWNER, {"provider": "mock"}),
        ("other-owner", "other", {}),
    ]:
        repo.create("accounts", owner_id, {"account_id": account_id, **fields}, record_id=account_id)

    monkeypatch.setattr(ccu, "sampling_enabled", lambda: True)

    def sample(_repo, owner_id, account_id):
        assert owner_id == OWNER
        return {
            "zero": {"balance": 0, "at": 10},
            "paid": {"balance": 13.25, "at": 11},
            "unavailable": None,
        }[account_id]

    monkeypatch.setattr(ccu, "sample_account_balance", sample)
    monkeypatch.setitem(app.dependency_overrides, current_user, lambda: {"id": OWNER})
    monkeypatch.setitem(app.dependency_overrides, get_store, lambda: repo.store)

    response = TestClient(app).get("/api/gpu/compute/cu-balance")
    assert response.status_code == 200
    balances = response.json()["data"]
    assert balances["total_balance"] == 13.25
    assert set(balances["snapshots"]) == {"zero", "paid", "unavailable"}
    assert balances["snapshots"]["zero"]["balance"] == 0
    assert balances["snapshots"]["unavailable"]["balance"] is None
    assert balances["snapshots"]["unavailable"]["error"] == "sample_failed"
    assert {row["source"] for row in repo.list("ccu_ledger", OWNER)} == {"manual"}


def test_cu_guard_http_validates_and_persists_owner_limits(tmp_path, monkeypatch):
    from app.common import current_user, get_store
    from app.main import app
    from fastapi.testclient import TestClient

    repo = make_repo(tmp_path)
    monkeypatch.setenv("GPU_DAILY_UNITS", "12.5")
    user = {"id": OWNER, "role": "admin"}
    monkeypatch.setitem(app.dependency_overrides, current_user, lambda: user)
    monkeypatch.setitem(app.dependency_overrides, get_store, lambda: repo.store)
    client = TestClient(app)
    path = "/api/gpu/cu/guard"
    assert client.get(path).json()["data"]["daily_limit_cu"] == 12.5
    monkeypatch.setenv("GPU_DAILY_UNITS", "4")
    assert client.get(path).json()["data"]["daily_limit_cu"] == 12.5
    for value in [-1, "2", True, 10**500]:
        assert client.put(path, json={"daily_limit_cu": value}).status_code == 422
    assert client.put(path, json={"daily_limit_cu": 0, "min_balance_cu": 2}).json()["data"]["status"] == "soft_daily_limit"
    reopened = make_repo(tmp_path)
    monkeypatch.setitem(app.dependency_overrides, get_store, lambda: reopened.store)
    usage = client.get("/api/gpu/usage").json()["data"]
    assert usage["guard"]["daily_limit_cu"] == 0
    assert usage["guard"]["min_balance_cu"] == 2
    user["role"] = "user"
    assert client.put(path, json={"daily_limit_cu": None}).status_code == 403
    user["role"] = "admin"
    assert client.put(path, json={"daily_limit_cu": None, "min_balance_cu": None}).json()["data"]["status"] == "ok"


def test_cu_sessions_and_daily_pagination_and_date_range(tmp_path, monkeypatch):
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    from app.common import current_user, get_store
    from app.config import settings
    from app.main import app
    from fastapi.testclient import TestClient

    repo = make_repo(tmp_path)
    monkeypatch.setattr(settings, "studio_timezone", "Asia/Shanghai")
    monkeypatch.setattr(ccu, "sampling_enabled", lambda: False)
    repo.create("accounts", OWNER, {"enabled": True}, record_id="acc-1")

    tz = ZoneInfo("Asia/Shanghai")
    base_day = datetime(2026, 11, 1, 10, 0, tzinfo=tz)
    balance = 100.0
    sample(repo, balance, (base_day - timedelta(days=1)).timestamp())

    # Create 12 distinct days, each with 1 closed session (alternating kind) and 1.0 CU debit.
    for idx in range(12):
        ts = (base_day + timedelta(days=idx)).timestamp()
        kind = "native_comfy" if idx % 2 == 0 else "single_job"
        open_session(repo, f"inst-{idx:02d}", kind, ts)
        balance = round(balance - 1.0, 6)
        sample(repo, balance, ts + 60)
        ccu.close_instance_session(repo, OWNER, f"inst-{idx:02d}", ts + 120, "confirmed_delete")

    monkeypatch.setitem(app.dependency_overrides, current_user, lambda: {"id": OWNER})
    monkeypatch.setitem(app.dependency_overrides, get_store, lambda: repo.store)
    client = TestClient(app)

    # /api/gpu/usage defaults recent_sessions to 10 items while reporting full count & actual total.
    usage = client.get("/api/gpu/usage").json()["data"]
    assert len(usage["recent_sessions"]) == 10
    assert usage["recent_sessions_count"] == 12
    assert usage["recent_sessions_actual_cu"] == pytest.approx(12.0)

    # /api/gpu/cu/sessions defaults to limit=10 and supports offset pagination.
    p1 = client.get("/api/gpu/cu/sessions").json()["data"]
    assert p1["limit"] == 10 and p1["offset"] == 0
    assert len(p1["sessions"]) == 10 and p1["total"] == 12
    assert p1["total_actual_cu"] == pytest.approx(12.0)

    p2 = client.get("/api/gpu/cu/sessions", params={"offset": 10}).json()["data"]
    assert len(p2["sessions"]) == 2 and p2["total"] == 12
    assert {row["instance_id"] for row in p1["sessions"]}.isdisjoint({row["instance_id"] for row in p2["sessions"]})

    # Date range + kind filtering on sessions.
    filtered = client.get(
        "/api/gpu/cu/sessions",
        params={"start_date": "2026-11-03", "end_date": "2026-11-07", "kind": "native_comfy"},
    ).json()["data"]
    assert filtered["total"] == 3  # Nov 3 (idx 2), Nov 5 (idx 4), Nov 7 (idx 6)
    assert filtered["total_actual_cu"] == pytest.approx(3.0)
    assert [row["local_date"] for row in filtered["sessions"]] == ["2026-11-07", "2026-11-05", "2026-11-03"]

    # Invalid date range returns 422 on both endpoints.
    assert client.get(
        "/api/gpu/cu/sessions",
        params={"start_date": "2026-11-10", "end_date": "2026-11-05"},
    ).status_code == 422
    assert client.get(
        "/api/gpu/cu/daily",
        params={"start_date": "2026-11-10", "end_date": "2026-11-05"},
    ).status_code == 422

    # /api/gpu/cu/daily defaults to 10 days per page with full range totals.
    d1 = client.get("/api/gpu/cu/daily").json()["data"]
    assert d1["limit"] == 10 and d1["offset"] == 0
    assert len(d1["days"]) == 10 and d1["total"] == 12
    assert d1["total_actual_cu"] == pytest.approx(12.0)
    assert d1["total_sessions"] == 12
    assert d1["days"][0]["date"] == "2026-11-12"

    d2 = client.get("/api/gpu/cu/daily", params={"offset": 10}).json()["data"]
    assert [day["date"] for day in d2["days"]] == ["2026-11-02", "2026-11-01"]
    assert d2["total"] == 12

    # Date range filtering on daily summary.
    dr = client.get(
        "/api/gpu/cu/daily",
        params={"start_date": "2026-11-04", "end_date": "2026-11-06"},
    ).json()["data"]
    assert [day["date"] for day in dr["days"]] == ["2026-11-06", "2026-11-05", "2026-11-04"]
    assert dr["total"] == 3
    assert dr["total_actual_cu"] == pytest.approx(3.0)
    assert dr["total_sessions"] == 3
