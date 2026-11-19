"""CU reconciliation: balance sampling, attribution and idle-runtime detection."""

import time

import pytest
from app.db import Store
from app.gpu import ccu
from app.gpu.repository import Repository
from app.gpu.scheduler import Scheduler

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


def test_reconcile_never_invents_or_returns_negative():
    assert ccu.reconcile(2500.0, 2499.825) == pytest.approx(0.175)
    assert ccu.reconcile(10, 11) is None
    assert ccu.reconcile(None, 5) is None
    assert ccu.reconcile(5, None) is None


@pytest.mark.parametrize("kind", ["workflow_smoke", "prep"])
def test_allocation_samples_missing_smoke_and_prep_baseline_and_records_delta(tmp_path, monkeypatch, kind):
    repo = make_repo(tmp_path)
    job, _ = repo.create_job(OWNER, {"execution_policy": {"provider": "colab"}}, "baseline")
    job = repo.update("jobs", job["id"], {
        "provider": "colab", "account_id": "acc-1", kind: True, "started_epoch": 120,
    }, OWNER)
    samples = iter([
        {"balance": 100.0, "at": 100.0},
        {"balance": 99.8, "at": 180.0},
    ])
    monkeypatch.setattr(ccu, "maybe_sample_balance", lambda *args: next(samples))
    scheduler = Scheduler(repo, clock=lambda: 180.0)
    job = scheduler._sample_allocation_baseline(job, {"create_attempts": 0})
    assert job["ccu_before"] == 100.0
    assert job["ccu_before_at"] == 100.0
    scheduler._record_usage(job, {"id": "lease-1"}, 180.0)
    measured = repo.get("jobs", job["id"], OWNER)
    assert measured["actual_units"] == pytest.approx(0.2)
    assert measured["ccu_source"] == "google_balance"
    assert repo.list("usage", OWNER)[0]["units_actual"] == pytest.approx(0.2)


@pytest.mark.parametrize("provider,before,attempts", [("mock", None, 0), ("colab", 0, 0), ("colab", None, 1)])
def test_allocation_does_not_resample_or_backfill_after_create(tmp_path, monkeypatch, provider, before, attempts):
    repo = make_repo(tmp_path)
    scheduler = Scheduler(repo)
    def forbidden(*args):
        raise AssertionError("Must not sample an existing baseline, mock, or started allocation")
    monkeypatch.setattr(ccu, "maybe_sample_balance", forbidden)
    job = {"provider": provider, "ccu_before": before}
    assert scheduler._sample_allocation_baseline(job, {"create_attempts": attempts}) is job


def test_allocation_sampling_failure_remains_fail_soft(tmp_path, monkeypatch):
    scheduler = Scheduler(make_repo(tmp_path))
    monkeypatch.setattr(ccu, "maybe_sample_balance", lambda *args: None)
    job = {"provider": "colab", "owner_id": OWNER, "account_id": "acc-1"}
    assert scheduler._sample_allocation_baseline(job, {}) is job


def test_measure_spend_uses_balance_delta_and_flags_contention(tmp_path, monkeypatch):
    repo = make_repo(tmp_path)
    job = {
        "id": "job-1",
        "owner_id": OWNER,
        "account_id": "acc-1",
        "provider": "colab",
        "ccu_before": 100.0,
    }
    monkeypatch.setattr(
        ccu,
        "maybe_sample_balance",
        lambda *args, **kwargs: {"balance": 99.5, "rate": 1.5, "assignments": 1, "at": 123.0},
    )
    scheduler = Scheduler(repo, clock=lambda: 200.0)

    patch = scheduler._measure_spend(job, None)
    assert patch["actual_units"] == pytest.approx(0.5)
    assert patch["ccu_after"] == 99.5
    assert patch["ccu_attribution"] == "exclusive"
    assert patch["ccu_source"] == "google_balance"

    repo.create(
        "leases",
        OWNER,
        {
            "provider": "colab",
            "account_id": "acc-1",
            "runtime_id": "studio-other1",
            "request_id": "11111111-1111-4111-8111-111111111111",
            "owner_job_id": "job-other",
            "state": "active",
        },
        record_id="lease-other",
    )
    contended = scheduler._measure_spend(job, None)
    assert contended["ccu_attribution"] == "contended"


def test_measure_spend_is_fail_soft_without_samples(tmp_path, monkeypatch):
    repo = make_repo(tmp_path)
    job = {"id": "job-2", "owner_id": OWNER, "account_id": "acc-1", "provider": "colab", "ccu_before": 10.0}
    monkeypatch.setattr(ccu, "maybe_sample_balance", lambda *args, **kwargs: None)
    scheduler = Scheduler(repo, clock=lambda: 300.0)

    patch = scheduler._measure_spend(job, None)
    assert "actual_units" not in patch
    assert patch["ccu_source"] == "after_missing"

    job_no_baseline = {"id": "job-3", "owner_id": OWNER, "account_id": "acc-1", "provider": "colab"}
    missing = scheduler._measure_spend(job_no_baseline, None)
    assert missing["ccu_source"] == "baseline_missing"
    assert "actual_units" not in missing


def test_tick_ccu_flags_dedupes_and_resolves_idle_runtime(tmp_path, monkeypatch):
    repo = make_repo(tmp_path)
    monkeypatch.setattr(ccu, "_state_path", lambda: tmp_path / "ccu_state.json")
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
    monkeypatch.setattr(ccu, "_state_path", lambda: tmp_path / "ccu_state.json")
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
    now = time.time()
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
    monkeypatch.setattr(
        ccu, "read_state", lambda: {"snapshots": {}, "anomalies": [], "updated_at": 5.0}
    )

    result = main.usage(repo.store, OWNER)
    reconciliation = result["reconciliation"]
    assert reconciliation["measured_jobs"] == 1
    assert reconciliation["unmeasured_jobs"] == 1
    assert reconciliation["measured_actual_units"] == pytest.approx(0.175)
    assert reconciliation["variance_units"] == pytest.approx(0.175 - 0.26)
    assert result["actual_units"] == pytest.approx(0.175)
    assert "ccu" in result and result["ccu"]["sampled_at"] == 5.0


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
    assert not ccu._state_path().exists()
