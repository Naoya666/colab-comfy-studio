"""Targeted regression tests for pre-launch P0/P1 audit fixes across Batches 1-4."""

from __future__ import annotations

import io
import os
import pickle
import tarfile
import time
import zipfile
from types import SimpleNamespace

import httpx
import pytest
from app import api_access, common, creation_api, native_outputs_routes, native_proxy, security
from app.config import settings
from app.db import Store
from app.gpu import accounts as accounts_mod
from app.gpu import (
    ccu,
    cu_guard,
    instances,
    model_sources,
    native,
    native_models,
    node_persistence,
    recovery,
    specs,
)
from app.gpu.repository import Repository

# ---------------------------------------------------------------------------
# Batch 1: Scheduler, Lifecycle, Multi-Account & CU Fuses
# ---------------------------------------------------------------------------


def test_p0_1_mock_and_auto_alloc_jobs_do_not_create_phantom_legacy_account(tmp_path):
    store = Store(f"sqlite:///{tmp_path / 'p0_1.db'}")
    repo = Repository(store)
    store.create("projects", "owner", {"title": "P"}, record_id="proj-1")

    mock_job, _ = repo.create_job(
        "owner",
        {
            "project_id": "proj-1",
            "provider": "mock",
            "execution_policy": {"provider": "mock", "lifecycle": "save"},
        },
        "idem-mock-1",
    )
    auto_job, _ = repo.create_job(
        "owner",
        {
            "project_id": "proj-1",
            "provider": "colab",
            "execution_policy": {"provider": "colab", "lifecycle": "smart"},
        },
        "idem-auto-1",
    )

    svc = accounts_mod.AccountService(repo, "owner")
    assert svc.list() == []
    assert repo.get("jobs", mock_job["id"], "owner").get("account_id") is None
    assert repo.get("jobs", auto_job["id"], "owner").get("account_id") is None

    # Genuine pre-v0.2 legacy row (no execution_policy, no provider="mock", no account_id) still triggers legacy migration.
    legacy_job = repo.create("jobs", "owner", {"state": "completed", "provider": "colab"}, project_id="proj-1")
    accts = svc.list()
    assert len(accts) == 1
    assert repo.get("accounts", accts[0]["account_id"], "owner").get("legacy_default") is True
    assert repo.get("jobs", legacy_job["id"], "owner")["account_id"] == accts[0]["account_id"]
    # Modern unassigned jobs still NOT hijacked by legacy default account!
    assert repo.get("jobs", mock_job["id"], "owner").get("account_id") is None
    assert repo.get("jobs", auto_job["id"], "owner").get("account_id") is None


def test_p0_2_batch_order_waits_for_earlier_batch_job(tmp_path):
    store = Store(f"sqlite:///{tmp_path / 'p0_2.db'}")
    repo = Repository(store)
    store.create("projects", "owner", {"title": "P"}, record_id="proj-1")
    repo.create(
        "accounts",
        "owner",
        {"account_id": "acc-1", "alias": "A1", "enabled": True, "allow_allocation": True, "connected": True},
        record_id="acc-1",
    )
    policy = {
        "provider": "colab",
        "account_id": "acc-1",
        "compute_spec": {"variant": "VARIANT_GPU", "accelerator": "T4", "shape": "SHAPE_STANDARD"},
        "lifecycle": "batch",
    }
    j1 = repo.create(
        "jobs",
        "owner",
        {
            "state": "running",
            "provider": "colab",
            "account_id": "acc-1",
            "batch_id": "batch-1",
            "batch_index": 0,
            "execution_policy": policy,
            "requested_spec": policy["compute_spec"],
            "snapshot": {"manifest": {"compute_spec": policy["compute_spec"]}},
        },
        project_id="proj-1",
    )
    j2 = repo.create(
        "jobs",
        "owner",
        {
            "state": "waiting_for_capacity",
            "provider": "colab",
            "account_id": "acc-1",
            "batch_id": "batch-1",
            "batch_index": 1,
            "execution_policy": policy,
            "requested_spec": policy["compute_spec"],
            "snapshot": {"manifest": {"compute_spec": policy["compute_spec"]}},
        },
        project_id="proj-1",
    )
    assert j1["id"] != j2["id"]
    svc = instances.InstanceService(repo)
    with pytest.raises(instances.CreationInstancePending):
        svc.allocate(j2)


def test_p0_5_retention_preserves_busy_and_drains_native_on_save(tmp_path):
    store = Store(f"sqlite:///{tmp_path / 'p0_5.db'}")
    repo = Repository(store)
    svc = instances.InstanceService(repo)

    # 1. Ephemeral instance rejects retention updates
    eph = repo.create(
        "instances",
        "owner",
        {"state": "busy", "provider": "colab", "instance_pool": instances.POOL_EPHEMERAL, "retention_mode": "save"},
    )
    with pytest.raises(instances.AllocationError):
        svc.retention("owner", eph["id"], "retain", 30, "k-eph")

    # 2. Busy creation instance keeps state="busy"
    cre = repo.create(
        "instances",
        "owner",
        {
            "state": "busy",
            "provider": "colab",
            "instance_pool": instances.POOL_CREATION,
            "retention_mode": "smart",
            "retain_idle_minutes": 15,
            "current_job_id": "job-busy",
        },
    )
    updated = svc.retention("owner", cre["id"], "retain", 20, "k-cre")
    assert updated["state"] == "busy"
    assert updated["retention_mode"] == "retain"
    assert updated["retain_idle_minutes"] == 20

    # 3. Native instance with mode="save" sets manual_stop_requested & native_draining on both instance & lease
    nat_lease = repo.create("leases", "owner", {"state": "idle", "native_comfy": True, "native_ready": True})
    nat = repo.create(
        "instances",
        "owner",
        {
            "state": "retained_idle",
            "provider": "colab",
            "instance_pool": instances.POOL_NATIVE,
            "native_comfy": True,
            "retention_mode": "retain",
            "lease_id": nat_lease["id"],
        },
    )
    updated_nat = svc.retention("owner", nat["id"], "save", 15, "k-nat")
    assert updated_nat["manual_stop_requested"] is True
    fresh_nat = repo.get("instances", nat["id"], "owner")
    assert fresh_nat["manual_stop_requested"] is True
    assert fresh_nat["native_draining"] is True
    fresh_lease = repo.get("leases", nat_lease["id"], "owner")
    assert fresh_lease["native_draining"] is True


def test_p0_6_p0_7_p0_8_cu_guard_fail_closed_disabled_account_and_native_soft_fuse(tmp_path, monkeypatch):
    store = Store(f"sqlite:///{tmp_path / 'p0_678.db'}")
    repo = Repository(store)
    now = 1_700_000_000.0
    repo.create(
        "accounts",
        "owner",
        {
            "account_id": "acc-1",
            "alias": "Main",
            "enabled": True,
            "allow_allocation": True,
            "connected": True,
            "auth_status": "connected",
        },
        record_id="acc-1",
    )
    guard = cu_guard.CuGuardService(repo, clock=lambda: now)
    guard.save("owner", {"min_balance_cu": 5.0, "daily_limit_cu": 10.0})

    # Seed low balance snapshot (4.0 <= floor 5.0) accompanied by a transient sample_failed error
    with ccu.transaction(repo, "owner") as session:
        ccu.setting(
            session,
            ccu._snapshot_key("owner", "acc-1"),
            {"balance": 4.0, "tier": "PRO", "error": "sample_failed", "sampled_at": now},
            "owner",
        )

    with store.db_lock(), store.Session.begin() as session:
        state_fail_closed = guard.state_in(session, "owner", fail_closed_on_sample_error=True)
    assert any(r["account_id"] == "acc-1" for r in state_fail_closed["hard_tripped_accounts"])
    with pytest.raises(common.APIError) as exc:
        guard.check("owner", "colab", "acc-1")
    assert exc.value.code == "CU_BALANCE_FLOOR_REACHED"

    # P0-7: Disabled account with an active instance stays monitored
    repo.update("accounts", "acc-1", {"enabled": False}, "owner")
    lease = repo.create(
        "leases",
        "owner",
        {"state": "active", "provider": "colab", "account_id": "acc-1", "native_comfy": True, "native_ready": False},
    )
    inst = repo.create(
        "instances",
        "owner",
        {
            "state": "allocating",
            "provider": "colab",
            "account_id": "acc-1",
            "instance_pool": instances.POOL_NATIVE,
            "native_comfy": True,
            "lease_id": lease["id"],
            "cu_fuse": "soft_daily_limit",  # Already set on instance, missing on lease
        },
    )
    monitored = ccu.monitored_accounts(repo, "owner")
    assert "acc-1" in monitored

    # P0-8: Soft fuse syncs native_draining + cu_fuse to booting native instance & lease even if inst.cu_fuse matched
    monkeypatch.setattr(
        guard,
        "state_in",
        lambda session, owner, **kw: {
            "enabled": True,
            "status": "soft_daily_limit",
            "soft_tripped": True,
            "daily_message": "daily cap",
            "hard_tripped_accounts": [],
        },
    )
    guard.enforce("owner")
    assert repo.get("instances", inst["id"], "owner")["native_draining"] is True
    fresh_lease = repo.get("leases", lease["id"], "owner")
    assert fresh_lease["native_draining"] is True
    assert fresh_lease["cu_fuse"] == "soft_daily_limit"


def test_p1_batch1_specs_handoff_and_account_delete(tmp_path):
    # Unknown spec fails closed as below workflow default
    assert specs.is_below_workflow_default(
        {"compute_spec": {"variant": "VARIANT_GPU", "accelerator": "L4", "shape": "SHAPE_STANDARD"}},
        {"variant": "VARIANT_GPU", "accelerator": "UNKNOWN_GPU", "shape": "SHAPE_STANDARD"},
    )

    store = Store(f"sqlite:///{tmp_path / 'p1_b1.db'}")
    repo = Repository(store)
    repo.create(
        "accounts",
        "owner",
        {"account_id": "acc-del", "alias": "Del", "enabled": True, "connected": True},
        record_id="acc-del",
    )
    repo.store.create("settings", "owner", {"policy": {"account_id": "acc-del", "mode": "smart"}}, record_id="creation-api:owner")
    repo.store.create("settings", "owner", {"policy": {"account_id": "acc-del"}}, record_id="native-api:owner")
    accounts_mod.AccountService(repo, "owner").delete("acc-del")
    assert repo.store.get("settings", "creation-api:owner", "owner")["policy"]["account_id"] is None
    assert repo.store.get("settings", "native-api:owner", "owner")["policy"]["account_id"] is None


# ---------------------------------------------------------------------------
# Batch 2: Auth, Local Trust, Native Allowlist, GZip & Tar/Weight Security
# ---------------------------------------------------------------------------


def test_p0_13_p0_14_local_trust_hardening(monkeypatch):
    def make_scope(peer_ip: str, headers: list[tuple[bytes, bytes]]):
        return {"type": "http", "client": (peer_ip, 12345), "headers": headers}

    # Duplicate X-Forwarded-For with loopback + public IP must be REJECTED
    scope_dup = make_scope("127.0.0.1", [(b"x-forwarded-for", b"127.0.0.1"), (b"x-forwarded-for", b"203.0.113.9")])
    assert api_access.is_trusted_local_client(scope_dup, {}) is False

    # RFC 7239 Forwarded header with non-loopback must be REJECTED
    scope_rfc = make_scope("127.0.0.1", [(b"forwarded", b"for=203.0.113.9;proto=https")])
    assert api_access.is_trusted_local_client(scope_rfc, {}) is False

    # Spoofed testclient/localhost in XFF must be REJECTED
    assert api_access.is_trusted_local_client(make_scope("127.0.0.1", [(b"x-forwarded-for", b"testclient")]), {}) is False
    assert api_access.is_trusted_local_client(make_scope("127.0.0.1", [(b"x-forwarded-for", b"localhost")]), {}) is False

    # Container bridge NAT IP rejected by default unless explicitly opted in
    monkeypatch.setattr(api_access, "_in_container", lambda: True)
    monkeypatch.delenv("STUDIO_TRUST_DOCKER_BRIDGE_LOOPBACK", raising=False)
    bridge_scope = make_scope("172.18.0.1", [(b"host", b"127.0.0.1:8189")])
    assert api_access.is_trusted_local_client(bridge_scope, {}) is False
    monkeypatch.setenv("STUDIO_TRUST_DOCKER_BRIDGE_LOOPBACK", "1")
    assert api_access.is_trusted_local_client(bridge_scope, {}) is True


def test_p0_15_studio_session_rejects_foreign_purpose_tokens(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "session_secret", "regression-test-session-secret-0123456789")
    now = int(time.time())
    native_sess = security.sign_session({"uid": "owner", "purpose": "native-session", "iat": now, "exp": now + 300})
    handoff = security.sign_session({"uid": "owner", "purpose": "native-handoff", "aud": "https://comfy.test", "iat": now, "exp": now + 60})
    studio_tok = security.sign_session({"uid": "owner", "purpose": "studio-session", "iat": now, "exp": now + 300})

    store = Store(f"sqlite:///{tmp_path / 'p0_15.db'}")
    store.create("users", "owner", {"username": "owner", "role": "admin"}, record_id="owner")

    def req_base(tok):
        return SimpleNamespace(
            method="GET",
            cookies={"studio_session": tok},
            headers={},
            app=SimpleNamespace(state=SimpleNamespace(store=store)),
        )

    with pytest.raises(common.APIError) as e1:
        common.current_user(req_base(native_sess))
    assert e1.value.status == 401
    with pytest.raises(common.APIError) as e2:
        common.current_user(req_base(handoff))
    assert e2.value.status == 401
    assert common.current_user(req_base(studio_tok))["id"] == "owner"


def test_p0_16_p1_27_native_weak_allowlist_and_dot_segment():
    # Allowed standard inference / queue / history / view / upload / WS endpoints
    for method, path, is_ws in [
        ("POST", "/prompt", False),
        ("POST", "/api/prompt", False),
        ("GET", "/queue", False),
        ("POST", "/queue", False),
        ("POST", "/interrupt", False),
        ("GET", "/history", False),
        ("GET", "/history/prompt-123", False),
        ("GET", "/view", False),
        ("POST", "/upload/image", False),
        ("POST", "/upload/mask", False),
        ("GET", "/ws", True),
        ("GET", "/system_stats", False),
        ("GET", "/object_info", False),
        ("GET", "/object_info/KSampler", False),
        ("GET", "/models/checkpoints", False),
    ]:
        assert api_access.is_native_weak_allowed(path, path.encode(), method=method, websocket=is_ws), f"should allow {method} {path}"

    # Blocked management / userdata / install / internal / dot-segment paths
    for method, path in [
        ("POST", "/model/install"),
        ("POST", "/snapshot/restore"),
        ("POST", "/userdata/evil.json"),
        ("GET", "/userdata/comfy.settings.json"),
        ("POST", "/free"),
        ("GET", "/manager/reboot"),
        ("POST", "/v2/manager/reboot"),
        ("GET", "/studio_native/state"),
        ("GET", "/foo/../studio_native/state"),
        ("POST", "/foo/../prompt"),
        ("DELETE", "/history/prompt-123"),
        ("GET", "/customnode/install"),
    ]:
        assert not api_access.is_native_weak_allowed(path, path.encode(), method=method), f"should block {method} {path}"

    # Normalized path blocks dot-segment bypass of /studio_native
    assert native_proxy._normalized_gateway_path(
        {"path": "/foo/../studio_native/state", "raw_path": b"/foo/../studio_native/state"}
    ) == "/studio_native/state"


def test_p1_30_p1_32_creation_history_and_content_disposition():
    rewritten_item = creation_api._strip_optional_compat_api_prefix(
        {"path": "/api/history/abc-123", "raw_path": b"/api/history/abc-123"}
    )
    assert rewritten_item["path"] == "/history/abc-123"
    rewritten_root = creation_api._strip_optional_compat_api_prefix(
        {"path": "/api/history", "raw_path": b"/api/history"}
    )
    assert rewritten_root["path"] == "/history"
    cd = native_outputs_routes._content_disposition('evil"\r\nX-Injected: 1\\foo.png')
    assert "\r" not in cd and "\n" not in cd and "\\" not in cd


def test_p0_17_p0_18_tar_env_security_and_weight_validation(tmp_path):
    # P0-17: nvidia protected + chained symlink escape blocked
    assert node_persistence._is_protected_site_top_name("nvidia") is True

    archive = tmp_path / "evil_env.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        sym = tarfile.TarInfo("site/escape")
        sym.type = tarfile.SYMTYPE
        sym.linkname = "../../outside"
        tf.addfile(sym)
        payload = b"pwned"
        f = tarfile.TarInfo("site/escape/pwn.txt")
        f.size = len(payload)
        tf.addfile(f, io.BytesIO(payload))

    dest = tmp_path / "dest"
    dest.mkdir()
    with pytest.raises(RuntimeError):
        node_persistence._safe_extract_env_archive(archive, dest)

    # P0-18 (a): duplicate ZIP entry rejected
    dup_zip = tmp_path / "dup.pth"
    safe_pkl = pickle.dumps({"a": 1}, protocol=2)
    with zipfile.ZipFile(dup_zip, "w") as zf:
        zf.writestr("archive/data.pkl", safe_pkl)
        zf.writestr("archive/./data.pkl", safe_pkl)
    with pytest.raises(model_sources.ModelError):
        model_sources.validate_weight_file(dup_zip, "dup.pth", dup_zip.stat().st_size, model_type="upscale_models")

    # P0-18 (b): 5th legacy pickle stream with unsafe GLOBAL opcode rejected
    class Evil:
        def __reduce__(self):
            return (os.system, ("echo pwn",))

    # Craft a minimal valid tensor-rebuilding 4th pickle stream (protocol 2) followed by a malicious 5th stream
    stream4_with_tensor = (
        b"\x80\x02"
        b"ctorch._utils\n_rebuild_tensor_v2\n"
        b"K\x01\x85R."
    )
    legacy = tmp_path / "legacy5.pth"
    buf = io.BytesIO()
    pickle.dump(model_sources.PYTORCH_LEGACY_MAGIC, buf, protocol=2)
    pickle.dump(1001, buf, protocol=2)
    pickle.dump({"little_endian": True}, buf, protocol=2)
    buf.write(stream4_with_tensor)
    pickle.dump(Evil(), buf, protocol=2)  # 5th stream!
    legacy.write_bytes(buf.getvalue())
    with pytest.raises(model_sources.ModelError):
        model_sources.validate_weight_file(legacy, "legacy5.pth", legacy.stat().st_size, model_type="upscale_models")


# ---------------------------------------------------------------------------
# Batch 3: L0-L3 Auto-Recovery & Native Busy Semantics
# ---------------------------------------------------------------------------


def test_p0_9_startup_grace_enforces_vps_outage_immediately(tmp_path, monkeypatch):
    monkeypatch.setenv("GPU_REAL_ENABLED", "1")
    monkeypatch.setenv("GPU_AUTO_RECOVERY_ENABLED", "1")
    monkeypatch.setenv("GPU_AUTO_RECOVERY_STARTUP_GRACE_SECONDS", "30")
    monkeypatch.setenv("GPU_AUTO_RECOVERY_MAX_VPS_OUTAGE_MINUTES", "18")

    store = Store(f"sqlite:///{tmp_path / 'p0_9.db'}")
    repo = Repository(store)
    repo.create(
        "accounts",
        "owner",
        {"account_id": "acc-1", "alias": "A1", "enabled": True, "allow_allocation": True, "connected": True},
        record_id="acc-1",
    )
    now = 1_700_000_000.0
    with store.db_lock(), store.Session.begin() as session:
        eligible, code, _ = recovery.evaluate_recovery_eligibility_in_tx(
            repo,
            session,
            owner_id="owner",
            pool=instances.POOL_NATIVE,
            account_id="acc-1",
            retention_mode="retain",
            hard_deadline_at=now + 3600,
            orig_idle_deadline_at=now + 3600,
            was_busy=False,
            rounds_used=0,
            now=now,
            process_started_at=now - 5.0,  # Inside 30s startup_grace!
            check_vps_outage=True,
            last_successful_probe_at=now - 25 * 60.0,  # 25 min outage!
        )
    assert eligible is False
    assert code == "aborted_vps_outage"


def test_p0_10_p0_11_local_outputs_persisted_and_cleaning_up_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    art_dir = tmp_path / "assets" / "artifacts"
    art_dir.mkdir(parents=True)
    obj_key = "a" * 32
    (art_dir / f"{obj_key}.data").write_bytes(b"png-bytes")

    assert recovery._local_outputs_persisted(
        {"outputs": [{"artifact_id": obj_key, "storage": {"provider": "local", "object_key": obj_key}}]}
    ) is True

    # P0-11: Creation job in cleaning_up / with cleanup_final_state must NOT be taken over
    monkeypatch.setenv("GPU_REAL_ENABLED", "1")
    monkeypatch.setenv("GPU_AUTO_RECOVERY_ENABLED", "1")
    store = Store(f"sqlite:///{tmp_path / 'p0_11.db'}")
    repo = Repository(store)
    repo.create("accounts", "owner", {"account_id": "acc-1", "enabled": True, "allow_allocation": True, "connected": True}, record_id="acc-1")
    job = repo.create(
        "jobs",
        "owner",
        {"state": "cleaning_up", "cleanup_final_state": "completed", "provider": "colab", "account_id": "acc-1"},
    )
    now = 1_700_000_000.0
    snap = {
        "owner_id": "owner",
        "instance_id": "inst-1",
        "pool": instances.POOL_CREATION,
        "account_id": "acc-1",
        "retention_mode": "smart",
        "was_busy": True,
        "current_job_id": job["id"],
        "hard_deadline_at": now + 3600,
        "orig_idle_deadline_at": now + 3600,
    }
    with store.db_lock(), store.Session.begin() as session:
        res = recovery.on_runtime_reclaimed_in_tx(repo, session, snap, now, process_started_at=now - 100)
    assert res is None


def test_p0_12_dead_tunnel_does_not_fabricate_busy_or_extend_idle(tmp_path, monkeypatch):
    monkeypatch.setenv("GPU_REAL_ENABLED", "0")
    store = Store(f"sqlite:///{tmp_path / 'p0_12.db'}")
    repo = Repository(store)
    inst = repo.create("instances", "owner", {"state": "retained_idle", "native_comfy": True})
    lease = repo.create(
        "leases",
        "owner",
        {
            "state": "idle",
            "native_comfy": True,
            "native_ready": True,
            "instance_id": inst["id"],
            "owner_job_id": "job-1",
            "runtime_id": "rt-1",
            "deadline_epoch": 50000.0,
            "native_last_busy": 1000.0,
            "native_last_upstream_busy": False,
        },
    )
    monkeypatch.setattr(native, "backend_for", lambda *a, **kw: ("https://internal.example.test", {"CF-Access-Client-Id": "id"}))
    monkeypatch.setattr(native, "keepalive_lease", lambda *a, **kw: False)

    def offline_handler(req):
        raise httpx.ConnectError("tunnel down", request=req)

    with httpx.Client(transport=httpx.MockTransport(offline_handler)) as c:
        assert native.busy_state(client=c) == ("unknown", "unreachable")
        assert native.busy(client=c) is True

    monkeypatch.setattr(native, "busy_state_for", lambda *a, **kw: ("unknown", "unreachable"))
    assert native.protect_lease(repo, lease, clock=lambda: 20000.0, check_remote=False) is True
    fresh = repo.get("leases", lease["id"], "owner")
    assert fresh["native_last_busy"] == 1000.0
    assert fresh["native_last_upstream_busy"] is False


# ---------------------------------------------------------------------------
# Batch 4: Workspace Lock, Local Models & Plugin Persistence
# ---------------------------------------------------------------------------


def test_p1_9_workspace_offline_status_blocks_releasing_and_deleting_leases(tmp_path):
    store = Store(f"sqlite:///{tmp_path / 'p1_9.db'}")
    repo = Repository(store)
    lease = repo.create("leases", "owner", {"state": "releasing", "native_comfy": True})
    assert native.workspace_offline_status(repo, "owner")[0] is False
    repo.update("leases", lease["id"], {"state": "deleting"}, "owner")
    assert native.workspace_offline_status(repo, "owner")[0] is False
    repo.update("leases", lease["id"], {"state": "deleted"}, "owner")
    assert native.workspace_offline_status(repo, "owner")[0] is True


def test_p1_11_p1_12_local_models_account_preservation_and_query_sanitization(tmp_path):
    # Non-secret query params preserved while tokens/keys stripped
    sanitized = native_models.LocalModels._sanitize_source(
        "https://civitai.com/api/download/models/123?type=Model&format=SafeTensor&token=SECRET_TOK&api_key=XYZ",
        "civitai",
    )
    assert sanitized is not None
    assert "type=Model" in sanitized["download_url"]
    assert "format=SafeTensor" in sanitized["download_url"]
    assert "SECRET_TOK" not in sanitized["download_url"]
    assert "api_key" not in sanitized["download_url"]

    # Multi-account Drive plan preserves explicit library account over native compute drive_account
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    local = native_models.LocalModels(
        tmp_path,
        model_dir,
        {
            "channel": "ch",
            "drive_account": "compute-acc",
            "drive_auth": {"library-acc": {"refresh_token": "r", "client_id": "c", "client_secret": "s"}},
        },
    )
    _, plan = local.plan(
        [{"category": "checkpoints", "name": "a.safetensors", "drive_file_id": "FileId_1234567890", "account": "library-acc"}],
        [],
    )
    assert plan["drive"]["account"] == "library-acc"


def test_p1_15_p1_16_git_bundle_validation_and_malformed_pyproject(tmp_path):
    node_dir = tmp_path / "my-plugin"
    node_dir.mkdir()
    (node_dir / ".studio-repo.pack").write_bytes(b"PACK")
    # Bad commit rejected
    with pytest.raises(RuntimeError):
        node_persistence._restore_git_bundle(node_dir, {"path": "my-plugin", "commit": "--upload-pack=evil"})
    # Unsafe repo URL rejected
    with pytest.raises(RuntimeError):
        node_persistence._restore_git_bundle(
            node_dir,
            {"path": "my-plugin", "commit": "a" * 40, "repo": "--evil-flag"},
        )

    # Malformed pyproject.toml & unborn .git repo do not crash _scan_single_plugin
    nodes = tmp_path / "custom_nodes"
    bad_plugin = nodes / "broken_node"
    bad_plugin.mkdir(parents=True)
    (bad_plugin / "init.py").write_text("x = 1\n")
    (bad_plugin / "pyproject.toml").write_text("this is [not valid toml = =")
    (bad_plugin / ".git").mkdir()
    item = node_persistence._scan_single_plugin(
        nodes, bad_plugin, True, snapshot_databases=False, check_secrets=False, use_disk_cache=False
    )
    assert item is not None
    assert item["path"] == "broken_node"
    assert item["version"] is None
    assert item["commit"] is None


# ---------------------------------------------------------------------------
# Follow-up Audit Fixes (1-4)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_audit_fix1_concurrent_prepare_studio_and_prepare_do_not_deadlock(tmp_path, monkeypatch):
    import asyncio
    import hashlib
    import threading

    model_dir = tmp_path / "models"
    (model_dir / "checkpoints").mkdir(parents=True)
    payload = b"safetensors-header-stub-data"
    digest = hashlib.sha256(payload).hexdigest()

    active_sections = 0
    max_overlap = 0
    overlap_lock = threading.Lock()

    # Direction A: prepare_studio holds exclusive lock first, then prepare() overlaps.
    studio_entered = threading.Event()
    studio_release = threading.Event()
    order_a: list[str] = []

    def fake_restore_weights(*args, **kwargs):
        nonlocal active_sections, max_overlap
        with overlap_lock:
            active_sections += 1
            max_overlap = max(max_overlap, active_sections)
        order_a.append("studio_start")
        studio_entered.set()
        assert studio_release.wait(5), "event loop deadlocked waiting for studio_release"
        order_a.append("studio_end")
        with overlap_lock:
            active_sections -= 1
        return []

    def fake_restore_loras(*args, **kwargs):
        return []

    monkeypatch.setattr(native_models.studio_worker, "_drive_cache", lambda p: None)
    monkeypatch.setattr(native_models.studio_worker, "_lora_drive_cache", lambda p: None)
    monkeypatch.setattr(native_models.studio_worker, "_restore_weights", fake_restore_weights)
    monkeypatch.setattr(native_models.studio_worker, "_restore_user_loras", fake_restore_loras)

    # Local hit file so prepare() runs cleanly after studio finishes.
    ckpt = model_dir / "checkpoints" / "a.safetensors"
    ckpt.write_bytes(payload)
    catalog = [{"model_type": "checkpoints", "filename": "a.safetensors", "sha256": digest, "size": len(payload)}]
    local = native_models.LocalModels(tmp_path, model_dir, {"channel": "ch", "catalog": catalog})

    orig_prepare_one = local.prepare_one

    def traced_prepare_one(target, plan, status):
        nonlocal active_sections, max_overlap
        with overlap_lock:
            active_sections += 1
            max_overlap = max(max_overlap, active_sections)
        order_a.append("native_start")
        try:
            return orig_prepare_one(target, plan, status)
        finally:
            order_a.append("native_end")
            with overlap_lock:
                active_sections -= 1

    monkeypatch.setattr(local, "prepare_one", traced_prepare_one)

    t_studio = asyncio.create_task(
        local.prepare_studio([], [], {"use_drive_cache": False}, lambda rows, require_files=False: {}, {}, 30)
    )
    assert await asyncio.to_thread(studio_entered.wait, 2)

    prompt = {"prompt": {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "a.safetensors"}}}}
    t_native = asyncio.create_task(local.prepare(prompt, [], catalog))
    # Event loop MUST remain responsive while t_native waits for t_studio!
    await asyncio.wait_for(asyncio.sleep(0.05), timeout=2.0)
    assert not t_native.done()

    studio_release.set()
    res_studio, res_native = await asyncio.wait_for(asyncio.gather(t_studio, t_native), timeout=5.0)
    assert res_studio == []
    assert res_native[0]["state"] == "ready"
    assert order_a == ["studio_start", "studio_end", "native_start", "native_end"]
    assert max_overlap == 1

    # Direction B: prepare() holds shared lock first, then prepare_studio() overlaps.
    native_entered = threading.Event()
    native_release = threading.Event()
    order_b: list[str] = []

    def blocking_prepare_one(target, plan, status):
        order_b.append("native_start")
        native_entered.set()
        assert native_release.wait(5), "event loop deadlocked waiting for native_release"
        res = orig_prepare_one(target, plan, status)
        order_b.append("native_end")
        return res

    def fast_restore_weights(*args, **kwargs):
        order_b.append("studio_run")
        return []

    monkeypatch.setattr(local, "prepare_one", blocking_prepare_one)
    monkeypatch.setattr(native_models.studio_worker, "_restore_weights", fast_restore_weights)

    t_native2 = asyncio.create_task(local.prepare(prompt, [], catalog))
    assert await asyncio.to_thread(native_entered.wait, 2)

    t_studio2 = asyncio.create_task(
        local.prepare_studio([], [], {"use_drive_cache": False}, lambda rows, require_files=False: {}, {}, 30)
    )
    await asyncio.wait_for(asyncio.sleep(0.05), timeout=2.0)
    assert not t_studio2.done()

    native_release.set()
    res_native2, res_studio2 = await asyncio.wait_for(asyncio.gather(t_native2, t_studio2), timeout=5.0)
    assert res_native2[0]["state"] == "ready"
    assert res_studio2 == []
    assert order_b == ["native_start", "native_end", "studio_run"]


def test_audit_fix2_cancel_during_ready_probe_transitions_to_cleaning_up(tmp_path):
    from app.gpu.compute import GpuConfig
    from app.gpu.scheduler import Scheduler

    store = Store(f"sqlite:///{tmp_path / 'fix2.db'}")
    repo = Repository(store)
    store.create("projects", "owner", {"title": "P"}, record_id="proj-1")
    now = 1_700_000_000.0
    inst = repo.create(
        "instances",
        "owner",
        {
            "state": "allocating",
            "provider": "colab",
            "account_id": "acc-1",
            "instance_pool": instances.POOL_CREATION,
        },
        project_id="proj-1",
    )
    job = repo.create(
        "jobs",
        "owner",
        {
            "state": "preparing",
            "provider": "colab",
            "account_id": "acc-1",
            "instance_id": inst["id"],
            "deadline_epoch": now + 3600,
            "requested_spec": {"variant": "VARIANT_GPU", "accelerator": "T4", "shape": "SHAPE_STANDARD"},
            "snapshot": {"manifest": {"compute_spec": {"variant": "VARIANT_GPU", "accelerator": "T4", "shape": "SHAPE_STANDARD"}}},
        },
        project_id="proj-1",
    )
    lease = repo.create(
        "leases",
        "owner",
        {
            "state": "active",
            "provider": "colab",
            "account_id": "acc-1",
            "instance_id": inst["id"],
            "owner_job_id": job["id"],
            "handle": {"runtime_name": "runtimes/rt-1", "endpoint": "ep-1"},
            "deadline_epoch": now + 3600,
        },
    )
    repo.update("jobs", job["id"], {"lease_id": lease["id"]}, "owner")

    class CancelDuringPollProvider:
        def inspect(self, handle):
            # Simulate concurrent user cancel arriving while provider.inspect() is in flight.
            repo.update("jobs", job["id"], {"cancel_requested": True}, "owner")
            return {"ready": True, "state": "RUNNING"}

    sched = Scheduler(
        repo,
        provider_factory=lambda _j: CancelDuringPollProvider(),
        config=GpuConfig(enabled=True, real_enabled=True),
        clock=lambda: now,
    )
    after = sched._prepare(repo.get("jobs", job["id"], "owner"))
    assert after["state"] == "cleaning_up"
    assert after["cleanup_final_state"] == "cancelled"
    assert after["requested_outcome"] == "cancelled"


def test_audit_fix3_explicit_unavailable_account_never_falls_back_to_other_drive_account(tmp_path):
    from app.gpu import native_workspace

    store = Store(f"sqlite:///{tmp_path / 'fix3.db'}")
    repo = Repository(store)
    repo.create(
        "accounts",
        "owner",
        {"account_id": "acc-a", "alias": "Account A (No Drive)", "enabled": True, "connected": False},
        record_id="acc-a",
    )
    repo.create(
        "accounts",
        "owner",
        {"account_id": "acc-b", "alias": "Account B (Has Drive)", "enabled": True, "connected": True},
        record_id="acc-b",
    )
    # Only Account B has an active Drive credential.
    repo.create(
        "credentials",
        "owner",
        {
            "provider": "google",
            "purpose": "drive",
            "account_id": "acc-b",
            "revoked": False,
            "encrypted_refresh_token": "enc-tok-b",
        },
    )

    with pytest.raises(common.APIError) as exc_unavail:
        native_workspace.delete_env_caches(repo, "owner", delete_all=True, account_id="acc-a")
    assert exc_unavail.value.code == "ACCOUNT_DRIVE_UNAVAILABLE"
    assert exc_unavail.value.status == 409

    with pytest.raises(common.APIError) as exc_missing:
        native_workspace.delete_env_caches(repo, "owner", delete_all=True, account_id="acc-does-not-exist")
    assert exc_missing.value.code == "ACCOUNT_NOT_FOUND"
    assert exc_missing.value.status == 404

    with pytest.raises(common.APIError) as exc_overview:
        native_workspace.get_workspace_overview(repo, "owner", refresh=True, account_id="acc-a")
    assert exc_overview.value.code == "ACCOUNT_DRIVE_UNAVAILABLE"


def test_audit_fix4_workspace_write_lock_expiry_cannot_resurrect_and_aborts_writes(tmp_path):
    from app.gpu import native_workspace
    from app.gpu.drive import DriveStorage

    store = Store(f"sqlite:///{tmp_path / 'fix4.db'}")
    repo = Repository(store)
    base_now = 1_700_000_000.0

    # Part A: Expired lock cannot be resurrected; active native lease also fails renewal.
    lock_id = native.acquire_workspace_write_lock(repo, "owner", reason="test", ttl_seconds=90.0, now=base_now)
    assert native.renew_workspace_write_lock(repo, "owner", lock_id, ttl_seconds=90.0, now=base_now + 30.0) is True
    assert native.renew_workspace_write_lock(repo, "owner", lock_id, ttl_seconds=90.0, now=base_now + 200.0) is False
    native.release_workspace_write_lock(repo, "owner", lock_id)

    lock_id2 = native.acquire_workspace_write_lock(repo, "owner", reason="test2", ttl_seconds=90.0, now=base_now)
    lease = repo.create("leases", "owner", {"state": "active", "native_comfy": True})
    assert native.renew_workspace_write_lock(repo, "owner", lock_id2, ttl_seconds=90.0, now=base_now + 10.0) is False
    repo.update("leases", lease["id"], {"state": "deleted"}, "owner")
    native.release_workspace_write_lock(repo, "owner", lock_id2)

    # Part B: Losing the lock mid-operation aborts subsequent Drive permanent deletions!
    deleted_ids: list[str] = []

    def drive_handler(req: httpx.Request) -> httpx.Response:
        if req.method == "GET" and req.url.path.endswith("/files"):
            q = req.url.params.get("q", "")
            if "AI-Studio" in q:
                return httpx.Response(200, json={"files": [{"id": "root-1", "name": "AI-Studio", "mimeType": "application/vnd.google-apps.folder"}]})
            if "'root-1' in parents" in q:
                return httpx.Response(200, json={"files": [{"id": "owner-1", "name": "owner", "mimeType": "application/vnd.google-apps.folder"}]})
            if "'owner-1' in parents" in q:
                return httpx.Response(200, json={"files": [{"id": "nc-1", "name": "native-comfy", "mimeType": "application/vnd.google-apps.folder"}]})
            if "'nc-1' in parents" in q:
                return httpx.Response(200, json={"files": [{"id": "ec-1", "name": "env-cache", "mimeType": "application/vnd.google-apps.folder"}]})
            if "'ec-1' in parents" in q:
                return httpx.Response(
                    200,
                    json={
                        "files": [
                            {"id": "f-1", "name": "k1.tar.gz", "size": "100"},
                            {"id": "f-2", "name": "k2.tar.gz", "size": "100"},
                            {"id": "f-3", "name": "k3.tar.gz", "size": "100"},
                        ]
                    },
                )
        if req.method == "DELETE":
            fid = req.url.path.rsplit("/", 1)[-1]
            deleted_ids.append(fid)
            if fid == "f-1":
                # Simulate lock expiry / loss before f-2 is deleted.
                key = native._workspace_lock_key("owner")
                repo.store.update("settings", key, {"expires_at": 1.0}, "owner")
            return httpx.Response(204)
        raise AssertionError(f"Unexpected Drive request: {req.method} {req.url}")

    fake_google = SimpleNamespace(access_token=lambda _purpose: "fake-drive-token")
    with httpx.Client(transport=httpx.MockTransport(drive_handler)) as client:
        drive = DriveStorage({"id": "ws-test", "owner_id": "owner"}, repo, client=client, google=fake_google)
        with pytest.raises(common.APIError) as exc_lost:
            native_workspace.delete_env_caches(repo, "owner", delete_all=True, drive=drive)
    assert exc_lost.value.code == "NATIVE_WORKSPACE_LOCK_LOST"
    assert exc_lost.value.status == 409
    assert deleted_ids == ["f-1"]

