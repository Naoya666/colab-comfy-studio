"""Lost CLI assignment responses must reconcile without creating runtimes."""
import time
import unittest
import uuid

import pytest
from app.db import Store
from app.gpu.colab import GoogleColabProvider, _uuid_to_web_safe_base64
from app.gpu.repository import Repository
from app.gpu.scheduler import Scheduler


@pytest.mark.parametrize("confirmed", [True, False])
def test_watchdog_reconciles_stuck_cli_allocation(monkeypatch, confirmed):
    repo = Repository(Store("sqlite://"))
    now = time.time()
    request_id = "11111111-1111-4111-8111-111111111111"
    nbh = _uuid_to_web_safe_base64(uuid.UUID(request_id))
    repo.create("jobs", "owner", {
        "state": "cleaning_up", "provider": "colab", "account_id": "account",
        "instance_id": "instance", "lease_id": "lease", "instance_completion_recorded": True,
        "cleanup_final_state": "failed", "error_code": "gpu_allocation_aborted",
    }, record_id="job")
    repo.create("instances", "owner", {
        "state": "releasing", "provider": "colab", "account_id": "account",
        "lease_id": "lease", "budget_reserved": 8,
    }, record_id="instance")
    repo.create("leases", "owner", {
        "state": "cleanup_pending", "provider": "colab", "account_id": "account",
        "instance_id": "instance", "owner_job_id": "job", "create_attempts": 1,
        "create_started": now - 120, "runtime_id": "studio-test", "request_id": request_id,
        "tunnel_nbh": nbh, "tunnel_assign_params": {"nbh": nbh},
        "handle": {"runtime_name": "runtimes/studio-test", "runtime_id": "studio-test",
                   "request_id": request_id, "owner_id": "owner", "job_id": "job",
                   "owner_job_id": "job", "intent": "gpu-studio"},
    }, record_id="lease")
    provider = GoogleColabProvider(repo, "owner", account_id="account")

    def request(method, path, *, params):
        assert (method, path, params) == ("GET", "/tun/m/assign", {"nbh": nbh})
        return {"token": "fake-xsrf", "nbh": nbh} if confirmed else {}

    def forbidden(*args):
        pytest.fail("an unresolved CLI intent must never cause a provider write")

    monkeypatch.setattr(provider, "_is_cli_tunnel", lambda lease: True)
    monkeypatch.setattr(provider, "_tunnel_request", request)
    monkeypatch.setattr(provider, "_tunnel_json", lambda value: value)
    monkeypatch.setattr(provider, "_tunnel_assignments", lambda: [])
    monkeypatch.setattr(provider, "delete", forbidden)
    scheduler = Scheduler(repo, provider_factory=lambda job: provider, clock=lambda: now)

    try:
        scheduler.watchdog()
    finally:
        provider.close()
    job = repo.get("jobs", "job", "owner")
    lease = repo.get("leases", "lease", "owner")
    instance = repo.get("instances", "instance", "owner")
    assert job["state"] == ("failed" if confirmed else "cleanup_pending")
    assert job["error_code"] == "gpu_allocation_aborted"
    assert lease["state"] == ("deleted" if confirmed else "cleanup_pending")
    assert instance["state"] == ("deleted" if confirmed else "releasing")


@pytest.mark.parametrize("lifecycle", ["smart", "retain"])
def test_creation_unconfirmed_tunnel_allocation_never_retains_idle_and_reconciles_remote(monkeypatch, lifecycle):
    """If POST /tun/m/assign times out on a creation-pool smart/retain job, cleanup must
    never mark the unconfirmed lease/instance retained_idle; it must reconcile and delete."""
    repo = Repository(Store("sqlite://"))
    clock = {"now": 1000.0}
    monkeypatch.setattr("app.gpu.colab.time.time", lambda: clock["now"])
    request_id = "22222222-2222-4222-8222-222222222222"
    nbh = _uuid_to_web_safe_base64(uuid.UUID(request_id))
    repo.create("accounts", "owner", {"account_id": "account", "alias": "acc", "enabled": True, "allow_allocation": True}, record_id="account")
    repo.create("jobs", "owner", {
        "state": "cleaning_up",
        "provider": "colab",
        "account_id": "account",
        "instance_id": "instance",
        "lease_id": "lease",
        "instance_pool": "creation",
        "cleanup_final_state": "failed",
        "error_code": "gpu_allocation_aborted",
        "error_message": "GPU allocation entered cleanup before execution; retry after release",
        "execution_policy": {"provider": "colab", "lifecycle": lifecycle, "retain_idle_minutes": 15},
    }, record_id="job")
    repo.create("instances", "owner", {
        "state": "allocating",
        "provider": "colab",
        "account_id": "account",
        "instance_pool": "creation",
        "current_job_id": "job",
        "lease_id": "lease",
        "retention_mode": lifecycle,
        "retain_idle_minutes": 15,
    }, record_id="instance")
    repo.create("leases", "owner", {
        "state": "cleanup_pending",
        "provider": "colab",
        "account_id": "account",
        "instance_id": "instance",
        "instance_pool": "creation",
        "owner_job_id": "job",
        "create_attempts": 1,
        "create_started": clock["now"] - 14.0,
        "runtime_id": "studio-timedout",
        "request_id": request_id,
        "tunnel_nbh": nbh,
        "tunnel_assign_params": {"nbh": nbh, "variant": "GPU", "accelerator": "L4"},
        "last_create_error": "Colab 隧道网络异常，远端结果可能未知，将保留租约进行协调",
        "handle": {
            "runtime_name": "runtimes/studio-timedout",
            "runtime_id": "studio-timedout",
            "request_id": request_id,
            "owner_id": "owner",
            "job_id": "job",
            "owner_job_id": "job",
            "intent": "gpu-studio",
        },
    }, record_id="lease")

    provider = GoogleColabProvider(repo, "owner", account_id="account")
    deleted_endpoints = []
    remote_assignments = [{"endpoint": "gpu-l4-late-created", "nbh": nbh, "variant": 1, "accelerator": "L4", "machineShape": 0}]

    def fake_request(method, path, *, params=None, headers=None, timeout=None):
        if method == "GET" and path == "/tun/m/assign":
            return {"endpoint": "gpu-l4-late-created", "nbh": nbh}
        raise AssertionError(f"unexpected request {method} {path}")

    def fake_delete(handle):
        saved = repo.get("leases", "lease", "owner")
        if not saved.get("tunnel_endpoint"):
            return False
        deleted_endpoints.append(saved["tunnel_endpoint"])
        remote_assignments.clear()
        return True

    monkeypatch.setattr(provider, "_is_cli_tunnel", lambda lease=None: True)
    monkeypatch.setattr(provider, "_tunnel_request", fake_request)
    monkeypatch.setattr(provider, "_tunnel_json", lambda value: value)
    monkeypatch.setattr(provider, "_tunnel_assignments", lambda: list(remote_assignments))
    monkeypatch.setattr(provider, "delete", fake_delete)

    scheduler = Scheduler(repo, provider_factory=lambda job: provider, clock=lambda: clock["now"])
    try:
        # 1) Right after timeout (<60s): instance must enter releasing (NOT retained_idle) and lease stays cleanup_pending.
        scheduler.tick()
        inst1 = repo.get("instances", "instance", "owner")
        lease1 = repo.get("leases", "lease", "owner")
        assert inst1["state"] == "releasing", f"Unconfirmed allocation must not retain_idle: {inst1}"
        assert lease1["state"] == "cleanup_pending", f"Lease must remain cleanup_pending: {lease1}"

        # 2) After 60s reconciliation window: discovers late remote endpoint and deletes it cleanly.
        clock["now"] += 70.0
        scheduler.tick()
        inst2 = repo.get("instances", "instance", "owner")
        lease2 = repo.get("leases", "lease", "owner")
        job2 = repo.get("jobs", "job", "owner")
        assert deleted_endpoints == ["gpu-l4-late-created"]
        assert lease2["state"] == "deleted"
        assert inst2["state"] == "deleted"
        assert job2["state"] == "failed"
    finally:
        provider.close()


class IntentRecoveryTests(unittest.TestCase):
    def setup_provider(self, response, assignments=None, age=120):
        lease = {
            'id': 'lease', 'request_id': '11111111-1111-4111-8111-111111111111',
            'create_attempts': 1, 'create_started': time.time() - age,
            'selected_spec': {'key': {'variant': 'VARIANT_GPU', 'accelerator': 'H100'}},
        }

        class Store:
            def update(self, kind, rid, patch, owner):
                lease.update(patch)
                return dict(lease)

        provider = GoogleColabProvider(Store(), 'owner')
        provider._owned = lambda handle: dict(lease)
        provider._is_cli_tunnel = lambda saved: True
        calls = []

        def request(method, path, **kwargs):
            calls.append((method, path))
            self.assertEqual(method, 'GET')
            return response

        provider._tunnel_request = request
        provider._tunnel_json = lambda value: value
        provider._tunnel_assignments = lambda: assignments or []
        return provider, lease, calls

    def unassigned(self):
        return {'token': 'fake-xsrf', 'nbh': _uuid_to_web_safe_base64(uuid.UUID('11111111-1111-4111-8111-111111111111'))}

    def test_confirmed_unassigned_intent_finishes_cleanup(self):
        provider, lease, calls = self.setup_provider(self.unassigned())
        self.assertTrue(provider.inspect({'runtime_name': 'runtimes/studio-test'})['deleted'])
        self.assertTrue(provider.delete({}))
        self.assertEqual(calls, [('GET', '/tun/m/assign')])
        self.assertEqual(lease['allocation_reconciliation'], 'exact_nbh_unassigned_and_absent')

    def test_recent_request_remains_unknown(self):
        provider, lease, calls = self.setup_provider(self.unassigned(), age=10)
        self.assertTrue(provider.inspect({'runtime_name': 'runtimes/studio-test'})['allocation_unknown'])
        self.assertFalse(provider.delete({}))
        self.assertFalse(calls)

    def test_empty_malformed_or_wrong_notebook_is_not_release_proof(self):
        for response in ({}, {'token': 'fake'}, {'nbh': 'wrong', 'token': 'fake'}):
            with self.subTest(response=response):
                provider, lease, calls = self.setup_provider(response)
                self.assertFalse(provider.inspect({'runtime_name': 'runtimes/studio-test'})['deleted'])
                self.assertFalse(provider.delete({}))

    def test_existing_assignment_endpoint_is_recovered(self):
        assignment = {'endpoint': 'owned-endpoint', 'nbh': self.unassigned()['nbh']}
        provider, lease, calls = self.setup_provider(assignment, [assignment])
        result = provider.inspect({'runtime_name': 'runtimes/studio-test'})
        self.assertFalse(result['deleted'])
        self.assertEqual(lease['tunnel_endpoint'], 'owned-endpoint')

    def test_list_match_overrides_unassigned_reply(self):
        assignment = {'endpoint': 'owned-endpoint', 'nbh': self.unassigned()['nbh']}
        provider, lease, calls = self.setup_provider(self.unassigned(), [assignment])
        self.assertFalse(provider.inspect({'runtime_name': 'runtimes/studio-test'})['deleted'])
        self.assertEqual(lease['tunnel_endpoint'], 'owned-endpoint')

    def test_failed_status_lookup_preserves_unknown(self):
        provider, lease, calls = self.setup_provider(self.unassigned())
        def fail(*args, **kwargs):
            raise RuntimeError('offline')
        provider._tunnel_request = fail
        with self.assertRaises(RuntimeError):
            provider.inspect({'runtime_name': 'runtimes/studio-test'})
        self.assertNotIn('operation_done', lease)


if __name__ == '__main__':
    unittest.main()
