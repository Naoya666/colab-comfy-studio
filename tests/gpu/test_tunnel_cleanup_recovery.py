"""Lost CLI assignment responses must reconcile without creating runtimes."""
import time
import unittest

from app.gpu.colab import GoogleColabProvider, _uuid_to_web_safe_base64
import uuid


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
