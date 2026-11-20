from app.db import Store
from app.gpu.compute import MockColabProvider, MockGPUExecutor
from app.gpu.repository import Repository
from app.gpu.scheduler import Scheduler


def test_real_instance_service_uses_latest_job_policy_after_manual_retention(tmp_path):
    repo = Repository(Store('sqlite:///' + str(tmp_path / 'flow.db')))
    now = [1000.0]
    provider, executor = MockColabProvider(), MockGPUExecutor()
    scheduler = Scheduler(repo, provider_factory=lambda _: provider, executor_factory=lambda _: executor, clock=lambda: now[0])
    def create(key, instance_id=None, *, lifecycle='smart', idle_minutes=2):
        policy = {'provider': 'mock', 'lifecycle': lifecycle, 'retain_idle_minutes': idle_minutes}
        if instance_id:
            policy.update(instance_id=instance_id, allocation='existing')
        return repo.create_job('owner-1', {
            'provider': 'mock', 'project_id': 'project-1',
            'snapshot': {'manifest': {'mock': True}},
            'execution_policy': policy, 'estimated_units': 0,
        }, key)[0]
    def finish(job):
        for _ in range(50):
            scheduler.tick()
            job = repo.get('jobs', job['id'], 'owner-1')
            if job['state'] in {'completed', 'failed', 'cancelled'}:
                break
        assert job['state'] == 'completed', job
        return job
    first = finish(create('first'))
    assert first.get('instance_id')
    instance = repo.get('instances', first['instance_id'], 'owner-1')
    assert instance['state'] == 'retained_idle', instance
    lease = first['lease_id']
    assert instance['retention_mode'] == 'smart'
    manual = scheduler.instances.retention('owner-1', instance['id'], 'retain', 5, 'manual-retain')
    assert manual['retention_mode'] == 'retain'
    assert repo.get('leases', lease, 'owner-1')['idle_deadline_at'] == 1300
    now[0] = 1121
    scheduler.watchdog()
    assert repo.get('instances', instance['id'], 'owner-1')['state'] == 'retained_idle'
    second = finish(create('second', instance['id'], idle_minutes=1))
    assert second['instance_id'] == first['instance_id']
    assert second['lease_id'] == lease
    assert second['retention_mode'] == 'smart'
    assert repo.get('leases', lease, 'owner-1')['idle_deadline_at'] == 1181

    third = finish(create('third', instance['id'], lifecycle='retain'))
    assert third['lease_id'] == lease
    assert third['retention_mode'] == 'retain'
    assert len(repo.list('instances', 'owner-1')) == 1
    now[0] = 1241
    scheduler.watchdog()
    assert repo.get('instances', instance['id'], 'owner-1')['state'] == 'deleted'
