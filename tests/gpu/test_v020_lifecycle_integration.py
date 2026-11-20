from app.db import Store
from app.gpu.compute import MockColabProvider, MockGPUExecutor
from app.gpu.mcp import MCPGPUExecutor
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


def test_generation_task_and_idle_deadlines_are_independent_on_reuse(tmp_path):
    repo = Repository(Store(f"sqlite:///{tmp_path / 'generation.db'}"))
    now = [1000.0]
    provider, executor = MockColabProvider(), MockGPUExecutor()
    scheduler = Scheduler(repo, provider_factory=lambda _: provider, executor_factory=lambda _: executor,
                          clock=lambda: now[0])
    owner = "owner-1"

    def submit(key, runtime, idle, instance_id=None):
        policy = {"provider": "mock", "lifecycle": "smart", "retain_idle_minutes": idle,
                  "max_runtime_minutes": runtime}
        if instance_id:
            policy.update(allocation="existing", instance_id=instance_id)
        return repo.create_job(owner, {"execution_policy": policy}, key, admitted={
            "execution_policy": policy, "snapshot": {"manifest": {"mock": True}},
            "generation_timeouts": True, "max_runtime_minutes": runtime, "project_id": "project-1",
        })[0]

    def advance(job, state):
        for _ in range(30):
            scheduler.tick()
            job = repo.get("jobs", job["id"], owner)
            if job["state"] == state:
                return job
        raise AssertionError(job)

    first = advance(submit("first", 2, 90), "completed")
    instance = repo.get("instances", first["instance_id"], owner)
    assert instance["idle_deadline_at"] == 6400
    assert instance.get("hard_deadline_at") is None
    now[0] = 1000 + 45 * 60
    scheduler.watchdog()

    second = advance(submit("second", 1500, 120, instance["id"]), "allocating")
    assert second["lease_id"] == first["lease_id"]
    assert second["task_deadline_epoch"] == now[0] + 1500 * 60
    payload = MCPGPUExecutor._payload(second, "test-execution", "/tmp")
    assert payload["max_runtime_minutes"] == 1500
    assert payload["deadline_epoch"] == second["task_deadline_epoch"]
    assert payload["generation_timeouts"] is True
    second = advance(second, "completed")
    assert repo.get("leases", second["lease_id"], owner)["idle_deadline_at"] == now[0] + 120 * 60
    retained = scheduler.instances.retention(owner, instance["id"], "retain", 150, "manual")
    assert retained["idle_deadline_at"] == now[0] + 150 * 60
    now[0] += 130 * 60
    scheduler.watchdog()
    assert repo.get("instances", instance["id"], owner)["state"] == "retained_idle"
    now[0] += 20 * 60 + 1
    scheduler.watchdog()
    assert repo.get("instances", instance["id"], owner)["state"] == "deleted"
