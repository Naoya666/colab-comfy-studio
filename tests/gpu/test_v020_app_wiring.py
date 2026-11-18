import pytest
from app.common import current_user
from app.db import Store
from app.main import app
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path):
    app.state.store = Store('sqlite:///' + str(tmp_path / 'wiring.db'))
    app.dependency_overrides[current_user] = lambda: {'id': 'owner', 'role': 'admin'}
    with TestClient(app) as session:
        yield session
    app.dependency_overrides.clear()


def test_instance_routes_are_mounted_and_have_real_repository_service(client):
    response = client.get('/api/gpu/instances')
    assert response.status_code == 200, response.text
    assert response.json()['data'] == []


def test_account_routes_mounted(client):
    response = client.get('/api/gpu/accounts')
    assert response.status_code == 200, response.text
    assert isinstance(response.json()['data'], list)


def test_instance_routes_require_authentication(client):
    app.dependency_overrides.pop(current_user, None)
    response = client.get('/api/gpu/instances')
    assert response.status_code == 401


def test_instance_history_clear_keeps_active_rows_and_related_records(client):
    repo = app.state.repo
    for owner, instance_id, state in (
        ("owner", "older", "deleted"),
        ("owner", "newer", "deleted"),
        ("owner", "cleanup", "cleanup_pending"),
        ("owner", "latest", "retained_idle"),
        ("other", "foreign", "deleted"),
    ):
        repo.create("instances", owner, {"state": state}, record_id=instance_id)
    repo.create("jobs", "owner", {"state": "completed", "instance_id": "older"}, record_id="saved-job")

    def instance_ids(path="/api/gpu/instances"):
        return [row["id"] for row in client.get(path).json()["data"]]

    assert instance_ids() == ["latest", "cleanup", "newer", "older"]
    assert client.post("/api/gpu/instances/history/clear").json()["data"] == {"cleared": 2}
    assert instance_ids() == ["latest", "cleanup"]
    assert instance_ids("/api/gpu/instances?state=deleted&include_cleared=true") == ["newer", "older"]
    assert client.get("/api/gpu/instances/older").json()["data"]["state"] == "deleted"
    assert client.get("/api/gpu/instances/older/jobs").json()["data"][0]["job_id"] == "saved-job"
    assert repo.get("instances", "foreign", "other").get("history_cleared_at") is None
    assert client.post("/api/gpu/instances/history/clear").json()["data"] == {"cleared": 0}
    repo.create("instances", "owner", {"state": "deleted"}, record_id="later")
    assert instance_ids() == ["later", "latest", "cleanup"]
