"""Exercise real authenticated HTTP routes against an isolated DB; no outbound transport."""
import pytest
from app.common import current_user
from app.config import settings
from app.db import Store
from app.gpu.repository import Repository
from app.main import app
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings,"encryption_key",Fernet.generate_key().decode())
    st=Store("sqlite:///:memory:")
    app.state.store=st
    app.dependency_overrides[current_user]=lambda:{"id":"library-http-owner","role":"admin"}
    with TestClient(app) as client:
        yield client,Repository(st)
    app.dependency_overrides.clear()


def test_catalog_queue_credentials_and_deletion_over_http(client):
    c,repo=client
    assert c.get('/api/gpu/model-library').json()['data']['items'] == []
    response=c.post('/api/gpu/model-library',json={"source":"https","download_url":"https://example.com/base.safetensors","filename":"base.safetensors","model_type":"checkpoints"})
    assert response.status_code == 201,response.text
    m=response.json()['data']
    first=c.post(f"/api/gpu/model-library/{m['id']}/prepare",json={})
    assert first.status_code == 202
    task=first.json()['data']
    assert c.post(f"/api/gpu/model-library/{m['id']}/prepare",json={}).json()['data']['id'] == task['id']
    assert c.post(f"/api/gpu/model-library/tasks/{task['id']}/cancel",json={}).json()['data']['state'] == 'cancelled'
    assert c.put('/api/gpu/model-library/source-settings/civitai',json={"token":"PRIVATE-KEY"}).json()['data']['civitai']
    assert 'PRIVATE-KEY' not in c.get('/api/gpu/model-library/source-settings').text
    assert 'encrypted_' not in c.get('/api/gpu/model-library/tasks').text
    assert c.patch(f"/api/gpu/model-library/{m['id']}",json={"display_name":"Base XL","architecture":"SDXL","sha256":"forged"}).status_code == 200
    assert repo.get('model_library',m['id'],'library-http-owner')['sha256'] is None
    assert c.delete(f"/api/gpu/model-library/{m['id']}").status_code == 200


def test_http_security_paths_and_ownership(client):
    c,repo=client
    bad=c.post('/api/gpu/model-library',json={"source":"https","download_url":"https://example.com/base","filename":"../escape.safetensors","model_type":"vae"})
    assert bad.status_code == 422
    foreign=repo.create('model_library','someone-else',{"display_name":"private","filename":"hidden.safetensors","model_type":"vae","architecture":"unknown","state":"ready"})
    assert c.get('/api/gpu/model-library/' + foreign['id']).status_code == 422
    assert c.get('/api/gpu/model-library').json()['data']['items'] == []


def test_png_import_scans_and_preserves_metadata_http(client):
    c,repo=client
    extraction={"workflow":{"nodes":[{"id":1,"type":"VAELoader","widgets_values":["v.safetensors"]}]},
                "prompt":{"1":{"class_type":"VAELoader","inputs":{"vae_name":"v.safetensors"}}},
                "metadata":[{"keyword":"author","text":"original"}]}
    saved=c.post('/api/gpu/workflows/import',json={"workflow_key":"http-png", "image_extraction":extraction})
    assert saved.status_code == 201,saved.text
    row=saved.json()['data']
    assert row['image_extraction'] == extraction
    dep=c.get(f"/api/gpu/workflows/{row['id']}/model-dependencies").json()['data']
    assert dep['dependencies'][0]['reference'] == 'v.safetensors'
    assert not dep['models_ready'] and not dep['runtime_validated']
