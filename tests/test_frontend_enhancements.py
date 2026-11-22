"""Tests for native telemetry, read-only Drive usage breakdown, native outputs gallery, and PWA manifest."""

from __future__ import annotations

import io
import json
from pathlib import Path

import httpx
import pytest
from app.config import settings
from app.db import Store
from app.gpu import drive_usage, native, native_outputs
from app.gpu.drive import FOLDER_MIME, DriveStorage
from app.gpu.google import GoogleService
from app.gpu.repository import Repository
from app.main import app
from fastapi.testclient import TestClient
from PIL import Image


def _png_bytes(color=(120, 180, 220), size=(64, 48)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def studio_client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(settings, "studio_password", "test-pass-123")
    monkeypatch.setattr(settings, "studio_api_token", "tok-test-admin")
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    store = Store(f"sqlite:///{tmp_path / 'studio.db'}")
    app.state.store = store
    app.state.repo = Repository(store)
    drive_usage.invalidate_drive_usage_cache()
    native_outputs.invalidate_native_outputs_cache()
    from app import native_routes
    native_routes._TELEMETRY_CACHE.clear()

    client = TestClient(app)
    headers = {"Authorization": "Bearer tok-test-admin"}
    yield client, store, headers
    drive_usage.invalidate_drive_usage_cache()
    native_outputs.invalidate_native_outputs_cache()
    native_routes._TELEMETRY_CACHE.clear()


def test_native_telemetry_inactive_and_cached(studio_client, monkeypatch):
    client, store, headers = studio_client
    repo = Repository(store)

    # 1. No active lease -> zero upstream requests
    resp = client.get("/api/gpu/native/telemetry", headers=headers)
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["active"] is False
    assert data["ready"] is False
    assert data["telemetry"] is None

    # 2. Active ready native lease -> parallel telemetry with 4s cache
    repo.create(
        "leases",
        "owner",
        {
            "state": "idle",
            "native_comfy": True,
            "native_ready": True,
            "instance_id": "inst-native-1",
            "account_id": "acc-1",
            "native_model_storage": {"model_storage_mode": "local", "cache_gb": 20, "cache_max_age_minutes": 60},
        },
        record_id="lease-native-1",
    )

    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/system_stats":
            return httpx.Response(
                200,
                json={
                    "system": {
                        "comfyui_version": "0.3.20",
                        "pytorch_version": "2.6.0+cu124",
                        "ram_total": 54 * 1024**3,
                        "ram_free": 40 * 1024**3,
                    },
                    "devices": [
                        {
                            "name": "NVIDIA L4",
                            "type": "cuda",
                            "vram_total": 24 * 1024**3,
                            "vram_free": 18 * 1024**3,
                            "torch_vram_total": 8 * 1024**3,
                            "torch_vram_free": 2 * 1024**3,
                        }
                    ],
                },
            )
        if request.url.path == "/queue":
            return httpx.Response(200, json={"queue_running": [["p1"]], "queue_pending": [["p2"], ["p3"]]})
        if request.url.path == "/studio_native/state":
            return httpx.Response(
                200,
                json={"busy": False, "busy_reason": None, "disk_free_bytes": 90 * 1024**3},
            )
        return httpx.Response(404)

    monkeypatch.setattr(native, "backend_for", lambda repo=None, owner="owner": ("https://tunnel.example", {"CF-Access-Client-Id": "id"}))
    orig_async_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda *a, **kw: orig_async_client(*a, transport=httpx.MockTransport(handler), **kw),
    )

    r1 = client.get("/api/gpu/native/telemetry", headers=headers)
    assert r1.status_code == 200
    d1 = r1.json()["data"]
    assert d1["active"] is True
    assert d1["ready"] is True
    t = d1["telemetry"]
    assert t["gpu_name"] == "NVIDIA L4"
    assert t["vram_used_bytes"] == 6 * 1024**3
    assert t["torch_vram_used_bytes"] == 6 * 1024**3
    assert t["ram_used_bytes"] == 14 * 1024**3
    assert t["queue_running"] == 1
    assert t["queue_pending"] == 2
    assert t["busy"] is True
    assert t["busy_reason"] == "queue"
    assert t["disk_free_bytes"] == 90 * 1024**3
    assert t["model_storage_mode"] == "local"
    assert len(calls) == 3

    # Second immediate call hits 4s TTL cache -> no additional tunnel probes
    r2 = client.get("/api/gpu/native/telemetry", headers=headers)
    assert r2.status_code == 200
    assert len(calls) == 3

    # Force refresh bypasses cache
    r3 = client.get("/api/gpu/native/telemetry?refresh=true", headers=headers)
    assert r3.status_code == 200
    assert len(calls) == 6


class _FakeGoogle(GoogleService):
    def __init__(self):
        pass

    def access_token(self, purpose: str = "drive") -> str:
        return "fake-drive-token"

    def status(self):
        return {"drive": {"connected": True}, "colab": {"connected": False}}


def _build_mock_drive(repo, tree_folders: dict[str, list[dict]], file_bytes: dict[str, bytes], trashed_out: list[str], quota: dict | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/about"):
            if quota is None:
                return httpx.Response(403, json={"error": "forbidden"})
            return httpx.Response(200, json={"storageQuota": quota})
        if path.endswith("/files") and request.method == "GET":
            q = request.url.params.get("q", "")
            if q.strip() == "trashed = false":
                all_entries = [
                    {**e, "parents": [parent_id]}
                    for parent_id, entries in tree_folders.items()
                    for e in entries
                    if e["id"] not in trashed_out
                ]
                return httpx.Response(200, json={"files": all_entries})
            # Match folder lookup by name & parent
            for parent_id, entries in tree_folders.items():
                if f"'{parent_id}' in parents" in q:
                    matched = [e for e in entries if e["id"] not in trashed_out]
                    if "name = '" in q:
                        target_name = q.split("name = '", 1)[1].split("'", 1)[0]
                        matched = [e for e in matched if e["name"] == target_name]
                    if f"mimeType = '{FOLDER_MIME}'" in q:
                        matched = [e for e in matched if e["mimeType"] == FOLDER_MIME]
                    return httpx.Response(200, json={"files": matched})
            return httpx.Response(200, json={"files": []})
        if "/files/" in path:
            fid = path.rsplit("/files/", 1)[1]
            if request.method == "PATCH":
                body = json.loads(request.content.decode())
                if body.get("trashed") is True:
                    trashed_out.append(fid)
                    return httpx.Response(200, json={"id": fid, "trashed": True})
            if request.method == "GET" and request.url.params.get("alt") == "media":
                raw = file_bytes.get(fid)
                if raw is None:
                    return httpx.Response(404)
                range_hdr = request.headers.get("range")
                if range_hdr and range_hdr.startswith("bytes="):
                    start_s, end_s = range_hdr.removeprefix("bytes=").split("-", 1)
                    start = int(start_s)
                    end = int(end_s) if end_s else len(raw) - 1
                    chunk = raw[start : end + 1]
                    return httpx.Response(
                        206,
                        content=chunk,
                        headers={
                            "Content-Range": f"bytes {start}-{end}/{len(raw)}",
                            "Content-Length": str(len(chunk)),
                        },
                    )
                return httpx.Response(200, content=raw, headers={"Content-Length": str(len(raw))})
        return httpx.Response(404)

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    return DriveStorage(
        {"id": "test", "owner_id": "owner"},
        repo,
        account_id="acc-drive",
        client=http_client,
        google=_FakeGoogle(),
    ), handler


def test_drive_storage_breakdown_readonly(studio_client, monkeypatch):
    client, store, headers = studio_client
    repo = Repository(store)

    # Register one indexed model in model_library
    repo.create(
        "model_library",
        "owner",
        {
            "display_name": "Anima",
            "filename": "anima.safetensors",
            "model_type": "diffusion_models",
            "state": "ready",
            "size": 1000,
            "drive_file_id": "file-model-1",
        },
        record_id="ml-1",
    )

    tree = {
        "root": [{"id": "f-root", "name": "AI-Studio", "mimeType": FOLDER_MIME}],
        "f-root": [{"id": "f-owner", "name": "owner", "mimeType": FOLDER_MIME}],
        "f-owner": [
            {"id": "f-models", "name": "models", "mimeType": FOLDER_MIME},
            {"id": "f-native", "name": "native-comfy", "mimeType": FOLDER_MIME},
            {"id": "f-rc", "name": "runtime-cache", "mimeType": FOLDER_MIME},
        ],
        "f-models": [
            {"id": "file-model-1", "name": "anima.safetensors", "mimeType": "application/octet-stream", "size": "1000"},
            {"id": "file-model-orphan", "name": "old.safetensors", "mimeType": "application/octet-stream", "size": "500"},
            {"id": "file-shortcut", "name": "alias", "mimeType": "application/vnd.google-apps.shortcut", "size": "9999"},
        ],
        "f-native": [
            {"id": "f-env", "name": "env-cache", "mimeType": FOLDER_MIME},
            {"id": "f-plug", "name": "plugins", "mimeType": FOLDER_MIME},
            {"id": "f-out", "name": "output", "mimeType": FOLDER_MIME},
            {"id": "f-user", "name": "user", "mimeType": FOLDER_MIME},
        ],
        "f-env": [{"id": "env1", "name": "native-env.tar.gz", "mimeType": "application/gzip", "size": "2000"}],
        "f-plug": [{"id": "plg1", "name": "LanPaint.tar.gz", "mimeType": "application/gzip", "size": "300"}],
        "f-out": [{"id": "out1", "name": "pic.png", "mimeType": "image/png", "size": "400"}],
        "f-user": [{"id": "usr1", "name": "wf.json", "mimeType": "application/json", "size": "50"}],
        "f-rc": [{"id": "rc1", "name": "cache.tar", "mimeType": "application/x-tar", "size": "250"}],
    }
    mock_drive, _ = _build_mock_drive(
        repo,
        tree,
        {},
        [],
        quota={"limit": "100000", "usage": "25000", "usageInDrive": "20000", "usageInDriveTrash": "1500"},
    )
    _orig_collect = drive_usage.collect_drive_usage
    monkeypatch.setattr(
        drive_usage,
        "collect_drive_usage",
        lambda r, owner, force_refresh=False: _orig_collect(r, owner, force_refresh=force_refresh, drive=mock_drive),
    )
    resp_ok = client.get("/api/gpu/model-library/drive-usage?refresh=true", headers=headers)
    assert resp_ok.status_code == 200
    result = resp_ok.json()["data"]
    assert result["connected"] is True
    assert result["quota"]["limit"] == 100000
    assert result["quota"]["usage_in_drive_trash"] == 1500
    assert result["studio_total_bytes"] == 1000 + 500 + 2000 + 300 + 400 + 50 + 250
    by_key = {c["key"]: c["bytes"] for c in result["categories"]}
    assert by_key["models_indexed"] == 1000
    assert by_key["models_other"] == 500
    assert by_key["native_env_cache"] == 2000
    assert by_key["native_plugins"] == 300
    assert by_key["native_outputs"] == 400
    assert by_key["native_workspace"] == 50
    assert by_key["studio_other"] == 250

    # Also test route disconnected fallback cleanly returns 200 connected=False
    monkeypatch.setattr(drive_usage, "collect_drive_usage", _orig_collect)
    resp = client.get("/api/gpu/model-library/drive-usage", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["data"]["connected"] is False


def test_native_outputs_gallery_full_lifecycle(studio_client, monkeypatch):
    client, store, headers = studio_client
    repo = Repository(store)

    # Create a Studio project for reference import
    proj = client.post("/api/gpu/projects", json={"title": "Test Project"}, headers=headers).json()["data"]
    project_id = proj["id"]

    png_data = _png_bytes()
    mp4_data = b"\x00\x00\x00\x18ftypmp42" + b"0" * 64
    trashed: list[str] = []
    tree = {
        "root": [{"id": "f-root", "name": "AI-Studio", "mimeType": FOLDER_MIME}],
        "f-root": [{"id": "f-owner", "name": "owner", "mimeType": FOLDER_MIME}],
        "f-owner": [
            {"id": "f-native", "name": "native-comfy", "mimeType": FOLDER_MIME},
            {"id": "secret-outside", "name": "secret.png", "mimeType": "image/png", "size": "100"},
        ],
        "f-native": [{"id": "f-out", "name": "output", "mimeType": FOLDER_MIME}],
        "f-out": [
            {"id": "f-sub", "name": "2026-11-22", "mimeType": FOLDER_MIME},
            {
                "id": "vid-1",
                "name": "clip.mp4",
                "mimeType": "video/mp4",
                "size": str(len(mp4_data)),
                "md5Checksum": "md5vid1",
                "modifiedTime": "2026-11-22T12:00:00Z",
                "videoMediaMetadata": {"width": 720, "height": 1280},
            },
        ],
        "f-sub": [
            {
                "id": "img-1",
                "name": "anima_00001_.png",
                "mimeType": "image/png",
                "size": str(len(png_data)),
                "md5Checksum": "md5img1",
                "modifiedTime": "2026-11-22T12:05:00Z",
                "imageMediaMetadata": {"width": 64, "height": 48},
            },
        ],
    }
    mock_drive, drive_handler = _build_mock_drive(
        repo, tree, {"img-1": png_data, "vid-1": mp4_data, "secret-outside": png_data}, trashed,
    )

    monkeypatch.setattr(
        native_outputs,
        "_open_drive",
        lambda r, owner, drive=None: (mock_drive, "acc-drive", "主盘", False),
    )
    orig_async_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda *a, **kw: orig_async_client(*a, transport=httpx.MockTransport(drive_handler), **kw),
    )

    # 1. List recursive outputs
    r_list = client.get("/api/gpu/native/outputs", headers=headers)
    assert r_list.status_code == 200
    body = r_list.json()["data"]
    assert body["connected"] is True
    assert body["total"] == 2
    assert body["subfolders"] == ["2026-11-22"]
    assert body["items"][0]["id"] == "img-1"
    assert body["items"][0]["subfolder"] == "2026-11-22"
    assert body["items"][0]["favorite"] is False

    # 2. Favorite toggle & filter
    r_fav = client.patch("/api/gpu/native/outputs/img-1", json={"favorite": True}, headers=headers)
    assert r_fav.status_code == 200
    assert r_fav.json()["data"]["favorite"] is True

    r_only = client.get("/api/gpu/native/outputs?favorite=only", headers=headers)
    assert [i["id"] for i in r_only.json()["data"]["items"]] == ["img-1"]

    # 3. Security boundary: file outside native-comfy/output/ is rejected
    assert client.get("/api/gpu/native/outputs/secret-outside/content", headers=headers).status_code == 404

    # 4. Thumbnail generation & local WebP disk cache
    r_thumb = client.get("/api/gpu/native/outputs/img-1/thumbnail", headers=headers)
    assert r_thumb.status_code == 200
    assert r_thumb.headers["content-type"] == "image/webp"
    cached_files = list((settings.data_dir / "native-thumb-cache").rglob("img-1-*.webp"))
    assert len(cached_files) == 1

    # 5. Streaming content with HTTP Range support
    r_range = client.get("/api/gpu/native/outputs/vid-1/content", headers={**headers, "Range": "bytes=0-15"})
    assert r_range.status_code == 206
    assert len(r_range.content) == 16
    assert r_range.headers["accept-ranges"] == "bytes"
    assert r_range.headers["content-range"].startswith("bytes 0-15/")

    # 6. Import as Studio reference asset
    r_imp = client.post(
        "/api/gpu/native/outputs/img-1/import-reference",
        json={"project_id": project_id},
        headers=headers,
    )
    assert r_imp.status_code == 201
    imported = r_imp.json()["data"]
    assert imported["project_id"] == project_id
    assert imported["filename"] == "anima_00001_.png"

    refs = client.get(f"/api/assets?project_id={project_id}", headers=headers).json()["data"]
    assert any(a["id"] == imported["id"] for a in refs)

    # 7. Trash output clears favorite & thumbnail cache
    r_del = client.delete("/api/gpu/native/outputs/img-1", headers=headers)
    assert r_del.status_code == 200
    assert "img-1" in trashed
    assert list((settings.data_dir / "native-thumb-cache").rglob("img-1-*.webp")) == []
    r_after = client.get("/api/gpu/native/outputs?refresh=true", headers=headers).json()["data"]
    assert [i["id"] for i in r_after["items"]] == ["vid-1"]


def test_pwa_manifest_and_icons_exist():
    public_dir = Path(__file__).resolve().parents[1] / "web" / "public"
    manifest_path = public_dir / "manifest.webmanifest"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["display"] == "standalone"
    assert manifest["start_url"] == "/"
    for icon in manifest["icons"]:
        icon_file = public_dir / icon["src"].lstrip("/")
        assert icon_file.is_file(), f"missing icon {icon_file}"
    assert (public_dir / "icons" / "apple-touch-icon.png").is_file()
