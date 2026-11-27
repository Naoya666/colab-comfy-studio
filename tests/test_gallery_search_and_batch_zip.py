"""Tests for gallery prompt/metadata search enrichment and Studio/Drive batch ZIP downloads."""

from __future__ import annotations

import io
import time
import zipfile
from urllib.parse import unquote

import httpx
from app import asset_routes, batch_zip, native_outputs_routes
from app.gpu import native_outputs
from app.gpu.drive import FOLDER_MIME
from app.gpu.repository import Repository
from app.security import sign_session
from app.storage import save_asset

from tests.gpu.test_routes_draft import gpu_client  # noqa: F811, F401
from tests.test_frontend_enhancements import _build_mock_drive, studio_client  # noqa: F811, F401
from tests.test_image_metadata_download import (
    _make_jpeg_with_metadata,
    _make_png_with_workflow_and_color,
    _make_webp_with_metadata,
    _png_chunk_types,
)


def test_batch_zip_filename_sanitization_and_deduplication():
    used: set[str] = set()
    first = batch_zip.unique_zip_entry_name("../../etc/passwd.png", used)
    assert first == "passwd.png"

    dup = batch_zip.unique_zip_entry_name("PASSWD.png", used)
    assert dup == "PASSWD (2).png"

    clean = batch_zip.unique_zip_entry_name(
        "ComfyUI_00001_.png",
        used,
        strip_metadata=True,
        strippable=True,
        subfolder="portraits/../nested/sub",
    )
    assert clean == "portraits/nested/sub/ComfyUI_00001_clean.png"

    clean_dup = batch_zip.unique_zip_entry_name(
        "ComfyUI_00001_.png",
        used,
        strip_metadata=True,
        strippable=True,
        subfolder="portraits/nested/sub",
    )
    assert clean_dup == "portraits/nested/sub/ComfyUI_00001_clean (2).png"

    # Non-strippable video keeps original name even when strip_metadata=True
    vid = batch_zip.unique_zip_entry_name("clip.mp4", used, strip_metadata=True, strippable=False)
    assert vid == "clip.mp4"


def test_gallery_enriches_prompt_negative_model_and_seed_for_search(gpu_client):  # noqa: F811
    client, store, user, project, _headers = gpu_client
    repo = Repository(store)
    png_raw = _make_png_with_workflow_and_color()

    job_with_comp = repo.create(
        "jobs",
        user["id"],
        {
            "project_id": project["id"],
            "model_id": "flux-krea-2",
            "workflow_key": "krea2-moss-dual-t2i",
            "workflow_version": "1.2.0",
            "state": "completed",
            "provider": "colab",
            "resolved_seed": 424242,
            "params": {"prompt": "masterpiece, cyberpunk neon rain street"},
            "prompt_composition": {
                "positive": {
                    "param_key": "prompt",
                    "main": "cyberpunk neon rain street",
                    "fixed": "masterpiece, best quality",
                    "enabled": True,
                    "position": "prefix",
                    "separator": "comma",
                },
                "negative": {
                    "param_key": "negative_prompt",
                    "main": "blurry, watermark",
                    "fixed": "lowres",
                    "enabled": True,
                    "position": "prefix",
                    "separator": "comma",
                },
            },
            "snapshot": {
                "resolved_params": {
                    "prompt": "masterpiece, best quality, cyberpunk neon rain street",
                    "negative_prompt": "lowres, blurry, watermark",
                    "seed": 424242,
                },
            },
        },
        project_id=project["id"],
    )
    asset1 = save_asset(
        store,
        user["id"],
        project["id"],
        png_raw,
        "neon.png",
        source_job_id=job_with_comp["id"],
    )

    # Secondary job with non-standard caption / custom negative param keys
    job_caption = repo.create(
        "jobs",
        user["id"],
        {
            "project_id": project["id"],
            "model_id": "minimax-h3",
            "workflow_key": "h3-video",
            "workflow_version": "1.0.0",
            "state": "completed",
            "provider": "mock",
            "params": {
                "scene_caption": "sakura petals drifting in golden sunlight",
                "extra_negative": "flicker, distorted hands",
                "seed": 99001,
            },
        },
        project_id=project["id"],
    )
    asset2 = save_asset(
        store,
        user["id"],
        project["id"],
        png_raw,
        "sakura.png",
        source_job_id=job_caption["id"],
    )

    res = client.get("/api/gpu/gallery")
    assert res.status_code == 200, res.text
    items = {item["id"]: item for item in res.json()["data"]}

    item1 = items[asset1["id"]]
    assert item1["prompt"] == "masterpiece, best quality, cyberpunk neon rain street"
    assert item1["main_prompt"] == "cyberpunk neon rain street"
    assert item1["negative_prompt"] == "lowres, blurry, watermark"
    assert item1["model_id"] == "flux-krea-2"
    assert item1["resolved_seed"] == 424242

    item2 = items[asset2["id"]]
    assert item2["prompt"] == "sakura petals drifting in golden sunlight"
    assert item2["negative_prompt"] == "flicker, distorted hands"
    assert item2["model_id"] == "minimax-h3"
    assert item2["resolved_seed"] == 99001
    assert "main_prompt" not in item2


def test_studio_batch_zip_download_original_clean_and_limits(gpu_client, monkeypatch):  # noqa: F811
    client, store, user, project, _headers = gpu_client
    png_raw = _make_png_with_workflow_and_color()
    jpg_raw = _make_jpeg_with_metadata()
    mp4_raw = b"\x00\x00\x00\x18ftypmp42" + b"0" * 128

    a1 = save_asset(store, user["id"], project["id"], png_raw, "shot.png", source_job_id="job-1")
    a2 = save_asset(store, user["id"], project["id"], png_raw, "shot.png", source_job_id="job-2")
    a3 = save_asset(store, user["id"], project["id"], jpg_raw, "photo.jpg", source_job_id="job-3")
    a4 = save_asset(store, user["id"], project["id"], mp4_raw, "clip.mp4", source_job_id="job-4")

    # 1. Original ZIP download
    t_orig = client.post(
        "/api/assets/batch-download",
        json={"asset_ids": [a1["id"], a2["id"], a3["id"], a4["id"]], "strip_metadata": False},
    )
    assert t_orig.status_code == 200, t_orig.text
    ticket_orig = t_orig.json()["data"]
    assert ticket_orig["item_count"] == 4
    assert ticket_orig["strippable_count"] == 3
    assert ticket_orig["skipped_strip_count"] == 0
    assert ticket_orig["filename"].endswith(".zip")
    assert "_clean.zip" not in ticket_orig["filename"]

    r_orig = client.get(ticket_orig["download_url"], headers={"Accept-Encoding": "gzip"})
    assert r_orig.status_code == 200, r_orig.text
    assert r_orig.headers["content-type"] == "application/zip"
    assert "content-encoding" not in r_orig.headers  # Must not be double-gzipped
    assert int(r_orig.headers["content-length"]) == len(r_orig.content)
    assert unquote(r_orig.headers["content-disposition"]).startswith("attachment;")

    with zipfile.ZipFile(io.BytesIO(r_orig.content), "r") as zf:
        names = zf.namelist()
        assert names == ["shot.png", "shot (2).png", "photo.jpg", "clip.mp4"]
        assert zf.read("shot.png") == png_raw
        assert zf.read("photo.jpg") == jpg_raw
        assert zf.read("clip.mp4") == mp4_raw
        for info in zf.infolist():
            assert info.compress_type == zipfile.ZIP_STORED

    # 2. Clean ZIP download: strips metadata from images, keeps video untouched!
    t_clean = client.post(
        "/api/assets/batch-download",
        json={"asset_ids": [a1["id"], a2["id"], a3["id"], a4["id"]], "strip_metadata": True},
    )
    assert t_clean.status_code == 200, t_clean.text
    ticket_clean = t_clean.json()["data"]
    assert ticket_clean["strippable_count"] == 3
    assert ticket_clean["skipped_strip_count"] == 1
    assert ticket_clean["filename"].endswith("_clean.zip")

    r_clean = client.get(ticket_clean["download_url"])
    assert r_clean.status_code == 200, r_clean.text
    with zipfile.ZipFile(io.BytesIO(r_clean.content), "r") as zf:
        names = zf.namelist()
        assert names == ["shot_clean.png", "shot_clean (2).png", "photo_clean.jpg", "clip.mp4"]
        clean_png = zf.read("shot_clean.png")
        assert b"KSampler" not in clean_png
        assert b"secret" not in clean_png
        assert b"tEXt" not in _png_chunk_types(clean_png)
        clean_jpg = zf.read("photo_clean.jpg")
        assert b"secret-jpeg-workflow" not in clean_jpg
        assert b"ICC_PROFILE\x00\x01\x01FAKE_ICC_DATA" in clean_jpg
        assert zf.read("clip.mp4") == mp4_raw

    # 3. Cross-user isolation on ticket download
    other_token = sign_session(
        {
            "uid": "other-user",
            "role": "user",
            "purpose": "studio-session",
            "iat": int(time.time()),
            "exp": int(time.time()) + 3600,
        }
    )
    r_other = client.get(
        ticket_clean["download_url"],
        headers={"Authorization": f"Bearer {other_token}"},
    )
    assert r_other.status_code == 404

    # 4. Expired ticket returns 410
    store.update(
        "settings",
        f"{batch_zip.TICKET_KEY_PREFIX}{ticket_clean['ticket_id']}",
        {"expires_at": time.time() - 5},
        user["id"],
    )
    r_expired = client.get(ticket_clean["download_url"])
    assert r_expired.status_code == 410

    # 5. Size limit enforcement (413)
    monkeypatch.setattr(asset_routes, "STUDIO_BATCH_ZIP_MAX_BYTES", 100)
    r_too_big = client.post(
        "/api/assets/batch-download",
        json={"asset_ids": [a1["id"]], "strip_metadata": False},
    )
    assert r_too_big.status_code == 413
    assert r_too_big.json()["error"]["code"] == "PAYLOAD_TOO_LARGE"

    # 6. Validation errors
    r_empty = client.post("/api/assets/batch-download", json={"asset_ids": []})
    assert r_empty.status_code == 422
    r_bad_flag = client.post("/api/assets/batch-download", json={"asset_ids": [a1["id"]], "strip_metadata": "yes"})
    assert r_bad_flag.status_code == 422


def test_native_drive_batch_zip_download_original_clean_and_limits(studio_client, monkeypatch):  # noqa: F811
    client, store, headers = studio_client
    repo = Repository(store)
    png_raw = _make_png_with_workflow_and_color()
    webp_raw = _make_webp_with_metadata(alpha=True)
    mp4_raw = b"\x00\x00\x00\x18ftypmp42" + b"0" * 96

    tree = {
        "root": [{"id": "f-root", "name": "AI-Studio", "mimeType": FOLDER_MIME}],
        "f-root": [{"id": "f-owner", "name": "owner", "mimeType": FOLDER_MIME}],
        "f-owner": [{"id": "f-native", "name": "native-comfy", "mimeType": FOLDER_MIME}],
        "f-native": [{"id": "f-out", "name": "output", "mimeType": FOLDER_MIME}],
        "f-out": [
            {"id": "f-sub", "name": "portraits", "mimeType": FOLDER_MIME},
            {
                "id": "img-webp",
                "name": "preview.webp",
                "mimeType": "image/webp",
                "size": str(len(webp_raw)),
                "md5Checksum": "md5webp01",
                "modifiedTime": "2026-11-27T10:00:00Z",
            },
            {
                "id": "vid-mp4",
                "name": "motion.mp4",
                "mimeType": "video/mp4",
                "size": str(len(mp4_raw)),
                "md5Checksum": "md5mp401",
                "modifiedTime": "2026-11-27T10:01:00Z",
            },
        ],
        "f-sub": [
            {
                "id": "img-png",
                "name": "ComfyUI_00001_.png",
                "mimeType": "image/png",
                "size": str(len(png_raw)),
                "md5Checksum": "md5png01",
                "modifiedTime": "2026-11-27T10:02:00Z",
            },
        ],
    }
    mock_drive, base_handler = _build_mock_drive(
        repo,
        tree,
        {"img-png": png_raw, "img-webp": webp_raw, "vid-mp4": mp4_raw},
        [],
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
        lambda *a, **kw: orig_async_client(*a, transport=httpx.MockTransport(base_handler), **kw),
    )
    native_outputs.invalidate_native_outputs_cache()

    # 1. Original Drive ZIP preserves subfolder structure and raw bytes
    t_orig = client.post(
        "/api/gpu/native/outputs/batch-download",
        json={"file_ids": ["img-png", "img-webp", "vid-mp4"], "strip_metadata": False},
        headers=headers,
    )
    assert t_orig.status_code == 200, t_orig.text
    ticket_orig = t_orig.json()["data"]
    assert ticket_orig["item_count"] == 3
    assert ticket_orig["strippable_count"] == 2
    assert ticket_orig["download_url"].startswith("/api/gpu/native/outputs/batch-download/")

    r_orig = client.get(ticket_orig["download_url"], headers=headers)
    assert r_orig.status_code == 200, r_orig.text
    assert r_orig.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(r_orig.content), "r") as zf:
        assert zf.namelist() == ["portraits/ComfyUI_00001_.png", "preview.webp", "motion.mp4"]
        assert zf.read("portraits/ComfyUI_00001_.png") == png_raw
        assert zf.read("preview.webp") == webp_raw
        assert zf.read("motion.mp4") == mp4_raw

    # 2. Clean Drive ZIP strips PNG/WebP metadata, appends _clean, keeps MP4 intact
    t_clean = client.post(
        "/api/gpu/native/outputs/batch-download",
        json={"file_ids": ["img-png", "img-webp", "vid-mp4"], "strip_metadata": True},
        headers=headers,
    )
    assert t_clean.status_code == 200, t_clean.text
    ticket_clean = t_clean.json()["data"]
    assert ticket_clean["skipped_strip_count"] == 1
    assert ticket_clean["filename"].endswith("_clean.zip")

    r_clean = client.get(ticket_clean["download_url"], headers=headers)
    assert r_clean.status_code == 200, r_clean.text
    with zipfile.ZipFile(io.BytesIO(r_clean.content), "r") as zf:
        assert zf.namelist() == ["portraits/ComfyUI_00001_clean.png", "preview_clean.webp", "motion.mp4"]
        assert b"KSampler" not in zf.read("portraits/ComfyUI_00001_clean.png")
        assert b"secret-webp-workflow" not in zf.read("preview_clean.webp")
        assert zf.read("motion.mp4") == mp4_raw

    # 3. Validation & size limits
    r_bad_id = client.post(
        "/api/gpu/native/outputs/batch-download",
        json={"file_ids": ["../bad"]},
        headers=headers,
    )
    assert r_bad_id.status_code == 400

    r_too_many = client.post(
        "/api/gpu/native/outputs/batch-download",
        json={"file_ids": [f"f-{i}" for i in range(51)]},
        headers=headers,
    )
    assert r_too_many.status_code == 422

    monkeypatch.setattr(native_outputs_routes, "NATIVE_BATCH_ZIP_MAX_BYTES", 64)
    r_too_big = client.post(
        "/api/gpu/native/outputs/batch-download",
        json={"file_ids": ["img-png"]},
        headers=headers,
    )
    assert r_too_big.status_code == 413
