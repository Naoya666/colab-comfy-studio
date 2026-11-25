"""Tests for lossless image metadata stripping and clean gallery download routes."""

from __future__ import annotations

import io
import json
import struct
import zlib
from urllib.parse import unquote

import httpx
import pytest
from app.gpu import native_outputs
from app.gpu.drive import FOLDER_MIME
from app.gpu.repository import Repository
from app.image_metadata import (
    ImageMetadataStripError,
    clean_download_filename,
    is_strippable_image_mime,
    strip_image_metadata,
)
from PIL import Image, PngImagePlugin

from tests.gpu.test_routes_draft import gpu_client  # noqa: F401
from tests.test_frontend_enhancements import _build_mock_drive, studio_client  # noqa: F401


def _png_chunk(ctype: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(ctype + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + ctype + payload + struct.pack(">I", crc)


def _png_chunk_types(data: bytes) -> list[bytes]:
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    types: list[bytes] = []
    offset = 8
    while offset < len(data):
        length = struct.unpack_from(">I", data, offset)[0]
        ctype = data[offset + 4 : offset + 8]
        types.append(ctype)
        offset += 12 + length
    return types


def _make_png_with_workflow_and_color() -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGBA", (16, 12), (42, 128, 220, 200))
    info = PngImagePlugin.PngInfo()
    info.add_text("workflow", json.dumps({"nodes": [{"id": 1, "type": "KSampler"}], "links": []}))
    info.add_itxt("prompt", json.dumps({"1": {"class_type": "CLIPTextEncode", "inputs": {"text": "secret prompt"}}}), zip=True)
    info.add_text("parameters", "secret a1111 params", zip=True)
    img.save(buf, format="PNG", pnginfo=info)
    base = buf.getvalue()

    # Inject sRGB, pHYs, eXIf, tIME, and a private metadata chunk before IEND, plus trailing bytes after IEND.
    iend_pos = base.rfind(b"\x00\x00\x00\x00IEND")
    assert iend_pos > 0
    srgb = _png_chunk(b"sRGB", b"\x00")
    phys = _png_chunk(b"pHYs", struct.pack(">IIB", 2835, 2835, 1))
    exif = _png_chunk(b"eXIf", b"MM\x00*\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00")
    t_time = _png_chunk(b"tIME", struct.pack(">HBBBBB", 2026, 11, 25, 12, 0, 0))
    private_chunk = _png_chunk(b"foOb", b"secret-private-node-data")
    return base[:iend_pos] + srgb + phys + exif + t_time + private_chunk + base[iend_pos:] + b"TRAILING_SECRET"


def _make_jpeg_with_metadata(*, progressive: bool = False) -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (24, 18), (180, 90, 45))
    exif = Image.Exif()
    exif[0x010E] = json.dumps({"workflow": "secret-jpeg-workflow"})
    img.save(buf, format="JPEG", quality=92, exif=exif.tobytes(), progressive=progressive)
    raw = buf.getvalue()
    assert raw[:2] == b"\xff\xd8"
    # Inject COM, APP2 ICC_PROFILE, APP1 XMP, and APP13 IPTC segments right after SOI.
    com_payload = b'{"prompt": "secret-comment-prompt"}'
    com_seg = b"\xff\xfe" + struct.pack(">H", len(com_payload) + 2) + com_payload
    xmp_payload = b"http://ns.adobe.com/xap/1.0/\x00<x:xmpmeta>secret-xmp</x:xmpmeta>"
    xmp_seg = b"\xff\xe1" + struct.pack(">H", len(xmp_payload) + 2) + xmp_payload
    iptc_payload = b"Photoshop 3.0\x00secret-iptc"
    iptc_seg = b"\xff\xed" + struct.pack(">H", len(iptc_payload) + 2) + iptc_payload
    icc_payload = b"ICC_PROFILE\x00\x01\x01FAKE_ICC_DATA"
    icc_seg = b"\xff\xe2" + struct.pack(">H", len(icc_payload) + 2) + icc_payload
    return raw[:2] + com_seg + xmp_seg + iptc_seg + icc_seg + raw[2:] + b"JPEG_TRAILING"


def _make_webp_with_metadata(*, animated: bool = False, alpha: bool = False) -> bytes:
    buf = io.BytesIO()
    exif = Image.Exif()
    exif[0x010E] = json.dumps({"workflow": "secret-webp-workflow"})
    xmp = b"<x:xmpmeta>secret-webp-xmp</x:xmpmeta>"
    if animated:
        f1 = Image.new("RGBA", (16, 16), (255, 0, 0, 255))
        f2 = Image.new("RGBA", (16, 16), (0, 255, 0, 128))
        f1.save(buf, format="WEBP", save_all=True, append_images=[f2], duration=100, loop=0, exif=exif.tobytes(), xmp=xmp)
    elif alpha:
        img = Image.new("RGBA", (16, 16), (10, 200, 150, 128))
        img.save(buf, format="WEBP", lossless=True, exif=exif.tobytes(), xmp=xmp)
    else:
        img = Image.new("RGB", (16, 16), (20, 120, 210))
        img.save(buf, format="WEBP", quality=90, exif=exif.tobytes(), xmp=xmp)
    return buf.getvalue()


def _make_gif_with_metadata() -> bytes:
    buf = io.BytesIO()
    f1 = Image.new("RGB", (12, 12), (255, 10, 10))
    f2 = Image.new("RGB", (12, 12), (10, 10, 255))
    f1.save(buf, format="GIF", save_all=True, append_images=[f2], duration=80, loop=0, comment=b"secret-gif-workflow")
    raw = buf.getvalue()
    assert raw.endswith(b"\x3b")
    # Inject XMP Application Extension right before GIF trailer (0x3B).
    xmp_ext = b"\x21\xff\x0bXMP DataXMP\x0asecret-xmp\x00"
    return raw[:-1] + xmp_ext + b"\x3b"


def test_clean_download_filename_and_mime_allowlist():
    assert clean_download_filename("portrait.png") == "portrait_clean.png"
    assert clean_download_filename("ComfyUI_00001_.png") == "ComfyUI_00001_clean.png"
    assert clean_download_filename("sample-01-.webp") == "sample-01-clean.webp"
    assert clean_download_filename("角色立绘.png") == "角色立绘_clean.png"
    assert clean_download_filename("my.v2.shot.jpeg") == "my.v2.shot_clean.jpeg"
    assert clean_download_filename("already_clean.png") == "already_clean.png"
    assert clean_download_filename("already-CLEAN.jpg") == "already-CLEAN.jpg"
    assert clean_download_filename("noext") == "noext_clean"
    assert clean_download_filename("nested/dir/a.png") == "a_clean.png"
    assert clean_download_filename("") == "image_clean"

    for mime in ("image/png", "image/jpeg", "image/webp", "image/gif", "IMAGE/PNG"):
        assert is_strippable_image_mime(mime) is True
    for mime in ("video/mp4", "video/webm", "application/json", "audio/mpeg", "", None):
        assert is_strippable_image_mime(mime) is False


def test_strip_png_losslessly_removes_workflow_and_preserves_color_and_pixels():
    raw = _make_png_with_workflow_and_color()
    assert b"secret prompt" in raw or b"KSampler" in raw
    assert b"TRAILING_SECRET" in raw

    cleaned = strip_image_metadata(raw, "image/png")
    chunks = _png_chunk_types(cleaned)
    assert b"IHDR" in chunks and b"IDAT" in chunks and chunks[-1] == b"IEND"
    assert b"sRGB" in chunks and b"pHYs" in chunks
    for forbidden in (b"tEXt", b"iTXt", b"zTXt", b"eXIf", b"tIME", b"foOb"):
        assert forbidden not in chunks
    assert b"KSampler" not in cleaned
    assert b"secret" not in cleaned
    assert b"TRAILING_SECRET" not in cleaned

    with Image.open(io.BytesIO(raw)) as orig_im, Image.open(io.BytesIO(cleaned)) as clean_im:
        assert orig_im.size == clean_im.size
        assert orig_im.mode == clean_im.mode
        assert orig_im.tobytes() == clean_im.tobytes()
        for key in ("workflow", "prompt", "parameters", "exif"):
            assert key not in clean_im.info


@pytest.mark.parametrize("progressive", [False, True])
def test_strip_jpeg_losslessly_removes_exif_xmp_com_iptc_and_keeps_icc(progressive: bool):
    raw = _make_jpeg_with_metadata(progressive=progressive)
    assert b"secret-jpeg-workflow" in raw
    assert b"secret-comment-prompt" in raw
    assert b"secret-xmp" in raw
    assert b"secret-iptc" in raw
    assert b"FAKE_ICC_DATA" in raw

    cleaned = strip_image_metadata(raw, "image/jpeg")
    assert b"secret-jpeg-workflow" not in cleaned
    assert b"secret-comment-prompt" not in cleaned
    assert b"secret-xmp" not in cleaned
    assert b"secret-iptc" not in cleaned
    assert b"JPEG_TRAILING" not in cleaned
    assert b"ICC_PROFILE\x00\x01\x01FAKE_ICC_DATA" in cleaned
    assert cleaned.endswith(b"\xff\xd9")

    # Pixel output must match identically (proving zero lossy re-encoding).
    with Image.open(io.BytesIO(raw)) as orig_im, Image.open(io.BytesIO(cleaned)) as clean_im:
        assert orig_im.size == clean_im.size
        assert orig_im.tobytes() == clean_im.tobytes()
        assert not clean_im.getexif()


@pytest.mark.parametrize("animated,alpha", [(False, False), (False, True), (True, False)])
def test_strip_webp_losslessly_removes_exif_xmp_and_preserves_frames_and_alpha(animated: bool, alpha: bool):
    raw = _make_webp_with_metadata(animated=animated, alpha=alpha)
    assert b"secret-webp-workflow" in raw
    assert b"secret-webp-xmp" in raw

    cleaned = strip_image_metadata(raw, "image/webp")
    assert b"secret-webp-workflow" not in cleaned
    assert b"secret-webp-xmp" not in cleaned
    assert b"EXIF" not in cleaned
    assert b"XMP " not in cleaned
    if not animated and not alpha:
        # Plain RGB WebP with only EXIF/XMP should simplify cleanly without an empty VP8X chunk.
        assert b"VP8X" not in cleaned

    with Image.open(io.BytesIO(raw)) as orig_im, Image.open(io.BytesIO(cleaned)) as clean_im:
        assert orig_im.size == clean_im.size
        assert getattr(orig_im, "n_frames", 1) == getattr(clean_im, "n_frames", 1)
        assert orig_im.tobytes() == clean_im.tobytes()
        assert not clean_im.getexif()
        assert "xmp" not in clean_im.info


def test_strip_gif_removes_comment_and_xmp_and_preserves_animation():
    raw = _make_gif_with_metadata()
    assert b"secret-gif-workflow" in raw
    assert b"secret-xmp" in raw
    assert b"NETSCAPE2.0" in raw

    cleaned = strip_image_metadata(raw, "image/gif")
    assert b"secret-gif-workflow" not in cleaned
    assert b"secret-xmp" not in cleaned
    assert b"NETSCAPE2.0" in cleaned
    assert cleaned.endswith(b"\x3b")

    with Image.open(io.BytesIO(raw)) as orig_im, Image.open(io.BytesIO(cleaned)) as clean_im:
        assert orig_im.n_frames == clean_im.n_frames == 2
        assert orig_im.tobytes() == clean_im.tobytes()
        assert "comment" not in clean_im.info


def test_strip_image_metadata_fails_closed_on_malformed_or_unsupported_input():
    with pytest.raises(ImageMetadataStripError):
        strip_image_metadata(b"not a png", "image/png")
    with pytest.raises(ImageMetadataStripError):
        strip_image_metadata(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR", "image/png")
    with pytest.raises(ImageMetadataStripError):
        strip_image_metadata(b"\xff\xd8\xff\xfe\x00\x04ab", "image/jpeg")
    with pytest.raises(ImageMetadataStripError):
        strip_image_metadata(b"RIFF\x04\x00\x00\x00WEBP", "image/webp")
    with pytest.raises(ImageMetadataStripError):
        strip_image_metadata(b"ftypmp42", "video/mp4")


def test_studio_asset_download_strip_metadata_route(gpu_client):  # noqa: F811
    client, _store, _user, project, _headers = gpu_client
    png_raw = _make_png_with_workflow_and_color()
    up_img = client.post(
        "/api/assets",
        data={"project_id": project["id"]},
        files={"file": ("角色立绘_0001_.png", png_raw, "image/png")},
    )
    assert up_img.status_code == 201, up_img.text
    asset = up_img.json()["data"]

    # 1. Original download preserves metadata, original filename, and base ETag.
    r_orig = client.get(f"/api/assets/{asset['id']}/download")
    assert r_orig.status_code == 200
    assert r_orig.content == png_raw
    orig_etag = r_orig.headers["etag"]
    assert not orig_etag.endswith('-clean"')

    # 2. Clean download strips metadata, adds _clean suffix, and uses a distinct ETag.
    r_clean = client.get(f"/api/assets/{asset['id']}/download", params={"strip_metadata": "true"})
    assert r_clean.status_code == 200, r_clean.text
    assert r_clean.headers["content-type"] == "image/png"
    clean_etag = r_clean.headers["etag"]
    assert clean_etag != orig_etag
    assert clean_etag.endswith('-clean"')
    disp = unquote(r_clean.headers["content-disposition"])
    assert disp.startswith("attachment;")
    assert "角色立绘_0001_clean.png" in disp
    assert b"KSampler" not in r_clean.content
    assert b"secret" not in r_clean.content
    assert b"tEXt" not in _png_chunk_types(r_clean.content)

    # 3. Conditional 304 respects clean ETag and never collides with original ETag.
    r_304 = client.get(
        f"/api/assets/{asset['id']}/download",
        params={"strip_metadata": "true"},
        headers={"If-None-Match": clean_etag},
    )
    assert r_304.status_code == 304
    r_miss = client.get(
        f"/api/assets/{asset['id']}/download",
        params={"strip_metadata": "true"},
        headers={"If-None-Match": orig_etag},
    )
    assert r_miss.status_code == 200

    # 4. Stored file on disk remains unmodified (original metadata still intact).
    r_after = client.get(f"/api/assets/{asset['id']}/download")
    assert r_after.status_code == 200
    assert r_after.content == png_raw

    # 5. Non-image asset (MP4 video) rejects strip_metadata with 422 instead of returning unstripped data.
    mp4_raw = b"\x00\x00\x00\x18ftypmp42" + b"0" * 64
    up_vid = client.post(
        "/api/assets",
        data={"project_id": project["id"]},
        files={"file": ("clip.mp4", mp4_raw, "video/mp4")},
    )
    assert up_vid.status_code == 201, up_vid.text
    vid_asset = up_vid.json()["data"]
    r_vid_clean = client.get(f"/api/assets/{vid_asset['id']}/download", params={"strip_metadata": "true"})
    assert r_vid_clean.status_code == 422
    assert r_vid_clean.json()["error"]["code"] == "STRIP_METADATA_UNSUPPORTED"


def test_native_drive_output_download_strip_metadata_route(studio_client, monkeypatch):  # noqa: F811
    client, store, headers = studio_client
    repo = Repository(store)
    png_raw = _make_png_with_workflow_and_color()
    webp_raw = _make_webp_with_metadata(alpha=True)
    mp4_raw = b"\x00\x00\x00\x18ftypmp42" + b"0" * 64

    tree = {
        "root": [{"id": "f-root", "name": "AI-Studio", "mimeType": FOLDER_MIME}],
        "f-root": [{"id": "f-owner", "name": "owner", "mimeType": FOLDER_MIME}],
        "f-owner": [{"id": "f-native", "name": "native-comfy", "mimeType": FOLDER_MIME}],
        "f-native": [{"id": "f-out", "name": "output", "mimeType": FOLDER_MIME}],
        "f-out": [
            {
                "id": "img-png",
                "name": "ComfyUI_00001_.png",
                "mimeType": "image/png",
                "size": str(len(png_raw)),
                "md5Checksum": "md5png01",
                "modifiedTime": "2026-11-25T12:00:00Z",
                "imageMediaMetadata": {"width": 16, "height": 12},
            },
            {
                "id": "img-webp",
                "name": "preview.webp",
                "mimeType": "image/webp",
                "size": str(len(webp_raw)),
                "md5Checksum": "md5webp01",
                "modifiedTime": "2026-11-25T12:01:00Z",
                "imageMediaMetadata": {"width": 16, "height": 16},
            },
            {
                "id": "vid-mp4",
                "name": "motion.mp4",
                "mimeType": "video/mp4",
                "size": str(len(mp4_raw)),
                "md5Checksum": "md5mp401",
                "modifiedTime": "2026-11-25T12:02:00Z",
                "videoMediaMetadata": {"width": 320, "height": 240},
            },
        ],
    }
    mock_drive, base_handler = _build_mock_drive(
        repo,
        tree,
        {"img-png": png_raw, "img-webp": webp_raw, "vid-mp4": mp4_raw},
        [],
    )

    media_fetches: list[str] = []

    def counting_handler(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("alt") == "media":
            media_fetches.append(request.url.path)
        return base_handler(request)

    monkeypatch.setattr(
        native_outputs,
        "_open_drive",
        lambda r, owner, drive=None: (mock_drive, "acc-drive", "主盘", False),
    )
    orig_async_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda *a, **kw: orig_async_client(*a, transport=httpx.MockTransport(counting_handler), **kw),
    )

    # 1. Original Drive download keeps workflow metadata untouched.
    r_orig = client.get("/api/gpu/native/outputs/img-png/content?download=true", headers=headers)
    assert r_orig.status_code == 200
    assert r_orig.content == png_raw
    assert 'filename="ComfyUI_00001_.png"' in r_orig.headers["content-disposition"]

    # 2. Clean Drive PNG download strips metadata, renames to _clean, and returns clean ETag.
    r_clean = client.get(
        "/api/gpu/native/outputs/img-png/content?download=true&strip_metadata=true",
        headers=headers,
    )
    assert r_clean.status_code == 200, r_clean.text
    assert r_clean.headers["content-type"] == "image/png"
    assert r_clean.headers["content-disposition"].startswith("attachment;")
    assert "ComfyUI_00001_clean.png" in unquote(r_clean.headers["content-disposition"])
    assert b"KSampler" not in r_clean.content
    assert b"secret" not in r_clean.content
    clean_etag = r_clean.headers["etag"]
    assert "native-clean-" in clean_etag

    # 3. Conditional 304 short-circuits before fetching media bytes from Drive again.
    fetches_before_304 = len(media_fetches)
    r_304 = client.get(
        "/api/gpu/native/outputs/img-png/content?download=true&strip_metadata=true",
        headers={**headers, "If-None-Match": clean_etag},
    )
    assert r_304.status_code == 304
    assert len(media_fetches) == fetches_before_304

    # 4. Clean Drive WebP download strips EXIF/XMP and renames cleanly.
    r_webp = client.get(
        "/api/gpu/native/outputs/img-webp/content?download=true&strip_metadata=true",
        headers=headers,
    )
    assert r_webp.status_code == 200
    assert "preview_clean.webp" in unquote(r_webp.headers["content-disposition"])
    assert b"secret-webp-workflow" not in r_webp.content
    assert b"secret-webp-xmp" not in r_webp.content

    # 5. Native video rejects strip_metadata with 422.
    r_vid = client.get(
        "/api/gpu/native/outputs/vid-mp4/content?download=true&strip_metadata=true",
        headers=headers,
    )
    assert r_vid.status_code == 422
    assert r_vid.json()["error"]["code"] == "STRIP_METADATA_UNSUPPORTED"
