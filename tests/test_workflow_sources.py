"""Unit and HTTP tests for remote workflow URL fetching, PNG extraction, and unified import."""
from __future__ import annotations

import json
import struct
import zlib

import httpx
import pytest
from app.common import current_user
from app.config import settings
from app.db import Store
from app.gpu.model_sources import ModelError, SourceHTTP
from app.gpu.repository import Repository
from app.gpu.workflow_sources import (
    extract_png_workflow,
    fetch_workflow_source,
    normalize_workflow_url,
)
from app.main import app
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    body = kind + payload
    crc = zlib.crc32(body) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", crc)


def _png_text(key: str, value, kind: bytes = b"tEXt", packed: bool = True) -> bytes:
    raw = value.encode("utf-8") if isinstance(value, str) else json.dumps(value).encode("utf-8")
    if kind == b"zTXt":
        payload = b"\x00" + zlib.compress(raw)
    elif kind == b"iTXt":
        payload = bytes([1 if packed else 0, 0, 0, 0]) + (zlib.compress(raw) if packed else raw)
    else:
        payload = raw
    return _png_chunk(kind, key.encode("latin-1") + b"\x00" + payload)


def _make_png(*text_chunks: bytes) -> bytes:
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00\x00\x00\x00")
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + b"".join(text_chunks)
        + _png_chunk(b"IDAT", idat)
        + _png_chunk(b"IEND", b"")
    )


SAMPLE_VISUAL = {
    "nodes": [
        {
            "id": 1,
            "type": "UNETLoader",
            "widgets_values": ["flux_dev_fp8.safetensors", "fp8_e4m3fn"],
            "properties": {
                "models": [
                    {
                        "name": "flux_dev_fp8.safetensors",
                        "url": "https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux_dev_fp8.safetensors",
                        "directory": "diffusion_models",
                    }
                ]
            },
        }
    ],
    "links": [],
}

SAMPLE_PROMPT = {
    "1": {
        "class_type": "UNETLoader",
        "inputs": {"unet_name": "flux_dev_fp8.safetensors", "weight_dtype": "fp8_e4m3fn"},
    }
}

SAMPLE_STUDIO_BUNDLE = {
    "workflow_key": "remote-demo",
    "model_id": "flux-dev",
    "manifest": {
        "workflow_key": "remote-demo",
        "model_id": "flux-dev",
        "version": "1.0.0",
        "compute_spec": "l4",
        "parameters": {},
        "dependencies": {
            "comfyui_commit": "0" * 40,
            "weight_files": [],
        },
    },
    "api_graph": SAMPLE_PROMPT,
    "object_info": {"UNETLoader": {"input": {"required": {"unet_name": [["flux_dev_fp8.safetensors"]]}}}},
}


def _mock_http(handler, *, resolver=None, tokens=None):
    return SourceHTTP(
        tokens or {},
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        resolver=resolver or (lambda _h: ["93.184.216.34"]),
    )


def test_normalize_workflow_url_supports_direct_signed_and_platform_conversions():
    # Direct HTTPS with temporary signed query params and fragment stripped
    url, fallback, note = normalize_workflow_url(
        "  https://cdn.discordapp.com/attachments/1/2/sample%20img.png?ex=6789&hm=deadbeef#top  "
    )
    assert url == "https://cdn.discordapp.com/attachments/1/2/sample%20img.png?ex=6789&hm=deadbeef"
    assert fallback is None
    assert note is None

    # GitHub blob -> raw.githubusercontent.com
    gh, gh_fb, gh_note = normalize_workflow_url(
        "https://github.com/comfyanonymous/ComfyUI_examples/blob/master/flux/flux_dev.png#L1"
    )
    assert gh == "https://raw.githubusercontent.com/comfyanonymous/ComfyUI_examples/master/flux/flux_dev.png"
    assert gh_fb is None
    assert gh_note and "GitHub" in gh_note

    # Hugging Face blob -> resolve (models & datasets)
    hf, _, hf_note = normalize_workflow_url("https://huggingface.co/org/repo/blob/main/workflows/a.json")
    assert hf == "https://huggingface.co/org/repo/resolve/main/workflows/a.json"
    assert hf_note and "Hugging Face" in hf_note

    hf_ds, _, _ = normalize_workflow_url("https://huggingface.co/datasets/org/ds/blob/main/demo.png")
    assert hf_ds == "https://huggingface.co/datasets/org/ds/resolve/main/demo.png"

    # Civitai image CDN resized -> original=true with fallback
    civ, civ_fb, civ_note = normalize_workflow_url(
        "https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/uuid-1/width=450,optimized=true/0001.jpeg"
    )
    assert civ == "https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/uuid-1/original=true/0001.jpeg"
    assert civ_fb == "https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/uuid-1/width=450,optimized=true/0001.jpeg"
    assert civ_note and "Civitai" in civ_note

    # Civitai already original=true -> untouched
    civ_orig, civ_orig_fb, civ_orig_note = normalize_workflow_url(
        "https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/uuid-1/original=true/0001.png"
    )
    assert civ_orig == "https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/uuid-1/original=true/0001.png"
    assert civ_orig_fb is None
    assert civ_orig_note is None

    # Rejected schemes / ports / userinfo
    for bad in [
        "",
        "http://example.com/wf.json",
        "https://user:pass@example.com/wf.json",
        "https://example.com:8443/wf.json",
        "ftp://example.com/wf.json",
    ]:
        with pytest.raises(ModelError):
            normalize_workflow_url(bad)


def test_extract_png_workflow_parity_with_frontend_parser():
    # tEXt, zTXt, compressed and uncompressed iTXt
    for kind, packed in [(b"tEXt", False), (b"zTXt", True), (b"iTXt", True), (b"iTXt", False)]:
        png_bytes = _make_png(_png_text("workflow", SAMPLE_VISUAL, kind, packed), _png_text("prompt", SAMPLE_PROMPT, kind, packed))
        out = extract_png_workflow(png_bytes)
        assert out["workflow"] == SAMPLE_VISUAL
        assert out["prompt"] == SAMPLE_PROMPT
        assert out["node_count"] == 1
        assert out["node_types"] == ["UNETLoader"]
        assert out["model_files"] == ["flux_dev_fp8.safetensors"]
        assert out["warnings"] == []

    # Workflow-only warns about missing API prompt
    wf_only = extract_png_workflow(_make_png(_png_text("workflow", SAMPLE_VISUAL)))
    assert "prompt" not in wf_only
    assert any("API" in w for w in wf_only["warnings"])

    # Invalid workflow does not discard valid prompt
    partial = extract_png_workflow(_make_png(_png_text("workflow", "{"), _png_text("prompt", SAMPLE_PROMPT)))
    assert partial["prompt"] == SAMPLE_PROMPT
    assert any("workflow 无效" in w for w in partial["warnings"])

    # A1111 parameters and empty PNG errors
    with pytest.raises(ModelError, match="A1111"):
        extract_png_workflow(_make_png(_png_text("parameters", "Steps: 20")))
    with pytest.raises(ModelError, match="没有找到 ComfyUI 元数据"):
        extract_png_workflow(_make_png())

    # Corrupted CRC
    corrupt = bytearray(_make_png(_png_text("prompt", SAMPLE_PROMPT)))
    corrupt[45] ^= 1
    with pytest.raises(ModelError, match="校验"):
        extract_png_workflow(bytes(corrupt))

    # Duplicate metadata
    with pytest.raises(ModelError, match="重复"):
        extract_png_workflow(_make_png(_png_text("prompt", SAMPLE_PROMPT), _png_text("prompt", SAMPLE_PROMPT)))

    # Decompression bomb > 4 MiB
    bomb_png = _make_png(_png_text("workflow", "x" * (4 * 1024 * 1024 + 100), b"zTXt"))
    with pytest.raises(ModelError, match="4 MiB"):
        extract_png_workflow(bomb_png)

    # Valid Latin-1 bytes undefined in Python cp1252 (0x81, 0x8d, 0x8f, 0x90, 0x9d) in unrelated text chunks
    # must NOT abort extraction of a valid ComfyUI prompt.
    latin1_author = _png_chunk(b"tEXt", b"Author\x00A\x81\x8d\x8f\x90\x9dB")
    latin1_png = _make_png(latin1_author, _png_text("prompt", SAMPLE_PROMPT))
    out_latin1 = extract_png_workflow(latin1_png)
    assert out_latin1["prompt"] == SAMPLE_PROMPT
    author_entry = next(m for m in out_latin1["metadata"] if m["keyword"] == "Author")
    assert author_entry["text"] == "A\x81\x8d\x8f\x90\x9dB"


def test_fetch_workflow_source_handles_png_json_civitai_fallback_and_rejections():
    png_bytes = _make_png(_png_text("workflow", SAMPLE_VISUAL), _png_text("prompt", SAMPLE_PROMPT))
    seen_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_paths.append(f"{request.headers.get('host')}{request.url.path}")
        if request.url.path.endswith("/original=true/0001.jpeg"):
            return httpx.Response(200, content=png_bytes, headers={"content-type": "image/png"})
        if request.url.path.endswith("/fallback-orig/original=true/0002.png"):
            return httpx.Response(404)
        if request.url.path.endswith("/fallback-orig/width=450/0002.png"):
            return httpx.Response(200, content=png_bytes, headers={"content-type": "image/png"})
        if request.url.path.endswith("/html"):
            return httpx.Response(200, text="<!DOCTYPE html><html><body>page</body></html>", headers={"content-type": "text/html"})
        if request.url.path.endswith("/jpg"):
            return httpx.Response(200, content=b"\xff\xd8\xff\xe0" + b"\x00" * 64, headers={"content-type": "image/jpeg"})
        if request.url.path.endswith("/bin"):
            return httpx.Response(200, content=b"PK\x03\x04" + b"\x00" * 64, headers={"content-type": "application/octet-stream"})
        if request.url.path.endswith("/bom.json"):
            return httpx.Response(
                200,
                content=b"\xef\xbb\xbf" + json.dumps(SAMPLE_VISUAL).encode("utf-8"),
                headers={"content-type": "application/json", "content-disposition": 'attachment; filename="my-wf.json"'},
            )
        return httpx.Response(404)

    http = _mock_http(handler)
    try:
        # Civitai thumbnail automatically switched to original=true and preserves PNG extraction
        res = fetch_workflow_source(
            http,
            "https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/uuid-1/width=450/0001.jpeg?sig=secret-signature",
        )
        assert res["kind"] == "png"
        assert res["source_host"] == "image.civitai.com"
        assert res["source_name"] == "0001.jpeg"
        assert "secret-signature" not in json.dumps(res)
        assert res["note"] and "Civitai" in res["note"]
        assert res["image_extraction"]["prompt"] == SAMPLE_PROMPT

        # Civitai original=true 404 falls back to requested URL cleanly
        res_fb = fetch_workflow_source(
            http,
            "https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/fallback-orig/width=450/0002.png",
        )
        assert res_fb["kind"] == "png"
        assert "note" not in res_fb

        # JSON with UTF-8 BOM + Content-Disposition filename
        res_json = fetch_workflow_source(http, "https://example.com/bom.json?token=temp123")
        assert res_json["kind"] == "json"
        assert res_json["source_name"] == "my-wf.json"
        assert res_json["payload"] == SAMPLE_VISUAL
        assert "temp123" not in json.dumps(res_json)

        # Rejections
        with pytest.raises(ModelError, match="HTML"):
            fetch_workflow_source(http, "https://example.com/html")
        with pytest.raises(ModelError, match="仅支持 PNG 原图或工作流 JSON"):
            fetch_workflow_source(http, "https://example.com/jpg")
        with pytest.raises(ModelError, match="既不是 PNG 原图"):
            fetch_workflow_source(http, "https://example.com/bin")
    finally:
        http.close()


def test_fetch_workflow_source_blocks_ssrf_private_ips(monkeypatch):
    # Real public_addresses resolver rejects loopback and private IPs
    http = SourceHTTP({}, client=httpx.Client(transport=httpx.MockTransport(lambda _r: httpx.Response(200))))
    monkeypatch.setattr(
        "app.gpu.model_sources.socket.getaddrinfo",
        lambda host, port, type=0: [(2, 1, 6, "", ("127.0.0.1", 443))],
    )
    try:
        with pytest.raises(ModelError, match="内部、保留或本地网络"):
            fetch_workflow_source(http, "https://internal.example/workflow.json")
    finally:
        http.close()


@pytest.fixture
def http_client(monkeypatch):
    monkeypatch.setattr(settings, "encryption_key", Fernet.generate_key().decode())
    st = Store("sqlite:///:memory:")
    app.state.store = st
    app.dependency_overrides[current_user] = lambda: {"id": "wf-url-owner", "role": "admin"}
    with TestClient(app) as client:
        yield client, Repository(st)
    app.dependency_overrides.clear()


def test_http_model_library_inspect_workflow_from_url_json_and_png(http_client, monkeypatch):
    c, _repo = http_client
    png_bytes = _make_png(_png_text("workflow", SAMPLE_VISUAL), _png_text("prompt", SAMPLE_PROMPT))

    def handler(request: httpx.Request) -> httpx.Response:
        if request.headers.get("host") == "raw.githubusercontent.com" and request.url.path == "/org/repo/main/wf.json":
            return httpx.Response(200, json=SAMPLE_VISUAL)
        if request.headers.get("host") == "cdn.example.com" and request.url.path == "/share/demo.png":
            return httpx.Response(200, content=png_bytes, headers={"content-type": "image/png"})
        return httpx.Response(404)

    monkeypatch.setattr(
        "app.gpu.model_library.SourceHTTP",
        lambda tokens=None: _mock_http(handler, tokens=tokens),
    )

    before_wfs = c.get("/api/gpu/workflows").json()["data"]

    # Non-admin user can still inspect remote workflow URLs in model library
    app.dependency_overrides[current_user] = lambda: {"id": "wf-url-owner", "role": "user"}

    resp_json = c.post(
        "/api/gpu/model-library/inspect-workflow",
        json={"url": "https://github.com/org/repo/blob/main/wf.json"},
    )
    assert resp_json.status_code == 200, resp_json.text
    data_json = resp_json.json()["data"]
    assert data_json["format"] == "visual"
    assert data_json["source_name"] == "wf.json"
    assert data_json["source_host"] == "raw.githubusercontent.com"
    assert "GitHub" in (data_json.get("source_note") or "")
    assert data_json["dependencies"][0]["reference"] == "flux_dev_fp8.safetensors"

    resp_png = c.post(
        "/api/gpu/model-library/inspect-workflow",
        json={"url": "https://cdn.example.com/share/demo.png?ex=abc&hm=123"},
    )
    assert resp_png.status_code == 200, resp_png.text
    data_png = resp_png.json()["data"]
    assert data_png["format"] == "both"
    assert data_png["source_name"] == "demo.png"
    assert data_png["source_host"] == "cdn.example.com"
    assert data_png["dependencies"][0]["reference"] == "flux_dev_fp8.safetensors"
    assert "hm=123" not in resp_png.text

    # Confirm zero workflow drafts were created
    app.dependency_overrides[current_user] = lambda: {"id": "wf-url-owner", "role": "admin"}
    after_wfs = c.get("/api/gpu/workflows").json()["data"]
    assert len(after_wfs) == len(before_wfs)


def test_http_workflows_fetch_url_conservative_policy_and_png_roundtrip(http_client, monkeypatch):
    c, _repo = http_client
    png_bytes = _make_png(_png_text("workflow", SAMPLE_VISUAL), _png_text("prompt", SAMPLE_PROMPT))
    latin1_png_bytes = _make_png(
        _png_chunk(b"tEXt", b"Author\x00A\x81\x8d\x8f\x90\x9dB"),
        _png_text("prompt", SAMPLE_PROMPT),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/bundle.json":
            return httpx.Response(200, json=SAMPLE_STUDIO_BUNDLE)
        if request.url.path == "/plain.json":
            return httpx.Response(200, json=SAMPLE_VISUAL)
        if request.url.path == "/example.png":
            return httpx.Response(200, content=png_bytes, headers={"content-type": "image/png"})
        if request.url.path == "/latin1.png":
            return httpx.Response(200, content=latin1_png_bytes, headers={"content-type": "image/png"})
        return httpx.Response(404)

    monkeypatch.setattr(
        "app.gpu.model_library.SourceHTTP",
        lambda tokens=None: _mock_http(handler, tokens=tokens),
    )

    # 1. Remote Studio bundle JSON -> returned ready to import
    r_bundle = c.post("/api/gpu/workflows/fetch-url", json={"url": "https://files.example.com/bundle.json"})
    assert r_bundle.status_code == 200, r_bundle.text
    d_bundle = r_bundle.json()["data"]
    assert d_bundle["kind"] == "bundle"
    assert d_bundle["bundle"]["workflow_key"] == "remote-demo"

    # 2. Remote plain ComfyUI JSON -> conservatively rejected with UNSUPPORTED_WORKFLOW_FORMAT
    r_plain = c.post("/api/gpu/workflows/fetch-url", json={"url": "https://files.example.com/plain.json"})
    assert r_plain.status_code == 422, r_plain.text
    err = r_plain.json()["error"]
    assert err["code"] == "UNSUPPORTED_WORKFLOW_FORMAT"
    assert "模型库" in err["message"]

    # 3. Remote PNG -> returns extraction summary, then imports cleanly as pending draft
    r_png = c.post("/api/gpu/workflows/fetch-url", json={"url": "https://files.example.com/example.png?sig=temp"})
    assert r_png.status_code == 200, r_png.text
    d_png = r_png.json()["data"]
    assert d_png["kind"] == "png"
    assert d_png["source_name"] == "example.png"
    assert d_png["source_host"] == "files.example.com"
    assert "sig=temp" not in r_png.text
    extraction = d_png["extraction"]
    assert extraction["workflow"] == SAMPLE_VISUAL
    assert extraction["prompt"] == SAMPLE_PROMPT

    imported = c.post(
        "/api/gpu/workflows/import",
        json={
            "workflow_key": "url-png-demo",
            "image_extraction": {
                "workflow": extraction["workflow"],
                "prompt": extraction["prompt"],
                "metadata": extraction["metadata"],
            },
        },
    )
    assert imported.status_code == 201, imported.text
    assert imported.json()["data"]["state"] == "draft"

    # 3b. Remote PNG with non-CP1252 Latin-1 Author metadata still returns 200 and extracts prompt cleanly
    r_latin1 = c.post("/api/gpu/workflows/fetch-url", json={"url": "https://files.example.com/latin1.png"})
    assert r_latin1.status_code == 200, r_latin1.text
    assert r_latin1.json()["data"]["extraction"]["prompt"] == SAMPLE_PROMPT

    # 4. Non-admin rejected with 403
    app.dependency_overrides[current_user] = lambda: {"id": "wf-url-owner", "role": "user"}
    r_forbidden = c.post("/api/gpu/workflows/fetch-url", json={"url": "https://files.example.com/bundle.json"})
    assert r_forbidden.status_code == 403


def test_non_finite_floats_in_png_and_remote_json_are_sanitized_for_json_response(http_client, monkeypatch):
    from app.gpu import model_dependencies as deps

    c, _repo = http_client

    raw_visual = (
        '{"nodes":['
        '{"id":1,"type":"UNETLoader","pos":[NaN,-Infinity],"size":[1e500,200],'
        '"widgets_values":["flux_dev_fp8.safetensors","fp8_e4m3fn"],"title":"keep NaN Infinity text"},'
        '{"id":2,"type":"LoraLoader","widgets_values":["style.safetensors",NaN,Infinity]}'
        '],"links":[]}'
    )
    raw_prompt = (
        '{"1":{"class_type":"UNETLoader","inputs":{"unet_name":"flux_dev_fp8.safetensors","weight_dtype":"fp8_e4m3fn","note":"literal NaN"}},'
        '"2":{"class_type":"LoraLoader","inputs":{"lora_name":"style.safetensors","strength_model":NaN,"strength_clip":-Infinity}},'
        '"3":{"class_type":"Power Lora Loader (rgthree)","inputs":{"lora_1":{"on":true,"lora":"detail.safetensors","strength":NaN,"strengthTwo":0.5}}}}'
    )
    nan_png = _make_png(_png_text("workflow", raw_visual), _png_text("prompt", raw_prompt))

    # 1. Pure-Python PNG extraction replaces NaN/Infinity/1e500 with None while keeping strings intact
    extracted = extract_png_workflow(nan_png)
    assert extracted["workflow"]["nodes"][0]["pos"] == [None, None]
    assert extracted["workflow"]["nodes"][0]["size"] == [None, 200]
    assert extracted["workflow"]["nodes"][0]["title"] == "keep NaN Infinity text"
    assert extracted["prompt"]["1"]["inputs"]["note"] == "literal NaN"
    assert extracted["prompt"]["2"]["inputs"]["strength_model"] is None
    assert extracted["prompt"]["2"]["inputs"]["strength_clip"] is None
    assert extracted["prompt"]["3"]["inputs"]["lora_1"]["strength"] is None
    assert extracted["prompt"]["3"]["inputs"]["lora_1"]["strengthTwo"] == 0.5
    assert any("workflow 包含非标准数值" in w for w in extracted["warnings"])
    assert any("prompt 包含非标准数值" in w for w in extracted["warnings"])
    json.dumps(extracted, allow_nan=False)

    # 2. Direct scan() on unsanitized in-memory Python floats still never emits non-finite strengths
    direct_scanned = deps.scan({
        "api_graph": {
            "2": {
                "class_type": "LoraLoader",
                "inputs": {"lora_name": "style.safetensors", "strength_model": float("nan"), "strength_clip": float("inf")},
            },
            "3": {
                "class_type": "Power Lora Loader (rgthree)",
                "inputs": {"lora_1": {"on": True, "lora": "detail.safetensors", "strength": float("nan"), "strengthTwo": 0.5}},
            },
        }
    })
    json.dumps(direct_scanned, allow_nan=False)
    by_node = {d["node_id"]: d for d in direct_scanned}
    assert by_node["2"]["strengths"] == {}
    assert by_node["2"]["required"] is True
    assert by_node["3"]["strengths"] == {"lora_1.strengthTwo": 0.5}
    assert by_node["3"]["required"] is True

    # 3. HTTP endpoints return 200 JSONResponse (allow_nan=False) for both remote PNG and remote JSON
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/nan.png":
            return httpx.Response(200, content=nan_png, headers={"content-type": "image/png"})
        if request.url.path == "/nan.json":
            return httpx.Response(200, content=raw_visual.encode("utf-8"), headers={"content-type": "application/json"})
        return httpx.Response(404)

    monkeypatch.setattr(
        "app.gpu.model_library.SourceHTTP",
        lambda tokens=None: _mock_http(handler, tokens=tokens),
    )

    r_fetch = c.post("/api/gpu/workflows/fetch-url", json={"url": "https://cdn.example.com/nan.png"})
    assert r_fetch.status_code == 200, r_fetch.text
    fetch_ext = r_fetch.json()["data"]["extraction"]
    assert fetch_ext["workflow"]["nodes"][0]["pos"] == [None, None]
    assert fetch_ext["prompt"]["2"]["inputs"]["strength_model"] is None

    r_import = c.post(
        "/api/gpu/workflows/import",
        json={
            "workflow_key": "nan-png-demo",
            "image_extraction": {
                "workflow": fetch_ext["workflow"],
                "prompt": fetch_ext["prompt"],
                "metadata": fetch_ext["metadata"],
            },
        },
    )
    assert r_import.status_code == 201, r_import.text
    assert r_import.json()["data"]["state"] == "draft"

    r_inspect_png = c.post("/api/gpu/model-library/inspect-workflow", json={"url": "https://cdn.example.com/nan.png"})
    assert r_inspect_png.status_code == 200, r_inspect_png.text
    inspect_png = r_inspect_png.json()["data"]
    refs_png = {d["reference"] for d in inspect_png["dependencies"]}
    assert {"flux_dev_fp8.safetensors", "style.safetensors", "detail.safetensors"}.issubset(refs_png)
    assert any("非标准数值" in w for w in inspect_png["warnings"])

    r_inspect_json = c.post("/api/gpu/model-library/inspect-workflow", json={"url": "https://cdn.example.com/nan.json"})
    assert r_inspect_json.status_code == 200, r_inspect_json.text
    inspect_json = r_inspect_json.json()["data"]
    assert any(d["reference"] == "flux_dev_fp8.safetensors" for d in inspect_json["dependencies"])
    assert any("非标准数值" in w for w in inspect_json["warnings"])
