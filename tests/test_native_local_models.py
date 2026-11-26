"""Exercise local preparation with in-process Comfy routes and simulated model sources."""
import asyncio
import fcntl
import hashlib
import io
import json
import os
import shutil
import sys
import threading
import zipfile
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from app.config import settings
from app.gpu import model_sources, native, native_models
from app.gpu.native_runtime import EXTENSION

DATA = b"model bytes for integrity tests"
SHA = hashlib.sha256(DATA).hexdigest()
PATH = "subdir/base.safetensors"
EXTERNAL = "https://huggingface.co/owner/repo/resolve/commit/base.safetensors"


def graph(name=PATH):
    return {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": name}},
            "2": {"class_type": "PreviewImage", "inputs": {"images": ["1", 0]}}}


def row(**patch):
    return {"model_type": "checkpoints", "filename": "base.safetensors", "relative_path": PATH,
            "sha256": SHA, "size": len(DATA), "drive_file_id": "known-file-id", "account": "drive-account",
            "source": {"download_url": EXTERNAL, "status": "verified"}, **patch}


class Stream(httpx.SyncByteStream):
    def __init__(self, body):
        self.body = body

    def __iter__(self):
        yield self.body


def media(body=DATA):
    return httpx.Response(200, headers={"content-length": str(len(body)), "etag": '"fixed"'}, stream=Stream(body))


@pytest.fixture
def local(tmp_path, monkeypatch):
    monkeypatch.setattr(model_sources, "public_addresses", lambda host: ["8.8.8.8"])
    # SourceHTTP captures its default resolver at import.
    original = native_models.SourceHTTP
    monkeypatch.setattr(native_models, "SourceHTTP",
                        lambda *args, **kw: original(*args, **kw, resolver=model_sources.public_addresses))
    monkeypatch.setattr(native_models, "RESERVE", 1)
    monkeypatch.setattr(native_models.time, "sleep", lambda _: None)
    instances = []

    def create(handler, catalog=(), **config):
        models = native_models.LocalModels(
            tmp_path, tmp_path / "models", {"catalog": list(catalog), "channel": "internal-channel", **config},
            http=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        instances.append(models)
        return models

    yield create
    for instance in instances:
        instance.executor.shutdown()
        instance.client.close()


@pytest.mark.asyncio
async def test_local_hit_uses_registered_hash_and_skips_both_remotes(local, tmp_path):
    model = local(lambda _: pytest.fail("local hit contacted a remote source"), [row()], models_folder="folder")
    target = tmp_path / "models/checkpoints" / PATH
    target.parent.mkdir(parents=True)
    target.write_bytes(DATA)
    for _ in range(2):
        result = await model.submit({"prompt": graph()})[1]
        assert result[0]["source"] == "local" and result[0]["cache_hit"] is True
    target.write_bytes(b"wrong")
    result = await model.submit({"prompt": graph()})[1]
    assert result[0]["state"] == "failed" and "身份" in result[0]["error"]
    assert target.read_bytes() == b"wrong"


@pytest.mark.asyncio
@pytest.mark.parametrize("external_ok", [True, False])
async def test_external_priority_and_hash_checked_drive_fallback(local, tmp_path, external_ok):
    calls = []

    def handler(req):
        host = req.headers["host"]
        calls.append(host)
        if host == "huggingface.co":
            assert req.headers["authorization"] == "Bearer model-token"
            return media() if external_ok else httpx.Response(403)
        if host == "oauth2.googleapis.com":
            return httpx.Response(200, json={"access_token": "drive-token", "expires_in": 3600})
        assert host == "www.googleapis.com"
        assert req.headers["authorization"] == "Bearer drive-token"
        assert req.url.path.endswith("/known-file-id")
        return media()

    model = local(handler, [row()], drive_auth={"drive-account": {}}, tokens={"huggingface": "model-token"})
    result = (await model.submit({"prompt": graph()})[1])[0]
    assert result["state"] == "ready"
    if external_ok:
        assert calls == ["huggingface.co"] and result["source"] == "external"
    else:
        assert calls[-2:] == ["oauth2.googleapis.com", "www.googleapis.com"]
        assert result["source"] == "drive" and "403" in result["fallback"]
    assert "model-token" not in repr(result) and "drive-token" not in repr(result)
    assert (tmp_path / "models/checkpoints" / PATH).read_bytes() == DATA


@pytest.mark.asyncio
@pytest.mark.parametrize("host", ["civitai.com", "civitai.red"])
@pytest.mark.parametrize("external_ok", [True, False])
async def test_catalogued_civitai_upscale_uses_external_source_or_drive(local, tmp_path, host, external_ok):
    name = "example.pth"
    url = f"https://{host}/api/download/models/123"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("archive/data.pkl", b"\x80\x02ctorch\nFloatStorage\n.")
    weight = buffer.getvalue()
    model = {
        "filename": name, "model_type": "upscale_models", "source": "civitai",
        "download_url": url, "source_verification": "verified", "state": "ready",
        "sha256": hashlib.sha256(weight).hexdigest(), "size": len(weight),
        "drive_file_id": "known-file-id", "drive_account_id": "drive-account",
    }
    catalog = native.local_catalog(SimpleNamespace(list=lambda *_: [model]), "owner")
    assert catalog[0]["source"]["kind"] == "civitai"

    def handler(req):
        req_host = req.headers["host"]
        if req_host == host:
            assert req.headers["authorization"] == "Bearer civ-key"
            return media(weight) if external_ok else httpx.Response(403)
        if req_host == "oauth2.googleapis.com":
            return httpx.Response(200, json={"access_token": "drive-token", "expires_in": 3600})
        assert req_host == "www.googleapis.com"
        return media(weight)

    models = local(handler, catalog, drive_auth={"drive-account": {}}, tokens={"civitai": "civ-key"})
    prompt = {"1": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": name}}}
    [result] = await models.submit({"prompt": prompt})[1]
    assert result["state"] == "ready"
    assert result["source"] == ("external" if external_ok else "drive")
    assert (tmp_path / "models/upscale_models" / name).read_bytes() == weight


@pytest.mark.asyncio
async def test_missing_ambiguous_and_distinct_subdirectory(local):
    models = local(lambda _: pytest.fail("an ambiguous or missing model triggered a download"), [
        row(), row(sha256="a" * 64, drive_file_id="another-id"),
        row(relative_path="different/base.safetensors", sha256="b" * 64),
        row(model_type="loras", sha256="c" * 64),
    ])
    problem = (await models.submit({"prompt": graph()})[1])[0]
    assert problem["state"] == "failed" and "多个版本" in problem["error"]
    problem = (await models.submit({"prompt": graph("base.safetensors")})[1])[0]
    assert problem["state"] == "failed" and "找不到明确来源" in problem["error"]


@pytest.mark.asyncio
async def test_verified_binding_resolves_workflow_link_conflict(local):
    hosts = []

    def handler(req):
        hosts.append(req.headers["Host"])
        return media()

    models = local(handler, [row()])
    body = {"prompt": graph(), "extra_data": {"extra_pnginfo": {"workflow": {
        "models": [{"name": PATH, "directory": "checkpoints", "url": "https://other.example/m",
                    "hash_type": "SHA256", "hash": SHA}]
    }}}}
    assert (await models.submit(body)[1])[0]["state"] == "ready"
    assert hosts == ["huggingface.co"]


@pytest.mark.asyncio
async def test_download_dedup_stays_responsive_and_fences_idle_probe(local, tmp_path):
    entered, release = threading.Event(), threading.Event()
    calls = []

    def handler(req):
        calls.append(req.url)
        entered.set()
        assert release.wait(5)
        return media()

    models = local(handler, [row()])
    first_id, first = models.submit({"prompt": graph()})
    second_id, second = models.submit({"prompt": graph()})
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        assert models.status(first_id)["state"] == models.status(second_id)["state"] == "preparing"
        assert not (tmp_path / "models/checkpoints" / PATH).exists()
        probe = EXTENSION[EXTENSION.index("def open_writes():"):EXTENSION.index("def index_status(")]
        namespace = {"Path": Path, "Request": lambda url, **kw: url,
                     "urlopen": lambda _, **kw: io.BytesIO(b'{"is_processing":false}'), "json": json, "os": os}
        exec(probe.replace("/content/ai-studio/native", str(tmp_path)), namespace)
        assert namespace["probe"]()["busy_reason"] == "model_preparation"
        with (tmp_path / "models-active.lock").open("rb") as handle:
            with pytest.raises(BlockingIOError):
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        release.set()
    assert (await first)[0]["state"] == (await second)[0]["state"] == "ready"
    assert len(calls) == 1
    assert namespace["probe"]()["busy"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("space_ok", [True, False])
async def test_hash_failure_and_insufficient_disk_never_publish(local, monkeypatch, tmp_path, space_ok):
    called = []
    models = local(lambda _: called.append(True) or media(b"x" * len(DATA)), [row(drive_file_id=None)])
    if not space_ok:
        monkeypatch.setattr(shutil, "disk_usage", lambda path: SimpleNamespace(free=1))
    outcome = (await models.submit({"prompt": graph()})[1])[0]
    assert outcome["state"] == "failed"
    assert ("SHA-256" if space_ok else "磁盘空间不足") in outcome["error"]
    assert bool(called) == space_ok
    assert not (tmp_path / "models/checkpoints" / PATH).exists()
    assert not list((tmp_path / "model-transfers").glob("*.partial"))


@pytest.mark.asyncio
async def test_range_resume_requires_same_external_identity(local, monkeypatch, tmp_path):
    monkeypatch.setattr(native_models, "CHUNK", 4)
    calls = []

    class Interrupted(httpx.SyncByteStream):
        def __iter__(self):
            yield DATA[:8]
            raise httpx.ReadError("simulated interruption")

    def handler(req):
        calls.append(dict(req.headers))
        if len(calls) == 1:
            return httpx.Response(200, headers={"content-length": str(len(DATA)), "etag": '"e"'}, stream=Interrupted())
        return httpx.Response(206, headers={"content-range": f"bytes 8-{len(DATA)-1}/{len(DATA)}", "etag": '"e"'},
                              stream=Stream(DATA[8:]))

    model = local(handler, [row(drive_file_id=None)])
    assert (await model.submit({"prompt": graph()})[1])[0]["state"] == "ready"
    assert calls[1]["range"] == "bytes=8-" and calls[1]["if-range"] == '"e"'
    assert (tmp_path / "models/checkpoints" / PATH).read_bytes() == DATA


def test_native_old_model_storage_defaults_validate_independently(monkeypatch):
    monkeypatch.setattr(settings, "native_comfy_cache_gb", 23)
    assert native.model_storage({"max_runtime_minutes": 8}) == {
        "model_storage_mode": "drive_mount", "cache_gb": 23, "cache_max_age_minutes": 60,
    }
    for payload in ({"cache_gb": True}, {"cache_max_age_minutes": 0}, {"model_storage_mode": "remote"}):
        with pytest.raises(Exception, match="无效|正整数"):
            native.model_storage(payload, validate=True)


def test_scan_uses_reachable_node_types_and_native_model_metadata():
    body = {"prompt": {
        **graph(), "3": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "unused.safetensors"}},
        "4": {"class_type": "Remote", "inputs": {"model_name": "gpt-image.safetensors"}},
    }, "partial_execution_targets": ["2"], "extra_data": {"extra_pnginfo": {"workflow": {
        "models": [{"name": PATH, "url": EXTERNAL, "hash": SHA, "hash_type": "SHA-256", "directory": "checkpoints"}],
    }}}}
    types = {"CheckpointLoaderSimple": object, "PreviewImage": SimpleNamespace(OUTPUT_NODE=True),
             "Remote": SimpleNamespace(OUTPUT_NODE=True, API_NODE=True)}
    assert [(d["name"], d["url"], d["sha256"]) for d in native_models.dependencies(body, [], node_types=types)] == [
        (PATH, EXTERNAL, SHA)
    ]
    assert list(native_models.dependencies({"prompt": {
        "1": {"class_type": "VAELoader", "inputs": {"vae_name": "pixel_space"}},
        "2": {"class_type": "RemoteApiModelLoader", "inputs": {"model": "remote-id.safetensors"}},
    }}, [])) == []


class Routes(dict):
    def method(self, verb, path):
        def register(handler):
            self[verb, path] = handler
            return handler
        return register

    def get(self, path):
        return self.method("GET", path)

    def post(self, path):
        return self.method("POST", path)


@pytest.mark.asyncio
async def test_runtime_lists_registered_names_without_files_and_prepares_before_validator(local, monkeypatch, tmp_path):
    async def mounted_request(method, path, data=None, headers=None):
        async def json_body():
            return data

        return SimpleNamespace(method=method, path=path, json=json_body, headers=httpx.Headers(headers),
                               match_info={"ident": path.split("/")[-1]})

    def response(data, status=200):
        return SimpleNamespace(body=json.dumps(data).encode(), status=status, content_type="application/json")

    web = SimpleNamespace(middleware=lambda f: f, json_response=response)
    monkeypatch.setitem(sys.modules, "aiohttp", SimpleNamespace(web=web))
    engine = SimpleNamespace(routes=Routes(), app=SimpleNamespace(middlewares=[]))
    models = local(lambda _: media(), [row()])
    monkeypatch.setattr(native_models, "LocalModels", lambda *args, **kwargs: models)
    (tmp_path / "local-model-config.json").write_text("{}")
    filename_cache = {"checkpoints": []}
    folder_paths = SimpleNamespace(models_dir=str(tmp_path / "models"), folder_names_and_paths={},
                                   filename_list_cache=filename_cache)
    native_models.install(engine, tmp_path, folder_paths, {})
    middleware = engine.app.middlewares[0]
    target = tmp_path / "models/checkpoints" / PATH
    definition = {"CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": [[]]}}}}

    async def initial_info(_request):
        return response(definition)

    result = await middleware(await mounted_request("GET", "/api/object_info"), initial_info)
    assert PATH in json.loads(result.body)["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
    assert not target.exists()
    signed = {"X-Studio-Model-Channel": "internal-channel"}
    prompt = {"prompt": graph(), "client_id": "websocket-client"}
    start = engine.routes["POST", "/studio_native/models/prepare"]
    submitted = await start(await mounted_request("POST", "/studio_native/models/prepare", {"workflow": prompt}, signed))
    ident = json.loads(submitted.body)["id"]
    await models.jobs[ident][0]

    async def upstream_validator(request):
        assert not filename_cache and target.read_bytes() == DATA
        assert (await request.json())["client_id"] == "websocket-client"
        return response({"prompt_id": "real-id", "number": 0, "node_errors": {}})

    result = await middleware(await mounted_request("POST", "/api/prompt", prompt, {
        **signed, "X-Studio-Model-Prepared": ident,
    }), upstream_validator)
    assert json.loads(result.body) == {"prompt_id": "real-id", "number": 0, "node_errors": {}}


@pytest.mark.asyncio
async def test_drive_only_fallback_adds_identity_for_older_model_rows(local, tmp_path):
    calls = []

    def handler(req):
        calls.append(req)
        if req.url.host == "oauth2.googleapis.com":
            return httpx.Response(200, json={"access_token": "drive-token", "expires_in": 3600})
        assert req.headers["authorization"] == "Bearer drive-token"
        if "alt" not in req.url.params:
            return httpx.Response(200, json={"id": "known-file-id", "size": str(len(DATA)),
                                              "mimeType": "application/octet-stream",
                                              "md5Checksum": hashlib.md5(DATA, usedforsecurity=False).hexdigest()})
        return media()

    model = local(handler, [row(source=None, sha256=None)], drive_auth={"drive-account": {}})
    result = (await model.submit({"prompt": graph()})[1])[0]
    assert result["state"] == "ready" and result["source"] == "drive" and result["integrity"] == "drive_md5"
    assert (tmp_path / "models/checkpoints" / PATH).read_bytes() == DATA
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_local_legacy_directory_alias_is_reused(local, tmp_path):
    model = local(lambda _: pytest.fail("existing legacy local path must not query Drive"), [
        row(model_type="unet"),
    ], models_folder="drive-models")
    target = tmp_path / "models/unet" / PATH
    target.parent.mkdir(parents=True)
    target.write_bytes(DATA)
    body = {"prompt": {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": PATH}}}}
    result = (await model.submit(body)[1])[0]
    assert result["state"] == "ready" and result["cache_hit"]
    assert not (tmp_path / "models/diffusion_models" / PATH).exists()


@pytest.mark.asyncio
async def test_url_only_verified_file_survives_comfy_process_restart(local, tmp_path):
    header = b'{"w":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}'
    data = len(header).to_bytes(8, "little") + header + b"data"
    model = local(lambda _: media(data))
    body = {"prompt": graph(), "extra_data": {"extra_pnginfo": {"workflow": {
        "models": [{"name": PATH, "directory": "checkpoints", "url": EXTERNAL}]
    }}}}
    assert (await model.submit(body)[1])[0]["integrity"] == "transport"
    restarted = local(lambda _: pytest.fail("complete local file must be reused after process restart"))
    result = (await restarted.submit(body)[1])[0]
    assert result["state"] == "ready" and result["cache_hit"]
    provenance = next((tmp_path / "model-transfers").glob("*.json"))
    assert EXTERNAL not in provenance.read_text()


@pytest.mark.asyncio
@pytest.mark.parametrize("layout", ["local_alias", "mounted_extra", "mount_first", "changed_before_queue"])
async def test_studio_prepare_checks_actual_native_loader_paths(local, monkeypatch, tmp_path, layout):
    def response(data, status=200):
        return SimpleNamespace(body=json.dumps(data).encode(), status=status, content_type="application/json")

    def request(path, body, **headers):
        async def read():
            return body

        return SimpleNamespace(method="POST", path=path, json=read,
                               headers={"X-Studio-Model-Channel": "internal-channel", **headers})

    monkeypatch.setitem(sys.modules, "aiohttp", SimpleNamespace(web=SimpleNamespace(
        middleware=lambda fn: fn, json_response=response,
    )))
    engine = SimpleNamespace(routes=Routes(), app=SimpleNamespace(middlewares=[]))
    models = local(lambda _: pytest.fail("Studio local reuse must not access native sources"))
    monkeypatch.setattr(native_models, "LocalModels", lambda *args, **kwargs: models)
    (tmp_path / "local-model-config.json").write_text("{}")
    model_dir = tmp_path / "models"
    output = tmp_path / "drive" / "output"
    alias = model_dir / "unet"
    target = alias / PATH
    target.parent.mkdir(parents=True)
    target.write_bytes(DATA)
    output_target = output / "diffusion_models" / PATH
    output_target.parent.mkdir(parents=True)
    output_target.write_bytes(b"mounted file")
    paths = [str(alias), str(model_dir / "diffusion_models"), str(output_target.parents[1])]
    if layout == "mounted_extra":
        paths.append(str(tmp_path / "drive" / "extra"))
    if layout == "mount_first":
        paths.insert(0, paths.pop())
    resolved = []

    def get_full_path(category, name):
        found = next((Path(root) / name for root in paths if (Path(root) / name).is_file()), None)
        resolved.append(found)
        return str(found) if found else None

    folder_paths = SimpleNamespace(
        models_dir=str(model_dir), folder_names_and_paths={"diffusion_models": (paths, {".safetensors"})},
        filename_list_cache={}, add_model_folder_path=lambda *args, **kwargs: None,
        get_output_directory=lambda: str(output), get_full_path=get_full_path,
    )
    native_models.install(engine, tmp_path, folder_paths, {})
    prompt = {"prompt": {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": PATH}}}}
    path = "/studio_native/models/prepare-studio"
    weight = {"repo_id": "test/model", "filename": "base.safetensors", "revision": "a" * 40,
              "sha256": SHA, "size": len(DATA), "target": "diffusion_models/" + PATH}
    ready = await engine.routes["POST", path](request(path, {
        "workflow": prompt, "weights": [weight], "loras": [], "timeout": 60,
    }))
    ident = json.loads(ready.body)["id"]
    task, _ = models.jobs[ident]
    if layout in {"mount_first", "mounted_extra"}:
        with pytest.raises(model_sources.ModelError, match="搜索路径"):
            await task
        assert models.status(ident)["state"] == "failed" and not resolved
        return
    await task
    assert models.status(ident)["models"][0]["source"] == "local"
    assert not (model_dir / "diffusion_models" / PATH).exists()

    async def queue(_):
        assert layout == "local_alias"
        return response({"prompt_id": "simulated"})

    if layout == "changed_before_queue":
        paths.insert(0, paths.pop())
    middleware = engine.app.middlewares[0]
    result = await middleware(request("/prompt", prompt, **{"X-Studio-Model-Prepared": ident}), queue)
    assert result.status == (422 if layout == "changed_before_queue" else 200)
    assert resolved and all(path == target for path in resolved)


def test_rgthree_lora_loader_stack_dependencies_scanned():
    from app.gpu import model_dependencies as deps

    prompt = {
        "1": {
            "class_type": "Lora Loader Stack (rgthree)",
            "inputs": {
                "lora_01": "style_a.safetensors",
                "strength_01": 0.6,
                "lora_02": "None",
                "strength_02": 1.0,
                "lora_03": "zero_weight.safetensors",
                "strength_03": 0.0,
                "lora_04": "style_b.safetensors",
                "strength_04": 0.35,
            },
        }
    }
    found = {d["field"]: d for d in deps.scan({"api_graph": prompt})}
    assert set(found) == {"lora_01", "lora_03", "lora_04"}
    assert found["lora_01"]["model_type"] == "loras" and found["lora_01"]["required"] is True
    assert found["lora_03"]["required"] is False
    assert found["lora_04"]["reference"] == "style_b.safetensors" and found["lora_04"]["required"] is True


def test_visual_ui_workflow_scans_gguf_rgthree_custom_node_types_and_skips_muted():
    class CustomLoader:
        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "weird_dict_kind": ({"custom": True},),
                    "model": ("MODEL",),
                    "seed": ("INT", {"default": 0, "control_after_generate": True}),
                    "custom_unet": (["custom_a.safetensors"],),
                }
            }

    class BrokenThirdPartyNode:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"unhashable": ({"bad": 1},)}}

    visual = {
        "nodes": [
            {"id": 1, "type": "UnetLoaderGGUF", "widgets_values": ["flux-q8.gguf"]},
            {
                "id": 2,
                "type": "Power Lora Loader (rgthree)",
                "widgets_values": [
                    {},
                    {"type": "PowerLoraLoaderHeaderWidget"},
                    {"on": True, "lora": "active_style.safetensors", "strength": 0.75},
                    {"on": False, "lora": "turned_off.safetensors", "strength": 1.0},
                ],
            },
            {
                "id": 3,
                "type": "Lora Loader Stack (rgthree)",
                "widgets_values": ["stack_1.safetensors", 0.5, "None", 1.0, "None", 1.0, "None", 1.0],
            },
            {"id": 4, "type": "CustomUnetLoader", "widgets_values": [42, "randomize", "custom_a.safetensors"]},
            {"id": 7, "type": "BrokenThirdPartyNode", "widgets_values": [True]},
            {"id": 5, "type": "VAELoader", "mode": 2, "widgets_values": ["muted_vae.safetensors"]},
            {
                "id": 6,
                "type": "VAELoader",
                "inputs": [{"name": "vae_name", "type": "COMBO", "link": 99}],
                "widgets_values": ["linked_override.safetensors"],
            },
        ]
    }
    old_rules = os.environ.get("MODEL_DEPENDENCY_RULES")
    os.environ["MODEL_DEPENDENCY_RULES"] = json.dumps({"CustomUnetLoader": {"custom_unet": "diffusion_models"}})
    try:
        found = list(
            native_models.dependencies(
                {"workflow": visual},
                [],
                visual=True,
                node_types={"CustomUnetLoader": CustomLoader, "BrokenThirdPartyNode": BrokenThirdPartyNode},
            )
        )
    finally:
        if old_rules is None:
            os.environ.pop("MODEL_DEPENDENCY_RULES", None)
        else:
            os.environ["MODEL_DEPENDENCY_RULES"] = old_rules
    pairs = [(d["category"], d["name"]) for d in found]
    assert pairs == [
        ("diffusion_models", "flux-q8.gguf"),
        ("loras", "active_style.safetensors"),
        ("loras", "stack_1.safetensors"),
        ("diffusion_models", "custom_a.safetensors"),
    ]


def test_visual_ui_workflow_expands_comfy_v1_subgraphs_and_outer_widget_overrides():
    sg_base = "7bd59413-cc00-475d-8413-f17feaa280d9"
    sg_upscale = "f188c0f7-bd94-4e06-8bf2-2359a280af3e"
    sg_muted = "6de93206-dd50-4fa4-b99f-67c48654c7b3"
    visual = {
        "nodes": [
            {
                "id": 343,
                "type": sg_base,
                "mode": 0,
                "inputs": [
                    {"name": "unet_name", "type": "COMBO", "widget": {"name": "unet_name"}, "link": None},
                    {"name": "clip_name", "type": "COMBO", "widget": {"name": "clip_name"}, "link": None},
                    {"name": "vae_name", "type": "COMBO", "widget": {"name": "vae_name"}, "link": None},
                ],
                "widgets_values": [
                    "anima-aesthetic-v1.1.safetensors",
                    "qwen_3_06b_base.safetensors",
                    "qwen_image_vae.safetensors",
                ],
            },
            {
                "id": 222,
                "type": sg_upscale,
                "mode": 0,
                "inputs": [
                    {"name": "image", "type": "IMAGE", "link": 35557},
                    {"name": "model_name", "type": "COMBO", "widget": {"name": "model_name"}, "link": None},
                ],
                "widgets_values": ["4xUltrasharp_4xUltrasharpV10.pt"],
            },
            {
                "id": 1335,
                "type": sg_muted,
                "mode": 2,
                "inputs": [{"name": "unet_name", "type": "COMBO", "widget": {"name": "unet_name"}, "link": None}],
                "widgets_values": ["muted_branch_model.safetensors"],
            },
        ],
        "definitions": {
            "subgraphs": [
                {
                    "id": sg_base,
                    "inputs": [
                        {"name": "unet_name", "type": "COMBO", "linkIds": [101]},
                        {"name": "clip_name", "type": "COMBO", "linkIds": [102]},
                        {"name": "vae_name", "type": "COMBO", "linkIds": [103]},
                    ],
                    "nodes": [
                        {
                            "id": 339,
                            "type": "UNETLoader",
                            "mode": 0,
                            "widgets_values": ["stale_default_unet.safetensors", "default"],
                            "inputs": [{"name": "unet_name", "type": "COMBO", "widget": {"name": "unet_name"}, "link": 101}],
                        },
                        {
                            "id": 340,
                            "type": "CLIPLoader",
                            "mode": 0,
                            "widgets_values": ["stale_default_clip.safetensors", "stable_diffusion", "default"],
                            "inputs": [{"name": "clip_name", "type": "COMBO", "widget": {"name": "clip_name"}, "link": 102}],
                        },
                        {
                            "id": 341,
                            "type": "VAELoader",
                            "mode": 0,
                            "widgets_values": ["stale_default_vae.safetensors"],
                            "inputs": [{"name": "vae_name", "type": "COMBO", "widget": {"name": "vae_name"}, "link": 103}],
                        },
                    ],
                },
                {
                    "id": sg_upscale,
                    "inputs": [
                        {"name": "image", "type": "IMAGE", "linkIds": [381]},
                        {"name": "model_name", "type": "COMBO", "linkIds": [384]},
                    ],
                    "nodes": [
                        {
                            "id": 154,
                            "type": "UpscaleModelLoader",
                            "mode": 0,
                            "widgets_values": ["RealESRGAN_x4plus_anime_6B.pth"],
                            "inputs": [{"name": "model_name", "type": "COMBO", "widget": {"name": "model_name"}, "link": 384}],
                        }
                    ],
                },
                {
                    "id": sg_muted,
                    "inputs": [{"name": "unet_name", "type": "COMBO", "linkIds": [901]}],
                    "nodes": [
                        {
                            "id": 999,
                            "type": "UNETLoader",
                            "mode": 0,
                            "widgets_values": ["muted_inner_default.safetensors", "default"],
                            "inputs": [{"name": "unet_name", "type": "COMBO", "widget": {"name": "unet_name"}, "link": 901}],
                        }
                    ],
                },
            ]
        },
    }
    found = list(native_models.dependencies({"workflow": visual}, [], visual=True))
    assert [(d["category"], d["name"]) for d in found] == [
        ("diffusion_models", "anima-aesthetic-v1.1.safetensors"),
        ("text_encoders", "qwen_3_06b_base.safetensors"),
        ("vae", "qwen_image_vae.safetensors"),
        ("upscale_models", "4xUltrasharp_4xUltrasharpV10.pt"),
    ]

    # Also support older subgraph nodes where outer_node["inputs"] omits unlinked widget slots.
    legacy_outer = {
        "nodes": [
            {
                "id": 167,
                "type": sg_base,
                "mode": 0,
                "inputs": [],
                "widgets_values": [
                    "krea2_turbo_int8_convrot.safetensors",
                    "qwen3vl_4b_fp8_scaled.safetensors",
                    "qwen_image_vae.safetensors",
                ],
            }
        ],
        "definitions": visual["definitions"],
    }
    legacy_found = list(native_models.dependencies({"workflow": legacy_outer}, [], visual=True))
    assert [(d["category"], d["name"]) for d in legacy_found] == [
        ("diffusion_models", "krea2_turbo_int8_convrot.safetensors"),
        ("text_encoders", "qwen3vl_4b_fp8_scaled.safetensors"),
        ("vae", "qwen_image_vae.safetensors"),
    ]


@pytest.mark.asyncio
async def test_concurrent_workflow_save_preserves_first_writer(local, monkeypatch):
    models = local(lambda _: pytest.fail("saving should use only the mounted filesystem"))
    start = threading.Barrier(2)
    publish = threading.Barrier(2)
    replace = Path.replace

    def pause_publish(path, target):
        try:
            publish.wait(timeout=0.5)
        except threading.BrokenBarrierError:
            pass
        return replace(path, target)

    def save(label):
        start.wait(timeout=5)
        return models.save_workflow("shared", {"nodes": [], "extra": {"label": label}})

    monkeypatch.setattr(Path, "replace", pause_publish)
    results = await asyncio.gather(*(asyncio.to_thread(save, label) for label in ("a", "b")), return_exceptions=True)
    assert sum(isinstance(result, native_models.WorkflowConflictError) for result in results) == 1
    winner = "b" if isinstance(results[0], native_models.WorkflowConflictError) else "a"
    assert models.read_workflow("comfy", "shared.json")["workflow"]["extra"]["label"] == winner


@pytest.mark.asyncio
async def test_drive_workflows_list_save_delete_and_prepare_by_ref(local, monkeypatch, tmp_path):
    def response(data, status=200):
        return SimpleNamespace(body=json.dumps(data).encode(), status=status, content_type="application/json")

    def request(method, path, body=None, query=None):
        async def read():
            return body

        return SimpleNamespace(
            method=method,
            path=path,
            query=query or {},
            json=read,
            headers=httpx.Headers({"X-Studio-Model-Channel": "internal-channel"}),
            match_info={"ident": path.split("/")[-1]},
        )

    monkeypatch.setitem(sys.modules, "aiohttp", SimpleNamespace(web=SimpleNamespace(
        middleware=lambda fn: fn, json_response=response, HTTPNotFound=RuntimeError,
    )))
    engine = SimpleNamespace(routes=Routes(), app=SimpleNamespace(middlewares=[]))
    models = local(lambda _: media(), [row()])
    monkeypatch.setattr(native_models, "LocalModels", lambda *args, **kwargs: models)
    (tmp_path / "local-model-config.json").write_text("{}")
    folder_paths = SimpleNamespace(models_dir=str(tmp_path / "models"), folder_names_and_paths={}, filename_list_cache={})
    native_models.install(engine, tmp_path, folder_paths, {})

    comfy_dir = tmp_path / "drive/native-comfy/user/default/workflows/anima"
    comfy_dir.mkdir(parents=True)
    ui_workflow = {
        "nodes": [{"id": 1, "type": "CheckpointLoaderSimple", "widgets_values": [PATH]}],
    }
    ui_file = comfy_dir / "original.json"
    ui_file.write_text(json.dumps(ui_workflow), encoding="utf-8")
    (comfy_dir / ".index.json").write_text("{}", encoding="utf-8")

    # Save an API workflow via route -> should land in warmup directory, not comfy UI folder.
    save_route = engine.routes["POST", "/studio_native/models/workflows"]
    saved_resp = await save_route(request("POST", "/studio_native/models/workflows", {
        "name": "tavern_api",
        "workflow": graph(),
        "overwrite": False,
    }))
    assert saved_resp.status == 201
    saved_item = json.loads(saved_resp.body)["workflow"]
    assert saved_item["source"] == "warmup" and saved_item["path"] == "tavern_api.json"
    assert (tmp_path / "drive/native-comfy/warmup/tavern_api.json").is_file()

    # Duplicate save without overwrite returns 409; with overwrite=True succeeds.
    conflict_resp = await save_route(request("POST", "/studio_native/models/workflows", {
        "name": "tavern_api.json",
        "workflow": graph(),
        "overwrite": False,
    }))
    assert conflict_resp.status == 409
    overwrite_resp = await save_route(request("POST", "/studio_native/models/workflows", {
        "name": "tavern_api.json",
        "workflow": graph(),
        "overwrite": True,
    }))
    assert overwrite_resp.status == 201

    wrapped_resp = await save_route(request("POST", "/studio_native/models/workflows", {
        "name": "wrapped",
        "workflow": {"workflow": ui_workflow},
    }))
    assert wrapped_resp.status == 201
    assert json.loads((comfy_dir.parent / "wrapped.json").read_text()) == ui_workflow

    mixed = {"workflow": ui_workflow, "prompt": graph()}
    mixed_item = models.save_workflow("mixed", mixed)
    assert mixed_item["source"] == "warmup"
    assert models.read_workflow("warmup", "mixed.json") == mixed

    # Path traversal and hidden files are rejected.
    for bad_name in ("../escape.json", ".hidden.json", "sub/.secret.json"):
        bad_resp = await save_route(request("POST", "/studio_native/models/workflows", {
            "name": bad_name,
            "workflow": graph(),
        }))
        assert bad_resp.status == 422

    # List returns both comfy and warmup workflows, excluding hidden .index.json.
    list_route = engine.routes["GET", "/studio_native/models/workflows"]
    listed = json.loads((await list_route(request("GET", "/studio_native/models/workflows", query={"refresh": "1"}))).body)
    assert {(w["source"], w["path"]) for w in listed["workflows"]} == {
        ("comfy", "anima/original.json"),
        ("comfy", "wrapped.json"),
        ("warmup", "mixed.json"),
        ("warmup", "tavern_api.json"),
    }

    # Prepare directly from Drive UI workflow reference without re-uploading JSON.
    prep_route = engine.routes["POST", "/studio_native/models/prepare"]
    prep_resp = await prep_route(request("POST", "/studio_native/models/prepare", {
        "workflow_ref": {"source": "comfy", "path": "anima/original.json"},
        "visual": True,
    }))
    assert prep_resp.status == 202
    ident = json.loads(prep_resp.body)["id"]
    prep_models = await models.jobs[ident][0]
    assert prep_models[0]["state"] == "ready"
    assert (tmp_path / "models/checkpoints" / PATH).read_bytes() == DATA

    # Deleting comfy workflow via Studio is forbidden; deleting warmup workflow succeeds.
    del_route = engine.routes["POST", "/studio_native/models/workflows/delete"]
    forbid_del = await del_route(request("POST", "/studio_native/models/workflows/delete", {
        "source": "comfy", "path": "anima/original.json",
    }))
    assert forbid_del.status == 422 and ui_file.is_file()

    ok_del = await del_route(request("POST", "/studio_native/models/workflows/delete", {
        "source": "warmup", "path": "tavern_api.json",
    }))
    assert ok_del.status == 200
    assert not (tmp_path / "drive/native-comfy/warmup/tavern_api.json").exists()
