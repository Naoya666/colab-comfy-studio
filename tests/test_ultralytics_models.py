"""Tests for ComfyUI Impact-Subpack Ultralytics bbox/segm model support."""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from app import creation_api, main as main_module
from app.db import Store
from app.gpu import model_dependencies as deps
from app.gpu import model_sources, native_models, native_runtime
from app.gpu.model_library import ModelLibrary, ModelTransfer
from app.gpu.model_sources import ModelError, SourceHTTP
from app.gpu.repository import Repository


def _make_ultralytics_pt_bytes(*, extra_global: tuple[str, str] | None = None) -> bytes:
    """Build a minimal valid PyTorch ZIP .pt archive with Ultralytics pickle globals."""
    buf = io.BytesIO()
    # Hand-craft a safe pickle protocol-2 stream referencing torch tensor rebuild + Ultralytics DetectionModel
    pkl = io.BytesIO()
    pkl.write(b"\x80\x02")
    # Push ultralytics.nn.tasks.DetectionModel and torch._utils._rebuild_tensor_v2 + torch.FloatStorage
    if extra_global is not None:
        mod, attr = extra_global
        pkl.write(f"c{mod}\n{attr}\n".encode("ascii"))
        pkl.write(b")\x81")  # NEWOBJ
    else:
        pkl.write(b"cultralytics.nn.tasks\nDetectionModel\n")
        pkl.write(b"cultralytics.nn.modules.block\nC2f\n")
        pkl.write(b"cthread\n" if False else b"ctorch.nn.modules.conv\nConv2d\n")
        pkl.write(b"ctorch\ndevice\n")
        pkl.write(b"c__builtin__\nset\n")
        pkl.write(b"ctorch._utils\n_rebuild_tensor_v2\n")
        pkl.write(b"ctorch\nFloatStorage\n")
        pkl.write(b"t")  # TUPLE
    pkl.write(b".")  # STOP
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("hand_yolov8s/data.pkl", pkl.getvalue())
        zf.writestr("hand_yolov8s/data/0", b"\x00" * 16)
    return buf.getvalue()


class _FakeDriveResponse:
    def __init__(self, status_code=200, data=None, headers=None):
        self.status_code = status_code
        self._data = data if data is not None else {}
        self.headers = headers or {}

    @property
    def is_success(self):
        return 200 <= self.status_code < 300

    def json(self):
        return self._data


class _FakeDrive:
    def __init__(self, files: dict[str, bytes] | None = None):
        self.base_url = "https://www.googleapis.com/drive/v3"
        self.upload_url = "https://www.googleapis.com/upload/drive/v3"
        self.account_id = "acct-default"
        self._owns_client = False
        self.files = dict(files or {})
        self.google = SimpleNamespace(access_token=lambda _scope: "fake-token")
        self._seq = 0

    def model_cache(self):
        return {"folder_id": "folder-models", "access_token": "fake-token"}

    def _folder(self, name, parent):
        return f"folder-{name}"

    def _q(self, val):
        return val

    def _ensure_ok(self, resp, _label):
        if resp.status_code >= 400:
            raise RuntimeError(f"HTTP {resp.status_code}")
        return resp.json()

    def _request(self, method, url, *, params=None, json_body=None, headers=None, content=None, retryable=True):
        if method == "POST" and url.endswith("/files") and (params or {}).get("uploadType") == "resumable":
            self._seq += 1
            fid = (json_body or {}).get("id") or f"drvfile{self._seq}"
            return _FakeDriveResponse(
                200,
                {"id": fid},
                headers={"location": f"https://www.googleapis.com/upload/drive/v3/files?upload_id={fid}"},
            )
        if method == "GET" and url.endswith("/files/generateIds"):
            self._seq += 1
            return _FakeDriveResponse(200, {"ids": [f"drvgen{self._seq}"]})
        if method == "GET" and "/files/" in url:
            fid = url.rsplit("/files/", 1)[1].split("?", 1)[0]
            if fid not in self.files:
                return _FakeDriveResponse(404, {})
            raw = self.files[fid]
            return _FakeDriveResponse(200, {
                "id": fid,
                "name": fid,
                "size": str(len(raw)),
                "mimeType": "application/octet-stream",
                "trashed": False,
                "md5Checksum": hashlib.md5(raw, usedforsecurity=False).hexdigest(),
                "sha256Checksum": hashlib.sha256(raw).hexdigest(),
            })
        if method == "GET" and url.endswith("/files"):
            return _FakeDriveResponse(200, {"files": []})
        if method == "PUT" and "upload_id=" in url:
            fid = url.split("upload_id=", 1)[1]
            data = b"".join(content) if content is not None else b""
            self.files[fid] = data
            return _FakeDriveResponse(200, {
                "id": fid,
                "name": fid,
                "size": str(len(data)),
                "mimeType": "application/octet-stream",
                "trashed": False,
                "md5Checksum": hashlib.md5(data, usedforsecurity=False).hexdigest(),
                "sha256Checksum": hashlib.sha256(data).hexdigest(),
            })
        return _FakeDriveResponse(200, {})


def test_directories_and_split_target_path():
    dmap = deps.directories()
    assert dmap["ultralytics_bbox"] == "ultralytics/bbox"
    assert dmap["ultralytics_segm"] == "ultralytics/segm"

    assert deps.split_target_path("ultralytics/bbox/hand_yolov8s.pt") == ("ultralytics_bbox", "hand_yolov8s.pt")
    assert deps.split_target_path("ultralytics/segm/nested/person_seg.pt") == ("ultralytics_segm", "nested/person_seg.pt")
    assert deps.split_target_path("models/ultralytics/bbox/hand_yolov8s.pt") == ("ultralytics_bbox", "hand_yolov8s.pt")
    assert deps.split_target_path("checkpoints/sdxl.safetensors") == ("checkpoints", "sdxl.safetensors")
    assert deps.split_target_path("unet/flux.safetensors") == ("diffusion_models", "flux.safetensors")


def test_native_runtime_ultralytics_paths_and_index(tmp_path):
    assert "ultralytics_bbox" in native_runtime.CATEGORIES
    assert "ultralytics_segm" in native_runtime.CATEGORIES

    cfg = native_runtime.model_paths(tmp_path / "mount")["studio_drive"]
    assert cfg["ultralytics_bbox"] == "models/ultralytics/bbox\nmodels/native-index/ultralytics/bbox"
    assert cfg["ultralytics_segm"] == "models/ultralytics/segm\nmodels/native-index/ultralytics/segm"

    comfy = tmp_path / "ComfyUI"
    comfy.mkdir()
    mount = tmp_path / "mount"
    mount.mkdir()
    native_runtime.prepare_paths(comfy, mount, local=True)
    assert (comfy / "models/ultralytics/bbox").is_dir()
    assert (comfy / "models/ultralytics/segm").is_dir()


def test_ultralytics_pt_security_validation(tmp_path):
    valid_bytes = _make_ultralytics_pt_bytes()
    valid_file = tmp_path / "hand_yolov8s.pt"
    valid_file.write_bytes(valid_bytes)

    # Allowed for ultralytics_bbox and ultralytics_segm
    model_sources.validate_weight_file(valid_file, "hand_yolov8s.pt", len(valid_bytes), model_type="ultralytics_bbox")
    model_sources.validate_weight_file(valid_file, "hand_yolov8s.pt", len(valid_bytes), model_type="ultralytics_segm")

    # Rejected for checkpoints / loras / upscale_models (upscale_models stays strict)
    with pytest.raises(ModelError):
        model_sources.validate_weight_file(valid_file, "hand_yolov8s.pt", len(valid_bytes), model_type="checkpoints")
    with pytest.raises(ModelError):
        model_sources.validate_weight_file(valid_file, "hand_yolov8s.pt", len(valid_bytes), model_type="upscale_models")

    # .pth is NOT allowed under ultralytics categories
    with pytest.raises(ModelError):
        model_sources.filename("hand_yolov8s.pth", model_type="ultralytics_bbox")
    with pytest.raises(ModelError):
        model_sources.validate_weight_file(valid_file, "hand_yolov8s.pth", len(valid_bytes), model_type="ultralytics_segm")

    # Malicious pickle payloads MUST be rejected under ultralytics_bbox/segm
    for dangerous in (("os", "system"), ("subprocess", "Popen"), ("builtins", "eval"), ("builtins", "__import__")):
        bad_bytes = _make_ultralytics_pt_bytes(extra_global=dangerous)
        bad_file = tmp_path / f"bad_{dangerous[0]}_{dangerous[1]}.pt"
        bad_file.write_bytes(bad_bytes)
        with pytest.raises(ModelError):
            model_sources.validate_weight_file(bad_file, bad_file.name, len(bad_bytes), model_type="ultralytics_bbox")


def test_ultralytics_detector_provider_dependency_extraction_and_matching():
    api_graph = {
        "10": {"class_type": "UltralyticsDetectorProvider", "inputs": {"model_name": "bbox/hand_yolov8s.pt"}},
        "11": {"class_type": "UltralyticsDetectorProvider", "inputs": {"model_name": "segm/custom/person_yolov8m-seg.pt"}},
    }
    scanned = deps.scan({"api_graph": api_graph})
    assert len(scanned) == 2
    by_node = {d["node_id"]: d for d in scanned}
    assert by_node["10"]["model_type"] == "ultralytics_bbox"
    assert by_node["10"]["reference"] == "hand_yolov8s.pt"
    assert deps.target_for(by_node["10"]) == "ultralytics/bbox/hand_yolov8s.pt"

    assert by_node["11"]["model_type"] == "ultralytics_segm"
    assert by_node["11"]["reference"] == "custom/person_yolov8m-seg.pt"
    assert deps.target_for(by_node["11"]) == "ultralytics/segm/custom/person_yolov8m-seg.pt"

    # Visual workflow with Windows backslash + properties.models hint
    visual = {
        "nodes": [
            {
                "id": 21,
                "type": "UltralyticsDetectorProvider",
                "widgets_values": ["bbox\\hand_yolov8s.pt"],
                "properties": {
                    "models": [
                        {
                            "name": "hand_yolov8s.pt",
                            "directory": "ultralytics/bbox",
                            "url": "https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8s.pt",
                        }
                    ]
                },
            }
        ]
    }
    vis_result = deps.inspect_workflow(visual, [
        {"id": "m-hand", "model_type": "ultralytics_bbox", "filename": "hand_yolov8s.pt", "state": "ready", "sha256": "a" * 64}
    ])
    assert len(vis_result["dependencies"]) == 1
    dep = vis_result["dependencies"][0]
    assert dep["model_type"] == "ultralytics_bbox"
    assert dep["reference"] == "hand_yolov8s.pt"
    assert dep["source_hint"] == "https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8s.pt"
    assert dep["candidates"] == ["m-hand"]


def test_model_library_upload_drive_import_source_binding_and_security(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_LIBRARY_SCRATCH", str(tmp_path / "scratch"))
    store = Store(f"sqlite:///{tmp_path / 'db.sqlite'}")
    repo = Repository(store)
    valid_bytes = _make_ultralytics_pt_bytes()
    valid_sha = hashlib.sha256(valid_bytes).hexdigest()
    bad_bytes = _make_ultralytics_pt_bytes(extra_global=("os", "system"))

    drive = _FakeDrive({"drv-valid": valid_bytes, "drv-bad": bad_bytes})

    def hf_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/api/models/Bingsu/adetailer/revision/"):
            return httpx.Response(200, json={
                "sha": "1" * 40,
                "siblings": [
                    {"rfilename": "hand_yolov8s.pt", "size": len(valid_bytes), "lfs": {"sha256": valid_sha, "size": len(valid_bytes)}}
                ],
            })
        if request.url.path.endswith("/hand_yolov8s.pt"):
            return httpx.Response(200, content=valid_bytes, headers={"content-length": str(len(valid_bytes))})
        return httpx.Response(404)

    http = SourceHTTP({}, client=httpx.Client(transport=httpx.MockTransport(hf_handler)), resolver=lambda _h: ["8.8.8.8"])
    lib = ModelLibrary(repo, "owner", drive=drive, http=http)

    # 1. Upload session allows .pt for ultralytics_bbox, rejects for checkpoints
    session = lib.upload_session("hand_yolov8s.pt", len(valid_bytes), "https://studio.example", model_type="ultralytics_bbox")
    assert session["upload_url"].startswith("https://www.googleapis.com/")
    with pytest.raises(ModelError):
        lib.upload_session("hand_yolov8s.pt", len(valid_bytes), "https://studio.example", model_type="checkpoints")

    # 2. Local upload & Drive import verify bytes (never skip validate_weight_file!)
    # Patch ModelTransfer.read_drive to read from our fake drive bytes
    def fake_read_drive(self, path):
        raw = drive.files[self.model["drive_file_id"]]
        path.write_bytes(raw)
        sha = hashlib.sha256(raw)
        md5 = hashlib.md5(raw, usedforsecurity=False)
        meta = drive._request("GET", f"{drive.base_url}/files/{self.model['drive_file_id']}").json()
        return meta, (sha, md5, len(raw))

    monkeypatch.setattr(ModelTransfer, "read_drive", fake_read_drive)

    uploaded = lib.add({
        "source": "upload",
        "drive_file_id": "drv-valid",
        "filename": "hand_yolov8s.pt",
        "model_type": "ultralytics_bbox",
    })
    task = lib.enqueue(uploaded["id"])
    ModelTransfer(lib, task).run()
    ready = lib.get(uploaded["id"])
    assert ready["state"] == "ready"
    assert ready["sha256"] == valid_sha

    # Source binding on uploaded .pt model
    bound = lib.bind_external_source(ready, "https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8s.pt")
    assert bound["status"] == "verified"
    assert bound["download_url"].endswith("/hand_yolov8s.pt")

    # Malicious .pt via Drive import MUST fail validation even though Drive returns sha256Checksum
    bad_row = lib.add({
        "source": "drive",
        "drive_file_id": "drv-bad",
        "filename": "evil_yolov8s.pt",
        "model_type": "ultralytics_bbox",
    })
    bad_task = lib.enqueue(bad_row["id"])
    ModelTransfer(lib, bad_task).run()
    assert lib.get(bad_row["id"])["state"] == "failed"
    assert "未授权的 pickle" in (lib.get(bad_row["id"]).get("error") or "")

    # 3. URL download + deduplication
    resolved = lib.resolve({
        "url": "https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8s.pt",
        "model_type": "ultralytics_bbox",
    })
    assert len(resolved) == 1
    assert resolved[0]["model_type"] == "ultralytics_bbox"
    added_1 = lib.add({**resolved[0], "model_type": "ultralytics_bbox"})
    added_2 = lib.add({**resolved[0], "model_type": "ultralytics_bbox"})
    assert added_1["id"] == added_2["id"]


def test_native_comfy_local_mode_creation_and_external_api(tmp_path, monkeypatch):
    valid_bytes = _make_ultralytics_pt_bytes()
    valid_sha = hashlib.sha256(valid_bytes).hexdigest()
    root = tmp_path / "native"
    root.mkdir()
    model_dir = tmp_path / "models"
    (model_dir / "ultralytics" / "bbox").mkdir(parents=True)
    (model_dir / "ultralytics" / "segm").mkdir(parents=True)

    url = "https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8s.pt"

    def transport(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/hand_yolov8s.pt"):
            return httpx.Response(200, stream=httpx.ByteStream(valid_bytes), headers={"content-length": str(len(valid_bytes))})
        return httpx.Response(404)

    config = {
        "channel": "chan-1",
        "catalog": [
            {
                "model_type": "ultralytics_bbox",
                "filename": "hand_yolov8s.pt",
                "sha256": valid_sha,
                "size": len(valid_bytes),
                "source": {"download_url": url, "status": "verified"},
            },
            {
                "model_type": "ultralytics_segm",
                "filename": "person_yolov8m-seg.pt",
                "sha256": "b" * 64,
                "size": 100,
                "source": {"download_url": "https://huggingface.co/Bingsu/adetailer/resolve/main/person_yolov8m-seg.pt", "status": "verified"},
            },
        ],
    }
    (root / "local-model-config.json").write_text(json.dumps(config))

    # 1. Native local preparation for UltralyticsDetectorProvider
    monkeypatch.setattr(
        native_models,
        "SourceHTTP",
        lambda tokens=None, client=None: SourceHTTP(tokens, client=client, resolver=lambda _h: ["8.8.8.8"]),
    )
    local = native_models.LocalModels(
        root,
        model_dir,
        config,
        http=httpx.Client(transport=httpx.MockTransport(transport)),
    )
    prompt = {
        "prompt": {
            "1": {"class_type": "UltralyticsDetectorProvider", "inputs": {"model_name": "bbox/hand_yolov8s.pt"}}
        }
    }
    rows = asyncio.run(local.prepare(prompt, [], config["catalog"]))

    assert len(rows) == 1
    assert rows[0]["state"] == "ready"
    assert rows[0]["category"] == "ultralytics_bbox"
    assert rows[0]["name"] == "hand_yolov8s.pt"
    dest = model_dir / "ultralytics" / "bbox" / "hand_yolov8s.pt"
    assert dest.is_file()
    assert dest.read_bytes() == valid_bytes

    # 2. Inventory scan & persistence for local mode
    inv = local.scan_inventory()
    assert inv["wild_models"] == []
    assert any(
        item["category"] == "ultralytics_bbox" and item["name"] == "hand_yolov8s.pt" and item["sha256"] == valid_sha
        for item in inv["recoverable_models"]
    )

    # 3. External API match_version & model_inventory
    workflow = {
        "api_graph": {
            "1": {"class_type": "UltralyticsDetectorProvider", "inputs": {"model_name": "bbox/hand_yolov8s.pt"}},
            "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "a hand"}},
        },
        "manifest": {
            "parameters": {
                "prompt": {"type": "string", "binding": "2.inputs.text", "default": "a hand"},
            },
            "weight_files": [
                {"source": "drive", "drive_file_id": "drv1", "sha256": valid_sha, "size": len(valid_bytes), "target": "ultralytics/bbox/hand_yolov8s.pt"}
            ],
        },
    }
    params, reason = creation_api.match_version(workflow["api_graph"], workflow)
    assert reason is None and params == {"prompt": "a hand"}
    tampered = {
        "1": {"class_type": "UltralyticsDetectorProvider", "inputs": {"model_name": "bbox/other.pt"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "a hand"}},
    }
    _, bad_reason = creation_api.match_version(tampered, workflow)
    assert bad_reason is not None

    inventory = creation_api.model_inventory([workflow])
    assert "hand_yolov8s.pt" in inventory["ultralytics_bbox"]
    assert "bbox/hand_yolov8s.pt" in inventory["ultralytics"]


def test_native_install_object_info_and_verify_studio_paths(tmp_path, monkeypatch):
    def fake_json_response(data, status=200):
        return SimpleNamespace(
            status=status,
            content_type="application/json",
            body=json.dumps(data).encode("utf-8"),
        )

    monkeypatch.setitem(sys.modules, "aiohttp", SimpleNamespace(web=SimpleNamespace(
        middleware=lambda fn: fn,
        json_response=fake_json_response,
        HTTPNotFound=RuntimeError,
    )))

    root = tmp_path / "native"
    root.mkdir()
    comfy = tmp_path / "ComfyUI"
    models_dir = comfy / "models"
    (models_dir / "ultralytics" / "bbox").mkdir(parents=True)
    (models_dir / "ultralytics" / "segm").mkdir(parents=True)
    output_dir = comfy / "output"
    output_dir.mkdir(parents=True)

    config = {
        "channel": "chan-1",
        "catalog": [
            {
                "model_type": "ultralytics_bbox",
                "filename": "hand_yolov8s.pt",
                "sha256": "a" * 64,
                "size": 123,
                "source": {"download_url": "https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8s.pt", "status": "verified"},
            },
            {
                "model_type": "ultralytics_segm",
                "filename": "person_yolov8m-seg.pt",
                "sha256": "b" * 64,
                "size": 456,
                "source": {"download_url": "https://huggingface.co/Bingsu/adetailer/resolve/main/person_yolov8m-seg.pt", "status": "verified"},
            },
        ],
    }
    (root / "local-model-config.json").write_text(json.dumps(config))

    folder_names_and_paths = {
        "ultralytics_bbox": ([str(models_dir / "ultralytics" / "bbox")], {".pt"}),
        "ultralytics_segm": ([str(models_dir / "ultralytics" / "segm")], {".pt"}),
    }

    def add_model_folder_path(cat, path, is_default=False):
        paths, _ = folder_names_and_paths.setdefault(cat, ([], set()))
        if path not in paths:
            if is_default:
                paths.insert(0, path)
            else:
                paths.append(path)

    def get_full_path(cat, name):
        for p in folder_names_and_paths.get(cat, ([], set()))[0]:
            cand = Path(p) / name
            if cand.is_file():
                return str(cand)
        return None

    folder_paths = SimpleNamespace(
        models_dir=str(models_dir),
        folder_names_and_paths=folder_names_and_paths,
        add_model_folder_path=add_model_folder_path,
        get_full_path=get_full_path,
        get_output_directory=lambda: str(output_dir),
        filename_list_cache={},
    )

    class _DummyRoutes:
        def get(self, _p):
            return lambda fn: fn

        def post(self, _p):
            return lambda fn: fn

    middlewares = []
    server = SimpleNamespace(routes=_DummyRoutes(), app=SimpleNamespace(middlewares=middlewares))
    native_models.install(server, root, folder_paths, node_types={})

    mw = middlewares[0]
    req = SimpleNamespace(method="GET", path="/object_info", headers={})

    async def fake_handler(_request):
        return fake_json_response({
            "UltralyticsDetectorProvider": {
                "input": {"required": {"model_name": [["bbox/existing.pt"]]}},
            }
        })

    resp = asyncio.run(mw(req, fake_handler))
    payload = json.loads(resp.body)
    opts = payload["UltralyticsDetectorProvider"]["input"]["required"]["model_name"][0]
    assert "bbox/existing.pt" in opts
    assert "bbox/hand_yolov8s.pt" in opts
    assert "segm/person_yolov8m-seg.pt" in opts
