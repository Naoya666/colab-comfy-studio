"""Offline tests for native ComfyUI telemetry collection and offline workspace management."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import httpx
import pytest
from app.common import APIError
from app.config import settings
from app.db import Store
from app.gpu import native, native_workspace
from app.gpu import node_persistence as np_mod
from app.gpu.drive import FOLDER_MIME, DriveStorage
from app.gpu.google import GoogleService
from app.gpu.repository import Repository
from app.main import app as studio_app
from app.security import sign_session
from starlette.testclient import TestClient


class FakeDriveBackend:
    """In-memory Google Drive v3 mock supporting folders, multipart create, media patch, move, trash, delete."""

    def __init__(self) -> None:
        self.seq = 0
        self.files: dict[str, dict] = {}
        self.content: dict[str, bytes] = {}
        self.trashed_calls: list[str] = []
        self.deleted_calls: list[str] = []
        self.root_id = "root"

    def _next_id(self) -> str:
        self.seq += 1
        return f"fid-{self.seq}"

    def mkdir(self, name: str, parent: str) -> str:
        for fid, meta in self.files.items():
            if not meta["trashed"] and meta["mimeType"] == FOLDER_MIME and meta["name"] == name and parent in meta["parents"]:
                return fid
        fid = self._next_id()
        self.files[fid] = {
            "id": fid,
            "name": name,
            "mimeType": FOLDER_MIME,
            "parents": [parent],
            "size": "0",
            "modifiedTime": "2026-11-24T01:00:00Z",
            "trashed": False,
        }
        return fid

    def put_file(self, name: str, parent: str, raw: bytes, *, mime: str = "application/json", modified: str = "2026-11-24T01:00:00Z", size_override: int | None = None) -> str:
        fid = self._next_id()
        self.files[fid] = {
            "id": fid,
            "name": name,
            "mimeType": mime,
            "parents": [parent],
            "size": str(len(raw) if size_override is None else size_override),
            "modifiedTime": modified,
            "trashed": False,
        }
        self.content[fid] = raw
        return fid

    def put_json(self, name: str, parent: str, obj: dict, **kwargs) -> str:
        return self.put_file(name, parent, json.dumps(obj, ensure_ascii=False).encode("utf-8"), **kwargs)

    def children_named(self, parent: str, name: str) -> list[dict]:
        return [
            m for m in self.files.values()
            if not m["trashed"] and parent in m["parents"] and m["name"] == name
        ]

    def handler(self, request: httpx.Request) -> httpx.Response:
        parsed = urlsplit(str(request.url))
        path = parsed.path
        qs = parse_qs(parsed.query)
        method = request.method.upper()

        # Upload endpoints
        if path.startswith("/upload/drive/v3/files"):
            upload_type = (qs.get("uploadType") or [""])[0]
            if method == "POST" and upload_type == "multipart":
                ct = request.headers.get("content-type", "")
                m = re.search(r"boundary=([^\s;]+)", ct)
                assert m, f"missing boundary in {ct}"
                boundary = ("--" + m.group(1)).encode()
                parts = [p for p in request.content.split(boundary) if p.strip() and p.strip() != b"--"]
                meta_part = parts[0].split(b"\r\n\r\n", 1)[1].rstrip(b"\r\n")
                data_part = parts[1].split(b"\r\n\r\n", 1)[1].rstrip(b"\r\n")
                meta = json.loads(meta_part.decode("utf-8"))
                fid = self.put_file(meta["name"], meta["parents"][0], data_part, mime=meta.get("mimeType", "application/json"))
                return httpx.Response(200, json=self.files[fid])
            if method == "PATCH" and upload_type == "media":
                fid = path.rsplit("/", 1)[-1]
                if fid not in self.files:
                    return httpx.Response(404, json={"error": "not found"})
                self.content[fid] = request.content
                self.files[fid]["size"] = str(len(request.content))
                self.files[fid]["modifiedTime"] = f"2026-11-24T02:00:{self.seq:02d}Z"
                return httpx.Response(200, json=self.files[fid])

        # Metadata / download endpoints
        if path == "/drive/v3/files" and method == "GET":
            q = (qs.get("q") or [""])[0]
            matches = [m for m in self.files.values() if not m["trashed"]]
            if "mimeType = 'application/vnd.google-apps.folder'" in q:
                matches = [m for m in matches if m["mimeType"] == FOLDER_MIME]
            if "mimeType != 'application/vnd.google-apps.folder'" in q:
                matches = [m for m in matches if m["mimeType"] != FOLDER_MIME]
            m_name = re.search(r"name = '([^']*)'", q)
            if m_name:
                target_name = m_name.group(1).replace("\\'", "'").replace("\\\\", "\\")
                matches = [m for m in matches if m["name"] == target_name]
            m_parent = re.search(r"'([^']*)' in parents", q)
            if m_parent:
                parent_id = m_parent.group(1)
                matches = [m for m in matches if parent_id in m["parents"]]
            return httpx.Response(200, json={"files": matches})

        if path == "/drive/v3/files" and method == "POST":
            body = json.loads(request.content.decode("utf-8"))
            if body.get("mimeType") == FOLDER_MIME:
                fid = self.mkdir(body["name"], body["parents"][0])
                return httpx.Response(200, json=self.files[fid])
            fid = self.put_file(body["name"], body["parents"][0], b"", mime=body.get("mimeType", "application/octet-stream"))
            return httpx.Response(200, json=self.files[fid])

        if path.startswith("/drive/v3/files/"):
            fid = unquote(path.rsplit("/", 1)[-1])
            if method == "GET" and (qs.get("alt") or [""])[0] == "media":
                if fid not in self.files or self.files[fid]["trashed"]:
                    return httpx.Response(404)
                return httpx.Response(200, content=self.content.get(fid, b""))
            if method == "PATCH":
                if fid not in self.files:
                    return httpx.Response(404)
                body = json.loads(request.content.decode("utf-8")) if request.content else {}
                if "trashed" in body:
                    self.files[fid]["trashed"] = body["trashed"]
                    if body["trashed"]:
                        self.trashed_calls.append(fid)
                add_p = (qs.get("addParents") or [""])[0]
                rem_p = (qs.get("removeParents") or [""])[0]
                if add_p or rem_p:
                    parents = list(self.files[fid]["parents"])
                    if rem_p in parents:
                        parents.remove(rem_p)
                    if add_p and add_p not in parents:
                        parents.append(add_p)
                    self.files[fid]["parents"] = parents
                return httpx.Response(200, json=self.files[fid])
            if method == "DELETE":
                self.deleted_calls.append(fid)
                if fid in self.files:
                    del self.files[fid]
                    self.content.pop(fid, None)
                return httpx.Response(204)

        return httpx.Response(404, json={"error": f"unhandled {method} {path}"})


def _make_fake_drive_storage(repo: Repository, owner_id: str, backend: FakeDriveBackend) -> DriveStorage:
    client = httpx.Client(transport=httpx.MockTransport(backend.handler))

    class FakeGoogle(GoogleService):
        def access_token(self, purpose: str = "drive") -> str:
            return "fake-drive-token"

    return DriveStorage(
        {"id": "test-ws", "owner_id": owner_id},
        repo,
        owner_id,
        account_id=None,
        client=client,
        google=FakeGoogle(repo, owner_id, client=client),
    )


def _seed_workspace(backend: FakeDriveBackend, owner: str = "owner") -> dict:
    ai = backend.mkdir("AI-Studio", "root")
    ow = backend.mkdir(owner, ai)
    nc = backend.mkdir("native-comfy", ow)
    user = backend.mkdir("user", nc)
    default = backend.mkdir("default", user)
    wfs = backend.mkdir("workflows", default)
    warmup = backend.mkdir("warmup", nc)
    plugins_dir = backend.mkdir("plugins", nc)
    objects_dir = backend.mkdir("objects", plugins_dir)
    history_dir = backend.mkdir("history", plugins_dir)
    env_cache_dir = backend.mkdir("env-cache", nc)

    # Archives
    sha_lanpaint = "a" * 64
    sha_kj = "b" * 64
    sha_ui = "c" * 64
    sha_single = "d" * 64
    sha_mgr_dis = "e" * 64
    sha_orphan = "f" * 64

    backend.put_file(f"{sha_lanpaint}.tar.gz", objects_dir, b"x" * 1000, mime="application/gzip", size_override=180 * 1024 * 1024)
    backend.put_file(f"{sha_kj}.tar.gz", objects_dir, b"x" * 100, mime="application/gzip", size_override=1 * 1024 * 1024)
    backend.put_file(f"{sha_ui}.tar.gz", objects_dir, b"x" * 100, mime="application/gzip", size_override=30 * 1024 * 1024)
    backend.put_file(f"{sha_single}.tar.gz", objects_dir, b"x" * 50, mime="application/gzip", size_override=2048)
    backend.put_file(f"{sha_mgr_dis}.tar.gz", objects_dir, b"x" * 50, mime="application/gzip", size_override=5 * 1024 * 1024)
    backend.put_file(f"{sha_orphan}.tar.gz", objects_dir, b"x" * 50, mime="application/gzip", size_override=50 * 1024 * 1024)

    plugins_list = sorted(
        [
            {"path": "LanPaint", "checksum": "1" * 64, "enabled": True, "repo": None, "commit": None, "version": "2.1.0", "archive": sha_lanpaint, "kind": "directory"},
            {"path": "comfyui-kjnodes", "checksum": "2" * 64, "enabled": True, "repo": None, "commit": None, "version": "1.5.4", "archive": sha_kj, "kind": "directory"},
            {"path": "comfyui-autocomplete-aaalice", "checksum": "3" * 64, "enabled": True, "repo": None, "commit": None, "version": "1.13.8", "archive": sha_ui, "kind": "directory"},
            {"path": "websocket_image_save.py", "checksum": "4" * 64, "enabled": True, "repo": None, "commit": None, "version": None, "archive": sha_single, "kind": "file"},
            {"path": "old-tool.disabled", "checksum": "5" * 64, "enabled": False, "repo": None, "commit": None, "version": "0.1", "archive": sha_mgr_dis, "kind": "directory"},
        ],
        key=lambda x: x["path"],
    )
    gen_head = np_mod.digest(plugins_list)
    head_manifest = {"schema": 1, "plugins": plugins_list, "snapshot": {"git_custom_nodes": {}}, "backed_up_at": 1700000100.0, "generation": gen_head}
    backend.put_json("head.json", plugins_dir, head_manifest)
    backend.put_json("pending.json", plugins_dir, head_manifest)
    backend.put_json(f"{gen_head}.json", history_dir, head_manifest, modified="2026-11-24T01:00:00Z")

    # Older history snapshot referencing sha_orphan (beyond keep_history=1)
    old_plugins = [{"path": "legacy-node", "checksum": "9" * 64, "enabled": True, "repo": None, "commit": None, "version": "0.1", "archive": sha_orphan, "kind": "directory"}]
    gen_old = np_mod.digest(old_plugins)
    old_manifest = {"schema": 1, "plugins": old_plugins, "snapshot": {"git_custom_nodes": {}}, "backed_up_at": 1600000000.0, "generation": gen_old}
    backend.put_json(f"{gen_old}.json", history_dir, old_manifest, modified="2026-11-20T01:00:00Z")

    # Workflows:
    # 1. wf1 uses LanPaint + unresolved third-party DLSS5NeuralRender
    wf1 = {
        "nodes": [
            {"id": 1, "type": "LanPaint_KSampler", "mode": 0, "properties": {"cnr_id": "LanPaint"}},
            {"id": 2, "type": "DLSS5NeuralRender", "mode": 2, "properties": {"aux_id": "storyAura/DLSS5-Comfyui"}},
            {"id": 3, "type": "KSampler", "mode": 0, "properties": {"cnr_id": "comfy-core"}},
            {"id": 4, "type": "MarkdownNote", "mode": 0, "properties": {}},
        ]
    }
    backend.put_json("工作流1.json", wfs, wf1)

    # 2. anima uses frontend-only KJNodes GetNode/SetNode inside a subgraph!
    sub_dir = backend.mkdir("anime", wfs)
    anima_wf = {
        "nodes": [
            {"id": 1, "type": "sg-uuid-1", "mode": 0, "properties": {"cnr_id": "comfy-core"}},
        ],
        "definitions": {
            "subgraphs": [
                {
                    "id": "sg-uuid-1",
                    "name": "Pipeline",
                    "nodes": [
                        {"id": 10, "type": "GetNode", "mode": 0, "properties": {"aux_id": "kijai/ComfyUI-KJNodes"}},
                        {"id": 11, "type": "SetNode", "mode": 0, "properties": {"aux_id": "kijai/ComfyUI-KJNodes"}},
                    ],
                }
            ]
        },
    }
    backend.put_json("Anima-修复版.json", sub_dir, anima_wf)

    # 3. Oversized workflow file (>20 MiB) should not OOM or crash overview
    backend.put_file("huge.json", wfs, b"{}", size_override=25 * 1024 * 1024)

    # Env caches: one normal v4, one legacy oversized v2
    backend.put_file("native-env-v4-good.tar.gz", env_cache_dir, b"z" * 10, mime="application/gzip", size_override=35 * 1024 * 1024)
    backend.put_json(
        "native-env-v4-good.meta.json",
        env_cache_dir,
        {"schema": 4, "key": "native-env-v4-good", "size": 35 * 1024 * 1024, "created_at": 1700000200.0, "added_count": 40, "spec": {"comfy_commit": "423cbd20511b", "python": "3.13", "gpu": "gpu123"}},
    )
    backend.put_file("native-env-v2-bloated.tar.gz", env_cache_dir, b"z" * 10, mime="application/gzip", size_override=1580 * 1024 * 1024)
    backend.put_json(
        "native-env-v2-bloated.meta.json",
        env_cache_dir,
        {"schema": 2, "key": "native-env-v2-bloated", "size": 1580 * 1024 * 1024, "created_at": 1690000000.0, "added_count": 141, "spec": {"comfy_commit": "423cbd20511b", "python": "3.13", "gpu": "gpu123"}},
    )

    return {
        "nc": nc,
        "wfs": wfs,
        "warmup": warmup,
        "plugins_dir": plugins_dir,
        "objects_dir": objects_dir,
        "history_dir": history_dir,
        "env_cache_dir": env_cache_dir,
        "gen_head": gen_head,
        "gen_old": gen_old,
        "sha_orphan": sha_orphan,
        "sha_lanpaint": sha_lanpaint,
    }


def test_parse_import_times_and_collect_plugin_telemetry(tmp_path: Path):
    comfy = tmp_path / "ComfyUI"
    nodes = comfy / "custom_nodes"
    (nodes / "comfyui-easy-use" / "web").mkdir(parents=True)
    (nodes / "comfyui-easy-use" / "web" / "main.js").write_text("console.log(1)")
    (nodes / "comfyui-easy-use" / "requirements.txt").write_text("numpy\n")
    (nodes / "comfyui-impact-pack").mkdir(parents=True)
    (nodes / "comfyui-impact-pack" / "install.py").write_text("print('ok')\n")
    (nodes / "websocket_image_save.py").write_text("NODE_CLASS_MAPPINGS = {}\n")
    (nodes / "studio_native_state.py").write_text("# internal\n")

    startup_log = f"""
\x1b[32m[INFO]\x1b[0m Prestartup times for custom nodes:
\x1b[32m[INFO]\x1b[0m    0.8 seconds: {nodes}/studio_native_guard
\x1b[32m[INFO]\x1b[0m [START] ComfyUI-Manager
\x1b[32m[INFO]\x1b[0m
Import times for custom nodes:
\x1b[32m[INFO]\x1b[0m    0.0 seconds: {nodes}/websocket_image_save.py
\x1b[32m[INFO]\x1b[0m    0.1 seconds: {nodes}/studio_native_state.py
\x1b[32m[INFO]\x1b[0m    0.7 seconds: {nodes}/zml-image
\x1b[32m[INFO]\x1b[0m    5.4 seconds: {nodes}/comfyui-impact-pack
\x1b[32m[INFO]\x1b[0m    8.4 seconds: {nodes}/comfyui-easy-use
\x1b[32m[INFO]\x1b[0m    1.2 seconds (IMPORT FAILED): {nodes}/broken-node
\x1b[32m[INFO]\x1b[0m Starting server
"""
    times, failed = np_mod.parse_import_times(startup_log, nodes)
    assert times == {
        "websocket_image_save.py": 0.0,
        "zml-image": 0.7,
        "comfyui-impact-pack": 5.4,
        "comfyui-easy-use": 8.4,
        "broken-node": 1.2,
    }
    assert "studio_native_state.py" not in times
    assert failed == ["broken-node"]

    class FakeEasyNode:
        __module__ = "custom_nodes.comfyui-easy-use.py.nodes"

    class FakeWsNode:
        __module__ = "custom_nodes.websocket_image_save"

    tel = np_mod.collect_plugin_telemetry(
        tmp_path,
        comfy,
        startup_log,
        node_class_mappings={"easy pipeIn": FakeEasyNode, "SaveImageWebsocket": FakeWsNode},
        object_info={"ZML_PresetResolution": {"python_module": "custom_nodes.zml-image"}},
        extension_web_dirs={},
        gpu_reported="NVIDIA L4",
        comfy_commit="423cbd20511b",
    )
    assert tel["schema"] == 1
    assert tel["import_times"]["comfyui-easy-use"] == 8.4
    assert tel["import_times"]["zml-image"] == 0.7
    assert tel["node_map"]["easy pipeIn"] == "comfyui-easy-use"
    assert tel["node_map"]["SaveImageWebsocket"] == "websocket_image_save.py"
    assert tel["node_map"]["ZML_PresetResolution"] == "zml-image"
    assert tel["plugins"]["zml-image"]["node_count"] == 1
    assert tel["plugins"]["comfyui-easy-use"] == {
        "node_count": 1,
        "js_count": 1,
        "has_requirements": True,
        "has_install_py": False,
    }
    assert tel["plugins"]["comfyui-impact-pack"]["has_install_py"] is True
    assert "studio_native_state.py" not in tel["plugins"]

    tel_digest = np_mod.save_plugin_telemetry(tmp_path, tel)
    assert len(tel_digest) == 64
    assert json.loads((tmp_path / "plugins-telemetry-digest.json").read_text())["digest"] == tel_digest


def test_sync_plugins_fetches_telemetry_incrementally(tmp_path: Path, monkeypatch):
    store = Store(f"sqlite:///{tmp_path}/studio.db")
    repo = Repository(store)
    monkeypatch.setattr(native, "backend", lambda: ("https://internal.example.test", {"CF-Access-Client-Id": "id", "CF-Access-Client-Secret": "sec"}))

    tel_calls = 0
    tel_payload = {
        "schema": 1,
        "collected_at": 1700000000.0,
        "import_times": {"LanPaint": 1.5},
        "plugins": {"LanPaint": {"node_count": 4, "js_count": 2, "has_requirements": False, "has_install_py": False}},
        "node_map": {"LanPaint_KSampler": "LanPaint"},
    }
    tel_digest = np_mod.digest(tel_payload)
    manifest = {"schema": 1, "plugins": [], "generation": np_mod.digest([])}

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal tel_calls
        if req.url.path == "/studio_native/state":
            return httpx.Response(200, json={"busy": False, "plugins": {"manifest": manifest, "telemetry_digest": tel_digest, "durable": True}})
        if req.url.path == "/studio_native/plugin-telemetry":
            tel_calls += 1
            return httpx.Response(200, json=tel_payload)
        return httpx.Response(404)

    lease = {"id": "l1", "owner_id": "owner", "gpu_reported": "NVIDIA L4"}
    with httpx.Client(transport=httpx.MockTransport(handler)) as c:
        native.sync_plugins(repo, lease, client=c)
        native.sync_plugins(repo, lease, client=c)  # Same digest -> should NOT refetch telemetry!

    assert tel_calls == 1
    saved = repo.store.get("settings", "native-plugins-" + np_mod.digest("owner")[:32], "owner")
    assert saved["telemetry_digest"] == tel_digest
    assert saved["telemetry"]["gpu_reported"] == "NVIDIA L4"
    assert saved["telemetry"]["import_times"]["LanPaint"] == 1.5


def test_workspace_overview_linked_cleanup_disable_enable_rollback_and_cache(tmp_path: Path):
    native_workspace.invalidate_workspace_cache()
    store = Store(f"sqlite:///{tmp_path}/studio.db")
    repo = Repository(store)
    backend = FakeDriveBackend()
    seeded = _seed_workspace(backend)
    drive = _make_fake_drive_storage(repo, "owner", backend)

    # Seed telemetry into settings
    settings_key = "native-plugins-" + np_mod.digest("owner")[:32]
    repo.store.create(
        "settings",
        "owner",
        {
            "telemetry_digest": "t" * 64,
            "telemetry": {
                "schema": 1,
                "collected_at": 1700000150.0,
                "gpu_reported": "NVIDIA L4",
                "import_times": {"LanPaint": 2.3, "comfyui-kjnodes": 0.8, "comfyui-autocomplete-aaalice": 0.4},
                "plugins": {
                    "LanPaint": {"node_count": 12, "js_count": 0, "has_requirements": True, "has_install_py": False},
                    "comfyui-kjnodes": {"node_count": 267, "js_count": 31, "has_requirements": True, "has_install_py": False},
                    "comfyui-autocomplete-aaalice": {"node_count": 0, "js_count": 26, "has_requirements": False, "has_install_py": False},
                    "websocket_image_save.py": {"node_count": 1, "js_count": 0, "has_requirements": False, "has_install_py": False},
                },
                "node_map": {"LanPaint_KSampler": "LanPaint"},
            },
        },
        record_id=settings_key,
    )

    overview = native_workspace.get_workspace_overview(repo, "owner", refresh=True, keep_history=1, drive=drive)
    assert overview["offline_writable"] is True
    assert overview["drive_connected"] is True

    wf_by_path = {w["path"]: w for w in overview["workflows"]["active"]}
    assert wf_by_path["工作流1.json"]["plugins"] == ["LanPaint"]
    assert wf_by_path["工作流1.json"]["unresolved_nodes"][0]["type"] == "DLSS5NeuralRender"
    # Subgraph frontend virtual GetNode/SetNode detected via aux_id!
    assert wf_by_path["anime/Anima-修复版.json"]["plugins"] == ["comfyui-kjnodes"]
    assert wf_by_path["huge.json"]["parse_error"]

    pl_by_path = {p["path"]: p for p in overview["plugins"]["active"]}
    assert pl_by_path["LanPaint"]["status_category"] == "in_use"
    assert pl_by_path["LanPaint"]["import_seconds"] == 2.3
    assert pl_by_path["comfyui-kjnodes"]["status_category"] == "in_use"
    assert pl_by_path["comfyui-autocomplete-aaalice"]["status_category"] == "unused_ui"
    assert pl_by_path["websocket_image_save.py"]["status_category"] == "unused"
    assert pl_by_path["old-tool.disabled"]["status_category"] == "manager_disabled"

    # Env cache flags
    ec_by_key = {c["key"]: c for c in overview["env_caches"]["items"]}
    assert ec_by_key["native-env-v4-good"]["flags"] == []
    assert set(ec_by_key["native-env-v2-bloated"]["flags"]) == {"legacy_schema", "oversized"}

    # Trying to disable in_use LanPaint without force_in_use is rejected!
    with pytest.raises(APIError) as exc_info:
        native_workspace.apply_plugin_action(repo, "owner", action="disable", paths=["LanPaint"], force_in_use=False, drive=drive)
    assert exc_info.value.code == "PLUGIN_IN_USE"

    # Archive 工作流1.json + linked offline disable of LanPaint in one step!
    res = native_workspace.apply_workflow_action(
        repo,
        "owner",
        action="archive",
        items=[{"source": "comfy", "path": "工作流1.json"}],
        also_plugins={"action": "disable", "paths": ["LanPaint"]},
        drive=drive,
    )
    assert len(res["processed"]) == 1
    assert res["plugins_result"]["changed_paths"] == ["LanPaint"]

    # Verify in-place overwrite: still strictly ONE pending.json and ONE disabled.json in Drive!
    assert len(backend.children_named(seeded["plugins_dir"], "pending.json")) == 1
    assert len(backend.children_named(seeded["plugins_dir"], "disabled.json")) == 1

    # Also disable single-file plugin websocket_image_save.py and manager-disabled plugin
    res2 = native_workspace.apply_plugin_action(
        repo, "owner", action="disable", paths=["websocket_image_save.py", "old-tool.disabled"], drive=drive
    )
    assert set(res2["changed_paths"]) == {"websocket_image_save.py", "old-tool.disabled"}
    assert len(backend.children_named(seeded["plugins_dir"], "pending.json")) == 1
    assert len(backend.children_named(seeded["plugins_dir"], "disabled.json")) == 1

    # Verify overview now shows LanPaint, websocket_image_save.py, old-tool.disabled in disabled list
    ov2 = native_workspace.get_workspace_overview(repo, "owner", refresh=True, keep_history=1, drive=drive)
    assert ov2["plugin_state"]["has_pending_offline_changes"] is True
    assert {w["path"] for w in ov2["workflows"]["archived"]} == {"工作流1.json"}
    assert {p["path"] for p in ov2["plugins"]["disabled"]} == {"LanPaint", "websocket_image_save.py", "old-tool.disabled"}

    # Restore archived workflow back to active & re-enable LanPaint + websocket_image_save.py + old-tool.disabled
    native_workspace.apply_workflow_action(
        repo, "owner", action="restore", items=[{"source": "archived", "path": "工作流1.json"}], drive=drive
    )
    en_res = native_workspace.apply_plugin_action(
        repo, "owner", action="enable", paths=["LanPaint", "websocket_image_save.py", "old-tool.disabled"], drive=drive
    )
    # Exact canonical entry restoration matches original head generation!
    assert en_res["generation"] == seeded["gen_head"]

    # Trash workflow uses Drive trash (not permanent delete)
    native_workspace.apply_workflow_action(
        repo, "owner", action="trash", items=[{"source": "comfy", "path": "工作流1.json"}], drive=drive
    )
    assert len(backend.trashed_calls) >= 1

    # Delete single bloated env-cache, then clear all
    del1 = native_workspace.delete_env_caches(repo, "owner", keys=["native-env-v2-bloated"], drive=drive)
    assert del1["deleted_keys"] == ["native-env-v2-bloated"]
    assert del1["deleted_files"] == 2
    del2 = native_workspace.delete_env_caches(repo, "owner", delete_all=True, drive=drive)
    assert del2["deleted_keys"] == ["native-env-v4-good"]

    # Orphan object cleanup with keep_history=1 removes sha_orphan and makes old snapshot non-restorable
    dry = native_workspace.cleanup_orphan_objects(repo, "owner", keep_history=1, dry_run=True, drive=drive)
    assert dry["deleted_count"] == 1
    assert dry["items"][0]["sha256"] == seeded["sha_orphan"]

    cleaned = native_workspace.cleanup_orphan_objects(repo, "owner", keep_history=1, dry_run=False, drive=drive)
    assert cleaned["deleted_count"] == 1
    assert cleaned["freed_bytes"] == 50 * 1024 * 1024

    # Attempting to rollback to old generation whose archive was cleaned must fail with 409 ROLLBACK_ARCHIVE_MISSING
    with pytest.raises(APIError) as rb_exc:
        native_workspace.rollback_plugins(repo, "owner", generation=seeded["gen_old"], drive=drive)
    assert rb_exc.value.code == "ROLLBACK_ARCHIVE_MISSING"

    # Rollback to head generation succeeds
    rb_ok = native_workspace.rollback_plugins(repo, "owner", generation=seeded["gen_head"], drive=drive)
    assert rb_ok["generation"] == seeded["gen_head"]


@pytest.mark.parametrize("failure", ["disabled_unreadable", "pending_corrupt", "history_unreadable", "incomplete_listing"])
def test_orphan_cleanup_requires_complete_verified_manifests(tmp_path, monkeypatch, failure):
    repo = Repository(Store(f"sqlite:///{tmp_path}/studio.db"))
    backend = FakeDriveBackend()
    seeded = _seed_workspace(backend)
    archive = seeded["sha_orphan"]
    disabled_id = backend.put_json("disabled.json", seeded["plugins_dir"], {
        "schema": 1, "plugins": [{"entry": {"path": "disabled-node", "archive": archive}}],
    })
    pending_id = backend.children_named(seeded["plugins_dir"], "pending.json")[0]["id"]
    history_id = backend.children_named(seeded["history_dir"], f"{seeded['gen_head']}.json")[0]["id"]
    original_request = backend.handler

    def request(req):
        if req.method == "GET" and req.url.params.get("alt") == "media":
            if (failure == "disabled_unreadable" and req.url.path.endswith("/" + disabled_id)
                    or failure == "history_unreadable" and req.url.path.endswith("/" + history_id)):
                return httpx.Response(503)
            if failure == "pending_corrupt" and req.url.path.endswith("/" + pending_id):
                return httpx.Response(200, json={"schema": 1, "plugins": [], "generation": "bad-checksum"})
        response = original_request(req)
        if failure == "incomplete_listing" and req.url.path == "/drive/v3/files" and req.url.params.get("pageSize") == "1000":
            return httpx.Response(200, json={**response.json(), "incompleteSearch": True})
        return response

    monkeypatch.setattr(backend, "handler", request)
    drive = _make_fake_drive_storage(repo, "owner", backend)
    monkeypatch.setattr(drive, "sleeper", lambda delay: None)
    try:
        with pytest.raises(APIError) as exc_info:
            native_workspace.cleanup_orphan_objects(repo, "owner", keep_history=1, drive=drive)
        assert exc_info.value.code == {
            "disabled_unreadable": "WORKSPACE_MANIFEST_UNREADABLE",
            "pending_corrupt": "WORKSPACE_MANIFEST_INVALID",
            "history_unreadable": "WORKSPACE_MANIFEST_UNREADABLE",
            "incomplete_listing": "WORKSPACE_DRIVE_INCOMPLETE",
        }[failure]
        assert backend.deleted_calls == []
    finally:
        drive.close()


def test_linked_workflow_action_can_retry_after_conflict_and_drive_write_failure(tmp_path, monkeypatch):
    repo = Repository(Store(f"sqlite:///{tmp_path}/studio.db"))
    backend = FakeDriveBackend()
    seeded = _seed_workspace(backend)
    pending_id = backend.children_named(seeded["plugins_dir"], "pending.json")[0]["id"]
    pending_base = backend.content[pending_id]
    workflow = {"source": "comfy", "path": "工作流1.json"}
    original_request = backend.handler
    reject_pending_write = False

    def request(req):
        if reject_pending_write and req.method == "PATCH" and req.url.path == f"/upload/drive/v3/files/{pending_id}":
            return httpx.Response(503)
        return original_request(req)

    monkeypatch.setattr(backend, "handler", request)
    drive = _make_fake_drive_storage(repo, "owner", backend)
    monkeypatch.setattr(drive, "sleeper", lambda delay: None)
    try:
        archived = backend.mkdir("workflows-archived", seeded["nc"])
        original_archive = backend.put_json(workflow["path"], archived, {"older": True})
        with pytest.raises(APIError) as conflict:
            native_workspace.apply_workflow_action(
                repo, "owner", action="archive", items=[workflow], overwrite=True,
                also_plugins={"action": "disable", "paths": ["comfyui-kjnodes"]}, drive=drive,
            )
        assert conflict.value.code == "PLUGIN_IN_USE"
        assert backend.trashed_calls == []
        assert backend.children_named(seeded["wfs"], workflow["path"])

        reject_pending_write = True
        with pytest.raises(APIError) as failed_write:
            native_workspace.apply_workflow_action(
                repo, "owner", action="archive", items=[workflow], overwrite=True,
                also_plugins={"action": "disable", "paths": ["LanPaint"]}, drive=drive,
            )
        assert failed_write.value.code == "WORKSPACE_DRIVE_ERROR"
        assert backend.children_named(seeded["wfs"], workflow["path"])
        assert backend.children_named(archived, workflow["path"])[0]["id"] == original_archive
        assert backend.content[pending_id] == pending_base
        assert backend.children_named(seeded["plugins_dir"], "disabled.json") == []

        reject_pending_write = False
        result = native_workspace.apply_workflow_action(
            repo, "owner", action="archive", items=[workflow], overwrite=True,
            also_plugins={"action": "disable", "paths": ["LanPaint"]}, drive=drive,
        )
        assert result["plugins_result"]["changed_paths"] == ["LanPaint"]
        assert backend.children_named(seeded["wfs"], workflow["path"]) == []
    finally:
        drive.close()


def test_plugin_mutations_and_conflict_checks_fail_closed_on_drive_errors(tmp_path, monkeypatch):
    repo = Repository(Store(f"sqlite:///{tmp_path}/studio.db"))
    backend = FakeDriveBackend()
    seeded = _seed_workspace(backend)
    disabled_id = backend.put_json("disabled.json", seeded["plugins_dir"], {
        "schema": 1,
        "plugins": [{"entry": {"path": "kept-disabled", "archive": seeded["sha_orphan"]}}],
    })
    disabled_before = backend.content[disabled_id]
    pending_id = backend.children_named(seeded["plugins_dir"], "pending.json")[0]["id"]
    pending_before = backend.content[pending_id]
    anima_dir = backend.children_named(seeded["wfs"], "anime")[0]["id"]
    anima_id = backend.children_named(anima_dir, "Anima-修复版.json")[0]["id"]
    original_request = backend.handler
    fail_target = {"mode": None}

    def request(req):
        if req.method == "GET" and req.url.params.get("alt") == "media":
            if fail_target["mode"] == "disabled_503" and req.url.path.endswith("/" + disabled_id):
                return httpx.Response(503)
            if fail_target["mode"] == "pending_503" and req.url.path.endswith("/" + pending_id):
                return httpx.Response(503)
            if fail_target["mode"] == "workflow_503" and req.url.path.endswith("/" + anima_id):
                return httpx.Response(503)
        response = original_request(req)
        if fail_target["mode"] == "incomplete" and req.url.path == "/drive/v3/files" and req.url.params.get("pageSize") == "1000":
            return httpx.Response(200, json={**response.json(), "incompleteSearch": True})
        return response

    monkeypatch.setattr(backend, "handler", request)
    drive = _make_fake_drive_storage(repo, "owner", backend)
    monkeypatch.setattr(drive, "sleeper", lambda delay: None)
    try:
        # 1. Transient 503 on disabled.json during plugin disable must NOT overwrite & wipe existing disabled.json
        fail_target["mode"] = "disabled_503"
        with pytest.raises(APIError) as exc1:
            native_workspace.apply_plugin_action(
                repo, "owner", action="disable", paths=["comfyui-autocomplete-aaalice"], force_in_use=True, drive=drive,
            )
        assert exc1.value.code == "WORKSPACE_MANIFEST_UNREADABLE"
        assert backend.content[disabled_id] == disabled_before
        assert backend.content[pending_id] == pending_before

        # 2. Transient 503 on pending.json during plugin disable must NOT silently fall back to head.json
        fail_target["mode"] = "pending_503"
        with pytest.raises(APIError) as exc2:
            native_workspace.apply_plugin_action(
                repo, "owner", action="disable", paths=["comfyui-autocomplete-aaalice"], force_in_use=True, drive=drive,
            )
        assert exc2.value.code == "WORKSPACE_MANIFEST_UNREADABLE"
        assert backend.content[pending_id] == pending_before

        # 3. Incomplete Drive listing blocks rollback_plugins
        fail_target["mode"] = "incomplete"
        with pytest.raises(APIError) as exc3:
            native_workspace.rollback_plugins(repo, "owner", generation=seeded["gen_head"], drive=drive)
        assert exc3.value.code == "WORKSPACE_DRIVE_INCOMPLETE"

        # 4. Transient 503 reading another active workflow JSON must fail closed during PLUGIN_IN_USE check
        fail_target["mode"] = "workflow_503"
        with pytest.raises(APIError) as exc4:
            native_workspace.apply_workflow_action(
                repo, "owner", action="archive", items=[{"source": "comfy", "path": "工作流1.json"}],
                also_plugins={"action": "disable", "paths": ["comfyui-kjnodes"]}, drive=drive,
            )
        assert exc4.value.code == "WORKSPACE_DRIVE_ERROR"
        # Neither workflow moved nor empty destination folder created on pre-check failure!
        assert backend.children_named(seeded["wfs"], "工作流1.json")
        assert backend.children_named(seeded["nc"], "workflows-archived") == []

        # 5. Clean 409 PLUGIN_IN_USE also creates zero empty destination folders in Drive
        fail_target["mode"] = None
        with pytest.raises(APIError) as exc5:
            native_workspace.apply_workflow_action(
                repo, "owner", action="archive", items=[{"source": "comfy", "path": "工作流1.json"}],
                also_plugins={"action": "disable", "paths": ["comfyui-kjnodes"]}, drive=drive,
            )
        assert exc5.value.code == "PLUGIN_IN_USE"
        assert backend.children_named(seeded["nc"], "workflows-archived") == []
    finally:
        drive.close()


def test_offline_lock_blocks_writes_when_native_active_and_blocks_admit_while_writing(tmp_path: Path, monkeypatch):
    store = Store(f"sqlite:///{tmp_path}/studio.db")
    repo = Repository(store)
    backend = FakeDriveBackend()
    _seed_workspace(backend)
    drive = _make_fake_drive_storage(repo, "owner", backend)

    # 1. Creation-pool lease on same account must NOT block native offline management
    repo.create(
        "leases",
        "owner",
        {"instance_id": "inst-c", "state": "idle", "native_comfy": False, "instance_pool": "creation"},
        record_id="lease-creation",
    )
    writable, reason = native.workspace_offline_status(repo, "owner")
    assert writable is True and reason is None

    # 2. Active native lease MUST block writes
    repo.create(
        "leases",
        "owner",
        {"instance_id": "inst-n", "state": "idle", "native_comfy": True, "native_ready": True},
        record_id="lease-native",
    )
    with pytest.raises(APIError) as exc1:
        native_workspace.delete_env_caches(repo, "owner", delete_all=True, drive=drive)
    assert exc1.value.code == "NATIVE_INSTANCE_ACTIVE"
    repo.update("leases", "lease-native", {"state": "deleted"}, "owner")

    # 3. Non-terminal native start job MUST block writes
    repo.create(
        "jobs",
        "owner",
        {"state": "allocating", "snapshot": {"native_comfy": True}},
        record_id="job-native-starting",
    )
    with pytest.raises(APIError) as exc2:
        native_workspace.apply_plugin_action(repo, "owner", action="disable", paths=["LanPaint"], force_in_use=True, drive=drive)
    assert exc2.value.code == "NATIVE_INSTANCE_ACTIVE"
    repo.update("jobs", "job-native-starting", {"state": "completed"}, "owner")

    # 4. Holding workspace write lock MUST block native.admit()
    lock_id = native.acquire_workspace_write_lock(repo, "owner", reason="test")
    try:
        monkeypatch.setattr(settings, "native_comfy_url", "https://comfy.example.test")
        monkeypatch.setattr(settings, "native_comfy_backend_url", "https://backend.example.test")
        monkeypatch.setenv("NATIVE_COMFY_COMMIT", "a" * 40)
        monkeypatch.setenv("GPU_REAL_ENABLED", "1")
        monkeypatch.setattr(native, "backend", lambda: ("https://backend.example.test", {"CF-Access-Client-Id": "i", "CF-Access-Client-Secret": "s"}))
        monkeypatch.setattr(native, "secret_file", lambda name: "token-value")
        with pytest.raises(APIError) as admit_exc:
            native.admit(
                repo,
                "owner",
                {"account_id": "acc-1", "spec": {"key": "L4"}},
                "idem-lock-test",
            )
        assert admit_exc.value.code == "NATIVE_WORKSPACE_LOCKED"
    finally:
        native.release_workspace_write_lock(repo, "owner", lock_id)


def test_workspace_http_routes_auth_and_contract(tmp_path: Path, monkeypatch):
    store = Store(f"sqlite:///{tmp_path}/studio.db")
    repo = Repository(store)
    backend = FakeDriveBackend()
    _seed_workspace(backend)
    drive = _make_fake_drive_storage(repo, "owner", backend)

    monkeypatch.setattr(native_workspace, "_open_drive", lambda r, o, drive=None: (drive_obj, False, "acc-1", "主盘"))
    drive_obj = drive
    monkeypatch.setattr(studio_app.state, "store", store, raising=False)
    monkeypatch.setattr(studio_app.state, "repo", repo, raising=False)

    monkeypatch.setattr(settings, "session_secret", "test-session-secret")
    monkeypatch.setattr(settings, "studio_api_token", "test-api-token")
    token = sign_session({"uid": "owner", "role": "admin", "exp": int(time.time()) + 3600})
    with TestClient(studio_app, base_url=settings.site_url) as client:
        # Unauthenticated rejected
        assert client.get("/api/gpu/native/workspace/overview").status_code == 401

        auth_headers = {"Authorization": f"Bearer {token}"}
        resp = client.get("/api/gpu/native/workspace/overview?refresh=true", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()["data"]
        assert body["drive_connected"] is True
        assert body["offline_writable"] is True
        assert len(body["workflows"]["active"]) == 3

        # Delete cache via DELETE endpoint
        del_resp = client.request("DELETE", "/api/gpu/native/workspace/env-cache", json={"keys": ["native-env-v2-bloated"]}, headers=auth_headers)
        assert del_resp.status_code == 200
        assert del_resp.json()["data"]["deleted_keys"] == ["native-env-v2-bloated"]
