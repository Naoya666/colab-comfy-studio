"""Offline source/Drive integration fakes: no platform credentials, CU or model bytes required."""
import copy
import hashlib
import json

import httpx
import pytest
from app.config import settings
from app.db import Store
from app.gpu import model_dependencies as deps
from app.gpu.model_library import ModelLibrary, ModelTransfer, transfer_lock
from app.gpu.model_sources import ModelError, SourceHTTP, Sources, filename, https_url, public_addresses
from app.gpu.registry import Registry
from app.gpu.repository import Repository
from app.gpu.runtime_worker import _restore_weights
from cryptography.fernet import Fernet

DATA = b"a small model payload"
HASH = hashlib.sha256(DATA).hexdigest()
MD5 = hashlib.md5(DATA, usedforsecurity=False).hexdigest()


@pytest.fixture
def repo(monkeypatch):
    monkeypatch.setattr(settings, "encryption_key", Fernet.generate_key().decode())
    return Repository(Store("sqlite:///:memory:"))


def source_http(handler, tokens=None):
    return SourceHTTP(tokens, client=httpx.Client(transport=httpx.MockTransport(handler)), resolver=lambda h: ["8.8.8.8"])


def model(lib, **extra):
    return lib.add({"filename": "base.safetensors", "model_type": "checkpoints", "architecture": "SDXL",
                    "source": "https", "download_url": "https://example.com/weights", **extra})


def workflow(repo, **extra):
    return repo.create("workflows", "owner", {"workflow_key": "test", "workflow_version": 1,
                       "manifest": {}, "api_graph": {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "base.safetensors"}}},
                       "state": "draft", **extra})


@pytest.mark.parametrize("path", ["../x.safetensors", "/x.safetensors", "x/../../a.safetensors", "a\\b.safetensors", "a.py", "a.ckpt", "a.pt", "a.bin", "a\x00.safetensors", "a:b.safetensors"])
def test_file_safety(path):
    with pytest.raises(ModelError):
        filename(path)


@pytest.mark.parametrize("url", ["http://example.com/a", "file:///etc/passwd", "https://u:p@example.com/a", "https://example.com:8443/a", "https://example.com/a?token=private", "https://example.com/a?X-Amz-Credential=private", "https://example.com/a?sig=private"])
def test_url_safety(url):
    with pytest.raises(ModelError):
        https_url(url)


@pytest.mark.parametrize("address", ["127.0.0.1", "10.1.1.1", "169.254.169.254", "::1", "::ffff:127.0.0.1", "100.64.0.1"])
def test_internal_dns_refused(monkeypatch, address):
    monkeypatch.setattr("socket.getaddrinfo", lambda *a, **k: [(None, None, None, None, (address, 443))])
    with pytest.raises(ModelError):
        public_addresses("unsafe.example")


def test_redirect_pins_dns_and_does_not_leak_authorization(caplog):
    calls = []
    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(302, headers={"location": "https://cdn.example/model?signature=private-signature",
                                               "set-cookie": "private=secret; Path=/"})
        return httpx.Response(200, content=DATA)
    http = source_http(handler, {"civitai": "secret"})
    with caplog.at_level("DEBUG"):
        with http.stream("https://civitai.com/api/download/models/1") as response:
            assert response.read() == DATA
    assert calls[0].url.host == "8.8.8.8" and calls[0].headers["host"] == "civitai.com"
    assert calls[0].extensions["sni_hostname"] == b"civitai.com"
    assert calls[0].headers["authorization"] == "Bearer secret"
    assert "authorization" not in calls[1].headers and "cookie" not in calls[1].headers
    assert "private-signature" not in caplog.text


def test_redirect_to_internal_network_rejected():
    http = source_http(lambda r: httpx.Response(302, headers={"location": "https://internal.example/x"}))
    http.resolver = lambda host: ["8.8.8.8"] if host == "example.com" else (_ for _ in ()).throw(ModelError("internal"))
    with pytest.raises(ModelError, match="internal"):
        with http.stream("https://example.com/x"):
            pass


@pytest.mark.parametrize("status,part", [(401,"鉴权"),(403,"许可"),(404,"不存在"),(429,"限流")])
def test_source_errors_safe(status, part):
    http = source_http(lambda r: httpx.Response(status, text="secret-token-must-not-leak"))
    with pytest.raises(ModelError, match=part) as error:
        http.json("https://example.com/model")
    assert "secret-token" not in str(error.value)


def test_hf_pins_commit_and_lists_files_without_whole_repo_download():
    http = source_http(lambda r: httpx.Response(200, json={"sha":"a"*40, "siblings":[
        {"rfilename":"folder/one.safetensors", "lfs":{"sha256":HASH,"size":len(DATA)}},
        {"rfilename":"two.gguf"}, {"rfilename":"code.py"}]}))
    files = Sources(http).resolve({"url":"https://huggingface.co/org/repo"})
    assert len(files) == 2 and files[0]["source_identity"]["revision"] == "a"*40
    assert files[0]["architecture"] == "unknown" and files[1]["expected_sha256"] is None
    selected = Sources(http).resolve({"url":"https://huggingface.co/org/repo/blob/main/folder/one.safetensors"})
    assert len(selected) == 1 and selected[0]["expected_size"] == len(DATA)


def test_civitai_version_and_hash_preserve_file_identity():
    version = {"id": 12, "modelId":1, "name":"v1", "baseModel":"Illustrious", "trainedWords":["style"],
               "model":{"name":"Example","type":"LORA"}, "files":[
                   {"id":31,"name":"a.safetensors","sizeKB":1.2,"hashes":{"SHA256":HASH},"downloadUrl":"https://civitai.com/api/download/models/12?type=Model"},
                   {"id":32,"name":"b.safetensors","hashes":{"AutoV2":"1234"},"downloadUrl":"https://civitai.com/api/download/models/12?fp=fp16"}]}
    sources = Sources(source_http(lambda r: httpx.Response(200, json=version)))
    files = sources.resolve({"url":"https://civitai.com/models/1?modelVersionId=12"})
    assert len(files) == 2 and files[0]["source_identity"] == {"model_id":1,"version_id":12,"file_id":31}
    assert files[0]["trigger_words"] == ["style"] and files[0]["expected_size"] is None
    assert files[1]["expected_sha256"] is None
    assert len(sources.resolve({"sha256":HASH})) == 1
    with pytest.raises(ModelError):
        sources.resolve({"sha256":"12345678"})


def test_dependency_scanner_avoids_prompts_and_handles_optional_lora():
    graph = {"1":{"class_type":"CheckpointLoaderSimple","inputs":{"ckpt_name":"base.safetensors"}},
             "2":{"class_type":"LoraLoader","inputs":{"lora_name":"lora.safetensors","strength_model":0,"strength_clip":0}},
             "3":{"class_type":"VAELoader","inputs":{"vae_name":"vae.safetensors"}},
             "4":{"class_type":"CLIPTextEncode","inputs":{"text":"photo file.safetensors embedding:negative"}},
             "5":{"class_type":"SaveImage","inputs":{"filename_prefix":"output.safetensors"}},
             "6":{"class_type":"UnknownModelLoader","inputs":{"name":"maybe.safetensors"}}}
    before = copy.deepcopy(graph)
    found = deps.scan({"api_graph":graph})
    assert len(found) == 5 and graph == before
    assert found[1]["required"] is False and found[-1]["status"] == "manual_review"
    assert not any(d["reference"] in {"file.safetensors","output.safetensors","maybe.safetensors"} for d in found)


def test_workflow_only_widgets_remain_non_executable():
    wf = {"image_extraction":{"workflow":{"nodes":[{"id":1,"type":"VAELoader","widgets_values":["v.safetensors"]},
                                                                  {"id":2,"type":"UnknownLoader","widgets_values":["unknown.safetensors"]}]}}}
    before = copy.deepcopy(wf)
    result = deps.scan(wf)
    assert result[0]["reference"] == "v.safetensors" and wf == before and "api_graph" not in wf
    assert result[1]["status"] == "manual_review"


def test_type_and_architecture_binding_guards():
    d = {"model_type":"checkpoints", "architecture":"SDXL"}
    with pytest.raises(ModelError, match="架构不兼容"):
        deps.bind(d, {"id":"m","model_type":"checkpoints","architecture":"FLUX"})
    with pytest.raises(ModelError, match="架构未知"):
        deps.bind(d, {"id":"m","model_type":"checkpoints","architecture":"unknown"})
    with pytest.raises(ModelError, match="类型"):
        deps.bind(d, {"id":"m","model_type":"vae","architecture":"SDXL"})


def test_filename_match_is_candidate_even_if_unique(repo):
    lib = ModelLibrary(repo,"owner")
    m = model(lib)
    w = workflow(repo)
    record = lib.dependency_record(w["id"])
    d = record["dependencies"][0]
    assert d["model_id"] is None and d["candidates"] == [m["id"]]
    assert not record["models_ready"]
    lib.bind(w["id"],d["id"],m["id"],confirm_unknown=True)
    assert len(lib.references(m["id"])) == 1
    with pytest.raises(ModelError, match="引用"):
        lib.delete(m["id"])
    lib.delete(m["id"],confirm_references=True)
    assert lib.dependency_record(w["id"])["dependencies"][0]["model_id"] is None


def test_encrypted_source_credentials_never_in_status(repo):
    lib = ModelLibrary(repo,"owner")
    lib.save_token("civitai","never-show-me")
    assert lib.tokens()["civitai"] == "never-show-me"
    assert "never-show-me" not in json.dumps(repo.list("credentials"))
    assert lib.token_status() == {"huggingface":False,"civitai":True}
    assert ModelLibrary(repo,"other").token_status()["civitai"] is False
    lib.save_token("civitai","")
    assert not lib.token_status()["civitai"]


def test_queue_idempotent_cancel_and_retry(repo):
    lib=ModelLibrary(repo,"owner")
    m=model(lib)
    first=lib.enqueue(m["id"])
    assert lib.enqueue(m["id"])["id"] == first["id"]
    lib.cancel(first["id"])
    assert lib.get(m["id"])["state"] == "failed"
    assert lib.enqueue(m["id"])["id"] != first["id"]
    assert ModelLibrary(repo,"other").models() == []


class FakeDrive:
    _owns_client = False
    base_url = "https://www.googleapis.com/drive/v3"
    upload_url = "https://www.googleapis.com/upload/drive/v3"
    def __init__(self):
        self.files = {}
        self.bytes = bytearray()
        self.uploads = 0
    def model_cache(self):
        return {"folder_id":"root","access_token":"t"*30}
    def _folder(self, name, parent):
        return parent + "_" + name
    def _q(self, value):
        return value
    def _ensure_ok(self, response, action):
        if response.status_code >= 300:
            raise ModelError("Drive " + action + "失败")
        return response.json()
    def _request(self, method, url, **kwargs):
        if method == "GET" and url.endswith('/generateIds'):
            return httpx.Response(200, json={"ids": ["remote"]})
        if method == "GET" and url.endswith('/files'):
            return httpx.Response(200, json={"files":list(self.files.values())})
        if method == "GET":
            value=self.files.get(url.rsplit('/',1)[-1])
            return httpx.Response(200 if value else 404, json=value or {})
        if method == "POST":
            self.uploads += 1
            self.bytes = bytearray()
            return httpx.Response(200, headers={"location":"https://www.googleapis.com/upload/drive/v3/files?upload_id=session"})
        if method == "PUT":
            for chunk in kwargs["content"]:
                self.bytes.extend(chunk)
            self.files["remote"]={"id":"remote","size":str(len(self.bytes)),"mimeType":"application/octet-stream", "trashed":False,
                                  "md5Checksum":hashlib.md5(self.bytes,usedforsecurity=False).hexdigest()}
            return httpx.Response(200,json=self.files["remote"])
        raise AssertionError(method)


def test_transfer_full_download_verify_upload_and_physical_dedup(repo):
    drive=FakeDrive()
    count=[]
    def handler(request):
        count.append(request)
        return httpx.Response(200,content=DATA)
    lib=ModelLibrary(repo,"owner",drive=drive,http=source_http(handler))
    a=model(lib,expected_sha256=HASH,expected_size=len(DATA))
    ModelTransfer(lib,lib.enqueue(a["id"])).run()
    ready=lib.get(a["id"])
    assert ready["state"] == "ready" and ready["sha256"] == HASH and ready["size"] == len(DATA)
    assert ready["integrity"] == "expected_hash_verified" and drive.uploads == 1
    b=model(lib,filename="different.safetensors",model_type="loras",expected_sha256=HASH)
    ModelTransfer(lib,lib.enqueue(b["id"])).run()
    assert lib.get(b["id"])["drive_file_id"] == ready["drive_file_id"]
    assert len(count) == 1 and drive.uploads == 1
    assert not list((settings.data_dir / "model-scratch").glob("transfer-*"))


def test_no_expected_hash_is_computed_but_not_source_approval(repo):
    lib=ModelLibrary(repo,"owner",drive=FakeDrive(),http=source_http(lambda r:httpx.Response(200,content=DATA)))
    a=model(lib)
    ModelTransfer(lib,lib.enqueue(a["id"])).run()
    m=lib.get(a["id"])
    assert m["sha256"] == HASH and m["integrity"] == "computed_hash_only"
    assert not m.get("review_approved")


def test_hash_failure_never_uploads_or_marks_ready(repo):
    drive=FakeDrive()
    lib=ModelLibrary(repo,"owner",drive=drive,http=source_http(lambda r:httpx.Response(200,content=DATA)))
    a=model(lib,expected_sha256="0"*64)
    task=lib.enqueue(a["id"])
    ModelTransfer(lib,task).run()
    assert lib.get(a["id"])["state"] == "failed" and "SHA-256" in lib.get(a["id"])["error"]
    assert drive.uploads == 0
    assert not list((settings.data_dir / "model-scratch").glob("transfer-*"))


def test_remote_missing_and_external_delete_policy(repo):
    lib=ModelLibrary(repo,"owner",drive=FakeDrive())
    a=model(lib,source="drive",drive_file_id="missing")
    with pytest.raises(ModelError, match="不存在"):
        lib.check_remote(a["id"])
    assert lib.get(a["id"])["state"] == "missing"
    with pytest.raises(ModelError, match="原文件"):
        lib.delete(a["id"],delete_file=True)


def test_restart_task_is_retryable_and_lock_is_exclusive(repo):
    drive=FakeDrive()
    lib=ModelLibrary(repo,"owner",drive=drive,http=source_http(lambda r:httpx.Response(200,content=DATA)))
    a=model(lib)
    task=lib.enqueue(a["id"])
    repo.update("model_downloads",task["id"],{"state":"uploading","attempt":1},"owner")
    ModelTransfer(lib,repo.get("model_downloads",task["id"],"owner")).run()
    assert repo.get("model_downloads",task["id"],"owner")["attempt"] == 2
    assert lib.get(a["id"])["state"] == "ready"
    with transfer_lock() as first:
        with transfer_lock() as second:
            assert first and not second


def test_runtime_reuses_local_only_current_manifest(tmp_path):
    target=tmp_path / "models/checkpoints/base.safetensors"
    target.parent.mkdir(parents=True)
    target.write_bytes(DATA)
    manifest={"weight_files":[{"source":"drive","drive_file_id":"remote-file-id-12345", "target":"checkpoints/base.safetensors","sha256":HASH,"size":len(DATA)}]}
    # No Drive cache/transport available: success proves only verified local bytes were used.
    records=_restore_weights(tmp_path,manifest,60,None,False)
    assert records[0]["source"] == "local" and target.read_bytes() == DATA


def test_runtime_mapping_copies_original_and_blocks_unreviewed_replacements():
    dep={"id":"d","node_id":"1","field":"ckpt_name","reference":"alias.safetensors","model_type":"checkpoints","required":True,"model_id":"m","expected_sha256":HASH}
    m={"id":"m","state":"ready","sha256":HASH,"size":len(DATA),"drive_file_id":"remote"}
    snapshot={"manifest":{"weight_files":[{"target":"checkpoints/alias.safetensors","sha256":HASH}]},"bound_graph":{"1":{"inputs":{"ckpt_name":"alias.safetensors"}}}}
    original=copy.deepcopy(snapshot)
    mapped=deps.runtime_copy(snapshot,[dep],[m])
    assert snapshot == original and mapped["manifest"]["weight_files"][0]["source"] == "drive"
    with pytest.raises(ModelError, match="审核"):
        deps.runtime_copy(snapshot,[dep],[{**m,"sha256":"0"*64}])


def test_png_import_preserves_original_and_adds_scan(repo):
    visual={"nodes":[{"id":1,"type":"VAELoader","widgets_values":["v.safetensors"]}]}
    row=Registry(repo).import_workflow("owner",{"workflow_key":"png", "image_extraction":{"workflow":visual}})
    assert row["image_extraction"]["workflow"] == visual and row["api_graph"] == {}
    assert row["model_dependency_scan"][0]["reference"] == "v.safetensors"
    assert row["state"] == "draft" and not row["smoke_test_passed"]


def test_nested_multi_lora_and_explicit_metadata_hints():
    graph={"1":{"class_type":"Power Lora Loader (rgthree)","inputs":{
        "lora_1":{"on":True,"lora":"style.safetensors","strength":0.8},
        "lora_2":{"on":False,"lora":"optional.safetensors","strength":1}}}}
    visual={"nodes":[{"id":1,"properties":{"models":[{"name":"style.safetensors","directory":"loras", "url":"https://example.com/style.safetensors","sha256":HASH}]}}]}
    found=deps.scan({"api_graph":graph,"editable_graph":visual})
    assert len(found) == 2 and found[0]["field"] == "lora_1.lora" and found[1]["required"] is False
    assert found[0]["source_hint"] == "https://example.com/style.safetensors" and found[0]["expected_sha256"] == HASH
    assert found[0]["strengths"] == {"lora_1.strength":0.8}


def test_enabling_optional_lora_changes_required_and_invalid_strength_rejected():
    d=deps.scan({"api_graph":{"1":{"class_type":"LoraLoaderModelOnly","inputs":{"lora_name":"a.safetensors","strength_model":0}}}})[0]
    assert not d["required"]
    m={"id":"m","model_type":"loras","architecture":"unknown"}
    assert deps.bind(d,m,confirm_unknown=True,strengths={"strength_model":1})["required"]
    with pytest.raises(ModelError):
        deps.bind(d,m,confirm_unknown=True,strengths={"new_field":1})
    with pytest.raises(ModelError):
        deps.bind(d,m,confirm_unknown=True,strengths={"strength_model":float('nan')})


def test_disabling_lora_changes_only_runtime_graph():
    d={"id":"d","node_id":"1","field":"lora_name","reference":"optional.safetensors","model_type":"loras","disabled":True,"strengths":{"strength_model":1}}
    snapshot={"manifest":{},"bound_graph":{"1":{"inputs":{"strength_model":1}}}}
    result=deps.runtime_copy(snapshot,[d],[])
    assert snapshot["bound_graph"]["1"]["inputs"]["strength_model"] == 1
    assert result["bound_graph"]["1"]["inputs"]["strength_model"] == 0


def test_library_model_registration_has_real_source_identity_not_fake_hf_revision(repo):
    lib=ModelLibrary(repo,"owner")
    m=model(lib,source="drive",drive_file_id="existing-file-123456")
    lib.update(m["id"],state="ready",sha256=HASH,size=len(DATA))
    family=lib.register_workflow_model(m["id"],{"model_key":"custom-sdxl","license":"user-reviewed","resource_profile":{"vram_gb":12}})
    assert family["source_kind"] == "drive" and family["revision"] is None
    assert not family["enabled"] and not family["review_approved"]


def test_storage_account_is_owned_and_cannot_move_existing_catalog(repo):
    lib=ModelLibrary(repo,"owner")
    account=repo.create("accounts","owner",{"alias":"Drive"})
    assert lib.save_storage(account["id"])["account_id"] == account["id"]
    model(lib)
    with pytest.raises(ModelError,match="不能直接切换"):
        lib.save_storage(None)
    with pytest.raises(ModelError,match="不属于"):
        ModelLibrary(repo,"other").save_storage(account["id"])


def test_same_hash_with_distinct_references_keeps_both_catalog_paths(repo):
    lib=ModelLibrary(repo,"owner")
    a=model(lib,expected_sha256=HASH)
    b=model(lib,filename="alias.safetensors",expected_sha256=HASH)
    c=model(lib, relative_path="nested/base.safetensors", expected_sha256=HASH)
    assert len({a["id"], b["id"], c["id"]}) == 3
    assert model(lib, expected_sha256=HASH)["id"] == a["id"]
    first = lib.enqueue(a["id"])
    assert lib.enqueue(a["id"])["id"] == first["id"]
    assert lib.enqueue(b["id"])["id"] != first["id"]


def test_optional_lora_is_not_restored_when_disabled():
    d={"id":"d","node_id":"2","field":"lora_name","reference":"optional.safetensors","model_type":"loras","required":False,"strengths":{"strength_model":0}}
    snapshot={"manifest":{"weight_files":[{"target":"loras/optional.safetensors"},{"target":"checkpoints/base.safetensors"}]},
              "bound_graph":{"2":{"inputs":{"strength_model":0}}}}
    result=deps.runtime_copy(snapshot,[d],[])
    assert [w['target'] for w in result['manifest']['weight_files']] == ['checkpoints/base.safetensors']
    assert len(snapshot['manifest']['weight_files']) == 2


def test_reverification_does_not_reidentify_mutated_file(repo,monkeypatch):
    lib=ModelLibrary(repo,"owner",drive=FakeDrive())
    a=model(lib,source="drive",drive_file_id="remote-file-1234567")
    lib.update(a['id'],state='ready',sha256=HASH,size=len(DATA))
    task=lib.enqueue(a['id'],'verify')
    def read_mutated(self,path):
        path.write_bytes(b'changed bytes')
        return {'id':'remote-file-1234567'}
    monkeypatch.setattr(ModelTransfer,'read_drive',read_mutated)
    ModelTransfer(lib,task).run()
    changed=lib.get(a['id'])
    assert changed['state'] == 'failed' and changed['sha256'] == HASH


def test_alternate_runtime_selection_requires_previously_approved_identity():
    old='a'*64
    dep={'id':'d','node_id':'1','field':'ckpt_name','reference':'base.safetensors','model_type':'checkpoints','model_id':'new','expected_sha256':old,'required':True}
    model={'id':'new','state':'ready','sha256':HASH,'size':len(DATA),'drive_file_id':'new-remote-file-1234567'}
    snapshot={'manifest':{'weight_files':[{'target':'checkpoints/base.safetensors','sha256':old}]},'bound_graph':{'1':{'inputs':{'ckpt_name':'base.safetensors'}}}}
    with pytest.raises(ModelError,match='审核'):
        deps.runtime_copy(snapshot,[dep],[model])
    mapped=deps.runtime_copy(snapshot,[dep],[model],approved={('1','ckpt_name'):{HASH}})
    assert mapped['manifest']['weight_files'][0]['sha256'] == HASH
    assert snapshot['manifest']['weight_files'][0]['sha256'] == old


def test_approved_alternates_require_matching_graph_pins_and_real_accepted_smoke(repo,monkeypatch):
    from app import review_routes
    m=repo.create('models','owner',{'model_id':'family'})
    w=workflow(repo,model_id=m['model_id'],manifest={'dependencies':{'comfyui_commit':'a'*40,'custom_nodes':[],'weight_files':[{'target':'checkpoints/base.safetensors','sha256':HASH}]}})
    job=repo.create('jobs','owner',{'state':'completed','provider':'colab','smoke_accepted':True,'acceptance_state':'accepted'})
    repo.create('workflows','owner',{**{k:v for k,v in w.items() if k not in {'id','owner_id','created_at','updated_at'}},
                         'workflow_version':2,'state':'published','review_approved':True,'smoke_test_kind':'real','smoke_job_id':job['id']})
    monkeypatch.setattr(review_routes,'_job_matches_any_smoke',lambda *args:True)
    assert deps.approved_alternates(repo,'owner',w) == {('1','ckpt_name'):{HASH}}
    incompatible=copy.deepcopy(w)
    incompatible['api_graph']['1']['class_type']='UnreviewedLoader'
    assert deps.approved_alternates(repo,'owner',incompatible) == {}


def test_drive_import_uses_native_sha_without_staging_large_file(repo,monkeypatch):
    drive=FakeDrive()
    drive.files['existing-file-1234567']={'id':'existing-file-1234567','mimeType':'application/octet-stream','trashed':False,
                                           'size':str(len(DATA)),'md5Checksum':MD5,'sha256Checksum':HASH}
    lib=ModelLibrary(repo,'owner',drive=drive)
    m=model(lib,source='drive',drive_file_id='existing-file-1234567',expected_sha256=HASH)
    monkeypatch.setattr(ModelTransfer,'read_drive',lambda *a:pytest.fail('native Drive SHA should avoid media transfer'))
    ModelTransfer(lib,lib.enqueue(m['id'])).run()
    assert lib.get(m['id'])['state'] == 'ready' and drive.uploads == 0


class WorkerKilled(BaseException):
    pass


class InterruptedDownload(httpx.SyncByteStream):
    def __init__(self, prefix):
        self.prefix = prefix

    def __iter__(self):
        yield self.prefix
        raise WorkerKilled


@pytest.mark.parametrize("range_behavior", ["resume", "ignored", "changed", "invalid", "corrupt"])
def test_interrupted_download_checks_prefix_and_range(repo, monkeypatch, range_behavior):
    from app.gpu import model_library

    monkeypatch.setattr(model_library, "CHECKPOINT_BYTES", model_library.BUFFER)
    prefix = b"p" * model_library.BUFFER
    payload = prefix + DATA
    requests = []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(200, stream=InterruptedDownload(prefix), headers={
                "ETag": '"fixed-v1"', "Content-Length": str(len(payload))
            })
        if "Range" not in request.headers or range_behavior == "ignored":
            return httpx.Response(200, content=payload, headers={"ETag": '"fixed-v1"'})
        assert request.headers["Range"] == f"bytes={len(prefix)}-"
        assert request.headers["If-Range"] == '"fixed-v1"'
        start = len(prefix) - int(range_behavior == "invalid")
        return httpx.Response(206, content=DATA, headers={
            "Content-Range": f"bytes {start}-{len(payload)-1}/{len(payload)}",
            "ETag": '"fixed-v2"' if range_behavior == "changed" else '"fixed-v1"',
        })

    drive = FakeDrive()
    lib = ModelLibrary(repo, "owner", drive=drive, http=source_http(handler))
    row = model(lib, expected_sha256=hashlib.sha256(payload).hexdigest(), expected_size=len(payload))
    task = lib.enqueue(row["id"])
    with pytest.raises(WorkerKilled):
        ModelTransfer(lib, task).run()
    if range_behavior == "corrupt":
        with (model_library.transfer_directory(task["id"]) / "weight.partial").open("r+b") as stream:
            stream.write(b"x")
    task = repo.get("model_downloads", task["id"], "owner")
    ModelTransfer(lib, task).run()
    state = repo.get("model_downloads", task["id"], "owner")
    assert state["state"] == "ready" and state["attempt"] == 2
    assert bytes(drive.bytes) == payload
    assert bool(state["restart_reason"]) == (range_behavior != "resume")
    assert state["resume_offset"] == (len(prefix) if range_behavior == "resume" else 0)
    if range_behavior == "corrupt":
        assert "Range" not in requests[1].headers


def test_complete_download_reused_after_worker_killed(repo, monkeypatch):
    drive = FakeDrive()
    calls = []
    lib = ModelLibrary(repo, "owner", drive=drive, http=source_http(
        lambda req: (calls.append(req), httpx.Response(200, content=DATA))[1]
    ))
    row = model(lib, expected_sha256=HASH, expected_size=len(DATA))
    task = lib.enqueue(row["id"])
    original = ModelTransfer.hash_file

    def killed(self, path, limit=None):
        raise WorkerKilled

    monkeypatch.setattr(ModelTransfer, "hash_file", killed)
    with pytest.raises(WorkerKilled):
        ModelTransfer(lib, task).run()
    monkeypatch.setattr(ModelTransfer, "hash_file", original)
    ModelTransfer(lib, repo.get("model_downloads", task["id"], "owner")).run()
    assert lib.get(row["id"])["state"] == "ready" and len(calls) == 1 and drive.uploads == 1


@pytest.mark.parametrize("mode", ["resume", "expired", "final-response-lost"])
def test_drive_resumes_confirmed_offset_and_dedupes_final_response(repo, monkeypatch, mode):
    from app.gpu import model_library
    from app.gpu.drive import DriveStorage, _DriveTransportError

    monkeypatch.setattr(model_library, "UPLOAD_CHUNK", 8)

    class ResumingDrive(FakeDrive):
        _file_resume_status = DriveStorage._file_resume_status
        _file_upload_offset = staticmethod(DriveStorage._file_upload_offset)
        _json = staticmethod(DriveStorage._json)

        def __init__(self):
            super().__init__()
            self.killed = False
            self.lost = False
            self.ranges = []

        def _request(self, method, url, **kwargs):
            if method != "PUT":
                return super()._request(method, url, **kwargs)
            content_range = kwargs["headers"]["Content-Range"]
            if content_range.startswith("bytes */"):
                if mode == "expired" or self.lost:
                    return httpx.Response(404)
                return httpx.Response(308, headers={"Range": f"bytes=0-{len(self.bytes)-1}"})
            self.ranges.append(content_range)
            begin = int(content_range.split()[1].split("-")[0])
            assert begin == len(self.bytes)
            for chunk in kwargs["content"]:
                assert len(chunk) <= model_library.BUFFER
                self.bytes.extend(chunk)
            assert int(kwargs["headers"]["Content-Length"]) == len(self.bytes) - begin
            if len(self.bytes) < len(DATA):
                if not self.killed and mode != "final-response-lost":
                    self.killed = True
                    raise WorkerKilled
                return httpx.Response(308, headers={"Range": f"bytes=0-{len(self.bytes)-1}"})
            self.files["remote"] = {
                "id": "remote", "size": str(len(self.bytes)), "md5Checksum": MD5, "sha256Checksum": HASH
            }
            if mode == "final-response-lost":
                self.lost = True
                raise _DriveTransportError("response disconnected")
            return httpx.Response(200, json=self.files["remote"])

    drive = ResumingDrive()
    source_calls = []
    lib = ModelLibrary(repo, "owner", drive=drive, http=source_http(
        lambda req: (source_calls.append(req), httpx.Response(200, content=DATA))[1]
    ))
    row = model(lib, expected_sha256=HASH, expected_size=len(DATA))
    task = lib.enqueue(row["id"])
    if mode == "final-response-lost":
        ModelTransfer(lib, task).run()
    else:
        with pytest.raises(WorkerKilled):
            ModelTransfer(lib, task).run()
        ModelTransfer(lib, repo.get("model_downloads", task["id"], "owner")).run()
    assert lib.get(row["id"])["state"] == "ready"
    assert len(repo.list("model_downloads", "owner")) == len(drive.files) == len(source_calls) == 1
    assert drive.uploads == (2 if mode == "expired" else 1)
    if mode == "resume":
        assert repo.get("model_downloads", task["id"], "owner")["resume_offset"] == 8
        assert drive.ranges[1] == f"bytes 8-15/{len(DATA)}"


def test_import_records_original_url_before_transfer_and_final_drive_id(repo):
    from app.gpu.model_library import external_source

    entered = "https://example.com/original"
    signed = "https://cdn.example.com/content?signature=redirect-secret"

    def handler(request):
        if request.headers["host"] == "example.com":
            return httpx.Response(302, headers={"Location": signed})
        return httpx.Response(200, content=DATA)

    lib = ModelLibrary(repo, "owner", drive=FakeDrive(), http=source_http(handler))
    [candidate] = lib.resolve({"url": entered, "filename": "base.safetensors", "model_type": "checkpoints"})
    model_row = lib.add(candidate)
    task = lib.enqueue(model_row["id"])
    assert task["original_source_url"] == entered
    ModelTransfer(lib, task).run()
    saved = lib.get(model_row["id"])
    assert external_source(saved) == {"url": entered, "download_url": entered, "status": "verified"}
    persisted_task = repo.get("model_downloads", task["id"], "owner")
    assert persisted_task["drive_file_id"] == saved["drive_file_id"] == "remote"
    assert persisted_task["original_source_url"] == entered
    assert "redirect-secret" not in repr((saved, persisted_task))
    assert not repo.list("model_library", "other")


def test_manual_binding_checks_metadata_and_removes_source_without_upload(repo):
    from app.gpu.model_library import external_source

    url = "https://huggingface.co/org/repo/blob/main/base.safetensors"
    drive = FakeDrive()
    meta_hash = HASH
    meta_size = len(DATA)

    def handler(request):
        return httpx.Response(200, json={"sha": "a" * 40, "siblings": [
            {"rfilename": "base.safetensors", "lfs": {"sha256": meta_hash, "size": meta_size}},
        ]})

    lib = ModelLibrary(repo, "owner", drive=drive, http=source_http(handler))
    record = lib.add({"source": "drive", "drive_file_id": "drive-copy", "filename": "base.safetensors",
                      "model_type": "checkpoints", "relative_path": "nested/base.safetensors"})
    record = lib.update(record["id"], state="ready", sha256=HASH, size=len(DATA))
    binding = lib.bind_external_source(record, url)
    assert binding["status"] == "verified" and binding["url"] == url
    assert "/resolve/" + "a" * 40 + "/" in binding["download_url"]
    after = lib.update(record["id"], external_source=binding)
    assert after["drive_file_id"] == "drive-copy" and after["relative_path"] == "nested/base.safetensors"
    generic = lib.bind_external_source(after, "https://example.com/file")
    assert generic["status"] == "unverified"
    meta_hash = "0" * 64
    with pytest.raises(ModelError, match="不匹配"):
        lib.bind_external_source(after, url)
    meta_hash = HASH
    meta_size += 1
    with pytest.raises(ModelError, match="不匹配"):
        lib.bind_external_source(after, url)
    assert external_source(lib.update(record["id"], external_source=lib.bind_external_source(after, None))) is None
    assert drive.uploads == 0


def test_hf_models_datasets_spaces_urls_metadata_and_download_links():
    from app.gpu.model_sources import parse_hf_url

    commit = "b" * 40
    requested_paths = []
    siblings = [
        {"rfilename": "upload/kazutake-hazano_v2_epoch28.safetensors", "lfs": {"sha256": HASH, "size": len(DATA)}},
        {"rfilename": "upload/nested/748cm_TA_EP4 (1).safetensors", "lfs": {"sha256": HASH, "size": len(DATA)}},
        {"rfilename": "upload/Anima-半写实动漫+style#1.safetensors", "lfs": {"sha256": HASH, "size": len(DATA)}},
        {"rfilename": "other/top.gguf", "size": 123},
        {"rfilename": "upload/readme.png", "size": 99},
    ]

    def handler(request):
        requested_paths.append(request.url.raw_path.decode())
        return httpx.Response(200, json={"sha": commit, "siblings": siblings})

    sources = Sources(source_http(handler))

    # 1. Models repo (resolve / blob / tree / fixed revision)
    m_res = sources.resolve({"url": "https://huggingface.co/org/model-repo/resolve/v1.0/upload/kazutake-hazano_v2_epoch28.safetensors?download=true"})
    assert requested_paths[-1] == "/api/models/org/model-repo/revision/v1.0?blobs=true"
    assert len(m_res) == 1
    assert m_res[0]["download_url"] == f"https://huggingface.co/org/model-repo/resolve/{commit}/upload/kazutake-hazano_v2_epoch28.safetensors"
    assert m_res[0]["source_identity"] == {
        "repo_id": "org/model-repo",
        "revision": commit,
        "filename": "upload/kazutake-hazano_v2_epoch28.safetensors",
    }

    m_tree = sources.resolve({"url": f"https://huggingface.co/org/model-repo/tree/{commit}/upload"})
    assert requested_paths[-1] == f"/api/models/org/model-repo/revision/{commit}?blobs=true"
    assert len(m_tree) == 3

    # 2. Datasets repo (resolve / blob / tree / repo_id with repo_type)
    ds_url = "https://huggingface.co/datasets/ACCC1380/Anima_Lora_Collection/resolve/main/upload/kazutake-hazano_v2_epoch28.safetensors"
    ds_res = sources.resolve({"url": ds_url})
    assert requested_paths[-1] == "/api/datasets/ACCC1380/Anima_Lora_Collection/revision/main?blobs=true"
    assert len(ds_res) == 1
    assert ds_res[0]["filename"] == "kazutake-hazano_v2_epoch28.safetensors"
    assert ds_res[0]["expected_sha256"] == HASH and ds_res[0]["expected_size"] == len(DATA)
    assert ds_res[0]["download_url"] == f"https://huggingface.co/datasets/ACCC1380/Anima_Lora_Collection/resolve/{commit}/upload/kazutake-hazano_v2_epoch28.safetensors"
    assert ds_res[0]["source_identity"] == {
        "repo_type": "datasets",
        "repo_id": "ACCC1380/Anima_Lora_Collection",
        "revision": commit,
        "filename": "upload/kazutake-hazano_v2_epoch28.safetensors",
    }

    ds_blob = sources.resolve({"url": "https://huggingface.co/datasets/ACCC1380/Anima_Lora_Collection/blob/main/upload/kazutake-hazano_v2_epoch28.safetensors"})
    assert len(ds_blob) == 1 and ds_blob[0]["download_url"] == ds_res[0]["download_url"]

    ds_tree_root = sources.resolve({"url": "https://huggingface.co/datasets/ACCC1380/Anima_Lora_Collection/tree/main"})
    assert len(ds_tree_root) == 4
    ds_tree_dir = sources.resolve({"url": "https://huggingface.co/datasets/ACCC1380/Anima_Lora_Collection/tree/main/upload/nested"})
    assert [f["filename"] for f in ds_tree_dir] == ["748cm_TA_EP4 (1).safetensors"]

    ds_by_repo = sources.resolve({
        "repo_type": "dataset",
        "repo_id": "ACCC1380/Anima_Lora_Collection",
        "revision": commit,
        "filename": "upload/kazutake-hazano_v2_epoch28.safetensors",
    })
    assert ds_by_repo[0]["download_url"] == ds_res[0]["download_url"]

    # 3. Spaces repo
    sp_res = sources.resolve({"url": "https://huggingface.co/spaces/org/demo-space/resolve/main/other/top.gguf"})
    assert requested_paths[-1] == "/api/spaces/org/demo-space/revision/main?blobs=true"
    assert len(sp_res) == 1
    assert sp_res[0]["download_url"] == f"https://huggingface.co/spaces/org/demo-space/resolve/{commit}/other/top.gguf"
    assert sp_res[0]["source_identity"]["repo_type"] == "spaces"

    # 4. Spaces, parentheses, UTF-8, '+', '#' and URL-encoded vs literal-space inputs
    encoded_space = sources.resolve({"url": "https://huggingface.co/datasets/ACCC1380/Anima_Lora_Collection/resolve/main/upload/nested/748cm_TA_EP4%20%281%29.safetensors"})
    literal_space = sources.resolve({"url": "https://huggingface.co/datasets/ACCC1380/Anima_Lora_Collection/resolve/main/upload/nested/748cm_TA_EP4 (1).safetensors"})
    assert len(encoded_space) == 1 and encoded_space == literal_space
    assert encoded_space[0]["download_url"] == f"https://huggingface.co/datasets/ACCC1380/Anima_Lora_Collection/resolve/{commit}/upload/nested/748cm_TA_EP4%20%281%29.safetensors"

    special = sources.resolve({"url": "https://huggingface.co/datasets/ACCC1380/Anima_Lora_Collection/blob/main/upload/Anima-%E5%8D%8A%E5%86%99%E5%AE%9E%E5%8A%A8%E6%BC%AB%2Bstyle%231.safetensors"})
    assert len(special) == 1
    assert special[0]["filename"] == "Anima-半写实动漫+style#1.safetensors"
    assert "%23" in special[0]["download_url"] and "#" not in special[0]["download_url"]
    assert parse_hf_url(special[0]["download_url"]) == (
        "datasets",
        "ACCC1380/Anima_Lora_Collection",
        commit,
        "upload/Anima-半写实动漫+style#1.safetensors",
        False,
    )


@pytest.mark.parametrize("bad_url", [
    "https://huggingface.co/datasets",
    "https://huggingface.co/datasets/ACCC1380",
    "https://huggingface.co/spaces/only-owner",
    "https://huggingface.co/only-owner",
    "https://huggingface.co/org/repo/resolve",
    "https://huggingface.co/org/repo/resolve/main",
    "https://huggingface.co/datasets/org/repo/blob/main",
    "https://huggingface.co/datasets/org/repo/tree",
    "https://huggingface.co/datasets/org/repo/unknown/main/a.safetensors",
    "https://huggingface.co/datasets/org/repo/resolve/main/upload/../secret.safetensors",
])
def test_hf_invalid_urls_rejected(bad_url):
    from app.gpu.model_sources import parse_hf_url

    with pytest.raises(ModelError):
        parse_hf_url(bad_url)


def test_hf_dataset_and_model_end_to_end_download_and_prepare_dependencies(repo):
    commit = "c" * 40
    calls = []

    def handler(request):
        host = request.headers["host"]
        path = request.url.raw_path.decode()
        calls.append((host, path, request.headers.get("authorization")))
        if path.startswith("/api/datasets/ACCC1380/Anima_Lora_Collection/revision/"):
            return httpx.Response(200, json={"sha": commit, "siblings": [
                {"rfilename": "upload/kazutake-hazano_v2_epoch28.safetensors", "lfs": {"sha256": HASH, "size": len(DATA)}},
            ]})
        if path.startswith("/api/models/org/model-repo/revision/"):
            return httpx.Response(200, json={"sha": commit, "siblings": [
                {"rfilename": "weights/base.safetensors", "lfs": {"sha256": HASH, "size": len(DATA)}},
            ]})
        if host == "huggingface.co" and path in {
            f"/datasets/ACCC1380/Anima_Lora_Collection/resolve/{commit}/upload/kazutake-hazano_v2_epoch28.safetensors",
            f"/org/model-repo/resolve/{commit}/weights/base.safetensors",
        }:
            return httpx.Response(302, headers={"location": "https://cdn.hf.example/blob?signature=signed"})
        if host == "cdn.hf.example":
            return httpx.Response(200, content=DATA)
        return httpx.Response(404)

    drive = FakeDrive()
    lib = ModelLibrary(repo, "owner", drive=drive, http=source_http(handler, {"huggingface": "hf-secret"}))

    # Datasets end-to-end: resolve -> add -> enqueue -> download -> verify -> Drive upload
    ds_url = "https://huggingface.co/datasets/ACCC1380/Anima_Lora_Collection/resolve/main/upload/kazutake-hazano_v2_epoch28.safetensors"
    [candidate] = lib.resolve({"url": ds_url})
    saved = lib.add({**candidate, "model_type": "loras"})
    assert saved["source_identity"]["repo_type"] == "datasets"
    assert saved["download_url"].startswith("https://huggingface.co/datasets/ACCC1380/Anima_Lora_Collection/resolve/")
    task = lib.enqueue(saved["id"])
    ModelTransfer(lib, task).run()
    ready = lib.get(saved["id"])
    assert ready["state"] == "ready" and ready["sha256"] == HASH and ready["size"] == len(DATA)
    assert ready["integrity"] == "expected_hash_verified" and drive.uploads == 1
    # Verify HF token was sent to huggingface.co and stripped on CDN redirect
    assert any(h == "huggingface.co" and p.startswith("/datasets/") and a == "Bearer hf-secret" for h, p, a in calls)
    assert all(a is None for h, _, a in calls if h == "cdn.hf.example")

    # prepare_dependencies preserves datasets prefix when weight specifies repo_type
    other_hash = "e" * 64
    wf = repo.create("workflows", "owner", {
        "workflow_key": "ds-dep",
        "workflow_version": 1,
        "manifest": {"weight_files": [{
            "repo_type": "datasets",
            "repo_id": "ACCC1380/Anima_Lora_Collection",
            "revision": commit,
            "filename": "upload/ds_alias.safetensors",
            "target": "loras/ds_alias.safetensors",
            "sha256": other_hash,
            "size": len(DATA),
        }]},
        "api_graph": {"1": {"class_type": "LoraLoader", "inputs": {"lora_name": "ds_alias.safetensors", "strength_model": 1, "strength_clip": 1}}},
        "state": "draft",
    })
    prep = lib.prepare_dependencies(wf["id"])
    assert len(prep["tasks"]) == 1
    dep_model = lib.get(prep["tasks"][0]["model_id"])
    assert dep_model["download_url"] == f"https://huggingface.co/datasets/ACCC1380/Anima_Lora_Collection/resolve/{commit}/upload/ds_alias.safetensors"
    assert dep_model["source_identity"]["repo_type"] == "datasets"

    # Existing models workflow remains unaffected (no /datasets/ prefix, no repo_type key)
    [m_candidate] = lib.resolve({"url": "https://huggingface.co/org/model-repo/blob/main/weights/base.safetensors"})
    m_saved = lib.add({**m_candidate, "model_type": "checkpoints"})
    assert "repo_type" not in m_saved["source_identity"]
    assert m_saved["download_url"] == f"https://huggingface.co/org/model-repo/resolve/{commit}/weights/base.safetensors"


def _make_safe_pytorch_zip() -> bytes:
    """Build a minimal valid PyTorch ZIP archive with safe pure-tensor pickle opcodes."""
    import io
    import zipfile

    # Protocol 2 pickle calling collections.OrderedDict and torch._utils._rebuild_tensor_v2 + torch.FloatStorage
    pkl = (
        b"\x80\x02"
        b"ccollections\nOrderedDict\n)\x81"
        b"ctorch._utils\n_rebuild_tensor_v2\n"
        b"ctorch\nFloatStorage\n"
        b"K\x00\x86R."
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("archive/data.pkl", pkl)
        zf.writestr("archive/data/0", b"\x00" * 16)
    return buf.getvalue()


def _make_safe_pytorch_legacy() -> bytes:
    """Build a minimal legacy PyTorch checkpoint (magic + protocol + sys_info + state_dict)."""
    import pickle
    from app.gpu.model_sources import PYTORCH_LEGACY_MAGIC

    s0 = pickle.dumps(PYTORCH_LEGACY_MAGIC, protocol=2)
    s1 = pickle.dumps(1001, protocol=2)
    s2 = pickle.dumps({"little_endian": True, "type_sizes": {"short": 2, "int": 4, "long": 4}}, protocol=2)
    s3 = (
        b"\x80\x02"
        b"ccollections\nOrderedDict\n)\x81"
        b"ctorch._utils\n_rebuild_tensor_v2\n"
        b"ctorch\nFloatStorage\n"
        b"K\x00\x86R."
    )
    return s0 + s1 + s2 + s3


def _make_malicious_pytorch_zip(use_stack_global: bool = False) -> bytes:
    import io
    import zipfile

    if use_stack_global:
        # Protocol 4 STACK_GLOBAL subprocess.Popen
        pkl = b"\x80\x04\x8c\nsubprocess\x8c\x05Popen\x93)\x81."
    else:
        # Protocol 2 GLOBAL os.system
        pkl = b"\x80\x02cos\nsystem\n(X\x02\x00\x00\x00idtR."
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("archive/data.pkl", pkl)
    return buf.getvalue()


def test_hf_ultrasharp_slash_revision_query_params_and_errors():
    commit = "d2db793b0909d31d651019f7aa17891616cc3327"
    ultra_sha = "a5812231fc936b42af08a5edba784195495d303d5b3248c24489ef0c4021fe01"
    requested = []

    def handler(request):
        path = request.url.raw_path.decode()
        requested.append(path)
        if path == "/api/models/MochaPixel/4XUltrasharpV10/revision/main?blobs=true":
            return httpx.Response(200, json={"sha": commit, "siblings": [
                {"rfilename": "4xUltrasharp_4xUltrasharpV10.pt", "lfs": {"sha256": ultra_sha, "size": 66961958}},
                {"rfilename": "README.md", "size": 42},
            ]})
        if path == "/api/models/org/slash-branch/refs":
            return httpx.Response(200, json={"branches": [{"name": "release/v1.0"}], "tags": [], "converts": []})
        if path == "/api/models/org/slash-branch/revision/release%2Fv1.0?blobs=true":
            return httpx.Response(200, json={"sha": commit, "siblings": [
                {"rfilename": "sub/model.safetensors", "lfs": {"sha256": HASH, "size": len(DATA)}},
            ]})
        if path == "/api/models/org/gated-repo/revision/main?blobs=true":
            return httpx.Response(403, text="gated")
        if path == "/api/models/org/empty-tree/revision/main?blobs=true":
            return httpx.Response(200, json={"sha": commit, "siblings": [{"rfilename": "docs/readme.md", "size": 10}]})
        return httpx.Response(404)

    sources = Sources(source_http(handler))

    # UltraSharp blob + resolve + tracking query params under upscale_models
    for url in (
        "https://huggingface.co/MochaPixel/4XUltrasharpV10/blob/main/4xUltrasharp_4xUltrasharpV10.pt",
        "https://huggingface.co/MochaPixel/4XUltrasharpV10/resolve/main/4xUltrasharp_4xUltrasharpV10.pt?download=true&utm_source=test",
    ):
        files = sources.resolve({"url": url, "model_type": "upscale_models"})
        assert len(files) == 1
        assert files[0]["filename"] == "4xUltrasharp_4xUltrasharpV10.pt"
        assert files[0]["model_type"] == "upscale_models"
        assert files[0]["expected_size"] == 66961958
        assert files[0]["expected_sha256"] == ultra_sha
        assert files[0]["download_url"] == f"https://huggingface.co/MochaPixel/4XUltrasharpV10/resolve/{commit}/4xUltrasharp_4xUltrasharpV10.pt"
        assert "repo_type" not in files[0]["source_identity"]

    # Reject .pt when non-upscale category is selected
    with pytest.raises(ModelError, match="upscale_models"):
        sources.resolve({
            "url": "https://huggingface.co/MochaPixel/4XUltrasharpV10/blob/main/4xUltrasharp_4xUltrasharpV10.pt",
            "model_type": "checkpoints",
        })

    # Branch containing '/' disambiguated via /refs
    slash_files = sources.resolve({"url": "https://huggingface.co/org/slash-branch/blob/release/v1.0/sub/model.safetensors"})
    assert len(slash_files) == 1 and slash_files[0]["filename"] == "model.safetensors"

    # Missing file inside existing repo -> clear 404
    with pytest.raises(ModelError, match="404"):
        sources.resolve({"url": "https://huggingface.co/MochaPixel/4XUltrasharpV10/blob/main/does_not_exist.safetensors"})

    # Missing repo -> 404
    with pytest.raises(ModelError, match="404"):
        sources.resolve({"url": "https://huggingface.co/org/nonexistent-repo"})

    # Gated repo -> 403 permission error
    with pytest.raises(ModelError, match="403"):
        sources.resolve({"url": "https://huggingface.co/org/gated-repo"})

    # Tree directory with no supported weights -> explicit directory error
    with pytest.raises(ModelError, match="目录"):
        sources.resolve({"url": "https://huggingface.co/org/empty-tree/tree/main/docs"})


def test_controlled_upscale_pt_pth_security_and_downloads(repo):
    commit = "d2db793b0909d31d651019f7aa17891616cc3327"
    legacy_pt = _make_safe_pytorch_legacy()
    legacy_sha = hashlib.sha256(legacy_pt).hexdigest()
    zip_pth = _make_safe_pytorch_zip()
    zip_sha = hashlib.sha256(zip_pth).hexdigest()
    gguf_bytes = b"GGUF\x03\x00\x00\x00" + b"\x00" * 32
    gguf_sha = hashlib.sha256(gguf_bytes).hexdigest()
    onnx_bytes = b"\x08\x07\x12\x04test" + b"\x00" * 32
    onnx_sha = hashlib.sha256(onnx_bytes).hexdigest()

    def handler(request):
        host = request.headers["host"]
        path = request.url.raw_path.decode()
        if path == "/api/models/MochaPixel/4XUltrasharpV10/revision/main?blobs=true":
            return httpx.Response(200, json={"sha": commit, "siblings": [
                {"rfilename": "4xUltrasharp_4xUltrasharpV10.pt", "lfs": {"sha256": legacy_sha, "size": len(legacy_pt)}},
            ]})
        if host == "huggingface.co" and path.endswith("/4xUltrasharp_4xUltrasharpV10.pt"):
            return httpx.Response(200, content=legacy_pt)
        if host == "api.github.com" and path == "/repos/xinntao/Real-ESRGAN/releases/tags/v0.2.2.4":
            return httpx.Response(200, json={"assets": [
                {"name": "RealESRGAN_x4plus_anime_6B.pth", "size": len(zip_pth), "digest": f"sha256:{zip_sha}"},
            ]})
        if host == "github.com" and path.endswith("/RealESRGAN_x4plus_anime_6B.pth"):
            return httpx.Response(200, content=zip_pth)
        if host == "example.com" and path == "/model.gguf":
            return httpx.Response(200, content=gguf_bytes)
        if host == "example.com" and path == "/model.onnx":
            return httpx.Response(200, content=onnx_bytes)
        return httpx.Response(404)

    drive = FakeDrive()
    lib = ModelLibrary(repo, "owner", drive=drive, http=source_http(handler, {"huggingface": "hf-tok"}))

    # 1. Valid legacy .pt upscale model from Hugging Face
    [pt_cand] = lib.resolve({
        "url": "https://huggingface.co/MochaPixel/4XUltrasharpV10/blob/main/4xUltrasharp_4xUltrasharpV10.pt",
        "model_type": "upscale_models",
    })
    pt_row = lib.add(pt_cand)
    ModelTransfer(lib, lib.enqueue(pt_row["id"])).run()
    pt_ready = lib.get(pt_row["id"])
    assert pt_ready["state"] == "ready" and pt_ready["sha256"] == legacy_sha and pt_ready["model_type"] == "upscale_models"

    # 2. Valid modern ZIP .pth upscale model from trusted GitHub Release (xinntao)
    [pth_cand] = lib.resolve({
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "model_type": "upscale_models",
    })
    assert pth_cand["expected_size"] == len(zip_pth) and pth_cand["expected_sha256"] == zip_sha
    pth_row = lib.add(pth_cand)
    ModelTransfer(lib, lib.enqueue(pth_row["id"])).run()
    pth_ready = lib.get(pth_row["id"])
    assert pth_ready["state"] == "ready" and pth_ready["sha256"] == zip_sha and pth_ready["integrity"] == "expected_hash_verified"

    # 3. Regression: GGUF & ONNX downloads still work
    gguf_row = lib.add({"filename": "m.gguf", "model_type": "diffusion_models", "source": "https",
                        "download_url": "https://example.com/model.gguf", "expected_sha256": gguf_sha})
    ModelTransfer(lib, lib.enqueue(gguf_row["id"])).run()
    assert lib.get(gguf_row["id"])["state"] == "ready"

    onnx_row = lib.add({"filename": "m.onnx", "model_type": "upscale_models", "source": "https",
                        "download_url": "https://example.com/model.onnx", "expected_sha256": onnx_sha})
    ModelTransfer(lib, lib.enqueue(onnx_row["id"])).run()
    assert lib.get(onnx_row["id"])["state"] == "ready"

    # 4. Non-upscale categories still block .pt / .pth
    for bad_cat in ("checkpoints", "loras", "vae"):
        with pytest.raises(ModelError):
            lib.add({**pth_cand, "model_type": bad_cat})

    # 5. Unauthorized .pt/.pth sources blocked (arbitrary host, untrusted GitHub owner, upload/drive)
    for bad_url in (
        "https://evil.example/upscaler.pt",
        "https://github.com/random-attacker/fake/releases/download/v1/upscaler.pth",
        "https://github.com/xinntao/Real-ESRGAN/raw/master/RealESRGAN_x4plus_anime_6B.pth",
    ):
        with pytest.raises(ModelError, match="未授权"):
            lib.resolve({"url": bad_url, "model_type": "upscale_models"})
        with pytest.raises(ModelError, match="未授权"):
            lib.add({"filename": "bad.pth", "model_type": "upscale_models", "source": "https", "download_url": bad_url})

    with pytest.raises(ModelError, match="受控"):
        lib.add({"filename": "local.pth", "model_type": "upscale_models", "source": "upload", "drive_file_id": "file-123"})


@pytest.mark.parametrize("filename_val,model_type,payload", [
    ("spoofed.safetensors", "checkpoints", _make_safe_pytorch_zip()),
    ("spoofed_legacy.safetensors", "checkpoints", _make_safe_pytorch_legacy()),
    ("spoofed.gguf", "diffusion_models", b"not-a-gguf-payload-at-all"),
    ("spoofed.onnx", "upscale_models", _make_safe_pytorch_zip()),
    ("html_page.safetensors", "checkpoints", b"<!DOCTYPE html><html>404</html>"),
    ("malicious_os.pth", "upscale_models", _make_malicious_pytorch_zip(use_stack_global=False)),
    ("malicious_stack.pth", "upscale_models", _make_malicious_pytorch_zip(use_stack_global=True)),
    ("corrupt.pt", "upscale_models", b"\x80\x02\x99\xfftruncated-corrupt-bytes"),
    ("fake_pt.pt", "upscale_models", b"<html>not a torch model</html>"),
])
def test_weight_file_validation_blocks_spoofing_malicious_pickle_and_corruption(repo, filename_val, model_type, payload):
    drive = FakeDrive()
    url = (
        f"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/{filename_val}"
        if filename_val.endswith((".pt", ".pth"))
        else f"https://example.com/{filename_val}"
    )
    lib = ModelLibrary(repo, "owner", drive=drive, http=source_http(lambda r: httpx.Response(200, content=payload)))
    row = lib.add({"filename": filename_val, "model_type": model_type, "source": "https", "download_url": url})
    ModelTransfer(lib, lib.enqueue(row["id"])).run()
    failed = lib.get(row["id"])
    assert failed["state"] == "failed" and drive.uploads == 0
    assert failed["error"]
