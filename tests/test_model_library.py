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
