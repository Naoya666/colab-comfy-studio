"""Authenticated model catalog API, separate from the existing reviewed workflow registry."""
from contextlib import contextmanager

from fastapi import APIRouter, Depends

from .common import APIError, admin, current_user, get_store, ok
from .config import settings
from .gpu.model_dependencies import directories
from .gpu.model_library import ModelLibrary, external_source
from .gpu.model_sources import ModelError, Sources
from .gpu.repository import Repository

router = APIRouter(prefix="/api/gpu")


@contextmanager
def library(st, user):
    lib = ModelLibrary(Repository(st), user["id"])
    try:
        yield lib
    except ModelError as exc:
        raise APIError("MODEL_LIBRARY_ERROR", str(exc), 422) from exc
    finally:
        lib.close()


def public_task(task):
    return {k: v for k, v in task.items() if not k.startswith("encrypted_")}


@router.get("/model-library")
def list_models(q: str = "", model_type: str = "", architecture: str = "", state: str = "", user=Depends(current_user), st=Depends(get_store)):
    with library(st, user) as lib:
        models = lib.models()
        visible = [m for m in models if (not q or q.lower() in f"{m['display_name']} {m['filename']}".lower())
                   and (not model_type or m["model_type"] == model_type)
                   and (not architecture or m["architecture"] == architecture) and (not state or m["state"] == state)]
        physical = {m["drive_file_id"]: m.get("size") or 0 for m in models if m.get("drive_file_id")}
        return ok({"items": [{**m, "external_source": external_source(m)} for m in visible], "types": directories(), "statistics": {"count": len(models), "bytes": sum(physical.values()),
                   "by_type": {t: {"count": sum(m["model_type"] == t for m in models),
                                  "bytes": sum({m["drive_file_id"]: m.get("size") or 0 for m in models if m["model_type"] == t and m.get("drive_file_id")}.values())}
                               for t in directories()}}})


@router.post("/model-library/resolve")
def resolve(body: dict, user=Depends(current_user), st=Depends(get_store)):
    with library(st, user) as lib:
        return ok({"files": lib.resolve(body), "requires_selection": True})


@router.get("/model-library/search")
def search(q: str, user=Depends(current_user), st=Depends(get_store)):
    with library(st, user) as lib:
        candidates, warnings = Sources(lib.http).search_with_warnings(q)
        return ok({"candidates": candidates, "warnings": warnings, "requires_confirmation": True})


@router.post("/model-library/inspect-workflow")
def inspect_workflow(body: dict, user=Depends(current_user), st=Depends(get_store)):
    with library(st, user) as lib:
        return ok(lib.inspect_workflow(body))


@router.get("/model-library/drive-usage")
def drive_usage(refresh: bool = False, user=Depends(current_user), st=Depends(get_store)):
    from .gpu.drive_usage import collect_drive_usage

    return ok(collect_drive_usage(Repository(st), user["id"], force_refresh=refresh))


@router.get("/model-library/storage-settings")
def storage_settings(user=Depends(current_user), st=Depends(get_store)):
    with library(st, user) as lib:
        return ok(lib.storage_settings())


@router.put("/model-library/storage-settings")
def save_storage(body: dict, user=Depends(current_user), st=Depends(get_store)):
    with library(st, user) as lib:
        return ok(lib.save_storage(body.get("account_id")))


@router.get("/model-library/source-settings")
def source_settings(user=Depends(current_user), st=Depends(get_store)):
    with library(st, user) as lib:
        return ok(lib.token_status())


@router.put("/model-library/source-settings/{source}")
def source_save(source: str, body: dict, user=Depends(current_user), st=Depends(get_store)):
    with library(st, user) as lib:
        return ok(lib.save_token(source, body.get("token")))


@router.post("/model-library/upload-session")
def upload_session(body: dict, user=Depends(current_user), st=Depends(get_store)):
    with library(st, user) as lib:
        return ok(lib.upload_session(body.get("filename"), body.get("size"), settings.site_url))


@router.get("/model-library/tasks")
def tasks(user=Depends(current_user), st=Depends(get_store)):
    with library(st, user) as lib:
        return ok([public_task(t) for t in lib.repo.list("model_downloads", user["id"])])


@router.post("/model-library/tasks/{task_id}/cancel")
def cancel(task_id: str, user=Depends(current_user), st=Depends(get_store)):
    with library(st, user) as lib:
        return ok(public_task(lib.cancel(task_id)))


@router.post("/model-library")
def add(body: dict, user=Depends(current_user), st=Depends(get_store)):
    with library(st, user) as lib:
        # Source adapters are re-run server-side to avoid trusting user-supplied platform identities.
        if body.get("source") in {"huggingface", "civitai"}:
            options = lib.resolve({
                "url": body.get("original_source_url") or body.get("source_url"),
                "model_type": body.get("model_type"),
            })
            selected = [f for f in options if f.get("source_identity") == body.get("source_identity")]
            if len(selected) != 1:
                raise ModelError("所选来源文件身份已变化，请重新解析")
            body = {**selected[0], **{k: body[k] for k in ("model_type", "architecture", "display_name", "relative_path") if k in body}}
        return ok(lib.add(body), status=201)


@router.post("/model-library/{model_id}/workflow-model")
def register_workflow_model(model_id: str, body: dict, user=Depends(current_user), st=Depends(get_store)):
    admin(user)
    with library(st, user) as lib:
        return ok(lib.register_workflow_model(model_id, body), status=201)


@router.get("/model-library/{model_id}")
def detail(model_id: str, user=Depends(current_user), st=Depends(get_store)):
    with library(st, user) as lib:
        row = lib.get(model_id)
        return ok({**row, "external_source": external_source(row), "references": lib.references(model_id)})


@router.patch("/model-library/{model_id}")
def update(model_id: str, body: dict, user=Depends(current_user), st=Depends(get_store)):
    with library(st, user) as lib:
        model = lib.get(model_id)
        patch = {k: str(body[k])[:300] for k in ("display_name", "architecture") if k in body}
        if "tags" in body:
            if not isinstance(body["tags"], list):
                raise ModelError("标签需为数组")
            patch["tags"] = [str(t)[:100] for t in body["tags"][:100]]
        if "external_source_url" in body:
            patch["external_source"] = lib.bind_external_source(model, body["external_source_url"])
        return ok(lib.update(model_id, **patch))


@router.post("/model-library/{model_id}/prepare")
def prepare(model_id: str, body: dict | None = None, user=Depends(current_user), st=Depends(get_store)):
    with library(st, user) as lib:
        return ok(public_task(lib.enqueue(model_id, (body or {}).get("action", "prepare"))), status=202)


@router.post("/model-library/{model_id}/check")
def check(model_id: str, user=Depends(current_user), st=Depends(get_store)):
    with library(st, user) as lib:
        return ok(lib.check_remote(model_id))


@router.delete("/model-library/{model_id}")
def delete(model_id: str, confirm_references: bool = False, delete_file: bool = False, user=Depends(current_user), st=Depends(get_store)):
    with library(st, user) as lib:
        return ok(lib.delete(model_id, confirm_references=confirm_references, delete_file=delete_file))


@router.get("/workflows/{workflow_id}/model-dependencies")
def dependencies(workflow_id: str, user=Depends(current_user), st=Depends(get_store)):
    with library(st, user) as lib:
        return ok(lib.dependency_record(workflow_id))


@router.post("/workflows/{workflow_id}/model-dependencies/{action}")
def dependency_action(workflow_id: str, action: str, body: dict | None = None, user=Depends(current_user), st=Depends(get_store)):
    body = body or {}
    with library(st, user) as lib:
        if action in {"scan", "match"}:
            return ok(lib.dependency_record(workflow_id, rescan=action == "scan"))
        if action == "prepare":
            return ok(lib.prepare_dependencies(workflow_id), status=202)
        if action == "adopt-cache":
            return ok(lib.adopt_cache(workflow_id), status=202)
        if action == "draft":
            admin(user)
            return ok(lib.export_bound_draft(workflow_id), status=201)
        if action == "bind":
            return ok(lib.bind(workflow_id, body.get("dependency_id"), body.get("model_id"),
                               confirm_unknown=body.get("confirm_unknown") is True,
                               strengths=body.get("strengths"), disabled=body.get("disabled") is True))
        raise ModelError("未知依赖操作")
