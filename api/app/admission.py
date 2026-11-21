"""Job admission domain service: validate, freeze inputs, estimate and persist."""
import copy

from .common import APIError, own
from .gpu import loras as lora_registry
from .gpu.compute import GpuConfig
from .gpu.inputs import resolve_input_images
from .gpu.manifest import bind_parameters, digest, resolve_parameters, validate_manifest
from .gpu.rates import estimate as estimate_units
from .gpu.rates import length_factor, mode_param
from .gpu.registry import Registry
from .gpu.repository import Repository
from .gpu.specs import compatibility_signature, resolve_compute
from .job_schemas import normalize_policy


def repo_registry(st, user):
    repo = Repository(st)
    registry = Registry(repo)
    registry.bootstrap(user["id"])
    return repo, registry


def _job_estimate(repo, owner, workflow, policy, cfg, params) -> dict:
    """Estimate the cost of the submitted workflow and inputs."""
    if policy["provider"] == "mock":
        return {"estimated_units": 0.0, "unit_rate": cfg.unit_rate}
    cap = policy["max_runtime_minutes"]
    manifest = workflow["manifest"]
    picked = mode_param(manifest)
    mode = params.get(picked[0]) if picked else None
    live_rate = None
    if policy.get("instance_id") and policy.get("account_id"):
        from .gpu.ccu import instance_live_rate

        with repo.store.db_lock(), repo.store.Session() as session:
            live_rate = instance_live_rate(session, owner, policy["account_id"], policy["instance_id"])
    spec = policy.get("resolved_compute_spec") or manifest.get("compute_spec")
    units, rate = estimate_units(repo, owner, workflow["workflow_key"], manifest, spec, cap,
                                 mode=mode, fallback_rate=cfg.unit_rate, live_rate_hourly=live_rate)
    if units is not None:
        units = round(min(units * length_factor(manifest, params), cap * rate), 3)
    return {"estimated_units": units, "unit_rate": rate}


def _resource(repo, kind, resource_id, user, code, message):
    """Read a v0.2 resource through the owner-scoped Repository boundary."""
    value = repo.get(kind, resource_id, user["id"])
    if not value:
        raise APIError(code, message, 404)
    return value


def _account_is_usable(account: dict) -> bool:
    status = str(account.get("status", account.get("state", "connected"))).lower()
    return (
        account.get("enabled", True) is not False
        and account.get("disabled", False) is not True
        and account.get("allow_allocation", True) is not False
        and status not in {"disabled", "needs_reauth", "error", "revoked"}
        and account.get("authorized", account.get("connected", True)) is not False
    )


def _validate_allocation(repo, user, policy: dict) -> dict:
    """Validate explicit account/instance requests without cross-identity fallback."""
    checked = dict(policy)
    account = None
    if checked.get("account_id"):
        account = _resource(repo, "accounts", checked["account_id"], user,
                            "ACCOUNT_NOT_FOUND", "指定的 Google 账号不存在或不属于当前用户")
        if not _account_is_usable(account):
            raise APIError("ACCOUNT_UNAVAILABLE", "指定账号已停用、未授权或不允许自动分配", 409)

    if checked.get("instance_id"):
        instance = _resource(repo, "instances", checked["instance_id"], user,
                             "INSTANCE_NOT_FOUND", "指定 GPU 实例不存在或不属于当前用户")
        state = str(instance.get("state", "")).lower()
        if state in {"released", "deleted", "releasing", "unknown", "cleanup_pending", "quarantined"}:
            raise APIError("INSTANCE_UNAVAILABLE", "指定 GPU 实例已释放或当前不可复用", 409)
        instance_account = instance.get("account_id")
        if instance_account and checked.get("account_id") and instance_account != checked["account_id"]:
            raise APIError("ACCOUNT_INSTANCE_CONFLICT", "指定账号与实例不属于同一身份", 422)
        if instance_account and not checked.get("account_id"):
            checked["account_id"] = instance_account
            account = _resource(repo, "accounts", instance_account, user,
                                "ACCOUNT_NOT_FOUND", "实例所属账号不存在或不属于当前用户")
            if not _account_is_usable(account):
                raise APIError("ACCOUNT_UNAVAILABLE", "实例所属账号已停用或未授权", 409)
        if instance.get("provider") and instance["provider"] != checked["provider"]:
            raise APIError("PROVIDER_INVALID", "指定实例与请求 Provider 不匹配", 422)
        if instance.get("current_job_id") or state in {"busy", "running"}:
            raise APIError("INSTANCE_BUSY", "指定 GPU 实例当前正在执行其他任务", 409)

    # Automatic account selection is only allowed to inspect eligible accounts;
    # it must never silently fail over from one identity to another.
    if checked["provider"] == "colab" and not checked.get("account_id"):
        accounts = repo.list("accounts", user["id"])
        usable = [item.get("account_id") or item["id"] for item in accounts if _account_is_usable(item)]
        if accounts and not usable:
            raise APIError("ACCOUNT_UNAVAILABLE", "没有允许分配且已授权的 Google 账号", 409)
        # #24: bind the only Colab-connected account before its OAuth status is checked;
        # with several connected accounts the instance service keeps choosing.
        connected = [aid for aid in usable if _google_status(repo, user["id"], aid)["colab"]["connected"]]
        if len(connected) == 1:
            checked["account_id"] = connected[0]
    return checked


def _google_status(repo, owner_id, account_id):
    from .gpu.google import GoogleService

    return GoogleService(repo, owner_id, account_id=account_id).status()


def _resolve_drive_account_id(repo, owner_id, account_id):
    from .gpu.accounts import AccountService

    return AccountService(repo, owner_id).resolve_drive_account_id(account_id)


def admit_job(st, user, body, key, *, validate_only=False, creation=None, before_create=None):
    repo, registry = repo_registry(st, user)
    if not key or len(key) > 200:
        raise APIError("IDEMPOTENCY_REQUIRED", "必须提供 Idempotency-Key，重试必须沿用原值")
    existing = next((j for j in repo.list("jobs", user["id"]) if j.get("idempotency_key") == key), None)
    if existing:
        same = (existing.get("creation_body_hash") == creation["creation_body_hash"]
                if creation and creation.get("creation_key_id") else existing.get("body_hash") == digest(body))
        if not same:
            raise APIError("IDEMPOTENCY_CONFLICT", "相同幂等键不能用于不同参数", 409)
        return existing
    project = own(st, "projects", body["project_id"], user)
    cfg = GpuConfig.from_env()
    policy = _validate_allocation(repo, user, normalize_policy(body["execution_policy"]))
    if policy["provider"] not in {"mock", "colab"}:
        raise APIError("PROVIDER_INVALID", "仅允许已配置的 mock 或 colab Provider")
    if policy["provider"] == "colab" and not cfg.real_enabled:
        raise APIError("REAL_GPU_DISABLED", "真实 GPU 默认关闭：请先配置 OAuth 并完成模型审查")
    try:
        workflow = registry.get_workflow(user["id"], body["workflow_key"], body["workflow_version"])
    except KeyError as exc:
        raise APIError("WORKFLOW_UNPUBLISHED", "工作流未发布或与模型不匹配", 404) from exc
    if workflow.get("state") != "published" or workflow["model_id"] != body["model_id"]:
        raise APIError("WORKFLOW_UNPUBLISHED", "工作流未发布或与模型不匹配")
    model = next((m for m in registry.models(user["id"]) if m["model_id"] == body["model_id"]), None)
    if not model or not model.get("enabled"):
        raise APIError("MODEL_UNAVAILABLE", "模型尚未通过许可、权重与硬件验证")
    if bool(model.get("is_mock")) != (policy["provider"] == "mock"):
        raise APIError("EXECUTOR_MISMATCH", "模拟模板只能使用 Mock，真实模型不能伪装模拟成功")
    validate_manifest(workflow["manifest"], workflow["api_graph"], workflow["object_info"])
    requested_spec, compute_warnings = resolve_compute(workflow["manifest"], policy.get("compute_spec"))
    policy["resolved_compute_spec"] = requested_spec.get("key") if requested_spec else None
    compute_fields = {
        "requested_spec": requested_spec,
        "compute_warnings": compute_warnings,
        "workflow_version_id": workflow["id"],
        "generation_timeouts": True,
    }
    preset = own(repo, "presets", body["preset_id"], user) if body.get("preset_id") else None
    if preset and preset["workflow_version_id"] != workflow["id"]:
        raise APIError("PRESET_INCOMPATIBLE", "预设属于不同工作流版本，请先迁移预设")
    params = resolve_parameters(
        workflow["manifest"], body["params"], preset.get("overrides") if preset else None
    )
    bind_params, input_files = resolve_input_images(
        lambda aid: own(st, "assets", aid, user), project["id"], workflow["manifest"], params
    )
    bound = bind_parameters(workflow["manifest"], workflow["api_graph"], bind_params)
    for asset_id in body["asset_ids"]:
        asset = own(st, "assets", asset_id, user)
        if asset["project_id"] != project["id"]:
            raise APIError("ASSET_PROJECT_MISMATCH", "参考素材不属于当前项目")
    snapshot = {
        "manifest": workflow["manifest"],
        "api_graph": workflow["api_graph"],
        "object_info": workflow["object_info"],
        "resolved_params": params,
        "bound_graph": bound,
        "input_files": input_files,
        "workflow_sha256": workflow["sha256"],
        "model_revision": model.get("revision"),
        "dependency_lock": workflow["manifest"].get("dependencies", {}),
    }
    snapshot["runtime_compute_spec"] = policy["resolved_compute_spec"]
    if creation:
        snapshot["creation_api"] = True
        if creation.get("warmup"):
            snapshot["creation_warmup"] = True
    if policy["provider"] == "colab":
        if not workflow.get("review_approved") or workflow.get("smoke_test_kind") != "real":
            raise APIError("REVIEW_REQUIRED", "真实工作流尚未通过后台审核及真实冒烟")
        snapshot["manifest"] = {**snapshot["manifest"], "state": "published", "reviewed": True}
        account = policy.get("account_id")
        if account:
            connected = _google_status(repo, user["id"], account)["colab"]["connected"]
        else:
            candidates = [row.get("account_id") or row["id"] for row in repo.list("accounts", user["id"])
                          if _account_is_usable(row) and row.get("provider", "colab") in {"colab", "google"}]
            connected = any(_google_status(repo, user["id"], key)["colab"]["connected"] for key in candidates or [None])
        if not connected:
            raise APIError("GOOGLE_NOT_CONNECTED", "请先授权对应 Google Colab 账户")
    # After the published stamp, so the curated LoRA weight files stay in the manifest the runtime restores.
    if body.get("loras"):
        _apply_loras(repo, user, workflow, body["loras"], snapshot, policy["provider"])
    from .gpu.model_dependencies import approved_alternates, bind, runtime_copy
    from .gpu.model_library import ModelLibrary
    from .gpu.model_sources import ModelError
    library = ModelLibrary(repo, user["id"])
    model_drive_account = None
    binding = None
    try:
        binding = next((r for r in repo.list("model_dependencies", user["id"])
                        if r.get("workflow_id") == workflow["id"]), None)
        if body.get("model_overrides"):
            binding = library.dependency_record(workflow["id"])
            dependencies = copy.deepcopy(binding["dependencies"])
            for override in body["model_overrides"]:
                dep = next((d for d in dependencies if d["id"] == override.get("dependency_id")), None)
                if dep is None:
                    raise ModelError("任务模型覆盖引用了未知依赖")
                replacement = library.get(override.get("model_id"))
                dep.update(bind(dep, replacement, confirm_unknown=override.get("confirm_unknown") is True,
                                strengths=override.get("strengths"), disabled=override.get("disabled") is True))
            binding = {**binding, "dependencies": dependencies}
        if binding:
            bound_models = []
            for ident in {d.get("model_id") for d in binding["dependencies"] if d.get("model_id")}:
                if policy["provider"] == "colab" and not creation:
                    bound_models.append(library.check_remote(ident))
                else:
                    bound_models.append(library.get(ident))
            drive_accounts = {m.get("drive_account_id") for m in bound_models if m.get("drive_account_id")}
            if len(drive_accounts) > 1:
                raise ModelError("本次模型来自不同 Drive 账号，请先统一存储账号")
            model_drive_account = next(iter(drive_accounts), None)
            snapshot = runtime_copy(snapshot, binding["dependencies"], bound_models,
                                    approved=approved_alternates(repo, user["id"], workflow))
    except ModelError as exc:
        raise APIError("MODEL_DEPENDENCIES_NOT_READY", str(exc), 409) from exc
    finally:
        library.close()
    environment = {k: v for k, v in snapshot.items() if k not in {"creation_api", "creation_warmup"}}
    compute_fields["compatibility_signature"] = compatibility_signature(
        policy["provider"], environment, requested_spec, workflow["workflow_key"], workflow["workflow_version"]
    )
    if creation:
        compute_fields.update(creation, creation_binding_digest=digest(
            {"dependencies": binding["dependencies"] if binding else None}))
        compute_fields["creation_fingerprint"] = compatibility_signature(
            policy["provider"], {k: v for k, v in environment.items() if k != "runtime_compute_spec"},
            None, workflow["workflow_key"], workflow["workflow_version"],
        )
    if validate_only:
        return {**body, **compute_fields, "execution_policy": policy,
                "requested_account_id": policy.get("account_id"),
                "requested_instance_id": policy.get("instance_id"),
                **({"drive_account_id": model_drive_account or _resolve_drive_account_id(repo, user["id"], policy.get("account_id"))} if model_drive_account or policy.get("account_id") else {}),
                "snapshot": snapshot, "params": params,
                "resolved_seed": params.get("seed"),
                "max_runtime_minutes": policy["max_runtime_minutes"],
                **_job_estimate(repo, user["id"], workflow, policy, cfg, params)}
    # A worker cannot observe a queued record without its immutable snapshot.
    with st.admission_lock() as acquired:
        if not acquired:
            raise APIError("SCHEDULER_BUSY", "队列正在同步，请用相同幂等键稍后重试", 503, True)
        from .gpu.native import guard_generation

        if repo.get("workflows", workflow["id"], user["id"]) is None:
            raise APIError("WORKFLOW_UNPUBLISHED", "工作流版本已删除", 409)
        if before_create:
            reused = before_create(repo, {**compute_fields, "snapshot": snapshot, "execution_policy": policy})
            if reused:
                return reused
            from .gpu.cu_guard import CuGuardService
            CuGuardService(repo).check(user["id"], policy["provider"], policy.get("account_id"), policy.get("instance_id"))
        else:
            guard_generation(repo, user["id"], policy)
        job, created = repo.create_job(user["id"], body, key, admitted={
            **compute_fields,
            "execution_policy": policy,
            **({"account_id": policy.get("account_id")} if policy.get("account_id") else {}),
            "requested_account_id": policy.get("account_id"),
            "requested_instance_id": policy.get("instance_id"),
            **({"drive_account_id": model_drive_account or _resolve_drive_account_id(repo, user["id"], policy.get("account_id"))} if model_drive_account or policy.get("account_id") else {}),
            "snapshot": snapshot,
            "params": params,
            "resolved_seed": params.get("seed"),
            "atelier_project_id": project.get("atelier_project_id"),
            "max_runtime_minutes": policy["max_runtime_minutes"],
            **_job_estimate(repo, user["id"], workflow, policy, cfg, params),
        })
        if created:
            repo.event(job["id"], user["id"], "queued", "任务已持久化；请求规格：" + str(requested_spec) + "；" + "；".join(compute_warnings))
    return job


def _apply_loras(repo, user, workflow, picks, snapshot, provider):
    """Curated picks become extra weight files; own picks are Drive files the runtime fetches."""
    if workflow["workflow_key"] not in lora_registry.H3_WORKFLOWS:
        raise APIError("LORA_UNSUPPORTED", "这个工作流不支持自定义 LoRA")
    curated = {c["id"]: c for c in lora_registry.CURATED if workflow["workflow_key"] in c["workflows"]}
    own_files = None
    chain, weights, user_files = [], [], []
    for pick in picks:
        ref, strength = str(pick.get("id") or ""), pick.get("strength", 0.8)
        if not isinstance(strength, int | float) or not -3.0 <= strength <= 3.0:
            raise APIError("VALIDATION_ERROR", "LoRA 强度需在 -3.0 到 3.0 之间")
        if ref in curated:
            weight = lora_registry.curated_weight(curated[ref])
            weights.append(weight)
            chain.append((weight["target"].removeprefix("loras/"), float(strength)))
            continue
        if own_files is None:
            own_files = {} if provider == "mock" else {f["id"]: f for f in _drive(repo, user).list_loras()}
        found = own_files.get(ref)
        if not found or not lora_registry.is_lora_file(found["name"], found["size"]):
            raise APIError("LORA_NOT_FOUND", "LoRA 文件不在 Drive 的 loras/h3 文件夹里，或不是 1.5GB 以内的 .safetensors")
        name = f"user/{found['id']}_{found['name']}"
        user_files.append({"id": found["id"], "name": found["name"], "size": found["size"], "md5Checksum": found["md5"], "target": name})
        chain.append((name, float(strength)))
    snapshot["bound_graph"] = lora_registry.insert_loras(snapshot["bound_graph"], chain)
    manifest = copy.deepcopy(snapshot["manifest"])
    manifest["weight_files"] = [*(manifest.get("weight_files") or []), *weights]
    allow = manifest.get("node_allowlist")
    if isinstance(allow, list) and "LoraLoaderModelOnly" not in allow:
        allow.append("LoraLoaderModelOnly")
    snapshot["manifest"] = manifest
    snapshot["lora_files"] = user_files
    snapshot["loras"] = [{"id": p.get("id"), "strength": p.get("strength", 0.8)} for p in picks]


def _drive(repo, user):
    from .gpu.drive import DriveStorage

    return DriveStorage({"owner_id": user["id"]}, repo)
