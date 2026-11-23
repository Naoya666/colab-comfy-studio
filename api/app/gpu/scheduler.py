"""Persistent short-tick scheduler for independent Colab GPU Studio API."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from app.common import APIError

from . import ccu
from .colab import RUNTIME_MISSING, ColabAllocationError, GoogleColabProvider, runtime_status
from .compute import (
    ComputeError,
    DisabledProvider,
    GpuConfig,
    ProviderUnavailable,
    default_executor_factory,
    default_provider_factory,
)
from .cu_guard import CuGuardService
from .google import GoogleError
from .manifest import digest
from .mcp import MCPError
from .newgpu.executor_plugins import TrustedExecutorAdapter
from .prep import PREP_SPEC_KEY, cache_key, create_prep_job, prep_idempotency_key, prep_jobs_for
from .scheduler_instances import SchedulerError, SchedulerInstancesMixin
from .scheduler_watchdog import ACTIVE_STATES, TERMINAL_STATES, SchedulerWatchdogMixin

MAX_OUTPUT_BYTES = 100 * 1024 * 1024


# Stage markers written by runtime_worker.py into status.json while a job is running.
_STAGE_TEXT = {
    "native_environment": "初始化原生 ComfyUI 和节点依赖",
    "native_system": "安装系统依赖与传输组件",
    "native_drive": "挂载 Google Drive 持久化目录",
    "native_models": "准备 ComfyUI 源码及模型目录",
    "native_plugins_restore": "并发恢复自定义节点归档",
    "native_env_cache": "恢复 Drive Python 环境缓存",
    "native_python": "安装 ComfyUI 与 Manager 基础依赖",
    "native_plugins_deps": "安装与校验自定义节点依赖",
    "native_tunnel": "建立原生 ComfyUI 代理连接",
    "restore_env": "正在恢复运行环境",
    "creation_python": "安装基础 Python 依赖",
    "restore_weights": "正在恢复模型文件",
    "restore_loras": "正在恢复用户 LoRA",
    "creation_warming": "加载模型并运行内部预热",
    "warm_reuse": "同一台机器已有模型，直接复用",
    "comfy_start": "正在启动 ComfyUI",
    "sampling": "采样中",
    "prep_env": "正在构建环境缓存",
    "prep_weights": "正在补齐模型缓存",
}


_PREP_TEXT = {
    "GPU job validating": "正在校验缓存准备任务",
    "GPU job validated": "缓存准备任务已校验",
    "GPU capacity available": "CPU 运行时可用",
    "GPU runtime allocated": "CPU 运行时已分配",
    "GPU runtime is ready": "CPU 运行时已就绪",
    "GPU execution started": "开始构建缓存",
    "GPU execution completed": "缓存构建完成",
    "GPU outputs persisted": "缓存已保存到 Drive",
    "GPU cleanup completed": "CPU 运行时已关闭（缓存保留在 Drive）",
    "GPU cleanup is pending": "正在关闭 CPU 运行时",
}


class Scheduler(SchedulerInstancesMixin, SchedulerWatchdogMixin):
    """Advance one persisted phase per tick; never hold the lock over adapter I/O."""

    def __init__(
        self,
        store,
        provider_factory: Callable | None = None,
        executor_factory: Callable | None = None,
        storage_factory: Callable | None = None,
        config: GpuConfig | None = None,
        clock: Callable[[], float] | None = None,
        instance_service: Any | None = None,
        instance_service_factory: Callable | None = None,
    ):
        # Accept either the base Store or the GPU Repository wrapped around it.
        from .instances import InstanceService
        from .repository import Repository

        self.store = store if isinstance(store, Repository) else Repository(store)
        self.db_store = self.store.store
        self.config = config or GpuConfig.from_env()
        self.clock = clock or time.time
        if instance_service is not None:
            self.instance_service = instance_service
        elif instance_service_factory is not None:
            self.instance_service = self._factory(instance_service_factory, {})
        else:
            self.instance_service = InstanceService(self.store, self.clock, self.config)
        self.instances = self.instance_service
        self.provider_factory = provider_factory or (
            lambda job: default_provider_factory(
                job, self.config, store=self.store, owner_id=job.get("owner_id")
            )
        )
        self.executor_factory = executor_factory or (
            lambda job: default_executor_factory(
                job, self.config, store=self.store, owner_id=job.get("owner_id")
            )
        )
        self.storage_factory = storage_factory

    def _factory(self, factory: Callable, job: dict):
        """Injected adapters take ``job`` and optionally the base Store."""
        import inspect

        parameters = inspect.signature(factory).parameters.values()
        positional = [p for p in parameters if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        if any(p.kind == p.VAR_POSITIONAL for p in parameters) or len(positional) >= 2:
            return factory(job, self.db_store)
        return factory(job)

    def _provider(self, job: dict):
        return self._factory(self.provider_factory, job)

    def _executor(self, job: dict):
        return self._factory(self.executor_factory, job)

    def _trusted_plugin_cleanup(self, job: dict, lease: dict | None) -> dict:
        """Run trusted-plugin lifecycle hooks without taking ownership from the provider.

        The provider remains authoritative for runtime deletion.  Plugin hook
        failures are therefore recorded as bounded internal status and never
        change the cleanup path or expose plugin exception text to the caller.
        """
        if not lease or not lease.get("handle") or not job.get("execution_id"):
            return job
        try:
            executor = self._executor(job)
        except Exception as exc:  # noqa: BLE001 - plugin discovery must not block provider cleanup
            logging.getLogger(__name__).warning(
                "trusted plugin executor unavailable during cleanup: %s", type(exc).__name__
            )
            return job
        if not isinstance(executor, TrustedExecutorAdapter):
            return job

        handle = self._lease_handle(lease)
        cancel_requested = bool(job.get("cancel_requested")) or job.get("requested_outcome") == "cancelled"
        if cancel_requested and job.get("trusted_plugin_cancel_succeeded") is not True:
            try:
                executor.cancel(handle, job)
                patch = {"trusted_plugin_cancel_succeeded": True, "trusted_plugin_cancel_error": None}
            except Exception as exc:  # noqa: BLE001 - provider cleanup remains authoritative
                logging.getLogger(__name__).warning("trusted plugin cancel failed: %s", type(exc).__name__)
                patch = {"trusted_plugin_cancel_error": "trusted_plugin_cancel_failed"}
            job = self._update("jobs", job["id"], patch, job.get("owner_id"))

        if job.get("trusted_plugin_cleanup_succeeded") is True:
            return job
        try:
            executor.cleanup(handle, job)
            patch = {"trusted_plugin_cleanup_succeeded": True, "trusted_plugin_cleanup_error": None}
        except Exception as exc:  # noqa: BLE001 - provider cleanup remains authoritative
            logging.getLogger(__name__).warning("trusted plugin cleanup failed: %s", type(exc).__name__)
            patch = {"trusted_plugin_cleanup_error": "trusted_plugin_cleanup_failed"}
        return self._update("jobs", job["id"], patch, job.get("owner_id"))

    def _list(self, kind: str, owner_id: str | None = None) -> list[dict]:
        return list(self.store.list(kind, owner_id=owner_id))

    def _get(self, kind: str, record_id: str, owner_id: str | None = None) -> dict | None:
        return self.store.get(kind, record_id, owner_id=owner_id)

    def _create(self, kind: str, owner_id: str, data: dict, record_id: str | None = None) -> dict:
        return self.store.create(kind, owner_id, data, record_id=record_id)

    def _update(self, kind: str, record_id: str, patch: dict, owner_id: str | None = None) -> dict:
        return self.store.update(kind, record_id, patch, owner_id=owner_id)

    def _event(self, job: dict, state: str, message: str, progress: float | None = None) -> None:
        self.store.event(job["id"], job.get("owner_id"), state, message, progress)

    def _set_state(self, job: dict, state: str, message: str, **patch) -> dict:
        if state in TERMINAL_STATES:
            # Worker and watchdog can both finish the same cleanup; only the first one records it.
            current = self.store.get("jobs", job["id"], job.get("owner_id")) or job
            if current.get("state") == state:
                return current
        if job.get("prep"):
            message = _PREP_TEXT.get(message, message.replace("GPU", "CPU"))
        return self.store.transition_job(
            job["id"], job.get("owner_id"), state, message, patch, self.clock(),
            expected_state=job.get("state"), expected_execution_id=job.get("execution_id"),
        )

    def _provider_name(self, job: dict) -> str:
        return job.get("provider") or (job.get("execution_policy") or {}).get("provider", "mock")

    def _effective_runtime(self, job: dict) -> int:
        requested = job.get("max_runtime_minutes")
        if requested is None:
            requested = (job.get("execution_policy") or {}).get("max_runtime_minutes")
        requested = self.config.max_runtime_minutes if requested is None else requested
        if type(requested) is not int or requested <= 0:
            raise SchedulerError("max runtime must be positive")
        return requested

    def _validate(self, job: dict) -> dict:
        if not self.config.enabled:
            return self._set_state(
                job,
                "failed",
                "GPU scheduler is disabled",
                error_code="gpu_disabled",
                error_message="GPU is disabled",
            )
        try:
            runtime = self._effective_runtime(job)
            self._check_cu(job)
            provider = self._provider(job)
            if self._provider_name(job) != "mock" and job.get("requested_spec") and not job.get("creation_fingerprint"):
                try:
                    self._select_spec(job, provider)
                except ProviderUnavailable as exc:
                    return self._begin_failure(job, "gpu_provider_unavailable", self._safe_error(exc))
                except Exception as exc:
                    return self._begin_failure(job, "gpu_spec_invalid", self._safe_error(exc))
        except APIError as exc:
            return self._begin_failure(job, exc.code, exc.message)
        except Exception as exc:
            return self._set_state(
                job,
                "failed",
                "GPU provider unavailable",
                error_code="gpu_provider_unavailable",
                error_message=self._safe_error(exc),
            )
        return self._set_state(job, "waiting_for_capacity", "GPU job validated",
                               max_runtime_minutes=runtime, validated_epoch=self.clock())

    def _check_cu(self, job):
        if self._provider_name(job) != "colab":
            return
        policy = job.get("execution_policy") or {}
        CuGuardService(self.store, self.clock).check(
            job["owner_id"], self._provider_name(job),
            job.get("account_id") or policy.get("account_id"), job.get("instance_id") or policy.get("instance_id"),
        )

    def _leases_for(self, owner_id: str | None = None) -> list[dict]:
        return self._list("leases", owner_id)

    @staticmethod
    def _lease_blocks(lease: dict) -> bool:
        return lease.get("state") in {"creating", "active", "busy", "idle", "cleanup_pending", "quarantined", "releasing", "deleting"}

    def _has_capacity_blocker(self, job: dict) -> bool:
        return any(self._lease_blocks(x) and x.get("owner_job_id") != job["id"] for x in self._leases_for())

    def _await_prep(self, job: dict) -> bool:
        """A GPU job waits for its workflow's CPU cache prep; the prep job is created on the first miss.

        One failed or cancelled prep stops the waiting (the job then runs uncached); it is never retried here.
        """
        from .instances import POOL_CREATION, instance_pool_of
        from .node_persistence import env_flag

        snapshot = job.get("snapshot") or {}
        sha = snapshot.get("workflow_sha256")
        weights = (snapshot.get("manifest") or {}).get("weight_files") or []
        if (
            self._provider_name(job) != "colab"
            or snapshot.get("prep")
            or snapshot.get("creation_api")
            or snapshot.get("system_smoke")
            or not sha
            or not weights
            or os.getenv("GPU_PREP", "on").strip().lower() == "off"
        ):
            return False
        # When every reviewed weight is already bound to Google Drive (Model Library) and
        # base Python dependencies are installed via fast parallel uv, CPU Prep has nothing
        # to cache for the GPU runtime.
        if (
            env_flag("CREATION_SKIP_DRIVE_PREP", True)
            and all(isinstance(w, dict) and w.get("source") == "drive" for w in weights)
        ):
            return False
        # If a live creation-pool instance already has this exact workflow warm in VRAM/disk
        # (e.g. warmed via Creation API), skip CPU prep so frontend requests can hot-reuse immediately.
        fp = job.get("creation_fingerprint")
        reuse_action = str((job.get("execution_policy") or {}).get("reuse_action") or job.get("reuse_action") or "auto")
        if fp and reuse_action not in {"new", "replace"}:
            for inst in self._list("instances", job.get("owner_id")):
                if (
                    instance_pool_of(inst) == POOL_CREATION
                    and inst.get("state") in {"ready", "retained_idle", "busy"}
                    and not inst.get("manual_stop_requested")
                    and not inst.get("runtime_deleted")
                    and (inst.get("creation_runtime") or {}).get("fingerprint") == fp
                    and (inst.get("creation_runtime") or {}).get("process")
                ):
                    return False
        drive_account_id = job.get("drive_account_id")
        if not drive_account_id and job.get("account_id"):
            from .accounts import AccountService

            drive_account_id = AccountService(self.store, job["owner_id"]).resolve_drive_account_id(job["account_id"])
        preps = prep_jobs_for(self.store, job["owner_id"], snapshot, drive_account_id=drive_account_id)
        done = next((p for p in preps if p["state"] == "completed"), None)
        if done:
            if done["id"] == job.get("prep_job_id") and job.get("prep_seconds") is None:
                # The prep ran for this job: show it as this job's first stage.
                spent = round(sum((done.get("worker_timings") or {}).values()), 1)
                self._update("jobs", job["id"], {"prep_seconds": spent}, job.get("owner_id"))
            return False
        active = next((p for p in preps if p["state"] in ACTIVE_STATES), None)
        if active:
            if job.get("prep_job_id") != active["id"]:
                self._update("jobs", job["id"], {"prep_job_id": active["id"]}, job.get("owner_id"))
            return True
        if preps:
            return False
        with self.db_store.dispatch_lock() as acquired:
            if not acquired:
                return True
            meta = {
                "workflow_sha256": sha,
                "model_id": job.get("model_id"),
                "workflow_key": job.get("workflow_key"),
                "workflow_version": job.get("workflow_version"),
                "account_id": job.get("account_id") or job.get("requested_account_id"),
                "drive_account_id": drive_account_id,
            }
            prep = create_prep_job(
                self.store, job["owner_id"], job.get("project_id"), snapshot, meta,
                prep_idempotency_key(cache_key(snapshot), drive_account_id or ""), for_job_id=job["id"],
            )
        self._update("jobs", job["id"], {"prep_job_id": prep["id"]}, job.get("owner_id"))
        self._event(job, "waiting_for_capacity", "准备缓存（CPU）：为这个任务自动准备环境和模型缓存，完成后自动开始", None)
        return True

    def _lease(self, job: dict) -> dict | None:
        if job.get("lease_id"):
            lease=self._get("leases", job["lease_id"], job.get("owner_id"))
            return lease if lease and lease.get('owner_job_id')==job['id'] else None
        return next(
            (x for x in self._leases_for(job.get("owner_id")) if x.get("owner_job_id") == job["id"]
             and not (job.get('attempt_retry_keys') and x.get('state')=='deleted')), None
        )

    def _ensure_lease(self, job: dict) -> dict:
        current = self._lease(job)
        if current:
            return current
        request_id, runtime_id = str(uuid.uuid4()), f"studio-{uuid.uuid4().hex}"
        now = self.clock()
        deadline = job.get("deadline_epoch")
        if deadline is None:
            deadline = now + self._effective_runtime(job) * 60
            if job.get("generation_timeouts"):
                deadline = job.get("task_deadline_epoch") or deadline
            elif job.get("creation_fingerprint"):
                deadline = now + self.config.max_runtime_minutes * 60
        from .instances import job_pool

        payload = {
            "owner_job_id": job["id"],
            "provider": self._provider_name(job),
            "instance_pool": job_pool(job),
            "state": "creating",
            "request_id": request_id,
            "runtime_id": runtime_id,
            "created_epoch": now,
            "last_seen": now,
            "delete_attempts": 0,
            "create_attempts": 0,
            "deadline_epoch": deadline,
        }
        if job.get("generation_timeouts"):
            payload["generation_timeouts"] = True
        for key in ("instance_id", "account_id"):
            if job.get(key) is not None:
                payload[key] = job[key]
        return self._create("leases", job["owner_id"], payload)

    @staticmethod
    def _safe_error(exc: Exception) -> str:
        """Expose only messages from deliberately safe adapter exception classes."""
        if not isinstance(exc, (ComputeError, GoogleError, MCPError)):
            logging.getLogger(__name__).warning("unexpected GPU error: %s: %s", type(exc).__name__, str(exc)[:300])
            return f"GPU operation failed ({type(exc).__name__})"
        message = str(exc).replace("token", "[redacted]").replace("secret", "[redacted]")
        return message[:1400] or "GPU operation failed"

    @staticmethod
    def _runtime_id(provider: str, runtime_id: str) -> str:
        if provider == "colab":
            return f"runtimes/{runtime_id}"
        return f"mock-runtime-{runtime_id}"

    def _lease_handle(self, lease: dict) -> dict | None:
        """The lease's runtime handle, always owned by the lease's current owner job.

        Lease reuse (handoff / idle attach) moves owner_job_id; providers verify the handle against it,
        so a handle that still names an earlier job is re-owned (and persisted) before any provider call.
        """
        handle = lease.get("handle")
        owner = lease.get("owner_job_id")
        if not isinstance(handle, dict) or not isinstance(owner, str):
            return handle
        if handle.get("job_id") == owner and handle.get("owner_job_id") == owner:
            return handle
        fixed = {**handle, "job_id": owner, "owner_job_id": owner}
        self._update("leases", lease["id"], {"handle": fixed}, lease.get("owner_id"))
        return fixed

    def _intent_handle(self, job: dict, lease: dict) -> dict | None:
        """Build only a lease-owned handle; never infer a platform runtime name."""
        runtime_id = lease.get("runtime_id")
        if not (
            isinstance(runtime_id, str)
            and lease.get("owner_job_id") == job.get("id")
            and isinstance(lease.get("request_id"), str)
            and lease.get("request_id")
        ):
            return None
        handle = {
            "runtime_name": self._runtime_id(self._provider_name(job), runtime_id),
            "runtime_id": runtime_id,
            "request_id": lease["request_id"],
            "job_id": job["id"],
            "owner_job_id": job["id"],
            "owner_id": job.get("owner_id"),
            "intent": "gpu-studio",
        }
        if lease.get("operation_name"):
            handle["operation_name"] = lease["operation_name"]
        if self._provider_name(job) == "mock":
            handle["mock"] = True
        return handle

    @staticmethod
    def _spec_key(spec: dict) -> str | None:
        """Colab runtimespecs are identified by their ``key`` (variant/accelerator/shape)."""
        value = spec.get("key")
        if isinstance(value, dict):
            from .specs import spec_key
            value = spec_key({"key": value})
            return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return value or None

    @staticmethod
    def _has_gpu(spec: dict) -> bool:
        key = spec.get("key")
        return isinstance(key, dict) and key.get("variant") in {"GPU", "VARIANT_GPU"}

    def _select_spec(self, job: dict, provider: Any) -> dict:
        try:
            specs = provider.list_specs()
        except Exception as exc:
            raise ComputeError(self._safe_error(exc)) from exc
        requested = job.get("requested_spec") or (job.get("execution_policy") or {}).get("requested_spec")
        requested_key = requested if isinstance(requested, str) else self._spec_key(requested or {})
        snapshot = job.get("snapshot") or {}
        smoke = snapshot.get("system_smoke") is True
        prep = snapshot.get("prep") is True
        # Workflow defaults and runtime overrides remain separate; never fall back to another GPU.
        reviewed_key = self._spec_key({"key": (snapshot.get("manifest") or {}).get("compute_spec")})
        if prep:
            requested_key = reviewed_key = self._spec_key({"key": PREP_SPEC_KEY})
        if requested_key is None:
            requested_key = reviewed_key
        if not (smoke or snapshot.get("workflow_smoke") is True or reviewed_key is not None):
            raise ComputeError("没有符合工作流runtime的GPU规格")
        if smoke and requested_key is None:
            raise ComputeError("system smoke必须绑定已选择的GPU规格")
        for spec in specs:
            if spec.get("eligible") is not True or self._has_gpu(spec) == prep:
                continue
            if requested_key is None or self._spec_key(spec) == requested_key:
                from .specs import check_requirements
                if not prep and not smoke:
                    check_requirements(snapshot.get("manifest") or {}, spec)
                return spec
        raise ComputeError("请求的 GPU 规格无申请资格或当前不可用；未切换其他 GPU：" + str(requested_key))

    def _recover_unknown_create(self, job: dict, lease: dict, provider: Any) -> str | bool:
        """Reconcile a create that may have succeeded before its response was lost."""
        handle = self._intent_handle(job, lease)
        if handle is None:
            return False
        try:
            if hasattr(provider, "recover_handle"):
                handle = provider.recover_handle(handle)
            status = provider.inspect(handle)
        except Exception as exc:
            self._update(
                "leases",
                lease["id"],
                {"last_error": self._safe_error(exc), "last_seen": self.clock()},
                job.get("owner_id"),
            )
            return False
        if status.get("deleted"):
            self._update("leases", lease["id"], {"last_seen": self.clock()}, job.get("owner_id"))
            return "absent"
        self._update(
            "leases",
            lease["id"],
            {"handle": handle, "state": "cleanup_pending", "last_seen": self.clock()},
            job.get("owner_id"),
        )
        recovered_lease = self._get("leases", lease["id"], job["owner_id"])
        if recovered_lease:
            ccu.open_instance_session(self.store, job, recovered_lease, lease.get("create_started") or self.clock())
        try:
            confirmed = bool(provider.delete(handle))
        except Exception as exc:
            confirmed = False
            error = self._safe_error(exc)
        else:
            error = None
        if confirmed:
            self._update(
                "leases", lease["id"], {"state": "deleted", "last_seen": self.clock()}, job.get("owner_id")
            )
            return "cleaned"
        self._update(
            "leases",
            lease["id"],
            {"last_error": error or "delete not confirmed", "last_seen": self.clock()},
            job.get("owner_id"),
        )
        return False

    def _sample_allocation_baseline(self, job: dict, lease: dict) -> dict:
        # Take the baseline before the first billable provider call.
        if (self._provider_name(job) != "colab" or job.get("ccu_before") is not None
                or int(lease.get("create_attempts") or 0) > 0):
            return job
        baseline = ccu.sample_and_record(self.store, job["owner_id"], job.get("account_id"), "alloc_before")
        if not baseline:
            return job
        return self._update("jobs", job["id"], {
            "ccu_before": baseline["balance"], "ccu_before_at": baseline["sampled_at"],
            "ccu_before_source": "google_balance",
        }, job["owner_id"])

    def _allocation(self, job: dict) -> dict:
        if job.get("selection_reason") in {"reuse_idle", "requested_instance"}:
            existing_lease = self._lease(job)
            if not existing_lease or not existing_lease.get("handle"):
                return self._begin_failure(job, "creation_lease_unavailable", "复用实例租约已不可用，已中止执行（未新开机器）")
        # Persist intent under the short lock, then release it before provider I/O.
        with self.db_store.dispatch_lock() as acquired:
            if not acquired:
                return job
            lease = self._ensure_lease(job)
        if job.get("lease_id") != lease["id"]:
            job = self._update("jobs", job["id"], {"lease_id": lease["id"]}, job.get("owner_id"))
        if lease.get("state") in {"cleanup_pending", "releasing", "deleting", "quarantined", "deleted"}:
            return self._begin_failure(job, "gpu_allocation_aborted", "GPU allocation entered cleanup before execution; retry after release")
        try:
            job, lease = self._bind_instance_lease(job, lease)
        except SchedulerError as exc:
            return self._begin_failure(job, "gpu_instance_conflict", str(exc))
        try:
            self._check_cu(job)
        except APIError as exc:
            return self._begin_failure(job, exc.code, exc.message)
        try:
            provider = self._provider(job)
        except Exception as exc:
            return self._begin_failure(job, "gpu_provider_unavailable", self._safe_error(exc))
        if lease.get("handle"):
            ccu.open_instance_session(self.store, job, lease, job.get("started_epoch") or lease.get("create_started") or self.clock())
            return self._set_state(job, "preparing", "GPU runtime allocation recovered")
        try:
            if isinstance(provider, DisabledProvider):
                raise ProviderUnavailable(provider.reason)
            spec = (
                {"provider": self._provider_name(job)}
                if self._provider_name(job) == "mock"
                else self._select_spec(job, provider)
            )
        except ProviderUnavailable as exc:
            return self._begin_failure(job, "gpu_provider_unavailable", self._safe_error(exc))
        except Exception as exc:
            return self._begin_failure(job, "gpu_spec_invalid", self._safe_error(exc))
        attempts = int(lease.get("create_attempts") or 0)
        if attempts > self.config.create_retries:
            return self._begin_failure(job, "gpu_create_failed", "GPU runtime allocation failed")
        job = self._sample_allocation_baseline(job, lease)
        try:
            self._check_cu(job)
        except APIError as exc:
            return self._begin_failure(job, exc.code, exc.message)
        attempts += 1
        lease = self._update(
            "leases",
            lease["id"],
            {
                "create_attempts": attempts,
                "create_started": self.clock(),
                "selected_spec": spec,
                "last_seen": self.clock(),
            },
            job.get("owner_id"),
        )
        try:
            handle = provider.create(spec, lease["request_id"], job["id"], lease["runtime_id"])
            if not isinstance(handle, dict) or not handle.get("runtime_name"):
                raise ComputeError("provider returned an unknown runtime handle")
            if hasattr(provider, "bind_owner"):
                handle = provider.bind_owner(handle, job.get("owner_id"))
            handle = {
                **handle,
                "runtime_id": lease["runtime_id"],
                "request_id": lease["request_id"],
                "job_id": job["id"],
                "owner_job_id": job["id"],
                "owner_id": job.get("owner_id"),
            }
        except ProviderUnavailable as exc:
            return self._begin_failure(job, "gpu_provider_unavailable", self._safe_error(exc))
        except ColabAllocationError as exc:
            return self._allocation_failed(job, exc)
        except Exception as exc:
            self._update("leases", lease["id"], {"last_create_error": self._safe_error(exc)}, job.get("owner_id"))
            recovered = self._recover_unknown_create(job, lease, provider)
            if recovered == "cleaned":
                return self._begin_failure(job, "gpu_create_failed", "GPU runtime allocation failed")
            if recovered is False and attempts > self.config.create_retries:
                return self._begin_failure(job, "gpu_create_failed", "GPU runtime allocation failed")
            return job
        lease = self._update(
            "leases",
            lease["id"],
            {"state": "active", "handle": handle, "last_seen": self.clock()},
            job.get("owner_id"),
        )
        ccu.open_instance_session(self.store, job, lease, self.clock())
        return self._set_state(
            job,
            "preparing",
            "GPU runtime allocated",
            request_id=lease["request_id"],
            runtime_id=lease["runtime_id"],
            deadline_epoch=lease["deadline_epoch"],
            started_epoch=self.clock(),
            started_at=self.clock(),
        )

    def _begin_failure(self, job: dict, code: str, message: str) -> dict:
        current = self.store.get("jobs", job["id"], job.get("owner_id"))
        if current and current.get("state") in TERMINAL_STATES | {"cleaning_up", "cleanup_pending"}:
            return current
        lease = self._lease(job)
        # No provider create was attempted, so a lease without a handle cannot
        # refer to an external runtime. Mark it reconciled without an API call.
        if lease and not lease.get("handle") and not int(lease.get("create_attempts") or 0):
            self._update(
                "leases", lease["id"], {"state": "deleted", "last_seen": self.clock()}, job.get("owner_id")
            )
        if lease or job.get("instance_id"):
            return self._set_state(
                job,
                "cleaning_up",
                message,
                requested_outcome="failed",
                error_code=code,
                error_message=message,
                cleanup_final_state="failed",
            )
        return self._set_state(
            job, "failed", message, requested_outcome="failed", error_code=code, error_message=message
        )

    def _allocation_failed(self, job: dict, exc: ColabAllocationError) -> dict:
        code = "gpu_capacity_unavailable" if exc.reason == "NO_RUNTIMES" else "gpu_allocation_failed"
        return self._begin_failure(job, code, str(exc))

    def _expired(self, job: dict) -> bool:
        lease = self._lease(job)
        deadline = job.get("deadline_epoch") or (lease or {}).get("deadline_epoch")
        return any(value is not None and self.clock() > float(value)
                   for value in (deadline, job.get("task_deadline_epoch")))

    def _prepare(self, job: dict) -> dict:
        lease = self._lease(job)
        if not lease or not lease.get("handle"):
            return self._begin_failure(job, "gpu_lease_missing", "GPU lease is missing")
        try:
            status = self._provider(job).inspect(self._lease_handle(lease))
        except Exception as exc:
            return self._begin_failure(job, "gpu_runtime_inspect_failed", self._safe_error(exc))
        if status.get("allocation_failed"):
            return self._allocation_failed(job, ColabAllocationError(
                status.get("allocation_error_code"), status.get("allocation_failure_reason")))
        if status.get("deleted"):
            return self._begin_failure(job, "gpu_runtime_missing", "GPU runtime disappeared")
        if status.get("ready"):
            from .instances import POOL_CREATION, job_pool

            reported = status.get("runtime_spec")
            if reported and job.get("requested_spec"):
                reused = (
                    job_pool(job) == POOL_CREATION
                    and job.get("selection_reason") in {"reuse_idle", "requested_instance"}
                )
                if reused:
                    from .specs import check_requirements, gpu_spec_order_settings, is_below_workflow_default

                    manifest = (job.get("snapshot") or {}).get("manifest") or {}
                    try:
                        check_requirements(manifest, reported)
                        order = gpu_spec_order_settings(self.db_store, job["owner_id"])["ordered_specs"]
                        if is_below_workflow_default(manifest, reported, order):
                            raise ValueError("复用实例实际 GPU 规格低于工作流默认最低规格")
                    except ValueError as exc:
                        return self._begin_failure(job, "gpu_spec_mismatch", str(exc))
                elif self._spec_key({"key": reported.get("key", reported)}) != self._spec_key(job["requested_spec"]):
                    return self._begin_failure(job, "gpu_spec_mismatch", "实际分配规格与请求不匹配，已停止执行并释放实例")
                job = self._update("jobs", job["id"], {"runtime_spec_reported": reported}, job.get("owner_id"))
                if job.get("instance_id"):
                    self._update("instances", job["instance_id"], {"runtime_spec_reported": reported}, job["owner_id"])
                self._event(job, "preparing", "实际分配规格：" + str(reported), None)
            job = self._mark_instance("mark_ready", job)
            return self._set_state(job, "loading", "GPU runtime is ready")
        return job

    def _load(self, job: dict) -> dict:
        lease = self._lease(job)
        if not lease:
            return self._begin_failure(job, "gpu_lease_missing", "GPU lease is missing")
        execution_id = job.get("execution_id")
        if execution_id:
            execution=self._get('executions',execution_id,job.get('owner_id'))
            if not execution or execution.get('state')!='starting':
                return self._set_state(job, "running", "GPU execution recovered")
        try:
            job = self.store.prepare_execution(job["id"], job["owner_id"], lease["id"], self.clock())
        except (KeyError, ValueError) as exc:
            # prepare_execution raises these only with its own fixed identity-check messages.
            # The lease is owned by this job (see _lease), so cleaning it up here is safe.
            return self._begin_failure(job, "gpu_execution_prepare_failed", f"Execution registration refused: {exc}")
        execution_id = job["execution_id"]
        if self._provider_name(job) == "colab" and job.get("creation_fingerprint") and job.get("instance_id"):
            self._update("instances", job["instance_id"], {"creation_runtime": None}, job["owner_id"])
        if (job.get("snapshot") or {}).get("native_comfy"):
            lease = self._update("leases", lease["id"], {"native_comfy": True}, job["owner_id"])
        try:
            # The Drive token is attached to an in-memory copy only; it is never persisted.
            cache = self._model_cache(job)
            executor_job = {
                **job,
                "drive_cache": cache,
                "use_drive_cache": os.getenv("GPU_MODEL_CACHE", "drive").strip().lower() == "drive",
                "deadline_epoch": job.get("deadline_epoch")
                if job.get("deadline_epoch") is not None
                else lease.get("deadline_epoch"),
            }
            if self._provider_name(job) == "colab" and (job.get("snapshot") or {}).get("lora_files"):
                executor_job["lora_drive_cache"] = self._lora_cache(job, cache)
            result = self._executor(job).start(
                self._lease_handle(lease), executor_job, execution_id
            )
        except Exception as exc:
            return self._begin_failure(job, "gpu_start_failed", self._safe_error(exc))
        current = self._get("jobs", job["id"], job["owner_id"])
        if current and current.get("cancel_requested") and current.get("error_code") == "CU_BALANCE_FLOOR_REACHED":
            return current
        if isinstance(result, dict) and result.get("execution_id") not in {None, execution_id}:
            return self._begin_failure(job, "gpu_execution_id_mismatch", "executor changed execution ID")
        job = self._mark_instance("job_busy", job)
        return self._set_state(job, "running", "GPU execution started")

    def _note_restored(self, job: dict, lease: dict, timings: dict) -> dict:
        """Remember on the lease which environment the runtime now has, and whether this job reused it."""
        signature = job.get("compatibility_signature")
        skipped = "warm_reuse" in timings
        if signature and lease.get("restored_signature") != signature:
            self._update("leases", lease["id"], {"restored_signature": signature}, job.get("owner_id"))
        return self._update("jobs", job["id"], {"restore_skipped": skipped}, job.get("owner_id"))

    def _model_cache(self, job: dict) -> dict | None:
        """Drive cache for reviewed model weights; None means download from Hugging Face."""
        if self._provider_name(job) != "colab":
            return None
        snapshot = job.get("snapshot") or {}
        manifest = snapshot.get("manifest") or {}
        weights = manifest.get("weight_files", manifest.get("weights")) or []
        if not weights:
            return None
        if (os.getenv("GPU_MODEL_CACHE", "drive").strip().lower() != "drive"
                and (snapshot.get("prep") or not any(item.get("source") == "drive" for item in weights))):
            return None
        try:
            from .drive import DriveStorage

            return DriveStorage(job, self.store).model_cache()
        except Exception as exc:  # noqa: BLE001 - the cache is an optimisation, not a requirement
            logging.getLogger(__name__).warning("model cache unavailable: %s", self._safe_error(exc))
            return None

    def _lora_cache(self, job: dict, model_cache: dict | None) -> dict | None:
        """Use the same owner-scoped Drive identity used to admit uploaded LoRAs."""
        from .accounts import AccountService
        from .drive import DriveStorage

        owner = job["owner_id"]
        accounts = AccountService(self.store, owner)
        model_account = accounts.resolve_drive_account_id(job.get("drive_account_id") or job.get("account_id"))
        if model_cache is not None and model_account in {None, accounts.default_account_id(owner)}:
            return model_cache
        storage = DriveStorage({"owner_id": owner}, self.store)
        try:
            return {"folder_id": storage.lora_folder(), "access_token": storage.google.access_token("drive")}
        except Exception as exc:
            logging.getLogger(__name__).warning("LoRA Drive authorization unavailable: %s", self._safe_error(exc))
            return None
        finally:
            storage.close()

    def _run(self, job: dict) -> dict:
        if float(job.get('poll_retry_at') or 0) > self.clock():
            return job
        lease = self._lease(job)
        if not lease:
            return self._begin_failure(job, "gpu_lease_missing", "GPU lease is missing")
        t_poll0 = time.monotonic()
        try:
            executor_job = {
                **job,
                "deadline_epoch": job.get("deadline_epoch")
                if job.get("deadline_epoch") is not None
                else lease.get("deadline_epoch"),
            }
            result = self._executor(job).poll(self._lease_handle(lease), executor_job)
        except Exception as exc:
            from .mcp import MCPTransient

            retries = int(job.get('poll_retry_count') or 0) + 1
            if isinstance(exc, MCPTransient) and retries <= 5:
                return self._set_state(job, 'running', 'GPU connection interrupted; retry pending',
                    poll_retry_count=retries, poll_retry_at=self.clock() + min(2**retries, 30))
            return self._begin_failure(job, "gpu_poll_failed", self._safe_error(exc))
        poll_s = time.monotonic() - t_poll0
        current = self._get("jobs", job["id"], job["owner_id"])
        if current and current.get("cancel_requested") and current.get("error_code") == "CU_BALANCE_FLOOR_REACHED":
            return current
        if job.get('poll_retry_count') or job.get('poll_retry_at'):
            job = self._set_state(job, 'running', 'GPU connection recovered', poll_retry_count=0, poll_retry_at=0)
        if result.get('execution_id') not in {None,job.get('execution_id')}:
            return self._begin_failure(job,'gpu_execution_id_mismatch','poll execution identity mismatch')
        if result.get("gpu_diagnostics") and (not job.get("gpu_reported")
                or (job.get("creation_fingerprint") and not job.get("gpu_diagnostics"))):
            job = self.store.record_gpu_diagnostics(job, result["gpu_diagnostics"]) or job
        state = result.get("state")
        timings = result.get("timings") if isinstance(result.get("timings"), dict) else None
        status_updated_at = result.get("status_updated_at")
        worker_age_s = (
            round(self.clock() - float(status_updated_at), 3)
            if isinstance(status_updated_at, (int, float))
            else None
        )
        if state == "completed":
            is_creation_api = bool((job.get("snapshot") or {}).get("creation_api"))
            raw_runtime = result.get("creation_runtime")
            if is_creation_api or (raw_runtime and job.get("creation_fingerprint") and job.get("instance_id")):
                runtime = raw_runtime if isinstance(raw_runtime, dict) else {}
                valid_runtime = (
                    runtime.get("instance_id") == job.get("instance_id")
                    and bool(runtime.get("process"))
                    and bool(runtime.get("environment"))
                    and runtime.get("workflow_sha256") == (job.get("snapshot") or {}).get("workflow_sha256")
                )
                if not valid_runtime:
                    if is_creation_api:
                        return self._begin_failure(job, "creation_runtime_missing", "ComfyUI 未返回有效的进程与环境记录")
                else:
                    runtime = {
                        **runtime,
                        "lease_id": lease["id"],
                        "fingerprint": job.get("creation_fingerprint"),
                        "workflow_key": job.get("workflow_key"),
                        "workflow_version": job.get("workflow_version"),
                        "devices": (job.get("gpu_diagnostics") or {}).get("devices", []),
                        "health_checked_at": self.clock(),
                    }
                    self._update("instances", job["instance_id"], {"creation_runtime": runtime}, job["owner_id"])
                    job = self._update("jobs", job["id"], {"creation_runtime": runtime}, job["owner_id"])
                    if job.get("warmup"):
                        if runtime.get("warmup_completed") is not True:
                            return self._begin_failure(job, "creation_warmup_incomplete", "预热图未完成")
                        return self._set_state(job, "cleaning_up", "模型预热图已完成",
                                               worker_timings=timings or {}, cleanup_final_state="completed")
            if (job.get("snapshot") or {}).get("native_comfy"):
                from app.config import settings

                from .native import model_storage, ready_for
                t_ready0 = time.monotonic()
                is_ready = ready_for(self.store, job["owner_id"])
                ready_s = time.monotonic() - t_ready0
                if not is_ready:
                    logging.getLogger(__name__).info(
                        "native_run_wait job=%s stage=%s poll_s=%.3f ready_s=%.3f worker_completed_age_s=%s",
                        job["id"], job.get("stage"), poll_s, ready_s, worker_age_s,
                    )
                    wait_started = job.get("native_proxy_wait_started") or self.clock()
                    wait_patch: dict[str, Any] = {
                        "stage": "native_tunnel",
                        "native_proxy_wait_started": wait_started,
                    }
                    if timings:
                        wait_patch["worker_timings"] = timings
                    if not job.get("native_proxy_wait_logged"):
                        self._event(job, "running", "等待代理健康检查及内部隧道鉴权验证", None)
                        wait_patch["native_proxy_wait_logged"] = True
                        return self._update("jobs", job["id"], wait_patch, job["owner_id"])
                    if job.get("native_proxy_wait_started") is None or (timings and timings != job.get("worker_timings")):
                        return self._update("jobs", job["id"], wait_patch, job["owner_id"])
                    return job
                if job.get("native_proxy_wait_started") is not None:
                    extra_tunnel = max(0.0, self.clock() - float(job["native_proxy_wait_started"]))
                    timings = dict(timings or job.get("worker_timings") or {})
                    base_tunnel = float(timings.get("native_tunnel") or 0.0)
                    timings["native_tunnel"] = round(base_tunnel + extra_tunnel, 1)
                t_mark0 = time.monotonic()
                deadline = min(float(lease["deadline_epoch"]),
                               float(job.get("native_session_deadline_epoch") or lease["deadline_epoch"]))
                lease = self._update("leases", lease["id"], {
                    "native_ready": True, "native_last_busy": self.clock(),
                    "native_bootstrap_job_id": job["id"],
                    "native_model_storage": model_storage(job.get("execution_policy")),
                    "deadline_epoch": deadline,
                    "native_idle_minutes": (job.get("execution_policy") or {}).get(
                        "retain_idle_minutes", settings.native_comfy_idle_minutes),
                }, job["owner_id"])
                self._update("instances", job["instance_id"], {
                    "native_comfy": True, "hard_deadline_at": deadline,
                }, job["owner_id"])
                job = self._update("jobs", job["id"], {
                    "deadline_epoch": deadline, "hard_deadline_at": deadline,
                    "native_proxy_wait_started": None,
                }, job["owner_id"])
                logging.getLogger(__name__).info(
                    "native_run_ready job=%s poll_s=%.3f ready_s=%.3f mark_ready_s=%.3f worker_completed_age_s=%s",
                    job["id"], poll_s, ready_s, time.monotonic() - t_mark0, worker_age_s,
                )
            job = self._note_restored(job, lease, timings or {})
        elif poll_s >= 3.0:
            logging.getLogger(__name__).info(
                "gpu_poll_slow job=%s state=%s stage=%s poll_s=%.3f",
                job["id"], state, result.get("stage"), poll_s,
            )
        if state == "running":
            stage = result.get("stage")
            now_epoch = self.clock()
            raw_stage_started = result.get("stage_started")
            valid_stage_started = (
                float(raw_stage_started)
                if isinstance(raw_stage_started, (int, float))
                and not isinstance(raw_stage_started, bool)
                and raw_stage_started > 0
                and abs(now_epoch - float(raw_stage_started)) <= 3600
                else None
            )
            resolved_stage_started = (
                min(valid_stage_started, now_epoch) if valid_stage_started is not None else now_epoch
            )
            if stage in _STAGE_TEXT and stage != job.get("stage"):
                updated = self._update(
                    "jobs",
                    job["id"],
                    {"stage": stage, "stage_started": resolved_stage_started, "worker_timings": timings or {}},
                    job.get("owner_id"),
                )
                self._event(updated, "running", _STAGE_TEXT[stage], None)
                return updated
            patch: dict[str, Any] = {}
            if stage in _STAGE_TEXT and job.get("stage_started") is None:
                patch["stage_started"] = resolved_stage_started
            if timings and timings != job.get("worker_timings"):
                patch["worker_timings"] = timings
            if patch:
                return self._update("jobs", job["id"], patch, job.get("owner_id"))
            return job
        if state == "failed":
            detail = str(result.get("message") or "")[:1400]
            if timings:
                job = self._update("jobs", job["id"], {"worker_timings": timings}, job.get("owner_id"))
            return self._begin_failure(
                job,
                result.get("error_code", "gpu_execution_failed"),
                "GPU execution failed" + (": " + detail if detail else ""),
            )
        if state != "completed":
            return self._begin_failure(job, "gpu_invalid_poll", "invalid GPU poll state")
        if 'artifacts' in result:
            from .output_protocol import verify_output
            try:
                manifest=result['artifacts']
                if not isinstance(manifest,list) or not 1 <= len(manifest) <= 100:
                    raise ComputeError('invalid artifact manifest count')
                if result.get('execution_id') != job.get('execution_id'):
                    raise ComputeError('artifact execution mismatch')
                verified=[verify_output(job,item)[0] for item in manifest]
                return self._set_state(job,'uploading','GPU execution completed',outputs=verified,
                                       output_bytes=sum(item['size_bytes'] for item in verified),
                                       worker_timings=timings or {})
            except Exception as exc:
                return self._begin_failure(job,'gpu_output_invalid',self._safe_error(exc))
        raw_outputs = result.get("outputs")
        if not isinstance(raw_outputs, list) or not raw_outputs:
            return self._begin_failure(job, "gpu_empty_outputs", "GPU execution produced no outputs")
        # Job snapshots are JSON columns. Keep binary provider outputs durable
        # without ever putting raw bytes or more than 100MB into the database.
        outputs, total = [], 0
        try:
            for output in raw_outputs:
                if not isinstance(output, dict):
                    raise ComputeError("GPU output format invalid")
                item = dict(output)
                content = item.get("content", b"")
                if item.get("content_encoding") == "base64":
                    content = base64.b64decode(content, validate=True)
                elif isinstance(content, str):
                    content = content.encode()
                if not isinstance(content, (bytes, bytearray)) or not content:
                    raise ComputeError("GPU output is empty")
                total += len(content)
                if total > MAX_OUTPUT_BYTES:
                    raise ComputeError("GPU output exceeds 100MB limit")
                item["content"] = base64.b64encode(bytes(content)).decode("ascii")
                item["content_encoding"] = "base64"
                outputs.append(item)
        except (ValueError, TypeError, ComputeError) as exc:
            return self._begin_failure(job, "gpu_output_invalid", self._safe_error(exc))
        return self._set_state(
            job,
            "uploading",
            "GPU execution completed",
            outputs=outputs,
            output_bytes=total,
            worker_timings=timings or {},
        )

    def _result_metadata(self, job: dict) -> dict:
        snapshot = job.get("snapshot") if isinstance(job.get("snapshot"), dict) else {}
        raw = json.dumps(
            snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
        ).encode()
        return {
            "job_id": job["id"],
            "snapshot_sha256": snapshot.get("workflow_sha256") or hashlib.sha256(raw).hexdigest(),
            "workflow_key": job.get("workflow_key"),
            "workflow_version": job.get("workflow_version"),
            "model_id": job.get("model_id"),
            "model_revision": snapshot.get("model_revision"),
            "resolved_params": snapshot.get("resolved_params", {}),
            "system_smoke": snapshot.get("system_smoke") is True,
            "gpu_reported": job.get("gpu_reported"),
            "gpu_diagnostics": job.get("gpu_diagnostics"),
        }

    def _write_local_metadata(self, job: dict, metadata: dict) -> dict:
        from app.config import settings

        root = Path(settings.data_dir) / "gpu-results" / job["owner_id"] / job["id"]
        root.mkdir(parents=True, exist_ok=True)
        target = root / "metadata.json"
        content = json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        tmp = target.with_suffix(".partial")
        tmp.write_bytes(content)
        tmp.replace(target)
        return {
            "filename": "metadata.json",
            "storage_key": str(target),
            "sha256": hashlib.sha256(content).hexdigest(),
            "mime": "application/json",
            "metadata": metadata,
        }

    def _upload(self, job: dict) -> dict:
        from .node_persistence import env_flag

        outputs, assets = job.get("outputs") or [], []
        report = None
        native_boot = bool((job.get("snapshot") or {}).get("native_comfy"))
        skip_native_drive = native_boot and env_flag("NATIVE_FAST_CONTROL_PLANE", True)
        owns_storage = False
        if self.storage_factory:
            storage = self._factory(self.storage_factory, job)
        elif not skip_native_drive and os.getenv("GPU_STORAGE", "local").strip().lower() == "drive":
            from .drive import DriveStorage

            storage = DriveStorage(job, self.store)
            owns_storage = True
        else:
            storage = None
        result_metadata = self._result_metadata(job)
        metadata_record = None
        try:
            if job.get('prep'):
                for output in outputs:
                    if 'storage' in output:
                        from .output_protocol import read_diagnostic_report
                        report=read_diagnostic_report(job,output)
                    elif isinstance(output.get('metadata'),dict):
                        report=output['metadata']
                if not isinstance(report,dict):
                    raise ComputeError('prep output missing verified diagnostic report')
                outputs=[]
                result_metadata['prep_report']=report
            for index, output in enumerate(outputs):
                if 'storage' in output:
                    from .output_protocol import commit_output
                    metadata = {**result_metadata, **{k: output[k] for k in ("comfy_node", "comfy_category") if k in output}}
                    assets.append(commit_output(self.db_store, job, output, metadata, storage=storage))
                    continue
                content = output.get("content", b"")
                if output.get("content_encoding") == "base64":
                    content = base64.b64decode(content, validate=True)
                elif isinstance(content, str):
                    content = content.encode()
                if (
                    not isinstance(content, (bytes, bytearray))
                    or not content
                    or len(content) > MAX_OUTPUT_BYTES
                ):
                    raise ComputeError("GPU output exceeds 100MB or is empty")
                metadata = {**result_metadata, **dict(output.get("metadata") or {})}
                if self._provider_name(job) == "mock":
                    metadata["mock"] = True
                else:
                    metadata.pop("mock", None)
                if storage is not None:
                    key = f"outputs/{job['id']}/{index}-{output.get('filename', 'output')}"
                    cloud_record = storage.put(
                        key, bytes(content), output.get("mime", "application/octet-stream")
                    )
                    if not storage.verify(cloud_record, bytes(content)):
                        raise ComputeError("stored GPU output failed verification")
                    metadata = {**metadata, "cloud_record": cloud_record}
                from app.storage import save_asset

                assets.append(save_asset(
                    self.db_store,
                    job["owner_id"],
                    job.get("project_id"),
                    bytes(content),
                    output.get("filename", "output"),
                    job["id"],
                    metadata,
                ))
            metadata_content = json.dumps(
                result_metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode()
            if storage is not None:
                metadata_record = storage.put(
                    f"outputs/{job['id']}/metadata.json", metadata_content, "application/json"
                )
                if not storage.verify(metadata_record, metadata_content):
                    raise ComputeError("stored GPU metadata failed verification")
            else:
                metadata_record = self._write_local_metadata(job, result_metadata)
        except Exception as exc:
            attempts = int(job.get("upload_attempts") or 0) + 1
            if attempts >= 3:
                return self._begin_failure(job, "gpu_upload_failed", self._safe_error(exc))
            return self._update("jobs", job["id"], {"upload_attempts": attempts}, job.get("owner_id"))
        finally:
            if owns_storage and hasattr(storage, "close"):
                storage.close()
        # Drive metadata file IDs are not authenticated application asset IDs.
        output_asset_ids = [a.get("id") for a in assets if a.get("id")]
        if metadata_record:
            assets.append(metadata_record)
        return self._set_state(
            job,
            "cleaning_up",
            "GPU outputs persisted",
            output_asset_ids=output_asset_ids,
            persisted_outputs=assets,
            **({'prep_report':report} if report is not None else {}),
            cleanup_final_state="completed",
        )

    def _close_cu_session(self, owner, instance_id, ended_at, reason):
        ccu.close_instance_session(self.store, owner, instance_id, ended_at, reason)

    def _record_usage(self, job: dict, lease: dict, stopped_at: float) -> None:
        if self._provider_name(job) == "mock":
            started = float(job.get("started_epoch") or lease.get("created_epoch") or stopped_at)
            self.store.record_usage_once(job["owner_id"], {
                "job_id": job["id"], "execution_id": job.get("execution_id"), "provider": "mock",
                "mock": True, "simulated": True, "units": 0.0, "units_estimate": 0.0, "units_actual": 0.0,
                "elapsed_minutes": max(0, stopped_at - started) / 60, "created_epoch": stopped_at,
                "started_at": started, "stopped_at": stopped_at, "source": "simulated",
            })
        else:
            self._close_cu_session(job["owner_id"], job.get("instance_id") or lease.get("instance_id"), stopped_at, "confirmed_delete")
        self._update("jobs", job["id"], {
            "stopped_at": stopped_at, "runtime_deleted": True,
            **({"actual_units": 0.0} if self._provider_name(job) == "mock" else {}),
        }, job["owner_id"])

    def _finish_cleanup(self, job: dict, state: str, message: str, **patch) -> dict:
        if state in {"completed", "cancelled"} and job.get("error_code") in {
            "gpu_cleanup_pending", "gpu_instance_completion_pending",
        }:
            patch.update(error_code=None, error_message=None)
        return self._set_state(job, state, message, cleanup_next_epoch=0, **patch)

    def _inspect_runtime(self, lease: dict, claim_token: str | None = None) -> str | None:
        """Check a bound runtime independently of Comfy, retention and idle deadlines."""
        from .native import _lease_identity

        if (
            lease.get("provider") != "colab"
            or lease.get("state") not in {"active", "idle", "releasing", "cleanup_pending"}
            or not all(lease.get(key) for key in ("account_id", "instance_id", "owner_job_id", "runtime_id", "request_id"))
            or not isinstance(lease.get("handle"), dict)
            or lease.get("cleanup_claim_token")
        ):
            return None
        # A CLI intent without an endpoint must go through cleanup's exact
        # notebook lookup before it can use bound-runtime inspection.
        if lease.get("tunnel_nbh") and not lease.get("tunnel_endpoint"):
            return None
        owner = lease["owner_id"]
        instance = self._get("instances", lease["instance_id"], owner)
        if (
            not instance or instance.get("state") == "deleted"
            or instance.get("lease_id") != lease["id"] or instance.get("account_id") != lease["account_id"]
            or instance.get("current_job_id") not in {None, lease["owner_job_id"]}
            or instance.get("cleanup_claim_token")
        ):
            return None
        identity = digest(_lease_identity(lease))
        instance_version = instance.get("version", 0)
        check = lease.get("provider_runtime_check") or {}
        if (
            check.get("identity") == identity and check.get("instance_version") == instance_version
            and self.clock() < float(check.get("next_check_at") or 0)
        ):
            return check["status"]
        job_id = lease["owner_job_id"]
        job_snapshot = self._get("jobs", job_id, owner) or {}
        if (
            job_snapshot.get("claim_token") not in {None, claim_token}
            and float(job_snapshot.get("claim_until") or 0) > self.clock()
        ):
            return "stale"
        handle = {**lease["handle"], "job_id": job_id, "owner_job_id": job_id, "inspection_lease_id": lease["id"]}
        try:
            provider = self._provider({
                "id": job_id, "owner_id": owner, "provider": "colab", "account_id": lease["account_id"],
            })
            try:
                status = runtime_status(provider.inspect(handle))
            finally:
                if isinstance(provider, GoogleColabProvider):
                    provider.close()
        except Exception:
            # Transport, authentication and provider errors carry no deletion evidence.
            status = "unknown"
        if not self._instance_call(
            "apply_runtime_inspection", lease, instance_version, job_snapshot, identity, status, self.clock(), claim_token,
        ):
            return "stale"
        return status

    def _cleanup(self, job: dict) -> dict:
        lease = self._lease(job)
        observation = None
        if lease:
            if not lease.get("force_stop"):
                observation = self._inspect_runtime(lease, job.get("claim_token"))
            if observation == "stale":
                return self._get("jobs", job["id"], job["owner_id"]) or job
            if observation == "missing" or lease.get("remote_delete_source") == RUNTIME_MISSING:
                current = self._get("jobs", job["id"], job["owner_id"]) or job
                final_state = current.get("cleanup_final_state") or "failed"
                self._record_usage(current, lease, self.clock())
                return self._finish_cleanup(current, final_state, "GPU cleanup completed")
        if not (lease or {}).get("force_stop"):
            job = self._trusted_plugin_cleanup(job, lease)
        instance = self._service_instance_for_job(job)
        retention_allowed = not (instance or {}).get("manual_stop_requested")
        if self._provider_name(job) == "colab":
            retention_allowed = retention_allowed and not CuGuardService(self.store, self.clock).state(job["owner_id"])["soft_tripped"]
        if lease and retention_allowed and job.get("batch_id") and job.get("cleanup_final_state") == "completed":
            waiting = [j for j in self._list("jobs", job["owner_id"])
                       if j.get("batch_id") == job["batch_id"] and j["state"] not in TERMINAL_STATES
                       and j["id"] != job["id"] and not j.get("cancel_requested")]
            waiting.sort(key=lambda j: j.get("queue_order", 0))
            next_job = waiting[0] if waiting else None
            if next_job:
                with self.db_store.dispatch_lock() as acquired:
                    if not acquired:
                        return job
                    transferred=self.store.handoff_lease(job['owner_id'],job['id'],next_job['id'],lease['id'],self.clock())
                if transferred:
                    msg = (
                        "Mock outputs verified; lease reused"
                        if self._provider_name(job) == "mock"
                        else "GPU outputs verified; lease reused"
                    )
                    self._event(job, "completed", msg)
                    return self._get("jobs", job["id"], job["owner_id"])
        final_state = job.get("cleanup_final_state") or job.get("requested_outcome") or "failed"
        from .instances import POOL_CREATION, job_pool, lease_allocation_unconfirmed

        if (
            final_state == "cancelled"
            and lease
            and lease.get("handle")
            and self._provider_name(job) == "colab"
            and job_pool(job) == POOL_CREATION
            and self._lifecycle(job) in {"smart", "retain"}
            and retention_allowed
            and not lease.get("force_stop")
            and not job.get("creation_interrupt_confirmed")
        ):
            interrupted = False
            try:
                executor = self._executor(job)
                interrupt_fn = getattr(executor, "interrupt_creation_service", None)
                if callable(interrupt_fn):
                    interrupted = interrupt_fn(self._lease_handle(lease), job) is True
            except Exception:
                pass
            if interrupted:
                job = self._update("jobs", job["id"], {"creation_interrupt_confirmed": True}, job.get("owner_id"))
            else:
                # Keep the lease unavailable until the provider confirms deletion.
                lease = self._update("leases", lease["id"], {"force_stop": True}, job["owner_id"])
        owns_instance = bool(
            instance is None
            or instance.get("current_job_id") in {None, job["id"]}
            or job.get("lease_id") is not None
        )
        if self._is_v020_job(job) and job.get("instance_id") and owns_instance:
            try:
                job, instance = self._complete_instance(job, final_state == "completed")
            except SchedulerError as exc:
                return self._set_state(
                    job,
                    "cleanup_pending",
                    "GPU instance completion is pending",
                    error_code=job.get("error_code") or "gpu_instance_completion_pending",
                    error_message=job.get("error_message") or str(exc),
                )
            can_retain_lease = not (lease and lease_allocation_unconfirmed(lease.get("state"), lease, job))
            if can_retain_lease and isinstance(instance, dict) and instance.get("state") == "retained_idle":
                with self.db_store.admission_lock() as acquired:
                    if not acquired:
                        return job
                    instance = self._service_instance_for_job(job)
                    if lease and instance and instance.get("state") == "retained_idle":
                        lease_patch = {"state": "idle", "current_job_id": None, "last_seen": self.clock()}
                        if instance.get("idle_deadline_at") is not None:
                            lease_patch["idle_deadline_at"] = instance["idle_deadline_at"]
                        if instance.get("hard_deadline_at") is not None:
                            lease_patch["deadline_epoch"] = instance["hard_deadline_at"]
                        elif lease.get("generation_timeouts") and instance.get("idle_deadline_at") is not None:
                            lease_patch["deadline_epoch"] = instance["idle_deadline_at"]
                        self._update("leases", lease["id"], lease_patch, job.get("owner_id"))
            if can_retain_lease and isinstance(instance, dict) and instance.get("state") == "retained_idle":
                if job.get("execution_id"):
                    self._update(
                        "executions",
                        job["execution_id"],
                        {"state": final_state, "finished_at": self.clock()},
                        job.get("owner_id"),
                    )
                return self._finish_cleanup(
                    job,
                    final_state,
                    "GPU cleanup completed; instance retained",
                    runtime_deleted=False,
                )
        policy=job.get('execution_policy') or {}
        if (lease and job.get('batch_id') and final_state=='completed'
                and policy.get('mode')=='smart' and self._provider_name(job)=='mock'
                and not self._is_v020_job(job)
                and self.clock()<float(lease.get('deadline_epoch') or 0)):
            with self.db_store.dispatch_lock() as acquired:
                if not acquired:
                    return job
                self._update('leases',lease['id'],{'state':'idle','current_job_id':None,
                    'compatibility_signature':job.get('compatibility_signature'),
                    'idle_deadline_at':min(self.clock()+int(policy.get('idle_seconds',120)),lease['deadline_epoch']),
                    'last_active_at':self.clock()},job['owner_id'])
                if job.get('execution_id'):
                    self._update('executions',job['execution_id'],{'state':'completed','finished_at':self.clock()},job['owner_id'])
                return self._set_state(job,'completed','Mock outputs verified; idle lease retained')
        if not lease:
            job = self._close_unstarted_instance(job, None)
            return self._finish_cleanup(job, final_state, "GPU cleanup completed")
        if lease.get("state") == "creating" and not lease.get("handle") and not lease.get("create_attempts"):
            lease = self._update("leases", lease["id"], {"state": "deleted", "last_seen": self.clock()}, job["owner_id"])
        if lease.get("state") == "deleted":
            job = self._close_unstarted_instance(job, lease)
            self._record_usage(job, lease, self.clock())
            return self._finish_cleanup(job, final_state, "GPU cleanup completed")
        next_epoch = float(lease.get("cleanup_next_epoch") or 0)
        if self.clock() < next_epoch or (observation == "unknown" and not lease.get("force_stop")):
            return job
        if lease.get("native_ready") and not lease.get("force_stop"):
            from .native import protect_lease
            if protect_lease(self.store, lease, self.clock, check_remote=observation is None):
                return job
        provider = self._provider(job)
        handle = self._lease_handle(lease) or self._intent_handle(job, lease)
        if handle is None:
            self._update(
                "leases",
                lease["id"],
                {
                    "state": "quarantined",
                    "last_error": "lease intent is incomplete",
                    "last_seen": self.clock(),
                },
                job.get("owner_id"),
            )
            return self._set_state(job, "failed", "GPU cleanup was refused", error_code="gpu_cleanup_refused")
        try:
            confirmed = False
            if not lease.get("force_stop"):
                status = provider.inspect(handle)
                observation = runtime_status(status) if lease.get("provider") == "colab" else None
                confirmed = observation == "missing" if observation else status.get("deleted") is True
            if not confirmed and (observation != "unknown" or lease.get("force_stop")):
                response = provider.delete(handle)
                # A truthy response is not enough: providers may return an
                # accepted/pending response before the remote runtime is gone.
                confirmed = response is True or (
                    isinstance(response, dict) and response.get("deleted") is True
                )
            error = None
        except Exception as exc:
            confirmed, error = False, self._safe_error(exc)
        attempts = int(lease.get("delete_attempts") or 0) + 1
        if confirmed:
            try:
                job = self._mark_instance_deleted(job)
            except SchedulerError as exc:
                confirmed = False
                error = str(exc)
        if not confirmed:
            delay = min(
                self.config.cleanup_retry_max_seconds,
                self.config.cleanup_retry_base_seconds * (2 ** max(0, attempts - 1)),
            )
            self._update(
                "leases",
                lease["id"],
                {
                    "state": "cleanup_pending",
                    "delete_attempts": attempts,
                    "cleanup_next_epoch": self.clock() + delay,
                    "last_error": error or "delete not confirmed",
                    "last_seen": self.clock(),
                },
                job.get("owner_id"),
            )
            return self._set_state(
                job,
                "cleanup_pending",
                "GPU cleanup is pending",
                cleanup_attempts=attempts,
                cleanup_next_epoch=self.clock() + delay,
                error_code=job.get("error_code") or "gpu_cleanup_pending",
            )
        self._update(
            "leases",
            lease["id"],
            {"state": "deleted", "delete_attempts": attempts, "last_seen": self.clock(),
             "remote_deleted": True, "remote_deleted_at": self.clock(),
             "remote_delete_source": "cleanup_provider_confirmation"},
            job.get("owner_id"),
        )
        self._record_usage(job, lease, self.clock())
        return self._finish_cleanup(job, final_state, "GPU cleanup completed")

    def _advance(self, job: dict) -> dict:
        state = job.get("state")
        if state in TERMINAL_STATES:
            return job
        if job.get("cancel_requested") and state not in {"cleaning_up", "cleanup_pending"}:
            if self._lease(job) or job.get("instance_id"):
                job = self._set_state(
                    job,
                    "cleaning_up",
                    "GPU cancellation requested",
                    requested_outcome="cancelled",
                    cleanup_final_state="cancelled",
                )
            else:
                return self._set_state(job, "cancelled", "GPU job cancelled", requested_outcome="cancelled")
        if job.get("state") == "cleanup_pending":
            return self._cleanup(job)
        if self._expired(job) and job.get("state") not in {
            "queued",
            "validating",
            "waiting_for_capacity",
            "cleaning_up",
            "cleanup_pending",
        }:
            if (job.get("snapshot") or {}).get("native_comfy") and job.get("stage") == "native_tunnel":
                return self._begin_failure(job, "native_proxy_unready", "原生代理健康检查超时：请检查 named tunnel、Access Service Auth 策略、service token 以及内部服务路由")
            return self._begin_failure(job, "gpu_timeout", "GPU job exceeded its runtime limit")
        state = job.get("state")
        if state == "queued":
            return self._set_state(job, "validating", "GPU job validating")
        if state == "validating":
            return self._validate(job)
        if state == "waiting_for_capacity":
            # CPU cache preparation owns the first wait.  Do not allocate a
            # durable/runtime instance until the prep decision is settled.
            try:
                self._check_cu(job)
            except APIError as exc:
                return self._begin_failure(job, exc.code, exc.message)
            if self._await_prep(job):
                return job if job.get("awaiting_prep") else self._update("jobs", job["id"], {"awaiting_prep": True}, job.get("owner_id"))
            if self._is_v020_job(job):
                allocated = self._allocate_instance(job)
                if allocated.get("state") in TERMINAL_STATES | {"cleaning_up", "cleanup_pending"}:
                    return allocated
                job = allocated
                if not job.get("lease_id") and job.get("instance_id"):
                    inst = self._service_instance_for_job(job)
                    if inst and inst.get("lease_id"):
                        job = self._reuse_instance_lease(job, inst)
                if not job.get("instance_id"):
                    return job
                if job.get("selection_reason") in {"reuse_idle", "requested_instance"}:
                    if not self._lease(job):
                        return self._begin_failure(job, "creation_lease_unavailable", "实例租约已不可复用，请重试")
            elif job.get('compatibility_signature'):
                with self.db_store.dispatch_lock() as acquired:
                    if not acquired:
                        return job
                    for idle in self._leases_for(job['owner_id']):
                        if self.store.attach_idle_lease(job['owner_id'],job['id'],idle['id'],self.clock()):
                            job=self._get('jobs',job['id'],job['owner_id'])
                            break
            if not self._is_v020_job(job) and self._has_capacity_blocker(job):
                return job
            task_deadline = ({"task_deadline_epoch": self.clock() + self._effective_runtime(job) * 60}
                             if job.get("generation_timeouts") or job.get("creation_fingerprint") else {})
            if job.get("generation_timeouts"):
                lease = self._lease(job)
                if lease and lease.get("generation_timeouts") and lease.get("hard_deadline_at") is None:
                    deadline = task_deadline["task_deadline_epoch"]
                    self._update("leases", lease["id"], {"deadline_epoch": deadline}, job["owner_id"])
                    task_deadline["deadline_epoch"] = deadline
            return self._set_state(job, "allocating", "GPU capacity available", awaiting_prep=False, **task_deadline)
        if state == "allocating":
            return self._allocation(job)
        if state == "preparing":
            return self._prepare(job)
        if state == "loading":
            return self._load(job)
        if state == "running":
            return self._run(job)
        if state == "uploading":
            return self._upload(job)
        if state == "cleaning_up":
            return self._cleanup(job)
        return self._begin_failure(job, "gpu_invalid_state", f"unknown GPU state: {state}")

    def _rr_state(self) -> dict[str, Any]:
        rr = getattr(self.db_store, "_scheduler_rr", None)
        if rr is None:
            rr = {"seq": 0, "stalled": {}}
            self.db_store._scheduler_rr = rr
        return rr

    def _claim_next_job(self) -> tuple[dict | None, str | None]:
        now = self.clock()
        rr = self._rr_state()
        stalled: dict[str, int] = rr["stalled"]
        jobs = self._list("jobs")
        active_ids = {job["id"] for job in jobs if job.get("state") in ACTIVE_STATES}
        for stale_id in [jid for jid in stalled if jid not in active_ids]:
            stalled.pop(stale_id, None)
        jobs.sort(
            key=lambda x: (
                # Cleanup goes first; a job parked on its cache prep goes last so it cannot starve its prep job.
                0 if x.get("state") in {"cleanup_pending", "cleaning_up"} else 2 if x.get("awaiting_prep") else 1,
                # Jobs that polled without changing state yield to newly queued/advancing jobs in FIFO round-robin.
                stalled.get(x["id"], 0),
                x.get("created_epoch", x.get("created_at", "")),
            )
        )
        for job in jobs:
            state = job.get("state")
            if state not in ACTIVE_STATES:
                continue
            # Do not repeatedly claim/unclaim a running job sitting in poll backoff unless cancelled or expired.
            if (
                state == "running"
                and float(job.get("poll_retry_at") or 0) > now
                and not job.get("cancel_requested")
                and not self._expired(job)
            ):
                continue
            until = float(job.get("claim_until") or 0)
            if job.get("claim_token") and until > now:
                continue
            token = uuid.uuid4().hex
            claimed = self._update(
                "jobs",
                job["id"],
                {"claim_token": token, "claim_until": now + self.config.claim_ttl_seconds},
                job.get("owner_id"),
            )
            return claimed, token
        return None, None

    def _release_claim(self, job_id: str, token: str | None, owner_id: str | None = None) -> None:
        if not token:
            return
        current = self._get("jobs", job_id, owner_id)
        if current and current.get("claim_token") == token:
            self._update("jobs", job_id, {"claim_token": None, "claim_until": None}, owner_id)

    def _next_job(self) -> dict | None:
        return next(
            (x for x in self._list("jobs") if x.get("state") in ACTIVE_STATES and not x.get("claim_token")),
            None,
        )

    def tick(self) -> dict:
        if not self.config.enabled:
            return {"acquired": True, "advanced": False, "reason": "disabled"}
        with self.db_store.dispatch_lock() as acquired:
            if not acquired:
                return {"acquired": False, "advanced": False, "reason": "busy"}
            job, claim = self._claim_next_job()
        if job is None:
            return {"acquired": True, "advanced": False}
        before = job.get("state")
        after = job
        try:
            after = self._advance(job)
        finally:
            self._release_claim(job["id"], claim, job.get("owner_id"))
        advanced = before != after.get("state")
        rr = self._rr_state()
        if after.get("state") in TERMINAL_STATES or advanced:
            rr["stalled"].pop(job["id"], None)
        else:
            rr["seq"] += 1
            rr["stalled"][job["id"]] = rr["seq"]
        return {
            "acquired": True,
            "advanced": advanced,
            "job_id": job["id"],
            "state": after.get("state"),
        }

    def cancel(self, owner_id: str, job_id: str) -> dict:
        job = self._get("jobs", job_id, owner_id)
        if not job:
            raise KeyError(job_id)
        if job.get("state") in TERMINAL_STATES:
            return job
        if job.get("state") == "queued" and not self._lease(job):
            return self._set_state(
                job, "cancelled", "GPU job cancelled", cancel_requested=True, requested_outcome="cancelled"
            )
        return self._update(
            "jobs", job_id, {"cancel_requested": True, "requested_outcome": "cancelled"}, owner_id
        )

    def stop(self, owner_id: str) -> dict:
        count = 0
        for job in self._list("jobs", owner_id):
            if job.get("state") not in TERMINAL_STATES:
                self.cancel(owner_id, job["id"])
                count += 1
        return {"owner_id": owner_id, "cancelled": count}


__all__ = ["Scheduler", "SchedulerError", "TERMINAL_STATES", "ACTIVE_STATES"]
