"""Instance lifecycle mixin extracted from scheduler.py."""

from __future__ import annotations

import logging
from typing import Any

from app.common import APIError

from .instances import CreationInstancePending


class SchedulerError(RuntimeError):
    pass


class SchedulerInstancesMixin:
    """InstanceService coordination, lease rebinding, and instance usage closing."""

    def _instance_call(self, method: str, *args, **kwargs):
        service = self.instance_service
        if service is None:
            return None
        # Callers may replace ``clock`` after construction; deadlines must share it.
        service.clock = self.clock
        return getattr(service, method)(*args, **kwargs)

    @staticmethod
    def _instance_id(instance: dict | None) -> str | None:
        if not isinstance(instance, dict):
            return None
        value = instance.get("instance_id") or instance.get("id")
        return value if isinstance(value, str) and value else None

    @staticmethod
    def _instance_epoch(value: Any) -> float | None:
        return None if value is None else float(value)

    def _instance_patch(self, instance: dict | None) -> dict:
        if not isinstance(instance, dict):
            return {}
        patch = {}
        instance_id = self._instance_id(instance)
        if instance_id:
            patch["instance_id"] = instance_id
        for key in (
            "account_id",
            "drive_account_id",
            "account_alias_snapshot",
            "instance_label",
            "instance_pool",
            "gpu_reported",
            "retention_mode",
            "retain_idle_minutes",
            "retained_until",
            "idle_deadline_at",
            "hard_deadline_at",
            "selection_reason",
        ):
            if key in instance and instance[key] is not None:
                patch[key] = instance[key]
        if instance.get("state") is not None:
            patch["instance_state"] = instance["state"]
        hard_deadline = self._instance_epoch(instance.get("hard_deadline_at"))
        if hard_deadline is not None:
            patch["deadline_epoch"] = hard_deadline
        return patch

    def _merge_instance(self, job: dict, instance: dict | None) -> dict:
        patch = self._instance_patch(instance)
        if not patch:
            return job
        return self._update("jobs", job["id"], patch, job.get("owner_id"))

    def _service_instances(self, *, states: tuple[str, ...] | None = None) -> list[dict]:
        if self.instance_service is None:
            return []
        if hasattr(self.store, "distinct_owners"):
            owners = self.store.distinct_owners("instances")
        else:
            owners = {job.get("owner_id") for job in self._list("jobs") if job.get("owner_id")}
        if not owners:
            return []
        if states is not None:
            result = []
            for owner in owners:
                for state in states:
                    result.extend(self.instance_service.list(owner, state=state))
            return result
        return [item for owner in owners for item in self.instance_service.list(owner)]

    def _service_instance_for_job(self, job: dict) -> dict | None:
        instance_id = job.get("instance_id")
        if not instance_id or self.instance_service is None:
            return None
        return self.instance_service.get(job.get("owner_id"), instance_id)

    def _lifecycle(self, job: dict) -> str:
        policy = job.get("execution_policy") or {}
        lifecycle = policy.get("lifecycle") or job.get("lifecycle") or policy.get("mode")
        if lifecycle in {"save", "batch", "smart", "retain"}:
            return lifecycle
        if policy.get("retention") in {True, "retain"} or job.get("retention_requested"):
            return "retain"
        return "save"

    def _instance_job(self, job: dict) -> dict:
        """Give InstanceService the legacy batch mode as an explicit lifecycle.

        Batch jobs created before v0.2 store ``mode=smart`` in their policy,
        while InstanceService deliberately reads ``lifecycle``.  Translate
        only the service-facing copy so the persisted legacy request remains
        unchanged and the instance service stays authoritative for retention.
        """
        lifecycle = self._lifecycle(job)
        policy = job.get("execution_policy") or {}
        if policy.get("lifecycle") == lifecycle:
            return job
        return {**job, "execution_policy": {**policy, "lifecycle": lifecycle}}

    def _is_v020_job(self, job: dict) -> bool:
        return self.instance_service is not None and not job.get("legacy_scheduler_only")

    def _allocate_instance(self, job: dict) -> dict:
        if not self._is_v020_job(job):
            return job
        if job.get("instance_id"):
            if job.get("selection_reason") in {"reuse_idle", "requested_instance"} and not self._lease(job):
                return self._reuse_instance_lease(job, self._service_instance_for_job(job))
            return job
        try:
            instance = self._instance_call("allocate", self._instance_job(job))
        except CreationInstancePending:
            return job
        except APIError as exc:
            return self._begin_failure(job, exc.code, exc.message)
        except Exception as exc:
            return self._begin_failure(job, "gpu_instance_allocation_failed", self._safe_error(exc))
        if instance is None:
            return job
        # InstanceService._claim_existing atomically takes over the lease and updates
        # the job row inside one transaction. Reload before merging so we never
        # overwrite lease_id / deadline_epoch / unit_rate with a stale job dict.
        job = self._get("jobs", job["id"], job.get("owner_id")) or job
        if not job.get("lease_id") and instance.get("lease_id"):
            job = self._reuse_instance_lease(job, instance)
        return self._merge_instance(job, instance)

    def _reuse_instance_lease(self, job: dict, instance: dict | None) -> dict:
        """Retarget a retained lease under the same admission fence as claim/allocate."""
        with self.db_store.admission_lock(timeout=1.0) as acquired:
            if not acquired:
                return job
            return self._reuse_instance_lease_locked(job, instance)

    def _reuse_instance_lease_locked(self, job: dict, instance: dict | None) -> dict:
        """Retarget the retained instance's lease instead of creating a second runtime."""
        if not self._is_v020_job(job) or not isinstance(instance, dict):
            return job
        lease_id = instance.get("lease_id")
        if not isinstance(lease_id, str) or not lease_id:
            return job
        lease = self._get("leases", lease_id, job.get("owner_id"))
        if not lease or lease.get("state") != "idle":
            return job
        deadline = lease.get("deadline_epoch")
        idle_deadline = lease.get("idle_deadline_at")
        if not lease.get("native_ready") and idle_deadline is not None and self.clock() >= float(idle_deadline):
            return job
        if deadline is not None and self.clock() >= float(deadline):
            return job
        handle = lease.get("handle")
        lease_patch = {
            "state": "active",
            "owner_job_id": job["id"],
            "current_job_id": job["id"],
            "idle_deadline_at": None,
            "last_active_at": self.clock(),
        }
        if isinstance(handle, dict):
            lease_patch["handle"] = {
                **handle,
                "job_id": job["id"],
                "owner_job_id": job["id"],
            }
        self._update("leases", lease_id, lease_patch, job.get("owner_id"))
        job_patch = {"lease_id": lease_id}
        if deadline is not None:
            job_patch["deadline_epoch"] = deadline
        return self._update("jobs", job["id"], job_patch, job.get("owner_id"))

    def _bind_instance_lease(self, job: dict, lease: dict) -> tuple[dict, dict]:
        if not self._is_v020_job(job) or not job.get("instance_id"):
            return job, lease
        if lease.get("instance_id") == job.get("instance_id") and lease.get("instance_bound"):
            return job, lease
        try:
            instance = self._instance_call("bind_lease", self._instance_job(job), lease)
        except Exception as exc:
            raise SchedulerError(f"instance lease binding failed: {self._safe_error(exc)}") from exc
        patch = {"instance_id": job["instance_id"], "instance_bound": True}
        if job.get("account_id"):
            patch["account_id"] = job["account_id"]
        lease = self._update("leases", lease["id"], patch, job.get("owner_id"))
        if instance:
            job = self._merge_instance(job, instance)
        return job, lease

    def _mark_instance(self, method: str, job: dict, **kwargs) -> dict:
        if not self._is_v020_job(job) or not job.get("instance_id"):
            return job
        try:
            instance = self._instance_call(method, self._instance_job(job), **kwargs)
        except Exception as exc:
            logging.getLogger(__name__).warning("instance %s failed: %s", method, self._safe_error(exc))
            return job
        return self._merge_instance(job, instance)

    def _complete_instance(self, job: dict, success: bool) -> tuple[dict, dict | None]:
        """Let InstanceService decide whether a completed job is retained or released."""
        if not self._is_v020_job(job) or not job.get("instance_id"):
            return job, None
        if job.get("instance_completion_recorded"):
            return job, self._service_instance_for_job(job)
        try:
            instance = self._instance_call("complete", self._instance_job(job), success=success)
        except Exception as exc:
            raise SchedulerError(f"instance completion failed: {self._safe_error(exc)}") from exc
        if (
            isinstance(instance, dict)
            and instance.get("state") == "retained_idle"
            and not instance.get("native_comfy")
            and self._lifecycle(job) == "smart"
        ):
            policy = job.get("execution_policy") or {}
            idle_seconds = policy.get("idle_seconds") if policy.get("mode") == "smart" else None
            if idle_seconds is not None:
                # Batch smart mode specifies seconds; apply it through the service's CAS path.
                instance = self._instance_call(
                    "retention",
                    job.get("owner_id"),
                    self._instance_id(instance),
                    "smart",
                    float(idle_seconds) / 60.0,
                    f"scheduler-smart:{job['id']}",
                )
        job = self._merge_instance(job, instance)
        job = self._update(
            "jobs", job["id"], {"instance_completion_recorded": True}, job.get("owner_id")
        )
        return job, instance if isinstance(instance, dict) else self._service_instance_for_job(job)

    def _record_instance_usage(self, job: dict, instance: dict | None, ended_at: float) -> None:
        """Close one idempotent instance-lifetime segment after remote deletion."""
        if not self._is_v020_job(job):
            return
        instance_id = job.get("instance_id") or self._instance_id(instance)
        if not instance_id:
            return
        instance = instance if isinstance(instance, dict) else {}
        if self._provider_name(job) == "colab":
            ended_at = self._instance_epoch(instance.get("deletion_confirmed_at")) or ended_at
            self._close_cu_session(job["owner_id"], instance_id, ended_at, "confirmed_delete")
            return
        started_at = (
            instance.get("allocation_requested_at")
            or job.get("started_epoch")
            or job.get("started_at")
        )
        confirmed_at = instance.get("deletion_confirmed_at") or ended_at
        started_epoch = self._instance_epoch(started_at)
        confirmed_epoch = self._instance_epoch(confirmed_at) or float(ended_at)
        elapsed_minutes = (
            max(0.0, confirmed_epoch - started_epoch) / 60.0
            if started_epoch is not None
            else 0.0
        )
        is_mock = self._provider_name(job) == "mock"
        rate = float(job.get("unit_rate") or self.config.unit_rate or 0.0)
        payload = {
            "instance_id": instance_id,
            "job_id": job.get("id"),
            "segment_key": f"instance-lifetime:{instance_id}",
            "allocation_requested_at": started_at,
            "deletion_confirmed_at": confirmed_at,
            "started_at": started_at,
            "ended_at": confirmed_at,
            "estimated_units": 0.0 if is_mock else elapsed_minutes * rate,
            # Provider billing is not available at deletion time; never invent CU.
            "actual_units": None,
        }
        self.instance_service.record_usage(job.get("owner_id"), payload)

    def _mark_instance_deleted(self, job: dict) -> dict:
        """Record deletion only after the provider has explicitly confirmed it."""
        if not self._is_v020_job(job) or not job.get("instance_id"):
            return job
        try:
            instance = self._instance_call(
                "mark_deleted", job.get("owner_id"), job["instance_id"],
                remote_deleted=True, expected_version=job.get("_cleanup_instance_version"),
            )
            self._record_instance_usage(job, instance, self.clock())
        except Exception as exc:
            raise SchedulerError(f"instance deletion confirmation failed: {self._safe_error(exc)}") from exc
        # Watchdog orphan cleanup has no job row to mirror; the ledger still
        # closes against the instance service and the lease can be finalized.
        if self._get("jobs", job["id"], job.get("owner_id")) is None:
            return job
        return self._merge_instance(job, instance)

    def _close_unstarted_instance(self, job: dict, lease: dict | None) -> dict:
        """任务在创建运行时之前就失败时，远端没有机器可删，实例直接收尾。

        否则实例会一直停在「释放中」，看起来像失败后还在继续跑。
        """
        if not self._is_v020_job(job) or not job.get("instance_id"):
            return job
        lease = lease or {}
        if lease.get("handle") or int(lease.get("create_attempts") or 0):
            return job
        instance = self._service_instance_for_job(job)
        if not instance or instance.get("state") != "releasing" or instance.get("lease_id") not in (None, lease.get("id")):
            return job
        instance = self._instance_call("mark_deleted", job.get("owner_id"), job["instance_id"], confirmed=True, expected_version=instance.get("version"))
        return self._merge_instance(job, instance)
