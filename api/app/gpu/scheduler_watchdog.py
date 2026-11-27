"""Watchdog, orphan-lease cleanup, and historical reconciliation mixin extracted from scheduler.py."""

from __future__ import annotations

import uuid

from . import ccu
from .colab import runtime_status
from .cu_guard import CuGuardService
from .scheduler_instances import SchedulerError

TERMINAL_STATES = {"completed", "failed", "cancelled"}
ACTIVE_STATES = {
    "queued",
    "validating",
    "waiting_for_capacity",
    "allocating",
    "preparing",
    "loading",
    "running",
    "uploading",
    "cleaning_up",
    "cleanup_pending",
}


class SchedulerWatchdogMixin:
    """Timeout recovery, orphan lease cleanup, and deleted-lease instance reconciliation."""

    def _cleanup_orphan_lease(self, lease: dict) -> bool:
        lease = self._get("leases", lease["id"], lease.get("owner_id")) or lease
        if lease.get("state") == "deleted":
            return False
        if lease.get("state") == "quarantined" and not lease.get("force_stop"):
            return False
        observation = None if lease.get("force_stop") else self._inspect_runtime(lease)
        if observation in {"missing", "stale", "unknown"}:
            return observation == "missing"
        if lease.get("native_ready") and not lease.get("force_stop"):
            from .native import protect_lease
            if protect_lease(self.store, lease, self.clock, check_remote=False):
                return False
        owner_job_id = lease.get("owner_job_id")
        owner_id = lease.get("owner_id")
        instance_id = lease.get("instance_id")
        if not instance_id:
            instance_id = (lease.get("data") or {}).get("instance_id")
        if not (
            isinstance(owner_job_id, str)
            and isinstance(owner_id, str)
            and lease.get("request_id")
            and lease.get("runtime_id")
        ):
            return False
        job = {
            "id": owner_job_id,
            "owner_id": owner_id,
            "provider": lease.get("provider", "mock"),
            "account_id": lease.get("account_id"),
            "instance_id": instance_id,
        }
        handle = self._lease_handle(lease) or self._intent_handle(job, lease)
        if handle is None:
            return False
        if self.clock() < float(lease.get("cleanup_next_epoch") or 0):
            return False
        if instance_id and any(
            other.get("id") != lease.get("id")
            and other.get("instance_id") == instance_id
            and self._lease_blocks(other)
            for other in self._leases_for(owner_id, active_only=True)
        ):
            return False

        token = uuid.uuid4().hex
        durable_instance = self._get("instances", instance_id, owner_id) if instance_id else None
        if durable_instance and durable_instance.get("state") in {"quarantined", "transferring"} and not lease.get("force_stop"):
            return False
        idle_cleanup = bool(
            lease.get("state") == "idle"
            and (
                not lease.get("native_comfy")
                or lease.get("native_draining")
                or lease.get("force_stop")
                or lease.get("native_bootstrap_failed")
                or (durable_instance and durable_instance.get("native_bootstrap_failed"))
            )
            and durable_instance
            and durable_instance.get("lease_id") == lease["id"]
        )
        use_claim = durable_instance is not None and (lease.get("state") != "idle" or idle_cleanup)
        claimed = self.store.claim_orphan_lease(
            owner_id, lease["id"], instance_id, token, self.clock(), idle_cleanup=idle_cleanup,
        ) if use_claim else None
        if use_claim and not claimed:
            return False
        if claimed:
            lease = claimed[0]
            instance = claimed[1]
            job["instance_id"] = instance_id
            if instance and instance.get("version") is not None:
                job["_cleanup_instance_version"] = instance["version"]
            handle = self._lease_handle(lease) or self._intent_handle(job, lease)

        def finish(confirmed, error=None, attempts=0):
            if confirmed:
                if claimed:
                    return bool(self.store.finish_orphan_lease(
                        owner_id, lease["id"], instance_id, token, self.clock(),
                        confirmed=True, error=error, attempts=attempts,
                    ))
                self._update("leases", lease["id"], {"state": "deleted", "last_seen": self.clock()}, owner_id)
                return True
            fresh_lease = self._get("leases", lease["id"], owner_id) or lease
            next_retry, _ = self._cleanup_retry_next_epoch(fresh_lease, attempts)
            delay = max(0.0, next_retry - self.clock())
            message = error or "delete not confirmed"
            finish_error = {"message": message, "retry_after": delay}
            if claimed:
                return bool(self.store.finish_orphan_lease(
                    owner_id, lease["id"], instance_id, token, self.clock(),
                    confirmed=False, error=finish_error, attempts=attempts,
                ))
            self._update("leases", lease["id"], {
                "state": "cleanup_pending", "delete_attempts": attempts,
                "cleanup_next_epoch": next_retry, "last_error": message,
                "last_seen": self.clock(),
            }, owner_id)
            return False

        # A creating row without a provider handle is only a durable placeholder;
        # claim/fence it, release the local reservation, and never call a provider.
        if lease.get("state") == "creating" and not lease.get("handle") and not int(lease.get("create_attempts") or 0):
            try:
                self._mark_instance_deleted(job)
            except SchedulerError:
                finish(False, "instance deletion confirmation failed")
                return False
            return finish(True)

        try:
            provider = self._provider(job)
            if hasattr(provider, "recover_handle"):
                handle = provider.recover_handle(handle)
            confirmed = False
            if not lease.get("force_stop"):
                status = provider.inspect(handle)
                observation = runtime_status(status) if lease.get("provider") == "colab" else None
                confirmed = observation == "missing" if observation else status.get("deleted") is True
            # Without a durable instance claim nothing fences a new lease on the
            # same instance, so recheck right before the remote delete.
            if not confirmed and not claimed and instance_id and any(
                other.get("id") != lease.get("id")
                and other.get("instance_id") == instance_id
                and self._lease_blocks(other)
                for other in self._leases_for(owner_id, active_only=True)
            ):
                finish(False, "same instance acquired by another live lease")
                return False
            if not confirmed and (observation != "unknown" or lease.get("force_stop")):
                response = provider.delete(handle)
                confirmed = response is True or (
                    isinstance(response, dict) and response.get("deleted") is True
                )
            error = None
        except Exception as exc:
            confirmed, error = False, self._safe_error(exc)

        attempts = int(lease.get("delete_attempts") or 0) + 1
        if confirmed:
            try:
                self._mark_instance_deleted(job)
            except SchedulerError as exc:
                confirmed = False
                error = str(exc)
        if confirmed:
            return finish(True, attempts=attempts)
        finish(False, error or "delete not confirmed", attempts)
        if attempts >= 3 and owner_id:
            from app import telegram_notify

            telegram_notify.notify_cleanup_failed(
                self.store,
                owner_id,
                lease_id=lease["id"],
                instance_id=instance_id,
                account_id=lease.get("account_id"),
                attempts=attempts,
                error=error,
                now=self.clock(),
            )
        return False

    def _historical_orphan_evidence(self, instance: dict, lease: dict, leases: list[dict], jobs: list[dict]):
        """Return a lease-owned inspection job only for an unambiguous old handoff."""
        owner = instance.get("owner_id")
        instance_id = self._instance_id(instance)
        if (
            not isinstance(owner, str)
            or not owner
            or not instance_id
            or instance.get("state") not in {"releasing", "cleanup_pending"}
            or lease.get("state") != "deleted"
            or lease.get("owner_id") != owner
            or lease.get("remote_delete_source") not in (None, "")
        ):
            return None
        account = instance.get("account_id")
        if not account or lease.get("account_id") != account:
            return None
        provider = lease.get("provider")
        if provider not in {"mock", "colab"} or instance.get("provider") not in {None, provider}:
            return None
        runtime_id = lease.get("runtime_id")
        if not isinstance(runtime_id, str) or not runtime_id or not lease.get("request_id"):
            return None
        if instance.get("runtime_id") not in (None, runtime_id):
            return None
        current_job_id = instance.get("current_job_id")
        owner_job_id = lease.get("owner_job_id")
        by_id = {row.get("id"): row for row in jobs}
        if not owner_job_id or not by_id.get(owner_job_id):
            return None
        if by_id[owner_job_id].get("owner_id") != owner or by_id[owner_job_id].get("state") not in TERMINAL_STATES:
            return None
        if current_job_id:
            current = by_id.get(current_job_id)
            if not current or current.get("owner_id") != owner or current.get("state") not in TERMINAL_STATES:
                return None
        lease_instance_id = lease.get("instance_id")
        if lease_instance_id in ("",):
            lease_instance_id = None
        lease_rebound = lease_instance_id not in (None, instance_id)
        owner_job = by_id[owner_job_id]
        if lease_rebound:
            successor_instance_id = owner_job.get("instance_id")
            if successor_instance_id != lease_instance_id or successor_instance_id == instance_id:
                return None
        related = [
            row for row in jobs
            if row.get("instance_id") == instance_id or row.get("id") in {current_job_id, owner_job_id}
        ]
        if any(row.get("owner_id") != owner or row.get("state") not in TERMINAL_STATES for row in related):
            return None
        for row in related:
            row_provider = row.get("provider") or (row.get("execution_policy") or {}).get("provider")
            allowed_instances = {None, instance_id}
            if row.get("id") == owner_job_id and lease_rebound:
                allowed_instances.add(lease_instance_id)
            if (
                row.get("account_id") not in (None, account)
                or row.get("instance_id") not in allowed_instances
                or row_provider not in (None, provider)
            ):
                return None
        for row in leases:
            if row.get("id") == lease.get("id") or row.get("state") == "deleted":
                continue
            if row.get("instance_id") == instance_id or row.get("runtime_id") == runtime_id:
                return None
        job = {
            "id": owner_job_id,
            "owner_id": owner,
            "provider": provider,
            "account_id": account,
            "instance_id": instance_id,
        }
        handle = self._intent_handle(job, lease)
        if not handle:
            return None
        if isinstance(lease.get("handle"), dict):
            handle = {**lease["handle"], **handle}
        return job, handle

    def _cleanup_historical_orphan_instance(self, instance: dict, lease: dict, leases: list[dict], jobs: list[dict]) -> bool:
        """Inspect, then persist deletion, for a legacy lease missing provenance."""
        now = self.clock()
        if any(now < float(row.get("cleanup_next_epoch") or 0) for row in (instance, lease)):
            return False
        evidence = self._historical_orphan_evidence(instance, lease, leases, jobs)
        if evidence is None:
            return False
        token = uuid.uuid4().hex
        claimed = self.store.claim_orphan_instance(instance["owner_id"], self._instance_id(instance), lease["id"], token, self.clock())
        if not claimed:
            return False
        claim_instance, claim_lease = claimed
        job, handle = evidence
        try:
            provider = self._provider(job)
            if hasattr(provider, "recover_handle"):
                handle = provider.recover_handle(handle)
            confirmed = runtime_status(provider.inspect(handle)) == "missing"
            error = None if confirmed else "remote deletion not unambiguously confirmed"
        except Exception as exc:
            confirmed, error = False, self._safe_error(exc)
        finished = bool(self.store.finish_orphan_instance(
            instance["owner_id"], self._instance_id(instance), lease["id"], token, self.clock(),
            confirmed=confirmed, error=error, expected_version=claim_instance.get("version"),
            expected_lease_version=claim_lease.get("version"),
        )) and confirmed
        if finished:
            self._close_cu_session(instance["owner_id"], self._instance_id(instance), self.clock(), "orphan_reconciliation")
        return finished

    def _reconcile_deleted_lease_instances(self, leases: list[dict] | None = None) -> int:
        """Converge old releasing rows only with durable remote-delete evidence."""
        if self.instance_service is None:
            return 0
        service_instances = self._service_instances(states=("releasing", "cleanup_pending"))
        if not service_instances:
            return 0
        all_leases = leases if leases is not None else self._leases_for()
        by_lease = {row.get("id"): row for row in all_leases if row.get("id")}
        recovered = 0
        jobs = None
        for instance in service_instances:
            if instance.get("state") not in {"releasing", "cleanup_pending"}:
                continue
            lease_id = instance.get("lease_id")
            lease = by_lease.get(lease_id)
            if not lease or lease.get("state") != "deleted":
                continue
            source = lease.get("remote_delete_source")
            if source in (None, ""):
                if jobs is None:
                    jobs = self._list("jobs")
                if self._cleanup_historical_orphan_instance(instance, lease, all_leases, jobs):
                    recovered += 1
                continue
            if lease.get("remote_deleted") is not True or source not in {
                "cleanup_provider_confirmation", "orphan_cleanup_provider_confirmation",
            }:
                continue
            if lease.get("owner_id") != instance.get("owner_id"):
                continue
            current_instance_id = self._instance_id(instance)
            lease_instance_json = (lease.get("data") or {}).get("instance_id")
            lease_instance_typed = lease.get("instance_id")
            if (
                lease_instance_json not in (None, "")
                and lease_instance_typed not in (None, "")
                and lease_instance_json != lease_instance_typed
            ):
                continue
            lease_instance_id = lease_instance_json or lease_instance_typed
            if not current_instance_id or lease_instance_id != current_instance_id:
                continue
            lease_account = lease.get("account_id")
            instance_account = instance.get("account_id")
            if (not lease_account or not instance_account or lease_account != instance_account):
                continue
            job = {
                "id": lease.get("owner_job_id") or f"orphan-instance:{instance.get('id')}",
                "owner_id": instance.get("owner_id"),
                "provider": lease.get("provider", "mock"),
                "instance_id": instance.get("id"),
                "_cleanup_instance_version": instance.get("version"),
            }
            try:
                self._mark_instance_deleted(job)
            except SchedulerError:
                continue
            recovered += 1
        return recovered

    def _probe_creation_process(self, job, instance, lease, *, observation: str | None = None):
        runtime = instance.get("creation_runtime") or {}
        now = float(self.clock())
        if (runtime.get("lease_id") != lease["id"]
                or not runtime.get("process")
                or now - float(runtime.get("health_checked_at") or 0) < 45):
            return
        try:
            executor = self._executor(job)
            probe = getattr(executor, "creation_process_health", None)
            healthy = probe(self._lease_handle(lease), job, runtime) if probe else None
        except Exception:
            return  # A temporary coordinator connection error is not proof of a dead process.
        if healthy is None:
            return
        should_self_heal_warmup = False
        heal_wf_key = runtime.get("workflow_key")
        heal_wf_ver = runtime.get("workflow_version")
        next_failures = int(runtime.get("health_failures") or 0) + 1
        vm_obs = observation
        if not healthy and next_failures >= 2 and vm_obs is None:
            # Run provider inspection OUTSIDE admission_lock to avoid blocking HTTP admissions
            # and prevent non-reentrant admission_lock self-deadlock / lock-order inversion.
            vm_obs = self._inspect_runtime(lease)
        if not healthy and next_failures >= 2 and vm_obs != "present":
            return
        with self.db_store.admission_lock() as acquired:
            if not acquired:
                return
            latest = self._get("instances", instance["id"], instance["owner_id"])
            if not (latest and latest["state"] == "retained_idle" and latest.get("creation_runtime") == runtime
                    and latest.get("lease_id") == lease["id"]):
                return
            if healthy:
                updated_rt = {**runtime, "health_checked_at": now, "health_failures": 0, "status": "healthy"}
                self._update("instances", instance["id"], {"creation_runtime": updated_rt}, instance["owner_id"])
                return
            failures = int(runtime.get("health_failures") or 0) + 1
            if failures < 2:
                updated_rt = {**runtime, "health_checked_at": now, "health_failures": failures}
                self._update("instances", instance["id"], {"creation_runtime": updated_rt}, instance["owner_id"])
                return
            # 2 consecutive failures and VM confirmed present (L2)
            updated_rt = {
                **runtime,
                "process": None,
                "status": "process_exited",
                "health_failures": failures,
                "health_checked_at": now,
            }
            patch: dict = {"creation_runtime": updated_rt}
            from app.config import settings as app_settings

            from .node_persistence import env_flag

            deadlines = [
                float(v)
                for v in (latest.get("idle_deadline_at"), latest.get("hard_deadline_at"), lease.get("idle_deadline_at"), lease.get("deadline_epoch"))
                if v is not None
            ]
            remaining_s = (min(deadlines) - now) if deadlines else 0.0
            recent_heals = [
                float(t) for t in (latest.get("in_vm_heal_timestamps") or [])
                if isinstance(t, (int, float)) and now - float(t) < 600.0
            ]
            if (
                env_flag("GPU_IN_VM_SELF_HEAL", app_settings.gpu_in_vm_self_heal)
                and runtime.get("warmup_completed") is True
                and remaining_s >= 600.0
                and len(recent_heals) < 2
                and heal_wf_key
                and heal_wf_ver is not None
            ):
                recent_heals.append(now)
                patch["in_vm_heal_timestamps"] = recent_heals
                should_self_heal_warmup = True
            self._update("instances", instance["id"], patch, instance["owner_id"])
            self.store.create("events", instance["owner_id"], {
                "instance_id": instance["id"],
                "job_id": job.get("id"),
                "state": "retained_idle",
                "message": "检测到创作池常驻 ComfyUI 进程连续 2 次探活失败，已标记进程退出并准备同机恢复",
            }, project_id=latest.get("project_id"))

        if should_self_heal_warmup:
            self._trigger_in_place_creation_warmup(instance, heal_wf_key, heal_wf_ver, now)

    def _trigger_in_place_creation_warmup(self, instance: dict, workflow_key: str, workflow_version, now: float) -> None:
        """L2: queue a lightweight in-place warmup reusing the same live creation VM."""
        try:
            from app import creation_api
            from app.admission import admit_job
            from app.job_schemas import JobInput, Policy

            owner = instance["owner_id"]
            wf = next(
                (
                    row
                    for row in creation_api.eligible(self.store, owner)
                    if row["workflow_key"] == workflow_key and str(row["workflow_version"]) == str(workflow_version)
                ),
                None,
            )
            if wf is None:
                return
            project_id = instance.get("project_id") or creation_api.get_project(self.db_store, owner)
            spec = instance.get("runtime_spec_reported") or instance.get("runtime_spec_requested")
            if isinstance(spec, dict) and "key" in spec:
                spec = spec["key"]
            mode = instance.get("retention_mode") if instance.get("retention_mode") in {"smart", "retain"} else "smart"
            retain_min = max(1, int(round(float(instance.get("retain_idle_minutes") or 15))))
            pol = Policy.model_validate({
                "provider": "colab",
                "account_id": instance.get("account_id"),
                "instance_id": instance["id"],
                "compute_spec": spec,
                "lifecycle": mode,
                "retain_idle_minutes": retain_min,
                "max_runtime_minutes": 30,
                "reuse_action": "reuse",
            }).model_dump()
            raw = JobInput(
                project_id=project_id,
                model_id=wf["model_id"],
                workflow_key=wf["workflow_key"],
                workflow_version=wf["workflow_version"],
                params={},
                execution_policy=pol,
            ).model_dump()
            idem = f"in-vm-heal-warmup:{instance['id']}:{int(now)}"
            admit_job(
                self.db_store,
                {"id": owner},
                raw,
                idem,
                creation={
                    "creation_key_id": None,
                    "creation_workflow_id": wf["id"],
                    "creation_existing_only": True,
                    "creation_body_hash": None,
                    "warmup": True,
                },
            )
        except Exception:
            pass

    def watchdog(self) -> dict:
        """Independent process entrypoint for timeout, orphan, and cleanup compensation."""
        with self.db_store.dispatch_lock() as acquired:
            if not acquired:
                return {"acquired": False, "recovered": 0, "reason": "busy"}
            candidates = self._leases_for(active_only=True)
        owners = {lease["owner_id"] for lease in candidates if lease.get("provider") == "colab"}
        guard = CuGuardService(self.store, self.clock)
        for owner in owners:
            ccu.sync_sessions(self.store, owner, self.clock())
            guard.enforce(owner)
        if owners:
            candidates = self._leases_for(active_only=True)
        recovered = 0
        observations = {lease["id"]: self._inspect_runtime(lease) for lease in candidates if not lease.get("force_stop")}
        recovered += sum(result == "missing" for result in observations.values())
        active_jobs = self._list_by_states("jobs", ACTIVE_STATES)
        jobs = [
            job for job in active_jobs
            if job.get("state") in {"cleanup_pending", "cleaning_up"} or self._expired(job)
        ]
        for job in jobs:
            with self.db_store.dispatch_lock() as acquired:
                if not acquired:
                    continue
                current = self._get("jobs", job["id"], job["owner_id"])
                if not current or current.get("state") != job["state"] or (
                    current.get("claim_token") and float(current.get("claim_until") or 0) > self.clock()
                ):
                    continue
                token = uuid.uuid4().hex
                claimed = self._update("jobs", job["id"], {
                    "claim_token": token, "claim_until": self.clock() + self.config.claim_ttl_seconds,
                }, job["owner_id"])
            try:
                recovered += self._advance(claimed).get("state") != claimed.get("state")
            finally:
                self._release_claim(job["id"], token, job["owner_id"])
        all_leases = self._leases_for(active_only=True)
        owner_job_ids = [lease.get("owner_job_id") for lease in all_leases if lease.get("owner_job_id")]
        if owner_job_ids:
            if hasattr(self.store, "list_jobs_by_ids"):
                job_by_id = self.store.list_jobs_by_ids(None, owner_job_ids)
            else:
                job_by_id = {job["id"]: job for job in self._list("jobs")}
        else:
            job_by_id = {}
        for lease in all_leases:
            observation = observations.get(lease["id"])
            if (
                lease.get("state") == "active"
                and lease.get("provider") == "colab"
                and not lease.get("force_stop")
                and not lease.get("cleanup_claim_token")
                and observation != "stale"
            ):
                from .native import keepalive_active_lease

                keepalive_active_lease(self.store, lease, self.clock, provider_factory=self._provider)
            associated = job_by_id.get(lease.get("owner_job_id"))
            if lease.get("state") == "deleted" or observation in {"stale", "missing_pending"} or (
                associated and (associated["owner_id"] != lease["owner_id"] or associated["state"] not in TERMINAL_STATES)
            ):
                continue
            instance_id = lease.get("instance_id")
            instance = self._get("instances", instance_id, lease["owner_id"]) if instance_id else None
            if (
                (lease.get("state") == "quarantined" or (instance and instance.get("state") == "quarantined"))
                and not lease.get("force_stop")
                and not (instance and instance.get("force_stop"))
            ):
                continue
            if instance and instance.get("state") == "transferring" and not lease.get("force_stop"):
                continue
            native_unprotected = False
            if lease.get("native_ready") and not lease.get("force_stop"):
                from .native import protect_lease
                if protect_lease(
                    self.store,
                    lease,
                    self.clock,
                    check_remote=observation is None,
                    provider_factory=self.provider_factory,
                ):
                    continue
                native_unprotected = True
            stopping = (
                native_unprotected
                or bool(lease.get("force_stop") or lease.get("native_draining"))
                or bool(
                    instance
                    and (
                        instance.get("manual_stop_requested")
                        or instance.get("force_stop")
                        or instance.get("native_draining")
                        or instance.get("state") in {"releasing", "cleanup_pending"}
                    )
                )
                or (lease.get("native_comfy") and lease.get("cu_fuse") == "soft_daily_limit")
            )
            if lease.get('state')=='idle' and not stopping and self.clock()<min(
                    float(lease.get('idle_deadline_at') or 0),float(lease.get('deadline_epoch') or 0)):
                from .instances import POOL_CREATION, POOL_EPHEMERAL, POOL_NATIVE, instance_pool_of

                pool = instance_pool_of(instance) if instance else None
                is_creation_pool = pool == POOL_CREATION
                if is_creation_pool and instance.get("creation_runtime"):
                    probe_job = associated or {
                        "id": lease.get("owner_job_id") or f"probe:{instance['id']}",
                        "owner_id": lease["owner_id"],
                        "provider": lease.get("provider", "colab"),
                        "account_id": lease.get("account_id"),
                        "instance_id": instance["id"],
                    }
                    self._probe_creation_process(probe_job, instance, lease, observation=observation)
                standby_keepalive_eligible = bool(
                    lease.get("provider") == "colab"
                    and not lease.get("native_ready")
                    and instance
                    and instance.get("state") == "retained_idle"
                    and (
                        (pool == POOL_CREATION and not lease.get("native_comfy") and instance.get("retention_mode") in {"smart", "retain"})
                        or (pool == POOL_EPHEMERAL and not lease.get("native_comfy") and instance.get("retention_mode") == "retain")
                        or (pool == POOL_NATIVE and instance.get("native_bootstrap_failed") is True and instance.get("retention_mode") == "retain")
                    )
                )
                if standby_keepalive_eligible:
                    from .native import keepalive_lease

                    keepalive_lease(self.store, lease, self.clock, provider_factory=self._provider, smart=True)
                continue
            expired = float(lease.get("deadline_epoch") or 0) < self.clock()
            if (
                associated is None and (expired or stopping)
            ) or (
                associated
                and associated.get("state") in TERMINAL_STATES
                and lease.get("owner_id") == associated.get("owner_id")
            ):
                if self._cleanup_orphan_lease(lease):
                    recovered += 1
        recovered += self._reconcile_deleted_lease_instances()
        from . import recovery

        recovered += recovery.tick(
            self.store,
            clock=self.clock,
            scheduler=self,
            process_started_at=getattr(self, "_started_at", None),
        )
        return {"acquired": True, "recovered": recovered}
