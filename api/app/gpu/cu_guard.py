"""Owner-wide daily CU fuse and per-account balance floor."""

import math
import os
import time
import uuid

from sqlalchemy import select

from app.common import APIError
from app.db import utcnow

from . import ccu
from .repository import (
    GpuAccountRow,
    GpuEventRow,
    GpuInstanceRow,
    GpuInstanceUsageSegmentRow,
    GpuJobRow,
    GpuLeaseRow,
)


class CuGuardService:
    def __init__(self, repo, clock=time.time):
        self.repo = repo
        self.clock = clock

    def _limits(self, session, owner, *, initialize=False):
        initial = os.getenv("GPU_DAILY_UNITS")
        try:
            daily = float(initial) if initial else None
        except ValueError:
            daily = None
        if daily is not None and (not math.isfinite(daily) or daily < 0):
            daily = None
        defaults = {"daily_limit_cu": daily, "min_balance_cu": None}
        key = f"cu-guard:{owner}"
        saved = ccu.setting(session, key)
        if initialize and not saved:
            ccu.setting(session, key, defaults, owner)
        return {**defaults, **saved}

    def state_in(self, session, owner, *, initialize=False, today=None, fail_closed_on_sample_error: bool = False):
        limits = self._limits(session, owner, initialize=initialize)
        if today is None:
            today = ccu.daily_spend(session, owner, ccu.local_date(self.clock()))
        daily, floor = limits["daily_limit_cu"], limits["min_balance_cu"]
        soft = daily is not None and today >= daily
        hard = []
        if floor is not None:
            active_accounts = set()
            for inst in session.scalars(
                select(GpuInstanceRow).where(GpuInstanceRow.owner_id == owner, GpuInstanceRow.state != "deleted")
            ):
                aid = inst.account_id or (inst.data or {}).get("account_id")
                if aid:
                    active_accounts.add(aid)
            for ls in session.scalars(
                select(GpuLeaseRow).where(GpuLeaseRow.owner_id == owner, GpuLeaseRow.state != "deleted")
            ):
                aid = ls.account_id or (ls.data or {}).get("account_id")
                if aid:
                    active_accounts.add(aid)
            for seg in session.scalars(
                select(GpuInstanceUsageSegmentRow).where(
                    GpuInstanceUsageSegmentRow.owner_id == owner,
                    GpuInstanceUsageSegmentRow.state.in_(("open", "reconciling")),
                )
            ):
                if seg.account_id:
                    active_accounts.add(seg.account_id)
            for account in session.scalars(select(GpuAccountRow).where(GpuAccountRow.owner_id == owner)):
                data = account.data or {}
                key = data.get("account_id") or account.id
                if not ccu.enabled_account(data):
                    is_colab = data.get("provider", "colab") in {"colab", "google"} and not data.get("is_mock")
                    if not (is_colab and key in active_accounts):
                        continue
                snapshot = ccu.setting(session, ccu._snapshot_key(owner, key))
                balance = snapshot.get("balance")
                if (fail_closed_on_sample_error or not snapshot.get("error")) and balance is not None and balance <= floor:
                    hard.append({"account_id": key, "alias": data.get("alias"), "balance": balance})
        daily_message = f"今日已扣 {today:g} CU，达到今日总上限 {daily:g} CU" if soft else None
        hard_message = "触及余额底线 " + f"{floor:g} CU 的账号：" + "、".join(
            f"{row['alias'] or row['account_id']}（{row['balance']:g} CU）" for row in hard
        ) if hard else None
        return {**limits, "today_actual_cu": today, "soft_tripped": soft, "hard_tripped_accounts": hard,
                "status": "hard_balance_floor" if hard else "soft_daily_limit" if soft else "ok",
                "message": hard_message or daily_message, "daily_message": daily_message}

    def state(self, owner):
        with self.repo.store.db_lock(), self.repo.store.Session() as session:
            if ccu.setting(session, f"cu-guard:{owner}"):
                return self.state_in(session, owner)
        with ccu.transaction(self.repo, owner) as session:
            return self.state_in(session, owner, initialize=True)

    def save(self, owner, values):
        limits = {}
        for key in ("daily_limit_cu", "min_balance_cu"):
            value = values.get(key)
            if value is not None:
                try:
                    valid = type(value) in (int, float) and 0 <= value and math.isfinite(value)
                except OverflowError:
                    valid = False
                if not valid:
                    raise APIError("INVALID_CU_LIMIT", "CU 限额必须为空或非负数", 422)
            limits[key] = value
        with ccu.transaction(self.repo, owner) as session:
            ccu.setting(session, f"cu-guard:{owner}", limits, owner)
        return self.state(owner)

    @staticmethod
    def filter_accounts(state, available):
        blocked = {row["account_id"]: row for row in state["hard_tripped_accounts"]}
        eligible = [key for key in available if key not in blocked]
        if available and not eligible:
            accounts = "、".join(f"{blocked[key]['alias'] or key}（{blocked[key]['balance']:g} CU）" for key in available)
            raise APIError("CU_BALANCE_FLOOR_REACHED", f"{accounts} 已触及余额底线 {state['min_balance_cu']:g} CU", 409)
        if state["soft_tripped"]:
            raise APIError("CU_DAILY_LIMIT_REACHED", state["daily_message"], 409)
        return eligible

    def check(self, owner, provider, account=None, instance=None):
        if provider != "colab":
            return
        if instance and not account:
            account = (self.repo.get("instances", instance, owner) or {}).get("account_id")
        available = [account] if account else [
            key for key, row in ccu.accounts(self.repo, owner).items()
            if row.get("allow_allocation") is not False and row.get("connected", True)
        ]
        state = self.state(owner)
        floor = state.get("min_balance_cu")
        if floor is not None and available:
            # If an account was already at/below the hard balance floor and only had a transient
            # sample_failed error, fail closed on new paid allocation unless a fresh sample confirms recovery.
            with self.repo.store.db_lock(), self.repo.store.Session() as session:
                stale_below = []
                for key in available:
                    snap = ccu.setting(session, ccu._snapshot_key(owner, key))
                    bal = snap.get("balance")
                    if snap.get("error") and bal is not None and bal <= floor:
                        stale_below.append((key, bal))
            if stale_below:
                hard_ids = {row["account_id"] for row in state["hard_tripped_accounts"]}
                extra = [
                    {"account_id": key, "alias": None, "balance": bal}
                    for key, bal in stale_below
                    if key not in hard_ids
                ]
                if extra:
                    state = {**state, "hard_tripped_accounts": [*state["hard_tripped_accounts"], *extra]}
        self.filter_accounts(state, available)

    def enforce(self, owner):
        """Persist stop/drain requests for the normal, confirmation-based cleanup paths."""
        with self.repo.store.dispatch_lock() as dispatch, self.repo.store.admission_lock() as acquired:
            if not dispatch or not acquired:
                return False
            with ccu.transaction(self.repo, owner) as session:
                state = self.state_in(session, owner, initialize=True)
                hard = {row["account_id"] for row in state["hard_tripped_accounts"]}
                candidates = session.scalars(select(GpuInstanceRow).where(
                    GpuInstanceRow.owner_id == owner, GpuInstanceRow.state != "deleted",
                ).with_for_update())
                for instance in candidates:
                    data = instance.data or {}
                    if data.get("provider") != "colab":
                        continue
                    inst_account_id = instance.account_id or data.get("account_id")
                    forced = inst_account_id in hard
                    if not forced and not state["soft_tripped"]:
                        self._restore(session, instance)
                        continue
                    if not forced and data.get("fuse_forced_stop"):
                        continue
                    lease = session.get(GpuLeaseRow, data.get("lease_id"), with_for_update=True) if data.get("lease_id") else None
                    job_id = data.get("current_job_id") or ((lease.data or {}).get("owner_job_id") if lease else None)
                    level = "hard_balance_floor" if forced else "soft_daily_limit"
                    native = bool(
                        data.get("instance_pool") == "native"
                        or data.get("native_comfy")
                        or (lease and (lease.data or {}).get("native_comfy"))
                    )
                    if data.get("cu_fuse") == level:
                        inst_patch = {}
                        if native and not data.get("native_draining"):
                            inst_patch["native_draining"] = True
                        if not forced and instance.state == "retained_idle":
                            instance.state = "releasing"
                            inst_patch.update(state="releasing", cu_fuse_previous_state="retained_idle")
                        if forced and instance.state not in {"releasing", "cleanup_pending", "quarantined"}:
                            instance.state = "releasing"
                            inst_patch.update(
                                state="releasing",
                                manual_stop_requested=True,
                                fuse_forced_stop=True,
                                force_stop=True,
                            )
                        if inst_patch:
                            instance.data = {**data, **inst_patch, "state": instance.state}
                            instance.version += 1
                        if lease and lease.owner_id == owner and lease.state != "deleted":
                            ldata = lease.data or {}
                            lease_patch = {}
                            if ldata.get("cu_fuse") != level:
                                lease_patch["cu_fuse"] = level
                            if native and not ldata.get("native_draining"):
                                lease_patch["native_draining"] = True
                            if forced and not ldata.get("force_stop"):
                                lease_patch["force_stop"] = True
                            if lease_patch:
                                lease.data = {**ldata, **lease_patch}
                                lease.version += 1
                        continue
                    patch = {"cu_fuse": level}
                    if forced:
                        patch.update(manual_stop_requested=True, fuse_forced_stop=True, force_stop=True)
                        if job_id:
                            job = session.get(GpuJobRow, job_id, with_for_update=True)
                            if job and job.owner_id == owner and job.state not in {"completed", "failed", "cancelled", "expired"}:
                                job.data = {**job.data, "cancel_requested": True, "error_code": "CU_BALANCE_FLOOR_REACHED",
                                            "error_message": "触发余额底线强制熔断，立即停止实例"}
                                if job.state not in {"cleaning_up", "cleanup_pending"}:
                                    if lease and lease.data.get("handle") and job.state in {"preparing", "loading", "running", "uploading"}:
                                        job.data = {**job.data, "claim_token": None, "claim_until": 0}
                                    job.state = "cleaning_up"
                                job.data = {**job.data, "state": job.state, "cleanup_final_state": "cancelled",
                                            "requested_outcome": "cancelled", **ccu.execution_finished(job.data, self.clock())}
                                job.version += 1
                    if native:
                        patch["native_draining"] = True
                    if forced or instance.state == "retained_idle":
                        patch["cu_fuse_previous_state"] = data.get("cu_fuse_previous_state", instance.state)
                        if instance.state not in {"cleanup_pending", "quarantined"}:
                            instance.state = "releasing"
                    instance.data = {**data, **patch, "state": instance.state}
                    instance.version += 1
                    if lease and lease.owner_id == owner and lease.state != "deleted":
                        lease.data = {**lease.data, "cu_fuse": level,
                                      **({"native_draining": True} if native else {}),
                                      **({"force_stop": True} if forced else {})}
                        lease.version += 1
                    message = "触发余额底线强制熔断，立即停止实例止损" if forced else (
                        "触发今日 CU 上限温和熔断，原生 ComfyUI 队列排空后释放实例" if native
                        else "触发今日 CU 上限温和熔断，当前任务完成后释放实例"
                    )
                    session.add(GpuEventRow(id=uuid.uuid4().hex, owner_id=owner, project_id=instance.project_id,
                                            state=instance.state, data={"instance_id": instance.id, "job_id": job_id,
                                            "message": message, "reason": level}, created_at=utcnow(), updated_at=utcnow()))
        return True

    @staticmethod
    def _restore(session, instance):
        data = instance.data or {}
        if data.get("cu_fuse") != "soft_daily_limit" or instance.state in {"cleanup_pending", "quarantined"}:
            return
        lease_id = data.get("lease_id")
        lease = session.get(GpuLeaseRow, lease_id, with_for_update=True) if lease_id else None
        if instance.state == "releasing" and (not lease or lease.state != "idle" or lease.data.get("cleanup_claim_token")):
            return
        if instance.state == "releasing":
            instance.state = data.get("cu_fuse_previous_state") or "retained_idle"
        retained = {key: value for key, value in data.items() if key not in {"cu_fuse", "cu_fuse_previous_state"}}
        if not data.get("manual_stop_requested") and data.get("native_draining"):
            retained["native_draining"] = False
        instance.data = {**retained, "state": instance.state}
        instance.version += 1
        if lease and lease.state != "deleted":
            lease.data = {key: value for key, value in lease.data.items() if key != "cu_fuse"}
            if not data.get("manual_stop_requested") and data.get("native_draining"):
                lease.data = {**lease.data, "native_draining": False}
            lease.version += 1
