"""Persistent queue and independent watchdog processes. No browser dependency."""

import argparse
import logging
import os
import signal
import time
from concurrent.futures import ThreadPoolExecutor

from .config import settings
from .db import Store
from .gpu.node_persistence import env_flag
from .gpu.scheduler import Scheduler

log = logging.getLogger("studio.worker")
running = True


def stop(*_):
    global running
    running = False


def _run_ccu_tick(store: Store) -> None:
    try:
        from .gpu.ccu import tick_ccu
        from .gpu.cu_guard import CuGuardService
        from .gpu.repository import Repository

        repo = Repository(store)
        owner_ids = sorted(
            {str(account["owner_id"]) for account in repo.list("accounts") if account.get("owner_id")}
        )
    except Exception as exc:  # noqa: BLE001 - never stop the queue for accounting
        log.warning("ccu tick owner scan failed: %s", type(exc).__name__)
        return
    for owner_id in owner_ids:
        try:
            tick_ccu(repo, owner_id)
            CuGuardService(repo).enforce(owner_id)
        except Exception as exc:  # noqa: BLE001 - isolate owner failures
            log.warning("ccu tick failed for owner=%s: %s", owner_id, type(exc).__name__)


def _run_telegram_dispatch(store: Store) -> None:
    try:
        from .telegram_notify import dispatch_pending

        dispatch_pending(store)
    except Exception as exc:  # noqa: BLE001 - never stop the queue for notifications
        log.warning("telegram dispatch failed: %s", type(exc).__name__)


def _run_telegram_periodic(store: Store) -> None:
    try:
        from .telegram_notify import periodic_tick

        periodic_tick(store)
    except Exception as exc:  # noqa: BLE001 - never stop the queue for notifications
        log.warning("telegram periodic tick failed: %s", type(exc).__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["worker", "watchdog", "model-worker"], default="worker", nargs="?")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=float, default=2)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    # Colab proxy credentials appear in HTTP query strings; never log URLs.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    store = Store(settings.database_url)
    from .gpu.repository import Repository

    repo = Repository(store)
    scheduler = Scheduler(repo)
    last_ccu_tick = 0.0
    last_tg_dispatch = 0.0
    last_tg_periodic = 0.0
    restart_pool = ThreadPoolExecutor(max_workers=1) if args.mode == "worker" and not args.once else None
    ccu_pool = ThreadPoolExecutor(max_workers=1) if args.mode == "worker" and not args.once else None
    tg_pool = ThreadPoolExecutor(max_workers=1) if args.mode == "worker" and not args.once else None
    restart_future = None
    ccu_future = None
    tg_future = None
    while running:
        advanced = False
        fast_control = env_flag("NATIVE_FAST_CONTROL_PLANE", True)
        try:
            if args.mode == "model-worker":
                from .gpu.model_library import transfer_tick
                result = transfer_tick(repo, stopping=lambda: not running)
            else:
                result = scheduler.watchdog() if args.mode == "watchdog" else scheduler.tick()
            if args.mode == "worker" and (restart_future is None or restart_future.done()):
                from .gpu.native_restart import tick as restart_tick

                if restart_future is not None:
                    completed, restart_future = restart_future, None
                    try:
                        completed.result()
                    except Exception as exc:  # noqa: BLE001 - isolate restart worker failures
                        log.warning("native restart tick failed: %s", type(exc).__name__)
                if restart_pool:
                    restart_future = restart_pool.submit(restart_tick, repo)
                else:
                    try:
                        restart_tick(repo)
                    except Exception as exc:  # noqa: BLE001
                        log.warning("native restart tick failed: %s", type(exc).__name__)
            if args.mode == "worker" and os.getenv("STUDIO_ATELIER_ENABLED", "0") == "1":
                from .atelier_routes import auto_export_tick

                auto_export_tick(store)
            if args.mode == "worker" and time.time() - last_ccu_tick >= 60:
                if ccu_future is not None and ccu_future.done():
                    completed_ccu, ccu_future = ccu_future, None
                    try:
                        completed_ccu.result()
                    except Exception as exc:  # noqa: BLE001 - never let ccu_future wedge
                        log.warning("ccu background tick failed: %s", type(exc).__name__)
                if ccu_future is None:
                    last_ccu_tick = time.time()
                    if fast_control and ccu_pool is not None:
                        ccu_future = ccu_pool.submit(_run_ccu_tick, store)
                    else:
                        _run_ccu_tick(store)
            if args.mode == "worker":
                now_ts = time.time()
                if tg_future is not None and tg_future.done():
                    completed_tg, tg_future = tg_future, None
                    try:
                        completed_tg.result()
                    except Exception as exc:  # noqa: BLE001
                        log.warning("telegram background tick failed: %s", type(exc).__name__)
                if tg_future is None and (now_ts - last_tg_dispatch >= 2.0 or now_ts - last_tg_periodic >= 60.0):
                    run_periodic = now_ts - last_tg_periodic >= 60.0
                    last_tg_dispatch = now_ts
                    if run_periodic:
                        last_tg_periodic = now_ts

                    def _tg_step(do_periodic=run_periodic):
                        if do_periodic:
                            _run_telegram_periodic(store)
                        _run_telegram_dispatch(store)

                    if tg_pool is not None:
                        tg_future = tg_pool.submit(_tg_step)
                    else:
                        _tg_step()
            advanced = bool(result.get("advanced"))
            if advanced:
                log.info("job=%s state=%s", result.get("job_id"), result.get("state"))
        except Exception as exc:
            # Exception bodies and provider HTTP responses can contain credentials.
            log.error("worker tick failed: %s", type(exc).__name__)
        if args.once:
            break
        if not (fast_control and advanced):
            time.sleep(max(0.1, args.interval))
    if restart_pool:
        restart_pool.shutdown()
    if ccu_pool:
        ccu_pool.shutdown()
    if tg_pool:
        tg_pool.shutdown()


if __name__ == "__main__":
    main()
