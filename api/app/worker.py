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
        for owner_id in owner_ids:
            tick_ccu(repo, owner_id)
            CuGuardService(repo).enforce(owner_id)
    except Exception as exc:  # noqa: BLE001 - never stop the queue for accounting
        log.warning("ccu tick failed: %s", type(exc).__name__)


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
    restart_pool = ThreadPoolExecutor(max_workers=1) if args.mode == "worker" and not args.once else None
    ccu_pool = ThreadPoolExecutor(max_workers=1) if args.mode == "worker" and not args.once else None
    restart_future = None
    ccu_future = None
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
                    completed.result()
                if restart_pool:
                    restart_future = restart_pool.submit(restart_tick, repo)
                else:
                    restart_tick(repo)
            if args.mode == "worker" and os.getenv("STUDIO_ATELIER_ENABLED", "0") == "1":
                from .atelier_routes import auto_export_tick

                auto_export_tick(store)
            if args.mode == "worker" and time.time() - last_ccu_tick >= 60:
                if ccu_future is not None and ccu_future.done():
                    ccu_future.result()
                    ccu_future = None
                if ccu_future is None:
                    last_ccu_tick = time.time()
                    if fast_control and ccu_pool is not None:
                        ccu_future = ccu_pool.submit(_run_ccu_tick, store)
                    else:
                        _run_ccu_tick(store)
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


if __name__ == "__main__":
    main()
