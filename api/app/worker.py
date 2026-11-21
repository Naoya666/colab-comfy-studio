"""Persistent queue and independent watchdog processes. No browser dependency."""

import argparse
import logging
import os
import signal
import time
from concurrent.futures import ThreadPoolExecutor

from .config import settings
from .db import Store
from .gpu.scheduler import Scheduler

log = logging.getLogger("studio.worker")
running = True


def stop(*_):
    global running
    running = False


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
    last_ccu_tick = 0.0
    restart_pool = ThreadPoolExecutor(max_workers=1) if args.mode == "worker" and not args.once else None
    restart_future = None
    while running:
        try:
            scheduler = Scheduler(store)
            if args.mode == "model-worker":
                from .gpu.model_library import transfer_tick
                from .gpu.repository import Repository
                result = transfer_tick(Repository(store), stopping=lambda: not running)
            else:
                result = scheduler.watchdog() if args.mode == "watchdog" else scheduler.tick()
            if args.mode == "worker" and (restart_future is None or restart_future.done()):
                from .gpu.native_restart import tick as restart_tick
                from .gpu.repository import Repository

                if restart_future is not None:
                    completed, restart_future = restart_future, None
                    completed.result()
                if restart_pool:
                    restart_future = restart_pool.submit(restart_tick, Repository(store))
                else:
                    restart_tick(Repository(store))
            if args.mode == "worker" and os.getenv("STUDIO_ATELIER_ENABLED", "0") == "1":
                from .atelier_routes import auto_export_tick

                auto_export_tick(store)
            if args.mode == "worker" and time.time() - last_ccu_tick >= 60:
                # Refresh account balances before applying the runtime fuses.
                last_ccu_tick = time.time()
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
            if result.get("advanced"):
                log.info("job=%s state=%s", result.get("job_id"), result.get("state"))
        except Exception as exc:
            # Exception bodies and provider HTTP responses can contain credentials.
            log.error("worker tick failed: %s", type(exc).__name__)
        if args.once:
            break
        time.sleep(max(0.1, args.interval))
    if restart_pool:
        restart_pool.shutdown()


if __name__ == "__main__":
    main()
