"""Dynamic in-process Uvicorn listeners for optional dedicated Creation/Native local ports."""

from __future__ import annotations

import asyncio
import os
import socket
from dataclasses import dataclass
from pathlib import Path

import uvicorn

from .common import APIError


def _default_bind_host(allow_lan: bool) -> str:
    if allow_lan or Path("/.dockerenv").exists() or os.getenv("STUDIO_CONTAINER") == "1":
        return "0.0.0.0"
    return "127.0.0.1"


def probe_port_available(port: int, *, allow_lan: bool = False) -> None:
    host = _default_bind_host(allow_lan)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
    except OSError as exc:
        raise APIError("PORT_IN_USE", f"独立端口 {port} 已被占用或无法绑定：{exc.strerror or exc}", 409) from exc
    finally:
        sock.close()


@dataclass
class _RunningListener:
    host: str
    port: int
    sock: socket.socket
    server: uvicorn.Server
    task: asyncio.Task


class _DedicatedPortASGI:
    def __init__(self, studio_app, kind: str):
        self.studio_app = studio_app
        self.kind = kind

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            while True:
                msg = await receive()
                if msg["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif msg["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        if self.kind == "creation":
            from .creation_api import dispatch_creation_scope

            await dispatch_creation_scope(self.studio_app, scope, receive, send, dedicated_port=True)
        else:
            from .native_proxy import dispatch_native_scope

            await dispatch_native_scope(self.studio_app, scope, receive, send, dedicated_port=True)


class PortListenerManager:
    def __init__(self, studio_app):
        self.studio_app = studio_app
        self._listeners: dict[str, _RunningListener] = {}
        self._errors: dict[str, str] = {}
        self._lock = asyncio.Lock()

    def status(self, kind: str) -> dict:
        item = self._listeners.get(kind)
        active = bool(item and not item.task.done())
        return {
            "active": active,
            "host": item.host if active else None,
            "port": item.port if active else None,
            "error": self._errors.get(kind),
        }

    async def apply(self, kind: str, port: int | None, *, allow_lan: bool = False, strict: bool = True) -> dict:
        async with self._lock:
            current = self._listeners.get(kind)
            target_host = _default_bind_host(allow_lan)
            if port is None:
                if current:
                    await self._stop_unlocked(kind)
                self._errors.pop(kind, None)
                return self.status(kind)

            if current and not current.task.done() and current.port == port and current.host == target_host:
                self._errors.pop(kind, None)
                return self.status(kind)

            same_port_rebind = bool(current and not current.task.done() and current.port == port)
            if same_port_rebind:
                await self._stop_unlocked(kind)

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind((target_host, port))
                sock.listen(128)
                sock.setblocking(False)
            except OSError as exc:
                sock.close()
                msg = f"独立端口 {port} 已被占用或无法绑定：{exc.strerror or exc}"
                self._errors[kind] = msg
                if strict:
                    raise APIError("PORT_IN_USE", msg, 409) from exc
                return self.status(kind)

            if current and not same_port_rebind:
                await self._stop_unlocked(kind)

            asgi_app = _DedicatedPortASGI(self.studio_app, kind)
            cfg = uvicorn.Config(
                asgi_app,
                host=target_host,
                port=port,
                access_log=False,
                log_level="error",
                lifespan="off",
            )
            server = uvicorn.Server(cfg)
            # Prevent uvicorn from installing process-wide SIGINT/SIGTERM handlers in sub-listeners.
            server.install_signal_handlers = lambda: None
            task = asyncio.create_task(server.serve(sockets=[sock]), name=f"studio-{kind}-port-{port}")
            self._listeners[kind] = _RunningListener(host=target_host, port=port, sock=sock, server=server, task=task)
            self._errors.pop(kind, None)
            # Yield briefly so uvicorn initializes the server loop cleanly.
            await asyncio.sleep(0.01)
            return self.status(kind)

    async def start_from_store(self, store, owner: str = "owner") -> None:
        from . import creation_api, native_api

        c_cfg = creation_api.config(store, owner)
        if c_cfg.get("listen_port"):
            await self.apply("creation", c_cfg["listen_port"], allow_lan=bool(c_cfg.get("allow_lan")), strict=False)
        n_pol = native_api.policy(store, owner)
        if n_pol.get("listen_port"):
            await self.apply("native", n_pol["listen_port"], allow_lan=bool(n_pol.get("allow_lan")), strict=False)

    async def _stop_unlocked(self, kind: str) -> None:
        item = self._listeners.pop(kind, None)
        if not item:
            return
        item.server.should_exit = True
        try:
            await asyncio.wait_for(item.task, timeout=2.0)
        except (TimeoutError, asyncio.CancelledError):
            item.task.cancel()
            await asyncio.gather(item.task, return_exceptions=True)
        finally:
            try:
                item.sock.close()
            except OSError:
                pass

    async def shutdown(self) -> None:
        async with self._lock:
            for kind in list(self._listeners):
                await self._stop_unlocked(kind)
