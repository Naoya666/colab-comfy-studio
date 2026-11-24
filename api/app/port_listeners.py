"""Dynamic in-process Uvicorn listeners for optional dedicated Creation/Native local ports."""

from __future__ import annotations

import asyncio
import os
import socket
from contextlib import nullcontext
from dataclasses import dataclass

import uvicorn

from .common import APIError


def _default_bind_host(allow_lan: bool) -> str:
    if allow_lan or os.getenv("STUDIO_CONTAINER_BIND_ALL") == "1":
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


class _DedicatedPortServer(uvicorn.Server):
    def capture_signals(self):
        return nullcontext()

    def install_signal_handlers(self):
        # Older Uvicorn releases use this hook.
        pass


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
            prev_host = current.host if same_port_rebind else None
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
                if same_port_rebind and prev_host:
                    # Best-effort restore of the previously working listener on same_port_rebind failure.
                    rollback_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    try:
                        rollback_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                        rollback_sock.bind((prev_host, port))
                        rollback_sock.listen(128)
                        rollback_sock.setblocking(False)
                        rb_server = _DedicatedPortServer(
                            uvicorn.Config(
                                _DedicatedPortASGI(self.studio_app, kind),
                                host=prev_host,
                                port=port,
                                access_log=False,
                                log_level="error",
                                lifespan="off",
                            )
                        )
                        rb_task = asyncio.create_task(
                            rb_server.serve(sockets=[rollback_sock]), name=f"studio-{kind}-port-{port}"
                        )
                        self._listeners[kind] = _RunningListener(
                            host=prev_host, port=port, sock=rollback_sock, server=rb_server, task=rb_task
                        )
                    except OSError:
                        rollback_sock.close()
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
            server = _DedicatedPortServer(cfg)
            task = asyncio.create_task(server.serve(sockets=[sock]), name=f"studio-{kind}-port-{port}")

            def _on_done(finished: asyncio.Task, _kind: str = kind, _sock: socket.socket = sock) -> None:
                try:
                    exc_done = finished.exception() if not finished.cancelled() else None
                except asyncio.CancelledError:
                    exc_done = None
                if exc_done is not None:
                    self._errors[_kind] = f"独立端口监听异常退出：{type(exc_done).__name__}"
                try:
                    _sock.close()
                except OSError:
                    pass

            task.add_done_callback(_on_done)
            self._listeners[kind] = _RunningListener(host=target_host, port=port, sock=sock, server=server, task=task)
            self._errors.pop(kind, None)
            # Wait cleanly for uvicorn to enter started state (bounded to 2s).
            loop = asyncio.get_running_loop()
            deadline = loop.time() + 2.0
            while not server.started and not task.done() and loop.time() < deadline:
                await asyncio.sleep(0.01)
            if task.done():
                self._listeners.pop(kind, None)
                msg = self._errors.get(kind) or f"独立端口 {port} 启动失败"
                self._errors[kind] = msg
                if strict:
                    raise APIError("PORT_IN_USE", msg, 409)
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
        except TimeoutError:
            item.server.force_exit = True
            item.task.cancel()
            await asyncio.gather(item.task, return_exceptions=True)
        except asyncio.CancelledError:
            item.server.force_exit = True
            item.task.cancel()
            await asyncio.gather(item.task, return_exceptions=True)
            raise
        finally:
            try:
                item.sock.close()
            except OSError:
                pass

    async def shutdown(self) -> None:
        async with self._lock:
            for kind in list(self._listeners):
                await self._stop_unlocked(kind)
