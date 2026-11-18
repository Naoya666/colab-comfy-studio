"""Full host reverse proxy adapted from FireTable/comfy-colab-kit apps/gate/proxy.py.

MIT design attribution: docs/licenses/comfy-colab-kit.txt. FastAPI/httpx adapter;
no frontend rewriting, raw encoded paths, streamed bodies, binary WebSockets.
"""
from __future__ import annotations

import asyncio
import secrets
import time
import uuid
from urllib.parse import urlsplit

import httpx
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse, StreamingResponse
from starlette.websockets import WebSocket, WebSocketDisconnect
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidHandshake

from .common import APIError
from .config import settings
from .gpu import native
from .security import read_session, sign_session

HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer",
       "trailers", "transfer-encoding", "upgrade", "host"}
PRIVATE = {"authorization", "cookie", "x-api-key", "x-forwarded-for", "x-forwarded-host", "x-forwarded-proto"}


def clean_headers(headers, *, request=False):
    connection = set(headers.get("connection", "").lower().replace(" ", "").split(","))
    return [(k, v) for k, v in headers.multi_items()
            if k.lower() not in HOP | connection
            and not (request and (k.lower() in PRIVATE or k.lower().startswith(("cf-", "sec-websocket", "x-studio-restart-"))))
            and not (not request and k.lower() == "set-cookie")]


def identity(headers, cookies):
    auth = headers.get("authorization", "")
    if auth.startswith("Bearer ") and settings.studio_api_token and secrets.compare_digest(auth[7:], settings.studio_api_token):
        return "owner"
    value = read_session(cookies.get("studio_native", ""))
    if value and value.get("purpose") == "native-session" and value.get("aud") == settings.native_comfy_url.rstrip("/"):
        return value.get("uid")
    return None


def check_origin(headers):
    origin = headers.get("origin")
    if origin and origin.rstrip("/") != settings.native_comfy_url.rstrip("/"):
        raise APIError("CSRF_ORIGIN", "跨站请求被拒绝", 403)


class NativeHostMiddleware:
    def __init__(self, app, studio_app):
        self.app = app
        self.studio_app = studio_app
        self._http_client = None

    def http_client(self):
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(None, connect=15), follow_redirects=False,
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=100),
            )
        return self._http_client

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            try:
                return await self.app(scope, receive, send)
            finally:
                if self._http_client is not None:
                    await self._http_client.aclose()
                    self._http_client = None
        if scope["type"] not in {"http", "websocket"} or not settings.native_comfy_url:
            return await self.app(scope, receive, send)
        headers = httpx.Headers(scope["headers"])
        if headers.get("host", "").lower() != urlsplit(settings.native_comfy_url).netloc.lower():
            return await self.app(scope, receive, send)
        if scope["type"] == "websocket":
            return await self.websocket(scope, receive, send)
        request = Request(scope, receive)
        try:
            if scope["path"] == "/_studio/session":
                if request.method != "POST" or headers.get("origin", "").rstrip("/") != settings.site_url.rstrip("/"):
                    raise APIError("CSRF_ORIGIN", "请从 Studio 打开原生 ComfyUI", 403)
                raw = await request.body()
                if len(raw) > 8192:
                    raise APIError("PAYLOAD_TOO_LARGE", "登录请求过大", 413)
                from urllib.parse import parse_qs
                ticket = read_session(parse_qs(raw.decode()).get("ticket", [""])[0])
                if not ticket or not ticket.get("uid") or ticket.get("purpose") != "native-handoff" or ticket.get("aud") != settings.native_comfy_url.rstrip("/"):
                    raise APIError("UNAUTHENTICATED", "登录凭证已过期，请从 Studio 重新打开", 401)
                now = int(time.time())
                token = sign_session({"uid": ticket["uid"], "aud": ticket["aud"], "purpose": "native-session",
                                      "iat": now, "exp": now + 43200})
                response = RedirectResponse("/", status_code=303)
                response.set_cookie("studio_native", token, max_age=43200, secure=True, httponly=True, samesite="strict")
                response.headers["Cache-Control"] = "no-store"
                return await response(scope, receive, send)
            owner = identity(headers, request.cookies)
            if not owner:
                raise APIError("UNAUTHENTICATED", "请从 Studio 的原生 ComfyUI 入口登录，API 使用 Bearer Token", 401)
            internal_path = scope["path"].removeprefix("/api").rstrip("/")
            if internal_path == "/studio_native/index":
                raise APIError("NOT_FOUND", "接口不存在", 404)
            if request.method not in {"GET", "HEAD", "OPTIONS"}:
                check_origin(headers)
            lease = await asyncio.to_thread(native.active_lease, self.studio_app.state.repo, owner)
            if not lease or not lease.get("native_ready"):
                raise APIError("NATIVE_NOT_READY", "请在 Studio 启动原生 ComfyUI", 503)
            if request.method not in {"GET", "HEAD", "OPTIONS"} and scope["path"] not in {"/interrupt", "/api/interrupt", "/queue", "/api/queue"}:
                instance = await asyncio.to_thread(self.studio_app.state.repo.get, "instances", lease.get("instance_id"), owner)
                if (instance or {}).get("manual_stop_requested") or time.time() >= float(lease.get("deadline_epoch") or float("inf")):
                    raise APIError("NATIVE_DRAINING", "会话已到运行时限或正在停止；等待已有任务及同步结束后重启", 409)
            return await self.http(request, lease, scope, receive, send)
        except APIError as exc:
            return await JSONResponse({"error": exc.code, "message": exc.message}, status_code=exc.status)(scope, receive, send)

    @staticmethod
    def target(scope):
        base, auth = native.backend()
        raw = scope.get("raw_path", scope["path"].encode()).decode("ascii")
        query = scope.get("query_string", b"").decode("ascii")
        return base + raw + ("?" + query if query else ""), base, auth

    async def http(self, request, lease, scope, receive, send):
        url, base, auth = self.target(scope)
        headers = clean_headers(httpx.Headers(scope["headers"]), request=True)
        headers += list(auth.items()) + [("Origin", base)]
        # Do not forward a browser Referer containing the public host to Comfy's origin checker.
        headers = [(k, v) for k, v in headers if k.lower() != "referer"]
        path = scope["path"].removeprefix("/api").rstrip("/")
        mutating = request.method not in {"GET", "HEAD", "OPTIONS"} or (
            request.method == "GET" and path in {"/manager/reboot", "/v2/manager/reboot", "/manager/queue/start"}
        )
        repo = self.studio_app.state.repo
        stop_heartbeat = asyncio.Event()
        write_id = uuid.uuid4().hex if mutating else None

        async def activity():
            while not stop_heartbeat.is_set():
                try:
                    await asyncio.wait_for(stop_heartbeat.wait(), 10)
                except TimeoutError:
                    await asyncio.to_thread(native.update_write, repo, lease, write_id)
        if mutating:
            await asyncio.to_thread(
                native.begin_write, repo, lease,
                allow_stopping=scope["path"] in {"/interrupt", "/api/interrupt", "/queue", "/api/queue"},
                write_id=write_id,
            )
        heartbeat = asyncio.create_task(activity()) if mutating else None
        downstream_started = False
        try:
            client = self.http_client()
            retryable = (not mutating and request.method in {"GET", "HEAD"}
                         and request.headers.get("content-length", "0") == "0"
                         and "transfer-encoding" not in request.headers)
            for attempt in range(3 if retryable else 1):
                req = client.build_request(request.method, httpx.URL(url), headers=headers,
                                           content=None if retryable else request.stream())
                # The pool must never replay upstream cookies on another browser request.
                req.headers.pop("cookie", None)
                try:
                    response = await client.send(req, stream=True)
                except httpx.TransportError:
                    if not retryable or attempt == 2:
                        raise
                    await asyncio.sleep(0.2 * (attempt + 1))
                    continue
                if retryable and response.status_code in {502, 503, 504} and attempt < 2:
                    await response.aclose()
                    await asyncio.sleep(0.2 * (attempt + 1))
                    continue
                break
            try:
                response_headers = clean_headers(response.headers)
                response_headers = [(k, v.replace(base, settings.native_comfy_url.rstrip("/")) if k.lower() == "location" else v)
                                    for k, v in response_headers]
                response_headers += [("X-Content-Type-Options", "nosniff"), ("Cache-Control", "no-store")]
                downstream = StreamingResponse(response.aiter_raw(), status_code=response.status_code)
                downstream.raw_headers = [(k.lower().encode("latin1"), v.encode("latin1")) for k, v in response_headers]
                downstream_started = True
                await downstream(scope, receive, send)
            finally:
                await response.aclose()
        except httpx.HTTPError:
            # Never include backend URLs/service credentials in browser-visible errors.
            if not downstream_started:
                await JSONResponse({"error": "NATIVE_UPSTREAM_UNAVAILABLE"}, status_code=502)(scope, receive, send)
        finally:
            stop_heartbeat.set()
            if heartbeat:
                try:
                    await heartbeat
                finally:
                    await asyncio.to_thread(native.update_write, repo, lease, write_id, finished=True)

    async def websocket(self, scope, receive, send):
        ws = WebSocket(scope, receive, send)
        try:
            owner = identity(ws.headers, ws.cookies)
            check_origin(ws.headers)
            lease = await asyncio.to_thread(native.active_lease, self.studio_app.state.repo, owner) if owner else None
            if not lease or not lease.get("native_ready"):
                return await ws.close(code=1008)
            url, base, auth = self.target(scope)
            headers = clean_headers(httpx.Headers(scope["headers"]), request=True)
            headers = [(k, v) for k, v in headers if k.lower() not in {"origin", "referer"}]
            headers += list(auth.items())
            protocols = scope.get("subprotocols") or None
            async with connect(url.replace("https://", "wss://", 1), origin=base, additional_headers=headers,
                               subprotocols=protocols, max_size=None, open_timeout=15) as upstream:
                await ws.accept(subprotocol=upstream.subprotocol)

                async def browser_to_upstream():
                    while True:
                        message = await ws.receive()
                        if message["type"] == "websocket.disconnect":
                            return
                        await upstream.send(message.get("bytes") if message.get("bytes") is not None else message["text"])

                async def upstream_to_browser():
                    async for message in upstream:
                        if isinstance(message, bytes):
                            await ws.send_bytes(message)
                        else:
                            await ws.send_text(message)
                tasks = [asyncio.create_task(browser_to_upstream()), asyncio.create_task(upstream_to_browser())]
                try:
                    await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                finally:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
        except (APIError, OSError, TimeoutError, ConnectionClosed, InvalidHandshake, WebSocketDisconnect):
            pass
        finally:
            if ws.application_state.name != "DISCONNECTED":
                await ws.close(code=1011)
