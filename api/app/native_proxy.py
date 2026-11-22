"""Full host and subpath/port reverse proxy adapted from FireTable/comfy-colab-kit apps/gate/proxy.py.

MIT design attribution: docs/licenses/comfy-colab-kit.txt. FastAPI/httpx adapter;
no frontend rewriting, raw encoded paths, streamed bodies, binary WebSockets.
"""
from __future__ import annotations

import asyncio
import json
import secrets
import time
import uuid
from urllib.parse import quote, urlencode

import httpx
from starlette.requests import ClientDisconnect, Request
from starlette.responses import JSONResponse, RedirectResponse, Response, StreamingResponse
from starlette.websockets import WebSocket, WebSocketDisconnect
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidHandshake

from . import native_api, native_model_api
from .api_access import (
    check_weak_origin,
    cors_headers_for,
    is_native_blocked_path,
    is_trusted_local_client,
    match_gateway_route,
    peek_path_code,
)
from .common import APIError
from .config import settings
from .gpu import native
from .security import read_session, sign_session

HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer",
       "trailers", "transfer-encoding", "upgrade", "host"}
PRIVATE = {"authorization", "cookie", "x-api-key", "x-forwarded-for", "x-forwarded-host", "x-forwarded-proto"}


def _header_bytes(name: str, value: str) -> bytes:
    try:
        return value.encode("latin1")
    except UnicodeEncodeError:
        if name.lower() == "content-disposition":
            import re
            match = re.search(r'filename="([^"]+)"', value)
            if match:
                raw_name = match.group(1)
                ascii_fallback = raw_name.encode("ascii", "replace").decode("ascii").replace('"', "_")
                utf8_star = quote(raw_name, safe="")
                rewritten = value[:match.start()] + f'filename="{ascii_fallback}"; filename*=UTF-8\'\'{utf8_star}' + value[match.end():]
                try:
                    return rewritten.encode("latin1")
                except UnicodeEncodeError:
                    pass
        return value.encode("latin1", "replace")


def clean_headers(headers, *, request=False):
    connection = set(headers.get("connection", "").lower().replace(" ", "").split(","))
    return [(k, v) for k, v in headers.multi_items()
            if k.lower() not in HOP | connection
            and not (request and (k.lower() in PRIVATE or k.lower().startswith(("cf-", "sec-websocket", "x-studio-restart-", "x-studio-model-"))))
            and not (not request and k.lower() == "set-cookie")]


def identity(headers, cookies):
    auth = headers.get("authorization", "")
    if auth.startswith("Bearer ") and settings.studio_api_token and secrets.compare_digest(auth[7:], settings.studio_api_token):
        return "owner"
    value = read_session(cookies.get("studio_native", ""))
    if settings.native_comfy_url and value and value.get("purpose") == "native-session" and value.get("aud") == settings.native_comfy_url.rstrip("/"):
        return value.get("uid")
    return None


def check_origin(headers):
    origin = headers.get("origin")
    if origin and origin.rstrip("/") != settings.native_comfy_url.rstrip("/"):
        raise APIError("CSRF_ORIGIN", "跨站请求被拒绝", 403)


class NativeGateway:
    def __init__(self, studio_app):
        self.studio_app = studio_app
        self._http_client = None

    def http_client(self):
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(None, connect=15), follow_redirects=False,
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=100),
            )
        return self._http_client

    async def close(self):
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    def _resolve_weak_access(self, repo, scope, headers, route_token: str | None) -> tuple[str, dict, dict[str, str]]:
        owner = "owner"
        pol = native_api.policy(repo.store, owner)
        if route_token:
            if pol["auth_mode"] not in {"url_path", "local_open"}:
                raise APIError("UNAUTHENTICATED", "当前未启用原生 ComfyUI 路径短码模式", 401)
            expected = peek_path_code(repo.store, f"native-api:{owner}", owner)
            if not expected or not secrets.compare_digest(route_token, expected):
                raise APIError("UNAUTHENTICATED", "原生 ComfyUI 路径短码无效或已重置", 401)
            cors = check_weak_origin(headers, via_token=True)
            return owner, pol, cors

        if pol["auth_mode"] != "local_open":
            raise APIError("UNAUTHENTICATED", "请从 Studio 的原生 ComfyUI 入口登录，API 使用 Bearer Token", 401)
        if not is_trusted_local_client(scope, headers, allow_lan=bool(pol.get("allow_lan"))):
            raise APIError(
                "UNAUTHENTICATED",
                "当前仅允许本机免密访问；局域网/容器互通请开启「允许局域网/容器私网访问」或改用 URL 路径短 Token 模式",
                401,
            )
        cors = check_weak_origin(headers, via_token=False, allowed_origins=pol.get("allowed_origins"))
        return owner, pol, cors

    def _preflight_cors(self, repo, scope, headers, route_token: str | None) -> dict[str, str]:
        req_headers = headers.get("access-control-request-headers", "").lower()
        if "authorization" in req_headers:
            return cors_headers_for(headers.get("origin"), wildcard_fallback=True)
        _, _, cors = self._resolve_weak_access(repo, scope, headers, route_token)
        return cors

    async def handle(self, scope, receive, send, *, entry: str, route_token: str | None):
        if scope["type"] == "websocket":
            return await self.websocket(scope, receive, send, route_token=route_token)
        headers = httpx.Headers(scope["headers"])
        request = Request(scope, receive)
        cors_headers: dict[str, str] = {}
        try:
            if scope["path"] == "/_studio/session":
                if entry != "host" or not settings.native_comfy_url:
                    raise APIError("NOT_FOUND", "接口不存在", 404)
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
                is_https = settings.native_comfy_url.startswith("https://")
                response.set_cookie("studio_native", token, max_age=43200, secure=is_https, httponly=True, samesite="strict")
                response.headers["Cache-Control"] = "no-store"
                return await response(scope, receive, send)

            repo = self.studio_app.state.repo
            if request.method == "OPTIONS" and headers.get("access-control-request-method"):
                cors_headers = self._preflight_cors(repo, scope, headers, route_token)
                return await Response(
                    status_code=204 if cors_headers else 403,
                    headers={"Cache-Control": "no-store", **cors_headers},
                )(scope, receive, send)

            auth = headers.get("authorization", "")
            key = None
            weak_policy = None
            if auth.startswith("Bearer nca_") and not (
                settings.studio_api_token and secrets.compare_digest(auth[7:], settings.studio_api_token)
            ):
                key = await asyncio.to_thread(native_api.authenticate, repo.store, auth[7:])
                cors_headers = cors_headers_for(headers.get("origin"), wildcard_fallback=True)
            owner = key["owner_id"] if key else identity(headers, request.cookies)
            if not owner:
                owner, weak_policy, cors_headers = await asyncio.to_thread(
                    self._resolve_weak_access, repo, scope, headers, route_token
                )

            internal_path = scope["path"].removeprefix("/api").rstrip("/")
            if internal_path == "/studio_native/index" or internal_path.startswith("/studio_native/models"):
                raise APIError("NOT_FOUND", "接口不存在", 404)

            if weak_policy is not None:
                if scope["path"] == "/_studio/api/status":
                    lease = await asyncio.to_thread(native.active_lease, repo, owner)
                    return await JSONResponse(
                        {"ready": bool(lease and lease.get("native_ready") and not native.restart_active(lease))},
                        headers={"Cache-Control": "no-store", **cors_headers},
                    )(scope, receive, send)
                if is_native_blocked_path(scope["path"], scope.get("raw_path")):
                    raise APIError("NATIVE_API_FORBIDDEN", "弱鉴权模式禁止访问 ComfyUI-Manager 与节点管理接口", 403)
            elif request.method not in {"GET", "HEAD", "OPTIONS"}:
                if key:
                    cors_headers = cors_headers_for(headers.get("origin"), wildcard_fallback=True)
                else:
                    check_origin(headers)

            path, file, payload = None, None, None
            if key:
                path = native_api.scope_path(scope["path"], request.method, scope["query_string"])
                await asyncio.to_thread(native_api.touch, repo.store, key)
            lease = await asyncio.to_thread(native.active_lease, repo, owner)
            if (path or scope["path"]) == "/_studio/api/status":
                return await JSONResponse(
                    {"ready": bool(lease and lease.get("native_ready") and not native.restart_active(lease))},
                    headers={"Cache-Control": "no-store", **cors_headers},
                )(scope, receive, send)

            submit = request.method == "POST" and scope["path"] in {"/prompt", "/api/prompt"}
            if key and submit:
                _, graph = await self.read_prompt(request)
                payload = json.dumps(native_api.submission(key, graph), allow_nan=False).encode()
            elif weak_policy is not None and submit and weak_policy.get("weak_workflow_mode") == "approved":
                _, graph = await self.read_prompt(request)
                payload = json.dumps(native_api.weak_submission(weak_policy, graph), allow_nan=False).encode()

            if lease and submit and not lease.get("native_ready"):
                await self.check_stopping(repo, lease, owner)
            if not lease or not lease.get("native_ready"):
                is_admin_token = bool(
                    settings.studio_api_token and auth.startswith("Bearer ")
                    and secrets.compare_digest(auth[7:], settings.studio_api_token)
                )
                if not submit or (not key and not is_admin_token and weak_policy is None):
                    raise APIError("NATIVE_NOT_READY", "请在 Studio 启动原生 ComfyUI", 503)
                if not key:
                    values = weak_policy or await asyncio.to_thread(native_api.policy, repo.store, owner)
                    if not values["auto_start"]:
                        raise APIError("NATIVE_NOT_READY", "请在 Studio 启动原生 ComfyUI", 503)
                    if payload is None:
                        raw_bytes, graph = await self.read_prompt(request)
                        if weak_policy is not None:
                            native_api.weak_submission(weak_policy, graph)
                            payload = raw_bytes
                        else:
                            native_api.submission({"mode": "trusted"}, graph)
                            payload = raw_bytes
                lease = await self.wait_for_start(
                    request, repo, owner, key,
                    api_key_label="weak-api" if weak_policy is not None else None,
                )
            if request.method not in {"GET", "HEAD", "OPTIONS"} and scope["path"] not in {"/interrupt", "/api/interrupt", "/queue", "/api/queue"}:
                await self.check_stopping(repo, lease, owner)
            if key:
                if path.startswith("/history/"):
                    prompt_id = path.rsplit("/", 1)[1]
                    record = await asyncio.to_thread(
                        repo.store.get, native_api.PROMPTS, native_api.prompt_record_id(key, lease, prompt_id), owner,
                    )
                    if not record:
                        raise APIError("NATIVE_API_NOT_FOUND", "任务不存在或已结束会话", 404)
                if path == "/view":
                    file = native_api.view_tuple(scope["query_string"].decode("ascii"))
                    record = await asyncio.to_thread(
                        repo.store.get, native_api.PROMPTS, native_api.view_id(key, lease, file), owner,
                    )
                    if not record:
                        raise APIError("NATIVE_API_NOT_FOUND", "图片不存在或不属于当前接入 Key", 404)
            local = False
            model_listing = internal_path == "/object_info" or internal_path.startswith(("/models/", "/object_info/"))
            if submit or (request.method == "GET" and model_listing):
                storage = await asyncio.to_thread(native.lease_storage, repo, lease)
                local = storage["model_storage_mode"] == "local"
            if local and submit and payload is None:
                payload, _ = await self.read_prompt(request, limit=16 * 1024**2)
            return await self.http(
                request, lease, scope, receive, send,
                key=key, path=path, file=file, payload=payload, local=local, cors_headers=cors_headers,
            )
        except APIError as exc:
            err_headers = {
                "Cache-Control": "no-store",
                **({"Retry-After": "5"} if exc.code == "NATIVE_STARTING" else {}),
                **cors_headers,
            }
            return await JSONResponse(
                {"error": exc.code, "message": exc.message}, status_code=exc.status, headers=err_headers,
            )(scope, receive, send)

    @staticmethod
    async def check_stopping(repo, lease, owner):
        instance = await asyncio.to_thread(repo.get, "instances", lease.get("instance_id"), owner)
        if (lease.get("native_draining") or (instance or {}).get("manual_stop_requested")
                or time.time() >= float(lease.get("deadline_epoch") or float("inf"))):
            raise APIError("NATIVE_DRAINING", "会话已到运行时限或正在停止；等待已有任务及同步结束后重启", 409)

    @staticmethod
    async def read_prompt(request, limit=native_api.MAX_WORKFLOW_BYTES):
        body = bytearray()
        try:
            async for chunk in request.stream():
                if len(body) + len(chunk) > limit:
                    raise APIError("PAYLOAD_TOO_LARGE", f"工作流请求不能超过 {limit // 1024**2} MiB", 413)
                body.extend(chunk)
            data = json.loads(body)
            json.dumps(data, allow_nan=False)
        except ClientDisconnect as exc:
            raise APIError("NATIVE_REQUEST_CLOSED", "连接已关闭，工作流未提交", 499) from exc
        except (ValueError, UnicodeError, RecursionError) as exc:
            raise APIError("NATIVE_API_INPUT", "请求必须为有效的 JSON 工作流", 400) from exc
        return bytes(body), data

    async def wait_for_start(self, request, repo, owner, key, *, api_key_label=None):
        deadline = time.monotonic() + native_api.START_WAIT_SECONDS
        admission_deadline = min(deadline, time.monotonic() + 2)
        while True:
            try:
                if api_key_label:
                    job = await asyncio.to_thread(
                        native_api.request_start, repo, owner, key, api_key_label=api_key_label
                    )
                else:
                    job = await asyncio.to_thread(native_api.request_start, repo, owner, key)
                break
            except APIError as exc:
                if exc.code != "SCHEDULER_BUSY" or time.monotonic() >= admission_deadline:
                    raise
                await asyncio.sleep(0.1)
        if not job:
            raise APIError("NATIVE_NOT_READY", "请在 Studio 启动原生 ComfyUI", 503)
        while True:
            if await request.is_disconnected():
                raise APIError("NATIVE_REQUEST_CLOSED", "连接已关闭，工作流未提交", 499)
            current = await asyncio.to_thread(repo.get, "jobs", job["id"], owner) or job
            if current.get("state") in native.TERMINAL - {"completed"}:
                raise APIError("NATIVE_START_FAILED", "原生 ComfyUI 启动失败，请在 Studio 查看启动记录", 503)
            lease = await asyncio.to_thread(native.active_lease, repo, owner)
            if lease:
                if lease.get("native_ready"):
                    if key:
                        await asyncio.to_thread(native_api.live_key, repo.store, key)
                    return lease
                await self.check_stopping(repo, lease, owner)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise APIError("NATIVE_STARTING", "原生 ComfyUI 正在启动，工作流尚未提交，请就绪后重试", 503)
            await asyncio.sleep(min(remaining, 1))

    @staticmethod
    def target(scope):
        base, auth = native.backend()
        raw = scope.get("raw_path", scope["path"].encode()).decode("ascii")
        query = scope.get("query_string", b"").decode("ascii")
        return base + raw + ("?" + query if query else ""), base, auth

    async def http(
        self, request, lease, scope, receive, send,
        *, key=None, path=None, file=None, payload=None, local=False, cors_headers=None,
    ):
        cors_headers = cors_headers or {}
        url, base, auth = self.target(scope)
        if key:
            url = base + quote(path, safe="/")
            if file:
                url += "?" + urlencode(dict(zip(("filename", "subfolder", "type"), file)))
        headers = clean_headers(httpx.Headers(scope["headers"]), request=True)
        # Strip client Origin and Referer before adding upstream Origin so ComfyUI's aiohttp
        # origin guard never receives duplicate/comma-joined Origin values on browser POSTs.
        headers = [(k, v) for k, v in headers if k.lower() not in {"origin", "referer"}]
        headers += list(auth.items()) + [("Origin", base)]
        path = path or scope["path"].removeprefix("/api").rstrip("/")
        mutating = request.method not in {"GET", "HEAD", "OPTIONS"} or (
            request.method == "GET" and path in {"/manager/reboot", "/v2/manager/reboot", "/manager/queue/start"}
        )
        repo = self.studio_app.state.repo
        if payload is not None:
            headers = [(k, v) for k, v in headers if k.lower() not in {"content-length", "content-type", "content-encoding"}]
            headers.append(("Content-Type", "application/json"))
        if key:
            headers = [(k, v) for k, v in headers if k.lower() != "accept-encoding"]
            headers.append(("Accept-Encoding", "identity"))
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
        reservation = None
        try:
            client = self.http_client()
            if key and mutating:
                reservation = await self.reserve_prompt(client, base, auth, repo, key, lease)
            if local:
                channel = native_model_api.headers(base, auth, lease)
                if request.method == "POST":
                    task = await native_model_api.start(client, base, channel, repo, lease, json.loads(payload), [], visual=False)
                    while True:
                        if await request.is_disconnected():
                            # The shared download can still be used by other requests. This prompt has never been sent.
                            return
                        state = await native_model_api.rpc(client, "GET", base, channel, "/prepare/" + task["id"])
                        if state["state"] != "preparing":
                            break
                        await asyncio.sleep(1)
                    if state["state"] != "ready":
                        from .gpu.native_models import model_problem

                        failures = [r for r in state["models"] if r["state"] == "failed"]
                        message = state.get("message") or "、".join(
                            f"{r['category']}/{r['name']} · {r['error']}" for r in failures
                        )[:3000]
                        problem = {**model_problem(message), "models": state["models"]}
                        return await JSONResponse(problem, status_code=422, headers=cors_headers)(scope, receive, send)
                    if await request.is_disconnected():
                        return
                    fresh = await asyncio.to_thread(native.active_lease, repo, lease["owner_id"])
                    if (not fresh or fresh["id"] != lease["id"] or not fresh.get("native_ready")
                            or native.restart_active(fresh)):
                        raise APIError("NATIVE_SESSION_CHANGED", "模型准备期间原生会话已改变，请重新提交", 409)
                    await self.check_stopping(repo, fresh, lease["owner_id"])
                    if key:
                        await asyncio.to_thread(native_api.live_key, repo.store, key)
                    headers.extend([("X-Studio-Model-Channel", channel["X-Studio-Model-Channel"]),
                                    ("X-Studio-Model-Prepared", task["id"])])
                else:
                    context = await asyncio.to_thread(native_model_api.runtime_context, repo, lease, credentials=False)
                    await native_model_api.rpc(client, "POST", base, channel, "/catalog", body=context)
            retryable = (not mutating and request.method in {"GET", "HEAD"}
                         and request.headers.get("content-length", "0") == "0"
                         and "transfer-encoding" not in request.headers)
            for attempt in range(3 if retryable else 1):
                req = client.build_request(request.method, httpx.URL(url), headers=headers,
                                           content=payload if payload is not None else None if retryable else request.stream(),
                                           timeout=45 if key else httpx.USE_CLIENT_DEFAULT)
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
                if key and response.is_redirect:
                    raise APIError("NATIVE_UPSTREAM_UNAVAILABLE", "原生 ComfyUI 返回了意外的重定向", 502)
                if key and response.is_success and (reservation or path.startswith("/history/")):
                    data = await self.read_upstream_json(response)
                    if reservation:
                        prompt_id = data.get("prompt_id")
                        if not isinstance(prompt_id, str) or not native_api.PROMPT_ID.fullmatch(prompt_id):
                            raise APIError("NATIVE_UPSTREAM_UNAVAILABLE", "ComfyUI 未返回有效任务 ID", 502)
                        await asyncio.to_thread(native_api.finish_submission, repo.store, key, lease, reservation, prompt_id)
                        reservation = None
                    else:
                        prompt_id = path.rsplit("/", 1)[1]
                        item = data.get(prompt_id)
                        await asyncio.to_thread(native_api.register_outputs, repo.store, key, lease, item)
                        data = {prompt_id: item} if item is not None else {}
                    return await JSONResponse(
                        data, status_code=response.status_code, headers={"Cache-Control": "no-store", **cors_headers}
                    )(scope, receive, send)
                response_headers = clean_headers(response.headers)
                public_origin = (settings.native_comfy_url or settings.site_url).rstrip("/")
                response_headers = [(k, v.replace(base, public_origin) if k.lower() == "location" else v)
                                    for k, v in response_headers]
                response_headers += [("X-Content-Type-Options", "nosniff"), ("Cache-Control", "no-store")]
                for ck, cv in cors_headers.items():
                    response_headers.append((ck, cv))
                downstream = StreamingResponse(response.aiter_raw(), status_code=response.status_code)
                downstream.raw_headers = [(k.lower().encode("latin1"), _header_bytes(k, v)) for k, v in response_headers]
                downstream_started = True
                await downstream(scope, receive, send)
            finally:
                await response.aclose()
        except httpx.HTTPError:
            # Never include backend URLs/service credentials in browser-visible errors.
            if not downstream_started:
                await JSONResponse(
                    {"error": "NATIVE_UPSTREAM_UNAVAILABLE"}, status_code=502, headers=cors_headers
                )(scope, receive, send)
        finally:
            stop_heartbeat.set()
            if heartbeat:
                try:
                    await heartbeat
                finally:
                    await asyncio.to_thread(native.update_write, repo, lease, write_id, finished=True)
            if reservation:
                await asyncio.to_thread(native_api.finish_submission, repo.store, key, lease, reservation)

    @staticmethod
    async def read_upstream_json(response):
        try:
            data = json.loads(await response.aread())
            if not isinstance(data, dict):
                raise ValueError
            return data
        except (ValueError, RecursionError) as exc:
            raise APIError("NATIVE_UPSTREAM_UNAVAILABLE", "原生 ComfyUI 返回了无效响应", 502) from exc

    async def reserve_prompt(self, client, base, auth, repo, key, lease):
        reservation = await asyncio.to_thread(native_api.reserve, repo.store, key, lease)
        if reservation:
            return reservation
        current = await asyncio.to_thread(native_api.live_key, repo.store, key)
        request = client.build_request("GET", base + "/queue", headers=auth, timeout=8)
        request.headers.pop("cookie", None)
        response = await client.send(request)
        if not response.is_success:
            raise APIError("NATIVE_UPSTREAM_UNAVAILABLE", "暂时无法检查 ComfyUI 队列", 502)
        data = await self.read_upstream_json(response)
        queues = [data.get(name) for name in ("queue_running", "queue_pending")]
        if any(not isinstance(queue, list) for queue in queues):
            raise APIError("NATIVE_UPSTREAM_UNAVAILABLE", "暂时无法检查 ComfyUI 队列", 502)
        queued = {entry[1] for queue in queues for entry in queue
                  if isinstance(entry, list) and len(entry) > 1 and isinstance(entry[1], str)}
        observed = {ident: entry["prompt"] for ident, entry in current.get("active", {}).items()
                    if entry.get("prompt") and entry["prompt"] not in queued}
        reservation = await asyncio.to_thread(native_api.reserve, repo.store, key, lease, observed)
        if not reservation:
            raise APIError("NATIVE_API_CONCURRENCY", "已达到此 Key 的并行任务上限，请等待任务完成", 429)
        return reservation

    async def websocket(self, scope, receive, send, *, route_token: str | None = None):
        ws = WebSocket(scope, receive, send)
        try:
            repo = self.studio_app.state.repo
            owner = identity(ws.headers, ws.cookies)
            if owner:
                check_origin(ws.headers)
            else:
                if is_native_blocked_path(scope["path"], scope.get("raw_path")):
                    return await ws.close(code=1008)
                owner, _, _ = await asyncio.to_thread(
                    self._resolve_weak_access, repo, scope, ws.headers, route_token
                )
            lease = await asyncio.to_thread(native.active_lease, repo, owner) if owner else None
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
            if ws.application_state.name == "CONNECTING":
                await ws.close(code=1008)
        finally:
            if ws.application_state.name != "DISCONNECTED":
                await ws.close(code=1011)


def get_native_gateway(studio_app) -> NativeGateway:
    gateway = getattr(studio_app.state, "_native_gateway", None)
    if gateway is None:
        gateway = NativeGateway(studio_app)
        studio_app.state._native_gateway = gateway
    return gateway


async def dispatch_native_scope(studio_app, scope, receive, send, *, dedicated_port: bool = False) -> bool:
    matched = match_gateway_route(
        scope,
        kind="native",
        dedicated_host_url=settings.native_comfy_url,
        dedicated_port=dedicated_port,
    )
    if matched is None:
        return False
    gateway = get_native_gateway(studio_app)
    await gateway.handle(
        matched["scope"], receive, send, entry=matched["entry"], route_token=matched["route_token"]
    )
    return True


class NativeHostMiddleware:
    def __init__(self, app, studio_app):
        self.app = app
        self.studio_app = studio_app

    @property
    def _http_client(self):
        gateway = getattr(self.studio_app.state, "_native_gateway", None)
        return gateway._http_client if gateway is not None else None

    @_http_client.setter
    def _http_client(self, value):
        gateway = getattr(self.studio_app.state, "_native_gateway", None)
        if gateway is not None:
            gateway._http_client = value

    def http_client(self):
        return get_native_gateway(self.studio_app).http_client()

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            try:
                return await self.app(scope, receive, send)
            finally:
                gateway = getattr(self.studio_app.state, "_native_gateway", None)
                if gateway is not None:
                    await gateway.close()
        if scope["type"] in {"http", "websocket"}:
            handled = await dispatch_native_scope(self.studio_app, scope, receive, send, dedicated_port=False)
            if handled:
                return
        return await self.app(scope, receive, send)
