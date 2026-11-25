"""Recoverable executor for the official Google Colab MCP endpoint.

The adapter has no in-memory execution registry.  Colab's execution directory
(status.json, meta.json and output files) is the source of truth, so a process
restart or a lost JSON-RPC response cannot cause a second worker launch.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import socket
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlsplit

import httpx

from app.security import decrypt_secret

from .google import GoogleService
from .node_persistence import env_flag
from .runtime_worker import (
    COMFY_SERVICE_ROOT,
    MAX_OUTPUT_BYTES,
    _instance_service_name,
    _service_api_url,
    creation_optimization_flags,
)

log = logging.getLogger("studio.mcp")
MCP_ENDPOINT = "https://colaboratory.googleapis.com/mcp"
_RUNTIME = re.compile(r"^runtimes/[a-zA-Z0-9][a-zA-Z0-9._~-]{0,127}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


_NETWORK_ATTEMPTS = 3
_INLINE_ARTIFACT_MAX_BYTES = 64 * 1024
_INLINE_ARTIFACT_TOTAL_MAX_BYTES = 256 * 1024
_DEFAULT_CHUNK_BYTES = 1024 * 1024
_TUNNEL_CHUNK_BYTES = 4 * 1024 * 1024
_MAX_CHUNK_BYTES = 8 * 1024 * 1024


def _prefer_ipv4_direct() -> bool:
    """Use IPv4 directly for Colab endpoints when no outbound proxy is configured.

    Avoids dual-stack read/open stalls observed on IPv4-only Docker bridges.
    """
    if os.getenv("COLAB_TUNNEL_FORCE_IPV4", "1").strip().lower() in {"0", "false", "off", "no"}:
        return False
    return not any(
        os.getenv(k) for k in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy")
    )


def _make_colab_http_client(timeout: float) -> httpx.Client:
    http_timeout = httpx.Timeout(timeout, connect=min(8.0, timeout))
    if _prefer_ipv4_direct():
        try:
            return httpx.Client(
                timeout=http_timeout,
                follow_redirects=False,
                transport=httpx.HTTPTransport(local_address="0.0.0.0"),
            )
        except OSError:
            pass
    return httpx.Client(timeout=http_timeout, follow_redirects=False)


def _connect_ipv4_socket(ws_url: str, timeout: float) -> socket.socket | None:
    if not _prefer_ipv4_direct():
        return None
    parsed = urlsplit(ws_url)
    host = parsed.hostname
    if not host:
        return None
    port = parsed.port or (443 if parsed.scheme == "wss" else 80)
    try:
        infos = socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_STREAM)
    except OSError:
        return None
    last_err: OSError | None = None
    for af, socktype, proto, _, sa in infos:
        sock = socket.socket(af, socktype, proto)
        try:
            sock.settimeout(timeout)
            sock.connect(sa)
            return sock
        except OSError as exc:
            last_err = exc
            sock.close()
    if last_err is not None:
        raise last_err
    return None


class MCPError(RuntimeError):
    """Safe Colab MCP adapter failure; upstream bodies are never exposed."""


class MCPTransient(MCPError):
    """Colab answered "unavailable"; every call made here is idempotent, so it is retried."""


_TRANSIENT = re.compile(r"unavailable|\b503\b|deadline|timed? ?out|temporar", re.I)
_SECRET = re.compile(r"(ya29\.[\w.-]+|Bearer\s+\S+|[A-Za-z0-9_-]{40,})")


def _error_detail(result: dict) -> str:
    """Secret-redacted, truncated reason from a failed tool result; goes to the backend log only."""
    parts = [result.get("executionError")]
    parts += [c.get("text") for c in result.get("content", []) if isinstance(c, dict)]
    text = " ".join(p if isinstance(p, str) else json.dumps(p) for p in parts if p)
    return _SECRET.sub("***", text).strip()[:300]


def _uuid4(value: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError, TypeError) as exc:
        raise MCPError("execution_id必须是UUID4") from exc
    if parsed.version != 4:
        raise MCPError("execution_id必须是UUID4")
    return str(parsed)


def _b64_json(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return base64.b64encode(raw).decode("ascii")


def build_bootstrap_code(execution_id: str, payload: dict, worker_source: str | None = None) -> str:
    """Return the one trusted code string accepted by ``execute_code``.

    Caller data is transported as base64 JSON, never interpolated as Python.
    The worker source is the checked-in runtime_worker.py, not user supplied
    code.  The O_EXCL lock and marker make repeated starts harmless.
    """
    execution_id = _uuid4(execution_id)
    if not isinstance(payload, dict):
        raise MCPError("worker payload必须是对象")
    if worker_source is None:
        worker_source = Path(__file__).with_name("runtime_worker.py").read_text(encoding="utf-8")
    if not isinstance(worker_source, str) or not worker_source:
        raise MCPError("trusted worker source is unavailable")
    is_native = bool((payload.get("snapshot") or {}).get("native_comfy"))
    lean_bootstrap = env_flag("CREATION_FAST_CONTROL_PLANE", True)
    extra_modules: dict[str, str] = {}
    if is_native or not lean_bootstrap:
        extra_modules["node_persistence.py"] = base64.b64encode(
            Path(__file__).with_name("node_persistence.py").read_bytes()
        ).decode("ascii")
        extra_modules["native_runtime.py"] = base64.b64encode(
            Path(__file__).with_name("native_runtime.py").read_bytes()
        ).decode("ascii")
    if (payload.get("native_secrets") or {}).get("model_storage_mode") == "local":
        for name in ("native_models.py", "model_dependencies.py", "model_sources.py"):
            extra_modules[name] = base64.b64encode(
                Path(__file__).with_name(name).read_bytes()
            ).decode("ascii")
    source_b64 = base64.b64encode(worker_source.encode("utf-8")).decode("ascii")
    payload_b64 = _b64_json(payload)
    execution_literal = repr(execution_id)
    return f"""# Colab GPU Studio fixed trusted bootstrap; no caller Python is executed.
import base64 as _base64
import json as _json
import os as _os
from pathlib import Path as _Path
import subprocess as _subprocess
import sys as _sys

_execution_id = {execution_literal}
_root = _Path("/content/ai-studio")
_dir = _root / _execution_id
_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
_lock = _dir / ".worker.lock"
_marker = _dir / "marker.json"
_status = _dir / "status.json"
_worker = _dir / "runtime_worker.py"
_payload = _dir / "payload.json"
_log = _dir / "worker.log"
_result = {{"execution_id": _execution_id, "directory": str(_dir), "started": False, "existing": False}}
if _marker.exists():
    _result["existing"] = True
elif _status.exists() or _lock.exists():
    _result["existing"] = True
else:
    try:
        # mode=x maps to an O_EXCL|O_CREAT exclusive lock.
        _lock.open("x").close()
    except FileExistsError:
        _result["existing"] = True
    else:
        for _filename, _content in {extra_modules!r}.items():
            (_dir / _filename).write_bytes(_base64.b64decode(_content))
        _worker.write_bytes(_base64.b64decode({source_b64!r}))
        _payload.touch(mode=0o600)
        _payload.chmod(0o600)
        _payload.write_bytes(_base64.b64decode({payload_b64!r}))
        _marker.write_text(_json.dumps({{"execution_id": _execution_id, "state": "reserved"}}, sort_keys=True), encoding="utf-8")
        _log_handle = _log.open("ab")
        _proc = _subprocess.Popen([_sys.executable, str(_worker), str(_payload)], cwd=str(_dir),
                                  stdin=_subprocess.DEVNULL, stdout=_log_handle, stderr=_subprocess.STDOUT,
                                  shell=False, start_new_session=True)
        _log_handle.close()
        _marker.write_text(_json.dumps({{"execution_id": _execution_id, "state": "launched", "pid": _proc.pid}}, sort_keys=True), encoding="utf-8")
        _result.update({{"started": True, "pid": _proc.pid}})
print(_json.dumps(_result, separators=(",", ":")))
"""


def build_teardown_code(execution_id: str, payload: dict, worker_source: str | None = None) -> str:
    """Return fixed trusted teardown code using the same O_EXCL lock + status.json protocol."""
    execution_id = _uuid4(execution_id)
    if not isinstance(payload, dict):
        raise MCPError("teardown payload必须是对象")
    if worker_source is None:
        worker_source = Path(__file__).with_name("runtime_worker.py").read_text(encoding="utf-8")
    if not isinstance(worker_source, str) or not worker_source:
        raise MCPError("trusted worker source is unavailable")
    allowed_fields = (
        "execution_id",
        "execution_dir",
        "from_pool",
        "to_pool",
        "native_bootstrap_failed",
        "clean_native_only",
        "expected_baseline_digest",
        "service_instance_id",
        "allow_mock_gpu",
    )
    clean_payload = {k: payload[k] for k in allowed_fields if k in payload}
    clean_payload["execution_id"] = execution_id
    clean_payload["execution_dir"] = f"/content/ai-studio/{execution_id}"
    source_b64 = base64.b64encode(worker_source.encode("utf-8")).decode("ascii")
    payload_b64 = _b64_json(clean_payload)
    execution_literal = repr(execution_id)
    return f"""# Colab GPU Studio fixed trusted teardown; no caller Python is executed.
import base64 as _base64
import json as _json
from pathlib import Path as _Path
import subprocess as _subprocess
import sys as _sys

_execution_id = {execution_literal}
_root = _Path("/content/ai-studio")
_dir = _root / _execution_id
_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
_lock = _dir / ".worker.lock"
_marker = _dir / "marker.json"
_status = _dir / "status.json"
_worker = _dir / "runtime_worker.py"
_payload = _dir / "payload.json"
_log = _dir / "worker.log"
_result = {{"execution_id": _execution_id, "directory": str(_dir), "started": False, "existing": False}}
if _marker.exists() or _status.exists() or _lock.exists():
    _result["existing"] = True
else:
    try:
        _lock.open("x").close()
    except FileExistsError:
        _result["existing"] = True
    else:
        _worker.write_bytes(_base64.b64decode({source_b64!r}))
        _payload.touch(mode=0o600)
        _payload.chmod(0o600)
        _payload.write_bytes(_base64.b64decode({payload_b64!r}))
        _marker.write_text(_json.dumps({{"execution_id": _execution_id, "state": "reserved", "kind": "teardown"}}, sort_keys=True), encoding="utf-8")
        _log_handle = _log.open("ab")
        _proc = _subprocess.Popen([_sys.executable, str(_worker), "teardown", str(_payload)], cwd=str(_dir),
                                  stdin=_subprocess.DEVNULL, stdout=_log_handle, stderr=_subprocess.STDOUT,
                                  shell=False, start_new_session=True)
        _log_handle.close()
        _marker.write_text(_json.dumps({{"execution_id": _execution_id, "state": "launched", "kind": "teardown", "pid": _proc.pid}}, sort_keys=True), encoding="utf-8")
        _result.update({{"started": True, "pid": _proc.pid}})
print(_json.dumps(_result, separators=(",", ":")))
"""


def build_poll_code(execution_id: str, *, inline_small_artifacts: bool = False) -> str:
    """Return a read-only status reader with a hard 100MB output bound."""
    execution_id = _uuid4(execution_id)
    encoded = base64.b64encode(execution_id.encode()).decode("ascii")
    inline_max = _INLINE_ARTIFACT_MAX_BYTES if inline_small_artifacts else 0
    inline_total_max = _INLINE_ARTIFACT_TOTAL_MAX_BYTES if inline_small_artifacts else 0
    return f"""# Colab GPU Studio fixed read-only recovery probe.
import base64 as _base64
import json as _json
from pathlib import Path as _Path

_execution_id = _base64.b64decode({encoded!r}).decode("ascii")
_dir = _Path("/content/ai-studio") / _execution_id
_status_path = _dir / "status.json"
_meta_path = _dir / "meta.json"
_limit = 100 * 1024 * 1024
_inline_max = {inline_max}
_inline_total_max = {inline_total_max}
_result = {{"execution_id": _execution_id, "state": "running", "outputs": []}}
if not _status_path.exists():
    _result["marker"] = (_dir / "marker.json").exists()
else:
    _status = _json.loads(_status_path.read_text(encoding="utf-8"))
    if isinstance(_status, dict):
        _result.update({{key: _status.get(key) for key in ("state", "error_code", "error_message", "stage", "timings", "stage_started", "updated_at", "gpu_diagnostics", "creation_runtime", "venv_isolated", "vm_baseline_digest", "clean_after_smoke", "cleaned", "teardown_report") if key in _status}})
    if _result.get("state") == "completed" and _meta_path.exists():
        _meta = _json.loads(_meta_path.read_text(encoding="utf-8"))
        _artifacts = _meta.get("artifacts") if isinstance(_meta, dict) else None
        if isinstance(_artifacts, list):
            if len(_artifacts) > 100 or len(_json.dumps(_artifacts)) > 1048576:
                raise ValueError("artifact metadata exceeds limit")
            _result["artifacts"] = _artifacts
            if _inline_max > 0:
                _inline = {{}}
                _inline_total = 0
                _root_res = _dir.resolve()
                for _idx, _a in enumerate(_artifacts):
                    if not isinstance(_a, dict) or "storage" in _a:
                        continue
                    _rel = _a.get("path")
                    _sz = _a.get("size_bytes")
                    if not isinstance(_rel, str) or type(_sz) is not int or not (0 < _sz <= _inline_max):
                        continue
                    if _inline_total + _sz > _inline_total_max:
                        continue
                    _fp = (_dir / _rel).resolve()
                    if _root_res not in _fp.parents or not _fp.is_file():
                        continue
                    if _fp.stat().st_size != _sz:
                        continue
                    _raw = _fp.read_bytes()
                    if len(_raw) == _sz:
                        _inline[str(_idx)] = _base64.b64encode(_raw).decode("ascii")
                        _inline_total += _sz
                if _inline:
                    _result["inline_artifacts"] = _inline
        _total = 0
        for _item in (_meta.get("outputs", []) if isinstance(_meta, dict) and _artifacts is None else []):
            if not isinstance(_item, dict):
                continue
            _relative = _item.get("path")
            if not isinstance(_relative, str):
                continue
            _file = (_dir / _relative).resolve()
            if _dir.resolve() not in _file.parents or not _file.is_file():
                continue
            _size = _file.stat().st_size
            if _size > _limit or _total + _size > _limit:
                _result = {{"execution_id": _execution_id, "state": "failed", "error_code": "gpu_output_too_large", "error_message": "output exceeds 100MB"}}
                break
            _content = _file.read_bytes()
            _total += len(_content)
            _result.setdefault("outputs", []).append({{**_item, "content": _base64.b64encode(_content).decode("ascii"), "content_encoding": "base64"}})
# A terminal status wins. For older markers without PID retain timeout recovery.
if _result.get('state') in ('running', 'starting', 'reserved'):
    _marker_path = _dir / 'marker.json'
    if _marker_path.is_file():
        _marker_data = _json.loads(_marker_path.read_text())
        _pid = _marker_data.get('pid')
        if type(_pid) is int and _pid > 0:
            import os as _os
            try:
                _os.kill(_pid, 0)
                _proc_stat = _Path('/proc') / str(_pid) / 'stat'
                _dead = _proc_stat.is_file() and _proc_stat.read_text().rsplit(')', 1)[1].split()[0] == 'Z'
            except ProcessLookupError:
                _dead = True
            except PermissionError:
                _dead = False
            if _dead:
                _result.update({{'state':'failed', 'error_code':'gpu_worker_exited',
                    'error_message':'Worker process exited before completion'}})
print(_json.dumps(_result, separators=(",", ":")))
"""


def build_chunk_code(execution_id: str, index: int, offset: int, chunk_bytes: int = _DEFAULT_CHUNK_BYTES) -> str:
    execution_id = _uuid4(execution_id)
    if (
        type(index) is not int
        or not 0 <= index < 100
        or type(offset) is not int
        or not 0 <= offset <= 2 * 1024**3
        or type(chunk_bytes) is not int
        or not 1 <= chunk_bytes <= _MAX_CHUNK_BYTES
    ):
        raise MCPError('invalid chunk request')
    return f"""# Trusted bounded artifact reader; no caller paths.
import base64 as _b
import json as _j
from pathlib import Path as _P
_e = {execution_id!r}
_root = _P('/content/ai-studio') / _e
_meta = _j.loads((_root / 'meta.json').read_text())
if _meta.get('execution_id') != _e:
    raise ValueError('execution mismatch')
_item = _meta['artifacts'][{index}]
_path = (_root / _item['path']).resolve()
if _root.resolve() not in _path.parents or not _path.is_file():
    raise ValueError('artifact path escaped')
with _path.open('rb') as _f:
    _f.seek({offset})
    _chunk = _f.read({chunk_bytes})
print(_j.dumps({{'execution_id': _e, 'offset': {offset}, 'content': _b.b64encode(_chunk).decode()}}))
"""


def _walk(value: Any):
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


class MCPGPUExecutor:
    is_mock = False

    def __init__(
        self,
        store,
        owner_id: str,
        *,
        account_id: str | None = None,
        google: GoogleService | None = None,
        client: httpx.Client | None = None,
        endpoint: str = MCP_ENDPOINT,
        timeout: float = 30.0,
        artifact_store=None,
    ):
        if store is None or not owner_id:
            raise MCPError("MCPGPUExecutor需要Repository和owner_id")
        self.artifact_store = artifact_store
        self.store = store
        self.owner_id, self.account_id = owner_id, account_id
        self.client = client or _make_colab_http_client(timeout)
        self._owns_client = client is None
        self.google = google or GoogleService(store, owner_id, account_id=account_id, client=self.client)
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        if self.endpoint != MCP_ENDPOINT:
            # Tests may inject a MockTransport endpoint, but production code must
            # not turn this adapter into an arbitrary outbound proxy.
            parsed = httpx.URL(self.endpoint)
            if parsed.scheme != "https" or not parsed.host:
                raise MCPError("Colab MCP endpoint未验证")

    @classmethod
    def from_environment(cls, store=None, owner_id=None, **kwargs):
        return cls(store, owner_id, **kwargs)

    def _rpc(self, method: str, params: dict) -> Any:
        token = self.google.access_token("colab")
        request_id = str(uuid.uuid4())
        body = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        headers = {
            "Authorization": "Bearer " + token,
            "Accept": "application/json,text/event-stream",
            "Content-Type": "application/json",
        }
        # Every call made here is safe to repeat (create_session has a fixed session id,
        # execute_code has an execution id, status probes only read), so a dropped or
        # timed-out connection is retried a few times instead of killing a running job.
        for attempt in range(_NETWORK_ATTEMPTS):
            try:
                response = self.client.post(self.endpoint, json=body, headers=headers)
                break
            except httpx.TransportError as exc:
                if attempt == _NETWORK_ATTEMPTS - 1:
                    raise MCPTransient(f"Colab MCP网络请求失败（{type(exc).__name__}，已重试{attempt}次）") from exc
                time.sleep(2 * (attempt + 1))
            except httpx.HTTPError as exc:
                raise MCPError(f"Colab MCP网络请求失败（{type(exc).__name__}）") from exc
        if response.status_code in {408, 429, 500, 502, 503, 504}:
            raise MCPTransient(f'Colab MCP temporarily unavailable (HTTP {response.status_code})')
        if response.status_code < 200 or response.status_code >= 300:
            raise MCPError(f"Colab MCP请求失败（HTTP {response.status_code}）")
        envelope = self._parse_response(response)
        if not isinstance(envelope, dict):
            raise MCPError("Colab MCP响应格式无效")
        if envelope.get("error"):
            error = envelope['error']
            message = error.get('message', '') if isinstance(error, dict) else ''
            code = error.get('code') if isinstance(error, dict) else None
            # Protocol/authorization failures are permanent, even if text happens
            # to contain a transient phrase. Never echo upstream messages.
            if code not in (-32700, -32600, -32601, -32602, 401, 403) and isinstance(message, str):
                permanent = re.search(r'permission|unauthori|forbidden|invalid|denied', message, re.I)
                if not permanent and _TRANSIENT.search(message):
                    raise MCPTransient('Colab MCP temporary RPC failure')
            raise MCPError("Colab MCP工具调用失败")
        return envelope.get("result")

    @staticmethod
    def _parse_response(response: httpx.Response) -> dict:
        content_type = response.headers.get("content-type", "").lower()
        if "text/event-stream" not in content_type:
            try:
                value = response.json()
            except (ValueError, json.JSONDecodeError) as exc:
                raise MCPError("Colab MCP JSON响应无效") from exc
            if isinstance(value, dict):
                return value
            raise MCPError("Colab MCP JSON响应不是对象")
        found = None
        for line in response.text.splitlines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                candidate = json.loads(data)
            except json.JSONDecodeError as exc:
                raise MCPError("Colab MCP SSE响应无效") from exc
            if isinstance(candidate, dict) and ("result" in candidate or "error" in candidate):
                found = candidate
        if found is None:
            raise MCPError("Colab MCP SSE响应为空")
        return found

    @staticmethod
    def _tool_value(result: Any) -> Any:
        if not isinstance(result, dict):
            raise MCPError("Colab MCP工具返回无效结构")
        if result.get("isError") or result.get("executionError"):
            detail = _error_detail(result)
            log.warning("Colab executor error: %s", detail)  # restricted backend log only
            name = (result.get("executionError") or {}).get("name") if isinstance(result.get("executionError"), dict) else None
            error = MCPTransient if _TRANSIENT.search(detail) else MCPError
            raise error(f"Colab固定执行程序报错（{name or '工具错误'}），详见后台日志")
        if result.get("outputTruncated"):
            raise MCPError("Colab文本传输被截断，结果未保存；请使用受控文件传输或降低产物大小")
        if isinstance(result.get("structuredContent"), dict):
            return MCPGPUExecutor._tool_value(result["structuredContent"])
        if "stdout" in result:
            text = result.get("stdout", "")
            if not isinstance(text, str):
                raise MCPError("Colab stdout结构无效")
            for line in reversed(text.splitlines()):
                try:
                    value = json.loads(line)
                    if isinstance(value, dict):
                        return value
                except json.JSONDecodeError:
                    continue
            raise MCPError("固定执行程序没有返回完整JSON状态")
        if "content" in result:
            for item in result.get("content", []):
                if (
                    isinstance(item, dict)
                    and item.get("type") == "text"
                    and isinstance(item.get("text"), str)
                ):
                    try:
                        return MCPGPUExecutor._tool_value(json.loads(item["text"]))
                    except json.JSONDecodeError:
                        continue
            raise MCPError("MCP工具没有返回可校验的JSON内容")
        return result

    def _tool(self, name: str, arguments: dict) -> Any:
        for attempt in range(_NETWORK_ATTEMPTS):
            try:
                return self._tool_value(self._rpc("tools/call", {"name": name, "arguments": arguments}))
            except MCPTransient:
                if attempt == _NETWORK_ATTEMPTS - 1:
                    raise
                time.sleep(3 * (attempt + 1))

    def _tunnel_lease(self, handle: dict) -> dict | None:
        if not isinstance(handle, dict) or handle.get("owner_id") != self.owner_id:
            return None
        runtime_id = handle.get("runtime_id")
        if not runtime_id and isinstance(handle.get("runtime_name"), str) and handle["runtime_name"].startswith("runtimes/"):
            runtime_id = handle["runtime_name"].split("/", 1)[1]
        if not runtime_id or not hasattr(self.store, "list"):
            return None
        req_id = handle.get("request_id")
        job_id = handle.get("owner_job_id") or handle.get("job_id")
        for lease in self.store.list("leases", self.owner_id):
            if lease.get("runtime_id") != runtime_id:
                continue
            if req_id and lease.get("request_id") != req_id:
                continue
            if job_id and lease.get("owner_job_id") != job_id:
                continue
            if lease.get("tunnel_endpoint"):
                return lease
        return None

    @staticmethod
    def _ws_pack_default(msg: dict) -> bytes:
        body = json.dumps(msg, default=str).encode("utf-8")
        # Jupyter's default protocol needs at least two buffers (json + zero-length
        # padding), otherwise strict servers reject the frame before reading it.
        buffers = [body, b""]
        nbufs = len(buffers)
        offsets = [4 * (nbufs + 1)]
        for index in range(nbufs - 1):
            offsets.append(offsets[-1] + len(buffers[index]))
        frame = bytearray(offsets[-1] + len(buffers[-1]))
        frame[0:4] = nbufs.to_bytes(4, "big")
        for index, offset in enumerate(offsets):
            start = 4 * (index + 1)
            frame[start : start + 4] = offset.to_bytes(4, "big")
        for index, buffer in enumerate(buffers):
            start = offsets[index]
            frame[start : start + len(buffer)] = buffer
        return bytes(frame)

    @staticmethod
    def _ws_unpack_default(raw: str | bytes) -> dict:
        if isinstance(raw, str):
            return json.loads(raw)
        nbufs = int.from_bytes(raw[:4], "big")
        if nbufs < 1:
            raise ValueError("invalid jupyter ws frame")
        offsets = [int.from_bytes(raw[4 * (i + 1) : 4 * (i + 2)], "big") for i in range(nbufs)]
        json_stop = offsets[1] if nbufs > 1 else len(raw)
        return json.loads(raw[offsets[0] : json_stop].decode("utf-8"))

    def _refresh_tunnel_proxy(self, lease: dict) -> tuple[str, str]:
        url = lease.get("proxy_url")
        enc_tok = lease.get("encrypted_proxy_token")
        exp = float(lease.get("proxy_token_expires_at") or 0)
        if url and enc_tok and (exp == 0 or exp > time.time() + 120):
            return str(url).rstrip("/"), decrypt_secret(enc_tok)
        from .colab import GoogleColabProvider

        provider = GoogleColabProvider(
            self.store,
            self.owner_id,
            account_id=self.account_id or lease.get("account_id"),
            client=self.client,
            google=self.google,
        )
        matched = provider._sync_tunnel_lease(lease)
        if matched is None:
            raise MCPError("Colab 隧道运行时已释放或不存在")
        fresh = self.store.get("leases", lease["id"], self.owner_id) or lease
        url = fresh.get("proxy_url")
        enc_tok = fresh.get("encrypted_proxy_token")
        if not url or not enc_tok:
            raise MCPError("Colab 隧道代理凭据缺失")
        return str(url).rstrip("/"), decrypt_secret(enc_tok)

    def _execute_tunnel_code(self, lease: dict, code: str, *, exec_timeout: float | None = None) -> dict:
        from websockets.exceptions import InvalidStatus
        from websockets.sync.client import connect as ws_connect

        fast_cp = env_flag("NATIVE_FAST_CONTROL_PLANE", True)
        base_timeout = float(getattr(self, "timeout", 30.0))
        override_timeout = getattr(self, "_exec_timeout_override", None)
        effective_timeout = float(exec_timeout if exec_timeout is not None else (override_timeout if override_timeout is not None else base_timeout))
        ws_open_timeout = min(6.0, effective_timeout) if fast_cp else min(15.0, effective_timeout)
        http_timeout = min(8.0, effective_timeout) if fast_cp else effective_timeout

        t_start = time.monotonic()
        proxy_url, proxy_token = self._refresh_tunnel_proxy(lease)
        refresh_s = time.monotonic() - t_start
        auth_params = {"colab-runtime-proxy-token": proxy_token, "authuser": "0"}
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-Colab-Client-Agent": "colab-cli",
            "X-Colab-Runtime-Proxy-Token": proxy_token,
        }

        def _ensure_kernel(current_id: str | None, *, verify_existing: bool) -> str:
            kid = current_id
            try:
                if kid and verify_existing:
                    r = self.client.get(
                        f"{proxy_url}/api/kernels/{kid}",
                        headers=headers,
                        params=auth_params,
                        timeout=http_timeout,
                    )
                    if r.status_code == 404:
                        kid = None
                    elif r.status_code >= 400:
                        raise MCPTransient(f"Colab kernel inspect HTTP {r.status_code}")
                if not kid:
                    rk = self.client.get(
                        f"{proxy_url}/api/kernels",
                        headers=headers,
                        params=auth_params,
                        timeout=http_timeout,
                    )
                    existing = rk.json() if rk.status_code == 200 else None
                    if isinstance(existing, list) and existing:
                        kid = existing[0].get("id")
                    else:
                        rc = self.client.post(
                            f"{proxy_url}/api/kernels",
                            json={"name": "python3"},
                            headers=headers,
                            params=auth_params,
                            timeout=http_timeout,
                        )
                        if rc.status_code >= 400:
                            raise MCPTransient(f"Colab kernel start HTTP {rc.status_code}")
                        kid = rc.json().get("id")
                    if not isinstance(kid, str) or not kid:
                        raise MCPError("Colab 隧道未能创建 Jupyter Kernel")
                    self.store.update("leases", lease["id"], {"kernel_id": kid}, self.owner_id)
                    lease["kernel_id"] = kid
                return kid
            except httpx.HTTPError as exc:
                raise MCPTransient(f"Colab kernel HTTP error ({type(exc).__name__})") from exc

        kernel_id = lease.get("kernel_id")
        skipped_kernel_verify = bool(kernel_id and fast_cp)
        t_kernel = time.monotonic()
        if not skipped_kernel_verify:
            kernel_id = _ensure_kernel(kernel_id, verify_existing=True)
        kernel_http_s = time.monotonic() - t_kernel

        ws_base = re.sub(r"^http", "ws", proxy_url, count=1)
        stdout_parts: list[str] = []
        error_info: dict | None = None
        ws_open_s = 0.0

        for ws_pass in range(2):
            session_id = uuid.uuid4().hex
            msg_id = uuid.uuid4().hex
            query = urlencode({**auth_params, "session_id": session_id})
            ws_url = f"{ws_base}/api/kernels/{kernel_id}/channels?{query}"
            req_msg = {
                "header": {
                    "msg_id": msg_id,
                    "username": "studio",
                    "session": session_id,
                    "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "msg_type": "execute_request",
                    "version": "5.3",
                },
                "parent_header": {},
                "metadata": {},
                "content": {
                    "code": code,
                    "silent": False,
                    "store_history": False,
                    "user_expressions": {},
                    "allow_stdin": False,
                    "stop_on_error": True,
                },
                "channel": "shell",
            }
            stdout_parts = []
            error_info = None
            reply_seen = False
            idle_seen = False
            deadline = time.monotonic() + effective_timeout
            t_ws = time.monotonic()
            ws_open_s = 0.0
            try:
                ws_kwargs: dict[str, Any] = {
                    "additional_headers": {
                        "X-Colab-Client-Agent": "colab-cli",
                        "X-Colab-Runtime-Proxy-Token": proxy_token,
                    },
                    "open_timeout": ws_open_timeout,
                    "close_timeout": 3.0 if fast_cp else 5.0,
                    "max_size": 32 * 1024 * 1024,
                }
                sock = _connect_ipv4_socket(ws_url, ws_open_timeout)
                if sock is not None:
                    ws_kwargs["sock"] = sock
                with ws_connect(ws_url, **ws_kwargs) as ws:
                    ws_open_s = time.monotonic() - t_ws
                    ws.send(self._ws_pack_default(req_msg))
                    while time.monotonic() < deadline:
                        remaining = max(0.5, deadline - time.monotonic())
                        raw = ws.recv(timeout=remaining)
                        msg = self._ws_unpack_default(raw)
                        if (msg.get("parent_header") or {}).get("msg_id") != msg_id:
                            continue
                        mtype = (msg.get("header") or {}).get("msg_type") or msg.get("msg_type")
                        content = msg.get("content") or {}
                        if mtype == "stream":
                            if content.get("name") == "stdout":
                                stdout_parts.append(str(content.get("text") or ""))
                        elif mtype in ("execute_result", "display_data"):
                            plain = (content.get("data") or {}).get("text/plain")
                            if isinstance(plain, str):
                                stdout_parts.append(plain + "\n")
                        elif mtype == "error":
                            error_info = {
                                "name": content.get("ename") or "ExecutionError",
                                "value": content.get("evalue") or "",
                            }
                        elif mtype == "execute_reply":
                            reply_seen = True
                            if content.get("status") == "error" and error_info is None:
                                error_info = {
                                    "name": content.get("ename") or "ExecutionError",
                                    "value": content.get("evalue") or "",
                                }
                        elif mtype == "status" and content.get("execution_state") == "idle":
                            idle_seen = True
                        if reply_seen and idle_seen:
                            break
                break
            except InvalidStatus as exc:
                status_code = getattr(getattr(exc, "response", None), "status_code", None)
                if status_code in {404, 410} and skipped_kernel_verify and ws_pass == 0:
                    log.info("tunnel_exec kernel_id stale (HTTP %s); recreating kernel", status_code)
                    t_recreate = time.monotonic()
                    kernel_id = _ensure_kernel(None, verify_existing=False)
                    kernel_http_s += time.monotonic() - t_recreate
                    skipped_kernel_verify = False
                    continue
                log.warning(
                    "tunnel_exec ws_status=%s total_s=%.3f refresh_s=%.3f kernel_http_s=%.3f ws_open_s=%.3f",
                    status_code, time.monotonic() - t_start, refresh_s, kernel_http_s, ws_open_s,
                )
                raise MCPTransient(f"Colab 隧道 WebSocket HTTP {status_code or 'error'}") from exc
            except TimeoutError as exc:
                log.warning(
                    "tunnel_exec timeout total_s=%.3f refresh_s=%.3f kernel_http_s=%.3f ws_open_s=%.3f",
                    time.monotonic() - t_start, refresh_s, kernel_http_s, ws_open_s,
                )
                raise MCPTransient("Colab 隧道 Kernel 执行超时") from exc
            except Exception as exc:
                if isinstance(exc, (MCPError, MCPTransient)):
                    raise
                log.warning(
                    "tunnel_exec ws_error=%s total_s=%.3f refresh_s=%.3f kernel_http_s=%.3f ws_open_s=%.3f",
                    type(exc).__name__, time.monotonic() - t_start, refresh_s, kernel_http_s, ws_open_s,
                )
                raise MCPTransient(f"Colab 隧道 WebSocket 异常（{type(exc).__name__}）") from exc
        total_s = time.monotonic() - t_start
        ws_exec_s = max(0.0, total_s - refresh_s - kernel_http_s - ws_open_s)
        if total_s >= 2.0:
            log.info(
                "tunnel_exec slow total_s=%.3f refresh_s=%.3f kernel_http_s=%.3f ws_open_s=%.3f ws_exec_s=%.3f",
                total_s, refresh_s, kernel_http_s, ws_open_s, ws_exec_s,
            )
        if error_info is not None:
            return {"isError": True, "executionError": error_info, "stdout": "".join(stdout_parts)}
        return {"stdout": "".join(stdout_parts)}

    def _session(self, handle: dict, job_id: str) -> str:
        if not isinstance(handle, dict) or handle.get("owner_id") != self.owner_id:
            raise MCPError("拒绝使用不属于当前owner的Colab租约")
        parent = handle.get("runtime_name")
        if not isinstance(parent, str) or not _RUNTIME.fullmatch(parent):
            raise MCPError("Colab runtime句柄未验证")
        if not isinstance(job_id, str) or not _SAFE_ID.fullmatch(job_id):
            raise MCPError("job id不适合作为稳定MCP sessionId")
        session_id = "studio-" + hashlib.sha256(job_id.encode()).hexdigest()[:32]
        if self._tunnel_lease(handle) is not None:
            return f"{parent}/sessions/{session_id}"
        result = self._tool("create_session", {"parent": parent, "sessionId": session_id})
        session = next(
            (
                value.get("name") or value.get("session")
                for value in _walk(result)
                if isinstance(value, dict)
                and isinstance(value.get("name") or value.get("session"), str)
                and (value.get("name") or value.get("session")).startswith(parent + "/sessions/")
            ),
            None,
        )
        if not session or not re.fullmatch(re.escape(parent) + r"/sessions/[a-z][a-z0-9-]{0,62}", session):
            raise MCPError("Colab MCP create_session未返回合法完整session资源名")
        return session

    @staticmethod
    def _payload(job: dict, execution_id: str, root: str) -> dict:
        snapshot = job.get("snapshot")
        if not isinstance(snapshot, dict):
            raise MCPError("job缺少不可变snapshot")
        # Explicit allow-list prevents credentials or arbitrary caller fields from
        # crossing into the remote runtime.
        payload = {
            "execution_id": execution_id,
            "owner_id": job.get("owner_id"),
            "output_transport": "mcp-chunks-v1",
            "job_id": job.get("id"),
            "execution_dir": root,
            "max_runtime_minutes": job.get("max_runtime_minutes", 10),
            "download_timeout_seconds": min(float(job.get("download_timeout_seconds", 600)), 600),
            "system_smoke": snapshot.get("system_smoke") is True,
            "snapshot": snapshot,
        }
        cache = job.get("drive_cache")
        if isinstance(cache, dict):
            if job.get("use_drive_cache") is False:
                payload["use_drive_cache"] = False
            payload["drive_cache"] = {
                "folder_id": cache.get("folder_id"),
                "runtime_folder_id": cache.get("runtime_folder_id"),
                "access_token": cache.get("access_token"),
            }
        if job.get("generation_timeouts"):
            payload["generation_timeouts"] = True
        if job.get("generation_timeouts") or snapshot.get("creation_api"):
            deadlines = [job.get("deadline_epoch"), job.get("task_deadline_epoch")]
            payload["deadline_epoch"] = min((float(d) for d in deadlines if d is not None), default=None)
        from .instances import POOL_CREATION, job_pool

        is_creation_pool = job_pool(job) == POOL_CREATION
        if snapshot.get("creation_api") or (is_creation_pool and job.get("instance_id")):
            payload["instance_id"] = job.get("instance_id")
            payload["use_resident_service"] = True
            if job.get("selection_reason") in {"reuse_idle", "requested_instance"}:
                payload["snapshot"] = {**snapshot, "runtime_compute_spec": None}
        if job.get("expected_baseline_digest"):
            payload["expected_baseline_digest"] = job["expected_baseline_digest"]
        if "lora_drive_cache" in job:
            loras = job["lora_drive_cache"]
            payload["lora_drive_cache"] = (
                {"folder_id": loras.get("folder_id"), "access_token": loras.get("access_token")}
                if isinstance(loras, dict) else None
            )
        return payload

    def _exec_code_on_handle(self, handle: dict, session: str, execution_id: str, code: str) -> Any:
        tunnel_lease = self._tunnel_lease(handle)
        if tunnel_lease is not None:
            fast_cp = env_flag("NATIVE_FAST_CONTROL_PLANE", True)
            for attempt in range(_NETWORK_ATTEMPTS):
                try:
                    return self._tool_value(self._execute_tunnel_code(tunnel_lease, code))
                except MCPTransient as exc:
                    if attempt == _NETWORK_ATTEMPTS - 1:
                        raise
                    delay = 0.5 * (attempt + 1) if fast_cp else 2 * (attempt + 1)
                    log.warning("tunnel_exec retry attempt=%d/%d delay_s=%.1f err=%s", attempt + 1, _NETWORK_ATTEMPTS, delay, type(exc).__name__)
                    time.sleep(delay)
        return self._tool("execute_code", {"session": session, "executionId": execution_id, "code": code})

    def creation_process_health(self, handle: dict, job: dict, runtime: dict) -> bool | None:
        """Read the process identity on an already allocated idle VM."""
        ident = runtime.get("process", "")
        if not isinstance(ident, str) or not re.fullmatch(r"[1-9][0-9]*:[0-9]+", ident):
            return False
        pid, start = ident.split(":")
        name = _instance_service_name(job["instance_id"])
        service_root = str(COMFY_SERVICE_ROOT / name)
        service = str(COMFY_SERVICE_ROOT / name / "ComfyUI")
        venv_bin = str(COMFY_SERVICE_ROOT / name / "venv" / "bin")
        loopback = _service_api_url("/system_stats")
        session = self._session(handle, job["id"])
        code = f"""import json, os, urllib.request
from pathlib import Path
proc = Path("/proc") / {pid!r}
healthy = False
try:
    stat = (proc / "stat").read_text().rsplit(")", 1)[1].split()
    raw_cmd = (proc / "cmdline").read_bytes()
    cmdline = [item.decode("utf-8", "replace") for item in raw_cmd.split(b"\\0") if item]
    venv_dir = Path({service_root!r}) / "venv"
    cmd_ok = bool(cmdline)
    if cmd_ok and venv_dir.is_dir():
        c0 = Path(cmdline[0])
        cmd_ok = c0.parent.resolve() == Path({venv_bin!r}).resolve() and c0.name.startswith("python")
    if stat[19] == {start!r} and os.readlink(proc / "cwd") == {service!r} and cmd_ok:
        with urllib.request.urlopen({loopback!r}, timeout=3) as response:
            healthy = response.status == 200
except (OSError, IndexError):
    pass
print(json.dumps({{"healthy": healthy}}))
"""
        result = self._exec_code_on_handle(handle, session, str(uuid.uuid4()), code)
        return result.get("healthy") if isinstance(result, dict) else None

    def start_teardown(self, handle: dict, job: dict, execution_id: str, teardown_payload: dict) -> dict:
        execution_id = _uuid4(execution_id)
        if not isinstance(job, dict) or not isinstance(job.get("id"), str):
            raise MCPError("job无效")
        session = self._session(handle, job["id"])
        code = build_teardown_code(execution_id, teardown_payload)
        result = self._exec_code_on_handle(handle, session, execution_id, code)
        return {
            "execution_id": execution_id,
            "session": session,
            "idempotent": bool(isinstance(result, dict) and result.get("existing")),
            "remote": result if isinstance(result, dict) else {},
        }

    def restart_kernel(self, handle: dict) -> bool:
        """Optionally restart the Colab Jupyter Kernel after clean teardown to free Kernel RAM."""
        tunnel_lease = self._tunnel_lease(handle)
        if tunnel_lease is None or not tunnel_lease.get("kernel_id"):
            return False
        try:
            proxy_url, proxy_token = self._refresh_tunnel_proxy(tunnel_lease)
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "X-Colab-Client-Agent": "colab-cli",
                "X-Colab-Runtime-Proxy-Token": proxy_token,
            }
            resp = self.client.post(
                f"{proxy_url}/api/kernels/{tunnel_lease['kernel_id']}/restart",
                headers=headers,
                params={"colab-runtime-proxy-token": proxy_token, "authuser": "0"},
                timeout=10.0,
            )
            return 200 <= resp.status_code < 300
        except Exception:
            return False

    def interrupt_creation_service(self, handle: dict, job: dict) -> bool:
        """Interrupt a resident prompt and confirm the worker can no longer submit work."""
        if not job.get("execution_id"):
            return False
        execution_id = _uuid4(job["execution_id"])
        interrupt_url = _service_api_url("/interrupt")
        queue_url = _service_api_url("/queue")
        session = self._session(handle, job["id"])
        code = f"""import json, time, urllib.request
from pathlib import Path
status_path = Path("/content/ai-studio") / {execution_id!r} / "status.json"
interrupted = False
try:
    req = urllib.request.Request({interrupt_url!r}, data=b"{{}}", method="POST",
                                 headers={{"Content-Type": "application/json"}})
    with urllib.request.urlopen(req, timeout=3) as resp:
        accepted = 200 <= resp.status < 300
    deadline = time.monotonic() + 4.0
    while accepted and time.monotonic() < deadline:
        has_status = status_path.is_file()
        status = json.loads(status_path.read_text()) if has_status else {{}}
        with urllib.request.urlopen({queue_url!r}, timeout=2) as qresp:
            qdata = json.loads(qresp.read().decode("utf-8"))
        queue_empty = qdata.get("queue_running") == qdata.get("queue_pending") == []
        if queue_empty and (not has_status or status.get("state") in {{"completed", "failed"}}):
            interrupted = True
            break
        time.sleep(0.3)
except Exception:
    pass
print(json.dumps({{"interrupted": interrupted}}))
"""
        result = self._exec_code_on_handle(handle, session, str(uuid.uuid4()), code)
        return bool(isinstance(result, dict) and result.get("interrupted"))

    def start(self, handle: dict, job: dict, execution_id: str) -> dict:
        execution_id = _uuid4(execution_id)
        if not isinstance(job, dict) or not isinstance(job.get("id"), str):
            raise MCPError("job无效")
        session = self._session(handle, job["id"])
        root = "/content/ai-studio/" + execution_id
        payload = self._payload(job, execution_id, root)
        if (job.get("snapshot") or {}).get("native_comfy"):
            from .native import native_optimization_flags, runtime_secrets
            payload["native_secrets"] = runtime_secrets(self.store, job)
            payload["native_plugin_generation"] = job["snapshot"].get("native_plugin_generation")
            payload["native_flags"] = native_optimization_flags()
        else:
            payload["creation_flags"] = creation_optimization_flags()
        code = build_bootstrap_code(execution_id, payload)
        result = self._exec_code_on_handle(handle, session, execution_id, code)
        return {
            "execution_id": execution_id,
            "session": session,
            "idempotent": bool(isinstance(result, dict) and result.get("existing")),
            "remote": result if isinstance(result, dict) else {},
        }

    def _stream_tunnel_artifact_chunks(
        self,
        lease: dict,
        execution_id: str,
        index: int,
        size_bytes: int,
        objects: Any,
        key: str,
        offset: int,
        *,
        chunk_bytes: int = _TUNNEL_CHUNK_BYTES,
    ) -> tuple[int, int]:
        """Stream remaining artifact chunks over a single persistent Jupyter WebSocket."""
        from websockets.exceptions import InvalidStatus
        from websockets.sync.client import connect as ws_connect

        base_timeout = float(getattr(self, "timeout", 30.0))
        ws_open_timeout = min(6.0, base_timeout)
        per_chunk_timeout = max(20.0, base_timeout)
        proxy_url, proxy_token = self._refresh_tunnel_proxy(lease)
        auth_params = {"colab-runtime-proxy-token": proxy_token, "authuser": "0"}
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-Colab-Client-Agent": "colab-cli",
            "X-Colab-Runtime-Proxy-Token": proxy_token,
        }
        kernel_id = lease.get("kernel_id")
        if not kernel_id:
            # Ensure a kernel exists before opening the streaming WS.
            self._execute_tunnel_code(lease, "print('{\"ok\":true}')", exec_timeout=10.0)
            kernel_id = lease.get("kernel_id")
            if not kernel_id:
                raise MCPTransient("Colab 隧道 Kernel 尚未就绪")
        ws_base = re.sub(r"^http", "ws", proxy_url, count=1)
        reads = 0

        for ws_pass in range(2):
            session_id = uuid.uuid4().hex
            query = urlencode({**auth_params, "session_id": session_id})
            ws_url = f"{ws_base}/api/kernels/{kernel_id}/channels?{query}"
            ws_kwargs: dict[str, Any] = {
                "additional_headers": {
                    "X-Colab-Client-Agent": headers["X-Colab-Client-Agent"],
                    "X-Colab-Runtime-Proxy-Token": proxy_token,
                },
                "open_timeout": ws_open_timeout,
                "close_timeout": 3.0,
                "max_size": 32 * 1024 * 1024,
            }
            sock = _connect_ipv4_socket(ws_url, ws_open_timeout)
            if sock is not None:
                ws_kwargs["sock"] = sock
            try:
                with ws_connect(ws_url, **ws_kwargs) as ws:
                    while offset < size_bytes:
                        reads += 1
                        code = build_chunk_code(execution_id, index, offset, chunk_bytes=chunk_bytes)
                        msg_id = uuid.uuid4().hex
                        req_msg = {
                            "header": {
                                "msg_id": msg_id,
                                "username": "studio",
                                "session": session_id,
                                "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                "msg_type": "execute_request",
                                "version": "5.3",
                            },
                            "parent_header": {},
                            "metadata": {},
                            "content": {
                                "code": code,
                                "silent": False,
                                "store_history": False,
                                "user_expressions": {},
                                "allow_stdin": False,
                                "stop_on_error": True,
                            },
                            "channel": "shell",
                        }
                        ws.send(self._ws_pack_default(req_msg))
                        stdout_parts: list[str] = []
                        error_info: dict | None = None
                        reply_seen = False
                        idle_seen = False
                        deadline = time.monotonic() + per_chunk_timeout
                        while time.monotonic() < deadline:
                            remaining = max(0.5, deadline - time.monotonic())
                            raw = ws.recv(timeout=remaining)
                            msg = self._ws_unpack_default(raw)
                            if (msg.get("parent_header") or {}).get("msg_id") != msg_id:
                                continue
                            mtype = (msg.get("header") or {}).get("msg_type") or msg.get("msg_type")
                            content = msg.get("content") or {}
                            if mtype == "stream":
                                if content.get("name") == "stdout":
                                    stdout_parts.append(str(content.get("text") or ""))
                            elif mtype in ("execute_result", "display_data"):
                                plain = (content.get("data") or {}).get("text/plain")
                                if isinstance(plain, str):
                                    stdout_parts.append(plain + "\n")
                            elif mtype == "error":
                                error_info = {
                                    "name": content.get("ename") or "ExecutionError",
                                    "value": content.get("evalue") or "",
                                }
                            elif mtype == "execute_reply":
                                reply_seen = True
                                if content.get("status") == "error" and error_info is None:
                                    error_info = {
                                        "name": content.get("ename") or "ExecutionError",
                                        "value": content.get("evalue") or "",
                                    }
                            elif mtype == "status" and content.get("execution_state") == "idle":
                                idle_seen = True
                            if reply_seen and idle_seen:
                                break
                        if error_info is not None:
                            reply = self._tool_value(
                                {"isError": True, "executionError": error_info, "stdout": "".join(stdout_parts)}
                            )
                        else:
                            reply = self._tool_value({"stdout": "".join(stdout_parts)})
                        if not isinstance(reply, dict) or reply.get("execution_id") != execution_id or reply.get("offset") != offset:
                            raise MCPError("artifact chunk identity mismatch")
                        try:
                            chunk = base64.b64decode(reply["content"], validate=True)
                        except (KeyError, ValueError, TypeError) as exc:
                            raise MCPError("artifact chunk encoding invalid") from exc
                        if not chunk or len(chunk) > chunk_bytes:
                            raise MCPError("artifact chunk size invalid")
                        offset = objects.append(key, self.owner_id, execution_id, offset, chunk)
                return offset, reads
            except InvalidStatus as exc:
                status_code = getattr(getattr(exc, "response", None), "status_code", None)
                if status_code in {404, 410} and ws_pass == 0:
                    lease["kernel_id"] = None
                    self._execute_tunnel_code(lease, "print('{\"ok\":true}')", exec_timeout=10.0)
                    kernel_id = lease.get("kernel_id")
                    if kernel_id:
                        continue
                raise MCPTransient(f"Colab 隧道产物 WebSocket HTTP {status_code or 'error'}") from exc
            except TimeoutError as exc:
                raise MCPTransient("Colab 隧道产物流式拉取超时") from exc
            except (MCPError, MCPTransient):
                raise
            except Exception as exc:
                raise MCPTransient(f"Colab 隧道产物流式拉取异常（{type(exc).__name__}）") from exc
        return offset, reads

    def poll(self, handle: dict, job: dict) -> dict:
        if not isinstance(job, dict) or not isinstance(job.get("id"), str):
            raise MCPError("job无效")
        fast_cp = env_flag("NATIVE_FAST_CONTROL_PLANE", True)
        base_timeout = float(getattr(self, "timeout", 30.0))
        poll_timeout = min(12.0, base_timeout) if fast_cp else None
        t_poll0 = time.monotonic()
        execution_id = _uuid4(job.get("execution_id"))
        session = self._session(handle, job["id"])
        probe_id = str(uuid.uuid4())
        code = build_poll_code(execution_id, inline_small_artifacts=fast_cp)
        prev_override = getattr(self, "_exec_timeout_override", None)
        self._exec_timeout_override = poll_timeout
        try:
            result = self._exec_code_on_handle(handle, session, probe_id, code)
        finally:
            self._exec_timeout_override = prev_override
        status_poll_s = time.monotonic() - t_poll0
        if not isinstance(result, dict):
            raise MCPError("Colab状态响应无效")
        if result.get("execution_id") not in {None, execution_id}:
            raise MCPError("Colab状态执行标识不匹配")
        state = result.get("state", "running")
        # The scheduler cleans gpu_diagnostics before saving it.
        evidence = {
            "execution_id": execution_id,
            "gpu_diagnostics": result.get("gpu_diagnostics"),
            "status_updated_at": result.get("updated_at"),
        }
        for extra_key in ("venv_isolated", "vm_baseline_digest", "clean_after_smoke", "cleaned", "teardown_report"):
            if extra_key in result:
                evidence[extra_key] = result.get(extra_key)
        if state in {"reserved", "starting", "running", None}:
            stage = result.get("stage")
            stage_started = result.get("stage_started")
            if status_poll_s >= 2.0:
                log.info("mcp_poll job=%s state=running stage=%s status_poll_s=%.3f", job["id"], stage, status_poll_s)
            return {
                **evidence,
                "state": "running",
                "outputs": [],
                "stage": stage if isinstance(stage, str) else None,
                "stage_started": (
                    float(stage_started)
                    if isinstance(stage_started, (int, float)) and not isinstance(stage_started, bool)
                    else None
                ),
                "timings": result.get("timings") if isinstance(result.get("timings"), dict) else {},
            }
        if state == "aborted_unsynced":
            return {
                **evidence,
                "state": "aborted_unsynced",
                "outputs": [],
                "error_code": result.get("error_code", "native_unsynced_data"),
                "message": str(result.get("error_message", "检测到未同步数据，已中止转移并保护实例"))[:1400],
                "timings": result.get("timings") if isinstance(result.get("timings"), dict) else {},
            }
        if state == "failed":
            return {
                **evidence,
                "state": "failed",
                "outputs": [],
                "error_code": result.get("error_code", "gpu_worker_failed"),
                "message": str(result.get("error_message", "GPU worker failed"))[:1400],
                "timings": result.get("timings") if isinstance(result.get("timings"), dict) else {},
            }
        if state != "completed":
            raise MCPError("Colab状态值无效")
        if result.get("creation_runtime") is not None or (job.get("snapshot") or {}).get("creation_api"):
            evidence["creation_runtime"] = result.get("creation_runtime")
            if (job.get("snapshot") or {}).get("creation_warmup"):
                return {**evidence, "state": "completed", "timings": result.get("timings") or {}}
        if 'artifacts' in result:
            artifacts=result['artifacts']
            if (not isinstance(artifacts,list) or not 1 <= len(artifacts) <= 100
                    or result.get('execution_id')!=execution_id
                    or len(json.dumps(artifacts))>1048576):
                raise MCPError('invalid artifact manifest')
            t_art0 = time.monotonic()
            chunk_reads = 0
            inline_reads = 0
            inline_map = result.get('inline_artifacts') if isinstance(result.get('inline_artifacts'), dict) else {}
            if any('storage' not in item for item in artifacts):
                from app.config import settings

                from .artifacts import LocalArtifactStore
                objects = self.artifact_store or LocalArtifactStore(settings.data_dir / 'assets' / 'artifacts')
                local = []
                for index, item in enumerate(artifacts):
                    value = objects.begin(self.owner_id, execution_id, item['filename'], item['mime'],
                        item['size_bytes'], item['sha256'])
                    key = value['artifact_id']
                    offset = objects.offset(key, self.owner_id, execution_id)
                    inline_b64 = inline_map.get(str(index)) if offset == 0 else None
                    if isinstance(inline_b64, str) and 0 < item['size_bytes'] <= _INLINE_ARTIFACT_MAX_BYTES:
                        try:
                            inline_chunk = base64.b64decode(inline_b64, validate=True)
                        except (ValueError, TypeError):
                            inline_chunk = b""
                        if len(inline_chunk) == item['size_bytes']:
                            offset = objects.append(key, self.owner_id, execution_id, 0, inline_chunk)
                            inline_reads += 1
                    tunnel_lease = (
                        self._tunnel_lease(handle)
                        if hasattr(self, "store") and env_flag("CREATION_FAST_CONTROL_PLANE", True)
                        else None
                    )
                    if offset < item['size_bytes'] and tunnel_lease is not None:
                        offset, streamed_reads = self._stream_tunnel_artifact_chunks(
                            tunnel_lease,
                            execution_id,
                            index,
                            item['size_bytes'],
                            objects,
                            key,
                            offset,
                        )
                        chunk_reads += streamed_reads
                    while offset < item['size_bytes']:
                        chunk_reads += 1
                        reply = self._exec_code_on_handle(
                            handle,
                            session,
                            str(uuid.uuid4()),
                            build_chunk_code(execution_id, index, offset),
                        )
                        if not isinstance(reply, dict) or reply.get('execution_id') != execution_id or reply.get('offset') != offset:
                            raise MCPError('artifact chunk identity mismatch')
                        try:
                            chunk = base64.b64decode(reply['content'], validate=True)
                        except (KeyError, ValueError, TypeError) as exc:
                            raise MCPError('artifact chunk encoding invalid') from exc
                        if not chunk or len(chunk) > 1024**2:
                            raise MCPError('artifact chunk size invalid')
                        offset = objects.append(key, self.owner_id, execution_id, offset, chunk)
                    verified = objects.verify(key, self.owner_id, execution_id)
                    if (job.get("snapshot") or {}).get("creation_api"):
                        verified.update(comfy_node=item.get("comfy_node"), comfy_category=item.get("comfy_category"))
                    local.append(verified)
                artifacts = local
            log.info(
                "mcp_poll job=%s state=completed status_poll_s=%.3f artifact_s=%.3f chunks=%d inlined=%d total_s=%.3f",
                job["id"], status_poll_s, time.monotonic() - t_art0, chunk_reads, inline_reads, time.monotonic() - t_poll0,
            )
            return {**evidence, 'state':'completed','artifacts':artifacts,
                    'timings':result.get('timings') or {}}
        outputs = []
        total = 0
        for item in result.get("outputs", []):
            if not isinstance(item, dict) or not isinstance(item.get("content"), str):
                continue
            try:
                content = base64.b64decode(item["content"], validate=True)
            except (ValueError, TypeError) as exc:
                raise MCPError("Colab输出编码无效") from exc
            total += len(content)
            if total > MAX_OUTPUT_BYTES:
                raise MCPError("Colab输出超过100MB上限")
            outputs.append(
                {
                    "filename": str(item.get("filename", "output.bin")),
                    "content": content,
                    "mime": str(item.get("mime", "application/octet-stream")),
                    "metadata": item.get("metadata") if isinstance(item.get("metadata"), dict) else {},
                }
            )
        timings = result.get("timings") if isinstance(result.get("timings"), dict) else {}
        return {**evidence, "state": "completed", "outputs": outputs, "timings": timings}

    def close(self) -> None:
        if self._owns_client:
            self.client.close()


__all__ = ["MCP_ENDPOINT", "MCPError", "MCPGPUExecutor", "build_bootstrap_code", "build_poll_code", "build_teardown_code"]
