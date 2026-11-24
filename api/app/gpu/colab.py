"""Colab v1beta runtime lifecycle + v1 long-running Operations (official discoveries)."""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from contextlib import contextmanager
from urllib.parse import urlencode

import httpx

from app.security import decrypt_secret, encrypt_secret

from .google import GoogleError, GoogleService

logger = logging.getLogger(__name__)

RUNTIME_MISSING = "provider_runtime_missing"
RUNTIME_MISSING_MESSAGE = "Colab 运行时已回收"
KEEPALIVE_INTERVAL_SECONDS = 60.0
KEEPALIVE_RETRY_SECONDS = 30.0
KERNEL_EXEC_INTERVAL_SECONDS = 240.0
TUNNEL_RECONCILE_WINDOW_SECONDS = 60.0

_SAFE_OUTCOME = re.compile(r"^[A-Z0-9_]{1,64}$")
_TUNNEL_OUTCOME_NUMERIC = {
    1: "QUOTA_DENIED_REQUESTED_VARIANTS",
    2: "QUOTA_EXCEEDED_USAGE_TIME",
    4: "SUCCESS",
    5: "DENYLISTED",
}


def runtime_status(result):
    if not isinstance(result, dict) or any(
        result.get(flag) for flag in ("allocation_unknown", "operation_pending", "ambiguous", "pending")
    ):
        return "unknown"
    if result.get("deleted") is True:
        return "missing"
    return "present" if result.get("deleted") is False else "unknown"


class ColabError(GoogleError):
    def __init__(
        self,
        message: str,
        *,
        error_code: str | None = None,
        category: str | None = None,
        phase: str | None = None,
        http_status: int | None = None,
        timeout_kind: str | None = None,
        elapsed_seconds: float | None = None,
        outcome: str | None = None,
        remote_state_unknown: bool = False,
        retryable_before_post: bool = False,
    ):
        super().__init__(message)
        self.error_code = error_code
        self.category = category
        self.phase = phase
        self.http_status = http_status
        self.timeout_kind = timeout_kind
        self.elapsed_seconds = round(float(elapsed_seconds), 1) if elapsed_seconds is not None else None
        self.outcome = outcome
        self.remote_state_unknown = bool(remote_state_unknown)
        self.retryable_before_post = bool(retryable_before_post)

    def to_diagnostics(self) -> dict:
        data = {}
        for key in (
            "error_code",
            "category",
            "phase",
            "http_status",
            "timeout_kind",
            "elapsed_seconds",
            "outcome",
        ):
            value = getattr(self, key, None)
            if value is not None:
                data[key] = value
        return data


class ColabAllocationError(ColabError):
    def __init__(
        self,
        code,
        reason,
        *,
        message: str | None = None,
        error_code: str | None = None,
        category: str | None = None,
        phase: str | None = None,
        http_status: int | None = None,
        elapsed_seconds: float | None = None,
        outcome: str | None = None,
        remote_state_unknown: bool = False,
    ):
        self.reason = reason
        # Only the numeric code is shown; Google's own text may carry private data.
        if message is None:
            if reason == "NO_RUNTIMES":
                message = "Colab 暂时无法分配运行时（NO_RUNTIMES）；请稍后重试"
            else:
                message = f"Colab 分配 Operation 失败（code {code}）"
        resolved_code = error_code or ("gpu_capacity_unavailable" if reason == "NO_RUNTIMES" else "gpu_allocation_failed")
        resolved_category = category or ("capacity" if reason == "NO_RUNTIMES" else "upstream")
        resolved_outcome = outcome
        if resolved_outcome is None and isinstance(reason, str):
            norm = reason.strip().upper()
            if _SAFE_OUTCOME.fullmatch(norm):
                resolved_outcome = norm
        super().__init__(
            message,
            error_code=resolved_code,
            category=resolved_category,
            phase=phase,
            http_status=http_status,
            elapsed_seconds=elapsed_seconds,
            outcome=resolved_outcome,
            remote_state_unknown=remote_state_unknown,
            retryable_before_post=False,
        )


def _normalize_outcome(raw) -> str | None:
    if isinstance(raw, bool) or raw is None:
        return None
    if isinstance(raw, int):
        return _TUNNEL_OUTCOME_NUMERIC.get(raw)
    if isinstance(raw, str):
        norm = raw.strip().upper()
        if norm.startswith("OUTCOME_"):
            norm = norm[len("OUTCOME_"):]
        if _SAFE_OUTCOME.fullmatch(norm):
            return norm
    return None


def _classify_httpx_error(
    exc: httpx.HTTPError,
    *,
    phase: str | None,
    mutating: bool,
    elapsed: float,
    channel: str,
) -> ColabError:
    if isinstance(exc, httpx.ReadTimeout):
        timeout_kind, label, category = "read", "响应超时", "timeout"
        error_code = "colab_assign_timeout" if phase in {"assign_get", "assign_post", "v1beta_create"} else "colab_network_error"
    elif isinstance(exc, httpx.ConnectTimeout):
        timeout_kind, label, category, error_code = "connect", "连接超时", "network", "colab_network_error"
    elif isinstance(exc, httpx.WriteTimeout):
        timeout_kind, label, category, error_code = "write", "发送超时", "timeout", "colab_network_error"
    elif isinstance(exc, httpx.PoolTimeout):
        timeout_kind, label, category, error_code = "pool", "连接池等待超时", "timeout", "colab_network_error"
    elif isinstance(exc, httpx.ProxyError):
        timeout_kind, label, category, error_code = "proxy", "代理连接失败", "network", "colab_network_error"
    elif isinstance(exc, httpx.ConnectError):
        timeout_kind, label, category, error_code = "connect", "网络连接失败", "network", "colab_network_error"
    else:
        timeout_kind, label, category, error_code = "transport", "网络传输中断", "network", "colab_network_error"

    if mutating:
        if timeout_kind == "read":
            message = (
                f"等待 Colab 分配响应超时（{elapsed:.1f}s），可能是上游算力排队或网络波动；"
                "远端结果待确认，正在安全释放租约，完成后可重试"
            )
        else:
            message = (
                f"Colab {channel}网络异常（{label}，耗时 {elapsed:.1f}s）；"
                "远端结果可能未知，正在确认释放租约，完成后可重试"
            )
    elif phase == "assign_get":
        message = f"Colab 分配握手失败（{label}，耗时 {elapsed:.1f}s），未发送开机请求；请检查网络或稍后重试"
    else:
        message = f"Colab {channel}网络异常（{label}，耗时 {elapsed:.1f}s），远端结果可能未知，将保留租约进行协调"

    return ColabError(
        message,
        error_code=error_code,
        category=category,
        phase=phase,
        timeout_kind=timeout_kind,
        elapsed_seconds=elapsed,
        remote_state_unknown=bool(mutating),
        retryable_before_post=(not mutating) and (phase == "assign_get"),
    )


def _classify_http_status(
    status_code: int,
    *,
    phase: str | None,
    mutating: bool,
    elapsed: float,
    channel: str,
) -> ColabError:
    if status_code in {401, 403}:
        return ColabError(
            f"Colab 账号授权无效或无访问权限（HTTP {status_code}，耗时 {elapsed:.1f}s）；请检查账号授权或重新连接",
            error_code="colab_auth_failed",
            category="auth",
            phase=phase,
            http_status=status_code,
            elapsed_seconds=elapsed,
            remote_state_unknown=False,
            retryable_before_post=False,
        )
    if status_code == 412:
        return ColabError(
            f"Colab 拒绝分配运行时（HTTP 412，可能达到并发或配额上限，耗时 {elapsed:.1f}s）",
            error_code="colab_quota_exceeded",
            category="quota",
            phase=phase,
            http_status=status_code,
            elapsed_seconds=elapsed,
            remote_state_unknown=False,
            retryable_before_post=False,
        )
    if status_code == 429:
        return ColabError(
            f"Colab 请求过于频繁或触发限流（HTTP 429，耗时 {elapsed:.1f}s）；请稍后重试",
            error_code="colab_rate_limited",
            category="quota",
            phase=phase,
            http_status=status_code,
            elapsed_seconds=elapsed,
            remote_state_unknown=False,
            retryable_before_post=False,
        )
    if 400 <= status_code < 500:
        return ColabError(
            f"Colab 拒绝了{channel}请求（HTTP {status_code}，耗时 {elapsed:.1f}s）",
            error_code="colab_request_rejected",
            category="upstream",
            phase=phase,
            http_status=status_code,
            elapsed_seconds=elapsed,
            remote_state_unknown=False,
            retryable_before_post=False,
        )
    if mutating:
        return ColabError(
            f"Colab 服务端暂时不可用（HTTP {status_code}，等待 {elapsed:.1f}s），可能是上游算力分配繁忙；"
            "正在确认远端未创建实例后释放租约，完成后可重试",
            error_code="colab_upstream_unavailable",
            category="upstream",
            phase=phase,
            http_status=status_code,
            elapsed_seconds=elapsed,
            remote_state_unknown=True,
            retryable_before_post=False,
        )
    return ColabError(
        f"Colab 服务端暂时不可用（HTTP {status_code}，耗时 {elapsed:.1f}s）；请稍后重试",
        error_code="colab_upstream_unavailable",
        category="upstream",
        phase=phase,
        http_status=status_code,
        elapsed_seconds=elapsed,
        remote_state_unknown=False,
        retryable_before_post=(phase == "assign_get"),
    )


def _classify_tunnel_outcome(payload: dict, *, elapsed: float) -> ColabError | None:
    norm = _normalize_outcome(payload.get("outcome"))
    if not norm or norm == "SUCCESS":
        return None
    if any(key in norm for key in ("QUOTA_DENIED", "NO_RUNTIME", "UNAVAILABLE")):
        return ColabAllocationError(
            "TUNNEL_DENIED",
            "NO_RUNTIMES",
            message=f"Colab 拒绝分配所请求的 GPU 规格（{norm}，耗时 {elapsed:.1f}s）；请切换 GPU 规格或稍后重试",
            error_code="gpu_capacity_unavailable",
            category="capacity",
            phase="assign_post",
            http_status=200,
            elapsed_seconds=elapsed,
            outcome=norm,
            remote_state_unknown=False,
        )
    if any(key in norm for key in ("QUOTA_EXCEEDED", "TOO_MANY", "LIMIT")):
        return ColabAllocationError(
            "TUNNEL_QUOTA",
            norm,
            message=f"Colab 账号配额或并发已达上限（{norm}，耗时 {elapsed:.1f}s），拒绝分配新运行时",
            error_code="colab_quota_exceeded",
            category="quota",
            phase="assign_post",
            http_status=200,
            elapsed_seconds=elapsed,
            outcome=norm,
            remote_state_unknown=False,
        )
    if any(key in norm for key in ("DENYLIST", "BANNED", "RESTRICTED")):
        return ColabAllocationError(
            "TUNNEL_DENYLISTED",
            norm,
            message=f"Colab 账号当前受限（{norm}，耗时 {elapsed:.1f}s），拒绝分配运行时",
            error_code="colab_auth_failed",
            category="auth",
            phase="assign_post",
            http_status=200,
            elapsed_seconds=elapsed,
            outcome=norm,
            remote_state_unknown=False,
        )
    return None


_CLI_PUBLIC_CLIENT_ID = "764086051850-6qr4p6gpi6hn506pt8ejuq83di341hur.apps.googleusercontent.com"
_TUNNEL_BASE_URL = "https://colab.research.google.com"
_XSSI_PREFIX = ")]}'\n"
_HIGH_MEM_ONLY_ACCELERATORS = frozenset({"L4", "V5E1", "V6E1"})


def _uuid_to_web_safe_base64(uuid_val: uuid.UUID) -> str:
    uuid_str = str(uuid_val)
    return uuid_str.replace("-", "_") + ("." * (44 - len(uuid_str)))


_RUNTIME_ID = re.compile(r"^[a-z](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_RUNTIME = re.compile(r"^runtimes/([a-z](?:[a-z0-9-]{0,61}[a-z0-9])?)$")
_OPERATION = re.compile(r"^operations/[A-Za-z0-9._~-]+$")
_VARIANTS = {"VARIANT_CPU", "VARIANT_GPU", "VARIANT_TPU"}
_VARIANT_ALIASES = {"CPU": "VARIANT_CPU", "GPU": "VARIANT_GPU", "TPU": "VARIANT_TPU"}
_SHAPES = {"SHAPE_STANDARD", "SHAPE_HIGHMEM"}
_SHAPE_ALIASES = {"STANDARD": "SHAPE_STANDARD", "HIGHMEM": "SHAPE_HIGHMEM"}


class GoogleColabProvider:
    is_mock = False

    def __init__(
        self,
        store,
        owner_id,
        *,
        account_id=None,
        client=None,
        google=None,
        base_url="https://colaboratory.googleapis.com",
        timeout=10.0,
        create_timeout=60.0,
    ):
        self.store, self.owner_id, self.account_id = store, owner_id, account_id
        self.create_timeout = max(float(timeout), float(create_timeout))
        if client is not None:
            self.client = client
        else:
            from .mcp import _make_colab_http_client

            self.client = _make_colab_http_client(timeout)
        self._owns_client = client is None
        self.google = google or GoogleService(store, owner_id, account_id=account_id, client=self.client)
        self.base_url = base_url.rstrip("/")

    def _create_http_timeout(self) -> httpx.Timeout:
        return httpx.Timeout(connect=10.0, read=self.create_timeout, write=15.0, pool=10.0)

    @contextmanager
    def _request_phase(self, phase: str, *, mutating: bool = False):
        prev = getattr(self, "_active_request_meta", (None, False))
        self._active_request_meta = (phase, bool(mutating))
        try:
            yield
        finally:
            self._active_request_meta = prev

    def _access_token(self, phase: str | None) -> str:
        t0 = time.monotonic()
        try:
            return self.google.access_token("colab")
        except GoogleError as exc:
            if phase not in {"assign_get", "assign_post", "v1beta_create"}:
                raise
            elapsed = time.monotonic() - t0
            if isinstance(exc.__cause__, httpx.HTTPError):
                raise ColabError(
                    f"刷新 Google 授权凭据时网络异常（耗时 {elapsed:.1f}s）；请检查网络或稍后重试",
                    error_code="colab_network_error",
                    category="network",
                    phase="auth",
                    timeout_kind="transport",
                    elapsed_seconds=elapsed,
                    remote_state_unknown=False,
                    retryable_before_post=True,
                ) from exc
            raise ColabError(
                str(exc) or "Google 账号授权无效，请重新连接账号",
                error_code="colab_auth_failed",
                category="auth",
                phase="auth",
                elapsed_seconds=elapsed,
                remote_state_unknown=False,
                retryable_before_post=False,
            ) from exc

    def _request(self, method, path, *, params=None, json=None, timeout=None):
        phase, mutating = getattr(self, "_active_request_meta", (None, False))
        token = self._access_token(phase)
        kwargs = {
            "params": params,
            "json": json,
            "headers": {"Authorization": "Bearer " + token, "Accept": "application/json"},
        }
        if timeout is not None:
            kwargs["timeout"] = timeout
        t0 = time.monotonic()
        try:
            response = self.client.request(method, self.base_url + path, **kwargs)
        except httpx.HTTPError as exc:
            raise _classify_httpx_error(
                exc, phase=phase, mutating=mutating, elapsed=time.monotonic() - t0, channel="API "
            ) from exc
        if response.status_code >= 400 and response.status_code != 404:
            raise _classify_http_status(
                response.status_code,
                phase=phase,
                mutating=mutating,
                elapsed=time.monotonic() - t0,
                channel="API ",
            )
        return response

    @staticmethod
    def _json(response):
        try:
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError()
            return payload
        except ValueError as exc:
            raise ColabError("Colab API 返回无效 JSON") from exc

    def _is_cli_tunnel(self, lease: dict | None = None) -> bool:
        if isinstance(lease, dict) and lease.get("tunnel_endpoint"):
            return True
        try:
            cred = self.google._credential("colab", account_id=self.account_id)
        except Exception:
            cred = None
        if isinstance(cred, dict):
            cid = str(cred.get("client_id") or "").strip()
            if cid == _CLI_PUBLIC_CLIENT_ID or cid.startswith("764086051850-"):
                return True
        return False

    def _tunnel_request(self, method: str, path: str, *, params: dict | None = None, headers: dict | None = None, timeout=None):
        phase, mutating = getattr(self, "_active_request_meta", (None, False))
        token = self._access_token(phase)
        req_params = {"authuser": "0"}
        if params:
            req_params.update(params)
        req_headers = {
            "Authorization": "Bearer " + token,
            "Accept": "application/json",
            "X-Colab-Client-Agent": "colab-cli",
        }
        if headers:
            req_headers.update(headers)
        req_kwargs = {"params": req_params, "headers": req_headers}
        if timeout is not None:
            req_kwargs["timeout"] = timeout
        t0 = time.monotonic()
        try:
            response = self.client.request(
                method,
                _TUNNEL_BASE_URL + path,
                **req_kwargs,
            )
        except httpx.HTTPError as exc:
            raise _classify_httpx_error(
                exc, phase=phase, mutating=mutating, elapsed=time.monotonic() - t0, channel="隧道"
            ) from exc
        if response.status_code >= 400 and response.status_code != 404:
            raise _classify_http_status(
                response.status_code,
                phase=phase,
                mutating=mutating,
                elapsed=time.monotonic() - t0,
                channel="隧道",
            )
        return response

    @staticmethod
    def _tunnel_json(response) -> dict:
        text = response.text or ""
        if text.startswith(_XSSI_PREFIX):
            text = text[len(_XSSI_PREFIX) :]
        if not text.strip():
            return {}
        try:
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError()
            return payload
        except ValueError as exc:
            raise ColabError("Colab 隧道返回无效 JSON") from exc

    def _tunnel_assignments(self) -> list[dict]:
        response = self._tunnel_request("GET", "/tun/m/assignments")
        if response.status_code != 200:
            raise ColabError("Colab 隧道分配列表不可用")
        payload = self._tunnel_json(response)
        rows = payload.get("assignments")
        if not isinstance(rows, list) or any(
            not isinstance(row, dict) or not isinstance(row.get("endpoint"), str) or not row["endpoint"]
            for row in rows
        ):
            raise ColabError("Colab 隧道分配列表无效")
        return rows

    def list_specs(self):
        if self._is_cli_tunnel():
            info = self._tunnel_json(self._tunnel_request("GET", "/tun/m/ccu-info"))
            specs = [
                {
                    "key": {"variant": "VARIANT_CPU", "accelerator": "NONE", "shape": "SHAPE_STANDARD"},
                    "eligible": True,
                }
            ]
            for gpu in info.get("eligibleGpus") or []:
                if isinstance(gpu, str) and gpu:
                    specs.append(
                        {
                            "key": {"variant": "VARIANT_GPU", "accelerator": gpu.upper(), "shape": "SHAPE_STANDARD"},
                            "eligible": True,
                        }
                    )
                    if gpu.upper() not in _HIGH_MEM_ONLY_ACCELERATORS:
                        specs.append(
                            {
                                "key": {"variant": "VARIANT_GPU", "accelerator": gpu.upper(), "shape": "SHAPE_HIGHMEM"},
                                "eligible": True,
                            }
                        )
            for tpu in info.get("eligibleTpus") or []:
                if isinstance(tpu, str) and tpu:
                    specs.append(
                        {
                            "key": {"variant": "VARIANT_TPU", "accelerator": tpu.upper(), "shape": "SHAPE_STANDARD"},
                            "eligible": True,
                        }
                    )
            return specs
        return self._json(self._request("GET", "/v1beta/runtimespecs")).get("runtimeSpecs", [])

    def subscription(self):
        if self._is_cli_tunnel():
            response = self._tunnel_request("GET", "/tun/m/ccu-info")
            if response.status_code != 200:
                raise ColabError("Colab 隧道余额查询不可用")
            info = self._tunnel_json(response)
            return {
                "name": "subscription",
                "tier": "SUBSCRIPTION_TIER_PRO" if float(info.get("currentBalance") or 0) > 0 else "SUBSCRIPTION_TIER_NONE",
                "paidComputeUnitsBalance": info.get("currentBalance"),
                "consumptionRateHourly": info.get("consumptionRateHourly", 0),
                "assignmentsCount": info.get("assignmentsCount", 0),
                "eligibleGpus": info.get("eligibleGpus", []),
            }
        return self._json(self._request("GET", "/v1beta/subscription"))

    def list_runtimes(self):
        if self._is_cli_tunnel():
            out = []
            for a in self._tunnel_assignments():
                endpoint = a.get("endpoint")
                if not endpoint:
                    continue
                variant_num = a.get("variant")
                variant = "VARIANT_GPU" if variant_num in (1, "GPU", "VARIANT_GPU") else ("VARIANT_TPU" if variant_num in (2, "TPU", "VARIANT_TPU") else "VARIANT_CPU")
                shape_num = a.get("machineShape")
                shape = "SHAPE_HIGHMEM" if shape_num in (1, "HIGH_RAM", "SHAPE_HIGHMEM") else "SHAPE_STANDARD"
                out.append(
                    {
                        "name": f"runtimes/{endpoint}",
                        "runtimeSpec": {
                            "variant": variant,
                            "accelerator": str(a.get("accelerator") or "NONE"),
                            "shape": shape,
                        },
                        "version": "colab-cli-tunnel",
                    }
                )
            return out
        # Never return Google's raw connectionInfo to routes.
        rows = self._json(self._request("GET", "/v1beta/runtimes")).get("runtimes", [])
        return [
            {"name": r.get("name"), "runtimeSpec": r.get("runtimeSpec"), "version": r.get("version")}
            for r in rows
        ]

    def _lease(self, runtime_id, request_id=None, job_id=None):
        for lease in self.store.list("leases", self.owner_id):
            if (
                lease.get("runtime_id") != runtime_id
                or not lease.get("owner_job_id")
                or not lease.get("request_id")
            ):
                continue
            if request_id and request_id != lease.get("request_id"):
                continue
            if job_id and job_id != lease.get("owner_job_id"):
                continue
            if lease.get("provider", "colab") != "colab":
                continue
            return lease
        raise ColabError("拒绝操作未登记归属的 runtime；不会触碰用户手动运行时")

    def _owned(self, handle):
        if not isinstance(handle, dict) or handle.get("owner_id") != self.owner_id:
            raise ColabError("Colab 租约归属无效")
        runtime = handle.get("runtime_name", "")
        match = _RUNTIME.fullmatch(runtime)
        if not match or handle.get("runtime_id") != match.group(1):
            raise ColabError("Colab runtime 句柄无效")
        lease = self._lease(
            match.group(1), handle.get("request_id"), handle.get("owner_job_id") or handle.get("job_id")
        )
        if handle.get("inspection_lease_id") and (
            handle["inspection_lease_id"] != lease["id"] or lease.get("account_id") != self.account_id
        ):
            raise ColabError("Colab 检查租约绑定已改变")
        return lease

    @staticmethod
    def _spec(spec):
        if not isinstance(spec, dict):
            raise ColabError("runtimeSpec 必须来自当前账户真实可用规格的 key")
        if "key" in spec:
            if spec.get("eligible") is not True:
                raise ColabError("runtimeSpec 必须明确标记 eligible=true")
            key = spec["key"]
        else:
            key = spec
        if not isinstance(key, dict):
            raise ColabError("runtimeSpec 必须来自当前账户真实可用规格的 key")
        key = dict(key)
        key["variant"] = _VARIANT_ALIASES.get(key.get("variant"), key.get("variant"))
        key["shape"] = _SHAPE_ALIASES.get(key.get("shape"), key.get("shape"))
        if (
            set(key) != {"variant", "accelerator", "shape"}
            or key.get("variant") not in _VARIANTS
            or key.get("shape") not in _SHAPES
        ):
            raise ColabError("runtimeSpec 必须来自当前账户真实可用规格的 key")
        if not isinstance(key.get("accelerator"), str) or not key["accelerator"]:
            raise ColabError("runtimeSpec accelerator 无效")
        return dict(key)

    @staticmethod
    def _operation_failure(operation):
        """Summarise a failed allocation Operation by its code and ErrorInfo reason only.

        Google's message and metadata may contain credentials or notebook data.
        """
        error = operation.get("error")
        if not error:
            return {}
        reason = next((d.get("reason") for d in error.get("details") or []
                       if d.get("@type") == "type.googleapis.com/google.rpc.ErrorInfo"), None)
        return {"allocation_failed": True,
                "allocation_error_code": error.get("code", "unknown"),
                "allocation_failure_reason": reason}

    @staticmethod
    def _raise_operation_error(
        operation,
        *,
        phase: str | None = None,
        http_status: int | None = None,
        elapsed_seconds: float | None = None,
    ):
        failure = GoogleColabProvider._operation_failure(operation)
        if failure:
            raise ColabAllocationError(
                failure["allocation_error_code"],
                failure["allocation_failure_reason"],
                phase=phase,
                http_status=http_status,
                elapsed_seconds=elapsed_seconds,
            )

    def _create_tunnel(self, lease: dict, key: dict, request_uuid: uuid.UUID, request_id: str, runtime_id: str) -> dict:
        nbh = _uuid_to_web_safe_base64(request_uuid)
        params: dict[str, str] = {"nbh": nbh}
        variant = key.get("variant")
        accelerator = str(key.get("accelerator") or "NONE").upper()
        shape = key.get("shape")
        if variant == "VARIANT_GPU":
            params["variant"] = "GPU"
            if accelerator and accelerator != "NONE":
                params["accelerator"] = accelerator
        elif variant == "VARIANT_TPU":
            params["variant"] = "TPU"
            if accelerator and accelerator != "NONE":
                params["accelerator"] = accelerator
        if shape == "SHAPE_HIGHMEM" and accelerator not in _HIGH_MEM_ONLY_ACCELERATORS:
            params["shape"] = "hm"

        self.store.update(
            "leases",
            lease["id"],
            {"tunnel_nbh": nbh, "tunnel_assign_params": params, "tunnel_post_sent": False},
            self.owner_id,
        )
        assign_timeout = self._create_http_timeout()
        t_get = time.monotonic()
        with self._request_phase("assign_get", mutating=False):
            first_resp = self._tunnel_request("GET", "/tun/m/assign", params=params, timeout=assign_timeout)
            try:
                first = self._tunnel_json(first_resp)
            except ColabError as exc:
                raise ColabError(
                    str(exc),
                    error_code="colab_invalid_response",
                    category="protocol",
                    phase="assign_get",
                    http_status=getattr(first_resp, "status_code", 200),
                    elapsed_seconds=time.monotonic() - t_get,
                    remote_state_unknown=False,
                    retryable_before_post=True,
                ) from exc
        if "endpoint" in first and isinstance(first.get("runtimeProxyInfo"), dict):
            assigned = first
            elapsed_post = time.monotonic() - t_get
        else:
            xsrf = first.get("token")
            if not isinstance(xsrf, str) or not xsrf:
                raise ColabError(
                    "Colab 隧道分配未返回 XSRF 令牌",
                    error_code="colab_invalid_response",
                    category="protocol",
                    phase="assign_get",
                    http_status=getattr(first_resp, "status_code", 200),
                    elapsed_seconds=time.monotonic() - t_get,
                    remote_state_unknown=False,
                    retryable_before_post=True,
                )
            self.store.update("leases", lease["id"], {"tunnel_post_sent": True}, self.owner_id)
            t_post = time.monotonic()
            with self._request_phase("assign_post", mutating=True):
                post_resp = self._tunnel_request(
                    "POST",
                    "/tun/m/assign",
                    params=params,
                    headers={"X-Goog-Colab-Token": xsrf},
                    timeout=assign_timeout,
                )
                elapsed_post = time.monotonic() - t_post
                try:
                    assigned = self._tunnel_json(post_resp)
                except ColabError as exc:
                    raise ColabError(
                        f"{exc}（耗时 {elapsed_post:.1f}s）；远端结果待确认，正在安全释放租约，完成后可重试",
                        error_code="colab_invalid_response",
                        category="protocol",
                        phase="assign_post",
                        http_status=getattr(post_resp, "status_code", 200),
                        elapsed_seconds=elapsed_post,
                        remote_state_unknown=True,
                        retryable_before_post=False,
                    ) from exc
        endpoint = assigned.get("endpoint")
        proxy = assigned.get("runtimeProxyInfo") or {}
        proxy_url = proxy.get("url")
        proxy_token = proxy.get("token")
        if not (isinstance(endpoint, str) and endpoint and isinstance(proxy_url, str) and proxy_url and isinstance(proxy_token, str) and proxy_token):
            denial = _classify_tunnel_outcome(assigned, elapsed=elapsed_post)
            if denial is not None:
                raise denial
            norm = _normalize_outcome(assigned.get("outcome"))
            raise ColabError(
                f"Colab 隧道分配响应缺少 endpoint 或代理连接信息（耗时 {elapsed_post:.1f}s）；"
                "远端结果待确认，正在安全释放租约，完成后可重试",
                error_code="colab_invalid_response",
                category="protocol",
                phase="assign_post",
                http_status=200,
                elapsed_seconds=elapsed_post,
                outcome=norm,
                remote_state_unknown=True,
                retryable_before_post=False,
            )
        ttl = int(proxy.get("tokenExpiresInSeconds") or 3600)
        op_name = f"operations/tunnel-{runtime_id}"
        self.store.update(
            "leases",
            lease["id"],
            {
                "operation_name": op_name,
                "operation_done": True,
                "last_create_error": None,
                "requested_spec": key,
                "tunnel_endpoint": endpoint,
                "tunnel_nbh": nbh,
                "proxy_url": proxy_url.rstrip("/"),
                "encrypted_proxy_token": encrypt_secret(proxy_token),
                "proxy_token_expires_at": time.time() + max(60, ttl - 60),
            },
            self.owner_id,
        )
        return {
            "runtime_name": "runtimes/" + runtime_id,
            "operation_name": op_name,
            "ready": True,
            "runtime_id": runtime_id,
            "request_id": request_id,
            "owner_id": self.owner_id,
        }

    def create(self, spec, request_id, job_id, runtime_id):
        if not isinstance(runtime_id, str) or not _RUNTIME_ID.fullmatch(runtime_id):
            raise ColabError("runtimeId 必须是字母开头的 RFC1034 标识")
        try:
            request_uuid = uuid.UUID(request_id)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ColabError("Colab requestId 必须是 UUID4") from exc
        if request_uuid.version != 4:
            raise ColabError("Colab requestId 必须是 UUID4")
        lease = self._lease(runtime_id, request_id, job_id)
        key = self._spec(spec)
        try:
            if self._is_cli_tunnel(lease):
                return self._create_tunnel(lease, key, request_uuid, request_id, runtime_id)
            t_post = time.monotonic()
            with self._request_phase("v1beta_create", mutating=True):
                response = self._request(
                    "POST",
                    "/v1beta/runtimes",
                    params={"requestId": request_id, "runtimeId": runtime_id},
                    json={"runtimeSpec": key},
                    timeout=self._create_http_timeout(),
                )
                elapsed_post = time.monotonic() - t_post
                try:
                    operation = self._json(response)
                except ColabError as exc:
                    raise ColabError(
                        f"{exc}（耗时 {elapsed_post:.1f}s）；远端结果待确认，正在安全释放租约，完成后可重试",
                        error_code="colab_invalid_response",
                        category="protocol",
                        phase="v1beta_create",
                        http_status=getattr(response, "status_code", 200),
                        elapsed_seconds=elapsed_post,
                        remote_state_unknown=True,
                    ) from exc
            name = operation.get("name")
            if not isinstance(name, str) or not _OPERATION.fullmatch(name):
                raise ColabError(
                    "创建响应缺少合法 Operation，保留既定 runtimeId 待协调",
                    error_code="colab_invalid_response",
                    category="protocol",
                    phase="v1beta_create",
                    http_status=getattr(response, "status_code", 200),
                    elapsed_seconds=elapsed_post,
                    remote_state_unknown=True,
                )
            self.store.update(
                "leases",
                lease["id"],
                {
                    "operation_name": name,
                    "operation_done": operation.get("done") is True,
                    "last_create_error": None,
                    "requested_spec": key,
                },
                self.owner_id,
            )
            self._raise_operation_error(
                operation,
                phase="v1beta_create",
                http_status=getattr(response, "status_code", 200),
                elapsed_seconds=elapsed_post,
            )
            return {
                "runtime_name": "runtimes/" + runtime_id,
                "operation_name": name,
                "ready": operation.get("done") is True,
                "runtime_id": runtime_id,
                "request_id": request_id,
                "owner_id": self.owner_id,
            }
        except ColabError as exc:
            logger.warning(
                "colab_create_failed job=%s lease=%s phase=%s code=%s category=%s http_status=%s timeout=%s outcome=%s elapsed_s=%s unknown=%s retryable=%s",
                job_id,
                lease.get("id"),
                exc.phase,
                exc.error_code,
                exc.category,
                exc.http_status,
                exc.timeout_kind,
                exc.outcome,
                exc.elapsed_seconds,
                exc.remote_state_unknown,
                exc.retryable_before_post,
            )
            raise

    def _operation(self, handle, lease):
        name = handle.get("operation_name") or lease.get("operation_name")
        if not name:
            return None
        if not _OPERATION.fullmatch(name):
            raise ColabError("Operation 名称无效")
        response = self._request("GET", "/v1/" + name)
        if response.status_code == 404:
            # Missing operation is not proof of runtime deletion; inspect runtime below.
            return None
        return self._json(response)

    def _sync_tunnel_lease(self, lease: dict) -> dict | None:
        endpoint = lease.get("tunnel_endpoint")
        assignments = self._tunnel_assignments()
        matched = next((a for a in assignments if a.get("endpoint") == endpoint), None) if endpoint else None
        if matched is None:
            return None
        proxy = matched.get("runtimeProxyInfo") or {}
        patch: dict = {}
        if isinstance(proxy.get("url"), str) and proxy["url"]:
            patch["proxy_url"] = proxy["url"].rstrip("/")
        if isinstance(proxy.get("token"), str) and proxy["token"]:
            patch["encrypted_proxy_token"] = encrypt_secret(proxy["token"])
            ttl = int(proxy.get("tokenExpiresInSeconds") or 3600)
            patch["proxy_token_expires_at"] = time.time() + max(60, ttl - 60)
        if patch:
            lease = self.store.update("leases", lease["id"], patch, self.owner_id)
        return matched

    def _reconcile_tunnel_intent(self, lease):
        """Resolve a lost assign response with read-only GETs, never another POST."""
        if lease.get("tunnel_endpoint") or lease.get("operation_done") or not lease.get("create_attempts"):
            return lease
        # Do not interpret an empty list while the original POST may still run.
        if time.time() - float(lease.get("create_started") or time.time()) < TUNNEL_RECONCILE_WINDOW_SECONDS:
            return lease
        params = lease.get("tunnel_assign_params")
        if not params:
            key = (lease.get("selected_spec") or {}).get("key") or lease.get("requested_spec") or {}
            params = {"nbh": _uuid_to_web_safe_base64(uuid.UUID(lease["request_id"]))}
            variant = key.get("variant")
            if variant in {"VARIANT_GPU", "VARIANT_TPU"}:
                params.update(variant=variant.removeprefix("VARIANT_"), accelerator=key.get("accelerator", "NONE"))
            if key.get("shape") == "SHAPE_HIGHMEM" and key.get("accelerator") not in _HIGH_MEM_ONLY_ACCELERATORS:
                params["shape"] = "hm"
        status = self._tunnel_json(self._tunnel_request("GET", "/tun/m/assign", params=params))
        endpoint = status.get("endpoint")
        if isinstance(endpoint, str) and endpoint:
            return self.store.update("leases", lease["id"], {
                "tunnel_endpoint": endpoint, "tunnel_nbh": params["nbh"], "operation_done": True,
            }, self.owner_id)
        # A valid unassigned response for this exact notebook plus absence from
        # assignments is required. Empty/malformed replies are not confirmation.
        if not isinstance(status.get("token"), str) or not status["token"] or status.get("nbh") != params["nbh"]:
            return lease
        assignments = self._tunnel_assignments()
        matched = next((a for a in assignments if a.get("nbh") == params["nbh"]), None)
        if matched:
            if not matched.get("endpoint"):
                return lease
            return self.store.update("leases", lease["id"], {
                "tunnel_endpoint": matched["endpoint"], "tunnel_nbh": params["nbh"], "operation_done": True,
            }, self.owner_id)
        return self.store.update("leases", lease["id"], {
            "operation_done": True, "tunnel_nbh": params["nbh"],
            "allocation_reconciliation": "exact_nbh_unassigned_and_absent",
        }, self.owner_id)

    @staticmethod
    def _assignment_spec(assignment):
        """Only report provider evidence; an assignment without hardware fields is unknown."""
        if not all(field in assignment for field in ("variant", "accelerator", "machineShape")):
            return None
        variant = assignment["variant"]
        variant = {0: "VARIANT_CPU", 1: "VARIANT_GPU", 2: "VARIANT_TPU"}.get(variant, variant)
        variant = _VARIANT_ALIASES.get(variant, variant)
        accelerator = str(assignment["accelerator"]).upper()
        if variant not in _VARIANTS or not accelerator or assignment["machineShape"] not in (0, 1, "STANDARD", "SHAPE_STANDARD", "HIGH_RAM", "HIGHMEM", "SHAPE_HIGHMEM"):
            return None
        shape = "SHAPE_HIGHMEM" if assignment["machineShape"] in (1, "HIGH_RAM", "HIGHMEM", "SHAPE_HIGHMEM") else "SHAPE_STANDARD"
        # CLI has no separate high-RAM choice for these accelerators.
        if accelerator in _HIGH_MEM_ONLY_ACCELERATORS:
            shape = "SHAPE_STANDARD"
        return {"variant": variant, "accelerator": accelerator, "shape": shape}

    def _persist_lease_patch(self, lease: dict, patch: dict) -> dict:
        store = getattr(self, "store", None)
        lease_id = lease.get("id")
        if store is not None and lease_id and hasattr(store, "update"):
            return store.update("leases", lease_id, patch, getattr(self, "owner_id", None))
        return {**lease, **patch}

    def inspect(self, handle):
        lease = self._owned(handle)
        if self._is_cli_tunnel(lease):
            reclaim_check = bool(handle.get("inspection_lease_id"))
            if not reclaim_check:
                lease = self._reconcile_tunnel_intent(lease)
            endpoint = lease.get("tunnel_endpoint")
            if not endpoint:
                unknown = reclaim_check or (
                    int(lease.get("create_attempts", 0)) > 0 and not lease.get("operation_done")
                )
                return {"ready": False, "deleted": not unknown, "allocation_unknown": unknown}
            if reclaim_check:
                matched = next((row for row in self._tunnel_assignments() if row["endpoint"] == endpoint), None)
            else:
                matched = self._sync_tunnel_lease(lease)
            if matched is None:
                return {"ready": False, "deleted": True}
            return {
                "ready": True,
                "deleted": False,
                "runtime_name": handle["runtime_name"],
                "runtime_spec": self._assignment_spec(matched),
                "version": "colab-cli-tunnel",
            }
        operation = self._operation(handle, lease)
        if operation is not None and operation.get("done") is not True:
            return {"ready": False, "deleted": False, "operation_pending": True}
        if operation is not None and operation.get("done") is True and not lease.get("operation_done"):
            lease = self._persist_lease_patch(lease, {"operation_done": True})
        # A failed Operation still needs the runtime GET: a runtime that exists must be cleaned up.
        failure = self._operation_failure(operation) if operation is not None else {}
        response = self._request("GET", "/v1beta/" + handle["runtime_name"])
        if response.status_code == 404:
            # No Operation + a previously attempted create whose response was lost is ambiguous.
            unknown = (
                operation is None
                and int(lease.get("create_attempts", 0)) > 0
                and not lease.get("operation_done")
                and lease.get("allocation_confirmed") is not True
            )
            return {"ready": False, "deleted": not unknown, "allocation_unknown": unknown, **failure}
        if failure:
            return {"ready": False, "deleted": False, **failure}
        runtime = self._json(response)
        ready = bool(runtime.get("connectionInfo"))
        if ready and (
            not lease.get("operation_done")
            or lease.get("allocation_confirmed") is not True
            or lease.get("last_create_error")
        ):
            lease = self._persist_lease_patch(
                lease,
                {"operation_done": True, "allocation_confirmed": True, "last_create_error": None},
            )
        return {
            "ready": ready,
            "deleted": False,
            "runtime_name": handle["runtime_name"],
            "runtime_spec": runtime.get("runtimeSpec"),
            "version": runtime.get("version"),
        }

    def connection_info(self, handle):
        lease = self._owned(handle)
        if self._is_cli_tunnel(lease):
            self._sync_tunnel_lease(lease)
            fresh = self.store.get("leases", lease["id"], self.owner_id) or lease
            url = fresh.get("proxy_url")
            enc_tok = fresh.get("encrypted_proxy_token")
            if not url or not enc_tok:
                raise ColabError("Colab connectionInfo 未就绪")
            return {
                "url": url,
                "token": decrypt_secret(enc_tok),
                "expire_time": fresh.get("proxy_token_expires_at"),
            }
        response = self._request("GET", "/v1beta/" + handle["runtime_name"])
        info = self._json(response).get("connectionInfo", {})
        if not info.get("url") or not info.get("token"):
            raise ColabError("Colab connectionInfo 未就绪")
        return {"url": info["url"], "token": info["token"], "expire_time": info.get("expireTime")}

    def _resolve_proxy_credentials(self, lease: dict, *, force_refresh: bool = False) -> tuple[str, str, dict]:
        url = lease.get("proxy_url")
        enc_tok = lease.get("encrypted_proxy_token")
        exp = float(lease.get("proxy_token_expires_at") or 0)
        if (
            not force_refresh
            and isinstance(url, str)
            and url
            and isinstance(enc_tok, str)
            and enc_tok
            and (exp == 0 or exp > time.time() + 180)
        ):
            return url.rstrip("/"), decrypt_secret(enc_tok), {}
        if self._is_cli_tunnel(lease):
            endpoint = lease.get("tunnel_endpoint")
            if not isinstance(endpoint, str) or not endpoint:
                raise ColabError("Colab 隧道缺少 endpoint")
            matched = next((a for a in self._tunnel_assignments() if a.get("endpoint") == endpoint), None)
            if matched is None:
                raise ColabError("Colab 隧道运行时不存在")
            proxy = matched.get("runtimeProxyInfo") or {}
            new_url = proxy.get("url")
            raw_tok = proxy.get("token")
            patch: dict = {}
            if isinstance(new_url, str) and new_url:
                url = new_url.rstrip("/")
                patch["proxy_url"] = url
            if isinstance(raw_tok, str) and raw_tok:
                ttl = int(proxy.get("tokenExpiresInSeconds") or 3600)
                patch["encrypted_proxy_token"] = encrypt_secret(raw_tok)
                patch["proxy_token_expires_at"] = time.time() + max(60, ttl - 60)
                if not isinstance(url, str) or not url:
                    raise ColabError("Colab 隧道缺少代理地址")
                return url.rstrip("/"), raw_tok, patch
            if isinstance(url, str) and url and isinstance(enc_tok, str) and enc_tok:
                return url.rstrip("/"), decrypt_secret(enc_tok), patch
            raise ColabError("Colab 隧道缺少代理凭据")
        handle = lease.get("handle")
        runtime_name = handle.get("runtime_name") if isinstance(handle, dict) else None
        if not isinstance(runtime_name, str) or not runtime_name:
            raise ColabError("Colab 运行时缺少 runtime_name")
        info = self._json(self._request("GET", "/v1beta/" + runtime_name)).get("connectionInfo") or {}
        if not info.get("url") or not info.get("token"):
            raise ColabError("Colab connectionInfo 未就绪")
        return str(info["url"]).rstrip("/"), str(info["token"]), {}

    def keep_alive(
        self,
        lease: dict,
        *,
        run_kernel_exec: bool = False,
        tfe_only: bool = False,
        ws_connect_fn=None,
    ) -> dict:
        """Send dual-layer Colab keep-alive without mutating DB state directly."""
        is_cli = self._is_cli_tunnel(lease)
        tfe_status = "skipped"
        if is_cli and isinstance(lease.get("tunnel_endpoint"), str) and lease["tunnel_endpoint"]:
            endpoint = lease["tunnel_endpoint"]
            try:
                token = self.google.access_token("colab")
                tfe_headers = {
                    "Authorization": "Bearer " + token,
                    "Accept": "application/json, text/plain, */*",
                    "X-Colab-Client-Agent": "colab-cli",
                    "X-Colab-Tunnel": "Google",
                }
                tfe_timeout = httpx.Timeout(connect=6.0, read=4.0, write=6.0, pool=6.0)
                with self.client.stream(
                    "GET",
                    f"{_TUNNEL_BASE_URL}/tun/m/{endpoint}/keep-alive/",
                    params={"authuser": "0"},
                    headers=tfe_headers,
                    timeout=tfe_timeout,
                ) as resp:
                    if 200 <= resp.status_code < 300:
                        cl = resp.headers.get("Content-Length")
                        if cl and cl.isdigit() and int(cl) <= 4096:
                            resp.read()
                        tfe_status = f"ok:{resp.status_code}"
                    else:
                        tfe_status = f"http:{resp.status_code}"
            except httpx.ReadTimeout:
                # TFE keep-alive streams may hold the response open after updating LastActiveTime.
                tfe_status = "read_timeout_ack"
            except Exception as exc:
                tfe_status = f"err:{type(exc).__name__}"

        if tfe_only:
            tfe_ok = (tfe_status.startswith("ok:") or tfe_status == "read_timeout_ack") if is_cli else False
            return {
                "ok": bool(tfe_ok),
                "tfe": tfe_status,
                "proxy": "skipped",
                "kernel_exec": "skipped",
                "lease_patch": {},
            }

        proxy_status = "skipped"
        kernel_exec_status = "skipped"
        lease_patch: dict = {}
        raw_kernel_id = lease.get("kernel_id")
        kernel_id = raw_kernel_id if isinstance(raw_kernel_id, str) and raw_kernel_id else None
        proxy_url = proxy_token = None
        try:
            proxy_url, proxy_token, patch = self._resolve_proxy_credentials(lease)
            lease_patch.update(patch)
            for attempt in range(2):
                auth_params = {"colab-runtime-proxy-token": proxy_token, "authuser": "0"}
                p_headers = {
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "X-Colab-Client-Agent": "colab-cli",
                    "X-Colab-Runtime-Proxy-Token": proxy_token,
                }
                target = f"{proxy_url}/api/kernels/{kernel_id}" if kernel_id else f"{proxy_url}/api/kernels"
                r = self.client.get(target, headers=p_headers, params=auth_params, timeout=8.0)
                if r.status_code in {401, 403} and attempt == 0:
                    proxy_url, proxy_token, patch = self._resolve_proxy_credentials(lease, force_refresh=True)
                    lease_patch.update(patch)
                    continue
                if kernel_id and r.status_code == 404:
                    kernel_id = None
                    r = self.client.get(f"{proxy_url}/api/kernels", headers=p_headers, params=auth_params, timeout=8.0)
                if r.status_code == 200:
                    if not kernel_id:
                        existing = r.json()
                        if isinstance(existing, list) and existing and isinstance(existing[0].get("id"), str) and existing[0]["id"]:
                            kernel_id = existing[0]["id"]
                        else:
                            rc = self.client.post(
                                f"{proxy_url}/api/kernels",
                                json={"name": "python3"},
                                headers=p_headers,
                                params=auth_params,
                                timeout=10.0,
                            )
                            if rc.status_code < 400:
                                created = rc.json().get("id")
                                if isinstance(created, str) and created:
                                    kernel_id = created
                        if kernel_id:
                            lease_patch["kernel_id"] = kernel_id
                    proxy_status = "ok" if kernel_id else "no_kernel"
                else:
                    proxy_status = f"http:{r.status_code}"
                break
        except Exception as exc:
            proxy_status = f"err:{type(exc).__name__}"

        if run_kernel_exec and proxy_status == "ok" and kernel_id and proxy_url and proxy_token:
            try:
                from .mcp import MCPGPUExecutor, _connect_ipv4_socket

                custom_ws = ws_connect_fn is not None
                if ws_connect_fn is None:
                    from websockets.sync.client import connect as ws_connect_fn
                ws_base = re.sub(r"^http", "ws", proxy_url, count=1)
                session_id = uuid.uuid4().hex
                msg_id = uuid.uuid4().hex
                query = urlencode({
                    "colab-runtime-proxy-token": proxy_token,
                    "authuser": "0",
                    "session_id": session_id,
                })
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
                        "code": "1 + 1",
                        "silent": False,
                        "store_history": False,
                        "user_expressions": {},
                        "allow_stdin": False,
                        "stop_on_error": True,
                    },
                    "channel": "shell",
                }
                reply_seen = False
                idle_seen = False
                has_error = False
                deadline = time.monotonic() + 8.0
                ws_kwargs = {
                    "additional_headers": {
                        "X-Colab-Client-Agent": "colab-cli",
                        "X-Colab-Runtime-Proxy-Token": proxy_token,
                    },
                    "open_timeout": 6.0,
                    "close_timeout": 3.0,
                    "max_size": 4 * 1024 * 1024,
                }
                if not custom_ws:
                    sock = _connect_ipv4_socket(ws_url, 6.0)
                    if sock is not None:
                        ws_kwargs["sock"] = sock
                with ws_connect_fn(ws_url, **ws_kwargs) as ws:
                    ws.send(MCPGPUExecutor._ws_pack_default(req_msg))
                    while time.monotonic() < deadline:
                        remaining = max(0.3, deadline - time.monotonic())
                        raw = ws.recv(timeout=remaining)
                        msg = MCPGPUExecutor._ws_unpack_default(raw)
                        if (msg.get("parent_header") or {}).get("msg_id") != msg_id:
                            continue
                        mtype = (msg.get("header") or {}).get("msg_type") or msg.get("msg_type")
                        content = msg.get("content") or {}
                        if mtype == "error":
                            has_error = True
                        elif mtype == "execute_reply":
                            reply_seen = True
                            if content.get("status") == "error":
                                has_error = True
                        elif mtype == "status" and content.get("execution_state") == "idle":
                            idle_seen = True
                        if reply_seen and idle_seen:
                            break
                kernel_exec_status = "ok" if (reply_seen and idle_seen and not has_error) else "incomplete"
            except Exception as exc:
                kernel_exec_status = f"err:{type(exc).__name__}"

        tfe_ok = (tfe_status.startswith("ok:") or tfe_status == "read_timeout_ack") if is_cli else True
        proxy_ok = proxy_status == "ok"
        exec_ok = (not run_kernel_exec) or kernel_exec_status == "ok"
        return {
            "ok": bool(tfe_ok and proxy_ok and exec_ok),
            "tfe": tfe_status,
            "proxy": proxy_status,
            "kernel_exec": kernel_exec_status,
            "lease_patch": lease_patch,
        }

    def delete(self, handle):
        lease = self._owned(handle)
        if self._is_cli_tunnel(lease):
            lease = self._reconcile_tunnel_intent(lease)
            endpoint = lease.get("tunnel_endpoint")
            if not endpoint:
                unknown = int(lease.get("create_attempts", 0)) > 0 and not lease.get("operation_done")
                return not unknown
            matched = self._sync_tunnel_lease(lease)
            if matched is None:
                return True
            unassign_get = self._tunnel_request("GET", f"/tun/m/unassign/{endpoint}")
            if unassign_get.status_code == 404:
                return True
            xsrf = self._tunnel_json(unassign_get).get("token")
            if not isinstance(xsrf, str) or not xsrf:
                raise ColabError("Colab 隧道释放缺少 XSRF 令牌")
            self._tunnel_request(
                "POST",
                f"/tun/m/unassign/{endpoint}",
                headers={"X-Goog-Colab-Token": xsrf},
            )
            return self._sync_tunnel_lease(lease) is None
        operation = self._operation(handle, lease)
        if operation is not None and operation.get("done") is not True:
            # Deleting a missing runtime during a still-pending create is not release confirmation.
            return False
        if operation is not None and operation.get("done") is True and not lease.get("operation_done"):
            lease = self._persist_lease_patch(lease, {"operation_done": True})
        # A failed completed Operation may still have allocated transient state; GET/DELETE regardless.
        current = self._request("GET", "/v1beta/" + handle["runtime_name"])
        if current.status_code == 404:
            if (
                operation is None
                and int(lease.get("create_attempts", 0)) > 0
                and not lease.get("operation_done")
                and lease.get("allocation_confirmed") is not True
            ):
                # Reconcile lost create response by looking for only our deterministic runtime metadata.
                response = self._request("GET", "/v1/operations")
                operations = self._json(response).get("operations", []) if response.status_code != 404 else []
                matches = [
                    o
                    for o in operations
                    if (o.get("metadata") or {}).get("runtime") == handle["runtime_name"]
                ]
                if not matches or any(o.get("done") is not True for o in matches):
                    return False
            return True
        self._request("DELETE", "/v1beta/" + handle["runtime_name"], params={"allowMissing": "true"})
        return self._request("GET", "/v1beta/" + handle["runtime_name"]).status_code == 404

    def close(self):
        if self._owns_client:
            self.client.close()
