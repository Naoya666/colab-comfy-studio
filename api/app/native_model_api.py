"""Authenticated Studio control calls for in-runtime model preparation."""
from __future__ import annotations

import asyncio

import httpx

from .common import APIError
from .gpu import native
from .gpu.accounts import AccountService
from .gpu.google import GoogleError, GoogleService
from .gpu.model_library import ModelLibrary
from .security import decrypt_secret

PATH = "/studio_native/models"


def runtime_context(repo, lease, *, credentials=True):
    owner = lease["owner_id"]
    lib = ModelLibrary(repo, owner)
    accounts = AccountService(repo, owner)
    default = accounts.default_account_id(owner)
    library_account = accounts.resolve_drive_account_id(lib.storage_settings().get("account_id")) or default
    mount_account = lease.get("native_drive_account_id")
    if not mount_account:
        job = repo.get("jobs", lease.get("native_bootstrap_job_id") or lease.get("owner_job_id"), owner) or {}
        mount_account = accounts.resolve_drive_account_id(job.get("drive_account_id") or job.get("account_id")) or default
    allowed = {library_account, mount_account}
    catalog = [row for row in native.local_catalog(repo, owner) if row["account"] in allowed]
    if not credentials:
        return {"catalog": catalog}
    drive_auth = {}
    for account in allowed:
        with httpx.Client(trust_env=False) as client:
            # The historical default account may use the pre-account credential slot.
            for slot in (account, None) if account == default else (account,):
                google = GoogleService(repo, owner, account_id=slot, client=client)
                cred = google._credential("drive", account_id=slot)
                if not cred or not cred.get("encrypted_refresh_token"):
                    continue
                try:
                    if cred.get("client_id") and cred.get("encrypted_client_secret"):
                        client_id = cred["client_id"]
                        secret = decrypt_secret(cred["encrypted_client_secret"])
                    else:
                        client_id, secret, _ = google._config()
                    drive_auth[account] = {"client_id": client_id, "client_secret": secret,
                                           "refresh_token": decrypt_secret(cred["encrypted_refresh_token"])}
                except (GoogleError, ValueError):
                    pass
                break
    return {"catalog": catalog, "tokens": lib.tokens(), "drive_auth": drive_auth}


def headers(base, auth, lease):
    return {**auth, "Origin": base, "X-Studio-Model-Channel": native.model_channel(lease["id"])}


async def rpc(client, method, base, auth, suffix, *, body=None):
    try:
        request = client.build_request(method, base + PATH + suffix, headers=auth, json=body, timeout=15)
        request.headers.pop("Cookie", None)
        response = await client.send(request)
        if response.status_code == 404 and method == "GET":
            raise APIError("NATIVE_MODEL_PREPARATION_MISSING", "准备记录已过期或实例已切换，请重新检查", 404)
        if response.status_code == 422:
            raise APIError("NATIVE_MODEL_INPUT", response.json()["error"]["message"], 422)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("Invalid model preparation response")
        return data
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        raise APIError("NATIVE_MODEL_UPSTREAM", "无法连接 Colab 模型准备服务，请确认当前实例就绪", 502) from exc


async def start(client, base, auth, repo, lease, workflow, dependencies, *, visual):
    context = await asyncio.to_thread(runtime_context, repo, lease)
    return await rpc(client, "POST", base, auth, "/prepare",
                     body={**context, "workflow": workflow, "dependencies": dependencies, "visual": visual})
