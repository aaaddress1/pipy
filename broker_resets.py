"""Saved resets over the Python extension or the original Oh My Pi broker protocol."""

import asyncio
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from auth_broker import AuthBrokerError, block_signature
from local_oauth import fingerprint
from quota_reset import PROVIDERS, ResetExecutor, account_key
from quota_service import now_ms
from reset_fence import marker_path


class NativeBrokerResets(ResetExecutor):
    def __init__(self, broker):
        from local_credentials import agent_db_path

        super().__init__(agent_db_path(broker.profile).resolve())
        self.broker = broker

    def record(self, row):
        identifier, provider = row["broker_id"], row["provider"]
        access = row["tokens"]["access_token"]
        return {"id": identifier, "provider": provider, "identityKey": row["identity"],
                "credential": {"type": "oauth", "access": access, "expires": row["expires"] * 1000, **row["account"]},
                "credentialRevision": fingerprint(access),
                "blocks": list(self.broker.active_blocks(row, self.broker.server_time()).values())}

    async def records(self, provider):
        async with self.broker.snapshot_lock:
            for _ in range(3):
                if await self.broker.fetch():
                    break
            else:
                raise AuthBrokerError("Auth Broker credential snapshot changed during reset verification.")
            return [self.record(row) for row in self.broker.rows.values() if row["kind"] == "oauth"
                    and row["provider"] in PROVIDERS.values() and (provider is None or provider == row["provider"])]

    async def prepare(self, record):
        if record["credential"]["expires"] > now_ms() + self.broker.refresh_skew * 1000:
            return record
        observed = self.broker.rows.get(record["id"])
        if (not observed or observed["kind"] != "oauth" or observed["provider"] != record["provider"]
            or self.record(observed)["credentialRevision"] != record["credentialRevision"]):
            return None
        try:
            await self.broker.refresh(observed, False)
        except AuthBrokerError:
            pass
        current = self.broker.rows.get(record["id"])
        if not current or current["kind"] != "oauth" or current["provider"] != record["provider"]:
            return None
        row = self.record(current)
        if (row["credential"]["expires"] > now_ms() and row["identityKey"] == record["identityKey"]
            and account_key(row) == account_key(record)):
            return row
        return None

    def pending_path(self, key):
        return Path(f"{marker_path(self.fence_root, key)}.pending.json")

    def pending_reset(self, key):
        try:
            with self.pending_path(key).open("rb") as file:
                raw = file.read(16_384)
        except FileNotFoundError:
            return None
        try:
            pending = json.loads(raw)
            if (not isinstance(pending, dict) or not isinstance(pending.get("creditId"), str)
                or not isinstance(pending.get("requestId"), str) or type(pending.get("remainingCount")) is not int
                or type(pending.get("expiresAt")) is not int or "program" not in pending):
                raise ValueError()
        except (ValueError, UnicodeError):
            raise OSError("Invalid saved-reset receipt. The previous redemption needs confirmation.") from None
        return pending if pending["expiresAt"] > now_ms() else None

    def save_pending_reset(self, key, pending):
        path = self.pending_path(key)
        if pending is None:
            path.unlink(missing_ok=True)
            return
        temporary = None
        try:
            with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=".reset-", delete=False) as file:
                temporary = Path(file.name)
                json.dump({**pending, "remainingCount": int(pending["remainingCount"]),
                           "expiresAt": now_ms() + 86_400_000}, file, separators=(",", ":"))
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, path)
        finally:
            if temporary:
                temporary.unlink(missing_ok=True)

    async def after_reset(self, record, scopes=(), preserve=()):
        provider, identifier = record["provider"], record["id"]

        async def current():
            fresh = await self.get(provider, identifier)
            if fresh and fresh["identityKey"] == record["identityKey"] and (
                fresh["credentialRevision"] == record["credentialRevision"] and account_key(fresh) == account_key(record)
            ):
                return fresh
            return None

        try:
            if scopes and await current():
                for scope, deadline in preserve:
                    await self.broker.write_block(identifier, provider, {
                        "providerKey": provider + ":oauth", "blockScope": scope, "blockedUntilMs": deadline,
                    })
                for observed in record["blocks"]:
                    key, scope = observed["providerKey"], observed["blockScope"]
                    if scope not in scopes:
                        continue
                    # The original broker only exposes unconditional slot deletes. Re-read
                    # the slot first and leave auth, policy and newer known blocks alone.
                    row = await current()
                    if not row:
                        break
                    latest = next((block for block in row["blocks"] if block["providerKey"] == key
                                   and block["blockScope"] == scope), None)
                    if latest and block_signature(latest) == block_signature(observed):
                        await self.broker.delete_block(identifier, (key, scope), observed)
        finally:
            await self.broker.invalidate_reset(provider, identifier)
            try:
                await self.broker.request("POST", "/v1/usage/stale", params={"provider": provider})
            finally:
                await self.broker.invalidate_reset(provider, identifier)


class BrokerResets:
    def __init__(self, broker):
        self.broker = broker
        self.mode = None
        self.native = None
        self.probe = asyncio.Lock()

    async def list(self, provider=None, identifier=None, *, evidence=False, active=None, timeout=10):
        params = {}
        for key, value in (("provider", provider), ("credentialId", identifier), ("active", active)):
            if value is not None:
                params[key] = value
        if evidence:
            params["evidence"] = "1"
        if self.mode is None:
            async with self.probe:
                if self.mode is None:
                    try:
                        rows = await self.extension_list(params, timeout)
                    except AuthBrokerError as error:
                        if error.status not in (404, 405):
                            raise
                        self.native = NativeBrokerResets(self.broker)
                        self.mode = "native"
                    else:
                        self.mode = "extension"
                        return rows
        if self.mode == "extension":
            return await self.extension_list(params, timeout)
        allowed = (lambda row: row["credentialId"] == identifier) if identifier is not None else None
        return await self.native.list(provider, allowed, evidence=evidence, active=active)

    async def extension_list(self, params, timeout):
        response = await self.broker.request("GET", "/v1/usage/resets", params=params, timeout=timeout)
        rows = self.broker.payload(response).get("accounts")
        if not isinstance(rows, list):
            raise AuthBrokerError("Auth Broker returned an invalid reset inventory.")
        return rows

    async def redeem(self, provider, identifier, credit_id=None, expected_count=None, identity_key=None):
        if self.native is not None:
            return await self.native.redeem(provider, identifier, credit_id, expected_count, identity_key)
        if self.mode != "extension":
            raise AuthBrokerError("Auth Broker reset inventory must be checked before redemption.")
        response = await self.broker.request("POST", "/v1/usage/resets/redeem", timeout=120, json={
            "provider": provider, "credentialId": identifier, "creditId": credit_id, "expectedCount": expected_count,
            "identityKey": identity_key,
        })
        return self.broker.payload(response)

    async def close(self):
        if self.native is not None:
            await self.native.close()
