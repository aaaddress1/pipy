# /// script
# requires-python = ">=3.11"
# dependencies = ["aiohttp>=3.9,<4", "httpx2", "cryptography", "xxhash"]
# ///
"""Host the Oh My Pi Auth Broker protocol on a native agent.db."""

import argparse
import asyncio
import hmac
import json
import logging
import math
import os
import re
import secrets
import sqlite3
from pathlib import Path
from time import time
from urllib.parse import urlsplit

from aiohttp import web

from auth_broker import AuthBrokerError
from auth_broker_cache import REMOTE_REFRESH, config_root, number
from local_credentials import AgentCredentialDB, CODEX_KEY, OAUTH_METADATA, agent_db_path
from local_oauth import LocalOAuth, can_refresh
from observed_usage import COUNTERS
from quota_reset import PROVIDERS as RESET_PROVIDERS, ResetService
from quota_service import QuotaService


CAPABILITIES = "OMP-Auth-Broker-Capabilities"
METER_CAPABILITY = "codex-meter-block-scopes"
INTERVAL_MS, SKEW_MS = 60_000, 300_000
SAFE_INTEGER = 2**53 - 1
logger = logging.getLogger(__name__)


def now_ms():
    return int(time() * 1000)


def wire_entry(record):
    return {key: record[key] for key in ("id", "provider", "credential", "identityKey")}


def meter_scopes(request):
    return METER_CAPABILITY in (part.strip() for part in request.headers.get(CAPABILITIES, "").split(","))


def project_blocks(blocks, canonical, now):
    active = [block for block in blocks if block["blockedUntilMs"] > now]
    if canonical:
        return active
    scopes = {"chat", "spark", "shared"}
    meters = [block for block in active if block["providerKey"] == CODEX_KEY and block["blockScope"] in scopes]
    if not meters:
        return active
    shared = {"providerKey": CODEX_KEY, "blockScope": "shared",
              "blockedUntilMs": max(block["blockedUntilMs"] for block in meters),
              "updatedAtMs": max(block["updatedAtMs"] for block in meters)}
    return sorted([block for block in active if block not in meters] + [shared],
                  key=lambda block: (block["providerKey"], block["blockScope"]))


def schedule(enabled, next_sweep, now):
    return {"enabled": enabled, "intervalMs": INTERVAL_MS if enabled else 0, "skewMs": SKEW_MS if enabled else 0,
            "nextSweepInMs": max(0, next_sweep - now) if enabled else SAFE_INTEGER}


def rotates_in(record, enabled, next_sweep, now):
    credential = record["credential"]
    if not enabled or credential["type"] != "oauth" or not can_refresh(record["provider"], credential):
        return None
    eligible = max(now, credential["expires"] - SKEW_MS)
    rounds = max(0, math.ceil((eligible - next_sweep) / INTERVAL_MS))
    return max(0, next_sweep + rounds * INTERVAL_MS - now)


def tag_headers(generation):
    return {"ETag": f'"{generation}"', "Cache-Control": "no-store", "Vary": CAPABILITIES}


def valid_string(value):
    return isinstance(value, str) and bool(value.strip())


def body_error(message="Invalid request body."):
    return web.HTTPBadRequest(text=json.dumps({"error": message}), content_type="application/json")


async def payload(request, *, optional=False):
    try:
        raw = await request.read()
        value = {} if optional and not raw else json.loads(raw)
    except (ValueError, UnicodeError, RecursionError):
        raise body_error() from None
    if not isinstance(value, dict):
        raise body_error()
    return value


def credential_payload(value):
    if not isinstance(value, dict):
        raise body_error()
    kind = value.get("type")
    if kind == "api_key" and valid_string(value.get("key")):
        if value.keys() <= {"type", "key", "source"} and value.get("source", "login") == "login":
            return value
    if kind == "oauth" and valid_string(value.get("access")) and number(value.get("expires")):
        refresh = value.get("refresh")
        if isinstance(refresh, str) and refresh != REMOTE_REFRESH and abs(value["expires"]) <= SAFE_INTEGER:
            strings = (*OAUTH_METADATA, "clientSecret")
            if (all(key not in value or isinstance(value[key], str) for key in strings)
                and ("authorizedAt" not in value or number(value["authorizedAt"]))
                and ("redirectUris" not in value or isinstance(value["redirectUris"], list)
                     and all(isinstance(uri, str) for uri in value["redirectUris"]))
                and value.get("inferenceRegion", "global") in ("global", "eu", "us")):
                return {key: item for key, item in value.items()
                        if key in (*strings, "type", "access", "refresh", "expires", "authorizedAt", "redirectUris")}
    raise body_error("Invalid credential.")


def usage_payload(data):
    allowed = {"installId", "hostname", "app", "entries"}
    if data.keys() - allowed or not valid_string(data.get("installId")) or not isinstance(data.get("entries"), list):
        raise body_error()
    if any(key in data and not isinstance(data[key], str) for key in ("hostname", "app")):
        raise body_error()
    for entry in data["entries"]:
        if (not isinstance(entry, dict) or not valid_string(entry.get("provider")) or not valid_string(entry.get("model"))
            or not number(entry.get("at")) or abs(entry["at"]) > SAFE_INTEGER
            or any(not number(entry.get(key)) or not 0 <= entry[key] <= SAFE_INTEGER for key in COUNTERS)):
            raise body_error()
    return data


def since_ms(request):
    parsed = re.match(r"^\s*[+-]?\d+", request.query.get("sinceMs", ""))
    return max(-SAFE_INTEGER, min(SAFE_INTEGER, int(parsed[0]))) if parsed and len(parsed[0]) < 32 else 0


class BrokerServer:
    def __init__(self, database, token, *, refresh=True, heartbeat=20):
        self.db, self.token = database, token.encode()
        self.oauth = LocalOAuth(database, SKEW_MS // 1000)
        self.refresh_enabled, self.heartbeat = refresh, heartbeat
        self.next_sweep = now_ms()
        self.generation, self.rows = None, {}
        self.changed = asyncio.Event()
        self.reloading = asyncio.Lock()
        self.refresh_locks = {}
        self.quota = QuotaService(database, self.oauth, self.refresh_locks)
        self.resets = ResetService(database, self.quota, self.oauth, self.refresh_locks)

    async def reload(self):
        async with self.reloading:
            generation, rows = await asyncio.to_thread(self.db.read_entries, None, self.generation)
            if rows is not None:
                self.generation, self.rows = generation, {row["id"]: row for row in rows}
                changed, self.changed = self.changed, asyncio.Event()
                changed.set()

    def snapshot(self, canonical):
        now = now_ms()
        enabled, next_sweep = self.refresh_enabled, self.next_sweep
        rows = []
        for record in self.rows.values():
            entry = {**wire_entry(record), "rotatesInMs": rotates_in(record, enabled, next_sweep, now)}
            if blocks := project_blocks(record["blocks"], canonical, now):
                entry["blocks"] = blocks
            rows.append(entry)
        return {"generation": self.generation, "generatedAt": now, "serverNowMs": now,
                "refresher": schedule(enabled, next_sweep, now), "credentials": rows}

    async def poll(self):
        while True:
            await asyncio.sleep(.25)
            try:
                await self.reload()
            except sqlite3.Error:
                logger.warning("Could not read credential database changes.")

    async def sweep(self):
        while True:
            started = now_ms()
            self.next_sweep = started + INTERVAL_MS
            try:
                await self.reload()
                due = [row for row in self.rows.values() if row["credential"]["type"] == "oauth"
                       and can_refresh(row["provider"], row["credential"]) and row["credential"]["expires"] <= started + SKEW_MS]
                await asyncio.gather(*(self.refresh_row(row, forced=False) for row in due), return_exceptions=True)
            except sqlite3.Error:
                logger.warning("Could not read credentials for the background refresh.")
            await asyncio.sleep(max(0, self.next_sweep - now_ms()) / 1000)

    async def refresh_row(self, observed, *, forced, reason=None):
        identifier = observed["id"]
        async with self.refresh_locks.setdefault(identifier, asyncio.Lock()):
            await self.reload()
            row = self.rows.get(identifier)
            if not row:
                return None
            if row["credential"]["type"] != "oauth":
                raise AuthBrokerError("Credential does not support OAuth refresh.")
            if row["credentialRevision"] == observed["credentialRevision"]:
                normalized = {"broker_id": identifier, "provider": row["provider"],
                              "tokens": {"access_token": row["credential"]["access"]},
                              "credential_revision": row["credentialRevision"]}
                await self.oauth.refresh(normalized, forced, reuse_recent=reason == "auth-recovery",
                                         disable_dead=reason != "usage-probe")
                await self.reload()
            return self.rows.get(identifier)

    async def snapshots(self, request):
        await self.reload()
        raw_tag = request.headers.get("If-None-Match", "").strip()
        tag = re.fullmatch(r'(?:W/)?"(\d+)"|(\d+)', raw_tag)
        try:
            raw_wait = float(request.query.get("wait", "0"))
        except ValueError:
            raw_wait = 0
        wait = min(30_000, max(0, raw_wait)) / 1000 if math.isfinite(raw_wait) else 0
        generation = self.generation
        if wait >= .001 and tag and (tag[1] or tag[2]).lstrip("0") == str(generation).lstrip("0"):
            try:
                await asyncio.wait_for(self.changed.wait(), math.floor(wait * 1000) / 1000)
            except TimeoutError:
                return web.Response(status=304, headers=tag_headers(generation))
        result = self.snapshot(meter_scopes(request))
        return web.json_response(result, headers=tag_headers(result["generation"]))

    async def stream(self, request):
        await self.reload()
        headers = {"Content-Type": "text/event-stream", "Cache-Control": "no-store", "Vary": CAPABILITIES,
                   "X-Accel-Buffering": "no"}
        response = web.StreamResponse(headers=headers)
        await response.prepare(request)
        canonical = meter_scopes(request)

        async def emit(kind, body):
            encoded = json.dumps({"kind": kind, **body}, ensure_ascii=False, separators=(",", ":"))
            await response.write(f"event: {kind}\ndata: {encoded}\n\n".encode())

        def content(rows):
            return {row["id"]: {key: value for key, value in row.items() if key != "rotatesInMs"} for row in rows}

        try:
            changed = self.changed
            snapshot = self.snapshot(canonical)
            seen = content(snapshot["credentials"])
            await emit("snapshot", snapshot)
            while True:
                try:
                    await asyncio.wait_for(changed.wait(), self.heartbeat)
                except TimeoutError:
                    await response.write(b": keepalive\n\n")
                    continue
                changed = self.changed
                snapshot = self.snapshot(canonical)
                state = {key: snapshot[key] for key in ("generation", "serverNowMs", "refresher")}
                entries = snapshot["credentials"]
                current = content(entries)
                for entry in entries:
                    identifier = entry["id"]
                    if seen.get(identifier) != current[identifier]:
                        await emit("entry", {**state, "entry": entry})
                for identifier in seen.keys() - current.keys():
                    await emit("removed", {**state, "id": identifier})
                seen = current
        except (ConnectionError, asyncio.CancelledError):
            pass
        return response

    async def credentials(self, request):
        data = await payload(request)
        if data.keys() != {"provider", "credential"} or not valid_string(data["provider"]):
            raise body_error()
        provider = data["provider"]
        await asyncio.to_thread(self.db.upsert, provider, credential_payload(data["credential"]))
        await self.reload()
        return web.json_response({"entries": [wire_entry(row) for row in self.rows.values() if row["provider"] == provider]})

    async def credential_action(self, request):
        identifier = int(request.match_info["id"])
        action = request.match_info["action"]
        await self.reload()
        observed = self.rows.get(identifier)
        if not observed:
            return web.json_response({"error": f"No credential with id={identifier}"}, status=404)
        if action == "refresh":
            try:
                row = await self.refresh_row(observed, forced=True, reason=request.query.get("reason"))
            except AuthBrokerError as error:
                return web.json_response({"error": str(error)}, status=500)
            if not row:
                return web.json_response({"error": f"No credential with id={identifier}"}, status=404)
            return web.json_response({"entry": wire_entry(row)})
        if action == "disable":
            data = await payload(request, optional=True)
            if data.keys() - {"cause"} or "cause" in data and not isinstance(data["cause"], str):
                raise body_error()
            changed = await asyncio.to_thread(self.db.disable, identifier, data.get("cause", "").strip()
                                              or "disabled via auth-broker")
        elif request.method == "POST":
            block = await payload(request)
            if (not {"providerKey", "blockScope", "blockedUntilMs"} <= block.keys()
                or block.keys() - {"providerKey", "blockScope", "blockedUntilMs", "updatedAtMs"}
                or not valid_string(block["providerKey"]) or not isinstance(block["blockScope"], str)
                or not number(block["blockedUntilMs"]) or abs(block["blockedUntilMs"]) > SAFE_INTEGER
                or "updatedAtMs" in block and not number(block["updatedAtMs"])):
                raise body_error()
            changed = await asyncio.to_thread(self.db.write_block, identifier, observed["provider"], block)
        else:
            slot = None
            if action == "block":
                block = await payload(request)
                if block.keys() != {"providerKey", "blockScope"} or not valid_string(block["providerKey"]) or not isinstance(block["blockScope"], str):
                    raise body_error()
                slot = block["providerKey"], block["blockScope"]
            changed = await asyncio.to_thread(self.db.delete_blocks, identifier, slot)
        await self.reload()
        return web.json_response({"ok": True}) if changed else web.json_response(
            {"error": f"No credential with id={identifier}"}, status=404,
        )

    async def disabled(self, request):
        rows = await asyncio.to_thread(self.db.disabled, request.query.get("provider"))
        return web.json_response({"generatedAt": now_ms(), "disabled": rows})

    async def usage(self, request):
        path, provider = request.path, request.query.get("provider")
        if path == "/v1/usage/observed":
            await asyncio.to_thread(self.db.record_client_usage, usage_payload(await payload(request)))
            return web.json_response({"ok": True})
        if path == "/v1/usage/stale":
            await self.quota.invalidate(provider)
            return web.json_response({"ok": True})
        if path == "/v1/usage/history":
            key, rows = "entries", await asyncio.to_thread(self.db.usage_history, provider, since_ms(request))
        elif path == "/v1/usage/clients":
            key, rows = "clients", await asyncio.to_thread(self.db.client_usage, since_ms(request))
        else:
            await self.reload()
            key, rows = "reports", await self.quota.reports(tuple(self.rows.values()), provider)
        return web.json_response({"generatedAt": now_ms(), key: rows})

    async def saved_resets(self, request):
        if request.method == "GET":
            provider, identifier = request.query.get("provider"), request.query.get("credentialId")
            if provider is not None and provider not in RESET_PROVIDERS.values():
                raise body_error("Unsupported reset provider.")
            if identifier is not None and (
                not re.fullmatch(r"[1-9][0-9]{0,17}", identifier) or int(identifier) > SAFE_INTEGER
            ):
                raise body_error()
            allowed = (lambda row: row["credentialId"] == int(identifier)) if identifier else None
            active = request.query.get("active")
            if active is not None and (not re.fullmatch(r"[1-9][0-9]{0,17}", active) or int(active) > SAFE_INTEGER):
                raise body_error()
            return web.json_response({"accounts": await self.resets.list(
                provider, allowed, evidence=request.query.get("evidence") == "1", active=int(active) if active else None,
            )})
        data = await payload(request)
        provider, identifier = data.get("provider"), data.get("credentialId")
        credit, expected, identity = data.get("creditId"), data.get("expectedCount"), data.get("identityKey")
        if (provider not in RESET_PROVIDERS.values() or type(identifier) is not int or not 0 < identifier <= SAFE_INTEGER
            or credit is not None and (not isinstance(credit, str) or not 0 < len(credit) <= 1024)
            or expected is not None and (type(expected) is not int or not 0 <= expected <= SAFE_INTEGER)
            or identity is not None and (not isinstance(identity, str) or not 0 < len(identity) <= 1024)):
            raise body_error()
        outcome = await self.resets.redeem(provider, identifier, credit, expected, identity)
        return web.json_response(outcome)

    def application(self):
        @web.middleware
        async def authorized(request, handler):
            if request.path != "/v1/healthz":
                match = re.fullmatch(r"Bearer\s+(.+)", request.headers.get("Authorization", ""), re.I)
                if not match or not hmac.compare_digest(self.token, match[1].strip().encode()):
                    return web.json_response({"error": "Unauthorized"}, status=401)
            try:
                return await handler(request)
            except (sqlite3.Error, OverflowError):
                return web.json_response({"error": "Credential database operation failed."}, status=500)

        async def health(_):
            return web.json_response({"ok": True, "version": "mini-pi"})

        async def start(_):
            await self.reload()
            tasks = [asyncio.create_task(self.poll())]
            if self.refresh_enabled:
                tasks.append(asyncio.create_task(self.sweep()))
            try:
                yield
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                await self.resets.close()
                await self.quota.close()
                await self.oauth.close()

        app = web.Application(middlewares=[authorized], client_max_size=1024 * 1024)
        app.cleanup_ctx.append(start)
        app.router.add_get("/v1/healthz", health)
        app.router.add_get("/v1/snapshot", self.snapshots)
        app.router.add_get("/v1/snapshot/stream", self.stream)
        app.router.add_post("/v1/credential", self.credentials)
        app.router.add_post(r"/v1/credential/{id:\d{1,18}}/{action:refresh|disable|block}", self.credential_action)
        app.router.add_delete(r"/v1/credential/{id:\d{1,18}}/{action:block|blocks}", self.credential_action)
        app.router.add_get("/v1/credentials/disabled", self.disabled)
        for path in ("usage", "usage/history", "usage/clients"):
            app.router.add_get("/v1/" + path, self.usage)
        for path in ("usage/observed", "usage/stale"):
            app.router.add_post("/v1/" + path, self.usage)
        app.router.add_get("/v1/usage/resets", self.saved_resets)
        app.router.add_post("/v1/usage/resets/redeem", self.saved_resets)
        return app


def load_token(path):
    override = os.getenv("OMP_AUTH_BROKER_TOKEN", "").strip()
    if override:
        return override
    try:
        return path.read_text(encoding="utf-8-sig").strip()
    except FileNotFoundError:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        token = secrets.token_hex(32)
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            return path.read_text(encoding="utf-8-sig").strip()
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            file.write(token + "\n")
        return token


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-db", type=Path, help="Oh My Pi schema v8 agent.db (created if missing)")
    parser.add_argument("--bind", default="127.0.0.1:8765")
    parser.add_argument("--no-refresh", action="store_true", help="disable the background refresher")
    args = parser.parse_args()
    profile = os.environ.get("OMP_PROFILE", os.environ.get("PI_PROFILE", "default")).strip().lower() or "default"
    try:
        bind = urlsplit("//" + args.bind)
        if not bind.hostname or bind.port is None or bind.username or bind.path or bind.query or bind.fragment:
            raise ValueError("Bind address must be host:port.")
        db = AgentCredentialDB(args.agent_db or agent_db_path(profile), create=True)
        token = load_token(config_root(profile) / "auth-broker.token")
        if not token or not token.isascii() or any(char.isspace() or ord(char) < 32 for char in token):
            raise ValueError("Invalid Auth Broker bearer token.")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    app = BrokerServer(db, token, refresh=not args.no_refresh).application()
    web.run_app(app, host=bind.hostname, port=bind.port, access_log=None, handler_cancellation=True)


if __name__ == "__main__":
    main()
