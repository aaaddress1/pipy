"""Stored-account saved resets for Claude and Codex."""

import asyncio
import os
import re
import sqlite3
from uuid import uuid4

import httpx2

import provider_quota as vendor
from auth_broker import AuthBrokerError
from auth_broker_cache import number
from broker_usage import exhausted, window
from local_oauth import jwt_payload, object_or_empty, parse_object, text
from quota_service import now_ms
from reset_fence import marker_path, read_marker, reset_lock, write_marker


PROVIDERS = {"claude": "anthropic", "codex": "openai-codex"}
COOLDOWN_MS = 60_000
TIERS = {"fable", "mythos"}
CODEX_PATH = "/wham/rate-limit-reset-credits"
ORG_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")


def account_details(record):
    credential = record["credential"]
    details = vendor.identity(credential)
    if record["provider"] == "openai-codex":
        claims = jwt_payload(credential["access"])
        for field, claim, name in (("accountId", vendor.AUTH_CLAIM, "chatgpt_account_id"),
                                   ("email", vendor.PROFILE_CLAIM, "email")):
            if field not in details and (value := text(object_or_empty(claims.get(claim)).get(name))):
                details[field] = value
    return details


def account_key(record):
    identity = account_details(record)
    account = identity.get("accountId") or identity.get("email")
    if account:
        return f"{record['provider']}|{identity.get('orgId', '-').lower()}|{account.lower()}"
    return None


def public_account(record):
    return {"credentialId": record["id"], "provider": record["provider"], "identityKey": record["identityKey"],
            **account_details(record)}


def codex_inventory(payload):
    fields = {"status": "status", "granted_at": "grantedAt", "expires_at": "expiresAt", "title": "title",
              "description": "description", "reset_type": "resetType", "redeem_started_at": "redeemStartedAt",
              "redeemed_at": "redeemedAt"}
    raw = payload.get("credits")
    credits = []
    for item in raw if isinstance(raw, list) else ():
        item = object_or_empty(item)
        if isinstance(item.get("id"), str) and item["id"]:
            credit = {"id": item["id"], **{dst: item[src] for src, dst in fields.items()
                                          if isinstance(item.get(src), str)}}
            credit["usable"] = credit.get("status", "available") == "available"
            credits.append(credit)
    count = vendor.numeric(payload.get("available_count"))
    count = max(0, int(count)) if count is not None else sum(credit["usable"] for credit in credits)
    return {"credits": credits, "availableCount": count, "redeemableCount": count, "eligible": count > 0}


def claude_block_changes(cleared, report):
    now = now_ms()
    if not cleared or not report or now - report["fetchedAt"] > 300_000:
        return (), ()
    limits = report["limits"]
    shared = [limit for limit in limits if limit["scope"].get("shared")]
    if not {"anthropic:5h", "anthropic:7d"} <= {limit["id"] for limit in shared}:
        return (), ()
    independent = {tier: [limit for limit in limits if limit["scope"].get("tier") == tier] for tier in TIERS}
    future = {limit["id"]: reset for rows in independent.values() for limit in rows
              if number(reset := window(limit).get("resetsAt")) and reset > now}
    global_limits = [limit for limit in limits if limit in shared or (
        limit["scope"].get("tier") and limit["id"] not in future
    )]
    gates = {"": global_limits, **{f"tier:{tier}": shared + rows for tier, rows in independent.items() if rows}}
    scopes = [scope for scope, rows in gates.items() if any(limit["id"] in cleared for limit in rows)
              and not any(limit["id"] not in cleared and exhausted(limit) for limit in rows)]
    preserve = [(f"tier:{tier}", future[limit["id"]]) for tier, rows in independent.items() for limit in rows
                if "" in scopes and limit["id"] in future and limit["id"] not in cleared and exhausted(limit)]
    return scopes, preserve


def parse_consumption(provider, response, credit):
    status = response.status_code
    payload = parse_object(response.content)
    if provider == "openai-codex":
        code = text(payload.get("code")) or ("reset" if response.is_success else f"http_{status}")
        return code, None
    if not response.is_success:
        return {401: "auth_error", 403: "auth_error", 429: "rate_limited"}.get(status, f"http_{status}"), None
    code = text(payload.get("result"))
    if not code or ("reason" in payload and not isinstance(payload["reason"], str)):
        return "malformed_response", None
    try:
        cleared = credit["clears"] if code == "reset" else None
        if "cleared" in payload:
            if not isinstance(payload["cleared"], list):
                return "malformed_response", None
            cleared = list(dict.fromkeys(map(vendor.reset_window, payload["cleared"])))
    except ValueError:
        return "malformed_response", None
    return {"already_used": "already_redeemed", "not_limited": "nothing_to_reset"}.get(code, code), cleared


class ResetExecutor:
    """Shared provider exchange using fresh account records and a backend reset ledger."""

    def __init__(self, fence_root):
        self.fence_root = fence_root
        self.client = httpx2.AsyncClient(timeout=25, follow_redirects=False)
        self.tasks = set()

    async def get(self, provider, identifier):
        return next((row for row in await self.records(provider) if row["id"] == identifier), None)

    @staticmethod
    def headers(record):
        headers = {"Authorization": "Bearer " + record["credential"]["access"], "Accept": "application/json"}
        if record["provider"] == "anthropic":
            version = os.getenv("PI_AI_CLAUDE_CODE_VERSION") or "2.1.280"
            return {**headers, "anthropic-beta": "oauth-2025-04-20",
                    "User-Agent": f"claude-cli/{version} (external, cli)"}
        headers["User-Agent"] = "omp/18.18.1"
        if account := account_details(record).get("accountId"):
            headers["ChatGPT-Account-Id"] = account
        return headers

    async def discover(self, record, headers, on_throttle=None):
        if record["provider"] == "anthropic":
            endpoint = vendor.CLAUDE_URL + "/usage"
            evidence = {}
            now = now_ms()
            offers = await vendor.claude_resets(self.client, endpoint, headers, {}, now, evidence=evidence, timeout=5,
                                               on_throttle=on_throttle)
            program = next((credit["program"] for credit in (offers or {}).get("credits", ())), None)
            return offers, vendor.parse_claude(evidence.get(program), record["credential"], endpoint, now)
        payload = await vendor.codex_optional(self.client, CODEX_PATH.lstrip("/"), headers)
        return (codex_inventory(payload) if payload is not None else None), None

    async def list(self, provider=None, allowed=None, *, evidence=False, active=None):
        rows = await self.records(provider)

        async def describe(row):
            account = public_account(row)
            retry = None

            def throttle(headers):
                nonlocal retry
                milliseconds = vendor.numeric(headers.get("Retry-After-Ms"))
                retry = max(0, milliseconds if milliseconds is not None else (
                    vendor.retry_delay(headers["Retry-After"], 0) * 1000 if "Retry-After" in headers else 0
                ))

            prepared = await self.prepare(row)
            offers, report = await self.discover(prepared, self.headers(prepared), throttle) if prepared else (None, None)
            if evidence:
                account["active"] = row["id"] == active
                if report:
                    account["report"] = report
                if retry is not None:
                    account["retryAfterMs"] = retry
                if key := account_key(row):
                    marker, stamp = read_marker(marker_path(self.fence_root, key))
                    if marker == "reset" and 0 <= now_ms() - stamp < COOLDOWN_MS:
                        account["recentResetAt"] = stamp
            return {**account, **offers} if offers is not None else {**account, "error": "credit_list_failed"}

        rows = [row for row in rows if allowed is None or allowed(public_account(row))]
        claude_rows = [row for row in rows if row["provider"] == "anthropic"]
        claude = [await describe(row) for row in sorted(claude_rows, key=lambda row: row["id"] != active)]
        codex = await asyncio.gather(*(describe(row) for row in rows if row["provider"] == "openai-codex"))
        return [*claude, *codex]

    async def organization(self, record, headers):
        org = account_details(record).get("orgId")
        if org is None:
            try:
                response = await self.client.get(vendor.CLAUDE_URL + "/profile", headers=headers, timeout=4)
            except httpx2.HTTPError:
                return None
            if response.is_success:
                data = parse_object(response.content)
                org = text(object_or_empty(data.get("organization")).get("uuid")) or text(data.get("organization_uuid"))
        return org if org and ORG_ID.fullmatch(org) else None

    async def redeem(self, provider, identifier, credit_id=None, expected_count=None, identity_key=None):
        stop = asyncio.Event()
        task = asyncio.create_task(self._redeem(provider, identifier, credit_id, expected_count, identity_key, stop))
        self.tasks.add(task)

        def finished(completed):
            self.tasks.discard(completed)
            if not completed.cancelled():
                completed.exception()

        task.add_done_callback(finished)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            stop.set()
            raise

    async def _redeem(self, provider, identifier, credit_id, expected_count, identity_key, stop):
        def result(code, **fields):
            return {"ok": code == "reset", "code": code, "provider": provider, "credentialId": identifier, **fields}

        record = await self.get(provider, identifier)
        if not record or identity_key is not None and record["identityKey"] != identity_key:
            return result("account_changed")
        key = account_key(record)
        if key is None:
            return result("account_identity_unavailable")
        marker = marker_path(self.fence_root, key)
        await asyncio.to_thread(marker.parent.mkdir, parents=True, exist_ok=True)
        async with reset_lock(marker, stop) as acquired:
            if not acquired:
                return result("reset_in_progress")
            state, stamp = read_marker(marker)
            if now_ms() - stamp < COOLDOWN_MS:
                return result("reset_recent" if state == "reset" else "reset_in_progress", attemptedAt=stamp)
            live = await self.get(provider, identifier)
            if not live or live["identityKey"] != record["identityKey"] or account_key(live) != key:
                return result("account_changed")
            record = await self.prepare(live)
            if not record:
                return result("auth_error")
            headers = self.headers(record)
            offers, report = await self.discover(record, headers)
            if offers is None:
                return result("credit_list_failed")
            claude = provider == "anthropic"
            credits, count = offers["credits"], offers["availableCount"]
            if claude:
                selected = offers.get("nextCreditId")
                if credit_id is not None and credit_id != selected:
                    return result("offer_changed", creditId=credit_id)
                credit = next((item for item in credits if item["id"] == selected and item["usable"]), None)
                if not offers["eligible"] or not credit or offers["redeemableCount"] < 1:
                    return result("ineligible" if count else "no_credit")
            else:
                if expected_count is not None and count < expected_count:
                    return result("spent")
                if not count:
                    return result("no_credit")
                choices = [item for item in credits if item["usable"] and (credit_id is None or item["id"] == credit_id)]
                if not choices:
                    return result("offer_changed" if credit_id is not None else "no_credit")
                credit = min(choices, key=lambda item: vendor.iso_ms(item.get("expiresAt")) or float("inf"))

            credit_id, program = credit["id"], credit.get("program")
            remaining = credit.get("remainingCount", count)
            pending = await asyncio.to_thread(self.pending_reset, key)
            if pending and not any(item["id"] == pending["creditId"] for item in credits):
                await asyncio.to_thread(self.save_pending_reset, key, None)
                pending = None
            if pending:
                if pending["creditId"] != credit_id:
                    return result("reset_unconfirmed", creditId=credit_id)
                if remaining < pending["remainingCount"]:
                    await self.after_reset(record)
                    await asyncio.to_thread(self.save_pending_reset, key, None)
                    return result("already_redeemed", creditId=credit_id)
                if pending["program"] == "juniper_tide":
                    return result("reset_unconfirmed", creditId=credit_id)

            request_id = pending["requestId"] if pending else str(uuid4())
            if claude:
                org = await self.organization(record, headers)
                if not org:
                    return result("organization_unavailable", creditId=credit_id)
                url = vendor.CLAUDE_URL.removesuffix("/oauth") + f"/organizations/{org}/reset_rate_limits"
                body = {"program": program}
                if program == "cedar_ember":
                    body.update(grant_id=credit_id, request_id=request_id)
            else:
                url = vendor.CODEX_URL + CODEX_PATH + "/consume"
                body = {"credit_id": credit_id, "redeem_request_id": request_id}
                if account := headers.get("ChatGPT-Account-Id"):
                    body["account_id"] = account
            latest = await self.get(provider, identifier)
            if not latest or latest["credentialRevision"] != record["credentialRevision"]:
                return result("account_changed")
            if stop.is_set():
                return result("cancelled")
            if claude and (expiry := vendor.iso_ms(credit.get("expiresAt"))) is not None and expiry <= now_ms():
                return result("unavailable", creditId=credit_id)
            started = now_ms()
            write_marker(marker, "pending", started)
            await asyncio.to_thread(self.save_pending_reset, key, {
                "creditId": credit_id, "requestId": request_id, "program": program, "remainingCount": remaining,
            })
            try:
                async with asyncio.timeout(25):
                    response = await self.client.post(url, json=body, headers=headers)
            except (httpx2.HTTPError, TimeoutError):
                return result("network_error", creditId=credit_id)
            code, cleared = parse_consumption(provider, response, credit)
            fields = {"creditId": credit_id}
            if cleared is not None:
                fields["cleared"] = cleared
            definitive = code in ("reset", "already_redeemed", "nothing_to_reset", "ineligible", "no_credit")
            first_rejected = not pending and (code == "cooldown" or response.status_code in (401, 403, 429))
            try:
                if code == "reset":
                    write_marker(marker, "reset", started)
                    fields["attemptedAt"] = started
                    scopes, preserve = claude_block_changes(cleared, report) if claude else (("", "chat", "spark", "shared"), ())
                    await self.after_reset(record, scopes, preserve)
                elif code in ("no_credit", "nothing_to_reset"):
                    write_marker(marker)
                elif code == "already_redeemed":
                    await self.after_reset(record)
                if definitive or first_rejected:
                    await asyncio.to_thread(self.save_pending_reset, key, None)
            except (OSError, sqlite3.Error):
                fields["cacheUpdated"] = False
            return result(code, **fields)

    async def close(self):
        await asyncio.gather(*self.tasks, return_exceptions=True)
        await self.client.aclose()


class ResetService(ResetExecutor):
    def __init__(self, database, quota, oauth, refresh_locks):
        super().__init__(database.path)
        self.db, self.oauth, self.refresh_locks = database, oauth, refresh_locks
        self.pending_reset, self.save_pending_reset = database.pending_reset, database.save_pending_reset
        self.after_reset = quota.after_reset

    async def records(self, provider):
        providers = (provider,) if provider in PROVIDERS.values() else tuple(PROVIDERS.values()) if provider is None else ()
        rows = await asyncio.to_thread(self.db.entries, providers)
        return [row for row in rows if row["credential"]["type"] == "oauth"]

    async def prepare(self, record):
        credential = record["credential"]
        if credential["expires"] > now_ms() + self.oauth.skew * 1000:
            return record
        identifier = record["id"]
        async with self.refresh_locks.setdefault(identifier, asyncio.Lock()):
            try:
                await self.oauth.refresh({
                    "broker_id": identifier, "provider": record["provider"],
                    "tokens": {"access_token": credential["access"]}, "credential_revision": record["credentialRevision"],
                }, False, disable_dead=False)
            except (AuthBrokerError, OSError):
                pass
            row = await self.get(record["provider"], identifier)
        if (row and row["credential"]["expires"] > now_ms() and row["identityKey"] == record["identityKey"]
            and account_key(row) == account_key(record)):
            return row
        return None


async def usage_reset(profile, database_path, provider, identifier, credit_id):
    from auth_broker import AuthBroker, account_pool, broker_config
    from broker_resets import BrokerResets
    from local_credentials import AgentCredentialDB, agent_db_path
    from local_oauth import LocalOAuth
    from quota_service import QuotaService

    pool = account_pool()

    def allowed(row):
        accounts, _ = pool.get(row["provider"], pool.get("*", (None, True)))
        return (identifier is None or row["credentialId"] == identifier) and (
            accounts is None or row["identityKey"] in accounts
        )

    def target(rows):
        return next((row for row in rows if allowed(row) and row["provider"] == provider), None)

    def changed(row):
        code = "credit_list_failed" if row else "account_changed"
        return {"ok": False, "code": code, "provider": provider, "credentialId": identifier}

    if database_path is None and (remote := broker_config(profile)):
        broker = AuthBroker(*remote, profile, model_providers=PROVIDERS.values())
        resets = BrokerResets(broker)
        try:
            rows = [row for row in await resets.list(provider, identifier, timeout=180) if allowed(row)]
            if identifier is None:
                return rows
            row = target(rows)
            if not row or row.get("error"):
                return changed(row)
            return await resets.redeem(provider, identifier, credit_id, row["availableCount"], row["identityKey"])
        finally:
            await resets.close()
            await broker.close()

    db = AgentCredentialDB(database_path or agent_db_path(profile))
    oauth = LocalOAuth(db, 300)
    refresh_locks = {}
    quota = QuotaService(db, oauth, refresh_locks)
    service = ResetService(db, quota, oauth, refresh_locks)
    try:
        rows = await service.list(provider, allowed)
        if identifier is None:
            return rows
        row = target(rows)
        if not row or row.get("error"):
            return changed(row)
        return await service.redeem(provider, identifier, credit_id, row["availableCount"], row["identityKey"])
    finally:
        await service.close()
        await quota.close()
        await oauth.close()


def parse_command(arguments):
    import argparse
    from pathlib import Path

    aliases = {**PROVIDERS, **{provider: provider for provider in PROVIDERS.values()}}
    parser = argparse.ArgumentParser(prog="mini_pi.py usage reset", description="List or spend a stored OAuth reset credit.")
    parser.add_argument("target", nargs="?", metavar="PROVIDER/ID")
    parser.add_argument("--provider", choices=aliases)
    parser.add_argument("--agent-db", type=Path)
    parser.add_argument("--credit-id")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(arguments)
    provider, identifier = aliases.get(args.provider), None
    if args.target:
        alias, _, raw_id = args.target.partition("/")
        if alias not in aliases or not re.fullmatch(r"[1-9][0-9]{0,17}", raw_id) or int(raw_id) > 2**53 - 1:
            parser.error("target must be claude/ID or codex/ID")
        if provider and provider != aliases[alias]:
            parser.error("target and --provider refer to different providers")
        provider, identifier = aliases[alias], int(raw_id)
    if args.credit_id is not None and (identifier is None or not 0 < len(args.credit_id) <= 1024):
        parser.error("--credit-id requires a target and a non-empty credit ID")
    return args, provider, identifier


def display(output, as_json):
    import json

    if as_json:
        return json.dumps(output, ensure_ascii=True)

    def single_line(value):
        return "".join(char if char.isprintable() else " " for char in value)

    if isinstance(output, dict):
        credit = f" ({output['creditId']})" if "creditId" in output else ""
        cache = " (local usage state update failed)" if output.get("cacheUpdated") is False else ""
        return single_line(f"{output['provider']}/{output['credentialId']} {output['code']}{credit}{cache}")
    rows = []
    for row in output:
        name = row.get("email") or row.get("accountId") or ""
        state = row.get("error") or f"{row['availableCount']} saved, {row['redeemableCount']} redeemable"
        credits = ", ".join(str(credit["id"]) for credit in row.get("credits", []) if credit["usable"])
        rows.append(single_line(f"{row['provider']}/{row['credentialId']} {name} {state}"
                                + (f" ({credits})" if credits else "")))
    return "\n".join(rows) or "No stored Claude or Codex OAuth accounts."


async def session_reset(profile, path, provider, identifier, credit_id, pool):
    from auth_broker import AuthBroker

    output = await usage_reset(profile, path, provider, identifier, credit_id)
    if pool and identifier is not None and output["code"] in ("reset", "already_redeemed"):
        await pool.invalidate_reset(provider, identifier)
        if isinstance(pool, AuthBroker):
            try:
                await pool.fetch()
            except AuthBrokerError:
                pass
    return output


def run_cli(arguments):
    import sys

    from mcp_oauth import active_profile

    args, provider, identifier = parse_command(arguments)
    try:
        output = asyncio.run(usage_reset(active_profile(), args.agent_db, provider, identifier, args.credit_id))
    except (AuthBrokerError, ValueError, OSError, sqlite3.Error) as error:
        print(f"mini-pi: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    print(display(output, args.json))
    return 1 if isinstance(output, dict) and not output["ok"] else 0
