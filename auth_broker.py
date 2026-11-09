"""Oh My Pi Auth Broker client and live MCP credential view."""

import asyncio
import json
import os
import re
from pathlib import Path
from time import monotonic, time
from urllib.parse import urlsplit

import httpx2

from auth_broker_cache import REMOTE_REFRESH, config_root, number
from broker_usage import (IDENTITY, PROVIDERS, exhausted, healable_scopes, match_report, merge_report,
                          overlay_key, parse_policies, plan_requirement, rank_oauth, usage_reports, usage_reset_times)
from observed_usage import ObservedUsage
from rate_limits import UsageBlock


PROFILE_PREFIX = "mcp_oauth:profile:"
USAGE_REPORT_TTL = 300_000
PROTECTED_SCOPES = frozenset(("auth", "account-policy"))


def block_signature(block):
    return block["blockedUntilMs"], block.get("updatedAtMs")


def request_scopes(provider, scope, model):
    scopes = {"", scope, *PROTECTED_SCOPES}
    if provider == "openai-codex" or scope in ("chat", "spark"):
        scopes.add("shared")
    if provider in ("openai-codex", "cursor") and model:
        scopes.add("model-policy:" + model.rsplit("/", 1)[-1].strip().lower())
    return scopes


class AuthBrokerError(OSError):
    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


class ModelCredentialError(OSError):
    pass


class ModelQuotaError(ModelCredentialError):
    pass


def account_pool():
    file = os.getenv("OMP_AUTH_BROKER_ACCOUNT_POOL_FILE", "").strip()
    if not file:
        return {}
    try:
        rules = json.loads(Path(file).read_text(encoding="utf-8-sig"))
        if not isinstance(rules, dict):
            raise ValueError()
        pool = {}
        for provider, rule in rules.items():
            if not provider or provider != provider.strip():
                raise ValueError()
            if type(rule) is bool:
                pool[provider] = (None if rule else frozenset(), rule)
                continue
            if isinstance(rule, list):
                rule = {"accounts": rule}
            if not isinstance(rule, dict) or rule.keys() - {"accounts", "apiKeys"}:
                raise ValueError()
            accounts, keys = rule.get("accounts"), rule.get("apiKeys", True)
            if (type(keys) is not bool or "accounts" in rule and (
                not isinstance(accounts, list) or not all(isinstance(value, str) and value and value == value.strip() for value in accounts)
            )):
                raise ValueError()
            pool[provider] = (frozenset(accounts) if accounts is not None else None, keys)
        return pool
    except (OSError, ValueError):
        raise ValueError("Invalid OMP_AUTH_BROKER_ACCOUNT_POOL_FILE.") from None


def dotted(config, key):
    current = config
    for segment in key.split("."):
        if not isinstance(current, dict) or segment not in current:
            current = config.get(key)
            break
        current = current[segment]
    return current


def config_value(config, key):
    current = dotted(config, key)
    if isinstance(current, str) and not current.startswith("!"):
        current = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", lambda match: os.getenv(match[1], ""), current)
        return (os.getenv(current) or current).strip()
    return None


def broker_settings(profile):
    root = config_root(profile)
    agent = root / "agent"
    if profile == "default":
        agent = Path(os.getenv("PI_CODING_AGENT_DIR") or root / "agent").expanduser()
    for name in ("config.yml", "config.yaml"):
        path = agent / name
        try:
            contents = path.read_text(encoding="utf-8-sig")
        except FileNotFoundError:
            continue
        import yaml

        try:
            config = yaml.safe_load(contents)
        except yaml.YAMLError:
            raise ValueError(f"Invalid Oh My Pi YAML in {path}.") from None
        if config is not None and not isinstance(config, dict):
            raise ValueError(f"Oh My Pi config {path} must be an object.")
        return config or {}
    return {}


def broker_config(profile):
    url = os.getenv("OMP_AUTH_BROKER_URL", "").strip()
    token = os.getenv("OMP_AUTH_BROKER_TOKEN", "").strip()
    if not url or not token:
        config = broker_settings(profile)
        url = url or config_value(config, "auth.broker.url")
        token = token or config_value(config, "auth.broker.token")
    if not url:
        return None
    if not token:
        try:
            token = (config_root(profile) / "auth-broker.token").read_text(encoding="utf-8-sig").strip()
        except FileNotFoundError:
            pass
    if not token:
        raise ValueError("Auth Broker URL is configured but its bearer token is missing.")
    return url, token


class CredentialPool:
    refresh_skew = 300

    def __init__(self, profile, model_providers=()):
        self.profile = profile
        self.model_providers = frozenset(model_providers)
        self.pool = account_pool()
        self.refresh_locks = {}
        settings = broker_settings(profile) if self.model_providers else {}
        self.policies, self.default_reserve = parse_policies(dotted(settings, "auth.accountPolicies"),
                                                            dotted(settings, "retry.usageReservePct"))
        self.usage_cache = {}
        self.on_usage = None
        self.usage_tasks = {}
        self.usage_overlays = {}
        self.usage_ingested = {}
        self.block_locks = {}
        # Bridge the time between a local mutation and the matching broker snapshot.
        self.pending_blocks = {}
        self.deleted_blocks = {}
        self.block_seen = {}
        self.timed_blocks = {}
        self.rows = {}
        self.clock_offset = 0
        self.snapshot_lock = asyncio.Lock()
        self.revision = 0
        self.local_revision = 0
        self.observed_usage = None

    def entry(self, row, offset=None):
        if not isinstance(row, dict) or type(row.get("id")) is not int or not isinstance(row.get("provider"), str):
            raise AuthBrokerError("Auth Broker returned an invalid credential entry.")
        provider, credential = row["provider"], row.get("credential")
        if not isinstance(credential, dict):
            raise AuthBrokerError("Auth Broker returned an invalid credential.")
        mcp = provider.startswith(("mcp_oauth:", "mcp_oauth_"))
        kind = credential.get("type")
        if not (mcp and kind == "oauth" or provider in self.model_providers and kind in ("oauth", "api_key")):
            return None
        if provider.startswith(PROFILE_PREFIX) and not provider.startswith(f"{PROFILE_PREFIX}{self.profile}:"):
            return None
        identity = row.get("identityKey")
        if identity is not None and not isinstance(identity, str):
            raise AuthBrokerError("Auth Broker returned an invalid account identity.")
        accounts, keys = self.pool.get(provider, self.pool.get("*", (None, True)))
        if kind == "api_key" and not keys or kind == "oauth" and accounts is not None and identity not in accounts:
            return None
        blocks = row.get("blocks", ())
        fields = {"broker_id": row["id"], "provider": provider, "kind": kind, "identity": identity,
                  "blocks": blocks if isinstance(blocks, list) else [],
                  "account": {field: credential[field] for field in IDENTITY if isinstance(credential.get(field), str)}}
        if kind == "api_key":
            key = credential.get("key")
            if not isinstance(key, str) or not key:
                raise AuthBrokerError("Auth Broker returned an invalid API key entry.")
            return {**fields, "tokens": {"access_token": key}, "source": credential.get("source"), "expires": None}
        access, expires = credential.get("access"), credential.get("expires")
        if credential.get("refresh") != REMOTE_REFRESH or not isinstance(access, str) or not access or not number(expires):
            raise AuthBrokerError("Auth Broker returned an invalid OAuth credential.")
        now = time()
        expiry = expires / 1000 + (self.clock_offset if offset is None else offset)
        client = {key: credential[field] for key, field in (
            ("client_id", "clientId"), ("client_secret", "clientSecret"), ("issuer", "clientIssuer"),
            ("token_endpoint_auth_method", "clientAuthMethod"), ("redirect_uris", "redirectUris"),
        ) if field in credential}
        tokens = {"access_token": access}
        if isinstance(credential.get("scope"), str):
            tokens["scope"] = credential["scope"]
        return {**fields, "tokens": tokens, "expires": expiry,
                "issued": now, "client": client, "alias": credential.get("miniPiAlias")}

    async def lookup(self, providers):
        await self.ensure()
        for provider in providers:
            row = next((row for row in self.rows.values() if row["provider"] == provider), None)
            if row:
                return row
        return None

    def server_time(self):
        return (time() - self.clock_offset) * 1000

    def active_blocks(self, row, now):
        identifier, provider, kind = row["broker_id"], row["provider"], row["kind"]
        pending = self.pending_blocks.get(identifier, {})
        deleted = self.deleted_blocks.get(identifier, {})
        blocks = {}
        for block in row.get("blocks", ()):
            if not isinstance(block, dict):
                continue
            key, scope, until = block.get("providerKey"), block.get("blockScope"), block.get("blockedUntilMs")
            if key not in (provider, f"{provider}:{kind}") or not isinstance(scope, str) or not number(until) or until <= now:
                continue
            slot = key, scope
            # A newer block in the same scope can carry the same deadline.
            if deleted.get(slot) == block_signature(block):
                continue
            if slot not in blocks or blocks[slot]["blockedUntilMs"] <= until:
                blocks[slot] = block
        for slot, block in tuple(pending.items()):
            until = block["blockedUntilMs"]
            mirrored = blocks.get(slot)
            if mirrored and mirrored["blockedUntilMs"] == until:
                # Keep the local probe delay even when the server rounds its timestamp down.
                self.block_seen[(identifier, slot, until)] = block["updatedAtMs"]
            if until <= now or (mirrored and mirrored["blockedUntilMs"] >= until):
                pending.pop(slot)
            else:
                blocks[slot] = block
        if not pending:
            self.pending_blocks.pop(identifier, None)
        for slot, signature in tuple(deleted.items()):
            if signature[0] <= now or slot in blocks and block_signature(blocks[slot]) != signature:
                deleted.pop(slot)
        if not deleted:
            self.deleted_blocks.pop(identifier, None)
        return blocks

    def probe_after(self, identifier, slot, block, now):
        until, updated = block_signature(block)
        key = identifier, slot, until
        if not number(updated):
            updated = self.block_seen.setdefault(key, now)
        updated = max(updated, self.block_seen.get(key, updated))
        return min(until, updated + USAGE_REPORT_TTL)

    def allow_unattributed(self, provider, restricted):
        return not restricted and self.pool.get(provider, self.pool.get("*", (None, True))) == (None, True)

    def overlay(self, row, now):
        entry = self.usage_overlays.get(overlay_key(row))
        return entry[1] if entry and entry[0] > now else None

    async def heal_blocks(self, provider, candidates, restricted):
        reports = await self.usage(provider)
        now = self.server_time()
        clears = []
        for observed in candidates:
            row = self.rows.get(observed["broker_id"])
            if not row or row["kind"] != "oauth" or row["identity"] != observed["identity"] or (
                overlay_key(row) != overlay_key(observed)
            ):
                continue
            report = match_report(reports, row, self.allow_unattributed(provider, restricted))
            if report and now - report["fetchedAt"] > USAGE_REPORT_TTL:
                report = None
            report = merge_report(report, self.overlay(row, monotonic()))
            if not report or now - report["fetchedAt"] > USAGE_REPORT_TTL:
                continue
            identifier = row["broker_id"]
            healed = healable_scopes(provider, report)
            for slot, block in self.active_blocks(row, now).items():
                if slot[1] in healed and self.probe_after(identifier, slot, block, now) <= min(now, report["fetchedAt"]):
                    clears.append(self.clear_block(row, slot, block))
        if clears:
            await asyncio.gather(*clears)

    async def model_rows(self, provider, scope, accounts, model=None, *, heal=False):
        await self.ensure()
        scopes = request_scopes(provider, scope, model)

        def candidates():
            return [row for row in self.rows.values() if row["provider"] == provider and (
                accounts is None or row["kind"] == "oauth" and row["identity"] in accounts
            )]

        now = self.server_time()
        rows = candidates()
        if heal and provider in PROVIDERS:
            self.block_seen = {key: first for key, first in self.block_seen.items() if key[0] in self.rows and key[2] > now}
            protected = PROTECTED_SCOPES if provider == "anthropic" else PROTECTED_SCOPES | {""}
            probes = []
            for row in rows:
                if row["kind"] != "oauth":
                    continue
                active = self.active_blocks(row, now)
                if any(slot[1] in protected for slot in active):
                    continue
                if any(slot[1] in scopes and not slot[1].startswith("model-policy:")
                       and self.probe_after(row["broker_id"], slot, block, now) <= now for slot, block in active.items()):
                    probes.append(row)
            if probes:
                await self.heal_blocks(provider, probes, accounts is not None)
                rows, now = candidates(), self.server_time()
        available, quota_blocked = [], False
        for row in rows:
            blocking = {slot[1] for slot in self.active_blocks(row, now) if slot[1] in scopes}
            if not blocking:
                available.append(row)
            elif not any(scope in PROTECTED_SCOPES or scope.startswith("model-policy:") for scope in blocking):
                quota_blocked = True
        return available, quota_blocked

    async def clear_block(self, observed, slot, block):
        identifier = observed["broker_id"]
        async with self.block_locks.setdefault(identifier, asyncio.Lock()):
            current = self.rows.get(identifier)
            latest = self.active_blocks(current, self.server_time()).get(slot) if current else None
            if not latest or current["kind"] != "oauth" or current["identity"] != observed["identity"] or (
                overlay_key(current) != overlay_key(observed) or block_signature(latest) != block_signature(block)
            ):
                return
            try:
                await self.delete_block(identifier, slot, block)
            except AuthBrokerError:
                return
            self.deleted_blocks.setdefault(identifier, {})[slot] = block_signature(block)
            self.timed_blocks.pop((identifier, slot), None)
            pending = self.pending_blocks.get(identifier, {})
            if pending.get(slot) == block:
                pending.pop(slot)
            self.revision += 1
            self.local_revision += 1

    async def block_model(self, identifier, provider, scope, seconds=None, *, until=None, observed=None, timed=False):
        async with self.block_locks.setdefault(identifier, asyncio.Lock()):
            row = self.rows.get(identifier)
            if not row or row["provider"] != provider or observed is not None and (
                row["kind"] != "oauth" or row["identity"] != observed["identity"] or overlay_key(row) != overlay_key(observed)
            ):
                return
            now = self.server_time()
            until = int(now + seconds * 1000) if until is None else until
            key = f'{provider}:{row["kind"]}'
            slot = key, scope
            existing = self.active_blocks(row, now).get(slot)
            if existing and existing["blockedUntilMs"] >= until:
                if timed and existing["blockedUntilMs"] == until:
                    self.timed_blocks[identifier, slot] = until
                return
            if timed and until > now:
                self.timed_blocks[identifier, slot] = until
            else:
                self.timed_blocks.pop((identifier, slot), None)
            block = {"providerKey": key, "blockScope": scope, "blockedUntilMs": until}
            self.pending_blocks.setdefault(identifier, {})[slot] = {**block, "updatedAtMs": now}
            self.deleted_blocks.get(identifier, {}).pop(slot, None)
            self.usage_cache.pop(provider, None)
            self.revision += 1
            self.local_revision += 1
            try:
                await self.write_block(identifier, provider, block)
            except AuthBrokerError:
                pass

    def quota_candidates(self, provider, scope, accounts, model, kind):
        now = self.server_time()
        scopes = request_scopes(provider, scope, model)
        self.timed_blocks = {key: until for key, until in self.timed_blocks.items() if until > now}
        candidates = {}
        for identifier, row in self.rows.items():
            if (row["provider"] != provider or row["kind"] != kind
                or accounts is not None and (kind != "oauth" or row["identity"] not in accounts)):
                continue
            blocks = {slot: block["blockedUntilMs"] for slot, block in self.active_blocks(row, now).items()
                      if slot[1] in scopes}
            until = max(blocks.values(), default=None)
            timed = max((at for slot, at in blocks.items() if self.timed_blocks.get((identifier, slot)) == at), default=None)
            candidates[identifier] = until, timed
        return candidates

    async def mark_usage_limit(self, identifier, provider, scope, accounts, model, deadline, has_hint):
        await self.ensure()
        row = self.rows.get(identifier)
        if not row or row["provider"] != provider:
            return UsageBlock(deadline, deadline)
        kind = row["kind"]
        latest = report_reset = None
        if kind == "oauth":
            reports = await self.usage(provider) if provider in PROVIDERS else []
            report = match_report(reports, row, self.allow_unattributed(provider, accounts is not None))
            report = merge_report(report, self.overlay(row, monotonic()))
            latest, report_reset = usage_reset_times(provider, model, report, self.server_time())
        _, prior = self.quota_candidates(provider, scope, accounts, model, kind).get(identifier, (None, None))
        # A complete report replaces a hintless guess in both the wait and persisted block.
        requested = deadline + self.server_time() - time() * 1000
        planned = report_reset if report_reset is not None and not has_hint else requested
        until = max(planned, latest or 0, prior or 0)
        await self.block_model(identifier, provider, scope, until=int(until), observed=row if kind == "oauth" else None,
                               timed=has_hint or until in (latest, report_reset))
        actual, _ = self.quota_candidates(provider, scope, accounts, model, kind).get(identifier, (None, None))
        to_wall = time() * 1000 - self.server_time()
        return UsageBlock(deadline, actual + to_wall if actual is not None else None,
                          report_reset + to_wall if report_reset is not None else None,
                          prior + to_wall if prior is not None else None)

    async def usage_candidates(self, provider, scope, accounts, model, active):
        await self.ensure()
        row = self.rows.get(active)
        kind = row["kind"] if row else "oauth" if accounts is not None else "api_key"
        offset = time() * 1000 - self.server_time()
        return {identifier: tuple(value + offset if value is not None else None for value in pair)
                for identifier, pair in self.quota_candidates(provider, scope, accounts, model, kind).items()}

    async def usage(self, provider):
        if (cached := self.usage_cache.get(provider)) and cached[0] > monotonic():
            return cached[1]
        task = self.usage_tasks.get(provider)
        if task is None or task.done():
            task = self.usage_tasks[provider] = asyncio.create_task(self.fetch_usage(provider))
        try:
            return await asyncio.wait_for(asyncio.shield(task), timeout=5)
        except TimeoutError:
            return []

    async def invalidate_reset(self, provider, identifier):
        task = self.usage_tasks.pop(provider, None)
        if task and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self.usage_cache.pop(provider, None)
        if (row := self.rows.get(identifier)) and row["provider"] == provider:
            key = overlay_key(row)
            self.usage_overlays.pop(key, None)
            self.usage_ingested.pop(key, None)
        self.local_revision += 1

    async def note_usage(self, observed, limits):
        current = self.rows.get(observed["broker_id"])
        key = overlay_key(observed)
        if not key or not current or current["kind"] != "oauth" or overlay_key(current) != key:
            return
        now = monotonic()
        if now - self.usage_ingested.get(key, float("-inf")) < 60 and not any(exhausted(limit) for limit in limits):
            return
        self.usage_ingested = {key: at for key, at in self.usage_ingested.items() if at + 60 > now}
        self.usage_overlays = {key: entry for key, entry in self.usage_overlays.items() if entry[0] > now}
        self.usage_ingested[key] = now
        previous = self.usage_overlays.get(key)
        report = {"fetchedAt": (time() - self.clock_offset) * 1000, "limits": limits,
                  "metadata": {**current["account"], "source": "ratelimit-headers"}}
        self.usage_overlays[key] = now + 15, merge_report(previous[1] if previous else None, report)

    async def choose_oauth(self, provider, model, rows, pinned, warm, restricted, scope=None):
        policies = self.policies.get(provider, ())
        fetch = (provider in PROVIDERS or policies) and (
            len(rows) > 1 or policies or provider in self.usage_tasks or plan_requirement(provider, model)
        )
        reports = await self.usage(provider) if fetch else []
        now = monotonic()
        overlays = {overlay_key(row): report for row in rows if (report := self.overlay(row, now))}
        if not fetch and not overlays:
            return next((row["broker_id"] for row in rows if row["broker_id"] == pinned), rows[0]["broker_id"])
        blocked = {}
        chosen = rank_oauth(provider, model, rows, reports, self.server_time(), pinned, warm,
                            policies, self.default_reserve, self.allow_unattributed(provider, restricted), overlays, blocked)
        if scope is not None and blocked:
            await asyncio.gather(*(self.block_model(row["broker_id"], provider, scope,
                                                   until=blocked[row["broker_id"]], observed=row)
                                   for row in rows if row["broker_id"] in blocked))
        return chosen

    async def close(self):
        for task in self.usage_tasks.values():
            task.cancel()
        await asyncio.gather(*self.usage_tasks.values(), return_exceptions=True)

    async def refresh(self, observed, forced):
        async with self.refresh_locks.setdefault(observed["broker_id"], asyncio.Lock()):
            current = self.rows.get(observed["broker_id"])
            if not current or current["tokens"] != observed["tokens"] or current.get("retry_at", 0) > time():
                return current
            if not forced and current["expires"] > time() + self.refresh_skew:
                return current
            return await self.refresh_entry(current, forced)


class AuthBroker(CredentialPool):
    def __init__(self, url, token, profile, cache=None, model_providers=()):
        try:
            parsed = urlsplit(url)
            if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username is not None or (
                parsed.query or parsed.fragment
            ):
                raise ValueError()
            httpx2.URL(url)
        except (ValueError, httpx2.InvalidURL):
            raise ValueError("Invalid Auth Broker URL.") from None
        if not token.isascii() or any(char.isspace() or ord(char) < 32 for char in token):
            raise ValueError("Invalid Auth Broker bearer token.")
        super().__init__(profile, model_providers)
        self.url = url.rstrip("/")
        self.cache = cache
        self.observed_usage = ObservedUsage(self.request, config_root("default") / "install-id")
        self.headers = {"Authorization": "Bearer " + token, "Accept": "application/json",
                        "OMP-Auth-Broker-Capabilities": "codex-meter-block-scopes"}
        self.generation = None
        self.task = self.idle = None
        self.stream_supported = True

    def client(self, timeout=10):
        return httpx2.AsyncClient(headers=self.headers, timeout=timeout, follow_redirects=False, trust_env=False)

    async def request(self, method, path, **kwargs):
        timeout = kwargs.pop("timeout", 10)
        try:
            async with self.client(timeout) as client:
                response = await client.request(method, self.url + path, **kwargs)
        except (httpx2.HTTPError, httpx2.InvalidURL):
            raise AuthBrokerError("Auth Broker request failed.") from None
        if response.status_code != 304 and not response.is_success:
            raise AuthBrokerError(f"Auth Broker returned HTTP {response.status_code}.", response.status_code)
        return response

    async def write_block(self, identifier, provider, block):
        await self.request("POST", f"/v1/credential/{identifier}/block", json=block, timeout=2)

    async def delete_block(self, identifier, slot, block):
        await self.request("DELETE", f"/v1/credential/{identifier}/block", timeout=2,
                           json={"providerKey": slot[0], "blockScope": slot[1]})

    @staticmethod
    def payload(response):
        try:
            value = response.json()
            if isinstance(value, dict):
                return value
        except ValueError:
            pass
        raise AuthBrokerError("Auth Broker returned invalid JSON.")

    def snapshot(self, document, *, cached=False):
        generation, entries = document.get("generation"), document.get("credentials")
        if type(generation) is not int or generation < 0 or not isinstance(entries, list):
            raise AuthBrokerError("Auth Broker returned an invalid snapshot.")
        offset = self.clock_offset
        if number(server_now := document.get("serverNowMs")):
            received = min(time(), document["generatedAt"] / 1000) if cached else time()
            offset = received - server_now / 1000
        rows = [self.entry(entry, offset) for entry in entries]
        self.rows = {row["broker_id"]: row for row in rows if row}
        self.generation = generation
        self.clock_offset = offset
        self.revision += 1
        if self.cache and not cached:
            self.cache.schedule(document)

    async def fetch(self, wait=False):
        headers = {"If-None-Match": f'"{self.generation}"'} if wait and self.generation is not None else {}
        path = "/v1/snapshot?wait=30000" if wait else "/v1/snapshot"
        revision = self.revision
        local_revision = self.local_revision
        response = await self.request("GET", path, headers=headers, timeout=32 if wait else 10)
        if response.status_code == 304:
            return False
        document = self.payload(response)
        generation = document.get("generation")
        if self.local_revision != local_revision:
            return False
        if self.revision == revision or type(generation) is int and (
            self.generation is None or generation >= self.generation
        ):
            self.snapshot(document)
            return True
        return False

    def apply_event(self, name, document):
        if not isinstance(document, dict) or document.get("kind") != name:
            raise AuthBrokerError("Auth Broker returned an invalid snapshot event.")
        if name == "snapshot":
            self.snapshot({key: value for key, value in document.items() if key != "kind"})
            return
        generation = document.get("generation")
        if type(generation) is not int or name not in ("entry", "removed"):
            raise AuthBrokerError("Auth Broker returned an invalid snapshot delta.")
        if self.generation is None or generation < self.generation:
            return
        if number(server_now := document.get("serverNowMs")):
            self.clock_offset = time() - server_now / 1000
        if name == "entry":
            entry = self.entry(document.get("entry"))
            if entry:
                self.rows[entry["broker_id"]] = entry
            else:
                self.rows.pop(document["entry"]["id"], None)
        else:
            identifier = document.get("id")
            if type(identifier) is not int:
                raise AuthBrokerError("Auth Broker returned an invalid removal event.")
            self.rows.pop(identifier, None)
        self.generation = generation
        self.revision += 1

    async def follow(self):
        delay = 1
        try:
            async with asyncio.timeout(20) as self.idle:
                while True:
                    try:
                        if self.stream_supported:
                            revision, local_revision = self.revision, self.local_revision
                            async with self.client(35) as client:
                                async with client.sse(self.url + "/v1/snapshot/stream") as stream:
                                    if stream.response.status_code == 404:
                                        self.stream_supported = False
                                        continue
                                    stream.response.raise_for_status()
                                    async for event in stream:
                                        document = json.loads(event.data)
                                        if event.event == "snapshot" and self.revision != revision:
                                            generation = document.get("generation") if isinstance(document, dict) else None
                                            if type(generation) is int and (generation < self.generation or (
                                                self.local_revision != local_revision and generation == self.generation
                                            )):
                                                continue
                                        self.apply_event(event.event, document)
                                        delay = 1
                            await asyncio.sleep(delay)
                        else:
                            generation = self.generation
                            await self.fetch(wait=True)
                            if generation == self.generation:
                                await asyncio.sleep(0.25)
                            delay = 1
                    except (AuthBrokerError, httpx2.HTTPError, httpx2.SSEError, ValueError):
                        await asyncio.sleep(delay)
                        delay = min(delay * 2, 15)
        except TimeoutError:
            pass
        finally:
            self.idle = None

    async def ensure(self):
        async with self.snapshot_lock:
            if self.task is None or self.task.done():
                if self.generation is None:
                    cached = await asyncio.to_thread(self.cache.read) if self.cache else None
                    if cached is not None:
                        self.snapshot(cached, cached=True)
                    else:
                        await self.fetch()
                self.task = asyncio.create_task(self.follow())
            elif self.idle is not None and not self.idle.expired():
                self.idle.reschedule(asyncio.get_running_loop().time() + 20)

    async def fetch_usage(self, provider):
        try:
            response = await self.request("GET", "/v1/usage", params={"provider": provider})
            reports = usage_reports(self.payload(response), provider)
        except AuthBrokerError:
            reports = []
        self.usage_cache[provider] = monotonic() + 15, reports
        if reports and self.on_usage:
            self.on_usage(provider, reports)
        return reports

    async def close(self):
        if self.task:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        await super().close()
        await self.observed_usage.close()
        if self.cache:
            await self.cache.close()

    async def refresh_entry(self, observed, forced):
        identifier = observed["broker_id"]
        path = f"/v1/credential/{identifier}/refresh" + ("?reason=auth-recovery" if forced else "")
        try:
            response = await self.request("POST", path)
        except AuthBrokerError as error:
            if error.status == 404:
                self.rows.pop(identifier, None)
                self.revision += 1
                self.local_revision += 1
            elif (current := self.rows.get(identifier)) and current["tokens"] == observed["tokens"]:
                self.rows[identifier] = {**current, "retry_at": time() + 15}
            raise
        entry = self.entry(self.payload(response).get("entry"))
        if not entry or entry["kind"] != "oauth" or entry["broker_id"] != identifier or entry["provider"] != observed["provider"]:
            raise AuthBrokerError("Auth Broker refresh returned a different credential.")
        current = self.rows.get(identifier)
        if not current or current["tokens"] != observed["tokens"]:
            return current
        entry["blocks"] = current["blocks"]
        self.rows[identifier] = entry
        self.revision += 1
        self.local_revision += 1
        return entry

    async def upload(self, provider, record):
        tokens, client = record["tokens"], record.get("client", {})
        refresh = tokens.get("refresh_token", "")
        if refresh == REMOTE_REFRESH:
            raise AuthBrokerError("Cannot upload a remote refresh sentinel.")
        expires = record.get("expires") or time() + 3600
        credential = {"type": "oauth", "access": tokens["access_token"], "refresh": refresh,
                      "expires": int(expires * 1000)}
        credential.update({key: value for key, value in (
            ("tokenUrl", record.get("token_url")), ("resource", record.get("resource")),
            ("authorizationUrl", record.get("authorization_url")), ("scope", tokens.get("scope")),
            ("clientId", client.get("client_id")), ("clientSecret", client.get("client_secret")),
            ("clientIssuer", client.get("issuer")), ("clientAuthMethod", client.get("token_endpoint_auth_method")),
            ("redirectUris", client.get("redirect_uris")), ("miniPiAlias", record.get("alias")),
        ) if value is not None})
        response = await self.request("POST", "/v1/credential", json={"provider": provider, "credential": credential})
        entries = self.payload(response).get("entries")
        if not isinstance(entries, list) or not any(row and row["provider"] == provider for row in map(self.entry, entries)):
            raise AuthBrokerError("Auth Broker did not confirm the saved MCP credential.")
        await self.fetch()

    async def disable(self, providers):
        await self.fetch()
        removed = False
        for identifier, row in tuple(self.rows.items()):
            if row["provider"] not in providers:
                continue
            response = await self.request("POST", f"/v1/credential/{identifier}/disable", json={"cause": "logged out by user"})
            if self.payload(response).get("ok") is not True:
                raise AuthBrokerError("Auth Broker did not confirm credential removal.")
            self.rows.pop(identifier, None)
            self.revision += 1
            self.local_revision += 1
            removed = True
        if removed and self.cache:
            await self.fetch()
