"""Durable quota cache and per-account provider polling."""

import asyncio
import logging
import random
import sqlite3
from collections import defaultdict
from time import time

from auth_broker import AuthBrokerError
from auth_broker_cache import number
from local_oauth import DeadGrant, parse_object, text
from provider_quota import VERSIONS, ProbeRejected, fetch_quota
from reset_fence import bun_hash


REPORT_TTL = 300_000
RETENTION = 86_400_000
FAILURE_TTL = {"anthropic": 60_000, "openai-codex": 10_000}
logger = logging.getLogger(__name__)


def now_ms():
    return int(time() * 1000)


def supports_quota(record):
    provider, credential = record["provider"], record["credential"]
    return provider in VERSIONS and (credential["type"] == "oauth" or (
        provider == "anthropic" and "sk-ant-oat" in credential["key"]
    ))


def usage_identity(credential, secret=None):
    parts = [credential["type"]]
    values = {key: text(credential.get(key)) for key in (
        "accountId", "email", "orgId", "projectId", "enterpriseUrl", "region", "inferenceRegion", "activeOrganizationId",
    )}
    for field, label in (("accountId", "account"), ("email", "email"), ("orgId", "org"), ("projectId", "project"),
                         ("enterpriseUrl", "enterprise"), ("region", "region"), ("inferenceRegion", "inference"),
                         ("activeOrganizationId", "selected-org")):
        if value := values[field]:
            parts.append(f"{label}:{value.lower() if field in ('email', 'enterpriseUrl') else value}")
    if not any(values[field] for field in ("accountId", "email", "orgId")):
        parts.append(f"secret:{bun_hash(secret):x}" if secret else "anonymous")
    return "|".join(parts)


def last_good(entry, now):
    report = entry.get("value") if isinstance(entry, dict) else None
    if (isinstance(report, dict) and number(report.get("fetchedAt")) and now < report["fetchedAt"] + RETENTION
        and isinstance(report.get("limits"), list) and isinstance(report.get("metadata"), dict)):
        return {key: value for key, value in report.items() if key != "raw"}
    return None


def fresh(entry, now):
    return entry and number(entry.get("expiresAt")) and entry["expiresAt"] > now


class QuotaService:
    def __init__(self, database, oauth, refresh_locks):
        self.db, self.oauth, self.refresh_locks = database, oauth, refresh_locks
        self.tasks = {}
        self.generations = defaultdict(int)
        self.serial = set()
        self.serial_locks = defaultdict(asyncio.Lock)
        self.cache_writes = asyncio.Lock()

    async def invalidate(self, provider):
        providers = (provider,) if provider else VERSIONS
        async with self.cache_writes:
            await asyncio.to_thread(self.db.stale_usage, provider)
            for name in providers:
                self.generations[name] += 1
                self.serial.add(name)

    async def after_reset(self, record, scopes=(), preserve=()):
        provider = record["provider"]
        key, _ = await self.key(record)
        async with self.cache_writes:
            await asyncio.to_thread(self.db.apply_reset, record, key, scopes, preserve)
            self.generations[provider] += 1
            self.serial.add(provider)

    async def reports(self, records, provider=None):
        groups = defaultdict(list)
        for record in records:
            if (not provider or record["provider"] == provider) and supports_quota(record):
                groups[record["provider"]].append(record)

        async def collect(name, rows):
            lock = self.serial_locks[name]
            if name in self.serial or lock.locked():
                async with lock:
                    if name in self.serial:
                        self.serial.discard(name)
                        return [await self.get(row) for row in rows]
            return await asyncio.gather(*(self.get(row) for row in rows))

        results = await asyncio.gather(*(collect(name, rows) for name, rows in groups.items()))
        reports = [item for group in results for item in group if item is not None]
        if provider not in VERSIONS:
            cached = await asyncio.to_thread(self.db.cached_reports, provider)
            reports.extend(report for report in cached if report.get("provider") not in VERSIONS)
        return reports

    async def key(self, record):
        provider, credential = record["provider"], record["credential"]
        secret = credential.get("key")
        if credential["type"] == "oauth" and not any(text(credential.get(field)) for field in ("accountId", "email", "orgId")):
            raw = parse_object(await asyncio.to_thread(self.db.oauth, record["id"], provider))
            secret = text(raw.get("refresh")) or credential["access"]
        account = usage_identity(credential, secret)
        return f"{VERSIONS[provider]}:{provider}:default:{account}", account

    async def get(self, record):
        provider = record["provider"]
        try:
            key, account = await self.key(record)
            while True:
                generation = self.generations[provider]
                existing = self.tasks.get(key)
                if existing is None or existing[1].done():
                    task = asyncio.create_task(self.probe(record, key, account, generation))
                    self.tasks[key] = generation, task

                    def done(finished, key=key):
                        active = self.tasks.get(key)
                        if active and active[1] is finished:
                            self.tasks.pop(key, None)
                        if not finished.cancelled():
                            finished.exception()

                    task.add_done_callback(done)
                else:
                    generation, task = existing
                report = await asyncio.shield(task)
                if generation == self.generations[provider]:
                    return report
                rows = await asyncio.to_thread(self.db.entries, (provider,))
                record = next((row for row in rows if row["id"] == record["id"]), None)
                if record is None or not supports_quota(record):
                    return None
                key, account = await self.key(record)
        except sqlite3.Error:
            logger.warning("Unable to read or save a provider quota report.")
            return None

    async def probe(self, record, key, account, generation):
        provider = record["provider"]
        started = now_ms()
        cached, failure = await asyncio.to_thread(self.db.quota_state, key)
        now = now_ms()
        previous = last_good(cached, now)
        if fresh(cached, now) or fresh(failure, now):
            return previous
        report = None
        rejected = False
        try:
            async with asyncio.timeout(10):
                credential = record["credential"]
                if credential["type"] == "oauth" and credential["expires"] <= now + 60_000:
                    try:
                        async with self.refresh_locks.setdefault(record["id"], asyncio.Lock()):
                            await self.oauth.refresh(
                                {"broker_id": record["id"], "provider": provider,
                                 "tokens": {"access_token": credential["access"]},
                                 "credential_revision": record["credentialRevision"]},
                                False, disable_dead=False,
                            )
                    except DeadGrant:
                        rejected = credential["expires"] <= now_ms()
                    except AuthBrokerError:
                        pass
                    # Check expiry, disable, and re-login again after acquiring/awaiting the refresh lease.
                    rows = await asyncio.to_thread(self.db.entries, (provider,))
                    current = next((item for item in rows if item["id"] == record["id"]), None)
                    if current is None or not supports_quota(current):
                        return None
                    record = current
                    credential = current["credential"]
                if credential["type"] == "api_key":
                    credential = {**credential, "access": credential["key"]}
                if not rejected:
                    report = await fetch_quota(provider, credential)
        except ProbeRejected:
            rejected = True
        except TimeoutError:
            pass

        if report and provider == "anthropic" and previous and "resetCredits" not in report and "resetCredits" in previous:
            report["resetCredits"] = previous["resetCredits"]

        async with self.cache_writes:
            if generation != self.generations[provider]:
                return report
            now = now_ms()
            if report is None and not rejected:
                report = previous if previous and now < previous["fetchedAt"] + RETENTION else None
                failed = True
            else:
                failed = rejected
            ttl = FAILURE_TTL[provider] if failed else REPORT_TTL
            expires = now + int(ttl * random.uniform(.75, 1.25))
            stored = await asyncio.to_thread(self.db.save_quota, key, account, record, report, expires,
                                             failed=failed, rejected=rejected, started=started)
            if stored is False:
                return None
        return report

    async def close(self):
        tasks = tuple(task for _, task in self.tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
