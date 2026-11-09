"""Read and update the native Oh My Pi v8 SQLite model credential store."""

import asyncio
import json
import os
import sqlite3
import sys
from contextlib import closing
from pathlib import Path
from time import monotonic, time

from auth_broker import AuthBrokerError, CredentialPool
from auth_broker_cache import REMOTE_REFRESH, config_root, number
from auth_schema import CODEX_KEY, EPOCH_SECONDS, VERSION, bootstrap
from broker_usage import IDENTITY, healable_scopes, usage_reports
from local_oauth import LocalOAuth, credential_identifiers, credential_identity, fingerprint
from observed_usage import COUNTERS
from quota_service import QuotaService


OAUTH_METADATA = (*IDENTITY, "orgName", "scope", "apiEndpoint", "enterpriseUrl", "region", "inferenceRegion",
                  "activeOrganizationId", "tokenUrl", "authorizationUrl", "resource", "clientId", "clientIssuer",
                  "clientAuthMethod", "miniPiAlias")


def stored_identity_matches(provider, existing, existing_key, incoming, incoming_key):
    if not incoming_key:
        return False
    if incoming_key == (existing_key or credential_identity(provider, existing)):
        return True
    values = credential_identifiers(incoming)
    org = values.get("org")
    if not org or not existing_key or (incoming_key != f"org:{org}" and not incoming_key.endswith(f"|org:{org}")):
        return False
    if existing_key == f"org:{org}":
        return True
    same_org = existing_key.endswith(f"|org:{org}")
    old = credential_identifiers(existing) if same_org else {}
    return any(value != org and (existing_key in (f"{key}:{value}", f"{key}:{value}|org:{org}")
                                  or same_org and old.get(key) == value)
               for key, value in values.items() if key != "org")


def agent_db_path(profile):
    root = config_root(profile)
    override = os.getenv("PI_CODING_AGENT_DIR") if profile == "default" else None
    agent = Path(override).expanduser().resolve() if override else root / "agent"
    if agent != root / "agent":
        return agent / "agent.db"
    if sys.platform in ("linux", "darwin") and (xdg := os.getenv("XDG_DATA_HOME")):
        data = Path(xdg) / "omp"
        if profile != "default":
            data = data / "profiles" / profile
        if data.is_dir():
            return data / "agent.db"
    return agent / "agent.db"


def decode_object(raw):
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


class AgentCredentialDB:
    def __init__(self, path, *, create=False):
        self.path = path.resolve()
        self.uri = self.path.as_uri() + "?mode=rw"
        try:
            if create:
                bootstrap(self.path)
            with closing(self.connect()) as db:
                version = db.execute("SELECT version FROM auth_schema_version WHERE id=1").fetchone()
        except sqlite3.Error:
            raise AuthBrokerError(f"Cannot open Oh My Pi credential storage at {self.path}.") from None
        if version != (VERSION,):
            raise AuthBrokerError(f"Oh My Pi credential storage requires schema v{VERSION}.")

    def connect(self):
        return sqlite3.connect(self.uri, uri=True, timeout=2)

    def entries(self, providers=None, *, after_revision=None):
        if providers is not None and not providers:
            return []
        return self.read_entries(providers, after_revision)[1]

    def read_entries(self, providers=None, after_revision=None):
        now = int(time() * 1000)
        params = tuple(providers) if providers is not None else ()
        condition = " AND c.provider IN (" + ",".join("?" for _ in params) + ")" if params else ""
        with closing(self.connect()) as db:
            db.execute("BEGIN")
            revision, = db.execute("SELECT revision FROM auth_change_revision WHERE id=1").fetchone()
            if revision == after_revision:
                return revision, None
            active = db.execute(f"""
                SELECT id, provider, credential_type, data, identity_key
                FROM auth_credentials AS c
                WHERE disabled_cause IS NULL {condition}
                ORDER BY id
            """, params).fetchall()
            blocks = {}
            for identifier, key, scope, deadline, updated in db.execute(f"""
                SELECT b.credential_id, b.provider_key, b.block_scope, b.blocked_until_ms, b.updated_at
                FROM auth_credential_blocks AS b JOIN auth_credentials AS c ON c.id = b.credential_id
                WHERE c.disabled_cause IS NULL {condition}
                  AND b.blocked_until_ms > ? AND NOT (b.provider_key = ? AND b.block_scope = 'shared')
                ORDER BY b.credential_id, b.provider_key, b.block_scope
            """, (*params, now, CODEX_KEY)):
                if number(deadline) and number(updated):
                    blocks.setdefault(identifier, []).append({
                        "providerKey": key, "blockScope": scope, "blockedUntilMs": deadline, "updatedAtMs": updated * 1000,
                    })
        entries = []
        for identifier, provider, kind, raw, identity in active:
            data = decode_object(raw)
            if data is None:
                continue
            if kind == "api_key" and isinstance(data.get("key"), str) and data["key"]:
                credential = {"type": kind, "key": data["key"]}
                if data.get("source") == "login":
                    credential["source"] = "login"
            elif kind == "oauth" and isinstance(data.get("access"), str) and data["access"] and number(data.get("expires")):
                credential = {"type": kind, "access": data["access"], "refresh": REMOTE_REFRESH, "expires": data["expires"]}
                credential.update({key: data[key] for key in OAUTH_METADATA if isinstance(data.get(key), str)})
                if number(data.get("authorizedAt")):
                    credential["authorizedAt"] = data["authorizedAt"]
                if isinstance(uris := data.get("redirectUris"), list) and all(isinstance(uri, str) for uri in uris):
                    credential["redirectUris"] = uris
            else:
                continue
            entries.append({
                "id": identifier, "provider": provider, "credential": credential,
                "identityKey": identity if isinstance(identity, str) else None, "blocks": blocks.get(identifier, []),
                "credentialRevision": fingerprint(raw),
            })
        return revision, entries

    def cached_reports(self, provider=None, *, fresh=True):
        now = int(time() * 1000)
        with closing(self.connect()) as db:
            rows = db.execute("SELECT value FROM cache WHERE key GLOB 'usage_cache:report:*' AND expires_at > ?",
                              (now // 1000,)).fetchall()
        reports = []
        for (raw,) in rows:
            cached = decode_object(raw)
            value = cached.get("value") if cached else None
            if not isinstance(value, dict) or not number(value.get("fetchedAt")) or not isinstance(value.get("limits"), list):
                continue
            if provider is not None and value.get("provider") != provider:
                continue
            if not fresh or number(cached.get("expiresAt")) and cached["expiresAt"] > now:
                reports.append({key: item for key, item in value.items() if key != "raw"})
        return sorted(reports, key=lambda report: report["fetchedAt"], reverse=True)

    def stale_usage(self, provider):
        with closing(self.connect()) as db, db:
            db.execute("""
                UPDATE cache SET value=json_set(value,'$.expiresAt',0)
                WHERE key GLOB 'usage_cache:report:*' AND json_valid(value)
                  AND (? IS NULL OR json_extract(value,'$.value.provider')=?)
            """, (provider, provider))

    def pending_reset(self, key):
        with closing(self.connect()) as db:
            row = db.execute("SELECT value FROM cache WHERE key=? AND expires_at>?",
                             ("mini_pi:reset_pending:" + key, int(time()))).fetchone()
        return decode_object(row[0]) if row else None

    def save_pending_reset(self, key, pending):
        key = "mini_pi:reset_pending:" + key
        with closing(self.connect()) as db, db:
            if pending is None:
                db.execute("DELETE FROM cache WHERE key=?", (key,))
            else:
                db.execute("""
                    INSERT INTO cache(key,value,expires_at) VALUES(?,?,?)
                    ON CONFLICT(key) DO UPDATE SET value=excluded.value,expires_at=excluded.expires_at
                """, (key, json.dumps(pending, separators=(",", ":")), int(time()) + 86_400))

    def apply_reset(self, record, quota_key, scopes, preserve):
        provider, identifier = record["provider"], record["id"]
        report_key = "usage_cache:report:" + quota_key
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            now = int(time() * 1000)
            db.execute("""
                INSERT INTO cache(key,value,expires_at) VALUES (?,?,?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value,expires_at=excluded.expires_at
            """, ("mini_pi:quota_reset_at:" + provider, str(now), now // 1000 + 60))
            db.execute("""
                UPDATE cache SET value=json_set(value,'$.expiresAt',0)
                WHERE key GLOB 'usage_cache:report:*' AND CASE WHEN json_valid(value)
                    THEN json_extract(value,'$.value.provider')=? ELSE 0 END
            """, (provider,))
            db.execute("""
                UPDATE cache SET value=json_remove(json_set(value,'$.expiresAt',0),'$.value.resetCredits')
                WHERE key=? AND json_valid(value)
            """, (report_key,))
            db.execute("""
                DELETE FROM cache WHERE key GLOB ? OR key GLOB ?
            """, (f"usage_cache:failure:report:*:{provider}:*", f"usage_cache:failure:report:{provider}:*"))
            row = db.execute("""
                SELECT data FROM auth_credentials WHERE id=? AND provider=? AND credential_type='oauth' AND disabled_cause IS NULL
            """, (identifier, provider)).fetchone()
            if not row or fingerprint(row[0]) != record["credentialRevision"]:
                return
            # A newer block can arrive while the provider processes the reset.
            for block in record["blocks"]:
                key, scope = block["providerKey"], block["blockScope"]
                if key in (provider, provider + ":oauth") and scope in scopes:
                    db.execute("""
                        DELETE FROM auth_credential_blocks WHERE credential_id=? AND provider_key=? AND block_scope=?
                            AND blocked_until_ms=? AND updated_at=?
                    """, (identifier, key, scope, block["blockedUntilMs"], int(block["updatedAtMs"] // 1000)))
            if not scopes or not preserve:
                return
            if db.execute("""
                SELECT 1 FROM auth_credential_blocks WHERE credential_id=? AND provider_key IN (?,?)
                  AND block_scope='' AND blocked_until_ms>?
            """, (identifier, provider, provider + ":oauth", now)).fetchone():
                return
            db.executemany(f"""
                INSERT INTO auth_credential_blocks(credential_id,provider_key,block_scope,blocked_until_ms,updated_at)
                VALUES(?,?,?,?,{EPOCH_SECONDS})
                ON CONFLICT(credential_id,provider_key,block_scope) DO UPDATE SET
                    blocked_until_ms=excluded.blocked_until_ms,updated_at=excluded.updated_at
                WHERE excluded.blocked_until_ms>auth_credential_blocks.blocked_until_ms
            """, ((identifier, provider + ":oauth", scope, until) for scope, until in preserve if until > now))

    def quota_state(self, key):
        now = int(time() * 1000)
        with closing(self.connect()) as db:
            rows = dict(db.execute("SELECT key,value FROM cache WHERE key IN (?,?) AND expires_at>?",
                                   (f"usage_cache:report:{key}", f"usage_cache:failure:report:{key}", now // 1000)))
        return decode_object(rows.get(f"usage_cache:report:{key}")), decode_object(
            rows.get(f"usage_cache:failure:report:{key}"),
        )

    def save_quota(self, key, account, record, report, expires, *, failed=False, rejected=False, started=None):
        identifier, provider = record["id"], record["provider"]
        now = int(time() * 1000)
        report_key, failure_key = f"usage_cache:report:{key}", f"usage_cache:failure:report:{key}"
        retention = (report["fetchedAt"] + 86_400_000) if report else expires
        values = [(report_key, json.dumps({"value": report, "expiresAt": min(expires, retention)}, separators=(",", ":")),
                   retention // 1000)]
        if failed:
            values.append((failure_key, json.dumps({"value": None, "expiresAt": expires}), expires // 1000))
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            if started is not None and db.execute("""
                SELECT 1 FROM cache WHERE key=? AND expires_at>? AND CAST(value AS INTEGER)>=?
            """, ("mini_pi:quota_reset_at:" + provider, now // 1000, started)).fetchone():
                return False
            row = db.execute("""
                SELECT data FROM auth_credentials WHERE id=? AND provider=? AND disabled_cause IS NULL
            """, (identifier, provider)).fetchone()
            if not row or fingerprint(row[0]) != record["credentialRevision"]:
                return
            current = set(db.execute("""
                SELECT provider_key,block_scope,blocked_until_ms,updated_at FROM auth_credential_blocks
                WHERE credential_id=? AND blocked_until_ms>? AND NOT (provider_key=? AND block_scope='shared')
            """, (identifier, now, CODEX_KEY)))
            observed = {(block["providerKey"], block["blockScope"], block["blockedUntilMs"], block["updatedAtMs"] // 1000)
                        for block in record["blocks"] if block["blockedUntilMs"] > now}
            unchanged = current == observed
            if not unchanged:
                # A probe that raced a new block cannot replace the report or heal the block.
                values = values[1:] if failed and not rejected else []
            db.executemany("""
                INSERT INTO cache(key,value,expires_at) VALUES (?,?,?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value,expires_at=excluded.expires_at
            """, values)
            if unchanged and not failed and report:
                db.execute("DELETE FROM cache WHERE key=?", (failure_key,))
                self.reconcile_quota(db, identifier, report)
                try:
                    self.record_quota_history(db, account, report)
                except sqlite3.Error:
                    pass

    @staticmethod
    def reconcile_quota(db, identifier, report):
        provider = report["provider"]
        now = int(time() * 1000)
        if now - report["fetchedAt"] > 300_000:
            return
        healed = healable_scopes(provider, report)
        rows = db.execute("""
            SELECT provider_key,block_scope,blocked_until_ms,updated_at FROM auth_credential_blocks
            WHERE credential_id=? AND provider_key IN (?,?) AND blocked_until_ms>?
              AND NOT (provider_key=? AND block_scope='shared')
        """, (identifier, provider, f"{provider}:oauth", now, CODEX_KEY)).fetchall()
        ready = min(now, report["fetchedAt"])
        db.executemany("""
            DELETE FROM auth_credential_blocks WHERE credential_id=? AND provider_key=? AND block_scope=?
        """, ((identifier, key, scope) for key, scope, until, updated in rows
              if scope in healed and min(until, updated * 1000 + 300_000) <= ready))

    @staticmethod
    def record_quota_history(db, account, report):
        recorded, provider = report["fetchedAt"], report["provider"]
        metadata = report["metadata"]
        for limit in report["limits"]:
            detail = limit.get("window", {})
            scope = limit["scope"]
            values = (recorded, metadata.get("email"), scope.get("accountId") or metadata.get("accountId"),
                      limit["label"], detail.get("label") or scope.get("windowId"),
                      limit["amount"].get("usedFraction"), limit.get("status"), detail.get("resetsAt"))
            prior = db.execute("""
                SELECT id,recorded_at FROM usage_history WHERE provider=? AND account_key=? AND limit_id=?
                ORDER BY recorded_at DESC LIMIT 1
            """, (provider, account, limit["id"])).fetchone()
            if prior and prior[1] // 3_600_000 == recorded // 3_600_000:
                db.execute("""
                    UPDATE usage_history SET recorded_at=?,email=?,account_id=?,label=?,window_label=?,used_fraction=?,
                        status=?,resets_at=? WHERE id=?
                """, (*values, prior[0]))
            else:
                db.execute("""
                    INSERT INTO usage_history(recorded_at,email,account_id,label,window_label,used_fraction,status,resets_at,
                        provider,account_key,limit_id) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """, (*values, provider, account, limit["id"]))

    def usage_history(self, provider, since):
        keys = ("recordedAt", "provider", "accountKey", "email", "accountId", "limitId", "label", "windowLabel",
                "usedFraction", "status", "resetsAt")
        with closing(self.connect()) as db:
            rows = db.execute("""
                SELECT recorded_at,provider,account_key,email,account_id,limit_id,label,window_label,used_fraction,status,resets_at
                FROM usage_history WHERE recorded_at>=? AND (? IS NULL OR provider=?) ORDER BY recorded_at
            """, (since, provider, provider)).fetchall()
        return [{key: value for key, value in zip(keys, row) if value is not None} for row in rows]

    def record_client_usage(self, payload):
        now = int(time() * 1000)
        install, app = payload["installId"], payload.get("app", "").strip()
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("""
                INSERT INTO clients (install_id,hostname,first_seen,last_seen) VALUES (?,?,?,?)
                ON CONFLICT(install_id) DO UPDATE SET hostname=COALESCE(excluded.hostname,hostname),last_seen=excluded.last_seen
            """, (install, payload.get("hostname"), now, now))
            for entry in payload["entries"]:
                identity = install, app, entry["provider"], entry["model"]
                at = entry["at"]
                counters = tuple(entry[key] for key in COUNTERS)
                bucket = db.execute("""
                    SELECT id FROM client_usage WHERE install_id=? AND app=? AND provider=? AND model=? AND recorded_at>=?
                    ORDER BY recorded_at DESC LIMIT 1
                """, (*identity, at - 300_000)).fetchone()
                if bucket:
                    db.execute("""
                        UPDATE client_usage SET recorded_at=?, requests=requests+?, input_tokens=input_tokens+?,
                        output_tokens=output_tokens+?, cache_read_tokens=cache_read_tokens+?,
                        cache_write_tokens=cache_write_tokens+?, cost_usd=cost_usd+? WHERE id=?
                    """, (at, *counters, bucket[0]))
                else:
                    db.execute("""
                        INSERT INTO client_usage (recorded_at,install_id,app,provider,model,requests,input_tokens,
                            output_tokens,cache_read_tokens,cache_write_tokens,cost_usd) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                    """, (at, *identity, *counters))

    def client_usage(self, since):
        with closing(self.connect()) as db:
            db.execute("BEGIN")
            clients = db.execute("SELECT install_id,hostname,first_seen,last_seen FROM clients ORDER BY last_seen DESC").fetchall()
            rows = db.execute("""
                SELECT install_id,app,provider,SUM(requests),SUM(input_tokens),SUM(output_tokens),SUM(cache_read_tokens),
                    SUM(cache_write_tokens),SUM(cost_usd)
                FROM client_usage WHERE recorded_at>=? GROUP BY install_id,app,provider
                ORDER BY install_id,SUM(input_tokens+output_tokens+cache_read_tokens+cache_write_tokens) DESC
            """, (since,)).fetchall()
        aggregates = {}
        for install, app, provider, *counts in rows:
            item = {"provider": provider, **dict(zip(COUNTERS, counts))}
            if app:
                item["app"] = app
            aggregates.setdefault(install, []).append(item)
        return [{"installId": install, **({"hostname": host} if host is not None else {}), "firstSeen": first,
                 "lastSeen": last, "providers": aggregates.get(install, [])} for install, host, first, last in clients]

    def upsert(self, provider, credential):
        kind = credential["type"]
        data = {key: value for key, value in credential.items() if key != "type"}
        raw = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        identity = credential_identity(provider, data) if kind == "oauth" else None
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            if kind == "oauth":
                db.execute(f"""
                    UPDATE auth_credentials SET disabled_cause='replaced by oauth login',updated_at={EPOCH_SECONDS}
                    WHERE provider=? AND credential_type='api_key' AND disabled_cause IS NULL
                """, (provider,))
            existing = db.execute("""
                SELECT id,data,identity_key,disabled_cause FROM auth_credentials
                WHERE provider=? AND credential_type=? ORDER BY id
            """, (provider, kind)).fetchall()
            active, disabled = None, []
            for identifier, current_raw, key, cause in existing:
                current = decode_object(current_raw)
                if current is None:
                    continue
                matches = (current.get("key") == data["key"] if kind == "api_key"
                           else stored_identity_matches(provider, current, key, data, identity))
                if cause is not None:
                    if matches or kind == "api_key":
                        disabled.append((identifier,))
                elif matches and active is None:
                    active = identifier, current_raw, key
                elif matches:
                    disabled.append((identifier,))
            if active is None:
                db.execute("""
                    INSERT INTO auth_credentials(provider,credential_type,data,identity_key) VALUES (?,?,?,?)
                """, (provider, kind, raw, identity))
            elif active[1:] != (raw, identity):
                db.execute(f"""
                    UPDATE auth_credentials SET data=?, identity_key=?, updated_at={EPOCH_SECONDS} WHERE id=?
                """, (raw, identity, active[0]))
            db.executemany("DELETE FROM auth_credentials WHERE id=?", disabled)

    def disable(self, identifier, cause):
        with closing(self.connect()) as db, db:
            return bool(db.execute(f"""
                UPDATE auth_credentials SET disabled_cause=?, updated_at={EPOCH_SECONDS}
                WHERE id=? AND disabled_cause IS NULL
            """, (cause, identifier)).rowcount)

    def disabled(self, provider):
        with closing(self.connect()) as db:
            rows = db.execute("""
                SELECT id,provider,credential_type,data,disabled_cause,updated_at FROM auth_credentials
                WHERE disabled_cause IS NOT NULL AND (? IS NULL OR provider=?) ORDER BY id
            """, (provider, provider)).fetchall()
        result = []
        for identifier, name, kind, raw, cause, updated in rows:
            item = {"id": identifier, "provider": name, "type": kind, "cause": cause, "disabledAtMs": updated * 1000}
            if kind == "oauth" and (credential := decode_object(raw)):
                item.update({key: credential[key] for key in ("email", "accountId", "orgId", "orgName")
                             if isinstance(credential.get(key), str)})
            result.append(item)
        return result

    def write_block(self, identifier, provider, block):
        key, scope, until = (block[field] for field in ("providerKey", "blockScope", "blockedUntilMs"))
        scopes = ("chat", "spark") if key == CODEX_KEY and scope == "shared" else (scope,)
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM auth_credential_blocks WHERE blocked_until_ms <= ?", (int(time() * 1000),))
            changed = 0
            for name in scopes:
                changed += db.execute(f"""
                    INSERT INTO auth_credential_blocks (credential_id, provider_key, block_scope, blocked_until_ms, updated_at)
                    SELECT id, ?, ?, ?, {EPOCH_SECONDS} FROM auth_credentials
                    WHERE id = ? AND provider = ? AND disabled_cause IS NULL
                    ON CONFLICT(credential_id, provider_key, block_scope) DO UPDATE SET
                        blocked_until_ms = MAX(blocked_until_ms, excluded.blocked_until_ms),
                        updated_at = excluded.updated_at
                """, (key, name, until, identifier, provider)).rowcount
            return bool(changed)

    def delete_blocks(self, identifier, slot=None):
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            if not db.execute("SELECT 1 FROM auth_credentials WHERE id=? AND disabled_cause IS NULL", (identifier,)).fetchone():
                return False
            if slot is None:
                db.execute("DELETE FROM auth_credential_blocks WHERE credential_id=?", (identifier,))
            else:
                db.execute("DELETE FROM auth_credential_blocks WHERE credential_id=? AND provider_key=? AND block_scope=?",
                           (identifier, *slot))
            return True

    def delete_block(self, identifier, slot, block):
        with closing(self.connect()) as db, db:
            db.execute("""
                DELETE FROM auth_credential_blocks
                WHERE credential_id = ? AND provider_key = ? AND block_scope = ?
                  AND blocked_until_ms = ? AND updated_at = ?
            """, (identifier, *slot, block["blockedUntilMs"], int(block["updatedAtMs"] // 1000)))

    def oauth(self, identifier, provider):
        with closing(self.connect()) as db:
            row = db.execute("""
                SELECT data FROM auth_credentials
                WHERE id=? AND provider=? AND credential_type='oauth' AND disabled_cause IS NULL
            """, (identifier, provider)).fetchone()
        return row[0] if row else None

    def claim_refresh(self, identifier, owner, ttl):
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            now = int(time() * 1000)
            prior = db.execute("""
                SELECT owner, expires_at_ms FROM auth_credential_refresh_leases WHERE credential_id=?
            """, (identifier,)).fetchone()
            claimed = db.execute(f"""
                INSERT INTO auth_credential_refresh_leases (credential_id, owner, expires_at_ms, updated_at)
                VALUES (?, ?, ?, {EPOCH_SECONDS})
                ON CONFLICT(credential_id) DO UPDATE SET
                    owner=excluded.owner, expires_at_ms=excluded.expires_at_ms, updated_at=excluded.updated_at
                WHERE auth_credential_refresh_leases.expires_at_ms <= ?
            """, (identifier, owner, now + ttl, now)).rowcount
            return bool(claimed), prior

    def renew_refresh(self, identifier, owner, ttl):
        with closing(self.connect()) as db, db:
            now = int(time() * 1000)
            return bool(db.execute(f"""
                UPDATE auth_credential_refresh_leases SET expires_at_ms=?, updated_at={EPOCH_SECONDS}
                WHERE credential_id=? AND owner=?
            """, (now + ttl, identifier, owner)).rowcount)

    def leave_refresh(self, identifier, owner, prior=None):
        with closing(self.connect()) as db, db:
            if prior:
                db.execute(f"""
                    UPDATE auth_credential_refresh_leases SET owner=?, expires_at_ms=?, updated_at={EPOCH_SECONDS}
                    WHERE credential_id=? AND owner=?
                """, (*prior, identifier, owner))
            else:
                db.execute("DELETE FROM auth_credential_refresh_leases WHERE credential_id=? AND owner=?",
                           (identifier, owner))

    def commit_refresh(self, identifier, provider, owner, expected, *, replacement=None, identity=None, disabled=None):
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            now = int(time() * 1000)
            fields = "data=?, identity_key=?" if replacement is not None else "disabled_cause=?"
            values = (replacement, identity) if replacement is not None else (disabled,)
            return bool(db.execute(f"""
                UPDATE auth_credentials SET {fields}, updated_at={EPOCH_SECONDS}
                WHERE id=? AND provider=? AND credential_type='oauth' AND disabled_cause IS NULL AND data=?
                  AND EXISTS (
                    SELECT 1 FROM auth_credential_refresh_leases
                    WHERE credential_id=? AND owner=? AND expires_at_ms > ?
                  )
            """, (*values, identifier, provider, expected, identifier, owner, now)).rowcount)


class LocalCredentials(CredentialPool):
    refresh_skew = 60

    def __init__(self, path, profile, model_providers):
        super().__init__(profile, model_providers)
        self.database = AgentCredentialDB(path)
        self.oauth = LocalOAuth(self.database, self.refresh_skew)
        self.quota = QuotaService(self.database, self.oauth, self.refresh_locks)

    async def sqlite(self, operation, *args):
        try:
            return await asyncio.to_thread(operation, *args)
        except sqlite3.Error:
            raise AuthBrokerError("Oh My Pi credential storage operation failed.") from None

    async def ensure(self):
        async with self.snapshot_lock:
            records = await self.sqlite(self.database.entries, self.model_providers)
            self.rows = {row["broker_id"]: {**row, "credential_revision": record["credentialRevision"]}
                         for record in records if (row := self.entry(record))}

    async def write_block(self, identifier, provider, block):
        await self.sqlite(self.database.write_block, identifier, provider, block)
        if block["providerKey"] == CODEX_KEY and block["blockScope"] == "shared":
            self.pending_blocks.get(identifier, {}).pop((CODEX_KEY, "shared"), None)
            await self.ensure()

    async def delete_block(self, identifier, slot, block):
        await self.sqlite(self.database.delete_block, identifier, slot, block)
        await self.ensure()

    def allow_unattributed(self, provider, restricted):
        return False

    async def fetch_usage(self, provider):
        try:
            entries = await self.sqlite(self.database.entries, (provider,))
            visible = [record for record in entries if self.entry(record)]
            reports = usage_reports({"reports": await self.quota.reports(visible, provider)}, provider)
        except (AuthBrokerError, sqlite3.Error):
            reports = []
        self.usage_cache[provider] = monotonic() + 15, reports
        if reports and self.on_usage:
            self.on_usage(provider, reports)
        return reports

    async def refresh_entry(self, observed, forced):
        try:
            await self.oauth.refresh(observed, forced)
        except AuthBrokerError:
            await self.ensure()
            current = self.rows.get(observed["broker_id"])
            if current and current["kind"] == "oauth" and current["expires"] > time() and (
                current["credential_revision"] != observed["credential_revision"]
            ):
                return current
            raise
        await self.ensure()
        current = self.rows.get(observed["broker_id"])
        if not current or current["kind"] != "oauth":
            return None
        return current

    async def close(self):
        await super().close()
        await self.quota.close()
        await self.oauth.close()
