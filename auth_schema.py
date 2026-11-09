"""Bootstrap Oh My Pi's native credential schema on a new SQLite vault."""

import os
import sqlite3
from contextlib import closing


VERSION = 8
EPOCH_SECONDS = "CAST(strftime('%s','now') AS INTEGER)"
CODEX_KEY = "openai-codex:oauth"


def mirror_trigger(event, shared):
    row = "OLD" if event == "DELETE" else "NEW"
    fields = "credential_id, provider_key, block_scope, blocked_until_ms, updated_at"
    target = f"credential_id={row}.credential_id AND provider_key={row}.provider_key"
    meters = "block_scope IN ('chat','spark')"
    if shared and event == "DELETE":
        operation = f"DELETE FROM auth_credential_blocks WHERE {target} AND {meters};"
    elif shared:
        operation = "".join(f"""
            INSERT INTO auth_credential_blocks ({fields})
            VALUES ({row}.credential_id,{row}.provider_key,'{scope}',{row}.blocked_until_ms,{row}.updated_at)
            ON CONFLICT(credential_id,provider_key,block_scope) DO UPDATE SET
                blocked_until_ms=MAX(auth_credential_blocks.blocked_until_ms,excluded.blocked_until_ms),
                updated_at=MAX(auth_credential_blocks.updated_at,excluded.updated_at);
        """ for scope in ("chat", "spark"))
    else:
        operation = f"""
            DELETE FROM auth_credential_blocks WHERE {target} AND block_scope='shared';
            INSERT INTO auth_credential_blocks ({fields})
            SELECT credential_id,provider_key,'shared',MAX(blocked_until_ms),MAX(updated_at)
            FROM auth_credential_blocks WHERE {target} AND {meters}
            GROUP BY credential_id,provider_key;
        """
    name = f"shared_{event.lower()}_to_meters" if shared else f"meter_{event.lower()}_to_shared"
    scope_check = "block_scope='shared'" if shared else meters
    return f"""
        CREATE TRIGGER IF NOT EXISTS auth_codex_{name} AFTER {event} ON auth_credential_blocks
        WHEN {row}.provider_key='{CODEX_KEY}' AND {row}.{scope_check} AND NOT EXISTS (
            SELECT 1 FROM auth_credential_block_mirror_guard WHERE credential_id={row}.credential_id
        )
        BEGIN
            INSERT OR IGNORE INTO auth_credential_block_mirror_guard (credential_id) VALUES ({row}.credential_id);
            {operation}
            DELETE FROM auth_credential_block_mirror_guard WHERE credential_id={row}.credential_id;
        END;
    """


def bootstrap(path):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
    except FileExistsError:
        pass
    with closing(sqlite3.connect(path, timeout=5)) as db:
        tables = {name for (name,) in db.execute("""
            SELECT name FROM sqlite_schema WHERE type='table' AND name IN ('auth_credentials','auth_schema_version')
        """)}
        if tables:
            return
        db.execute("PRAGMA journal_mode=WAL")
        db.executescript(f"""
            BEGIN IMMEDIATE;
            CREATE TABLE IF NOT EXISTS auth_schema_version (id INTEGER PRIMARY KEY CHECK(id=1), version INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS auth_credentials (
                id INTEGER PRIMARY KEY AUTOINCREMENT, provider TEXT NOT NULL, credential_type TEXT NOT NULL,
                data TEXT NOT NULL, disabled_cause TEXT DEFAULT NULL, identity_key TEXT DEFAULT NULL,
                created_at INTEGER NOT NULL DEFAULT ({EPOCH_SECONDS}), updated_at INTEGER NOT NULL DEFAULT ({EPOCH_SECONDS})
            );
            CREATE INDEX IF NOT EXISTS idx_auth_provider ON auth_credentials(provider);
            CREATE INDEX IF NOT EXISTS idx_auth_provider_identity ON auth_credentials(provider,identity_key)
                WHERE identity_key IS NOT NULL;
            CREATE TABLE IF NOT EXISTS auth_credential_blocks (
                credential_id INTEGER NOT NULL, provider_key TEXT NOT NULL, block_scope TEXT NOT NULL DEFAULT '',
                blocked_until_ms INTEGER NOT NULL, updated_at INTEGER NOT NULL,
                PRIMARY KEY (credential_id,provider_key,block_scope)
            );
            CREATE INDEX IF NOT EXISTS idx_auth_credential_blocks_expires ON auth_credential_blocks(blocked_until_ms);
            CREATE TABLE IF NOT EXISTS auth_credential_refresh_leases (
                credential_id INTEGER PRIMARY KEY, owner TEXT NOT NULL, expires_at_ms INTEGER NOT NULL, updated_at INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_auth_credential_refresh_leases_expires
                ON auth_credential_refresh_leases(expires_at_ms);
            CREATE TABLE IF NOT EXISTS auth_credential_block_mirror_guard (credential_id INTEGER PRIMARY KEY) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS auth_change_revision (id INTEGER PRIMARY KEY CHECK(id=1), revision INTEGER NOT NULL);
            INSERT OR IGNORE INTO auth_change_revision VALUES (1,0);
            CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT NOT NULL, expires_at INTEGER NOT NULL);
            CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache(expires_at);
            CREATE TABLE IF NOT EXISTS usage_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT, recorded_at INTEGER NOT NULL, provider TEXT NOT NULL,
                account_key TEXT NOT NULL, email TEXT, account_id TEXT, limit_id TEXT NOT NULL, label TEXT NOT NULL,
                window_label TEXT, used_fraction REAL, status TEXT, resets_at INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_usage_history_series ON usage_history(provider,account_key,limit_id,recorded_at);
            CREATE INDEX IF NOT EXISTS idx_usage_history_recorded ON usage_history(recorded_at);
            CREATE TABLE IF NOT EXISTS clients (
                install_id TEXT PRIMARY KEY, hostname TEXT, first_seen INTEGER NOT NULL, last_seen INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS client_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT, recorded_at INTEGER NOT NULL, install_id TEXT NOT NULL,
                app TEXT NOT NULL DEFAULT '', provider TEXT NOT NULL, model TEXT NOT NULL, requests INTEGER NOT NULL,
                input_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL, cache_read_tokens INTEGER NOT NULL,
                cache_write_tokens INTEGER NOT NULL, cost_usd REAL NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_client_usage_series ON client_usage(install_id,provider,model,recorded_at);
            CREATE INDEX IF NOT EXISTS idx_client_usage_recorded ON client_usage(recorded_at);
            {"".join(mirror_trigger(event, shared) for event in ("INSERT", "UPDATE", "DELETE") for shared in (True, False))}
            {"".join(f'''
                CREATE TRIGGER IF NOT EXISTS auth_change_revision_{table}_{event.lower()} AFTER {event} ON {table}
                BEGIN UPDATE auth_change_revision SET revision=revision+1 WHERE id=1; END;
            ''' for table in ("auth_credentials", "auth_credential_blocks") for event in ("INSERT", "UPDATE", "DELETE"))}
            INSERT OR IGNORE INTO auth_schema_version VALUES (1,{VERSION});
            COMMIT;
        """)
