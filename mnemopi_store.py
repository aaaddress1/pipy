"""Oh My Pi BEAM bank storage and on-disk SQLite compatibility."""

import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from itertools import islice
from pathlib import Path
from uuid import uuid4

from mnemopi_legacy_vectors import delete_legacy_vectors
from mnemopi_veracity import unlink_fact_source
from reset_fence import bun_hash
from session_jsonl import timestamp


DB_NAME = "mnemopi.db"
SCHEMA = Path(__file__).with_name("mnemopi_schema.sql")
MEMORY_TABLES = {"working": "working_memory", "episodic": "episodic_memory"}
LIVE_EMBEDDING = """
    EXISTS(SELECT 1 FROM working_memory WHERE id=? AND coalesce(embed_text,content)=?
           AND superseded_by IS NULL AND (valid_until IS NULL OR valid_until>?))
    OR EXISTS(SELECT 1 FROM episodic_memory WHERE id=? AND content=?
              AND superseded_by IS NULL AND (valid_until IS NULL OR valid_until>?))
"""
PROJECTIONS = ("memoria_facts", "memoria_timelines", "memoria_instructions", "memoria_preferences", "memoria_kg")
COMMON_COLUMNS = {
    "veracity": "TEXT DEFAULT 'unknown'",
    "memory_type": "TEXT DEFAULT 'unknown'",
    "recall_count": "INTEGER DEFAULT 0",
    "last_recalled": "TIMESTAMP DEFAULT NULL",
    "valid_until": "TIMESTAMP DEFAULT NULL",
    "superseded_by": "TEXT DEFAULT NULL",
    "scope": "TEXT DEFAULT 'global'",
    "author_id": "TEXT DEFAULT NULL",
    "author_type": "TEXT DEFAULT NULL",
    "channel_id": "TEXT DEFAULT NULL",
    "trust_tier": "TEXT DEFAULT 'STATED'",
    "validator": "TEXT DEFAULT NULL",
    "validated_at": "TIMESTAMP DEFAULT NULL",
    "validation_count": "INTEGER DEFAULT 0",
    "event_date": "TEXT DEFAULT NULL",
    "event_date_precision": "TEXT DEFAULT 'unknown'",
    "temporal_tags": "TEXT DEFAULT '[]'",
    "corrected_by": "INTEGER DEFAULT NULL",
}
UPGRADES = {
    "episodic_memory": {
        "tier": "INTEGER DEFAULT 1",
        "degraded_at": "TEXT",
        "binary_vector": "BLOB",
    },
    "working_memory": {
        "embed_text": "TEXT DEFAULT NULL",
        "consolidated_at": "TEXT",
    },
    "memoria_facts": {
        "version_id": "INTEGER DEFAULT 0",
        "previous_value": "TEXT",
        "updated_msg_idx": "INTEGER",
        "valid_from_msg_idx": "INTEGER",
        "valid_to_msg_idx": "INTEGER",
        "source_memory_id": "TEXT",
    },
}


def base36(number):
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    result = ""
    while number:
        number, remainder = divmod(number, 36)
        result = digits[remainder] + result
    return result or "0"


def limit_bank(name):
    if len(name) <= 64:
        return name
    digest = base36(bun_hash(name))
    return (name[:63 - len(digest)].rstrip("-") or "bank") + "-" + digest


def sanitize_bank(name):
    return limit_bank(re.sub(r"[^a-zA-Z0-9_-]+", "-", name.strip()).strip("-"))


def project_bank(workspace):
    label = sanitize_bank(workspace.name) or "default"
    return limit_bank(label + "-" + base36(bun_hash(str(workspace))))


def scoped_banks(database, workspace, configured, mode):
    shared = sanitize_bank(configured) or "default"
    name = limit_bank("-".join(part for part in (sanitize_bank(configured), project_bank(workspace)) if part))
    if mode == "global":
        return shared, shared, {shared: database}
    root = database.parent / "banks"
    banks = {name: root / name / DB_NAME}
    if mode == "per-project-tagged":
        banks.setdefault(shared, database)
    if root.is_dir():
        candidates = (entry for entry in root.iterdir()
                      if entry.name not in banks and entry.is_dir() and not entry.is_symlink())
        for entry in islice(candidates, 64):
            path = entry / DB_NAME
            if path.is_symlink():
                continue
            try:
                with open_bank(path, readonly=True) as db:
                    safe, = db.execute("""
                        SELECT CASE
                          WHEN EXISTS(SELECT 1 FROM working_memory
                                      WHERE json_extract(metadata_json, '$.cwd') IS NOT ?1) THEN 0
                          ELSE EXISTS(SELECT 1 FROM working_memory WHERE json_extract(metadata_json, '$.cwd') = ?1)
                        END
                    """, (str(workspace),)).fetchone()
                if safe:
                    banks[entry.name] = path
            except sqlite3.Error:
                continue
    return name, shared, banks


@contextmanager
def open_bank(path, *, readonly=False, write=False):
    db = sqlite3.connect(path.as_uri() + "?mode=ro" if readonly else path, uri=readonly, timeout=5)
    db.row_factory = sqlite3.Row
    try:
        with db:
            if write:
                db.execute("BEGIN IMMEDIATE")
            yield db
    finally:
        db.close()


def schema_statements():
    statement = ""
    for line in SCHEMA.read_text("utf-8").splitlines():
        if line.startswith("--") or not line.strip():
            continue
        statement += line + "\n"
        if sqlite3.complete_statement(statement):
            yield statement
            statement = ""


def columns(db, table):
    return {row["name"] for row in db.execute(f"PRAGMA table_info({table})")}


def initialize(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open_bank(path) as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=NORMAL")
        db.execute("BEGIN IMMEDIATE")
        statements = tuple(schema_statements())
        for sql in statements:
            if sql.startswith("CREATE TABLE"):
                db.execute(sql)
        changes = {**UPGRADES}
        for table in MEMORY_TABLES.values():
            changes[table] = {**COMMON_COLUMNS, **changes[table]}
        for table in PROJECTIONS[1:]:
            changes[table] = {"source_memory_id": "TEXT"}
        for table, expected in changes.items():
            missing = expected.keys() - columns(db, table)
            for column in missing:
                db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {expected[column]}")
            if "consolidated_at" in missing:
                db.execute("UPDATE working_memory SET consolidated_at=? WHERE consolidated_at IS NULL", (timestamp(),))
        for sql in statements:
            if not sql.startswith("CREATE TABLE"):
                db.execute(sql)


def visible(alias="m"):
    return (f"({alias}.session_id=? OR {alias}.scope='global' OR {alias}.channel_id=?) AND "
            f"{alias}.superseded_by IS NULL AND ({alias}.valid_until IS NULL OR {alias}.valid_until>?)")


def fact_visible(db, alias="f"):
    global_scope = f" OR {alias}.scope='global'" if "scope" in columns(db, "facts") else ""
    retired = (f"NOT EXISTS (SELECT 1 FROM {table} m WHERE m.id={alias}.source_msg_id "
               "AND (m.superseded_by IS NOT NULL OR m.valid_until<=?))" for table in MEMORY_TABLES.values())
    return " AND ".join([f"({alias}.session_id=?{global_scope})", *retired])


def fact_row(db, bank, identifier):
    scope = " OR scope='global'" if "scope" in columns(db, "facts") else ""
    row = db.execute(f"SELECT * FROM facts WHERE fact_id=? AND (session_id=?{scope})", (identifier, bank)).fetchone()
    if not row:
        return None
    source = None
    for table in MEMORY_TABLES.values():
        source = db.execute(f"SELECT valid_until,superseded_by FROM {table} WHERE id=?", (row["source_msg_id"],)).fetchone()
        if source:
            break
    fields = {key: row[key] for key in ("subject", "predicate", "object", "source_msg_id")}
    return {"id": identifier, "content": " ".join(row[key] for key in ("subject", "predicate", "object")),
            "source": "facts", "store": "fact", "bank": bank, "timestamp": row["timestamp"],
            "importance": row["confidence"], "session_id": row["session_id"], "created_at": row["created_at"],
            "metadata_json": json.dumps(fields, ensure_ascii=False), **(dict(source) if source else {})}


def row_by_id(db, bank, identifier):
    for kind, table in MEMORY_TABLES.items():
        where, params = ("id=?", (identifier,)) if kind == "working" else (
            "id=? AND (session_id=? OR scope='global')", (identifier, bank))
        if row := db.execute(f"SELECT * FROM {table} WHERE {where}", params).fetchone():
            return {**dict(row), "bank": bank, "store": kind}
    return fact_row(db, bank, identifier)


def purge_working_artifacts(db, identifier, *, keep_annotations=False):
    facts = [row[0] for row in db.execute("SELECT fact_id FROM facts WHERE source_msg_id=?", (identifier,))]
    for table, key in (("facts", "source_msg_id"), ("memory_embeddings", "memory_id"), ("triples", "source"),
                       *((table, "source_memory_id") for table in PROJECTIONS)):
        db.execute(f"DELETE FROM {table} WHERE {key}=?", (identifier,))
    unlink_fact_source(db, identifier)
    if not keep_annotations:
        db.execute("DELETE FROM annotations WHERE memory_id=?", (identifier,))
    gist = "gist_" + identifier
    if "id" in columns(db, "gists"):
        db.execute("DELETE FROM gists WHERE id=?", (gist,))
    if {"source", "target"} <= columns(db, "graph_edges"):
        db.executemany("DELETE FROM graph_edges WHERE source=? OR target=?",
                       ((key, key) for key in [identifier, gist, *facts]))


def trim_working(db, bank):
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    rows = db.execute("""
        SELECT id FROM working_memory WHERE session_id=? AND consolidated_at IS NULL AND trust_tier IS NOT 'IMPORTED'
          AND (timestamp<? OR id NOT IN (
              SELECT id FROM working_memory WHERE session_id=? AND consolidated_at IS NULL
                  AND trust_tier IS NOT 'IMPORTED' ORDER BY timestamp DESC, rowid DESC LIMIT 1000
          ))
    """, (bank, cutoff, bank)).fetchall()
    for (identifier,) in rows:
        db.execute("DELETE FROM working_memory WHERE id=?", (identifier,))
        purge_working_artifacts(db, identifier)


class Bank:
    def __init__(self, name, path, *, polyphonic=False):
        self.name, self.path = name, path
        self.polyphonic, self.polyphonic_ready = polyphonic, False
        initialize(path)

    def retain(self, content, *, source, importance, metadata, veracity, memory_type, embed_text=None, through=None):
        now = timestamp()
        with open_bank(self.path, write=True) as db:
            if through is not None and self.cursor(db, metadata["session_id"]) >= through:
                return None
            row = db.execute("SELECT id FROM working_memory WHERE content=? AND session_id=? LIMIT 1",
                             (content, self.name)).fetchone()
            trust = "STATED"
            if row and through is None:
                identifier = row["id"]
                db.execute("""
                    UPDATE working_memory SET importance=MAX(importance,?), timestamp=?, source=?, scope='bank',
                    author_id='coding-agent', author_type='agent', channel_id=?, memory_type=?,
                    veracity=CASE WHEN ? != 'unknown' THEN ? ELSE veracity END, trust_tier=?,
                    embed_text=COALESCE(?,embed_text), consolidated_at=NULL
                    WHERE id=? AND session_id=?
                """, (importance, now, source, self.name, memory_type, veracity, veracity, trust, embed_text,
                      identifier, self.name))
            else:
                identifier = sha256((content + "\0" + now + "\0" + uuid4().hex).encode()).hexdigest()[:16]
                db.execute("""
                    INSERT INTO working_memory
                    (id,content,embed_text,source,timestamp,session_id,importance,metadata_json,scope,
                     author_id,author_type,channel_id,veracity,memory_type,trust_tier)
                    VALUES (?,?,?,?,?,?,?,?,'bank','coding-agent','agent',?,?,?,?)
                """, (identifier, content, embed_text, source, now, self.name, importance,
                      json.dumps(metadata, ensure_ascii=False, separators=(",", ":")), self.name, veracity, memory_type, trust))
                annotations = [("occurred_on", now[:10])]
                if source not in ("", "conversation", "user", "assistant"):
                    annotations.append(("has_source", source))
                db.executemany("INSERT OR IGNORE INTO annotations(memory_id,kind,value) VALUES (?,?,?)",
                               ((identifier, kind, value) for kind, value in annotations))
                trim_working(db, self.name)
        return identifier

    def cursor(self, db, session):
        rows = db.execute("""
            SELECT CASE WHEN json_type(metadata_json,'$.retained_through_user_turn') IS NOT 'integer'
                        THEN content END AS content,json_extract(metadata_json,'$.source_id') AS source_id,
                   json_extract(metadata_json,'$.retained_through_user_turn') AS through
            FROM working_memory WHERE source='coding-agent-transcript' AND json_valid(metadata_json)
            AND json_extract(metadata_json,'$.session_id')=? ORDER BY rowid
        """, (session,))
        count = 0
        for row in rows:
            if type(row["through"]) is int:
                count = max(count, row["through"])
            elif isinstance(row["source_id"], str) and (row["source_id"] == session or row["source_id"].startswith(session + "-")):
                count = max(count, row["content"].splitlines().count("[role: user]"))
        return count

    def get(self, identifier):
        with open_bank(self.path) as db:
            return row_by_id(db, self.name, identifier)

    def edit(self, operation, identifier, *, content=None, importance=None, replacement=None):
        with open_bank(self.path, write=True) as db:
            row = row_by_id(db, self.name, identifier)
            if not row:
                return "not_found"
            store = row["store"]
            if store == "fact":
                return "not_editable"
            if operation == "invalidate":
                table = MEMORY_TABLES[store]
                now = timestamp()
                result = db.execute(f"""
                    UPDATE {table} SET valid_until=?,superseded_by=?
                    WHERE id=? AND (session_id=? OR scope='global')
                """, (now, replacement, identifier, self.name))
                if result.rowcount:
                    db.execute("UPDATE triples SET valid_until=? WHERE source=? AND valid_until IS NULL", (now, identifier))
                    unlink_fact_source(db, identifier)
                return "invalidated" if result.rowcount else "not_found"
            if store != "working" or row["session_id"] != self.name:
                return "not_found"
            if operation == "forget":
                db.execute("DELETE FROM working_memory WHERE id=?", (identifier,))
                purge_working_artifacts(db, identifier)
                return "deleted"
            db.execute("""
                UPDATE working_memory SET content=coalesce(?,content), importance=coalesce(?,importance),
                embed_text=CASE WHEN ? IS NULL OR ?=content THEN embed_text ELSE NULL END WHERE id=?
            """, (content, importance, content, content, identifier))
            if content is not None and content != row["content"]:
                purge_working_artifacts(db, identifier, keep_annotations=True)
            return "updated"

    def save_embedding(self, identifier, text, model, vector):
        self.save_embeddings([(identifier, text, vector)], model)

    def save_embeddings(self, items, model, *, missing_only=False):
        if not items:
            return
        now = timestamp()
        action = "IGNORE" if missing_only else "REPLACE"
        with open_bank(self.path, write=True) as db:
            db.executemany(f"""
                INSERT OR {action} INTO memory_embeddings(memory_id,embedding_json,model)
                SELECT ?,?,? WHERE {LIVE_EMBEDDING}
            """, ((identifier, json.dumps(vector, separators=(",", ":")), model, identifier, text, now,
                   identifier, text, now) for identifier, text, vector in items))

    def stats(self):
        with open_bank(self.path) as db:
            return {table: db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                    for table in ("working_memory", "episodic_memory", "facts", "memory_embeddings", "gists", "graph_edges",
                                  "consolidated_facts", "conflicts")}

    def clear(self):
        with open_bank(self.path, write=True) as db:
            delete_legacy_vectors(db)
            names = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            for table in ("working_memory", "episodic_memory", "facts", "memory_embeddings", "annotations", "triples",
                          "scratchpad", "memory_validations", "consolidation_log", "gists", "graph_edges", "consolidated_facts",
                          "conflicts", *PROJECTIONS):
                if table in names:
                    db.execute(f"DELETE FROM {table}")
        self.polyphonic_ready = False
