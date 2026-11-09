"""Import and export the upstream Mnemopi BEAM 1.0 backup format."""

import json
import math
import sqlite3
from contextlib import suppress

from mnemopi_legacy_vectors import insert_legacy_vector, legacy_vector_type
from mnemopi_store import open_bank, purge_working_artifacts
from session_jsonl import timestamp


MEMORY_FIELDS = {
    "content": "", "source": None, "timestamp": None, "session_id": "default", "importance": .5,
    "metadata_json": "{}", "valid_until": None, "superseded_by": None, "scope": "session",
    "recall_count": 0, "last_recalled": None, "created_at": None, "veracity": "unknown",
    "memory_type": "unknown", "author_id": None, "author_type": None, "channel_id": None,
    "trust_tier": "STATED", "event_date": None, "event_date_precision": "unknown", "temporal_tags": "[]",
}
SCRATCH_FIELDS = {"content": "", "session_id": "default", "created_at": None, "updated_at": None}
LOG_FIELDS = {"session_id": "default", "items_consolidated": 0, "summary_preview": "", "created_at": None}
VERACITY = frozenset(("true", "false", "stated", "inferred", "tool", "imported", "unknown"))


def export_bank(bank):
    sections = (
        ("working_memory", ("id", *MEMORY_FIELDS, "consolidated_at", "embed_text"), "timestamp"),
        ("episodic_memory", ("rowid", "id", *MEMORY_FIELDS, "summary_of"), "timestamp"),
        ("scratchpad", ("id", *SCRATCH_FIELDS), "updated_at"),
        ("consolidation_log", ("id", *LOG_FIELDS), "created_at"),
    )
    with open_bank(bank.path, readonly=True) as db:
        db.execute("BEGIN")
        result = {"mnemopi_export": {"version": "1.0", "export_date": timestamp(), "source_db": str(bank.path),
                                     "component": "beam"}, "episodic_embeddings": []}
        for table, fields, order in sections:
            result[table] = [dict(row) for row in db.execute(
                f"SELECT {','.join(fields)} FROM {table} ORDER BY session_id,{order}")]
    return result


def section(data, name):
    rows = data.get(name)
    return (item if isinstance(item, dict) else {} for item in rows) if isinstance(rows, list) else ()


def row_values(item, fields):
    values = []
    for key, fallback in fields.items():
        value = item.get(key, fallback)
        values.append(value if value is None or type(value) in (str, int, float, bool) else fallback)
    return values


def identifier(item):
    value = item.get("id")
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value) if isinstance(value, (str, int, float)) else ""


def number(value):
    if value is None or isinstance(value, str) and not value.strip():
        return 0.
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return math.nan


def import_memories(db, data, table, stats, imported_at, force, vector_type):
    working = table == "working_memory"
    fields = (MEMORY_FIELDS | {"consolidated_at": imported_at, "embed_text": None} if working else
              MEMORY_FIELDS | {"summary_of": ""})
    sql = f"INSERT INTO {table}(id,{','.join(fields)}) VALUES ({','.join('?' for _ in range(len(fields) + 1))})"
    mapped = {}
    for item in section(data, table):
        id_ = identifier(item)
        if not id_:
            continue
        existing = db.execute(f"SELECT rowid FROM {table} WHERE id=?", (id_,)).fetchone()
        if existing and not force:
            stats["skipped"] += 1
            continue
        if existing:
            if not working and vector_type:
                with suppress(sqlite3.Error):
                    db.execute("DELETE FROM vec_episodes WHERE rowid=?", (existing[0],))
            db.execute(f"DELETE FROM {table} WHERE id=?", (id_,))
            purge_working_artifacts(db, id_)
        truth = item.get("veracity")
        truth = truth.strip().lower() if isinstance(truth, str) else "unknown"
        values = item | {"veracity": truth if truth in VERACITY else "unknown"}
        if working and values.get("consolidated_at") is None:
            values["consolidated_at"] = imported_at
        inserted = db.execute(sql, (id_, *row_values(values, fields))).lastrowid
        stats["overwritten" if existing else "inserted"] += 1
        if not working and math.isfinite(old_id := number(item.get("rowid", math.nan))):
            mapped[old_id] = inserted
    return mapped


def import_bank(bank, data, *, force=False):
    if not isinstance(data, dict):
        raise ValueError("Import file must contain a Mnemopi export object.")
    stats = {"working_memory": dict(inserted=0, skipped=0, overwritten=0),
             "episodic_memory": dict(inserted=0, skipped=0, overwritten=0, embeddings_inserted=0),
             "scratchpad": dict(inserted=0, updated=0), "consolidation_log": dict(inserted=0)}
    imported_at = timestamp()
    with open_bank(bank.path, write=True) as db:
        vector_type = legacy_vector_type(db)
        import_memories(db, data, "working_memory", stats["working_memory"], imported_at, force, vector_type)
        mapped = import_memories(db, data, "episodic_memory", stats["episodic_memory"], imported_at, force, vector_type)
        if vector_type and mapped:
            for item in section(data, "episodic_embeddings"):
                rowid = mapped.get(number(item.get("rowid", math.nan)))
                raw = item.get("embedding")
                if rowid is None or not isinstance(raw, list):
                    continue
                if not db.execute("SELECT 1 FROM episodic_memory WHERE rowid=?", (rowid,)).fetchone():
                    continue
                embedding = [number(value) for value in raw]
                if all(map(math.isfinite, embedding)):
                    with suppress(sqlite3.Error):
                        insert_legacy_vector(db, vector_type, rowid, embedding)
                        stats["episodic_memory"]["embeddings_inserted"] += 1
        for item in section(data, "scratchpad"):
            id_ = identifier(item)
            if not id_:
                continue
            exists = db.execute("SELECT 1 FROM scratchpad WHERE id=?", (id_,)).fetchone()
            db.execute("""
                INSERT INTO scratchpad(id,content,session_id,created_at,updated_at) VALUES (?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET content=excluded.content,session_id=excluded.session_id,
                    created_at=excluded.created_at,updated_at=excluded.updated_at
            """, (id_, *row_values(item, SCRATCH_FIELDS)))
            stats["scratchpad"]["updated" if exists else "inserted"] += 1
        for item in section(data, "consolidation_log"):
            db.execute("INSERT INTO consolidation_log(session_id,items_consolidated,summary_preview,created_at) VALUES (?,?,?,?)",
                       row_values(item, LOG_FIELDS))
            stats["consolidation_log"]["inserted"] += 1
    bank.polyphonic_ready = False
    return stats
