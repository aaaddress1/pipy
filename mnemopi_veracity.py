"""Source-aware consolidation of Mnemopi knowledge graph facts."""

import json
from hashlib import sha256
from unicodedata import normalize

from session_jsonl import timestamp


VERACITY = {"stated": 1., "inferred": .7, "tool": .5, "imported": .6, "unknown": .8}
MULTI_VALUED = frozenset(("related_to", "mentions", "references", "uses", "has", "depends_on", "knows"))


def fact_id(subject, predicate, object_):
    digest = sha256()
    for value in (subject, predicate, object_):
        if not isinstance(value, str) or not value:
            raise ValueError("Fact subjects, predicates, and objects must be nonempty text.")
        data = normalize("NFC", value).encode("utf-8")
        digest.update(str(len(data)).encode("ascii") + b":")
        digest.update(data)
    return "cf_" + digest.hexdigest()[:24]


def sources(raw):
    try:
        parsed = json.loads(raw or "[]")
    except (ValueError, TypeError):
        return []
    return [source for source in parsed if isinstance(source, str)] if isinstance(parsed, list) else []


def source_veracity(db, identifier):
    for table in ("working_memory", "episodic_memory"):
        if row := db.execute(f"SELECT veracity FROM {table} WHERE id=?", (identifier,)).fetchone():
            return row[0] if row[0] in VERACITY else "unknown"
    return "unknown"


def update_confidence(confidence, veracity):
    return min(1., confidence + (1 - confidence) * VERACITY.get(veracity, VERACITY["unknown"]) * .3)


def consolidate_fact(db, subject, predicate, object_, source, *, veracity=None):
    identifier = fact_id(subject, predicate, object_)
    existing = db.execute("""
        SELECT id,confidence,mention_count,sources_json FROM consolidated_facts
        WHERE id=? OR (subject=? AND predicate=? AND object=?) LIMIT 1
    """, (identifier, subject, predicate, object_)).fetchone()
    known = sources(existing["sources_json"]) if existing else []
    if source and source in known:
        return existing["id"]
    if source:
        known.append(source)
    truth = veracity if veracity in VERACITY else source_veracity(db, source)
    now = timestamp()
    if existing:
        confidence = update_confidence(existing["confidence"], truth) if existing["mention_count"] else VERACITY[truth] * .5
        db.execute("""
            UPDATE consolidated_facts SET confidence=?,mention_count=?,last_seen=?,sources_json=?,veracity=?,updated_at=?
            WHERE id=?
        """, (confidence, existing["mention_count"] + 1, now,
              json.dumps(known, ensure_ascii=False), truth, now, existing["id"]))
        return existing["id"]
    db.execute("""
        INSERT INTO consolidated_facts
          (id,subject,predicate,object,confidence,mention_count,first_seen,last_seen,sources_json,veracity)
        VALUES (?,?,?,?,?,1,?,?,?,?)
    """, (identifier, subject, predicate, object_, VERACITY[truth] * .5, now, now,
          json.dumps(known, ensure_ascii=False), truth))
    if predicate.lower() not in MULTI_VALUED:
        db.execute("""
            INSERT INTO conflicts(fact_a_id,fact_b_id,conflict_type)
            SELECT ?,id,'contradiction' FROM consolidated_facts WHERE subject=? AND predicate=? AND object<>?
        """, (identifier, subject, predicate, object_))
    return identifier


def backfill_facts(db):
    now = timestamp()
    rows = db.execute("""
        SELECT trim(k.subject) AS subject,trim(k.predicate) AS predicate,trim(k.object) AS object,k.source_memory_id
        FROM memoria_kg k WHERE k.source_memory_id IS NOT NULL AND k.source_memory_id<>''
          AND trim(k.subject)<>'' AND trim(k.predicate)<>'' AND trim(k.object)<>''
          AND (EXISTS(SELECT 1 FROM working_memory w WHERE w.id=k.source_memory_id AND w.superseded_by IS NULL
                      AND (w.valid_until IS NULL OR w.valid_until>?))
               OR EXISTS(SELECT 1 FROM episodic_memory e WHERE e.id=k.source_memory_id AND e.superseded_by IS NULL
                         AND (e.valid_until IS NULL OR e.valid_until>?)))
          AND NOT EXISTS (
              SELECT 1 FROM consolidated_facts f,json_each(CASE WHEN json_valid(f.sources_json) THEN f.sources_json ELSE '[]' END) s
              WHERE f.subject=trim(k.subject) AND f.predicate=trim(k.predicate) AND f.object=trim(k.object)
                AND s.value=k.source_memory_id
          )
        GROUP BY 1,2,3,4 ORDER BY min(k.id)
    """, (now, now)).fetchall()
    for row in rows:
        consolidate_fact(db, *row)


def unlink_fact_source(db, identifier):
    now = timestamp()
    rows = db.execute("""
        SELECT id,sources_json FROM consolidated_facts f
        WHERE EXISTS(SELECT 1 FROM json_each(CASE WHEN json_valid(f.sources_json) THEN f.sources_json ELSE '[]' END)
                     WHERE value=?)
    """, (identifier,)).fetchall()
    for row in rows:
        remaining = [source for source in sources(row["sources_json"]) if source != identifier]
        evidence = [source_veracity(db, source) for source in remaining]
        confidence = VERACITY[evidence[0]] * .5 if evidence else 0.
        for truth in evidence[1:]:
            confidence = update_confidence(confidence, truth)
        db.execute("""
            UPDATE consolidated_facts SET sources_json=?,mention_count=?,confidence=?,veracity=?,updated_at=? WHERE id=?
        """, (json.dumps(remaining, ensure_ascii=False), len(remaining), confidence,
              evidence[-1] if evidence else "unknown", now, row["id"]))


def resolve_conflict(db, conflict_id, winner):
    row = db.execute("SELECT fact_a_id,fact_b_id,resolution FROM conflicts WHERE id=?", (conflict_id,)).fetchone()
    if not row or row["resolution"] is not None or winner not in (row["fact_a_id"], row["fact_b_id"]):
        return False
    loser = row["fact_b_id"] if winner == row["fact_a_id"] else row["fact_a_id"]
    now = timestamp()
    db.execute("UPDATE consolidated_facts SET superseded_by=?,updated_at=? WHERE id=?", (winner, now, loser))
    db.execute("UPDATE conflicts SET resolution=?,resolved_at=? WHERE id=?", ("superseded_by_" + winner, now, conflict_id))
    return True
