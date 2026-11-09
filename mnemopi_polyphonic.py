"""Bounded Mnemopi recall voices and reciprocal rank fusion."""

import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from math import exp

from mnemopi_graph import NOUN, TOKEN
from mnemopi_recall import cosine, count_recalls, preview, rank_bank, vector
from mnemopi_store import MEMORY_TABLES, fact_visible, open_bank, visible
from mnemopi_veracity import backfill_facts, sources
from session_jsonl import timestamp


VOICES = ("hybrid", "vector", "graph", "fact", "temporal")
RRF_K = 60
MAX_SCORE = len(VOICES) / (RRF_K + 1)
GRAPH_DIRECT = 32
GRAPH_SEEDS = 16
GRAPH_DEPTH = 2
GRAPH_FRONTIER = 32
GRAPH_EDGE_ROWS = 512
GRAPH_RESULTS = 64
TEMPORAL_TERMS = ("yesterday", "today", "recent", "last", "latest", "this week", "this month", "ago", "before")
FACT_STOP = frozenset("""
a an the is are was were be been have has had do does did will would could should may might can shall must
i you he she it we they me him her us them my your his its our their mine yours hers ours theirs
what which who whom where when why how this that these those of in to for on with at by from as into through
during before after above below between under and but or nor not so than too very just about also really
actually basically simply if then else while because though although
""".split())


def voice_enabled(name):
    return os.getenv("MNEMOPI_VOICE_" + name.upper(), "").strip().lower() not in ("0", "false", "no", "off")


def placeholders(items):
    return ",".join("?" for _ in items)


def visibility(bank, now):
    return bank, bank, now


def live_ids(db, bank, now, identifiers):
    if not identifiers:
        return set()
    params = (*identifiers, *visibility(bank, now))
    where = f"m.id IN ({placeholders(identifiers)}) AND {visible()}"
    return {row[0] for table in MEMORY_TABLES.values()
            for row in db.execute(f"SELECT m.id FROM {table} m WHERE {where}", params)}


def hybrid_voice(baseline):
    unique = dict.fromkeys(row["id"] for row in baseline if row["id"])
    details = {row["id"]: row["score"] for row in reversed(baseline)}
    return [(identifier, 1 / rank, {"hybrid_rank": rank, "hybrid_score": details[identifier]})
            for rank, identifier in enumerate(unique, 1)]


def vector_voice(db, bank, now, query_vector, model):
    if not query_vector:
        return []
    sql = " UNION ALL ".join(f"""
        SELECT m.id,e.embedding_json,'{kind}' AS store FROM memory_embeddings e
        JOIN {table} m ON m.id=e.memory_id WHERE e.model IS ? AND {visible()}
    """ for kind, table in MEMORY_TABLES.items())
    best = {}
    args = (model, *visibility(bank, now)) * len(MEMORY_TABLES)
    for row in db.execute(sql + " LIMIT 50000", args):
        try:
            value = cosine(query_vector, vector(json.loads(row["embedding_json"])))
        except (ValueError, OverflowError):
            continue
        if value < .65:
            continue
        identifier = row["id"]
        score = (value + 1) / 2
        if identifier not in best or score > best[identifier][1]:
            best[identifier] = (identifier, score, {"similarity": score, "cosine_similarity": value,
                                                    "embedding_tier": row["store"], "backend": "memory_embeddings"})
    return sorted(best.values(), key=lambda hit: (-hit[1], hit[0]))[:20]


def graph_voice(db, bank, now, query):
    best = {}

    def seed(identifier, score, details):
        if identifier and (identifier not in best or best[identifier][1] < score):
            best[identifier] = (identifier, score, details)

    nouns = {}
    for noun in NOUN.findall(query):
        nouns.setdefault(noun.lower(), noun)
    for entity in nouns.values():
        escaped = entity.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        for row in db.execute("""
            SELECT id,memory_id,text FROM gists WHERE participants_json LIKE ? ESCAPE '\\' ORDER BY timestamp DESC
        """, ('%"' + escaped + '"%',)):
            identifier = row["memory_id"] or row["id"].removeprefix("gist_")
            seed(identifier, .6, {"entity": entity, "gist": row["text"]})
        for row in db.execute("""
            SELECT source_msg_id,subject,predicate,object,confidence FROM facts
            WHERE subject=? ORDER BY confidence DESC,timestamp DESC
        """, (entity,)):
            confidence = .5 if row["confidence"] is None else row["confidence"]
            seed(row["source_msg_id"], confidence * .5, {"entity": entity,
                 "fact": " ".join(row[key] for key in ("subject", "predicate", "object"))})
    candidates = list(best)
    active = set()
    for offset in range(0, len(candidates), 400):
        active.update(live_ids(db, bank, now, candidates[offset:offset + 400]))
    direct = [hit for identifier, hit in best.items() if identifier in active]
    direct.sort(key=lambda hit: -hit[1])
    results = direct[:GRAPH_DIRECT]
    frontier = [hit[0] for hit in results[:GRAPH_SEEDS]]
    roots = {identifier: identifier for identifier in frontier}
    seen = set(best)
    for depth in range(1, GRAPH_DEPTH + 1):
        if not frontier or len(results) >= GRAPH_RESULTS:
            break
        slots = placeholders(frontier)
        rows = db.execute(f"""
            SELECT source,target,weight FROM graph_edges
            WHERE (source IN ({slots}) OR target IN ({slots})) AND edge_type='ctx' AND weight>=.3
            ORDER BY weight DESC,id LIMIT ?
        """, (*frontier, *frontier, GRAPH_EDGE_ROWS))
        expanding = set(frontier)
        adjacent = {}
        for row in rows:
            start, end = row["source"], row["target"]
            if start not in expanding:
                start, end = end, start
            if end not in seen and not end.startswith("gist_"):
                adjacent.setdefault(end, (start, row["weight"]))
        frontier = []
        keys = list(adjacent)
        reachable = set()
        for offset in range(0, len(keys), 400):
            reachable.update(live_ids(db, bank, now, keys[offset:offset + 400]))
        for end, (start, weight) in adjacent.items():
            seen.add(end)
            if end not in reachable:
                continue
            roots[end] = roots[start]
            if len(frontier) < GRAPH_FRONTIER:
                frontier.append(end)
            results.append((end, .4 / depth, {"seed": roots[end], "edge_type": "ctx", "depth": depth, "weight": weight}))
            if len(results) == GRAPH_RESULTS:
                break
    return results


def fact_voice(db, query):
    words = TOKEN.findall(query.lower())
    candidates = {}
    for index, word in enumerate(words):
        if word not in FACT_STOP and len(word) >= 3:
            candidates[word] = None
        if index + 1 < len(words) and (word not in FACT_STOP or words[index + 1] not in FACT_STOP):
            candidates[word + " " + words[index + 1]] = None
    if not candidates:
        return []
    best = {}
    keys = list(candidates)
    for offset in range(0, len(keys), 400):
        chunk = keys[offset:offset + 400]
        for row in db.execute(f"""
            SELECT * FROM consolidated_facts WHERE lower(subject) IN ({placeholders(chunk)})
              AND confidence>=.5 AND superseded_by IS NULL ORDER BY confidence DESC,mention_count DESC
        """, chunk):
            for identifier in sources(row["sources_json"]):
                identifier = identifier.strip()
                if not identifier or identifier in best and best[identifier][1] >= row["confidence"]:
                    continue
                best[identifier] = (identifier, row["confidence"], {"fact_id": row["id"], "subject": row["subject"],
                                    "predicate": row["predicate"], "object": row["object"], "mentions": row["mention_count"]})
    return list(best.values())


def temporal_voice(db, bank, now, query, anchor=None):
    if not any(term in query.lower() for term in TEMPORAL_TERMS):
        return []
    moment = datetime.fromisoformat(now)
    upper = None
    if anchor:
        try:
            moment = datetime.fromisoformat(anchor)
            if moment.tzinfo is None:
                moment = moment.replace(tzinfo=timezone.utc)
            upper = moment.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        except ValueError:
            pass
    cutoff = (moment - timedelta(days=7)).astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    rows = db.execute(f"""
        SELECT m.id,m.timestamp,m.importance FROM working_memory m
        WHERE m.timestamp>? AND (? IS NULL OR m.timestamp<=?) AND {visible()}
        ORDER BY m.timestamp DESC LIMIT 20
    """, (cutoff, upper, upper, *visibility(bank, now)))
    results = []
    for row in rows:
        if not row["timestamp"]:
            continue
        try:
            then = datetime.fromisoformat(row["timestamp"])
        except ValueError:
            continue
        if then.tzinfo is None:
            then = then.replace(tzinfo=timezone.utc)
        age = max(0., (moment - then).total_seconds() / 86_400)
        importance = row["importance"] or 0.
        results.append((row["id"], exp(-age / 7) * importance, {"age_days": age, "importance": importance}))
    return results


def fuse(voices):
    merged = {}
    for voice, hits in voices:
        ranked = set()
        for identifier, score, details in sorted(hits, key=lambda hit: (-hit[1], hit[0])):
            if identifier in ranked:
                continue
            ranked.add(identifier)
            item = merged.setdefault(identifier, {"combined_score": 0., "voice_scores": {}, "metadata": {}})
            contribution = 1 / (RRF_K + len(ranked))
            item["combined_score"] += contribution
            item["voice_scores"][voice] = contribution
            item["metadata"].update(details)
    return sorted(merged.items(), key=lambda item: (-item[1]["combined_score"], item[0]))


def hydrate(db, bank, now, identifier, prior):
    if prior and prior["store"] == "fact":
        valid = db.execute(f"SELECT 1 FROM facts f WHERE f.fact_id=? AND {fact_visible(db)}",
                           (identifier, bank, now, now)).fetchone()
        return prior if valid else None
    for kind, table in MEMORY_TABLES.items():
        if row := db.execute(f"SELECT m.* FROM {table} m WHERE m.id=? AND {visible()}",
                             (identifier, *visibility(bank, now))).fetchone():
            return prior or {**dict(row), "store": kind}
    return None


def diversity(db, bank, now, fused, baseline, limit):
    previous = {row["id"]: row for row in reversed(baseline)}
    picks, taken = [], []
    for identifier, fused_row in fused:
        row = hydrate(db, bank, now, identifier, previous.get(identifier))
        if row is None:
            continue
        tokens = {token for token in TOKEN.findall(row["content"].lower()) if len(token) >= 2}
        if any(tokens & earlier and len(tokens & earlier) / len(tokens | earlier) > .8 for earlier in taken):
            continue
        taken.append(tokens)
        picks.append({**row, "bank": bank, "content": preview(row["content"]), "voice_scores": fused_row["voice_scores"],
                      "combined_score": fused_row["combined_score"], "polyphonic": fused_row["metadata"],
                      "score": round(fused_row["combined_score"] / MAX_SCORE, 4)})
        if len(picks) == limit:
            break
    return picks


def prepare_fact_voice(bank):
    if not bank.polyphonic_ready and voice_enabled("fact"):
        try:
            with open_bank(bank.path, write=True) as db:
                backfill_facts(db)
            bank.polyphonic_ready = True
        except sqlite3.Error:
            # Preserve recall and try the backfill again on the next lookup.
            pass


def rank_polyphonic_bank(bank, query, limit, query_vector=None, model=None, *, anchor=None, update_counts=True):
    if limit <= 0:
        return []
    prepare_fact_voice(bank)
    baseline = rank_bank(bank, query, limit * 2, query_vector, model, update_counts=False) if voice_enabled("hybrid") else []
    now = timestamp()
    with open_bank(bank.path) as db:
        voices = (
            ("hybrid", hybrid_voice(baseline)),
            ("vector", vector_voice(db, bank.name, now, query_vector, model) if voice_enabled("vector") else []),
            ("graph", graph_voice(db, bank.name, now, query) if voice_enabled("graph") else []),
            ("fact", fact_voice(db, query) if voice_enabled("fact") else []),
            ("temporal", temporal_voice(db, bank.name, now, query, anchor) if voice_enabled("temporal") else []),
        )
        results = diversity(db, bank.name, now, fuse(voices), baseline, limit)
        if update_counts:
            count_recalls(db, bank, results)
        return results
