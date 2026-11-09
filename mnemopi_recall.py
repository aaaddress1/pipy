"""BEAM lexical and dense candidate retrieval, scoring, and diversification."""

import json
import math
import os
import re
from datetime import datetime, timezone

import regex

from mnemopi_store import MEMORY_TABLES, fact_visible, open_bank, visible
from session_jsonl import timestamp


GROUPS = """
database db datastore data_store
password pass pwd passwd credential secret token
config configuration settings cfg setup
error bug issue fault failure crash exception traceback
fix repair resolve solve patch correct address
deploy deployment release ship push rollout
server host machine vm instance node vps
api endpoint interface service
key token credential secret api_key
user account profile identity person
model llm ai provider gpt claude gemini
speed fast quick performance latency throughput
memory recall remember storage retention
search find lookup query retrieve locate
file document doc text note
code script program source implementation
test verify check validate probe examine
backup snapshot copy save archive
install setup configure bootstrap init
update upgrade refresh renew sync
delete remove destroy purge clean wipe erase
list show display enumerate catalog
time date when timestamp schedule
url link address uri path
health status check pulse alive up
service daemon process systemd worker
port socket bind listen
network internet connection connectivity dns
ssh terminal shell remote connect
git commit push pull repo repository branch
log output stdout stderr trace debug
cron schedule job task timer periodic
email mail message inbox smtp
image picture photo screenshot graphic
browser web page site navigate chrome
monitor watch observe track survey
alert notify notification warning ping
migrate transfer move relocate port
compare diff versus vs contrast
save store persist preserve keep
"""
SYNONYMS = {word: tuple(group.split()) for group in GROUPS.strip().splitlines() for word in group.split()}
STOP = frozenset("a an and are as at be by for from how i in is it of on or that the this to was what when where who with".split())
FACT_FILLER = frozenset("active current currently d know latest ll m please present re recent remind remember s t tell today ve".split())
TOKEN = regex.compile(r"[\p{L}\p{N}_]+")
CJK = regex.compile(r"[\p{Han}\p{Hiragana}\p{Katakana}\p{Hangul}]")
CURRENT_QUERY = re.compile(r"\b(?:now|current|currently|latest|recent|today|active|present)\b", re.I)
CURRENT_CONTENT = re.compile(r"\b(?:current|currently|latest|now|active|present)\b", re.I)
HISTORIC_CONTENT = re.compile(r"\b(?:was|previous|previously|legacy|old|stale|former|deprecated)\b", re.I)
VERACITY = {"stated": 1., "inferred": .7, "tool": .5, "imported": .6, "unknown": .8}
INTENT_PATTERNS = (
    ("temporal", (
        r"\b(when|last|yesterday|today|tomorrow|ago|before|after|since|until|during|recently|lately)\b",
        r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"\b(this|next|last)\s+(week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        r"\b\d+\s+(day|week|month|year|hour|minute)s?\s+(ago|from now|later|earlier)\b",
    )),
    ("factual", (
        r"\bwhat\s+is\b",
        r"\bwho\s+is\b",
        r"\bwhere\s+is\b",
        r"\b(definition|define|explain|meaning)\b",
        r"\bhow\s+(many|much|long|far)\b",
    )),
    ("entity", (
        r"\b(tell\s+me\s+about|what\s+do\s+you\s+know\s+about)\b",
        r"\b(who\s+is|what\s+does)\s+[a-z]+\b",
        r"\b(about|regarding|concerning)\s+[a-z]+\b",
    )),
    ("preference", (
        r"\b(prefer|like|dislike|want|hate|love|enjoy|favorite|best|worst)\b",
        r"\b(should\s+i|would\s+you|do\s+you\s+recommend)\b",
        r"\b(choose|pick|select|option|choice|decide)\b",
    )),
    ("procedural", (
        r"\bhow\s+(to|do|can|should|would)\b",
        r"\b(step|process|procedure|workflow|guide|tutorial)\b",
        r"\b(setup|install|configure|build|deploy|run|execute|start|stop)\b",
    )),
)
INTENT_WEIGHTS = {
    "temporal": (0.6, 1.5, 0.8),
    "factual": (1.0, 1.2, 0.9),
    "entity": (1.1, 1.0, 1.3),
    "preference": (0.9, 0.8, 1.5),
    "procedural": (1.3, 0.9, 0.7),
    "general": (1.0, 1.0, 1.0),
}


def tokenize(text):
    return [word for word in TOKEN.findall(text.lower()) if word not in STOP]


def groups(query):
    extra = {"branding": ("positioning", "wording", "headline")}
    extra.update(dict.fromkeys(("preference", "prefer", "preferred"), ("wants", "want", "prefers")))
    return [tuple(dict.fromkeys(part for option in (*SYNONYMS.get(word, (word,)), *extra.get(word, ()))
                                for part in tokenize(option))) for word in tokenize(query)]


def fts_query(query, tokens):
    return " OR ".join('"' + word.replace('"', '""') + '"' for word in (tokens[:12] or [query.strip()]))


def word_form(left, right):
    shorter, longer = (left, right) if len(left) <= len(right) else (right, left)
    if len(shorter) < 4 or len(shorter) == len(longer):
        return False
    if longer.startswith(shorter) and longer[len(shorter):] in ("s", "es", "d", "ed", "ing", "er", "ers", "ment", "ments"):
        return True
    if not longer.startswith(shorter[:-1]):
        return False
    ending = longer[len(shorter) - 1:]
    return shorter.endswith("e") and ending in ("ing", "ion", "ions") or shorter.endswith("y") and ending in ("ies", "ied")


def relevance(query_groups, content):
    if not query_groups:
        return 0.
    tokens = tokenize(content)
    content_set = set(tokens)
    content_set.update(part for token in tokens if "_" in token for part in token.split("_") if part)
    lower = content.lower()
    matched = partial = 0
    for group in query_groups:
        if any(word in content_set or CJK.search(word) and word in lower for word in group):
            matched += 1
        elif any(word_form(word, term) for word in group for term in content_set):
            partial += 1
    if len(query_groups) > 1:
        return (matched + partial * .5) / len(query_groups)
    if not matched:
        return .35 if partial else 0.
    token = query_groups[0][0]
    repeats = lower.count(token) if CJK.search(token) else tokens.count(token)
    return .7 + min(max(repeats - 1, 0), 3) * .1


def vector(raw):
    if not isinstance(raw, list) or not raw or any(type(value) not in (int, float) or not math.isfinite(value) for value in raw):
        return None
    return raw


def cosine(left, right):
    if not right or len(left) != len(right):
        return 0.
    norm = math.hypot(*left) * math.hypot(*right)
    return max(0., sum(a * b for a, b in zip(left, right)) / norm) if norm else 0.


def ranked(rows, key):
    if not rows:
        return {}
    low, high = min(row["rank"] for row in rows), max(row["rank"] for row in rows)
    return {row[key]: 1 - (row["rank"] - low) / (high - low or 1) for row in rows}


def memory_candidates(db, kind, params, expression, query_vector, model, limit):
    table = MEMORY_TABLES[kind]
    mirror, key = ("fts_working", "id") if kind == "working" else ("fts_episodes", "rowid")
    fts = ranked(list(db.execute(f"""
        SELECT f.{key},f.rank FROM {mirror} f WHERE {mirror} MATCH ?
          AND EXISTS (SELECT 1 FROM {table} m WHERE m.{key}=f.{key} AND {visible()})
        ORDER BY f.rank,f.{key} LIMIT ?
    """, (expression, *params, limit))), key)
    dense = {}
    if query_vector:
        for row in db.execute(f"""
            SELECT m.{key} AS candidate,e.embedding_json
            FROM (SELECT rowid,* FROM {table} m WHERE {visible()} ORDER BY timestamp DESC LIMIT 500) m
            JOIN memory_embeddings e ON e.memory_id=m.id AND e.model IS ?
        """, (*params, model)):
            try:
                similarity = cosine(query_vector, vector(json.loads(row["embedding_json"])))
            except (ValueError, OverflowError):
                continue
            dense[row["candidate"]] = similarity
    keys = tuple(dict.fromkeys([*fts, *sorted(dense, key=dense.get, reverse=True)[:limit]]))
    if keys:
        placeholders = ",".join("?" for _ in keys)
        rows = db.execute(f"SELECT rowid,* FROM {table} m WHERE m.{key} IN ({placeholders}) AND {visible()}",
                          (*keys, *params))
    else:
        rows = db.execute(f"SELECT rowid,* FROM {table} m WHERE {visible()} ORDER BY timestamp DESC LIMIT 500", params)
    return [(dict(row), fts.get(row[key], 0.), dense.get(row[key], 0.)) for row in rows]


def number(name, default):
    try:
        value = float(os.getenv("MNEMOPI_" + name, default))
        return value if math.isfinite(value) else default
    except ValueError:
        return default


def preview(content):
    return content if len(content) <= 500 else content[:499] + "…"


def recency(stamp, now, half_life=72):
    if not stamp:
        return 0.
    try:
        moment = datetime.fromisoformat(stamp)
    except ValueError:
        return 0.
    if not moment.tzinfo:
        moment = moment.replace(tzinfo=timezone.utc)
    return math.exp(-max(0., (now - moment).total_seconds()) / (3600 * max(.001, half_life)))


def score(row, kind, fts, dense, query_groups, threshold, now, current, weights):
    text = row.get("embed_text") or row["content"]
    lexical = relevance(query_groups, text)
    if lexical < threshold and dense < .65:
        return None
    vector_weight, fts_weight, importance_weight = weights
    importance = row["importance"] if row["importance"] is not None else .5
    keyword = max(lexical, fts * .6)
    if kind == "episodic":
        value = max(dense * vector_weight + fts * fts_weight + importance * importance_weight, lexical * .8)
        tier = row.get("tier") or 1
        value *= max(0., number(f"TIER{min(tier, 3)}_WEIGHT", (1., .85, .7)[min(tier, 3) - 1]))
    else:
        value = keyword * (1 - importance_weight) * .6 + importance * importance_weight + keyword**2 * .08
        if dense > 0:
            value = value * .8 + dense * .2
    truth = row["veracity"] or "unknown"
    trust = 0. if truth == "false" else 1. if truth in ("true", "likely_true") else max(
        0., number(truth.upper() + "_WEIGHT", VERACITY[truth]) if truth in VERACITY else number("UNKNOWN_WEIGHT", .8))
    decay = recency(row["timestamp"], now)
    value *= (.7 + .3 * decay) * trust
    if current:
        half_life = number("TEMPORAL_HALFLIFE_HOURS", 24)
        value *= 1 + .45 * max(recency(row["timestamp"], now, half_life), recency(row["event_date"], now, half_life * 2))
        if CURRENT_CONTENT.search(text):
            value *= 1.35
        if HISTORIC_CONTENT.search(text):
            value *= .72
    return {**row, "content": preview(row["content"]), "score": round(value, 4), "store": kind,
            "keyword_score": round(lexical, 4), "dense_score": round(dense, 4), "fts_score": round(fts, 4)}


def facts(db, bank, now, expression, tokens, query, limit):
    params = (bank, now, now)
    where = fact_visible(db)
    rows = list(db.execute(f"""
        SELECT f.rowid,f.fact_id,f.subject,f.predicate,f.object,f.timestamp,f.confidence,t.rank
        FROM fts_facts t JOIN facts f ON f.rowid=t.rowid
        WHERE fts_facts MATCH ? AND {where} ORDER BY t.rank,f.rowid LIMIT ?
    """, (expression, *params, limit * 3)))
    if not rows:
        by_id = {}
        for token in tokens[:6]:
            like = "%" + token.replace("_", "\\_") + "%"
            for row in db.execute(f"""
                SELECT f.rowid,f.fact_id,f.subject,f.predicate,f.object,f.timestamp,f.confidence,0 AS rank
                FROM facts f WHERE (subject LIKE ? ESCAPE '\\' OR predicate LIKE ? ESCAPE '\\' OR object LIKE ? ESCAPE '\\')
                AND {where} LIMIT ?
            """, (like, like, like, *params, limit)):
                by_id[row["rowid"]] = row
        rows = list(by_id.values())
    ranks = ranked(rows, "rowid")
    results = []
    for row in rows:
        parts = [part for part in (row["subject"], row["predicate"]) if part.lower() not in ("entity", "fact")]
        searchable = " ".join([*parts, row["object"]])
        useful = [token for token in tokenize(query) if token not in FACT_FILLER or token in tokenize(searchable)]
        lexical = relevance(groups(" ".join(useful)), searchable)
        confidence = row["confidence"] if row["confidence"] is not None else .5
        value = round(lexical * (.7 + confidence * .2 + ranks[row["rowid"]] * .1), 4)
        if value > 0:
            results.append({"id": row["fact_id"], "content": preview(row["object"] or searchable), "source": "facts",
                            "timestamp": row["timestamp"], "importance": confidence, "store": "fact", "score": value})
    return sorted(results, key=lambda row: -row["score"])[:limit]


def mmr(rows, limit):
    remaining = [(row, set(row["content"].lower().split())) for row in sorted(rows, key=lambda row: -row["score"])]
    picked = []
    while remaining and len(picked) < limit:
        def adjusted(item):
            row, tokens = item
            similarity = max((len(tokens & other) / len(tokens | other) for _, other in picked if tokens | other), default=0.)
            return .7 * row["score"] - .3 * similarity
        best = max(range(len(remaining)), key=lambda index: adjusted(remaining[index]))
        picked.append(remaining.pop(best))
    return [row for row, _ in picked]


def adjusted_weights(query):
    lower = query.lower()
    intent, score = "general", 0
    for category, patterns in INTENT_PATTERNS:
        matches = sum(bool(re.search(pattern, lower)) for pattern in patterns)
        if matches > score:
            intent, score = category, matches
    base = (max(0., number("VEC_WEIGHT", .5)), max(0., number("FTS_WEIGHT", .3)),
            max(0., number("IMPORTANCE_WEIGHT", .2)))
    if not any(base):
        base = (.5, .3, .2)
    weights = tuple(weight * bias for weight, bias in zip(base, INTENT_WEIGHTS[intent]))
    total = sum(weights)
    return tuple(weight / total for weight in weights)


def diversify_by_coverage(rows, tokens, limit):
    pool, selected, covered = list(rows), [], set()
    terms = set(tokens)
    while pool and len(selected) < limit:
        best = max(range(len(pool)), key=lambda index: pool[index]["score"] + .06 * sum(
            word in terms and word not in covered for word in tokenize(pool[index]["content"])))
        row = pool.pop(best)
        selected.append(row)
        covered.update(terms.intersection(tokenize(row["content"])))
    return selected


def count_recalls(db, bank, picks):
    now = timestamp()
    params = bank.name, bank.name, now
    for kind, table in MEMORY_TABLES.items():
        db.executemany(f"""
            UPDATE {table} AS m SET recall_count=coalesce(recall_count,0)+1,last_recalled=?
            WHERE id=? AND {visible()}
        """, ((now, row["id"], *params) for row in picks if row["store"] == kind))


def rank_bank(bank, query, limit, query_vector=None, model=None, *, update_counts=True):
    query_groups = groups(query)
    tokens = list(dict.fromkeys(word for group in query_groups for word in group))
    threshold = (.08, .08, .18, .34)[len(tokens)] if len(tokens) <= 3 else .22
    expression = fts_query(query, tokens)
    now = timestamp()
    moment = datetime.fromisoformat(now)
    params = bank.name, bank.name, now
    current = bool(CURRENT_QUERY.search(query))
    weights = adjusted_weights(query)
    scored = []
    with open_bank(bank.path) as db:
        for kind in MEMORY_TABLES:
            candidates = memory_candidates(db, kind, params, expression, query_vector, model, max(limit * 6, 50))
            for row, fts, dense in candidates:
                if result := score(row, kind, fts, dense, query_groups, threshold, moment, current, weights):
                    scored.append(result)
        best = {}
        for row in sorted(scored, key=lambda item: -item["score"]):
            best.setdefault(row["content"], row)
        working = {row["id"] for row in best.values() if row["store"] == "working"}
        survivors = [row for row in best.values() if row["store"] == "working"
                     or not working.intersection((row.get("summary_of") or "").split(","))]
        if len(tokens) >= 4 and len(survivors) > limit * 2:
            survivors = diversify_by_coverage(survivors, tokens, limit * 2)
        picks = mmr(survivors, limit * 2)
        fact_limit = min(limit, max(3, (limit + 1) // 2))
        if fact_limit > 0:
            picks.extend(facts(db, bank.name, now, expression, tokens, query, fact_limit))
        picks = mmr(picks, limit)
        if update_counts:
            count_recalls(db, bank, picks)
    return [{**row, "bank": bank.name} for row in picks]
