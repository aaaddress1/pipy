"""BEAM age-gated working-memory promotion and episodic tier degradation."""

import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from uuid import uuid4

import regex

from mnemopi_recall import VERACITY, number
from mnemopi_extract import project_patterns
from mnemopi_graph import ingest_graph
from mnemopi_store import open_bank
from runtime import pack


CATEGORY = dict((
    ("PREFERENCE", "PREF"),
    ("TRAIT", "TRAIT"),
    ("STATUS", "STAT"),
    ("INSTRUCTION", "INST"),
    ("PROJECT", "PROJ"),
    ("LOCATION", "LOC"),
    ("FAMILY", "FAM"),
    ("OCCUPATION", "OCC"),
    ("DECISION", "DEC"),
    ("EVENT", "EVT"),
    ("TOOL", "TOOL"),
    ("FACT", "FACT"),
    ("OPINION", "OPN"),
))
PHRASES = (
    ("User asked ", "ASK "),
    ("User wants ", "WANT "),
    ("User prefers ", "PREF "),
    ("User likes ", "LIKE "),
    ("User dislikes ", "DISLIKE "),
    ("User is ", "IS "),
    ("User has ", "HAS "),
    ("User built ", "BUILT "),
    ("User asked for ", "ASK "),
    ("User requested ", "REQ "),
    ("Married to ", "MARRIED→"),
    ("Email: ", "@"),
    ("GitHub: ", "GH:"),
    ("Location: ", "LOC:"),
    ("Phone: ", "PH:"),
    ("User email is ", "@"),
    ("User voice message ", "VM "),
    ("User stack: ", "STACK|"),
    ("Full-stack developer", "FSDEV"),
    ("Software Developer", "SDEV"),
    ("AI Systems Engineer", "AIENG"),
    ("real-time", "RT"),
    ("Real-time", "RT"),
    ("bilingual", "bi"),
    ("Bilingual", "bi"),
    ("self-hosted", "selfhost"),
    ("automation", "auto"),
    ("transcription", "transc"),
    ("translation", "transl"),
)
STRUCTURE = (
    (" - ", " | "),
    (" -- ", " | "),
    (" | ", " | "),
    (", ", " | "),
    (" and ", "+"),
    (" or ", "/"),
    (" for ", "→"),
    (" to ", "→"),
    (" with ", " w/ "),
    (" over ", ">"),
    (" instead of ", "!>"),
    (" because of ", "∵"),
    (" due to ", "∵"),
    (" using ", "→"),
    (" built ", "→"),
    (" in ", ":"),
    (" at ", "@"),
    (" on ", "@"),
    (" from ", "<-"),
)
WORD = r"[\p{L}\p{N}_]"
STATUS = (("working correctly", "OK"), ("working", "OK"), ("complete", "DONE"), ("completed", "DONE"))
TRUNCATION = "\n[... sleep_consolidation episode truncated by maxEpisodeChars ...]"


def pattern(text):
    start = f"(?<!{WORD})" if regex.match(WORD, text[0]) else ""
    end = f"(?!{WORD})" if regex.match(WORD, text[-1]) else ""
    return regex.compile(start + regex.escape(text) + end)


PHRASE_PATTERNS = [(pattern(text), value) for text, value in sorted(PHRASES, key=lambda pair: -len(pair[0]))]
STATUS_PATTERNS = [(pattern(text), value) for text, value in STATUS]


def encode(text):
    if not text or "|" in text and len(text.strip().split()) <= 3:
        return text
    text = text.strip()
    name, separator, rest = text.partition(": ")
    if separator and name in CATEGORY:
        text = CATEGORY[name] + "|" + rest
    for phrase, value in PHRASE_PATTERNS:
        text = phrase.sub(value, text)
    for phrase, value in STRUCTURE:
        text = text.replace(phrase, value)
    text = re.sub(r"\(\s*", "(", text).replace(" )", ")")
    for phrase, value in STATUS_PATTERNS:
        text = phrase.sub(value, text)
    return text.strip()


def batches(rows, budget):
    batch, size = [], 0
    for row in rows:
        length = len(row["content"])
        if batch and size + 3 + length > budget:
            yield batch, size
            batch, size = [], 0
        size += length + (3 if batch else 0)
        batch.append(row)
    if batch:
        yield batch, size


def veracity(rows):
    weights = {**VERACITY, "true": 1., "false": 0.}
    counts = Counter(row["veracity"] for row in rows if row["veracity"] in weights and row["veracity"] != "unknown")
    return min(counts, key=lambda key: (-counts[key], weights[key])) if counts else "unknown"


def key_signal(content, maximum):
    sentences = [text.strip() for text in re.split(r"(?<=[.!?])\s+", content) if text.strip()]
    if not sentences:
        return content[:maximum]
    def weight(sentence):
        terms = r"\b(?:prefer|always|never|deadline|release|version|decided|important|must|should)\b"
        return 2 * len(re.findall(r"\b[A-Z][a-zA-Z0-9_-]+\b", sentence)) + len(re.findall(terms, sentence, re.I))
    selected, used = [], 0
    for index, sentence in sorted(enumerate(sentences), key=lambda item: -weight(item[1])):
        if selected and used + len(sentence) + 1 > maximum:
            continue
        selected.append((index, sentence))
        used += len(sentence) + 1
        if used >= maximum:
            break
    text = " ".join(sentence for _, sentence in sorted(selected))
    return text if len(text) <= maximum else text[:max(0, maximum - 6)].strip() + " [...]"


def iso(moment):
    return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def degrade(db, now):
    count = max(1, int(number("DEGRADE_BATCH", 100)))
    maximum = max(7, int(number("TIER3_MAX_CHARS", 300)))
    targets = [(tier, row) for tier, days, limit in ((1, 30, count), (2, 180, max(1, count // 2)))
               for row in db.execute("""
                   SELECT id,content FROM episodic_memory WHERE tier=? AND created_at<? ORDER BY created_at LIMIT ?
               """, (tier, iso(now - timedelta(days=max(0, number(f"TIER{tier + 1}_DAYS", days)))), limit))]
    for tier, row in targets:
        old = row["content"]
        content = old[:800] if tier == 1 else key_signal(old, maximum) if len(old) > maximum else old
        db.execute("UPDATE episodic_memory SET content=?,tier=?,degraded_at=? WHERE id=?",
                   (content, tier + 1, iso(now), row["id"]))
        if content != old:
            db.execute("DELETE FROM memory_embeddings WHERE memory_id=?", (row["id"],))
            db.execute("UPDATE episodic_memory SET binary_vector=NULL WHERE id=?", (row["id"],))


def sleep_bank(bank, *, all_sessions=False):
    now = datetime.now(timezone.utc)
    stamp = iso(now)
    cutoff = iso(now - timedelta(hours=max(0, int(number("WM_TTL_HOURS", 24)) // 2)))
    limit = max(1, int(number("SLEEP_BATCH", 5000)))
    maximum = max(1, int(number("MAX_EPISODE_CHARS", 100_000)))
    episodes, retained = [], 0
    with open_bank(bank.path, write=True) as db:
        sessions = [bank.name]
        if all_sessions:
            sessions = [row[0] for row in db.execute("""
                SELECT coalesce(session_id,'default') FROM working_memory WHERE timestamp<? AND consolidated_at IS NULL
                GROUP BY coalesce(session_id,'default') ORDER BY min(timestamp)
            """, (cutoff,))]
        for session in sessions:
            grouped = defaultdict(list)
            for row in db.execute("""
                SELECT id,coalesce(embed_text,content) AS content,source,scope,valid_until,veracity
                FROM working_memory WHERE coalesce(session_id,'default')=? AND timestamp<? AND consolidated_at IS NULL
                ORDER BY timestamp LIMIT ?
            """, (session, cutoff, limit)):
                grouped[row["source"] or "unknown"].append(row)
            count = sum(len(rows) for rows in grouped.values())
            if not count:
                continue
            start = len(episodes)
            for source, rows in grouped.items():
                prefix = f"[{source}] "
                for batch, size in batches(rows, max(0, maximum - len(prefix))):
                    content = prefix + encode(" | ".join(row["content"] for row in batch))
                    metadata = {"original_count": len(batch), "source": source, "llm_used": False}
                    if len(content) > maximum:
                        content = (content[:maximum] if maximum <= len(TRUNCATION) else
                                   content[:maximum - len(TRUNCATION)].rstrip() + TRUNCATION)
                        metadata.update(truncated=True, original_chars=size, max_chars=maximum)
                    identifier = sha256((content + "\0" + stamp + "\0" + uuid4().hex).encode()).hexdigest()[:16]
                    ids = [row["id"] for row in batch]
                    scope = "global" if any(row["scope"] == "global" for row in batch) else "session"
                    expiry = min((row["valid_until"] for row in batch if row["valid_until"]), default=None)
                    db.execute("""
                        INSERT INTO episodic_memory(id,content,source,timestamp,session_id,importance,metadata_json,summary_of,
                            valid_until,scope,author_id,author_type,channel_id,memory_type,veracity,created_at)
                        VALUES (?,?,'sleep_consolidation',?,?,.6,?,?,?,?,'coding-agent','agent',?,'unknown',?,?)
                    """, (identifier, content, stamp, session, pack(metadata), ",".join(ids), expiry, scope,
                          bank.name, veracity(batch), stamp))
                    project_patterns(db, session, identifier, content, polyphonic=bank.polyphonic)
                    ingest_graph(db, session, identifier, content)
                    db.executemany("UPDATE working_memory SET consolidated_at=? WHERE id=?", ((stamp, id_) for id_ in ids))
                    episodes.append((identifier, content))
            db.execute("INSERT INTO consolidation_log(session_id,items_consolidated,summary_preview,created_at) VALUES (?,?,?,?)",
                       (session, count, f"{len(episodes) - start} summaries (aaak) from {count} items", stamp))
            retained += count
        if retained:
            degrade(db, now)
    return retained, episodes
