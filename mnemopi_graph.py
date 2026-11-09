"""Mnemopi episodic gist and relationship ingestion."""

import json
import re
import sqlite3
from itertools import chain

import regex

from session_jsonl import timestamp


WORD = r"[\p{L}\p{N}_]"
NOUN = regex.compile(rf"(?<!{WORD})(\p{{Lu}}\p{{Ll}}+(?:\s+\p{{Lu}}\p{{Ll}}+)?)(?!{WORD})")
PRONOUN = regex.compile(rf"(?<!{WORD})(I|you|we|they|he|she|it|me|us|them|him|her)(?!{WORD})", regex.I)
TOKEN = regex.compile(r"[\p{L}\p{N}_-]+")
CONTENT_STOP = set("the and for with that this from into onto about was were are is has have had she he they them their our new".split())
CLAUSE_BREAK = r"(?:and|but|or|nor|yet|then|because|so|while|which|that|who|whom|whose|when|where|if|since|although|though|unless|until|whereas)\b"
ARTICLE = r"(?:(?:an?|the)\s+)?"
SUBJECT = r"\b([A-Z][a-zA-Z\s]+?)"
NEXT_CLAUSE = ARTICLE + r"(?:[A-Z][a-zA-Z\d]*[ \t]+)*[a-zA-Z\d]+\s+(?:is|has|uses?|using|used|works?)\b"
TEMPORAL = (
    (r"\b(yesterday|today|tomorrow|now|soon|later|earlier)\b", "point_in_time"),
    (r"\b(last\s+week|last\s+month|last\s+year|next\s+week)\b", "point_in_time"),
    (r"\b(since|from|starting)\b.*\b(until|to|through|end)\b", "duration"),
    (r"\b(between|from)\b.*\b(and|to)\b", "range"),
    (r"\b\d{1,2}:\d{2}\s*(AM|PM)?\b", "point_in_time"),
    (r"\b\d{4}-\d{2}-\d{2}\b", "point_in_time"),
)
PLACE = re.compile(r"\b(?:at|in|from)\s+([A-Z][a-zA-Z\s]+?)(?:\s+(?:yesterday|today|tomorrow|now|last|next|on|at)\b|$)")
GENERIC_PLACE = re.compile(r"\b(office|home|work|school|hospital|store|restaurant|building|room)\b", re.I)
EMOTION = (
    ("positive", "happy excited great awesome love enjoy glad pleased"),
    ("negative", "sad angry frustrated upset hate disappointed worried"),
    ("neutral", "fine okay alright normal standard"),
)


def phrase(word):
    token = rf"{word}(?:['-]{word})*\b"
    return rf"({token}(?:[ \t]+(?!{CLAUSE_BREAK}|{NEXT_CLAUSE}){token})*)"


FACT_PATTERNS = (
    (re.compile(SUBJECT + r"\s+is\s+" + ARTICLE + phrase("[a-zA-Z]+")), "is", .7),
    (re.compile(SUBJECT + r"\s+has\s+" + ARTICLE + phrase(r"[a-zA-Z\d]+")), "has", .6),
    (re.compile(SUBJECT + r"\s+(?:uses?|using|used)\s+" + ARTICLE + phrase("[a-zA-Z]+")), "uses", .6),
    (re.compile(SUBJECT + r"\s+works?\s+(?:at|for|with)\s+([A-Z][a-zA-Z]*(?:[ \t]+[A-Z][a-zA-Z]*)*)\b"), "works_at", .7),
)


def participants(text):
    names = {}
    for name in chain(NOUN.findall(text), PRONOUN.findall(text)):
        names.setdefault(name.lower(), name)
        if len(names) >= 5:
            break
    return list(names.values())


def graph_facts(text):
    count = 0
    text = text[:4096]
    for pattern, predicate, confidence in FACT_PATTERNS:
        for match in pattern.finditer(text):
            subject, object_ = (value.strip() for value in match.groups())
            if len(subject) <= 2 or len(object_) <= 2:
                continue
            yield subject, predicate, object_, confidence
            count += 1
            if count == 5:
                return


def content_terms(text):
    return {token for token in TOKEN.findall(text.lower()) if len(token) >= 3 and token not in CONTENT_STOP}


def entities(db, identifier):
    return {value.strip().lower() for row in db.execute("SELECT subject,object FROM facts WHERE source_msg_id=?", (identifier,))
            for value in row if isinstance(value, str) and value.strip()}


def add_edge(db, source, target, kind, weight, stamp):
    db.execute("""
        INSERT INTO graph_edges(source,target,edge_type,weight,timestamp) VALUES (?,?,?,?,?)
        ON CONFLICT(source,target,edge_type) DO UPDATE SET weight=excluded.weight,timestamp=excluded.timestamp
    """, (source, target, kind, weight, stamp))


def link_memories(db, identifier, text, stamp):
    source_tokens, source_entities = content_terms(text), entities(db, identifier)
    live = set()
    rows = db.execute("""
        SELECT id,content FROM working_memory WHERE id<>? AND superseded_by IS NULL AND (valid_until IS NULL OR valid_until>?)
        UNION ALL
        SELECT id,content FROM episodic_memory WHERE id<>? AND superseded_by IS NULL AND (valid_until IS NULL OR valid_until>?)
        UNION ALL
        SELECT memory_id,text FROM gists g WHERE memory_id<>?
            AND NOT EXISTS(SELECT 1 FROM working_memory w WHERE w.id=g.memory_id)
            AND NOT EXISTS(SELECT 1 FROM episodic_memory e WHERE e.id=g.memory_id)
    """, (identifier, stamp, identifier, stamp, identifier))
    for target, content in rows:
        if target in live:
            continue
        live.add(target)
        target_tokens = content_terms(content)
        overlap = len(source_tokens & target_tokens)
        lexical = round(overlap / len(source_tokens | target_tokens), 3) if overlap else 0.
        references = 0.
        if source_entities:
            target_entities = entities(db, target)
            if target_entities:
                references = round(len(source_entities & target_entities) / max(len(source_entities), len(target_entities)), 3)
        if lexical >= .35:
            add_edge(db, identifier, target, "related_to", lexical, stamp)
        if references > 0:
            add_edge(db, identifier, target, "references", references, stamp)
        weight = lexical if lexical >= .35 else references
        if weight >= .35:
            add_edge(db, identifier, target, "ctx", weight, stamp)


def ingest_graph(db, session, identifier, text):
    """Graph enrichment remains best-effort within the caller's transaction."""
    stamp, gist = timestamp(), "gist_" + identifier
    sentence = re.split(r"[.!?]+", text, maxsplit=1)[0].strip()
    summary = sentence[:100] if len(sentence) > 10 else text[:100].strip()
    location = PLACE.search(text) or GENERIC_PLACE.search(text)
    lower = text.lower()
    emotion = next((name for name, words in EMOTION if any(word in lower for word in words.split())), None)
    scope = next((scope for pattern, scope in TEMPORAL if re.search(pattern, text, re.I)), None)
    db.execute("SAVEPOINT mnemopi_graph")
    try:
        db.execute("""
            INSERT OR REPLACE INTO gists(id,text,timestamp,participants_json,location,emotion,time_scope,memory_id)
            VALUES (?,?,?,?,?,?,?,?)
        """, (gist, summary, stamp, json.dumps(participants(text), ensure_ascii=False),
              location[1].strip() if location else None, emotion, scope, identifier))
        add_edge(db, identifier, gist, "ctx", 1., stamp)
        for index, (subject, predicate, object_, confidence) in enumerate(graph_facts(text)):
            fact = f"fact_{identifier}_{index}"
            db.execute("""
                INSERT OR IGNORE INTO facts(fact_id,session_id,subject,predicate,object,timestamp,source_msg_id,confidence)
                VALUES (?,?,?,?,?,?,?,?)
            """, (fact, session, subject, predicate, object_, stamp, identifier, confidence))
            add_edge(db, gist, fact, "rel", confidence, stamp)
        link_memories(db, identifier, text, stamp)
    except sqlite3.Error:
        db.execute("ROLLBACK TO mnemopi_graph")
    finally:
        db.execute("RELEASE mnemopi_graph")
