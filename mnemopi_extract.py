"""Mnemopi fact extraction and source-linked MEMORIA projections."""

import json
import os
import re
import sqlite3
from hashlib import sha256
from itertools import islice
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from auth_broker import dotted
from mnemopi_store import open_bank
from mnemopi_veracity import consolidate_fact
from session_jsonl import timestamp


SYSTEM = (Path(__file__).with_name("prompts") / "mnemopi-extraction-system.md").read_text("utf-8").strip()
REMOTE_PROMPT = """You are an expert structured memory extractor for Mnemopi v3.0+ MEMORIA tables.
The user message below may be in English, German, Russian, or another language.
First detect the language, then extract ONLY high-signal, long-term relevant items.
Categories to extract (return valid JSON only, no extra text):
- facts: persistent user metrics, states, knowledge, or personal data
  (Examples: 'my name is X', 'I work at Y', 'server runs on port 8080')
- instructions: rules or commands directed at me the agent
  (Examples: 'always use tabs', 'never delete logs', 'call me boss')
- preferences: likes, dislikes, and their evolution
  (Examples: 'I like dark mode', 'I prefer Python over Go')
- timelines: real events with dates/times
  (Examples: 'release on 2024-12-01', 'meeting next Tuesday')
- kg: knowledge-graph triples in subject-predicate-object form

Rules:
- Only extract persistent, non-transient content. Ignore weather, one-off chat, system text.
- Use semantic understanding — do NOT rely on English keywords.
- Preserve original casing and language.
- If nothing qualifies, return empty arrays.

Return JSON in this exact format:
{"facts": [], "instructions": [], "preferences": [], "timelines": [], "kg": []}

User message: {text}

Extraction:"""
FACT_FIELDS = ("fact", "text", "content", "value", "statement")
CATEGORY_FIELDS = {"facts": FACT_FIELDS, "instructions": ("instruction", "rule", *FACT_FIELDS),
                   "preferences": ("preference", *FACT_FIELDS),
                   "timelines": ("description", "event", "timeline", "date", *FACT_FIELDS)}
DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
VERSION = re.compile(r"\bv?\d+\.\d+(?:\.\d+)?(?:[-+][A-Za-z0-9.]+)?\b")
METRIC = re.compile(r"(\d+(?:[.,]\d+)?)\s*(ms|sec|seconds?|minutes?|hours?|days?|weeks?|months?|%|KB|MB|GB|TB|"
                    r"rows?|columns?|roles?|features?|bugs?|commits?|cards?|users?|items?|tests?|APIs?|endpoints?|sprints?|tickets?)\b", re.I)
TIMELINE = re.compile(r"\b(?:release|deadline|meeting|launch|ship|shipped|due|start|started|finish|finished)\b", re.I)
RELATION = re.compile(r"\b([A-Z][A-Za-z0-9_-]{2,})\s+(?:is|uses|runs|owns|depends on)\s+([^.!?;]{2,80})")
BOUNDARY = re.compile(r"\n|\||`|[.!?](?=\s|$)|;")
METRIC_STOP = set("the and for was of to a an in on at by is are has had not but or".split())
HEURISTIC = (
    (r"my name is\s+", "The user's name is "),
    (r"i (?:am|work as)\s+(?:an?\s+)?", "The user is "),
    (r"i work (?:at|for)\s+", "The user works at "),
    (r"i (?:live in|am based in)\s+", "The user lives in "),
    (r"i (?:use|uses|am using)\s+", "The user uses "),
    (r"i (?:like|love|prefer|enjoy)\s+", "The user prefers "),
    (r"i (?:hate|dislike|do not like|don't like)\s+", "The user dislikes "),
)


def normalized(value):
    return re.sub(r"[.!?]+$", "", value.strip()) if isinstance(value, str) else ""


def category_text(items, fields, *, joined=False):
    if not isinstance(items, list):
        return []
    values = []
    for item in items:
        if isinstance(item, dict):
            parts = [item[key].strip() for key in fields if isinstance(item.get(key), str) and item[key].strip()]
            item = " ".join(parts if joined else parts[:1])
        if text := normalized(item):
            values.append(text)
            if len(values) == 5:
                break
    return values


def parse_categories(output):
    categories = {key: [] for key in (*CATEGORY_FIELDS, "kg")}
    if not isinstance(output, str) or not output.strip() or output.strip().upper() == "NO_FACTS":
        return categories
    raw = output.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I) if raw.startswith("```") else raw
    if raw.startswith("{"):
        try:
            value = json.loads(raw)
        except ValueError:
            matches = re.findall(r'"([^"]{10,})"', output)
            if matches:
                categories["facts"] = list(islice(filter(None, map(normalized, matches)), 5))
                return categories
        else:
            if isinstance(value, dict):
                for key, fields in CATEGORY_FIELDS.items():
                    categories[key] = category_text(value.get(key), fields, joined=key == "timelines")
                for item in value["kg"] if isinstance(value.get("kg"), list) else ():
                    if isinstance(item, dict):
                        item = [item.get(key) for key in ("subject", "predicate", "object")]
                    if isinstance(item, list) and len(item) >= 3 and all(triple := tuple(map(normalized, item[:3]))):
                        categories["kg"].append(triple)
                        if len(categories["kg"]) == 5:
                            break
                return categories
    for line in output.splitlines():
        line = re.sub(r"^[\s\d.\-*]+", "", line).strip()
        if len(line) > 10 and (fact := normalized(line)):
            categories["facts"].append(fact)
            if len(categories["facts"]) == 20:
                break
    return categories


def heuristic_facts(text):
    facts = {}
    for clause in re.split(r"[.!?;]+|\s+(?:and|but)\s+", " ".join(text.split()), flags=re.I):
        for pattern, prefix in HEURISTIC:
            if match := re.search(r"\b" + pattern + r"([^,.!?;]+)", clause, re.I):
                fact = normalized(prefix + match[1])
                if len(fact) > 10:
                    facts.setdefault(fact, None)
        if match := re.search(r"\b(?:i|you)\s+(always|never)\s+([^,.!?;]+)", clause, re.I):
            fact = normalized(f"Instruction: {match[1].lower()} {match[2]}")
            if len(fact) > 10:
                facts.setdefault(fact, None)
        if len(facts) >= 5:
            break
    return list(islice(facts, 5))


def extraction_window(text, maximum=8192):
    marker = "\n\n[...]\n\n"
    if len(text) <= maximum:
        return text
    if maximum <= len(marker) + 16:
        return text[-maximum:] if maximum else ""
    budget = maximum - len(marker)
    half = budget // 2
    return text[:half] + marker + text[-(budget - half):]


def clean_output(text):
    for marker in ("<|assistant|>", "<|user|>", "</s>"):
        text = text.replace(marker, "")
    text = re.sub(r"^(?:\s*<think>.*?</think>)+\s*", "", text.strip(), flags=re.I | re.S)
    text = re.sub(r"^(?:Summarize the following memories|Preserve facts).*?[.!?:]\s*", "", text, flags=re.I | re.S)
    text = re.sub(r"^Source:.*?\n", "", text, count=1, flags=re.I | re.M)
    return re.sub(r"^\s*[-*]\s.*\n", "", text, flags=re.M).strip()


def setting(settings, name, environment, default=""):
    value = dotted(settings, "mnemopi." + name)
    return value.strip() if isinstance(value, str) and value.strip() else os.getenv(environment, default).strip()


class FactExtractor:
    def __init__(self, settings, client):
        self.client = client
        mode = dotted(settings, "mnemopi.llmMode")
        self.mode = mode if mode in ("none", "smol", "remote") else "smol"
        self.complete = None
        self.endpoint = setting(settings, "llmBaseUrl", "MNEMOPI_LLM_BASE_URL").rstrip("/") if self.mode == "remote" else ""
        self.key = setting(settings, "llmApiKey", "MNEMOPI_LLM_API_KEY") if self.endpoint else ""
        self.model = setting(settings, "llmModel", "MNEMOPI_LLM_MODEL", "local") or "local"
        url = urlsplit(self.endpoint)
        if self.endpoint and (url.scheme not in ("http", "https") or not url.hostname or url.username is not None
                              or url.query or url.fragment):
            raise ValueError("mnemopi.llmBaseUrl must be an HTTP(S) URL without credentials, query, or fragment.")
        limit = re.match(r"^[+-]?\d+", os.getenv("MNEMOPI_LLM_MAX_TOKENS", "").strip())
        self.maximum = max(1, int(limit[0])) if limit else 2048
        flag = os.getenv("MNEMOPI_LLM_ENABLED", "").strip().lower()
        self.enabled = not flag or flag in ("1", "true", "yes", "on")

    def remote(self, prompt, signal):
        headers = {"Authorization": "Bearer " + self.key} if self.key else {}
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
                "max_tokens": self.maximum, "stop": ["</s>", "<|user|>"]}
        for attempt in range(3):
            signal.check()
            try:
                result = self.client.post(self.endpoint + "/chat/completions", headers=headers, json=body, timeout=60)
                if result.is_success:
                    content = result.json()["choices"][0]["message"]["content"]
                    return clean_output(content) if isinstance(content, str) else None
                if result.status_code not in (408, 425, 429, 500, 502, 503, 504):
                    return None
            except httpx.TransportError:
                pass
            except (ValueError, KeyError, IndexError, TypeError):
                return None
            if attempt < 2:
                signal.wait(2**attempt)
        return None

    def extract(self, text, signal):
        text = extraction_window(text)
        raw = None
        if self.enabled and self.mode == "smol" and self.complete:
            raw = self.complete(SYSTEM, text, self.maximum, signal)
        elif self.enabled and self.mode == "remote" and self.endpoint:
            template = os.getenv("MNEMOPI_EXTRACTION_PROMPT") or REMOTE_PROMPT
            raw = self.remote(template.replace("{text}", text).replace("{lang}", "en"), signal)
        categories = parse_categories(raw)
        if not any(categories.values()):
            categories["facts"] = heuristic_facts(text)
        return categories


class Projections:
    def __init__(self, db, session, source, *, polyphonic=False):
        self.db, self.session, self.source = db, session, source
        self.now = timestamp()
        self.polyphonic = polyphonic

    def unique(self, table, key, fields):
        names = tuple(fields)
        values = tuple(fields.values())
        conditions = " AND ".join(f"{name} IS ?" for name in key)
        return self.db.execute(f"""
            INSERT INTO {table} ({', '.join(names)}) SELECT {', '.join('?' for _ in values)}
            WHERE NOT EXISTS (SELECT 1 FROM {table} WHERE {conditions})
        """, (*values, *(fields[name] for name in key))).rowcount > 0

    def fact(self, kind, key, value, context, importance):
        identity = "\0".join((self.session, kind, key, value, self.source))
        identifier = sha256(identity.encode()).hexdigest()[:16]
        inserted = self.db.execute("""
            INSERT OR IGNORE INTO facts(fact_id,session_id,subject,predicate,object,timestamp,source_msg_id,confidence)
            VALUES (?,?,?,?,?,?,?,?)
        """, (identifier, self.session, key, kind, value, self.now, self.source, importance))
        if inserted.rowcount:
            self.db.execute("""
                INSERT INTO memoria_facts(session_id,message_idx,fact_type,key,value,context_snippet,importance,timestamp,source_memory_id)
                VALUES (?,0,?,?,?,?,?,?,?)
            """, (self.session, kind, key, value, context, importance, self.now, self.source))

    def preference(self, content, topic):
        self.unique("memoria_preferences", ("session_id", "preference", "source_memory_id"), {
            "session_id": self.session, "message_idx": 0, "preference": content, "topic": topic,
            "context_snippet": content, "source_memory_id": self.source,
        })

    def instruction(self, content, context):
        self.unique("memoria_instructions", ("session_id", "instruction", "source_memory_id"), {
            "session_id": self.session, "message_idx": 0, "instruction": content, "active": 1,
            "context_snippet": context, "source_memory_id": self.source,
        })

    def timeline(self, content, date):
        self.unique("memoria_timelines", ("session_id", "date", "description", "source_memory_id"), {
            "session_id": self.session, "date": date, "message_idx": 0, "description": content,
            "source": "extraction", "source_memory_id": self.source,
        })

    def triple(self, subject, predicate, object_):
        triple = {"subject": subject, "predicate": predicate, "object": object_}
        inserted = self.unique("memoria_kg", ("session_id", *triple, "source_memory_id"), {
            "session_id": self.session, **triple, "message_idx": 0, "confidence": .65, "source_memory_id": self.source,
        })
        self.unique("triples", (*triple, "source"), {**triple, "valid_from": self.now, "source": self.source, "confidence": .65})
        if inserted and self.polyphonic:
            self.db.execute("SAVEPOINT mnemopi_fact")
            try:
                consolidate_fact(self.db, subject, predicate, object_, self.source)
            except sqlite3.Error:
                self.db.execute("ROLLBACK TO mnemopi_fact")
            finally:
                self.db.execute("RELEASE mnemopi_fact")

    def categories(self, items, clean):
        for category in CATEGORY_FIELDS:
            for value in items[category]:
                value = clean(value)
                if not value:
                    continue
                self.fact("entity", "fact", value, value, .7)
                if category == "instructions":
                    self.instruction(value, value)
                elif category == "preferences":
                    self.preference(value, None)
                elif category == "timelines":
                    match = DATE.search(value)
                    self.timeline(value, match[0] if match else None)
                elif category == "facts":
                    if preference := re.match(r"^The user (?:prefers|dislikes) (.+)$", value, re.I):
                        self.preference(value, preference[1])
                    if instruction := re.match(r"^Instruction: (.+)$", value, re.I):
                        self.instruction(instruction[1], value)
        for triple in items["kg"]:
            values = tuple(map(clean, triple))
            if all(values):
                self.triple(*values)


def store_categories(bank, identifier, expected, items, clean):
    if not any(items.values()):
        return
    with open_bank(bank.path, write=True) as db:
        row = db.execute("""
            SELECT session_id FROM working_memory WHERE id=? AND content=? AND superseded_by IS NULL
                AND (valid_until IS NULL OR valid_until>?)
        """, (identifier, expected, timestamp())).fetchone()
        if row:
            Projections(db, row["session_id"], identifier, polyphonic=bank.polyphonic).categories(items, clean)


def project_patterns(db, session, identifier, text, *, polyphonic=False):
    if len(text) > 50_000:
        return
    projections = Projections(db, session, identifier, polyphonic=polyphonic)
    def snippet(index, width=50):
        return " ".join(text[max(0, index - width):index + width].split())

    for match in islice(METRIC.finditer(text), 10):
        unit = match[2].lower()
        if unit.endswith("s") and not unit.endswith("ms"):
            unit = unit[:-1]
        window = text[max(0, match.start() - 50):match.start()]
        boundary = max((part.start() for part in BOUNDARY.finditer(window)), default=-1)
        words = [re.sub(r"""[.,:;!?()\[\]"'`*_]""", "", word) for word in window[boundary + 1:].split()]
        words = [word.lower() for word in words if len(word) > 2 and word.lower() not in METRIC_STOP]
        key = "_".join([*words[-3:], "pct" if unit == "%" else unit])
        projections.fact("metric", key, match[1] + match[2], snippet(match.start()), .65)
    for match in DATE.finditer(text):
        context = snippet(match.start(), 100)
        projections.fact("date", "iso_date", match[0], context, .5)
        if TIMELINE.search(context):
            projections.timeline(context, match[0])
    for match in VERSION.finditer(text):
        projections.fact("version", "version", match[0], snippet(match.start()), .6)
    projections.categories({key: heuristic_facts(text) if key == "facts" else [] for key in (*CATEGORY_FIELDS, "kg")}, str)
    for match in RELATION.finditer(text):
        projections.triple(match[1], "related_to", " ".join(match[2].split()))
