"""Remote Mnemopi embeddings and atomic bank-wide model reconciliation."""

import json
import math
import os
import re
import time
from array import array
from collections import OrderedDict
from threading import Lock
from urllib.parse import urlsplit

import httpx

from auth_broker import dotted
from cancellable_http import active_turn
from mnemopi_extract import extraction_window
from mnemopi_legacy_vectors import delete_legacy_vectors
from mnemopi_recall import vector
from mnemopi_store import open_bank
from rate_limits import retry_after
from reset_fence import try_lock
from session_jsonl import timestamp


REBUILD_BATCH = 128
QUERY_CACHE_MAX = 512
RETRYABLE = frozenset((408, 425, 429, 500, 502, 503, 504))
EMBEDDABLE = """
    SELECT id,coalesce(embed_text,content) AS text FROM working_memory
    WHERE superseded_by IS NULL AND (valid_until IS NULL OR valid_until>:now)
    UNION ALL
    SELECT id,content AS text FROM episodic_memory
    WHERE superseded_by IS NULL AND (valid_until IS NULL OR valid_until>:now)
"""


def env_int(name, fallback, minimum=0):
    raw = os.getenv("MNEMOPI_" + name, "").strip()
    match = re.match(r"^[+-]?\d+", raw)
    try:
        value = int(match[0]) if match else fallback
    except ValueError:
        return fallback
    return value if value >= minimum else fallback


def matrix_response(response, size):
    data = response.json().get("data")
    if not isinstance(data, list) or len(data) != size or any(not isinstance(row, dict) for row in data):
        return None
    indices = [row.get("index") for row in data]
    if any(index is not None for index in indices):
        if any(type(index) is not int for index in indices) or sorted(indices) != list(range(size)):
            return None
        data = sorted(data, key=lambda row: row["index"])
    matrix = []
    for row in data:
        coordinates = vector(row.get("embedding"))
        if coordinates is None:
            return None
        rounded = list(array("f", coordinates))
        if not all(map(math.isfinite, rounded)) or matrix and len(rounded) != len(matrix[0]):
            return None
        matrix.append(rounded)
    return matrix


class Embeddings:
    def __init__(self, settings, client, clean, notice):
        self.client, self.clean, self.notice = client, clean, notice
        token = os.getenv("MNEMOPI_EMBEDDING_API_KEY") or dotted(settings, "mnemopi.embeddingApiKey") or ""
        self.token = token if isinstance(token, str) else ""
        endpoint = os.getenv("MNEMOPI_EMBEDDING_API_URL") or dotted(settings, "mnemopi.embeddingApiUrl") or ""
        self.endpoint = endpoint.strip().rstrip("/") if isinstance(endpoint, str) else ""
        url = urlsplit(self.endpoint)
        if self.endpoint and (url.scheme not in ("https", "http") or not url.hostname or url.username is not None
                              or url.query or url.fragment):
            raise ValueError("mnemopi.embeddingApiUrl must be an HTTP(S) URL without credentials, query, or fragment.")
        disabled = os.getenv("MNEMOPI_NO_EMBEDDINGS", "").strip().lower()
        if dotted(settings, "mnemopi.noEmbeddings") is True or disabled in ("1", "true", "yes", "on"):
            self.endpoint = ""
        multilingual = dotted(settings, "mnemopi.embeddingVariant") == "multilingual"
        default_model = "intfloat/multilingual-e5-large" if multilingual else "BAAI/bge-base-en-v1.5"
        model = os.getenv("MNEMOPI_EMBEDDING_MODEL") or dotted(settings, "mnemopi.embeddingModel") or default_model
        self.model = model.strip() if isinstance(model, str) and model.strip() else default_model
        self.batch_size = min(REBUILD_BATCH, env_int("EMBEDDING_BATCH_SIZE", 64, 1))
        self.max_chars = env_int("EMBEDDING_MAX_INPUT_CHARS", 8192)
        self.queries = OrderedDict()
        self.lock = Lock()

    def request_batch(self, texts):
        headers = {"Authorization": "Bearer " + self.token} if self.token else {}
        body = {"model": self.model, "input": texts}
        signal = active_turn()
        for attempt in range(3):
            if signal:
                signal.check()
            delay = 2**attempt
            try:
                response = self.client.post(self.endpoint + "/embeddings", headers=headers, timeout=30, json=body)
                if response.is_success:
                    return matrix_response(response, len(texts))
                if response.status_code not in RETRYABLE:
                    break
                hint = retry_after(response)
                if hint is not None:
                    if hint > 30:
                        break
                    delay = max(delay, hint)
            except httpx.TransportError:
                pass
            except (AttributeError, ValueError, TypeError, OverflowError):
                break
            if attempt < 2:
                signal.wait(delay) if signal else time.sleep(delay)
        self.notice("Mnemopi embedding request failed.")
        return None

    def many(self, texts):
        if not self.endpoint or not texts:
            return None
        matrix = []
        for offset in range(0, len(texts), self.batch_size):
            part = [self.clean(text) for text in texts[offset:offset + self.batch_size]]
            if self.max_chars:
                part = [extraction_window(text, self.max_chars) for text in part]
            rows = self.request_batch(part)
            if rows is None or matrix and len(rows[0]) != len(matrix[0]):
                return None
            matrix.extend(rows)
        return matrix

    def query(self, text):
        if not self.endpoint or not text:
            return None
        with self.lock:
            if text in self.queries:
                self.queries.move_to_end(text)
                return self.queries[text]
        rows = self.many([text])
        if rows:
            with self.lock:
                self.queries[text] = rows[0]
                self.queries.move_to_end(text)
                while len(self.queries) > QUERY_CACHE_MAX:
                    self.queries.popitem(last=False)
            return rows[0]
        return None

    def reconcile(self, bank, signal):
        if not self.endpoint:
            return
        release = try_lock(bank.path.with_name(bank.path.name + ".embedding-rebuild.lock"))
        if not release:
            return
        try:
            with open_bank(bank.path) as db:
                self.rebuild(db, bank, signal)
        finally:
            release()

    def rebuild(self, db, bank, signal):
        stale = db.execute("SELECT 1 FROM memory_embeddings WHERE model IS NOT ? LIMIT 1", (self.model,)).fetchone()
        dimension = None
        if stale:
            probe = self.many(["mnemopi embedding probe"])
            if not probe:
                self.notice("Mnemopi embedding model probe failed. Stored vectors were retained.")
                return
            dimension = len(probe[0])
        db.execute("PRAGMA temp_store=FILE")
        db.execute("""CREATE TEMP TABLE embedding_rebuild (
            seq INTEGER PRIMARY KEY, memory_id TEXT NOT NULL, text TEXT NOT NULL, embedding_json TEXT
        )""")
        missing = "" if stale else """AND NOT EXISTS (
            SELECT 1 FROM memory_embeddings e WHERE e.memory_id=m.id AND e.model IS :model
        )"""
        db.execute(f"""
            INSERT INTO embedding_rebuild(memory_id,text)
            SELECT id,text FROM ({EMBEDDABLE}) m WHERE trim(text)<>'' {missing}
        """, {"now": timestamp(), "model": self.model})
        db.commit()
        cursor = 0
        while True:
            signal.check()
            rows = db.execute("""
                SELECT seq,memory_id,text FROM embedding_rebuild WHERE seq>? ORDER BY seq LIMIT ?
            """, (cursor, REBUILD_BATCH)).fetchall()
            if not rows:
                break
            cursor = rows[-1]["seq"]
            rows = [row for row in rows if row["text"].strip()]
            if not rows:
                continue
            matrix = self.many([row["text"] for row in rows])
            if matrix is None or dimension is not None and len(matrix[0]) != dimension:
                if stale:
                    self.notice("Mnemopi embedding rebuild failed. Stored vectors were retained.")
                return
            dimension = len(matrix[0])
            signal.check()
            if stale:
                db.executemany("UPDATE embedding_rebuild SET embedding_json=? WHERE seq=?",
                               ((json.dumps(value, separators=(",", ":")), row["seq"])
                                for row, value in zip(rows, matrix, strict=True)))
                db.commit()
            else:
                bank.save_embeddings([(row["memory_id"], row["text"], value)
                                      for row, value in zip(rows, matrix, strict=True)], self.model, missing_only=True)
        if stale:
            signal.check()
            self.swap(db)

    def swap(self, db):
        db.execute("BEGIN IMMEDIATE")
        db.execute("DELETE FROM memory_embeddings WHERE model IS NOT ?", (self.model,))
        db.execute("UPDATE episodic_memory SET binary_vector=NULL WHERE binary_vector IS NOT NULL")
        delete_legacy_vectors(db)
        db.execute(f"""
            INSERT OR IGNORE INTO memory_embeddings(memory_id,embedding_json,model)
            SELECT s.memory_id,s.embedding_json,:model FROM embedding_rebuild s
            WHERE s.embedding_json IS NOT NULL AND (
                EXISTS(SELECT 1 FROM working_memory w WHERE w.id=s.memory_id AND coalesce(w.embed_text,w.content)=s.text
                       AND w.superseded_by IS NULL AND (w.valid_until IS NULL OR w.valid_until>:now))
                OR EXISTS(SELECT 1 FROM episodic_memory e WHERE e.id=s.memory_id AND e.content=s.text
                          AND e.superseded_by IS NULL AND (e.valid_until IS NULL OR e.valid_until>:now))
            )
        """, {"now": timestamp(), "model": self.model})
