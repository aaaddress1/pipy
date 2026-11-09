"""Scoped four-tier recall cache with cross-connection SQLite invalidation."""

import sqlite3
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from threading import RLock
from time import monotonic

from mnemopi_polyphonic import VOICES, hydrate, prepare_fact_voice, rank_polyphonic_bank, voice_enabled
from mnemopi_recall import cosine, count_recalls, rank_bank
from session_jsonl import timestamp


@dataclass
class Entry:
    inserted: float
    words: frozenset
    embedding: tuple | None
    results: list


class RecallCache:
    def __init__(self, bank, *, max_size=256, ttl=300):
        self.bank = bank
        self.db = sqlite3.connect(bank.path, check_same_thread=False, timeout=5)
        self.db.row_factory = sqlite3.Row
        self.lock = RLock()
        self.entries = OrderedDict()
        self.revision = self.read_version()
        self.max_size, self.ttl = max(0, int(max_size)), max(0, ttl)
        self.hits = [0] * 4
        self.misses = 0

    def read_version(self):
        return self.db.execute("PRAGMA data_version").fetchone()[0]

    def refresh(self):
        version = self.read_version()
        if version != self.revision:
            self.entries.clear()
            self.revision = version
        return version

    def expire(self, now):
        expired = [key for key, entry in self.entries.items() if now - entry.inserted > self.ttl]
        for key in expired:
            del self.entries[key]

    def match(self, key, query, words, embedding):
        if key in self.entries:
            return key, 0
        scope, _ = key
        eligible = [(cached_key, cached) for cached_key, cached in self.entries.items() if cached_key[0] == scope]
        best, maximum = None, 0.
        raw_words = set(query.lower().split())
        if embedding:
            for cached_key, cached in eligible:
                similarity = cosine(embedding, cached.embedding)
                if similarity >= .88:
                    return cached_key, 1
                if similarity < .78 or similarity <= maximum:
                    continue
                union = raw_words | cached.words
                overlap = len(raw_words & cached.words) / len(union) if union else 0.
                if overlap >= .15:
                    best, maximum = cached_key, similarity
        if best:
            return best, 2
        for cached_key, cached in eligible:
            shared = len(words & cached.words)
            if shared >= 2 and shared >= len(words) * .7 and shared >= len(cached.words) * .5:
                return cached_key, 3
        return None

    def valid_rows(self, rows):
        now = timestamp()
        return [row for row in rows if hydrate(self.db, self.bank.name, now, row["id"], row) is not None]

    def recall(self, query, limit, embedding, model, *, polyphonic=False):
        if polyphonic:
            prepare_fact_voice(self.bank)
        ranking = rank_polyphonic_bank if polyphonic else rank_bank
        scope = (limit, model, polyphonic, tuple(voice_enabled(voice) for voice in VOICES) if polyphonic else ())
        normalized = " ".join(sorted(word.lower() for word in query.split() if len(word) > 1))
        words = frozenset(normalized.split())
        key = scope, normalized
        with self.lock:
            before = self.refresh()
            self.expire(monotonic())
            if hit := self.match(key, query, words, embedding):
                cached_key, tier = hit
                with self.db:
                    self.db.execute("BEGIN IMMEDIATE")
                    if self.refresh() == before:
                        entry = self.entries[cached_key]
                        rows = self.valid_rows(entry.results)
                        if len(rows) == len(entry.results):
                            count_recalls(self.db, self.bank, rows)
                            self.hits[tier] += 1
                            self.entries.move_to_end(cached_key)
                            return deepcopy(rows)
                        self.entries.pop(cached_key, None)
                before = self.refresh()
            self.misses += 1
            rows = ranking(self.bank, query, limit, embedding, model, update_counts=False)
            with self.db:
                self.db.execute("BEGIN IMMEDIATE")
                if self.refresh() == before:
                    self.entries[key] = Entry(monotonic(), words, tuple(embedding) if embedding else None, deepcopy(rows))
                    self.entries.move_to_end(key)
                    while len(self.entries) > self.max_size:
                        self.entries.popitem(last=False)
                else:
                    rows = self.valid_rows(rows)
                count_recalls(self.db, self.bank, rows)
            return rows

    def stats(self):
        with self.lock:
            return {"hits": sum(self.hits), "misses": self.misses,
                    **{f"tier{index}_hits": hits for index, hits in enumerate(self.hits, 1)}, "size": len(self.entries)}

    def close(self):
        with self.lock:
            self.db.close()
