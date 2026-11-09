"""Hindsight bank scoping, transcript memory, and the durable tool retain queue."""

import json
import os
import random
import re
import subprocess
import time
from collections import deque
from concurrent.futures import CancelledError, TimeoutError
from datetime import datetime
from difflib import unified_diff
from hashlib import sha256
from pathlib import Path
from threading import Condition, Event, RLock
from urllib.parse import quote, urlsplit
from uuid import uuid4

import httpx

from auth_broker import dotted
from memory import redact_memory
from memory_files import atomic_text
from memory_transcript import MEMORY_TAGS, NON_CONVERSATIONAL, conversation_messages, conversation_text, substantive, user_turns
from reset_fence import bun_hash, shared_file_lock
from runtime import database
from cancellable_http import active_turn, turn_scope
from handoff import Cancellation
from session_graph import ANCESTORS


CONTROLS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
EVENT = re.compile(r"omp-retain-[0-9a-f]{32}\.json\Z")
DAY = 86_400
REFUSED = {400, 413, 422}
RECALL_PREAMBLE = ("Relevant memories from past conversations (prioritize recent when conflicting). "
                   "Only use memories that are directly useful to continue this conversation; ignore the rest:")


class HindsightError(ValueError):
    def __init__(self, operation, status=None):
        self.status = status
        super().__init__(f"Hindsight {operation} failed" + (f" (HTTP {status})." if status else "."))


def config_value(settings, key, default, env=None, choices=None):
    override = os.getenv("HINDSIGHT_" + env) if env else None
    value = dotted(settings, "hindsight." + key)
    if override:
        if type(default) is bool:
            truthy = {"1", "y", "true", "yes", "on", "Y", "TRUE", "YES", "ON"}
            value = override in truthy
        elif type(default) is int:
            if match := re.match(r"^\s*[+-]?\d+", override):
                value = int(match[0])
        elif not choices or override in choices:
            value = override
    if type(value) is not type(default) or (choices and value not in choices):
        return default
    return value


def project_label(workspace):
    try:
        git = subprocess.run(["git", "-C", str(workspace), "rev-parse", "--path-format=absolute", "--git-common-dir"],
                             capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3)
        if git.returncode == 0 and git.stdout.strip():
            common = Path(git.stdout.strip())
            workspace = common.parent if common.name == ".git" else common
    except (OSError, subprocess.TimeoutExpired):
        pass
    return workspace.name.lower() or "unknown"


def retain_time():
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


VISIBLE_ROW = " AND ".join(f"json_type(m.value, '$.{key}') IS NULL" for key in NON_CONVERSATIONAL)


def conversation_turn_batches(db, session, start):
    rows = db.execute(ANCESTORS + """
        SELECT n.id,n.batch,json_extract(n.payload,'$.memoryUserTurns'),
               CASE WHEN json_type(n.payload,'$.memoryUserTurns') IS NULL THEN b.messages END
        FROM ancestry a JOIN session_nodes n ON n.id=a.id
        JOIN batches b ON b.session=n.session AND b.seq=n.batch
        WHERE n.kind='batch' AND n.batch>=? ORDER BY n.batch
    """, (session.leaf, session.id, session.id, start))
    older, counts = [], []
    for node, seq, count, raw in rows:
        if count is None:
            messages = json.loads(raw)
            count = user_turns(messages)
            older.append((count, any("_memory_recall_changes" in message for message in messages), session.id, node))
        counts.append((seq, count))
    if older:
        db.executemany("UPDATE session_nodes SET payload=json_set(coalesce(payload,'{}'),'$.memoryUserTurns',?,"
                       "'$.memoryRecallChanges',?) "
                       "WHERE session=? AND id=?", older)
    return counts


def conversation_turns(session, *, after_reset=False):
    with session.writer, database(session.path) as db:
        start = session.transcript_start if after_reset else 0
        return sum(count for _, count in conversation_turn_batches(db, session, start))


def conversation(session, *, after_reset=False, after_turn=0, last_turns=None):
    if last_turns is not None and last_turns <= 0:
        return []
    with session.writer, database(session.path) as db:
        start = session.transcript_start if after_reset else 0
        counts = conversation_turn_batches(db, session, start) if after_turn or last_turns is not None else ()
        if last_turns is not None:
            after_turn = max(after_turn, sum(count for _, count in counts) - last_turns)
        remaining = after_turn
        if after_turn:
            for seq, count in counts:
                if count > remaining:
                    start = seq
                    break
                remaining -= count
            else:
                return []
        rows = db.execute(ANCESTORS + f"""
            SELECT json_extract(m.value,'$.role'), json_extract(m.value,'$.content'), json_type(m.value,'$.content'),
                   coalesce(json_extract(source.metadata,'$.timestamp'),json_extract(n.payload,'$.timestamp'))
            FROM ancestry a JOIN session_nodes n ON n.id=a.id
            JOIN batches b ON b.session=n.session AND b.seq=n.batch JOIN json_each(b.messages) m
            LEFT JOIN session_entry_sources source ON source.node=n.id AND source.session=n.session
            WHERE n.kind='batch' AND n.batch>=? AND json_extract(m.value,'$.role') IN ('user','assistant')
            AND {VISIBLE_ROW} ORDER BY n.batch,CAST(m.key AS INTEGER)
        """, (session.leaf, session.id, session.id, start)).fetchall()
    entries = []
    waiting_for_user = bool(after_turn)
    for role, content, kind, stamp in rows:
        text = conversation_text(json.loads(content) if kind == "array" else content)
        if not substantive(text):
            continue
        if waiting_for_user:
            if role != "user":
                continue
            if remaining:
                remaining -= 1
                continue
            waiting_for_user = False
        entries.append((role, text, stamp))
    return entries


def retention_prefix(digest, messages):
    for message in messages:
        for value in message:
            data = (value or "").encode("utf-8", errors="replace")
            digest.update(len(data).to_bytes(8, "big"))
            digest.update(data)
    return digest


def recall_query(history, latest, turns, limit):
    latest = latest.strip()
    if turns <= 1:
        return latest[:limit] if limit > 0 else latest
    boundaries = [index for index, (role, _, _) in enumerate(history) if role == "user"]
    prior = turns - 1
    history = history[boundaries[-prior] if len(boundaries) >= prior else 0:]
    entries = []
    for role, content, _ in history:
        content = MEMORY_TAGS.sub("", content).strip()
        if any(char.isalnum() for char in content):
            safe = re.sub(r"\n(?=(?:user|assistant|system|tool): )", "\n  ", content)
            entries.append(f"{role}: {safe}")
    while entries:
        query = "Prior context:\n\n" + "\n".join(entries) + "\n\n" + latest
        if limit <= 0 or len(query) <= limit:
            return query
        entries.pop(0)
    return latest[:limit] if limit > 0 else latest


def persisted_recall(session, scope):
    with database(session.path) as db:
        rows = db.execute(ANCESTORS + """
            SELECT n.kind,n.payload,
                CASE WHEN json_type(n.payload,'$.memoryUserTurns') IS NULL THEN
                    CASE WHEN b.messages LIKE '%"_memory_recall_changes"%' THEN b.messages END
                WHEN json_extract(n.payload,'$.memoryRecallChanges')=1 THEN b.messages END
            FROM ancestry a JOIN session_nodes n ON n.id=a.id LEFT JOIN batches b ON b.seq=n.batch AND b.session=n.session
            ORDER BY n.id DESC
        """, (session.leaf, session.id, session.id))
        turn_followed = False
        reported = None
        for kind, payload, messages in rows:
            if kind == "batch":
                turn_followed = True
                if messages and reported is None:
                    for message in reversed(json.loads(messages)):
                        changes = message.get("_memory_recall_changes")
                        if (isinstance(changes, dict) and changes.get("scope") == scope
                                and isinstance(changes.get("memories"), list)):
                            reported = changes["memories"]
                            break
            if kind == "reset":
                break
            if kind != "foreign":
                continue
            entry = json.loads(payload)
            name, data = entry.get("customType"), entry.get("data")
            if entry.get("type") == "reset_boundary" or name == "memory_recall_clear" and data == scope:
                break
            if not turn_followed or not isinstance(data, dict) or data.get("scope") != scope:
                continue
            if name == "memory_recall_report" and reported is None and isinstance(data.get("memories"), list):
                reported = data["memories"]
            if name == "memory_recall" and isinstance(data.get("text"), str):
                return data if reported is None else {**data, "memories": reported}
    return None


class Hindsight:
    def __init__(self, settings, workspace, agent_dir, session, executor, notice, *, primary=True, client=None):
        get = lambda key, default, env=None, choices=None: config_value(settings, key, default, env, choices)
        self.url = get("apiUrl", "http://localhost:8888", "API_URL").strip().rstrip("/")
        self.token = get("apiToken", "", "API_TOKEN")
        url = urlsplit(self.url)
        if self.url and (url.scheme not in ("http", "https") or not url.hostname or url.username is not None
                         or url.query or url.fragment):
            raise ValueError("hindsight.apiUrl must be an HTTP(S) server URL without credentials, query, or fragment.")
        mode = get("scoping", "per-project-tagged", "SCOPING", {"global", "per-project", "per-project-tagged"})
        base = get("bankId", "", "BANK_ID").strip() or "omp"
        prefix = get("bankIdPrefix", "").strip()
        project = project_label(workspace) if mode != "global" else ""
        self.bank = "-".join(part for part in (prefix, base, project if mode == "per-project" else "") if part)
        self.tags = ["project:" + project] if mode == "per-project-tagged" else []
        self.mode = mode
        self.global_scope = get("globalScope", False) and mode != "per-project"
        self.missions = {field: mission for field, mission in (
            ("reflect_mission", get("bankMission", "", "BANK_MISSION").strip()),
            ("retain_mission", get("retainMission", "").strip()),
        ) if mission}
        self.auto_recall = primary and get("autoRecall", True, "AUTO_RECALL")
        self.auto_retain = primary and get("autoRetain", True, "AUTO_RETAIN")
        self.retain_mode = get("retainMode", "full-session", "RETAIN_MODE", {"full-session", "last-turn"})
        self.every = max(1, get("retainEveryNTurns", 3, "RETAIN_EVERY_N_TURNS"))
        self.overlap = max(0, get("retainOverlapTurns", 2))
        self.retain_context = get("retainContext", "omp")
        self.budget = get("recallBudget", "mid", "RECALL_BUDGET", {"low", "mid", "high"})
        self.recall_tokens = max(1, get("recallMaxTokens", 1024, "RECALL_MAX_TOKENS"))
        self.recall_turns = max(0, get("recallContextTurns", 1, "RECALL_CONTEXT_TURNS"))
        self.query_limit = get("recallMaxQueryChars", 800, "RECALL_MAX_QUERY_CHARS")
        self.types = get("recallTypes", ["world", "experience"])
        self.timeouts = {operation: max(1, get(operation + "TimeoutMs", default, operation.upper() + "_TIMEOUT_MS")) / 1000
                         for operation, default in (("request", 30_000), ("recall", 30_000),
                                                    ("retain", 60_000), ("reflect", 120_000))}
        self.mental_enabled = primary and get("mentalModelsEnabled", True)
        self.mental_seed = get("mentalModelAutoSeed", True)
        self.mental_budget = get("mentalModelMaxRenderChars", 16_000)
        self.client = client or httpx.Client(trust_env=False)
        self.own_client = client is None
        self.session, self.executor, self.notice = session, executor, notice
        self.condition, self.stopping = Condition(), Event()
        self.ensured = False
        self.recall = None
        self.recall_epoch = session.navigation_epoch
        self.mental = ""
        self.last_retained = 0
        self.last_transcript = None
        self.retained_transcript_id = None
        self.retained_prefix = None
        self.retention_queue = deque()
        self.last_window_ms = 0
        self.drain_job = self.retain_job = self.mental_job = None
        self.drain_at = None
        self.bank_lock = RLock()
        self.mental_signal = Cancellation(reason="Hindsight mental-model load cancelled.")
        self.root = agent_dir / "hindsight-outbox"
        normalized = str(httpx.URL(self.url)).rstrip("/") if self.url else ""
        destination = sha256((normalized + "\0" + self.bank).encode()).hexdigest()[:16]
        self.outbox = self.root / destination
        self.scope = json.dumps(["hindsight", self.url, format(bun_hash(self.token), "x"),
                                 self.bank, self.tags, "any" if self.tags else None], ensure_ascii=False, separators=(",", ":"))
        with database(session.path) as db:
            created = db.execute("SELECT json_extract(header, '$.timestamp') FROM session_imports WHERE session=?",
                                 (session.id,)).fetchone()
            if not created or not isinstance(created[0], str):
                created = db.execute(ANCESTORS + """
                    SELECT json_extract(n.payload, '$.timestamp')
                    FROM ancestry a JOIN session_nodes n ON n.id=a.id
                    WHERE n.kind='batch' AND json_type(n.payload, '$.timestamp')='text'
                    ORDER BY n.id LIMIT 1
                """, (session.leaf, session.id, session.id)).fetchone()
        self.start_time = created[0] if created and isinstance(created[0], str) else retain_time()
        if self.url and primary:
            self.schedule_drain()
            if self.mental_enabled:
                self.mental_job = executor.submit(self.start_mental, self.mental_signal)

    def clean(self, text):
        if self.token:
            text = text.replace(self.token, "[REDACTED]")
        return redact_memory(CONTROLS.sub("", text))

    def request(self, method, path="", *, operation="request", body=None, params=None, allow_missing=False):
        headers = {"User-Agent": "omp-hindsight", "Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        try:
            response = self.client.request(method, self.url + "/v1/default/banks/" + quote(self.bank, safe="") + path,
                                           headers=headers, json=body, params=params, timeout=self.timeouts[operation])
        except httpx.RequestError:
            raise HindsightError(operation) from None
        if response.status_code == 404 and allow_missing:
            return None
        if not response.is_success:
            raise HindsightError(operation, response.status_code)
        try:
            result = response.json() if response.content else {}
            if not isinstance(result, dict) and not (path.endswith("/history") and isinstance(result, list)):
                raise ValueError("Expected a JSON object or list.")
            return result
        except ValueError:
            raise HindsightError(operation) from None

    def ensure_bank(self):
        with self.bank_lock:
            if not self.ensured:
                try:
                    self.request("PUT", body=self.missions)
                    self.ensured = True
                except HindsightError:
                    pass

    def lookup(self, query, *, reflect=False, context=None):
        if not isinstance(query, str) or not query.strip() or context is not None and not isinstance(context, str):
            raise ValueError("Provide a nonempty query and optional text context.")
        body = {"query": self.clean(MEMORY_TAGS.sub("", query)).strip(), "budget": "low" if reflect else self.budget}
        if self.tags:
            body.update(tags=self.tags, tags_match="any")
        if reflect:
            self.ensure_bank()
            if context:
                body["context"] = self.clean(MEMORY_TAGS.sub("", context))
            response = self.request("POST", "/reflect", operation="reflect", body=body)
            text = response.get("text")
            return self.clean(text).strip() if isinstance(text, str) and text.strip() else "No relevant information found to reflect on."
        body["max_tokens"] = self.recall_tokens
        if self.types:
            body["types"] = self.types
        response = self.request("POST", "/memories/recall", operation="recall", body=body)
        results = response.get("results") or []
        if not isinstance(results, list):
            raise HindsightError("recall")
        rows = [row for row in results if isinstance(row, dict) and isinstance(row.get("text"), str)]
        return rows, "\n\n".join(self.clean(
            "- " + row["text"] + (f" [{row['type']}]" if row.get("type") else "")
            + (f" ({row['mentioned_at']})" if row.get("mentioned_at") else "")
        ) for row in rows)

    def context(self, history, latest):
        if not latest.strip():
            return ""
        try:
            _, text = self.lookup(recall_query(history, latest, self.recall_turns, self.query_limit))
        except HindsightError as error:
            self.notice(str(error))
            return None
        return f"<memories>\n{RECALL_PREAMBLE}\n\n{text}\n</memories>" if text else ""

    def prepare(self, latest):
        if self.recall_epoch != self.session.navigation_epoch:
            self.recall, self.recall_epoch = None, self.session.navigation_epoch
        if not self.auto_recall or self.recall is not None:
            return
        if (entry := persisted_recall(self.session, self.scope)) is not None:
            self.recall = self.clean(entry["text"])
            return
        if not latest.strip():
            return
        history = conversation(self.session, after_reset=True, last_turns=self.recall_turns - 1)
        result = self.context(history, latest)
        if signal := active_turn():
            signal.check()
        if result is not None:
            return {"type": "custom", "customType": "memory_recall",
                    "data": {"version": 1, "scope": self.scope, "text": result}}

    def commit(self, entry):
        self.recall = entry["data"]["text"]

    def reset_context(self):
        self.recall = None
        self.mental = ""
        self.mental_signal.cancel()
        self.mental_job = None
        if self.url and self.mental_enabled:
            self.mental_signal = Cancellation(reason="Hindsight mental-model load cancelled.")
            self.mental_job = self.executor.submit(self.start_mental, self.mental_signal, False)

    def compaction_context(self, messages):
        for index in range(len(messages) - 1, -1, -1):
            if messages[index][0] == "user" and messages[index][1].strip():
                return self.context(messages[:index], messages[index][1])
        return ""

    def command(self, argument):
        verb, _, rest = argument.strip().partition(" ")
        verb = verb.lower() or "view"
        if verb == "view":
            self.settle_mental(wait=True)
            return self.instructions() or "Memory payload is empty."
        if verb in ("stats", "queue", "diagnose"):
            queued = sum(1 for file in self.outbox.iterdir() if EVENT.fullmatch(file.name)) if self.outbox.is_dir() else 0
            return f"Hindsight bank {self.bank} ({self.mode}). Queued memories: {queued}."
        if verb in ("clear", "reset"):
            self.flush()
            self.recall, self.mental = "", ""
            self.mental_signal.cancel()
            self.mental_job = None
            self.session.record_metadata({"type": "custom", "customType": "memory_recall_clear", "data": self.scope})
            return "Cleared the local Hindsight context. The remote memory bank is managed on the server."
        if verb in ("sync", "enqueue", "rebuild"):
            self.flush()
            with self.session.writer:
                history = conversation(self.session, after_reset=True)
                transcript_id = self.session.transcript_id
            job = self.schedule_retention(history, transcript_id, force=True)
            if verb == "sync":
                job.result()
                return "Hindsight retention ran."
            return "Hindsight retention enqueued."
        if verb == "mm":
            return self.mental_command(rest)
        raise ValueError("Use /memory view, stats, diagnose, queue, sync, enqueue, rebuild, clear, or mm.")

    def flush(self):
        try:
            if self.drain():
                self.schedule_drain()
        except (OSError, HindsightError):
            self.schedule_drain()

    def mental_command(self, argument):
        verb, _, identifier = argument.strip().partition(" ")
        identifier = identifier.strip()
        verb = verb.lower() or "list"
        if verb == "reload":
            model = self.load_mental(False)
            if model is not None:
                self.mental = model
            self.mental_job = None
            return "Reloaded the Hindsight mental-model context." if model is not None else "Hindsight mental-model reload failed."
        if verb == "seed":
            self.load_mental(True)
            return "Checked and seeded missing Hindsight mental models."
        if verb == "list":
            rows = self.mental_models("metadata")
            return "\n".join(f"{row.get('id', '')}  {row.get('name', '')}" for row in rows) or "No mental models."
        if verb == "refresh" and not identifier:
            rows = self.mental_models("metadata")
            targets = [row["id"] for row in rows if row.get("id") and (row.get("trigger") or {}).get("refresh_after_consolidation")]
            for name in targets:
                self.mental_command("refresh " + name)
            return f"Queued refreshes for {len(targets)} mental models."
        if not identifier or verb not in ("show", "history", "refresh", "delete"):
            raise ValueError("Use /memory mm list, show <id>, history <id>, refresh [id], delete <id>, seed, or reload.")
        path = "/mental-models/" + quote(identifier, safe="")
        if verb == "show":
            model = self.request("GET", path, params={"detail": "content"}, allow_missing=True)
            return self.clean(model.get("content") or "No content yet.") if model else "Mental model not found."
        if verb == "delete":
            response = self.request("DELETE", path, allow_missing=True)
            return "Deleted mental model." if response is not None else "Mental model not found."
        if verb == "refresh":
            self.request("POST", path + "/refresh")
            return "Queued mental-model refresh."
        history = self.request("GET", path + "/history")
        if isinstance(history, dict):
            history = history.get("items")
        if not history or not isinstance(history, list):
            return "No mental-model history."
        model = self.request("GET", path, params={"detail": "content"}, allow_missing=True) or {}
        current = self.clean(model.get("content") or "")
        changes = []
        for entry in history:
            previous = self.clean(entry.get("previous_content") or "")
            changes.append("\n".join(unified_diff(previous.splitlines()[:1000], current.splitlines()[:1000],
                                                 "previous", "current", lineterm=""))[:16_000])
            current = previous
        return "\n\n".join(change for change in changes if change) or "No content changes."

    def publish(self, items, call_id=None):
        if not isinstance(items, list) or not items or len(items) > 1000:
            raise ValueError("Provide between 1 and 1000 memories.")
        call_id = call_id or uuid4().hex
        events = []
        for index, item in enumerate(items):
            if not isinstance(item, dict) or set(item) - {"content", "context", "scope"}:
                raise ValueError("A memory accepts content, context, and scope.")
            content, context = item.get("content"), item.get("context", self.retain_context)
            scope = item.get("scope", "project")
            if not isinstance(content, str) or not isinstance(context, str):
                raise ValueError("Memory content and context must be text.")
            if scope != "project" and not (scope == "global" and self.global_scope):
                raise ValueError("Global retention requires hindsight.globalScope and a shared bank.")
            content, context = (self.clean(MEMORY_TAGS.sub("", value)).strip() for value in (content, context))
            if not content:
                raise ValueError("Memory content is empty.")
            identity = json.dumps([self.session.id, call_id, index, scope, context, content],
                                  ensure_ascii=False, separators=(",", ":"))
            identifier = "omp-retain-" + sha256(identity.encode()).hexdigest()[:32]
            events.append({"version": 1, "id": identifier, "scope": scope,
                           "tags": [self.clean(tag) for tag in self.tags] if scope == "project" else [],
                           "timestamp": retain_time(), "content": content, "context": context,
                           "sessionId": self.clean(self.session.id)})
        self.outbox.mkdir(mode=0o700, parents=True, exist_ok=True)
        with shared_file_lock(self.outbox / "queue.lock", timeout=5):
            existing = {file.name for file in self.outbox.iterdir() if EVENT.fullmatch(file.name)}
            fresh = [event for event in events if event["id"] + ".json" not in existing]
            if len(existing) + len(fresh) > 1000:
                raise ValueError("Hindsight outbox is full. Memory was not queued.")
            written = []
            try:
                for event in fresh:
                    file = self.outbox / (event["id"] + ".json")
                    atomic_text(file, json.dumps(event, ensure_ascii=False, separators=(",", ":")), durable=True)
                    written.append(file)
                if written and os.name == "posix":
                    directory = os.open(self.outbox, os.O_RDONLY)
                    try:
                        os.fsync(directory)
                    finally:
                        os.close(directory)
            except OSError:
                for file in written:
                    file.unlink(missing_ok=True)
                raise
        self.schedule_drain(0 if len(existing) + len(fresh) >= 16 else 5)
        return len(events)

    def schedule_drain(self, delay=0):
        with self.condition:
            deadline = time.monotonic() + delay
            self.drain_at = deadline if self.drain_at is None else min(deadline, self.drain_at)
            if self.drain_job is None:
                self.drain_job = self.executor.submit(self.retry_drain)
            self.condition.notify()

    def retry_drain(self):
        failures = 0
        while True:
            with self.condition:
                if self.drain_at is None:
                    self.drain_job = None
                    return
                delay = self.drain_at - time.monotonic()
                if delay > 0 and not self.stopping.is_set():
                    self.condition.wait(delay)
                    continue
                self.drain_at = None
            try:
                busy = self.drain()
                failures = 0
            except (OSError, HindsightError) as error:
                busy = False
                failures += 1
                if failures == 1:
                    self.notice(f"Hindsight outbox will retry queued memories ({error}).")
            if not self.stopping.is_set() and (busy or failures):
                delay = 5 if busy else min(1800, 5 * 2 ** min(10, failures - 1)) * random.uniform(.8, 1.2)
                self.schedule_drain(delay)

    def quarantine(self, file):
        target = self.root / "quarantine"
        target.mkdir(mode=0o700, exist_ok=True)
        file.replace(target / file.name)
        os.utime(target / file.name, None)
        self.notice(f"Hindsight moved {file.name} to quarantine.")

    def drain(self):
        if not self.root.is_dir():
            return False
        now = time.time()
        retry = False
        for destination in self.root.iterdir():
            if not destination.is_dir() or destination.is_symlink():
                continue
            if destination.name == "quarantine":
                for file in destination.iterdir():
                    if EVENT.fullmatch(file.name) and file.stat().st_mtime < now - 30 * DAY:
                        file.unlink(missing_ok=True)
                continue
            try:
                with shared_file_lock(destination / "drain.lock"):
                    with shared_file_lock(destination / "queue.lock"):
                        pending = []
                        for file in destination.iterdir():
                            if not EVENT.fullmatch(file.name):
                                continue
                            timestamp = file.stat().st_mtime
                            if timestamp < now - 7 * DAY:
                                self.quarantine(file)
                            elif destination == self.outbox:
                                event = self.read_event(file)
                                if event:
                                    pending.append((timestamp, file, event))
                                else:
                                    self.quarantine(file)
                    if destination == self.outbox and pending:
                        self.ensure_bank()
                        groups = {}
                        for _, file, event in sorted(pending):
                            groups.setdefault(tuple(sorted(event["tags"])), []).append((file, event))
                        for batch in groups.values():
                            for index in range(0, len(batch), 16):
                                self.deliver(batch[index:index + 16])
            except TimeoutError:
                retry |= destination == self.outbox
        return retry

    def read_event(self, file):
        try:
            if file.is_symlink():
                return None
            event = json.loads(file.read_text(encoding="utf-8"))
            if (event["version"] == 1 and event["id"] + ".json" == file.name
                    and event["scope"] in ("project", "global") and isinstance(event["tags"], list)
                    and all(isinstance(tag, str) for tag in event["tags"])
                    and all(isinstance(event[key], str) for key in ("content", "context", "sessionId", "timestamp"))):
                datetime.fromisoformat(event["timestamp"])
                return event
        except (ValueError, KeyError, TypeError):
            pass
        return None

    def deliver(self, batch):
        items = []
        for _, event in batch:
            item = {"content": self.clean(event["content"]), "context": self.clean(event["context"]),
                    "timestamp": event["timestamp"], "metadata": {"session_id": self.clean(event["sessionId"])},
                    "document_id": event["id"], "update_mode": "replace"}
            if event["scope"] == "project" and event["tags"]:
                item["tags"] = [self.clean(tag) for tag in event["tags"]]
            items.append(item)
        try:
            self.request("POST", "/memories", operation="retain", body={"items": items, "async": True})
        except HindsightError as error:
            if error.status not in REFUSED:
                raise
            if len(batch) == 1:
                self.quarantine(batch[0][0])
            else:
                for event in batch:
                    self.deliver([event])
        else:
            for file, _ in batch:
                file.unlink(missing_ok=True)

    def retain(self, messages, *, transcript_id, force=False):
        count = sum(role == "user" for role, _, _ in messages)
        same_transcript = transcript_id == self.retained_transcript_id
        retained = self.last_retained if same_transcript else 0
        if not count or not force and count - retained < self.every:
            return
        document = transcript_id
        started = self.start_time if transcript_id == self.session.id else next(
            (stamp for _, _, stamp in messages if stamp), None) or retain_time()
        if self.retain_mode == "last-turn":
            boundaries = [index for index, (role, _, _) in enumerate(messages) if role == "user"]
            window = self.every + self.overlap
            messages = messages[boundaries[-window] if len(boundaries) >= window else 0:]
            self.last_window_ms = max(self.last_window_ms + 1, int(time.time() * 1000))
            document += "-" + str(self.last_window_ms)
        full = self.retain_mode == "full-session"
        digest = sha256() if full else None
        rendered, offset = None, 0
        if full and not force and same_transcript and self.retained_prefix:
            previous_count, previous_key = self.retained_prefix
            if previous_count <= len(messages) and (
                retention_prefix(digest, messages[:previous_count]).digest() == previous_key
            ):
                rendered, offset = self.last_transcript, previous_count
            else:
                digest = sha256()
        sections = []
        for role, text, timestamp in messages[offset:]:
            text = self.clean(MEMORY_TAGS.sub("", text)).strip()
            if any(char.isalnum() for char in text):
                stamp = f"[timestamp: {timestamp}]\n" if timestamp else ""
                sections.append(f"[role: {role}]\n{stamp}{text}\n[{role}:end]")
        transcript = "\n\n".join([rendered, *sections] if rendered else sections)
        if not transcript or not force and same_transcript and self.retain_mode == "full-session" and transcript == self.last_transcript:
            return
        item = {"content": transcript, "document_id": document, "context": self.clean(self.retain_context),
                "metadata": {"session_id": self.clean(transcript_id)}, "timestamp": started}
        if self.tags:
            item["tags"] = self.tags
        try:
            self.ensure_bank()
            self.request("POST", "/memories", operation="retain", body={"items": [item], "async": True})
        except HindsightError as error:
            self.notice(str(error))
        else:
            prefix = (len(messages), retention_prefix(digest, messages[offset:]).digest()) if full else None
            with self.condition:
                self.last_retained, self.last_transcript = count, transcript
                self.retained_transcript_id, self.retained_prefix = transcript_id, prefix

    def schedule_retention(self, messages, transcript_id, *, force=False):
        with self.condition:
            record = messages, transcript_id, force
            if self.retain_mode == "full-session" and self.retention_queue and not force and (
                self.retention_queue[-1][1:] == (transcript_id, False)
            ):
                self.retention_queue[-1] = record
            else:
                self.retention_queue.append(record)
            if self.retain_job is None or self.retain_job.done():
                self.retain_job = self.executor.submit(self.drain_retention)
            return self.retain_job

    def drain_retention(self):
        while True:
            with self.condition:
                if not self.retention_queue:
                    self.retain_job = None
                    return
                messages, transcript_id, force = self.retention_queue.popleft()
            self.retain(messages, transcript_id=transcript_id, force=force)

    def end_turn(self):
        if self.auto_retain:
            with self.session.writer:
                transcript_id = self.session.transcript_id
                with self.condition:
                    retained = self.last_retained if transcript_id == self.retained_transcript_id else 0
                if conversation_turns(self.session, after_reset=True) - retained >= self.every:
                    history = conversation(self.session, after_reset=True)
                    self.schedule_retention(history, transcript_id)
        self.schedule_drain()

    def mental_models(self, detail="content"):
        items = []
        while True:
            page = self.request("GET", "/mental-models", params={"detail": detail, "limit": 100, "offset": len(items)})
            rows = page.get("items") or []
            if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
                raise HindsightError("mental-model listing")
            items.extend(rows)
            total = page.get("total")
            if len(rows) < 100 or isinstance(total, int) and len(items) >= total:
                return items

    def start_mental(self, signal, seed=None):
        try:
            with turn_scope(signal):
                signal.check()
                return self.load_mental(seed)
        except CancelledError:
            return ""

    def load_mental(self, seed=None):
        try:
            self.ensure_bank()
            should_seed = self.mental_seed if seed is None else seed
            if should_seed:
                seeds = json.loads(Path(__file__).with_name("prompts").joinpath("hindsight-seeds.json").read_text("utf-8"))
                existing = self.mental_models("metadata")
                for seed in seeds["seeds"]:
                    if self.stopping.is_set():
                        return ""
                    if self.mode not in seed["scopes"]:
                        continue
                    tags = self.tags if seed["projectTagged"] else []
                    name = seed["id"]
                    if tags:
                        name += "-" + (re.sub(r"[^A-Za-z0-9._-]+", "-", tags[0][8:]).strip("-") or "project")
                    if any(row.get("id") in (name, seed["id"]) and sorted(row.get("tags") or []) == sorted(tags)
                           for row in existing):
                        continue
                    body = {key: seed[key] for key in ("name", "source_query", "max_tokens", "trigger")}
                    body["id"] = name
                    if tags:
                        body["tags"] = tags
                    try:
                        self.request("POST", "/mental-models", body=body)
                    except HindsightError:
                        continue
            models = sorted(self.mental_models(), key=lambda row: (row.get("name", ""), row.get("id", "")))
        except (HindsightError, OSError) as error:
            self.notice(f"Hindsight mental-model load failed ({error}).")
            return None
        sections = [(self.clean(row["name"]), self.clean(row["content"]).strip())
                    for row in models if isinstance(row.get("content"), str) and row["content"].strip()
                    and isinstance(row.get("name"), str)
                    and (not self.tags or not row.get("tags") or any(tag in self.tags for tag in row["tags"]))]
        wrapper = "<mental_models>\n{}\n</mental_models>"
        available = self.mental_budget - len(wrapper.format(""))
        if not sections or available < 64:
            return ""
        body = "\n\n".join(f"# {title}\n{content}" for title, content in sections)
        if len(body) > available:
            marker = "\n\n…[mental-model snapshot truncated at render budget]"
            share = max(1, (available - len(marker) - sum(len(title) + 5 for title, _ in sections)) // len(sections))
            body = "\n\n".join(f"# {title}\n{content[:share]}" for title, content in sections)
            body = body[:available - len(marker)].rstrip() + marker
        return wrapper.format(body)

    def settle_mental(self, *, wait=False):
        if self.mental_job and (wait or self.mental_job.done()):
            try:
                loaded = self.mental_job.result(timeout=1.5 if wait else None)
            except TimeoutError:
                return
            if loaded is not None:
                self.mental = loaded
            self.mental_job = None

    def instructions(self):
        return "\n\n".join(part for part in (self.mental, self.recall) if part)

    def close(self):
        if not self.url:
            if self.own_client:
                self.client.close()
            return
        self.stopping.set()
        self.mental_signal.cancel()
        self.schedule_drain()
        try:
            for job in (self.retain_job, self.drain_job, self.mental_job):
                if job:
                    job.result()
        finally:
            if self.own_client:
                self.client.close()
