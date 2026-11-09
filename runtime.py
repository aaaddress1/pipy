"""Persistent sessions and speculative context maintenance."""

import json
import sys
import time
from concurrent.futures import CancelledError, Future
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from functools import partial, wraps
from pathlib import Path
from sqlite3 import Error as SQLiteError, connect
from threading import RLock
from uuid import uuid4

from openai import APIStatusError

import handoff
import session_exit as exit_log
import session_graph as graph
from context_tokens import ContextCounter
from service_runtime import SCHEMA as SERVICE_SCHEMA
import snapcompact
from cancellable_http import active_turn, turn_scope
from compaction_policy import Budget
from context_windows import BOUNDARY, NOTE_LIMIT, NOTE_TYPE, REMINDER, ROLLOVER, note_message, render_message, render_metadata
from model_catalog import ANTHROPIC_COMPACTION_APIS, RESPONSES_APIS
from recovery import StreamInterrupted, TRANSPORT_ERRORS
from shake import AUTO, RESCUE, artifact_id, plan
from session_artifacts import read_numbered
from tool_pruning import IDLE_FLUSH_SECONDS, WARM_SUFFIX, prune_plan, without_useless


def pack(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def tokens(value):
    return (len(pack(value).encode("utf-8")) + 3) // 4


def responses_replay_scope(route):
    return [route.api, route.provider(), route.endpoint(), route.auth, route.upstream]


def responses_replacement_tokens(items, route=None):
    return ContextCounter(route).responses(items)


def native_provider(route):
    return route.broker_provider or (route.catalog_model or {}).get("provider") or {
        "https://api.openai.com/v1": "openai", "https://chatgpt.com/backend-api": "openai-codex",
        "https://api.anthropic.com": "anthropic", "https://api.anthropic.com/v1": "anthropic",
    }.get(route.endpoint(), route.provider())


def responses_replay_matches(native, route):
    if isinstance(route, str):
        return native.get("route") == route
    if route.api not in RESPONSES_APIS:
        return False
    if native.get("route") == route.identity() or native.get("scope") == responses_replay_scope(route):
        return True
    return native.get("provider") == native_provider(route)


def paused_turn(message):
    details = message.get("_stop_details")
    return message["role"] == "assistant" and isinstance(details, dict) and details.get("type") == "pause_turn"


def portable_message(message, thinking_route=None):
    private = ("_anthropic", "_anthropic_import", "_history_at", "_journal_at", "_history_text_parts", "_responses",
               "_thinking", "_branch_from_node", "_rpc", "_tool_details", "_tool_error",
               "_mcp_notification", "_input_transformations", "_chat_thinking_route", "_irc_messages", "_service_runs",
               "_ttsr_rules", "_ttsr_kind", "_ttsr_aborted", "_tool_stop", "_stop_details", "_pause_exhausted", "_dsml_leak",
               "_thinking_loop_redirect", "_tool_pruned_at", "_tool_useless", "_session_transfer", "_goal_message",
               "_todo_message", "_todo_snapshot", "_todo_entry_id", "_parent_assignment", "_compaction_continue",
               "_tool_pending")
    if thinking_route and message.get("_chat_thinking_route") == thinking_route and message.get("_thinking"):
        return {**{key: value for key, value in message.items() if key not in private},
                "reasoning_content": message["_thinking"]}
    if any(key in message for key in private):
        return {key: value for key, value in message.items() if key not in private}
    return message


def developer_content(message):
    content = message["content"]
    if message.get("_branch_summary"):
        return graph.SUMMARY_CONTEXT.replace("{{summary}}", content)
    if any(message.get(key) for key in (
        "_async_jobs", "_irc_messages", "_service_runs", "_ttsr_rules", "_memory_recall_changes",
        "_thinking_loop_redirect", "_session_transfer", "_goal_message", "_todo_message", "_compaction_continue",
    )):
        return content
    if isinstance(content, list):
        return [{"type": "text", "text": "[Agent recovery]"}, *content]
    return "[Agent recovery]\n" + content


@contextmanager
def database(path):
    db = connect(path, timeout=15)
    try:
        with db:
            yield db
    finally:
        db.close()


def initialize(path):
    with database(path) as db:
        db.execute("PRAGMA journal_mode = WAL")
        db.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY, workspace TEXT NOT NULL, updated REAL NOT NULL,
                first_kept INTEGER NOT NULL DEFAULT 0, summary TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS batches (
                seq INTEGER PRIMARY KEY, session TEXT NOT NULL, messages TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS batches_session ON batches(session, seq);
            CREATE TABLE IF NOT EXISTS artifacts (
                session TEXT NOT NULL, id TEXT NOT NULL, content TEXT NOT NULL,
                PRIMARY KEY (session, id)
            );
            CREATE TABLE IF NOT EXISTS context_elisions (
                session TEXT NOT NULL, seq INTEGER NOT NULL, message_index INTEGER NOT NULL, content TEXT NOT NULL,
                PRIMARY KEY (session, seq, message_index)
            );
            CREATE TABLE IF NOT EXISTS native_compactions (
                session TEXT PRIMARY KEY, payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS model_switches (
                seq INTEGER PRIMARY KEY, session TEXT NOT NULL, primary_route TEXT NOT NULL,
                active_route TEXT NOT NULL, reason TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS model_switches_session ON model_switches(session, seq);
            CREATE TABLE IF NOT EXISTS session_recaps (
                seq INTEGER PRIMARY KEY, session TEXT NOT NULL, created REAL NOT NULL, content TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS session_recaps_session ON session_recaps(session, seq);
            CREATE TABLE IF NOT EXISTS extractions (
                session TEXT PRIMARY KEY, version REAL NOT NULL, content TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memories (
                workspace TEXT PRIMARY KEY, source TEXT NOT NULL, content TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memory_rollouts (
                session TEXT PRIMARY KEY, version REAL NOT NULL, summary TEXT NOT NULL, slug TEXT
            );
            CREATE TABLE IF NOT EXISTS memory_artifacts (
                workspace TEXT PRIMARY KEY, source TEXT NOT NULL, files TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS mcp_tool_cache (
                workspace TEXT NOT NULL, config TEXT NOT NULL, tools TEXT NOT NULL, expires REAL NOT NULL,
                PRIMARY KEY (workspace, config)
            );
        """)
        db.executescript(graph.SCHEMA)
        db.executescript(SERVICE_SCHEMA)


def session_write(method):
    @wraps(method)
    def serialized(session, *args, **kwargs):
        with session.writer:
            return method(session, *args, **kwargs)
    return serialized


UNREAD = object()


class Session:
    def __init__(self, path, session_id, workspace):
        self.path = path
        self.id = session_id
        self.writer = RLock()
        self.archive_route, self.archive_shape, self.archive_threshold = None, "auto", None
        self._archive_view = None
        self.archive_images_disabled = False
        self.navigation_epoch = 0
        self.fresh_provider_id = None
        self.tool_replay = None
        self.timestamp_batches = False
        self._context_notes = self._rollover_request = UNREAD
        with database(path) as db:
            db.execute(
                "INSERT OR IGNORE INTO sessions(id, workspace, updated) VALUES (?, ?, ?)",
                (session_id, workspace, time.time()),
            )
            stored_workspace, self.first_kept, self.summary = db.execute(
                "SELECT workspace, first_kept, summary FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if stored_workspace != workspace:
                raise ValueError("Session belongs to a different workspace.")
            db.execute("""INSERT OR IGNORE INTO session_titles
                          SELECT session,json_extract(header,'$.title'),
                                 CASE json_extract(header,'$.titleSource') WHEN 'auto' THEN 'auto' ELSE 'user' END
                          FROM session_imports WHERE session=? AND json_type(header,'$.title')='text'
                          AND trim(json_extract(header,'$.title'))<>''""", (session_id,))
            native = db.execute("SELECT payload FROM native_compactions WHERE session = ?", (session_id,)).fetchone()
            self.native_summary = json.loads(native[0]) if native else None
            self.leaf = graph.initialize_session(db, session_id, self.first_kept, self.summary, self.native_summary)
            from session_jsonl import upgrade_responses_compactions

            if upgrade_responses_compactions(db, session_id):
                self.first_kept, self.summary, self.native_summary = graph.checkpoint(db, session_id, self.leaf)
                graph.projection(db, session_id, self.first_kept, self.summary, self.native_summary)
            self.reset_node, self.transcript_start = graph.transcript_boundary(db, session_id, self.leaf)
            self.prompt_cache_key = graph.cache_key(db, session_id, self.reset_node)
            self.batches = self.read_batches(db, self.first_kept)

    def read_batches(self, db, start, before=None):
        return graph.stored_batches(db, self.id, self.leaf, start, before)

    @property
    def transcript_id(self):
        return self.fresh_provider_id or (self.prompt_cache_key if self.reset_node is not None else self.id)

    @property
    def request_cache_key(self):
        return self.fresh_provider_id or self.prompt_cache_key

    def fresh_provider_session(self):
        previous = self.transcript_id
        self.fresh_provider_id = str(uuid4())
        return previous, self.fresh_provider_id

    def tree(self):
        with database(self.path) as db:
            return list(graph.tree(db, self.id, self.leaf))

    def fork(self, workspace, new_id, target=None, *, messages=(), repair_tail=False):
        with database(self.path) as db:
            leaf = graph.fork(
                db, self.id, new_id, self.leaf if target is None else target, workspace, time.time(),
                whole=target is None,
            )
            if repair_tail:
                leaf = graph.repair_fork_tail(db, new_id, leaf)
            for message in messages:
                seq = db.execute("INSERT INTO batches(session,messages) VALUES (?,?)", (new_id, pack([message]))).lastrowid
                leaf = graph.append(db, new_id, leaf, "batch", seq, graph.batch_payload([message]))
            if messages or repair_tail:
                graph.set_head(db, new_id, leaf)

    def label(self, target, name):
        with database(self.path) as db:
            graph.destination(db, self.id, target)
            if name:
                db.execute("INSERT OR REPLACE INTO session_labels VALUES (?,?,?)", (self.id, target, name))
            else:
                db.execute("DELETE FROM session_labels WHERE session = ? AND node = ?", (self.id, target))

    def title(self):
        with database(self.path) as db:
            row = db.execute("SELECT title FROM session_titles WHERE session=?", (self.id,)).fetchone()
        return row[0] if row else None

    def set_title(self, name, *, automatic=False):
        name = " ".join("".join(char if char.isprintable() or char == "\u200d" else " " for char in name).split())[:256]
        if not name:
            return False
        with database(self.path) as db:
            sql = "INSERT OR IGNORE" if automatic else "INSERT OR REPLACE"
            return bool(db.execute(f"{sql} INTO session_titles VALUES (?,?,?)",
                                   (self.id, name, "auto" if automatic else "user")).rowcount)

    def branch_preview(self, target, summarize=False):
        with database(self.path) as db:
            destination, draft = graph.destination(db, self.id, target)
            abandoned = graph.abandoned(db, self.id, self.leaf, destination) if summarize else []
            return destination, draft, [portable_message(message) for message in abandoned]

    @session_write
    def navigate(self, target, summary=None):
        with database(self.path) as db:
            if target is not None:
                graph.destination(db, self.id, target)
            if summary:
                batch = [{"role": "developer", "content": summary, "_branch_summary": True, "_branch_from_node": self.leaf}]
                seq = db.execute(
                    "INSERT INTO batches(session,messages) VALUES (?,?)", (self.id, pack(batch)),
                ).lastrowid
                target = graph.append(db, self.id, target, "batch", seq)
            cut, context, native = graph.checkpoint(db, self.id, target)
            batches = graph.stored_batches(db, self.id, target, cut)
            reset_node, start = graph.transcript_boundary(db, self.id, target)
            cache_key = graph.cache_key(db, self.id, reset_node)
            graph.set_head(db, self.id, target)
            graph.projection(db, self.id, cut, context, native)
            self.touch(db)
        self.leaf, self.first_kept, self.summary = target, cut, context
        self.native_summary, self.batches = native, batches
        self.reset_node, self.transcript_start, self.prompt_cache_key = reset_node, start, cache_key
        self.fresh_provider_id = None
        self.tool_replay = None
        self._archive_view = None
        self.navigation_epoch += 1
        self._context_notes = self._rollover_request = UNREAD

    @session_write
    def reset_context(self):
        dropped = sum(len(batch) for _, batch in self.batches) + bool(self.summary)
        with database(self.path) as db:
            db.execute("BEGIN IMMEDIATE")
            head, = db.execute("SELECT leaf FROM session_heads WHERE session=?", (self.id,)).fetchone()
            if head != self.leaf:
                raise RuntimeError("Session changed in another process. Reopen it before clearing context.")
            last = graph.last_batch(db, self.id, self.leaf)
            leaf = graph.append(db, self.id, self.leaf, "reset", payload=last)
            graph.set_head(db, self.id, leaf)
            graph.projection(db, self.id, last + 1, "", None)
            cache_key = graph.cache_key(db, self.id, leaf)
            self.touch(db)
        self.leaf = self.reset_node = leaf
        self.first_kept = self.transcript_start = last + 1
        self.prompt_cache_key = cache_key
        self.fresh_provider_id = None
        self.tool_replay = None
        self.summary, self.native_summary, self.batches = "", None, []
        self._archive_view = None
        self.archive_images_disabled = False
        self.navigation_epoch += 1
        self._context_notes = self._rollover_request = None
        return dropped

    def archived_responses_prefix(self):
        native = self.native_summary
        if not native or "items" not in native:
            return self.summary, self.first_kept, []
        summary, start = native["portable_summary"], native["portable_first_kept"]
        with database(self.path) as db:
            return summary, start, self.read_batches(db, start, self.first_kept)

    @session_write
    def restore_portable(self, route):
        # Route selection changes the model's view of the checkpoint, not the journal.
        with database(self.path) as db:
            cut, summary, native = graph.checkpoint(db, self.id, self.leaf)
            if not native:
                return
            if "items" in native:
                split_tool = native.get("provider") and graph.has_late_tool_results(db, self.id, self.leaf, cut - 1)
                if split_tool or not responses_replay_matches(native, route):
                    cut, summary = native["portable_first_kept"], native["portable_summary"]
                    archive = native.get("portable_archive")
                    native = {"snapcompact": archive} if archive else native.get("portable_handoff")
            elif native.get("anthropic_provider") and graph.has_late_tool_results(db, self.id, self.leaf, cut - 1):
                cut, summary, native = self.transcript_start, "", None
            else:
                return
            if (cut, summary, native) == (self.first_kept, self.summary, self.native_summary):
                return
            self.batches = self.read_batches(db, cut)
        self.first_kept, self.summary, self.native_summary = cut, summary, native
        self._archive_view = None
        self._rollover_request = UNREAD

    def configure_archive(self, route, shape, threshold):
        self.archive_route = route
        self.archive_shape = shape
        self.archive_threshold = threshold
        self._archive_view = None
        self.archive_images_disabled = False

    def archive_view(self):
        archive = (self.native_summary or {}).get("snapcompact")
        if not archive:
            return None
        if self._archive_view is None:
            route = self.archive_route
            if route and self.archive_images_disabled:
                route = replace(route, vision=False)
            self._archive_view = snapcompact.history(archive, route, self.archive_shape, self.archive_threshold)
        return self._archive_view

    def disable_archive_images(self):
        view = self.archive_view()
        if not view or not any(part["type"] == "image_url" for part in view[1]):
            return False
        self.archive_images_disabled = True
        self._archive_view = None
        return True

    def apply_elisions(self, replacements):
        for seq, batch in self.batches:
            for index, message in enumerate(batch):
                if (seq, index) in replacements:
                    batch[index] = graph.apply_elision(message, replacements[seq, index])

    @session_write
    def archive_elisions(self, replacements, artifacts, *, check=None):
        with database(self.path) as db:
            db.executemany(
                "INSERT OR IGNORE INTO artifacts(session, id, content) VALUES (?, ?, ?)",
                ((self.id, identifier, text) for identifier, text in artifacts.items()),
            )
            if check:
                check()
            leaf = graph.append(
                db, self.id, self.leaf, "elision",
                payload=[(seq, index, text) for (seq, index), text in replacements.items()],
            )
            graph.set_head(db, self.id, leaf)
            if check:
                check()
        self.leaf = leaf
        self.apply_elisions(replacements)

    @session_write
    def drop_images(self, check=None):
        if check:
            check()
        with database(self.path) as db:
            batches = graph.stored_batches(db, self.id, self.leaf)
        replacements, removed = {}, 0
        for seq, batch in batches:
            if check:
                check()
            for index, message in enumerate(batch):
                _, count = graph.without_images(message)
                if count:
                    replacements[seq, index] = {"drop_images": True}
                    removed += count
        if replacements:
            self.archive_elisions(replacements, {}, check=check)
            self._rollover_request = UNREAD
        return removed

    def save_artifact(self, text):
        identifier = artifact_id(text)
        with database(self.path) as db:
            db.execute("INSERT OR IGNORE INTO artifacts VALUES (?, ?, ?)", (self.id, identifier, text))
        return identifier

    def artifact(self, identifier, offset, limit):
        with database(self.path) as db:
            row = db.execute(
                "SELECT content FROM artifacts WHERE session = ? AND id = ?",
                (self.id, identifier),
            ).fetchone()
        if row is not None:
            content = row[0]
            return content[offset:offset + limit], len(content)
        directory = Path(self.path).parent / "artifacts" / self.id
        if (numbered := read_numbered(directory, identifier, offset, limit)) is not None:
            return numbered
        raise ValueError("Artifact not found in this session.")

    @session_write
    def append(self, messages, *, before=None):
        attempts = [message["_usage_attempt"] for message in messages if "_usage_attempt" in message]
        stored = [{key: value for key, value in message.items() if key != "_usage_attempt"} for message in messages] if attempts else messages
        with database(self.path) as db:
            seq = db.execute(
                "INSERT INTO batches(session, messages) VALUES (?, ?)", (self.id, pack(stored)),
            ).lastrowid
            leaf = graph.append(db, self.id, self.leaf, "foreign", payload=before) if before else self.leaf
            stamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds") if self.timestamp_batches else None
            payload = graph.batch_payload(messages, stamp)
            leaf = graph.append(db, self.id, leaf, "batch", seq, payload)
            injected = list(dict.fromkeys(name for message in messages for name in message.get("_ttsr_rules", ())))
            if injected:
                leaf = graph.append(db, self.id, leaf, "foreign", payload={"type": "ttsr_injection", "injectedRules": injected})
            graph.set_head(db, self.id, leaf)
            self.consume_notifications(db, messages)
            if attempts:
                db.executemany("DELETE FROM session_usage_attempts WHERE session = ? AND id = ?",
                               ((self.id, identifier) for identifier in attempts))
            self.touch(db)
        self.leaf = leaf
        if attempts:
            for message in messages:
                message.pop("_usage_attempt", None)
        self.batches.append((seq, messages))
        return seq

    def record_usage_attempt(self, payload, ancestors=()):
        with database(self.path) as db:
            insert = """INSERT INTO session_usage_attempts(session,parent,payload)
                SELECT ?,?,json_set(?, '$._after_node', COALESCE(MAX(id),0)) FROM session_nodes WHERE session = ?"""
            identifier = db.execute(insert, (self.id, self.leaf, pack(payload), self.id)).lastrowid
            for owner in ancestors:
                mirror = {**payload, "purpose": "task", "agentPurpose": payload["purpose"],
                          "agentSessionId": self.id, "agentAttemptId": identifier}
                db.execute(insert, (owner.id, owner.leaf, pack(mirror), owner.id))
            return identifier

    def finish_usage_attempt(self, identifier, reason, ancestors=()):
        with database(self.path) as db:
            db.execute("""UPDATE session_usage_attempts SET payload = json_set(payload, '$.stopReason', ?)
                          WHERE session = ? AND id = ?""", (reason, self.id, identifier))
            db.executemany("""UPDATE session_usage_attempts SET payload = json_set(payload, '$.stopReason', ?)
                              WHERE session = ? AND json_extract(payload, '$.agentSessionId') = ?
                              AND json_extract(payload, '$.agentAttemptId') = ?""",
                           ((reason, owner.id, self.id, identifier) for owner in ancestors))

    @session_write
    def update_tool_result(self, seq, index, message):
        with database(self.path) as db:
            leaf = graph.append(db, self.id, self.leaf, "tool_update", payload=[seq, index, message])
            graph.set_head(db, self.id, leaf)
            self.consume_notifications(db, [message])
            self.touch(db)
        self.leaf = leaf
        for batch_seq, batch in reversed(self.batches):
            if batch_seq == seq:
                batch[index] = message
                break

    def record_tool_start(self, call):
        self.record_metadata(exit_log.custom(exit_log.TOOL_START, exit_log.tool_start(call)))

    def conversation_tail(self):
        return next(((seq, batch, message) for seq, batch in reversed(self.batches) for message in reversed(batch)
                     if message["role"] in ("user", "assistant", "tool") and not any(
                         key in message for key in ("_goal_message", "_async_jobs", "_mcp_notification", "_service_runs")
                     )), (None, (), None))

    def pending_tools(self, seq, batch):
        pending = {item["tool_call_id"] for item in batch if exit_log.pending_result(item)}
        if not pending:
            return []
        with database(self.path) as db:
            rows = db.execute(graph.ANCESTORS + """
                SELECT json_extract(n.payload,'$.data') FROM ancestry a JOIN session_nodes n ON n.id=a.id
                WHERE n.kind='foreign' AND json_extract(n.payload,'$.customType')=?
                AND n.id > (SELECT id FROM session_nodes WHERE session=? AND kind='batch' AND batch=?) ORDER BY n.id
            """, (self.leaf, self.id, self.id, exit_log.TOOL_START, self.id, seq))
            starts = {item["toolCallId"]: item for raw, in rows if isinstance(item := json.loads(raw or "null"), dict)
                      and isinstance(item.get("toolCallId"), str)}
        return [starts.get(call["id"], {"toolCallId": call["id"], "toolName": call["function"]["name"]})
                for call in batch[0].get("tool_calls", ()) if call["id"] in pending]

    @session_write
    def record_exit(self, kind, reason):
        seq, batch, tail = self.conversation_tail()
        pending = self.pending_tools(seq, batch)
        if tail is None or kind == "normal" and not pending and not any(
            message["role"] == "assistant" for _, items in self.batches for message in items
        ):
            return
        data = {"reason": reason, "kind": kind, "recordedAt": exit_log.timestamp()}
        if pending:
            data["pendingToolCalls"] = pending
        self.record_metadata(exit_log.custom(exit_log.SESSION_EXIT, data))

    @session_write
    def recover_interrupted(self, *, assume_exit=False, model=None):
        seq, batch, tail = self.conversation_tail()
        if tail is None:
            return False
        with database(self.path) as db:
            row = db.execute(graph.ANCESTORS + """
                SELECT json_extract(n.payload,'$.data') FROM ancestry a JOIN session_nodes n ON n.id=a.id
                WHERE n.kind='foreign' AND json_extract(n.payload,'$.customType')=?
                AND n.id > (SELECT id FROM session_nodes WHERE session=? AND kind='batch' AND batch=?)
                AND n.id > coalesce((SELECT max(update_node.id) FROM ancestry path
                    JOIN session_nodes update_node ON update_node.id=path.id
                    WHERE update_node.kind='tool_update' AND json_extract(update_node.payload,'$[0]')=?),0)
                ORDER BY n.id DESC LIMIT 1
            """, (self.leaf, self.id, self.id, exit_log.SESSION_EXIT, self.id, seq, seq)).fetchone()
        recorded = json.loads(row[0] or "null") if row else None
        if not isinstance(recorded, dict) or recorded.get("kind") not in ("normal", "signal", "fatal", "process_exit"):
            recorded = None
        if recorded and recorded["kind"] == "normal" and not recorded.get("pendingToolCalls"):
            return False
        if exit_log.interrupted(tail):
            return assume_exit
        if not recorded and not assume_exit or tail["role"] == "assistant" and not tail.get("tool_calls"):
            return False
        previous = next((item for _, items in reversed(self.batches) for item in reversed(items)
                         if item["role"] == "assistant"), None)
        if tail["role"] == "tool" and previous and exit_log.stop_reason(previous) in ("error", "aborted"):
            return False
        for index, item in enumerate(batch):
            if exit_log.pending_result(item):
                self.update_tool_result(seq, index, {"role": "tool", "tool_call_id": item["tool_call_id"],
                    "content": exit_log.UNKNOWN_RESULT, "_tool_error": True})
        self.append([exit_log.abort_message(recorded.get("recordedAt") if recorded else exit_log.timestamp(),
                                           previous if previous and previous.get("_rpc") else model)])
        return not assume_exit or not recorded or recorded["kind"] != "normal"

    @session_write
    def prepare_tool_replay(self):
        if len(self.batches) < 2 or len(self.batches[-1][1]) != 1 or not exit_log.interrupted(self.batches[-1][1][0]):
            return False
        seq, batch = self.batches[-2]
        calls = exit_log.runnable_calls(batch[0])
        if not calls or len(batch) != len(calls) + 1 or {item.get("tool_call_id") for item in batch[1:]
                if item.get("_tool_error") is True} != {call["id"] for call in calls}:
            return False
        with database(self.path) as db:
            row = db.execute("""SELECT n.id,b.messages FROM session_nodes n JOIN batches b ON n.batch=b.seq
                              WHERE n.session=? AND n.kind='batch' AND n.batch=?""", (self.id, seq)).fetchone()
        if row is None or not all(exit_log.pending_result(item) for item in json.loads(row[1])[1:]):
            return False
        self.navigate(row[0])
        self.tool_replay = self.batches[-1]
        return True

    def consume_notifications(self, db, messages):
        ids = {row["id"] for message in messages for row in message.get("_irc_messages", ())}
        if ids:
            db.executemany("UPDATE peer_messages SET consumed=1 WHERE recipient=? AND seq=?",
                           ((self.id, identifier) for identifier in ids))
        services = {row["id"] for message in messages for row in message.get("_service_runs", ())}
        if services:
            db.executemany("UPDATE service_runs SET consumed=1 WHERE owner=? AND id=? AND finished IS NOT NULL",
                           ((self.id, identifier) for identifier in services))

    @session_write
    def record_metadata(self, *payloads):
        with database(self.path) as db:
            leaf = self.leaf
            for payload in payloads:
                leaf = graph.append(db, self.id, leaf, "foreign", payload=payload)
            graph.set_head(db, self.id, leaf)
            self.touch(db)
        self.leaf = leaf

        if any(payload.get("customType") == NOTE_TYPE for payload in payloads):
            self._context_notes = UNREAD

    @session_write
    def context_notes(self):
        if self._context_notes is not UNREAD:
            return self._context_notes
        self._context_notes = self.read_context_notes()
        return self._context_notes

    def read_context_notes(self):
        with database(self.path) as db:
            rows = db.execute(graph.ANCESTORS + """
                SELECT n.id,n.kind,n.payload,s.source_id FROM ancestry a JOIN session_nodes n ON n.id=a.id
                LEFT JOIN session_entry_sources s ON s.session=n.session AND s.node=n.id
                WHERE n.kind='reset' OR (n.kind='foreign' AND json_extract(n.payload,'$.customType')=?)
                ORDER BY n.id DESC
            """, (self.leaf, self.id, self.id, NOTE_TYPE))
            for node, kind, payload, source in rows:
                if kind == "reset":
                    break
                entry = json.loads(payload)
                data = entry.get("data")
                if (entry.get("type") == "custom" and isinstance(data, dict) and data.keys() == {"version", "text"}
                        and type(data["version"]) is int
                        and data["version"] == 1 and isinstance(text := data["text"], str)):
                    try:
                        if len(text.encode("utf-8")) <= NOTE_LIMIT:
                            return {"entryId": source or entry.get("id") or str(node), "text": text}
                    except UnicodeEncodeError:
                        continue
        return None

    @session_write
    def save_context_notes(self, text, epoch, check=None):
        if check:
            check()
        if epoch != self.navigation_epoch:
            raise ValueError("The session branch changed before the notebook could be saved.")
        size = len(text.encode("utf-8"))
        if size > NOTE_LIMIT:
            raise ValueError(f"Context notes exceed the {NOTE_LIMIT:,} UTF-8 byte limit.")
        identifier = uuid4().hex
        self.record_metadata({"type": "custom", "customType": NOTE_TYPE, "id": identifier,
                              "timestamp": datetime.now(timezone.utc).isoformat(), "data": {"version": 1, "text": text}})
        return {"entryId": identifier, "text": text, "bytes": size}

    def full_history(self):
        with self.writer, database(self.path) as db:
            leaf = self.leaf
            rows = graph.nodes(db, self.id, leaf)
            batches = dict(db.execute(graph.ANCESTORS + """
                SELECT b.seq,b.messages FROM ancestry a JOIN session_nodes n ON n.id=a.id
                JOIN batches b ON b.session=n.session AND b.seq=n.batch
            """, (leaf, self.id, self.id)))
            sources = dict(db.execute("SELECT node,source_id FROM session_entry_sources WHERE session=?", (self.id,)))
        yield "# Current branch — full history"
        yield ""
        records = [(node, parent, kind, seq, json.loads(raw) if raw else None) for node, parent, kind, seq, raw in rows]
        updated = {(payload[0], payload[1]) for _, _, kind, _, payload in records if kind == "tool_update"}
        identifiers, anchors = {None: "root"}, {}
        for node, parent, kind, seq, payload in records:
            if kind == "elision":
                identifiers[node] = identifiers[parent]
                continue
            identifier = sources.get(node) or (payload.get("id") if isinstance(payload, dict) else None) or str(node)
            identifiers[node] = identifier
            yield f"## Entry {identifier} · {'message' if kind in ('batch', 'tool_update') else kind}"
            yield f"Parent: {identifiers[parent]}"
            if kind == "batch":
                anchors[seq] = identifier
                for position, message in enumerate(json.loads(batches[seq])):
                    if (seq, position) not in updated:
                        yield from render_message(message)
            elif kind == "tool_update":
                yield from render_message(payload[2])
            else:
                yield from render_metadata(kind, payload, anchors)
            yield ""

    def context_boundary(self):
        with database(self.path) as db:
            row = db.execute(graph.ANCESTORS + """
                SELECT MAX(n.id) FROM ancestry a JOIN session_nodes n ON n.id=a.id
                WHERE n.kind IN ('compact','reset')
            """, (self.leaf, self.id, self.id)).fetchone()
            return row[0] or 0

    def rollover_request(self, cut, committed):
        if not committed:
            return self.read_rollover_request(cut, False)
        if self._rollover_request is UNREAD:
            self._rollover_request = self.read_rollover_request(cut, True)
        return self._rollover_request

    def read_rollover_request(self, cut, committed):
        with database(self.path) as db:
            leaf = self.leaf
            if committed:
                entry = db.execute(graph.ANCESTORS + """
                    SELECT n.parent FROM ancestry a JOIN session_nodes n ON n.id=a.id
                    WHERE n.kind='compact' ORDER BY n.id DESC LIMIT 1
                """, (leaf, self.id, self.id)).fetchone()
                if entry:
                    leaf = entry[0]
            _, start = graph.transcript_boundary(db, self.id, leaf)
            parents = {owner for owner, in db.execute("SELECT owner FROM subagent_links WHERE child=?", (self.id,))}
            rows = db.execute(graph.ANCESTORS + """
                , candidates(seq, position, message) AS (
                    SELECT b.seq,j.key,j.value FROM ancestry a JOIN session_nodes n ON n.id=a.id
                    JOIN batches b ON b.session=n.session AND b.seq=n.batch JOIN json_each(b.messages) j
                    WHERE b.seq>=? AND (json_extract(j.value,'$.role')='user'
                        OR json_type(j.value,'$._parent_assignment')='true'
                        OR json_type(j.value,'$._irc_messages')='array')
                    UNION ALL
                    SELECT json_extract(n.payload,'$[0]'),json_extract(n.payload,'$[1]'),json_extract(n.payload,'$[2]')
                    FROM ancestry a JOIN session_nodes n ON n.id=a.id WHERE n.kind='tool_update'
                    AND json_extract(n.payload,'$[0]')>=? AND json_type(n.payload,'$[2]._irc_messages')='array'
                ) SELECT seq,position,message FROM candidates ORDER BY seq DESC,position DESC
            """, (leaf, self.id, self.id, start, start))
            for seq, position, raw in rows:
                message = json.loads(raw)
                user = message["role"] == "user" and not any(message.get(key) for key in (
                    "_mcp_notification", "_async_jobs", "_service_runs", "_irc_messages", "_goal_message",
                ))
                parent = message.get("_parent_assignment") or any(
                    row.get("sender") in parents for row in message.get("_irc_messages", ())
                )
                if user or parent:
                    if seq >= cut:
                        return None
                    projected = graph.stored_batches(db, self.id, self.leaf, seq, seq + 1)[0][1][position]
                    return {"role": "user", "content": projected["content"]}
        return None

    def goal_record(self):
        with database(self.path) as db:
            row = db.execute(graph.ANCESTORS + """
                SELECT n.payload FROM ancestry a JOIN session_nodes n ON n.id = a.id
                WHERE n.kind = 'foreign' AND json_extract(n.payload, '$.type') = 'mode_change'
                ORDER BY n.id DESC LIMIT 1
            """, (self.leaf, self.id, self.id)).fetchone()
        return json.loads(row[0]) if row else {}

    def todo_snapshots(self):
        for _, _, _, phases in self.todo_records():
            yield phases

    def todo_records(self):
        with database(self.path) as db:
            rows = db.execute(graph.ANCESTORS + """
                , tool_results(node, position, message) AS (
                    SELECT n.id, CAST(part.key AS INTEGER), part.value
                    FROM ancestry a JOIN session_nodes n ON n.id = a.id
                    JOIN batches b ON b.session = n.session AND b.seq = n.batch
                    JOIN json_each(b.messages) part
                    WHERE n.kind = 'batch' AND b.messages LIKE '%"_todo_snapshot":true%'
                    UNION ALL
                    SELECT n.id, 0, json_extract(n.payload, '$[2]')
                    FROM ancestry a JOIN session_nodes n ON n.id = a.id WHERE n.kind = 'tool_update'
                ), snapshots(node, position, entry, phases) AS (
                    SELECT node, position, json_extract(message, '$._todo_entry_id'),
                           json_extract(message, '$._tool_details.phases') FROM tool_results
                    WHERE json_extract(message, '$._todo_snapshot') = 1
                    UNION ALL
                    SELECT n.id, 0, json_extract(n.payload, '$.id'), json_extract(n.payload, '$.data.phases')
                    FROM ancestry a JOIN session_nodes n ON n.id = a.id
                    WHERE n.kind = 'foreign' AND json_extract(n.payload, '$.type') = 'custom'
                      AND json_extract(n.payload, '$.customType') = 'user_todo_edit'
                )
                SELECT t.node, t.position, COALESCE(s.source_id, t.entry), t.phases FROM snapshots t
                LEFT JOIN session_entry_sources s ON s.node = t.node AND s.session = ?
                WHERE t.phases IS NOT NULL ORDER BY t.node DESC, t.position DESC
            """, (self.leaf, self.id, self.id, self.id))
            for node, position, entry, phases in rows:
                if isinstance(phases, str):
                    try:
                        yield node, position, entry, json.loads(phases)
                    except ValueError:
                        continue

    @session_write
    def identify_todo(self, node, position):
        entry = uuid4().hex
        with database(self.path) as db:
            kind, seq = db.execute("SELECT kind,batch FROM session_nodes WHERE session = ? AND id = ?",
                                   (self.id, node)).fetchone()
            if kind == "batch":
                db.execute("UPDATE batches SET messages=json_set(messages, ?, ?) WHERE session=? AND seq=?",
                           (f"$[{position}]._todo_entry_id", entry, self.id, seq))
            else:
                key = "$[2]._todo_entry_id" if kind == "tool_update" else "$.id"
                db.execute("UPDATE session_nodes SET payload=json_set(payload, ?, ?) WHERE session=? AND id=?",
                           (key, entry, self.id, node))
        return entry

    def todo_hud_visibility(self, entry, fingerprint):
        with database(self.path) as db:
            row = db.execute(graph.ANCESTORS + """
                SELECT json_extract(n.payload, '$.data.visibility')
                FROM ancestry a JOIN session_nodes n ON n.id = a.id
                WHERE n.kind = 'foreign' AND json_extract(n.payload, '$.customType') = 'todo_hud_state'
                  AND json_extract(n.payload, '$.data.sourceEntryId') = ?
                  AND json_extract(n.payload, '$.data.fingerprint') = ?
                  AND json_extract(n.payload, '$.data.visibility') IN ('revealed','dismissed')
                ORDER BY n.id DESC LIMIT 1
            """, (self.leaf, self.id, self.id, entry, fingerprint)).fetchone()
        return row[0] if row else None

    def ancestor_ids(self):
        with database(self.path) as db:
            return graph.ancestor_ids(db, self.id, self.leaf)

    def last_thinking_change(self):
        with database(self.path) as db:
            row = db.execute(graph.ANCESTORS + """
                SELECT n.payload FROM ancestry a JOIN session_nodes n ON n.id = a.id
                WHERE n.kind = 'foreign' AND json_extract(n.payload, '$.type') = 'thinking_level_change'
                ORDER BY n.id DESC LIMIT 1
            """, (self.leaf, self.id, self.id)).fetchone()
        return json.loads(row[0]) if row else {}

    def injected_rules(self):
        with database(self.path) as db:
            rows = db.execute(graph.ANCESTORS + """
                SELECT DISTINCT rule.value FROM ancestry a JOIN session_nodes n ON n.id=a.id
                JOIN json_each(n.payload, '$.injectedRules') rule
                WHERE n.id>? AND n.kind='foreign' AND json_extract(n.payload, '$.type')='ttsr_injection' AND rule.type='text'
            """, (self.leaf, self.id, self.id, self.reset_node or 0))
            return {name for name, in rows}

    def touch(self, db):
        db.execute("UPDATE sessions SET updated = ? WHERE id = ?", (time.time(), self.id))

    def cache_idle(self):
        with database(self.path) as db:
            updated, = db.execute("SELECT updated FROM sessions WHERE id=?", (self.id,)).fetchone()
        return time.time() - updated >= IDLE_FLUSH_SECONDS

    def last_route(self, primary):
        with database(self.path) as db:
            return graph.last_route(db, self.id, self.leaf, primary)

    @session_write
    def record_route(self, primary, active, reason):
        with database(self.path) as db:
            db.execute(
                "INSERT INTO model_switches(session, primary_route, active_route, reason) VALUES (?, ?, ?, ?)",
                (self.id, primary, active, reason),
            )
            leaf = graph.append(db, self.id, self.leaf, "route", payload=[primary, active])
            graph.set_head(db, self.id, leaf)
        self.leaf = leaf

    def record_recap(self, text):
        with database(self.path) as db:
            db.execute(
                "INSERT INTO session_recaps(session, created, content) VALUES (?, ?, ?)",
                (self.id, time.time(), text),
            )

    def can_continue(self):
        message = next((message for _, batch in reversed(self.batches) for message in reversed(batch)
                        if "_session_transfer" not in message and not exit_log.cutoff(message)), None)
        if message is None and (covered := (self.native_summary or {}).get("replay_through")):
            with database(self.path) as db:
                batches = self.read_batches(db, covered, covered + 1)
            message = batches[-1][1][-1] if batches else None
        if message is None:
            return False
        return message["role"] in ("user", "tool", "developer") or paused_turn(message) and not message.get("_pause_exhausted")

    def messages(
        self, system, summary=None, cut=0, *, include_native=False, include_archive=False, before=None, method=None,
        thinking_route=None, for_compaction=False,
    ):
        committed = summary is None
        archive = self.archive_view() if include_archive and committed else None
        metadata = self.native_summary if summary is None else None
        native = metadata if include_native else None
        if metadata:
            method = metadata.get("method")
        summary = self.summary if summary is None else summary
        head = [{"role": "system", "content": system}]
        if archive:
            lead, blocks, _ = archive
            content = [{"type": "text", "text": lead}, *blocks]
            if not any(part["type"] == "image_url" for part in blocks):
                content = "\n".join(part["text"] for part in content)
            head.append({"role": "user", "content": content})
        elif summary or native and ("items" in native or "block" in native):
            context = summary if method == ROLLOVER else handoff.CONTEXT.replace("{{summary}}", summary) if method == "handoff" else (
                "Summary of earlier conversation. Build on prior work without repeating it. "
                "Follow the latest user request and carry forward constraints on unfinished work. "
                "Requests from completed turns are historical.\n"
                f"<summary>\n{summary}\n</summary>"
            )
            head.append({
                "role": "user",
                "content": context,
            })
            if native and ("items" in native or "block" in native):
                key = "_responses_compaction" if "items" in native else "_anthropic_compaction"
                head[-1][key] = native
        if method == ROLLOVER and (request := self.rollover_request(max(cut, self.first_kept) if committed else cut, committed)):
            head.append(request)
        provenance = {"_compaction_context": True} if for_compaction else {}
        if provenance and len(head) > 1:
            head[-1].update(provenance)
        messages = head + [
            {**provenance, "role": "user", "content": developer_content(message),
             **({"_history_at": message["_history_at"]} if include_native and "_history_at" in message else {})}
            if message["role"] == "developer" else (message if include_native else portable_message(message, thinking_route))
            for seq, batch in self.batches if seq >= cut and (before is None or seq < before)
            for message in batch
            if not exit_log.cutoff(message)
            and (include_native or message.get("content") or not (paused_turn(message) or message.get("_dsml_leak")))
        ]
        if (notes := self.context_notes()) and notes["text"]:
            insert = 1
            if len(messages) > 1 and any(key in messages[1] for key in ("_anthropic_compaction", "_responses_compaction")):
                insert = 2
                if insert < len(messages) and messages[insert]["role"] == "assistant":
                    insert += 1
                while insert < len(messages) and messages[insert]["role"] == "tool":
                    insert += 1
            messages.insert(insert, {**provenance, **note_message(notes["text"])})
        return messages

    @session_write
    def compact(self, cut, summary, native=None, *, check=None):
        if native and "items" in native:
            previous = self.native_summary or {}
            native = {
                **native,
                "portable_summary": previous.get("portable_summary", self.summary),
                "portable_first_kept": previous.get("portable_first_kept", self.first_kept),
            }
            portable_archive = previous.get("portable_archive") or previous.get("snapcompact")
            if portable_archive:
                native["portable_archive"] = portable_archive
            portable_handoff = previous.get("portable_handoff") or (
                previous if previous.get("method") in ("handoff", ROLLOVER) else None
            )
            if portable_handoff:
                native["portable_handoff"] = portable_handoff
        with database(self.path) as db:
            if check:
                check()
            leaf = graph.append(db, self.id, self.leaf, "compact", payload=[cut, summary, native])
            graph.set_head(db, self.id, leaf)
            graph.projection(db, self.id, cut, summary, native)
            if check:
                check()
        self.leaf = leaf
        self.first_kept, self.summary = cut, summary
        self.native_summary = native
        self._rollover_request = UNREAD
        self._archive_view = None
        self.batches = [(seq, batch) for seq, batch in self.batches if seq >= cut]

    @session_write
    def update_compaction_metadata(self, node, *, tokens_after, warning=None, check=None):
        if node is None:
            return
        with database(self.path) as db:
            row = db.execute(graph.ANCESTORS + """
                SELECT n.id,n.kind,n.payload FROM ancestry a JOIN session_nodes n ON n.id=a.id
                WHERE n.kind IN ('compact','reset') ORDER BY n.id DESC LIMIT 1
            """, (self.leaf, self.id, self.id)).fetchone()
            if not row or row[:2] != (node, "compact"):
                return
            cut, summary, existing = json.loads(row[2])
            metadata = {**(existing or {}), "tokensAfter": tokens_after}
            if warning:
                metadata["warning"] = warning
            if metadata == existing:
                return
            if check:
                check()
            db.execute("UPDATE session_nodes SET payload=? WHERE session=? AND id=?",
                       (pack([cut, summary, metadata]), self.id, node))
            graph.projection(db, self.id, cut, summary, metadata)
            if check:
                check()
        self.native_summary = metadata

    def compaction_diagnostics(self):
        data = self.native_summary or {}
        result = {key: data[key] for key in ("tokensBefore", "tokensAfter")
                  if type(data.get(key)) is int and data[key] >= 0}
        if isinstance(warning := data.get("warning"), str) and warning.strip():
            result["warning"] = warning
        return result

    def file_operations(self, before):
        with database(self.path) as db:
            rows = graph.stored_batches(db, self.id, self.leaf, self.transcript_start, before, apply_elisions=False)
            return handoff.file_operations(message for _, batch in rows for message in batch)


@dataclass(frozen=True)
class Snapshot:
    boundary: int
    first_kept: int
    context_tokens: int
    transcript: str
    navigation_epoch: int = 0
    leaf: int | None = None
    pending_from: int | None = None


@dataclass(frozen=True)
class CompactionSummary:
    text: str
    native: dict | None = None

    def through(self, seq):
        return replace(self, native={**self.native, "replay_through": seq})


def snap_summary(route, shape, threshold, payload):
    messages, summary, previous, room = payload
    archive = snapcompact.compact(messages, summary, previous, route, shape, room)
    return archive_summary(archive, route, shape, threshold)


def archive_summary(archive, route, shape, threshold):
    lead, blocks, _ = snapcompact.history(archive, route, shape, threshold)
    counter = ContextCounter(route)
    cost = counter.text(lead) + counter.content(blocks)
    return CompactionSummary(lead, {"snapcompact": archive, "replacement_tokens": cost})


@dataclass(frozen=True)
class Speculation:
    method: str
    snapshot: Snapshot
    future: Future
    cancellation: handoff.Cancellation | None = None


class Compaction:
    def __init__(
        self, session, window, percent, keep_recent, summarize, executor, methods=("native", "shake", "soft"),
        native_compact=None, on_context_change=None, prepare_handoff=None, context_projection=None, disable_inline=None,
        on_event=None, thinking_route=None, async_enabled=True, auto_enabled=True, mid_turn_enabled=True,
        budget=None, context_windows=None, supersede_reads=True, drop_useless=True, pruning_route=None, tool_schemas=None,
    ):
        self.session = session
        self.window = window
        self.budget = budget or Budget(percent=percent, keep_recent=keep_recent)
        self.auto_enabled = auto_enabled
        self.async_enabled = async_enabled
        self.mid_turn_enabled = mid_turn_enabled
        self.context_windows = context_windows or (lambda: False)
        self.supersede_reads, self.drop_useless = supersede_reads, drop_useless
        self.pruning_route = pruning_route or (lambda: None)
        self.context_reminder = None
        self.threshold = self.budget.trigger(window)
        self.summarize = summarize
        self.executor = executor
        self.methods = methods
        self.native_compact = native_compact
        self.prepare_handoff = prepare_handoff
        self.context_projection = context_projection
        self.tool_schemas = tool_schemas or (lambda: ())
        self.counter = ContextCounter(session.archive_route)
        self.thinking_route = thinking_route or (lambda: None)
        self.disable_inline = disable_inline
        self.on_context_change = on_context_change or (lambda: None)
        self.on_event = on_event
        self.pending = None
        self.native_speculation_failed = False
        self.last_cuts = {}
        self.measured = None
        self.provider_context = 0
        self.threshold_dead_end = None
        self.continuation_blocked = False
        self.last_commit_node = None

    def discard_pending(self):
        if self.pending:
            if self.pending.cancellation:
                self.pending.cancellation.cancel()
            self.pending.future.cancel()
            self.pending = None

    def change_model(self, window, summarize, native_compact=None, *, budget=None):
        self.window = window
        if budget is not None:
            self.budget = budget
        self.threshold = self.budget.trigger(window)
        self.summarize = summarize
        self.native_compact = native_compact
        self.counter = ContextCounter(self.session.archive_route)
        self.reset_context()

    def reset_context(self):
        self.on_context_change()
        self.discard_pending()
        self.last_cuts.clear()
        self.measured = None
        self.provider_context = 0
        self.threshold_dead_end = None
        self.continuation_blocked = False
        self.last_commit_node = None
        self.native_speculation_failed = False

    def observe(self, messages, usage):
        self.provider_context = 0
        if not usage:
            return
        prompt = getattr(usage, "context_prompt_tokens", None)
        if prompt is None:
            prompt = usage.prompt_tokens or 0
        if prompt:
            self.measured = (self.context_cost(messages), prompt)
        context = getattr(usage, "context_tokens", None)
        if context is None:
            context = getattr(usage, "total_tokens", None) or prompt + (getattr(usage, "completion_tokens", 0) or 0)
        self.provider_context = max(0, context)

    def context_cost(self, messages, native_tokens=None):
        return self.context_sizes(messages, native_tokens)[0]

    def context_sizes(self, messages, native_tokens, schemas=None):
        tools = self.counter.tools(self.tool_schemas() if schemas is None else schemas)
        stored = self.counter.messages(messages) + tools + (native_tokens or 0)
        if native_tokens is None and self.context_projection:
            projected = self.context_projection(messages)
            if projected is not messages:
                return self.counter.messages(projected) + tools, stored
        return stored, stored

    def messages(self, system, *args, **kwargs):
        kwargs.setdefault("include_native", True)
        kwargs.setdefault("include_archive", True)
        return self.session.messages(system, *args, thinking_route=self.thinking_route(), **kwargs)

    def estimate(self, messages, native_tokens=None, *, floor_stored=False, schemas=None):
        estimated, stored = self.context_sizes(messages, native_tokens, schemas)
        if self.measured:
            estimate_at_request, actual_at_request = self.measured
            estimated = max(estimated, actual_at_request + estimated - estimate_at_request)
        return max(estimated, stored) if floor_stored else estimated

    @property
    def prompt_budget(self):
        return self.budget.prompt_budget(self.window)

    def fits_prompt(self, system):
        return self.prompt_budget is None or self.estimate(self.messages(system), floor_stored=True) <= self.prompt_budget

    def cut_index(self, pending_from=None):
        if len(self.session.batches) < 2:
            return None
        budget = self.budget.recent_budget(self.window)
        batches = self.session.batches
        kept = 0
        cut = len(batches) - 1
        for index in range(cut, -1, -1):
            kept += self.counter.messages(batches[index][1])
            if kept > budget:
                break
            cut = index
        if pending_from is not None:
            cut = next((index for index, (seq, _) in enumerate(batches[:cut]) if seq >= pending_from), cut)
        return cut or None

    def pending_input_start(self):
        start = None
        for seq, batch in reversed(self.session.batches):
            for message in reversed(batch):
                if message["role"] == "assistant":
                    return start
                if message["role"] in ("user", "developer"):
                    start = seq
        return start

    def snapshot(self, estimated, pending_from=None):
        if (cut := self.cut_index(pending_from)) is None:
            return None
        discarded = [message for _, batch in self.session.batches[:cut] for message in batch]
        prefix = [portable_message(message) for message in without_useless(discarded)]
        payload = pack({"previous_summary": self.session.summary, "discarded_messages": prefix})
        return Snapshot(self.session.first_kept, self.session.batches[cut][0], estimated, payload,
                        self.session.navigation_epoch, self.session.leaf, pending_from)

    def shake_for_body_read_timeout(self, check=None, replay_prefixes=None):
        return self.shake(AUTO, tool_results_only=True, check=check, replay_prefixes=replay_prefixes)

    def prune_tool_results(self, *, cold_check=False):
        if not self.session.batches or self.context_windows():
            return
        stale = self.supersede_reads or self.drop_useless
        bulk = bool(not cold_check and self.auto_enabled and self.methods)
        if not (stale or bulk) or cold_check and (self.session.can_continue() or not self.session.cache_idle()):
            return
        route = self.pruning_route()
        model = (route.catalog_model or {}) if route else {}
        anthropic = route and route.api in ANTHROPIC_COMPACTION_APIS
        lookback = model.get("promptCacheLookback") or (20 if anthropic else None)
        if type(lookback) is not int or lookback <= 0:
            lookback = None
        prefix_bound = (model.get("thinking") or {}).get("prefixBinding") is True
        if anthropic and not prefix_bound:
            prefix_bound = any(
                part.get("type") in ("thinking", "redacted_thinking")
                for _, batch in self.session.batches for message in batch
                for part in (message.get("_anthropic") or {}).get("blocks", ())
            )
        warm_limit = 0 if prefix_bound else WARM_SUFFIX
        cancellation = active_turn()

        def apply(*, bulk=False, limit=warm_limit):
            edits, savings = prune_plan(
                self.session.batches, lambda message: tokens(portable_message(message)),
                supersede=self.supersede_reads and not bulk, drop_useless=self.drop_useless,
                bulk=bulk, warm_limit=limit, lookback=lookback,
            )
            if not edits:
                return
            timestamp = int(time.time() * 1000)
            replacements = {position: {"pruned_content": notice, "pruned_at": timestamp} for position, notice in edits.items()}
            self.session.archive_elisions(replacements, {}, check=cancellation.check if cancellation else None)
            self.on_context_change()
            self.discard_pending()
            self.last_cuts.clear()
            self.threshold_dead_end = None
            self.native_speculation_failed = False
            print(f"[pruning: {len(edits)} tool results, ~{savings} tokens reclaimed]", file=sys.stderr)

        with self.session.writer:
            try:
                if stale:
                    apply(limit=None if cold_check else warm_limit)
                if bulk:
                    apply(bulk=True)
            except SQLiteError:
                print("[pruning: could not persist tool result updates]", file=sys.stderr)

    def shake(self, preset, *, tool_results_only=False, check=None, replay_prefixes=None, before=None):
        if check:
            check()
        batches = self.session.batches
        if before is not None:
            batches = [(seq, batch) for seq, batch in batches if seq < before]
        replacements, artifacts, savings = plan(
            batches, tokens, preset, tool_results_only=tool_results_only,
            message_tokens=lambda message: tokens(portable_message(message)),
        )
        if not savings:
            return False
        elided = len(replacements)
        if replay_prefixes:
            for seq, batch in self.session.batches:
                for index, message in enumerate(batch):
                    if saved := message.get("_responses"):
                        if prefix := replay_prefixes.get((saved["route"], saved["prefix"])):
                            replacements[seq, index] = {"responses_prefix": prefix}
        try:
            self.session.archive_elisions(replacements, artifacts, check=check)
        except SQLiteError:
            print("[shake: could not save recovery artifacts]", file=sys.stderr)
            return False
        self.on_context_change()
        self.discard_pending()
        self.last_cuts.clear()
        self.native_speculation_failed = False
        self.provider_context = 0
        self.threshold_dead_end = None
        print(f"[shake: reclaimed ~{savings} tokens in {elided} messages]", file=sys.stderr)
        return True

    def maintain(self, system, reason="threshold", focus=None, *, methods=None, phase="pre", promote=None):
        if phase == "post" or reason in ("overflow", "incomplete"):
            self.last_commit_node = None
        if not self.auto_enabled and reason != "manual":
            return False
        pending_from = self.pending_input_start() if reason == "threshold" and phase == "pre" else None
        if methods is None and not focus and self.context_windows() and reason in (
            "manual", "threshold", "overflow", "incomplete", "idle",
        ):
            if reason == "threshold" and (self.threshold is None or phase == "mid" and not self.mid_turn_enabled):
                return False
            return self.rollover(system, reason, phase=phase, promote=promote, pending_from=pending_from)
        methods = self.methods if methods is None else methods
        if not methods:
            return False
        retrying = reason in ("overflow", "incomplete")
        if reason == "threshold" and (self.threshold is None or phase == "mid" and not self.mid_turn_enabled):
            return False
        estimated = self.estimate(self.messages(system), floor_stored=reason == "threshold")
        if reason == "threshold":
            if phase in ("mid", "post"):
                estimated = max(estimated, self.provider_context)
            if self.threshold_dead_end is not None:
                if self.threshold_dead_end == self.threshold_cut():
                    return False
                self.threshold_dead_end = None
            if estimated <= self.threshold:
                if estimated < self.threshold:
                    self.start_speculation(system, estimated, pending_from=pending_from)
                return False
            if self.start_speculation(system, estimated, pending_from=pending_from):
                return False
            if promote and promote():
                return False

        if self.pending:
            if reason in ("manual", "idle") or not self.pending.future.done():
                self.discard_pending()
            else:
                self.check_speculation(system, estimated, pending_from=pending_from)
        changed = False
        for index, method in enumerate(methods):
            if not self.method_usable(method, reason):
                continue
            if method == "native" and reason != "manual" and self.native_speculation_failed and any(
                self.method_usable(candidate, reason) for candidate in methods[index + 1:]
            ):
                continue
            if self.pending and self.pending.method != method:
                self.discard_pending()
            if method == "shake":
                reclaimed = self.maintenance_attempt(reason, method, partial(self.shake, AUTO, before=pending_from))
                if reason == "idle":
                    return reclaimed
                if reclaimed:
                    changed = True
                    if retrying and self.fits_prompt(system):
                        return True
                    if not retrying and (
                        self.threshold is None or self.estimate(self.messages(system)) <= self.threshold * 80 // 100
                    ):
                        return True
            else:
                recovery_band = self.prompt_budget if retrying else None
                if method == "snapcompact" and reason == "threshold" and self.threshold:
                    recovery_band = self.threshold * 80 // 100
                attempt = partial(
                    self.summarize_history, system, reason != "threshold", method, recovery_band,
                    pending_only=method == "handoff" and reason in ("overflow", "media"),
                    automatic=reason != "manual", focus=focus, pending_from=pending_from,
                )
                if self.maintenance_attempt(reason, method, attempt):
                    return True
        if reason in ("threshold", "overflow", "incomplete") and self.shrink_archive(system):
            if not retrying or self.fits_prompt(system):
                return True
            changed = True
        if retrying and "shake" in methods:
            changed = self.shake(RESCUE) or changed
        if retrying and (not changed or not self.fits_prompt(system)) and self.drop_images(system):
            if self.fits_prompt(system):
                return True
            return self.maintain(system, reason, focus, methods=methods, phase=phase, promote=promote)
        if reason == "threshold" and phase in ("mid", "post") and not changed:
            self.threshold_dead_end = self.threshold_cut()
        return changed and (not retrying or self.fits_prompt(system))

    def post_threshold_ready(self, system, schemas, changed):
        if self.threshold is None or not self.auto_enabled:
            return True
        messages = self.messages(system)
        estimate = self.estimate(messages, floor_stored=True)
        if not changed and (self.threshold_dead_end is None or self.pending
                            or max(estimate, self.provider_context) <= self.threshold):
            return True

        def residual():
            messages = self.messages(system)
            return self.estimate(messages, floor_stored=True, schemas=schemas)

        band = self.threshold * 80 // 100
        after = residual()
        cancellation = active_turn()
        check = cancellation.check if cancellation else None
        if after > band and not self.context_windows():
            for rescue in (partial(self.shrink_archive, system), partial(self.shake, RESCUE, check=check),
                           partial(self.drop_images, system)):
                if check:
                    check()
                if rescue():
                    after = residual()
                    if after <= band:
                        break
        warning = None
        if after > band:
            self.continuation_blocked = True
            self.threshold_dead_end = self.threshold_cut()
            warning = "Compaction could not create enough headroom. Automatic continuation was paused."
        self.session.update_compaction_metadata(self.last_commit_node, tokens_after=after, warning=warning, check=check)
        if warning:
            if self.on_event:
                self.on_event("notice", level="warning", source="compaction", message=warning)
            print(f"[{warning}]", file=sys.stderr)
        return warning is None

    def remind_context_notes(self, system):
        if not self.auto_enabled or self.threshold is None or not self.context_windows():
            return
        estimate = max(self.provider_context, self.estimate(self.messages(system), floor_stored=True))
        lead = min(32_000, max(8_192, self.threshold // 8))
        if 0 < self.threshold - estimate <= lead:
            boundary = self.session.context_boundary()
            if boundary != self.context_reminder:
                self.context_reminder = boundary
                self.session.append([{"role": "developer", "content": REMINDER, "_context_note_reminder": True}])

    def rollover(self, system, reason, *, phase="mid", promote=None, pending_from=None):
        if not self.window or not self.context_windows():
            return False
        with self.session.writer:
            before = self.messages(system)
            estimate = self.estimate(before, floor_stored=True)
            if phase in ("mid", "post"):
                estimate = max(estimate, self.provider_context)
            if reason == "threshold" and (estimate <= self.threshold or promote and promote()):
                return False
            cut = self.cut_index(pending_from)
            target = self.session.batches[cut][0] if cut is not None else None
            boundary = self.session.navigation_epoch, self.session.first_kept, target
            if reason == "threshold" and self.threshold_dead_end == boundary:
                return False
            if cut is None:
                if reason == "threshold":
                    self.threshold_dead_end = boundary
                return False
            snapshot = Snapshot(self.session.first_kept, target, estimate, "",
                                self.session.navigation_epoch, self.session.leaf, pending_from)
            result = CompactionSummary(BOUNDARY, {"method": ROLLOVER, "tokensBefore": estimate})
            limit = self.prompt_budget if reason in ("overflow", "incomplete") else None

            def commit_rollover():
                if cancellation := active_turn():
                    cancellation.check()
                return self.commit(system, before, snapshot, result, limit)

            changed = self.maintenance_attempt("threshold" if reason == "explicit" else reason, "soft", commit_rollover)
            if not changed and reason == "threshold":
                self.threshold_dead_end = boundary
            return changed

    def threshold_cut(self):
        index = self.cut_index()
        return self.session.navigation_epoch, self.session.first_kept, self.session.batches[index][0] if index else None

    def method_usable(self, method, reason):
        if reason == "media" and method == "snapcompact":
            return False
        if reason in ("overflow", "media") and method == "handoff":
            return bool(self.pending and self.pending.method == method and self.pending.future.done())
        return method == "shake" or self.available(method)

    def maintenance_attempt(self, reason, method, attempt):
        if not self.on_event or reason not in ("threshold", "overflow", "idle", "incomplete"):
            return attempt()
        action = {"native": "remote", "soft": "context-full"}.get(method, method)
        self.on_event("auto_compaction_start", reason=reason, action=action)
        try:
            changed = attempt()
        except BaseException as error:
            self.on_event("auto_compaction_end", action=action, aborted=isinstance(error, (KeyboardInterrupt, CancelledError)),
                          willRetry=False, errorMessage=str(error))
            raise
        self.on_event("auto_compaction_end", action=action, aborted=False, skipped=not changed,
                      willRetry=changed and reason in ("overflow", "incomplete"))
        return changed

    def available(self, method):
        if method == "native":
            return self.native_compact is not None
        if method == "snapcompact":
            return bool(
                self.session.archive_route and self.session.archive_route.vision and not self.session.archive_images_disabled
            )
        if method == "handoff":
            return self.prepare_handoff is not None
        return method == "soft"

    def handoff_request(self, system, snapshot, focus, automatic, cancellation):
        files = self.session.file_operations(snapshot.first_kept)
        options = {"before": snapshot.pending_from} if snapshot.pending_from is not None else {}
        request = self.prepare_handoff(system, focus, automatic, cancellation, **options)

        def run(_):
            document = request()
            return CompactionSummary(handoff.with_files(document, files), {"method": "handoff", "files": files})

        return run, None

    def handoff(self, system, focus=None):
        if not self.prepare_handoff:
            raise RuntimeError("Handoff requires an active model.")
        self.discard_pending()
        if sum(len(batch) for _, batch in self.session.batches) < 2:
            raise RuntimeError("Nothing to hand off (no messages yet).")
        before = self.messages(system)
        snapshot = self.snapshot(self.estimate(before))
        if not snapshot:
            raise RuntimeError("Nothing to hand off (already compacted or all messages are recent).")
        cancellation = handoff.Cancellation(active_turn())
        try:
            request, payload = self.handoff_request(system, snapshot, focus, False, cancellation)
            result = request(payload)
            if not result.text.strip():
                raise RuntimeError("Handoff generation produced no content.")
            if not self.commit(system, before, snapshot, result, require_reduction=False):
                raise RuntimeError("Handoff does not fit the current context window.")
        finally:
            cancellation.cancel()

    def recover_media(self, system):
        archive = self.session.disable_archive_images()
        inline = self.disable_inline() if self.disable_inline else False
        removed = self.drop_images(system)
        if (archive or inline) and not removed:
            self.on_context_change()
            self.discard_pending()
            self.last_cuts.clear()
            self.measured = None
            self.provider_context = 0
            self.threshold_dead_end = None
        if archive or inline or removed:
            print("[compaction: replaying context as text after a media rejection]", file=sys.stderr)
            if not self.window or self.estimate(self.messages(system)) < self.window:
                return True
        return self.maintain(system, reason="media")

    def drop_images(self, system=None):
        if system is not None and not any(graph.without_images(message)[1] for message in self.messages(system)):
            return 0
        cancellation = active_turn()
        removed = self.session.drop_images(cancellation.check if cancellation else None)
        if removed:
            node = self.last_commit_node
            self.reset_context()
            self.last_commit_node = node
            print(f"[compaction: removed {removed} image blocks from the active context]", file=sys.stderr)
        return removed

    def shrink_archive(self, system):
        archive = (self.session.native_summary or {}).get("snapcompact")
        route = self.session.archive_route
        if not archive or not archive["frames"] or not self.available("snapcompact"):
            return False
        messages = self.messages(system)
        cutoff = self.threshold * 80 // 100 if self.threshold else self.window
        fixed = self.estimate(messages, 0)
        if cutoff is not None and fixed >= cutoff:
            return False
        shape_name = self.session.archive_shape
        source = archive["text"] if snapcompact.includes_thinking(route) else snapcompact.without_thinking(archive["text"])
        shape = snapcompact.resolve_shape(route, shape_name, source)
        cap = len(archive["frames"]) - 1
        if cutoff is not None:
            cap = min(cap, snapcompact.frame_budget(route, shape, cutoff - fixed))
        current = self.session.first_kept
        snapshot = Snapshot(current, current, self.estimate(messages), "", self.session.navigation_epoch)
        for count in dict.fromkeys((cap, 0)):
            reduced = snapcompact.build_archive(source, shape, count, archive["truncated"], archive["files"])
            reduced["shape_reader"] = snapcompact.reader_shape_key(route, shape_name)
            result = archive_summary(reduced, route, shape_name, self.threshold)
            if self.commit(system, messages, snapshot, result, cutoff):
                return True
        return False

    def summary_request(self, method, system, snapshot, cancellation=None, automatic=True, focus=None, *, speculative=False):
        cancellation = cancellation or handoff.Cancellation(active_turn(), automatic=automatic)

        def with_cancel(request, payload):
            def run(value):
                with turn_scope(cancellation):
                    cancellation.check()
                    return request(value)
            return run, payload

        if method == "handoff":
            return self.handoff_request(
                system, snapshot, handoff.AUTO_FOCUS if automatic else focus, automatic,
                cancellation,
            )
        if method == "native":
            if focus:
                system += "\n\nAdditional compaction instructions: " + focus
            return with_cancel(self.native_request(system, snapshot, speculative), None)
        native = self.session.native_summary or {}
        prior, _, archived = self.session.archived_responses_prefix()
        archive = native.get("snapcompact") or native.get("portable_archive")
        if method == "snapcompact":
            prefix = [msg for _, batch in archived for msg in batch]
            prefix.extend(msg for seq, batch in self.session.batches if seq < snapshot.first_kept for msg in batch)
            fixed = self.context_cost(self.messages(system, "", snapshot.first_kept), 0)
            room = int(self.threshold * 0.6) - fixed if self.threshold else None
            return (
                partial(snap_summary, self.session.archive_route, self.session.archive_shape, self.threshold),
                (without_useless(prefix), prior, archive, room),
            )
        summarize = partial(self.summarize, focus=focus) if focus else self.summarize
        if archive or "items" in native:
            payload = json.loads(snapshot.transcript)
            payload["previous_summary"] = snapcompact.plain(archive["text"]) if archive else prior
            payload["discarded_messages"][:0] = [
                portable_message(message) for message in without_useless([msg for _, batch in archived for msg in batch])
            ]
            return with_cancel(summarize, pack(payload))
        return with_cancel(summarize, snapshot.transcript)

    def native_request(self, system, snapshot, speculative):
        route = self.session.archive_route
        batches = self.session.batches
        cut, pending_from = snapshot.first_kept, snapshot.pending_from
        anthropic = route and route.api in ANTHROPIC_COMPACTION_APIS
        files_due_before = float("inf")
        if anthropic:
            # Anthropic merges adjacent messages on the same side of a turn.
            # Move past that run so the retained signed assistant sees the same wire boundary.
            end = pending_from if pending_from is not None else batches[-1][0] + 1
            cut = next((right_seq for (_, left), (right_seq, right) in zip(batches, batches[1:])
                        if right_seq >= cut and right[0]["role"] not in ("developer", "system")
                        and (left[-1]["role"] == "assistant") != (right[0]["role"] == "assistant")),
                       end)
            cut = min(cut, end)
            if first := next((batch[0] for seq, batch in batches if seq >= cut), None):
                last = next((message for seq, batch in reversed(batches) if seq < cut
                             for message in reversed(batch) if message["role"] != "tool"), None)
                boundary = first if first["role"] == "assistant" else last
                files_due_before = boundary.get("_history_at", float("inf")) if boundary else float("-inf")
        responses = route and route.api in RESPONSES_APIS
        prefix = self.session.messages(system, include_native=True, include_archive=True, before=cut,
                                       for_compaction=responses)
        covered = next(seq for seq, _ in reversed(batches) if seq < cut)
        whole = None
        if responses:
            whole = self.session.messages(system, include_native=True, include_archive=True, before=pending_from,
                                          for_compaction=True)
        last = next(seq for seq, _ in reversed(batches) if pending_from is None or seq < pending_from)
        compact = self.native_compact
        session_id = self.session.transcript_id
        if speculative:
            session_id = f"{session_id}:spec:{uuid4().hex[:16]}"

        def run(_):
            if whole is not None:
                result = compact(whole, prefix=(covered, prefix), session_id=session_id, speculative=speculative)
                return result if "replay_through" in result.native else result.through(last)
            result = compact(prefix, files_due_before=files_due_before) if anthropic else compact(prefix)
            return result.through(covered) if isinstance(result, CompactionSummary) and result.native else result

        return run

    def start_speculation(self, system, estimated, *, pending_from=None):
        """Start a side request if useful and return whether threshold maintenance should defer."""
        if self.threshold is None or not self.auto_enabled:
            return False
        if not self.async_enabled:
            self.discard_pending()
            return False
        invalidated = self.check_speculation(system, estimated, pending_from=pending_from, refresh=estimated < self.threshold)
        if invalidated and estimated > self.threshold:
            return False
        method = next((item for item in self.methods if self.method_usable(item, "threshold")
                       and not (item == "native" and self.native_speculation_failed)), None)
        if method not in ("native", "handoff", "soft"):
            return False
        lead = min(32_000, max(8192, self.threshold // 8))
        above = estimated > self.threshold
        cap = min(self.threshold + lead, self.window - 8192)
        if estimated < self.threshold - lead or (above and estimated >= cap):
            return False
        if self.pending:
            return above and not self.pending.future.done()
        snapshot = self.snapshot(estimated, pending_from)
        if not snapshot or snapshot.first_kept == self.last_cuts.get(method):
            return False
        cancellation = handoff.Cancellation(automatic=True)
        try:
            request, payload = self.summary_request(method, system, snapshot, cancellation, speculative=True)
            self.pending = Speculation(method, snapshot, self.executor.submit(request, payload), cancellation)
        except Exception as error:
            cancellation.cancel()
            print(f"[compaction: could not prepare background request ({type(error).__name__})]", file=sys.stderr)
            return False
        print("[compaction: background summary started]", file=sys.stderr)
        return above

    def check_speculation(self, system, estimated, *, pending_from=None, refresh=False):
        """Return true when an armed result was discarded and needs a fresh cut."""
        if not self.pending:
            return False
        pending = self.pending
        snapshot, future = pending.snapshot, pending.future
        if (not self.async_enabled or snapshot.navigation_epoch != self.session.navigation_epoch
            or snapshot.boundary != self.session.first_kept or future.cancelled()):
            self.discard_pending()
            return False
        if not future.done():
            return False
        if error := future.exception():
            if pending.method == "native":
                transient = isinstance(error, (CancelledError, StreamInterrupted, *TRANSPORT_ERRORS))
                if isinstance(error, APIStatusError):
                    transient = error.status_code >= 500 or error.status_code in (408, 409, 425, 429)
                self.native_speculation_failed = not transient
            print(f"[compaction: background request failed ({type(error).__name__})]", file=sys.stderr)
            self.discard_pending()
        else:
            result = future.result()
            summary = CompactionSummary(result) if isinstance(result, str) else result
            native = summary.native if isinstance(summary, CompactionSummary) else None
            grew = refresh and estimated - snapshot.context_tokens > max(self.budget.keep_recent, 8192)
            band = self.armed_recovery_band(snapshot, summary)
            if (grew and not (native and native.get("block"))) or not self.compaction_costs(
                system, self.messages(system), snapshot, summary, band, True, pending_from=pending_from, armed=True,
            ):
                self.discard_pending()
                return True
        return False

    def armed_recovery_band(self, snapshot, summary, default=None):
        native = summary.native if isinstance(summary, CompactionSummary) else None
        if self.threshold and self.session.leaf != snapshot.leaf and not (native and native.get("block")):
            band = self.threshold * 80 // 100
            return min(default, band) if default is not None else band
        return default

    def summarize_history(self, system, force, method, recovery_band=None, *, pending_only=False, automatic=True,
                          focus=None, pending_from=None):
        messages = self.messages(system)
        estimated = self.estimate(messages)
        armed = self.pending if self.pending and self.pending.method == method else None
        if pending_only and not armed:
            return False
        try:
            if armed:
                self.pending = None
                result = armed.future.result()
                band = self.armed_recovery_band(armed.snapshot, result, recovery_band)
                if self.commit(system, messages, armed.snapshot, result, band, pending_from=pending_from, armed=True):
                    return True
            if pending_only:
                return False
            snapshot = self.snapshot(estimated, pending_from)
            if not snapshot or not force and snapshot.first_kept == self.last_cuts.get(method):
                return False
            self.last_cuts[method] = snapshot.first_kept
            request, payload = self.summary_request(method, system, snapshot, automatic=automatic, focus=focus)
            return self.commit(system, messages, snapshot, request(payload), recovery_band)
        except CancelledError:
            self.last_cuts.pop(method, None)
            raise
        except Exception as error:
            print(f"[compaction: summary request failed ({type(error).__name__})]", file=sys.stderr)
            return False

    def compaction_costs(self, system, before, snapshot, summary, recovery_band, require_reduction, *,
                         pending_from=None, armed=False):
        if (
            not isinstance(summary, CompactionSummary) or not summary.text.strip() or snapshot.boundary != self.session.first_kept
            or snapshot.navigation_epoch != self.session.navigation_epoch
        ):
            return None
        native = summary.native or {}
        if armed and (route := self.session.archive_route):
            if "items" in native and not responses_replay_matches(native, route):
                return None
            if "block" in native and (route.api not in ANTHROPIC_COMPACTION_APIS or not route.native_compaction
                    or not (native.get("route") == route.identity() or native.get("anthropic_provider") == native_provider(route))):
                return None
        cut = native["replay_through"] + 1 if "replay_through" in native else snapshot.first_kept
        if any(boundary is not None and cut > boundary for boundary in (snapshot.pending_from, pending_from)):
            return None
        if snapshot.leaf is not None and self.session.leaf != snapshot.leaf:
            with database(self.session.path) as db:
                if not graph.intact_compaction_prefix(db, self.session.id, self.session.leaf, snapshot.leaf, cut):
                    return None
        signed = self.counter.anthropic_snapshot(native) if "block" in native else None
        external_history = "items" in native or "snapcompact" in native or signed is not None
        after = self.messages(system, "" if external_history else summary.text, cut, method=native.get("method"))
        extra_tokens = self.counter.responses(native["items"]) if "items" in native else (
            signed if signed is not None else native.get("replacement_tokens", 0)
        )
        estimated, stored = self.context_sizes(before, None)
        before_tokens = stored if armed else estimated
        after_tokens = self.context_sizes(after, extra_tokens)[1]
        projected = after_tokens if armed else self.estimate(after, extra_tokens)
        if (require_reduction and after_tokens >= before_tokens) or (self.window and projected >= self.window):
            return None
        if recovery_band is not None and projected > recovery_band:
            return None
        return cut, before_tokens, after_tokens

    def commit(self, system, before, snapshot, result, recovery_band=None, *, require_reduction=True,
               pending_from=None, armed=False):
        summary = CompactionSummary(result) if isinstance(result, str) else result
        cancellation = active_turn()
        check = cancellation.check if cancellation else None
        with self.session.writer:
            if check:
                check()
            costs = self.compaction_costs(system, before, snapshot, summary, recovery_band, require_reduction,
                                          pending_from=pending_from, armed=armed)
            if costs is None:
                return False
            cut, before_tokens, after_tokens = costs
            self.on_context_change()
            metadata = {"tokensBefore": snapshot.context_tokens, "tokensAfter": after_tokens, **(summary.native or {})}
            self.session.compact(cut, summary.text, metadata, check=check)
            self.last_commit_node = self.session.leaf
        self.discard_pending()
        self.last_cuts.clear()
        self.measured = None
        self.provider_context = 0
        self.threshold_dead_end = None
        self.native_speculation_failed = False
        print(f"[compaction: {before_tokens} → {after_tokens} estimated tokens]", file=sys.stderr)
        return True
