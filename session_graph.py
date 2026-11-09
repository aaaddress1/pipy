"""Persistent per-session ancestry for batches and context rewrites."""

import json
import re
from pathlib import Path
from uuid import NAMESPACE_URL, uuid4, uuid5

from memory_transcript import user_turns
from session_exit import pending_result, placeholder


SCHEMA = """
CREATE TABLE IF NOT EXISTS session_heads (
    session TEXT PRIMARY KEY, leaf INTEGER
);
CREATE TABLE IF NOT EXISTS session_nodes (
    id INTEGER PRIMARY KEY, session TEXT NOT NULL, parent INTEGER,
    kind TEXT NOT NULL, batch INTEGER, payload TEXT
);
CREATE INDEX IF NOT EXISTS session_nodes_session ON session_nodes(session, id);
CREATE TABLE IF NOT EXISTS session_labels (
    session TEXT NOT NULL, node INTEGER NOT NULL, label TEXT NOT NULL,
    PRIMARY KEY(session, node)
);
CREATE TABLE IF NOT EXISTS session_titles (
    session TEXT PRIMARY KEY, title TEXT NOT NULL, source TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS session_origins (
    session TEXT PRIMARY KEY, parent TEXT NOT NULL, prompt_cache_key TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS session_imports (
    session TEXT PRIMARY KEY, header TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS session_entry_sources (
    session TEXT NOT NULL, node INTEGER NOT NULL, source_id TEXT NOT NULL, metadata TEXT NOT NULL,
    PRIMARY KEY(session, node), UNIQUE(session, source_id)
);
CREATE TABLE IF NOT EXISTS session_usage_attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT, session TEXT NOT NULL, parent INTEGER, payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS session_usage_attempts_session ON session_usage_attempts(session, id);
CREATE TABLE IF NOT EXISTS subagent_links (
    owner TEXT NOT NULL, name TEXT COLLATE NOCASE NOT NULL, child TEXT NOT NULL, result TEXT,
    PRIMARY KEY(owner, name)
);
CREATE INDEX IF NOT EXISTS subagent_links_child ON subagent_links(child);
CREATE TABLE IF NOT EXISTS subagent_specs (child TEXT PRIMARY KEY, data TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS subagent_workpool_batches (child TEXT PRIMARY KEY, job TEXT NOT NULL, task TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS subagent_tombstones (child TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS subagent_deliveries (
    id INTEGER PRIMARY KEY AUTOINCREMENT, owner TEXT NOT NULL, child TEXT NOT NULL,
    summary TEXT NOT NULL, report TEXT NOT NULL, settled_leaf INTEGER, consumed INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS subagent_deliveries_owner ON subagent_deliveries(owner, consumed, id);
CREATE INDEX IF NOT EXISTS subagent_deliveries_child ON subagent_deliveries(child, settled_leaf);
CREATE TABLE IF NOT EXISTS speculative_task_roots (session TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS peer_messages (
    seq INTEGER PRIMARY KEY AUTOINCREMENT, root TEXT NOT NULL, sender TEXT NOT NULL, recipient TEXT NOT NULL,
    sender_name TEXT NOT NULL, body TEXT NOT NULL, flags INTEGER NOT NULL, consumed INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS peer_messages_recipient ON peer_messages(root,recipient,consumed,seq);
CREATE INDEX IF NOT EXISTS peer_messages_sender ON peer_messages(root,sender,seq);
"""
SPECULATIVE_TASK_TREE = """WITH RECURSIVE speculative_tasks(session) AS (
    SELECT session FROM speculative_task_roots
    UNION ALL
    SELECT l.child FROM subagent_links l JOIN speculative_tasks s ON l.owner=s.session
)
"""
JOURNAL_TABLES = (
    "batches", "artifacts", "context_elisions", "native_compactions", "model_switches", "session_recaps",
    "session_heads", "session_nodes", "session_labels", "session_titles", "session_origins", "session_imports",
    "session_entry_sources", "session_usage_attempts",
)


def task_subtree(db, root):
    return [sid for sid, in db.execute("""
        WITH RECURSIVE subtree(sid,depth) AS (
            SELECT ?,0 UNION ALL SELECT l.child,s.depth+1 FROM subagent_links l JOIN subtree s ON l.owner=s.sid
        ) SELECT sid FROM subtree ORDER BY depth DESC
    """, (root,))]


def delete_task_subtree(db, children):
    for table in (*JOURNAL_TABLES, "extractions", "memory_rollouts", "speculative_task_roots"):
        db.executemany(f"DELETE FROM {table} WHERE session = ?", ((sid,) for sid in children))
    db.executemany("DELETE FROM peer_messages WHERE sender = ? OR recipient = ?", ((sid, sid) for sid in children))
    db.executemany("DELETE FROM subagent_specs WHERE child = ?", ((sid,) for sid in children))
    db.executemany("DELETE FROM subagent_workpool_batches WHERE child = ?", ((sid,) for sid in children))
    db.executemany("DELETE FROM subagent_tombstones WHERE child = ?", ((sid,) for sid in children))
    db.executemany("DELETE FROM subagent_deliveries WHERE owner = ? OR child = ?", ((sid, sid) for sid in children))
    db.executemany("DELETE FROM subagent_links WHERE owner = ? OR child = ?", ((sid, sid) for sid in children))
    db.executemany("DELETE FROM sessions WHERE id = ?", ((sid,) for sid in children))


def drop_journal(db, session):
    for table in JOURNAL_TABLES:
        db.execute(f"DELETE FROM {table} WHERE session = ?", (session,))
    db.execute("DELETE FROM subagent_deliveries WHERE owner=? OR child=?", (session, session))
    db.execute("DELETE FROM subagent_workpool_batches WHERE child=?", (session,))
    db.execute("DELETE FROM sessions WHERE id = ?", (session,))


PROMPTS = Path(__file__).with_name("prompts")
SUMMARY_CONTEXT = (PROMPTS / "branch-summary-context.md").read_text(encoding="utf-8").strip()
SUMMARY_PROMPT = "\n\n".join(
    (PROMPTS / name).read_text(encoding="utf-8").strip()
    for name in ("branch-summary-preamble.md", "branch-summary.md")
)

ANCESTORS = """
WITH RECURSIVE ancestry(id, parent) AS (
    SELECT id, parent FROM session_nodes WHERE id = ? AND session = ?
    UNION ALL
    SELECT n.id, n.parent FROM session_nodes n JOIN ancestry a ON n.id = a.parent
    WHERE n.session = ?
)
"""


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def append(db, session, parent, kind, batch=None, payload=None):
    return db.execute(
        "INSERT INTO session_nodes(session,parent,kind,batch,payload) VALUES (?,?,?,?,?)",
        (session, parent, kind, batch, encode(payload) if payload is not None else None),
    ).lastrowid


def set_head(db, session, leaf):
    db.execute("INSERT OR REPLACE INTO session_heads VALUES (?,?)", (session, leaf))


def transfer_details(value):
    if not isinstance(value, dict) or not isinstance(value.get("kind"), str):
        return None
    fields = {"arrived": ("fromCwd", "cwd"), "departed": ("to",), "forked": ("source", "to")}.get(value["kind"])
    if fields and all(isinstance(value.get(key), str) for key in fields):
        return {key: value[key] for key in ("kind", *fields)}
    return None


def transfer_message(kind, **fields):
    templates = {
        "arrived": "This session was moved here from another machine. It ran in `{fromCwd}` there; "
                   "its working directory is now `{cwd}`. Paths from earlier turns under `{fromCwd}` now live under "
                   "`{cwd}`. Running processes, shell state, and files outside the working directory did not move with it.",
        "departed": "This session was moved to {to} and continues there. This copy stops here.",
        "forked": "This session had moved to {to} when it was resumed here, so it continues as a fork of the stale local "
                  "copy `{source}`. Work done on {to} after the move is not part of this conversation.",
    }
    return {"role": "developer", "content": templates[kind].format_map(fields),
            "_session_transfer": {"kind": kind, **fields}}


def batch_payload(messages, timestamp=None):
    payload = {"memoryUserTurns": user_turns(messages)}
    if any("_memory_recall_changes" in message for message in messages):
        payload["memoryRecallChanges"] = True
    if timestamp:
        payload["timestamp"] = timestamp
    for message in reversed(messages):
        if marker := transfer_details(message.get("_session_transfer")):
            payload["sessionTransfer"] = marker
            break
    return payload


def last_transfer(db, session):
    row = db.execute("""
        SELECT json_extract(payload, '$.sessionTransfer') FROM session_nodes
        WHERE session = ? AND kind = 'batch' AND json_type(payload, '$.sessionTransfer') = 'object'
        ORDER BY id DESC LIMIT 1
    """, (session,)).fetchone()
    if row:
        return transfer_details(json.loads(row[0]))
    imported = db.execute("SELECT json_extract(header, '$._pipyTransferIndexed') FROM session_imports WHERE session = ?",
                          (session,)).fetchone()
    if imported is None or imported[0]:
        return None
    for raw, in db.execute("SELECT messages FROM batches WHERE session = ? AND instr(messages, ?) > 0 ORDER BY seq DESC",
                           (session, '"_session_transfer":')):
        payload = batch_payload(json.loads(raw))
        if payload:
            return payload["sessionTransfer"]
    return None


def initialize_session(db, session, cut, summary, native):
    row = db.execute("SELECT leaf FROM session_heads WHERE session = ?", (session,)).fetchone()
    if row is not None:
        return row[0]
    leaf = None
    for seq, in db.execute("SELECT seq FROM batches WHERE session = ? ORDER BY seq", (session,)):
        leaf = append(db, session, leaf, "batch", seq)
    edits = db.execute(
        "SELECT seq,message_index,content FROM context_elisions WHERE session = ? ORDER BY seq,message_index", (session,),
    ).fetchall()
    if edits:
        leaf = append(db, session, leaf, "elision", payload=edits)
    if cut or summary or native:
        leaf = append(db, session, leaf, "compact", payload=[cut, summary, native])
    route = db.execute("""
        SELECT primary_route, active_route FROM model_switches WHERE session = ? ORDER BY seq DESC LIMIT 1
    """, (session,)).fetchone()
    if route:
        leaf = append(db, session, leaf, "route", payload=route)
    set_head(db, session, leaf)
    return leaf


def ancestor_ids(db, session, leaf):
    return {identifier for identifier, in db.execute(ANCESTORS + "SELECT id FROM ancestry", (leaf, session, session))}


def nodes(db, session, leaf, after=0):
    return db.execute(ANCESTORS + """
        SELECT n.id,n.parent,n.kind,n.batch,n.payload
        FROM ancestry a JOIN session_nodes n ON n.id = a.id WHERE n.id > ? ORDER BY n.id
    """, (leaf, session, session, after)).fetchall()


def intact_compaction_prefix(db, session, leaf, snapshot_leaf, cut):
    found, rewritten = db.execute("""
        WITH RECURSIVE suffix(id,parent,kind,payload) AS (
            SELECT id,parent,kind,payload FROM session_nodes WHERE id=? AND session=?
            UNION ALL
            SELECT n.id,n.parent,n.kind,n.payload FROM session_nodes n JOIN suffix s ON n.id=s.parent
            WHERE n.session=? AND n.id>=?
        )
        SELECT coalesce(max(n.id = ?),0), coalesce(max(n.id > ? AND (
            n.kind IN ('compact','reset') OR
            (n.kind='tool_update' AND json_extract(n.payload,'$[0]') < ?) OR
            (n.kind='tool_retry' AND json_extract(n.payload,'$') < ?) OR
            (n.kind='elision' AND EXISTS (
                SELECT 1 FROM json_each(CASE WHEN n.kind='elision' THEN n.payload ELSE '[]' END) e
                WHERE json_extract(e.value,'$[0]') < ?
            ))
        )),0) FROM suffix n
    """, (leaf, session, session, snapshot_leaf, snapshot_leaf, snapshot_leaf, cut, cut, cut)).fetchone()
    return bool(found and not rewritten)


def without_images(message):
    """Strip projected image blocks while preserving text, tool details and pairing."""
    if message.get("role") not in ("user", "developer", "tool"):
        return message, 0
    content = message.get("content")
    details = message.get("_tool_details")
    image_types = ("image", "image_url", "input_image")

    def keep(parts):
        return [part for part in parts if not isinstance(part, dict) or part.get("type") not in image_types]

    changes, removed = {}, 0
    if isinstance(content, list):
        kept = keep(content)
        if len(kept) != len(content):
            removed += len(content) - len(kept)
            changes["content"] = kept or [{"type": "text", "text": "[image removed]"}]
    if isinstance(details, dict) and isinstance(images := details.get("images"), list):
        kept = keep(images)
        if len(kept) != len(images):
            removed += len(images) - len(kept)
            changes["_tool_details"] = {**details, "images": kept}
    return ({**message, **changes}, removed) if removed else (message, 0)


def apply_elision(message, edit):
    if isinstance(edit, dict):
        if edit.get("drop_images") is True:
            return without_images(message)[0]
        if "pruned_at" in edit:
            return {**message, "content": edit["pruned_content"], "_tool_pruned_at": edit["pruned_at"]}
        if saved := message.get("_responses"):
            return {**message, "_responses": {**saved, "prefix": edit["responses_prefix"]}}
        return message
    return {**message, "content": edit}


def stored_batches(db, session, leaf, start=0, before=None, *, apply_elisions=True):
    rows = db.execute(ANCESTORS + """
        SELECT b.seq,b.messages FROM ancestry a
        JOIN session_nodes n ON n.id = a.id JOIN batches b ON n.batch = b.seq
        WHERE b.session = ? AND b.seq >= ? AND (? IS NULL OR b.seq < ?) ORDER BY b.seq
    """, (leaf, session, session, session, start, before, before))
    batches = [(seq, json.loads(raw)) for seq, raw in rows]
    by_seq = dict(batches)
    for kind, data in db.execute(ANCESTORS + """
        SELECT n.kind,n.payload FROM ancestry a JOIN session_nodes n ON n.id = a.id
        WHERE (n.kind = 'tool_update' AND json_extract(n.payload,'$[0]') >= ?
            AND (? IS NULL OR json_extract(n.payload,'$[0]') < ?)) OR n.kind = 'tool_retry'
            OR (? AND n.kind = 'elision') ORDER BY n.id
    """, (leaf, session, session, start, before, before, apply_elisions)):
        if kind == "tool_update":
            seq, index, message = json.loads(data)
            if seq in by_seq and 0 <= index <= len(by_seq[seq]):
                by_seq[seq][index:index + 1] = [message]
        elif kind == "tool_retry":
            if (batch := by_seq.get(json.loads(data))) is not None:
                batch[1:] = map(placeholder, batch[0]["tool_calls"])
        else:
            for seq, index, edit in json.loads(data):
                if seq in by_seq and 0 <= index < len(by_seq[seq]):
                    by_seq[seq][index] = apply_elision(by_seq[seq][index], edit)
    return batches


def active_raw_batches(db, session, before=None):
    row = db.execute("SELECT leaf FROM session_heads WHERE session = ?", (session,)).fetchone()
    if row is not None:
        return stored_batches(db, session, row[0], before=before, apply_elisions=False)
    return [
        (seq, json.loads(raw)) for seq, raw in db.execute(
            "SELECT seq,messages FROM batches WHERE session = ? AND (? IS NULL OR seq < ?) ORDER BY seq",
            (session, before, before),
        )
    ]


def repair_fork_tail(db, session, leaf):
    start = checkpoint(db, session, leaf)[0]
    answered = set()
    for seq, batch in reversed(stored_batches(db, session, leaf, start, apply_elisions=False)):
        for message in reversed(batch):
            if message["role"] == "tool" and not pending_result(message):
                answered.add(message.get("tool_call_id"))
            if message["role"] != "assistant":
                continue
            missing = [call["id"] for call in message.get("tool_calls", ()) if call["id"] not in answered]
            if missing:
                limited = message.get("_tool_stop") == "length"
                content = ("Tool call was not executed because the assistant reached its output limit. "
                           "Split the work into smaller tool calls." if limited else
                           "The parent session may still be running this tool; its outcome is unknown.")
                positions = {item.get("tool_call_id"): index for index, item in enumerate(batch) if pending_result(item)}
                next_index = len(batch)
                for call_id in missing:
                    result = {"role": "tool", "tool_call_id": call_id, "content": content, "_tool_error": True}
                    if limited:
                        result["_tool_details"] = {"__synthetic": True, "source": "assistant_stop_length", "executed": False}
                    position = positions.get(call_id, next_index)
                    if position == next_index:
                        next_index += 1
                    leaf = append(db, session, leaf, "tool_update", payload=[seq, position, result])
            return leaf
    return leaf


def checkpoint(db, session, leaf):
    row = db.execute(ANCESTORS + """
        SELECT n.kind,n.payload FROM ancestry a JOIN session_nodes n ON n.id = a.id
        WHERE n.kind IN ('compact','reset') ORDER BY n.id DESC LIMIT 1
    """, (leaf, session, session)).fetchone()
    if not row:
        return 0, "", None
    kind, payload = row
    return (json.loads(payload) + 1, "", None) if kind == "reset" else json.loads(payload)


def has_late_tool_results(db, session, leaf, covered):
    return db.execute(ANCESTORS + """
        , path AS (SELECT n.id,n.kind,n.payload FROM ancestry a JOIN session_nodes n ON n.id=a.id)
        SELECT 1 FROM path WHERE kind='tool_update' AND json_extract(payload,'$[0]')<=?
        AND id > (SELECT max(id) FROM path WHERE kind='compact') LIMIT 1
    """, (leaf, session, session, covered)).fetchone() is not None


def transcript_boundary(db, session, leaf):
    row = db.execute(ANCESTORS + """
        SELECT n.id,n.payload FROM ancestry a JOIN session_nodes n ON n.id=a.id
        WHERE n.kind='reset' ORDER BY n.id DESC LIMIT 1
    """, (leaf, session, session)).fetchone()
    return (row[0], json.loads(row[1]) + 1) if row else (None, 0)


def last_batch(db, session, leaf):
    return db.execute(ANCESTORS + """
        SELECT coalesce(max(n.batch),0) FROM ancestry a JOIN session_nodes n ON n.id=a.id
    """, (leaf, session, session)).fetchone()[0]


def first_batch_at(db, session, leaf, cut):
    row = db.execute(ANCESTORS + """
        SELECT MIN(n.batch) FROM ancestry a JOIN session_nodes n ON n.id = a.id
        WHERE n.batch >= ?
    """, (leaf, session, session, cut)).fetchone()
    return row[0]


def last_route(db, session, leaf, primary):
    row = db.execute(ANCESTORS + """
        SELECT n.payload FROM ancestry a JOIN session_nodes n ON n.id = a.id
        WHERE n.kind = 'route' ORDER BY n.id DESC LIMIT 1
    """, (leaf, session, session)).fetchone()
    if row:
        previous_primary, active = json.loads(row[0])
        if previous_primary == primary:
            return active
    return None


def cache_key(db, session, boundary=None):
    if boundary is not None:
        return str(uuid5(NAMESPACE_URL, f"mini-pi:reset:{session}:{boundary}"))
    row = db.execute("SELECT prompt_cache_key FROM session_origins WHERE session = ?", (session,)).fetchone()
    return row[0] if row else session


def fork(db, source, target, leaf, workspace, timestamp, *, whole):
    if whole:
        entries = db.execute("""
            SELECT id,parent,kind,batch,payload FROM session_nodes WHERE session = ? ORDER BY id
        """, (source,)).fetchall()
    else:
        entries = nodes(db, source, leaf)
        if not entries:
            raise ValueError(f"Session entry {leaf} was not found.")
    db.execute("INSERT INTO sessions(id,workspace,updated) VALUES (?,?,?)", (target, workspace, timestamp))
    db.execute("INSERT INTO session_titles SELECT ?,title,source FROM session_titles WHERE session=?", (target, source))
    db.execute("INSERT INTO session_origins VALUES (?,?,?)", (target, source, cache_key(db, source) if whole else target))
    mapped, batches = {None: None}, {0: 0}

    def translate(cut, parent):
        if cut in batches:
            return batches[cut]
        first = first_batch_at(db, source, parent, cut)
        return batches[first] if first is not None else batches[last_batch(db, source, parent)] + 1

    for identifier, parent, kind, seq, data in entries:
        payload = json.loads(data) if data else None
        if kind == "batch":
            raw, = db.execute("SELECT messages FROM batches WHERE session = ? AND seq = ?", (source, seq)).fetchone()
            batches[seq] = db.execute("INSERT INTO batches(session,messages) VALUES (?,?)", (target, raw)).lastrowid
            seq = batches[seq]
        elif kind == "compact":
            cut, summary, native = payload
            if native and "portable_first_kept" in native:
                native["portable_first_kept"] = translate(native["portable_first_kept"], parent)
            if native and "display_first_kept" in native:
                native["display_first_kept"] = batches[native["display_first_kept"]]
            if native and "replay_through" in native:
                native["replay_through"] = batches[native["replay_through"]]
                cut = native["replay_through"] + 1
            else:
                cut = translate(cut, parent)
            payload = [cut, summary, native]
        elif kind == "reset":
            payload = batches[payload]
        elif kind == "elision":
            payload = [(batches[original], index, text) for original, index, text in payload if original in batches]
        elif kind == "tool_update":
            batch, index, message = payload
            payload = [batches[batch], index, message]
        elif kind == "tool_retry":
            payload = batches[payload]
        mapped[identifier] = append(db, target, mapped[parent], kind, seq, payload)
    for identifier, name in db.execute("SELECT node,label FROM session_labels WHERE session = ?", (source,)):
        if identifier in mapped:
            db.execute("INSERT INTO session_labels VALUES (?,?,?)", (target, mapped[identifier], name))
    db.execute("INSERT INTO artifacts SELECT ?,id,content FROM artifacts WHERE session = ?", (target, source))
    children = db.execute("SELECT name,child,result FROM subagent_links WHERE owner=?", (source,)).fetchall()
    for name, child, data in children:
        copied = str(uuid4())
        row = db.execute("""SELECT s.workspace,h.leaf FROM sessions s JOIN session_heads h ON h.session=s.id
                            WHERE s.id=?""", (child,)).fetchone()
        if row:
            child_workspace, child_leaf = row
            fork(db, child, copied, child_leaf, child_workspace, timestamp, whole=True)
        if data:
            result = json.loads(data)
            result["sessionId"] = copied
            data = encode(result)
        db.execute("INSERT INTO subagent_links(owner,name,child,result) VALUES (?,?,?,?)", (target, name, copied, data))
        db.execute("INSERT INTO subagent_tombstones SELECT ? FROM subagent_tombstones WHERE child=?", (copied, child))
        if spec := db.execute("SELECT data FROM subagent_specs WHERE child=?", (child,)).fetchone():
            saved = json.loads(spec[0])
            saved["sid"] = copied
            db.execute("INSERT INTO subagent_specs(child,data) VALUES (?,?)", (copied, encode(saved)))
    db.execute("INSERT INTO session_imports SELECT ?,header FROM session_imports WHERE session = ?", (target, source))
    for identifier, source_id, metadata in db.execute(
        "SELECT node,source_id,metadata FROM session_entry_sources WHERE session = ?", (source,),
    ):
        if identifier in mapped:
            db.execute("INSERT INTO session_entry_sources VALUES (?,?,?,?)", (target, mapped[identifier], source_id, metadata))
    attempts = db.execute("SELECT parent,payload FROM session_usage_attempts WHERE session = ? ORDER BY id", (source,)).fetchall()
    for parent, data in attempts:
        if parent in mapped:
            payload = json.loads(data)
            payload["_after_node"] = mapped.get(payload.get("_after_node"), mapped[parent]) or 0
            db.execute("INSERT INTO session_usage_attempts(session,parent,payload) VALUES (?,?,?)",
                       (target, mapped[parent], encode(payload)))
    new_leaf = mapped[leaf]
    set_head(db, target, new_leaf)
    projection(db, target, *checkpoint(db, target, new_leaf))
    return new_leaf


def projection(db, session, cut, summary, native):
    db.execute("UPDATE sessions SET first_kept = ?, summary = ? WHERE id = ?", (cut, summary, session))
    if native:
        db.execute("INSERT OR REPLACE INTO native_compactions VALUES (?,?)", (session, encode(native)))
    else:
        db.execute("DELETE FROM native_compactions WHERE session = ?", (session,))


def destination(db, session, target):
    if target is None:
        return None, None
    row = db.execute("""
        SELECT n.parent,b.messages FROM session_nodes n
        LEFT JOIN batches b ON b.seq = n.batch WHERE n.session = ? AND n.id = ?
    """, (session, target)).fetchone()
    if row is None:
        raise ValueError(f"Session entry {target} was not found.")
    parent, raw = row
    if raw:
        batch = json.loads(raw)
        if len(batch) == 1 and batch[0]["role"] == "user":
            content = batch[0]["content"]
            if isinstance(content, list):
                content = "\n".join(part["text"] for part in content if part["type"] == "text")
            return parent, content
    return target, None


def abandoned(db, session, source, target):
    common = ancestor_ids(db, session, target) & ancestor_ids(db, session, source)
    branch = nodes(db, session, source, after=max(common, default=0))
    first = next((seq for _, _, kind, seq, _ in branch if kind == "batch"), None)
    raw = dict(stored_batches(db, session, source, start=first)) if first is not None else {}
    abandoned_batches = {seq for _, _, kind, seq, _ in branch if kind == "batch"}
    output = []
    for _, _, kind, seq, payload in branch:
        if kind == "batch":
            output.extend(raw[seq])
        elif kind == "tool_update":
            batch, _, message = json.loads(payload)
            if batch not in abandoned_batches:
                output.append(message)
        elif kind == "compact":
            _, summary, metadata = json.loads(payload)
            if metadata and "snapcompact" in metadata:
                summary = metadata["snapcompact"]["text"]
            if summary and not (metadata and "items" in metadata):
                output.append({"role": "developer", "content": summary})
    return output


def tree(db, session, leaf):
    active = ancestor_ids(db, session, leaf)
    children = {}
    rows = db.execute("""
        SELECT n.id,n.parent,n.kind,substr(
            CASE WHEN n.kind = 'tool_update' THEN n.payload ELSE b.messages END, 1, 500
        ),l.label
        FROM session_nodes n LEFT JOIN batches b ON b.seq = n.batch
        LEFT JOIN session_labels l ON l.session = n.session AND l.node = n.id
        WHERE n.session = ? ORDER BY n.id
    """, (session,))
    for identifier, parent, kind, preview, label in rows:
        children.setdefault(parent, []).append((identifier, kind, preview, label))
    roots = sorted(children.get(None, ()), key=lambda item: (item[0] not in active, item[0]))
    stack = [(item, 0) for item in reversed(roots)]
    while stack:
        (identifier, kind, preview, label), depth = stack.pop()
        if preview:
            role = re.search(r'"role":"([^"]+)"', preview)
            content = re.search(r'"content":"((?:[^"\\]|\\.)*)', preview)
            kind = role[1] if role else kind
            preview = content[1][:110] if content else "(tool batch)"
        mark = ">" if identifier == leaf else "*" if identifier in active else " "
        tag = f" [{label}]" if label else ""
        yield f"{mark} {'  ' * min(depth, 16)}{identifier} {kind}{tag} {preview or ''}".rstrip()
        ordered = sorted(children.get(identifier, ()), key=lambda item: (item[0] not in active, item[0]))
        stack.extend((item, depth + 1) for item in reversed(ordered))
