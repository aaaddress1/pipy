"""Import and export oh-my-pi session journals without replaying tools."""

import gzip
import heapq
import json
import math
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

import session_graph as graph
from runtime import database, initialize, pack, responses_replacement_tokens
from session_artifacts import PIPY_ARTIFACT, copy_sidecar, install_sidecar, sidecar, stored_artifacts
from session_blobs import hydrate_images
from session_exit import pending_result, placeholder
from shake import artifact_id, block_ranges, tool_text
from snapcompact_inline import is_failed


VERSION = 3
TITLE_BYTES = 256
SHAKE_REFERENCE = re.compile(r"\[shaken ~[0-9]+ tokens — recover: artifact://([a-f0-9]{64})\]")


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def mapping(value):
    return value if isinstance(value, dict) else {}


def journal_time(value):
    if type(value) is int or type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return int((parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)).timestamp() * 1000)
        except (ValueError, OverflowError, OSError):
            pass
    return None


def title_slot(header):
    title = header.get("title") if isinstance(header.get("title"), str) else ""
    record = {"type": "title", "v": 1, "title": "", "updatedAt": header["timestamp"], "pad": ""}
    if header.get("titleSource") in ("auto", "user"):
        record["source"] = header["titleSource"]
    low, high = 0, len(title)
    while low < high:
        middle = (low + high + 1) // 2
        record["title"] = title[:middle]
        if len(pack(record).encode("utf-8")) + 1 <= TITLE_BYTES:
            low = middle
        else:
            high = middle - 1
    record["title"] = title[:low]
    record["pad"] = " " * (TITLE_BYTES - len(pack(record).encode("utf-8")) - 1)
    return record


def read_journal(path):
    with path.open("rb") as probe:
        compressed = probe.read(2) == b"\x1f\x8b"
    opener = gzip.open if compressed else open
    header, title, rows, skipped = None, {}, [], 0
    with opener(path, "rt", encoding="utf-8-sig") as source:
        for line in source:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except ValueError:
                skipped += 1
                continue
            if not isinstance(entry, dict) or not isinstance(entry.get("type"), str):
                skipped += 1
            elif entry["type"] == "title" and header is None:
                title = entry
            elif entry["type"] == "session" and header is None:
                header = entry
            elif header is None:
                raise ValueError("Session journal is missing its header.")
            else:
                rows.append(entry)
    if not header:
        raise ValueError("Session journal is missing its header.")
    version = header.get("version", 1)
    if type(version) is not int or not 1 <= version <= VERSION:
        raise ValueError(f"Unsupported session journal version {version!r}.")
    if isinstance(title.get("title"), str):
        header = {**header, "title": title["title"]}
        if title.get("source") in ("auto", "user"):
            header["titleSource"] = title["source"]
    if version == 1:
        for index, entry in enumerate(rows, 1):
            entry["id"] = f"legacy-{index}"
            entry["parentId"] = f"legacy-{index - 1}" if index > 1 else None
            first = entry.get("firstKeptEntryIndex")
            if entry["type"] == "compaction" and type(first) is int and 1 <= first <= len(rows):
                entry["firstKeptEntryId"] = f"legacy-{first}"
    return header, rows, skipped


def ordered_rows(rows):
    by_id = {}
    for index, entry in enumerate(rows):
        identifier = entry.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in by_id:
            raise ValueError("Session journal contains a missing or duplicate entry id.")
        by_id[identifier] = index
    children, roots, missing = defaultdict(list), [], 0
    for index, entry in enumerate(rows):
        parent = entry.get("parentId")
        if isinstance(parent, str) and parent in by_id:
            children[parent].append(index)
        else:
            if parent is not None:
                missing += 1
            entry["parentId"] = None
            roots.append(index)
    heapq.heapify(roots)
    order = []
    while roots:
        entry = rows[heapq.heappop(roots)]
        order.append(entry)
        for index in children[entry["id"]]:
            heapq.heappush(roots, index)
    if len(order) != len(rows):
        raise ValueError("Session journal contains a parent cycle.")
    return order, missing


def text_and_images(content):
    if isinstance(content, str):
        return content
    output = []
    for part in content if isinstance(content, list) else ():
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text" and isinstance(part.get("text"), str):
            output.append({"type": "text", "text": part["text"]})
        if part.get("type") == "image":
            mime, data = part.get("mimeType"), part.get("data")
            if mime in ("image/png", "image/jpeg", "image/gif", "image/webp") and isinstance(data, str) and not data.startswith("blob:"):
                output.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}})
            else:
                output.append({"type": "text", "text": "[Image data remains in the source session.]"})
    if any(block["type"] == "image_url" for block in output):
        return output
    return "\n".join(block["text"] for block in output)


def valid_elision(edit):
    if isinstance(edit, dict):
        if set(edit) == {"drop_images"}:
            return edit["drop_images"] is True
        if set(edit) == {"pruned_content", "pruned_at"}:
            return isinstance(edit["pruned_content"], str) and type(edit["pruned_at"]) is int and edit["pruned_at"] >= 0
        return set(edit) == {"responses_prefix"} and isinstance(edit["responses_prefix"], str)
    return isinstance(edit, str) or isinstance(edit, list) and all(
        isinstance(part, dict) and (
            part.get("type") == "text" and isinstance(part.get("text"), str)
            or part.get("type") == "image_url" and isinstance(part.get("image_url"), dict)
            and isinstance(part["image_url"].get("url"), str)
        ) for part in edit
    )


def source_anthropic_turn(message):
    if message.get("api") not in ("anthropic-messages", "bedrock-converse-stream"):
        return None
    content = message.get("content")
    if not isinstance(content, list) or not all(isinstance(item, dict) for item in content) or not all(
        isinstance(message.get(field), str) for field in ("provider", "model")
    ):
        return None
    saved = {key: message[key] for key in ("provider", "model", "content", "api")}
    if isinstance(stop := message.get("stopReason"), str):
        saved["stopReason"] = stop
    return saved


def import_message(entry):
    kind = entry["type"]
    if kind == "branch_summary":
        return {"role": "developer", "content": str(entry.get("summary") or ""), "_branch_summary": True}
    if kind == "custom_message":
        message = {"role": "developer", "content": text_and_images(entry.get("content"))}
        if entry.get("customType") in ("eager-todo-prelude", "mid-run-todo-nudge", "todo-reminder", "todo-error-reminder"):
            message["_todo_message"] = entry["customType"]
        if entry.get("customType") in ("goal-mode-context", "goal-continuation", "goal-budget-limit", "guided-goal-interview"):
            role = "developer" if entry["customType"] == "guided-goal-interview" else "user"
            return {**message, "role": role, "_goal_message": entry["customType"]}
        if entry.get("customType") == "thinking-loop-redirect":
            message["_thinking_loop_redirect"] = True
        if entry.get("customType") == "compaction-auto-continue":
            message["_compaction_continue"] = True
        if entry.get("customType") == "session-transfer" and (marker := graph.transfer_details(entry.get("details"))):
            message["_session_transfer"] = marker
        if entry.get("customType") == "memory-recall-changes":
            message["_memory_recall_changes"] = mapping(entry.get("details"))
        if entry.get("customType") == "async-result":
            message["_async_jobs"] = mapping(entry.get("details")) or {"jobs": []}
        if entry.get("customType") == "launch-completion":
            rows = mapping(entry.get("details")).get("daemons")
            if isinstance(rows, list):
                message["_service_runs"] = [row for row in rows if isinstance(row, dict) and isinstance(row.get("id"), str)]
        if entry.get("customType") == "irc:incoming":
            details = mapping(entry.get("details"))
            rows = details.get("messages")
            if isinstance(rows, list):
                message["_irc_messages"] = [row for row in rows if isinstance(row, dict) and type(row.get("id")) is int]
            if details.get("fromParent") is True:
                message["_parent_assignment"] = True
        if entry.get("customType") == "ttsr-injection":
            rules = mapping(entry.get("details")).get("rules")
            if isinstance(rules, list):
                message["_ttsr_rules"] = [name for name in rules if isinstance(name, str)]
                message["_ttsr_kind"] = mapping(entry.get("details")).get("kind", "prose")
        return message
    if kind != "message":
        return None
    message = entry.get("message")
    if not isinstance(message, dict):
        raise ValueError("Session journal contains an invalid message.")
    local = mapping(entry.get("miniPi")).get("message")
    role = message.get("role")
    expected = "tool" if role == "toolResult" else role
    if isinstance(local, dict) and local.get("role") == expected:
        if role == "assistant" and "_anthropic_import" not in local and (native := source_anthropic_turn(message)):
            local = {**local, "_anthropic_import": native}
        if role == "toolResult" and message.get("toolName") == "todo" and not message.get("isError"):
            details = local.get("_tool_details")
            if isinstance(details, dict) and details.get("op") != "view" and isinstance(details.get("phases"), list):
                return {**local, "_todo_snapshot": True}
        return local
    if role == "user":
        result = {"role": "user", "content": text_and_images(message.get("content"))}
        if message.get("liveSteered") is True:
            result["liveSteered"] = True
        if message.get("attribution") == "agent" and plain_text(result["content"]).startswith("[MCP notification] "):
            result["_mcp_notification"] = True
        return result
    if role == "toolResult":
        result = {
            "role": "tool", "tool_call_id": message.get("toolCallId", ""),
            "content": text_and_images(message.get("content")),
        }
        if isinstance(message.get("details"), dict):
            result["_tool_details"] = message["details"]
            if (message.get("toolName") == "todo" and not message.get("isError")
                    and message["details"].get("op") != "view" and isinstance(message["details"].get("phases"), list)):
                result["_todo_snapshot"] = True
        if type(message.get("isError")) is bool:
            result["_tool_error"] = message["isError"]
        if type(message.get("prunedAt")) in (int, float):
            result["_tool_pruned_at"] = message["prunedAt"]
        if message.get("useless") is True:
            result["_tool_useless"] = True
        return result
    if role == "assistant":
        content = message.get("content") or ()
        calls, reasoning = [], []
        for item in content if isinstance(content, list) else ():
            if not isinstance(item, dict):
                continue
            if item.get("type") == "toolCall" and isinstance(item.get("id"), str) and isinstance(item.get("name"), str):
                calls.append({
                    "id": item["id"], "type": "function",
                    "function": {"name": item["name"], "arguments": pack(item.get("arguments", {}))},
                })
            elif item.get("type") == "thinking" and isinstance(item.get("thinking"), str):
                reasoning.append(item["thinking"])
        result = {"role": "assistant", "content": plain_text(text_and_images(content)) or None}
        if calls:
            result["tool_calls"] = calls
        if reasoning:
            result["_thinking"] = "\n".join(reasoning)
        if native := source_anthropic_turn(message):
            result["_anthropic_import"] = native
        if message.get("stopReason") in ("aborted", "error", "length"):
            result["_rpc"] = {key: message[key] for key in ("api", "provider", "model", "stopReason", "errorMessage")
                              if key in message}
            if calls and message["stopReason"] == "length":
                result["_tool_stop"] = "length"
        if not calls and message.get("stopReason") == "stop" and mapping(message.get("stopDetails")).get("type") == "pause_turn":
            result["_stop_details"] = {"type": "pause_turn"}
        if isinstance(message.get("inputTransformations"), list):
            result["_input_transformations"] = message["inputTransformations"]
        return result
    if role in ("custom", "hookMessage", "developer"):
        return {"role": "developer", "content": text_and_images(message.get("content"))}
    return None


def plain_text(content):
    return content if isinstance(content, str) else "\n".join(
        block["text"] for block in content or () if block.get("type") == "text"
    )


def text_only(messages):
    output = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            text = "\n".join(
                block["text"] if block["type"] == "text" else "[Image omitted for text-only model.]"
                for block in content
            )
            message = {**message, "content": text}
        output.append(message)
    return output


def insert_batch(db, session, parent, batch):
    seq = db.execute("INSERT INTO batches(session,messages) VALUES (?,?)", (session, pack(batch))).lastrowid
    return graph.append(db, session, parent, "batch", seq, graph.batch_payload(batch)), seq


def source_metadata(entry):
    record = {key: entry[key] for key in ("timestamp", "runtime", "fromId") if key in entry}
    message = mapping(entry.get("message"))
    record["message"] = {
        key: message[key] for key in (
            "timestamp", "api", "provider", "model", "usage", "stopReason", "errorMessage", "toolName", "isError",
        ) if key in message
    }
    return record


RESPONSES_PRESERVE_KEYS = ("openaiRemoteCompaction", "openaiCompactionV2")
ANTHROPIC_PRESERVE_KEY = "anthropicCompaction"


def compaction_diagnostics(entry):
    result = {key: entry[key] for key in ("tokensBefore", "tokensAfter")
              if type(entry.get(key)) is int and entry[key] >= 0}
    if isinstance(warning := entry.get("warning"), str) and warning.strip():
        result["warning"] = warning
    return result


def covered_batch(db, session, parent, replay_node):
    if replay_node is None:
        return None
    path = graph.nodes(db, session, parent)
    if not any(node == replay_node for node, _, _, _, _ in path):
        return None
    start = graph.transcript_boundary(db, session, parent)[1]
    covered = max((seq for node, _, kind, seq, _ in path if node <= replay_node and kind == "batch"), default=0)
    if not covered or covered < start:
        return None
    # A single pipy batch includes the assistant's calls and all their results.
    if any(node > replay_node and kind == "tool_update" and json.loads(raw)[0] <= covered
           for node, _, kind, _, raw in path):
        return None
    return covered


def upstream_anthropic_compaction(db, session, parent, entry, mapped):
    saved = mapping(mapping(entry.get("preserveData")).get(ANTHROPIC_PRESERVE_KEY))
    if not all(isinstance(saved.get(key), str) and saved[key] for key in ("provider", "content")):
        return None
    wire_names = {"signature": "signature", "encryptedContent": "encrypted_content"}
    state = {wire: saved[key] for key, wire in wire_names.items() if isinstance(saved.get(key), str) and saved[key]}
    if not state:
        return None
    if "toolChanges" in saved:
        changes = saved["toolChanges"]
        if not isinstance(changes, list) or not all(isinstance(change, dict) for change in changes):
            return None
        state["tool_changes"] = changes
    native = {"block": {"type": "compaction", "content": saved["content"], **state},
              "anthropic_provider": saved["provider"], **compaction_diagnostics(entry)}
    metadata = {key: saved[key] for key in ("filesText", "model") if isinstance(saved.get(key), str)}
    if type(used := saved.get("usedTokens")) is int or type(used) is float and math.isfinite(used):
        metadata["usedTokens"] = used
    if saved.get("exactTail") is True:
        metadata["exactTail"] = True
    if isinstance(files := saved.get("retainedFiles"), list):
        metadata["retainedFiles"] = [{"text": row["text"], "after": row["after"]} for row in files
                                     if isinstance(row, dict) and isinstance(row.get("text"), str)
                                     and (type(row.get("after")) is int
                                          or type(row.get("after")) is float and math.isfinite(row["after"]))]
    if (committed := journal_time(entry.get("timestamp"))) is not None:
        native["anthropic_committed_at"] = committed
    native["anthropic_metadata"] = metadata
    first = entry.get("firstKeptEntryId")
    if first == "":
        anchor = entry.get("providerReplayThroughEntryId")
        replay_node = mapped.get(anchor) if isinstance(anchor, str) else parent if anchor is None else None
        if not (covered := covered_batch(db, session, parent, replay_node)):
            return None
        cut = covered + 1
        native["replay_through"] = covered
    elif isinstance(first, str) and (first_node := mapped.get(first)):
        path = graph.nodes(db, session, parent)
        start = graph.transcript_boundary(db, session, parent)[1]
        cut = next((seq for node, _, kind, seq, _ in path if node == first_node and kind == "batch"), None)
        if cut is None or cut < start or any(node > first_node and kind == "tool_update" and json.loads(raw)[0] < cut
                                              for node, _, kind, _, raw in path):
            return None
    else:
        return None
    return [cut, str(entry.get("summary") or saved["content"]), native]


def upstream_responses_compaction(db, session, parent, entry, mapped):
    source = mapping(entry.get("preserveData"))
    saved = mapping(next((source[key] for key in RESPONSES_PRESERVE_KEYS if key in source), None))
    provider, items = saved.get("provider"), saved.get("replacementHistory")
    if not isinstance(provider, str) or not provider or not isinstance(items, list) or not all(
        isinstance(item, dict) for item in items
    ):
        return None
    anchor = entry.get("providerReplayThroughEntryId")
    replay_node = mapped.get(anchor) if isinstance(anchor, str) else parent if anchor is None else None
    if not (covered := covered_batch(db, session, parent, replay_node)):
        return None
    portable_cut, portable_summary, previous = graph.checkpoint(db, session, parent)
    previous = previous or {}
    native = {
        "items": items, "provider": provider, "replay_through": covered,
        "replacement_tokens": responses_replacement_tokens(items),
        "portable_first_kept": previous.get("portable_first_kept", portable_cut),
        "portable_summary": previous.get("portable_summary", portable_summary),
        "source_fields": {key: value for key, value in saved.items() if key not in ("provider", "replacementHistory")},
        **compaction_diagnostics(entry),
    }
    first = entry.get("firstKeptEntryId")
    if isinstance(first, str) and (first_node := mapped.get(first)):
        row = db.execute("SELECT batch FROM session_nodes WHERE id=? AND session=? AND kind='batch'",
                         (first_node, session)).fetchone()
        if row and graph.first_batch_at(db, session, parent, row[0]) == row[0]:
            native["display_first_kept"] = row[0]
    if archive := previous.get("portable_archive") or previous.get("snapcompact"):
        native["portable_archive"] = archive
    if handoff := previous.get("portable_handoff") or (
        previous if previous.get("method") in ("handoff", "experimental-context-rollover") else None
    ):
        native["portable_handoff"] = handoff
    return [covered + 1, str(entry.get("summary") or ""), native]


def upgrade_responses_compactions(db, session):
    rows = db.execute("""
        SELECT id,parent,payload FROM session_nodes WHERE session=? AND kind='foreign'
        AND json_extract(payload,'$.type')='compaction'
        AND (json_type(payload,'$.preserveData.openaiRemoteCompaction')='object'
             OR json_type(payload,'$.preserveData.openaiCompactionV2')='object') ORDER BY id
    """, (session,)).fetchall()
    if not rows:
        return False
    mapped = dict(db.execute("SELECT source_id,node FROM session_entry_sources WHERE session=?", (session,)))
    upgraded = False
    for identifier, parent, raw in rows:
        if payload := upstream_responses_compaction(db, session, parent, json.loads(raw), mapped):
            db.execute("UPDATE session_nodes SET kind='compact',payload=? WHERE session=? AND id=?",
                       (pack(payload), session, identifier))
            upgraded = True
    return upgraded


def import_compaction(db, session, parent, entry, batches, mapped):
    metadata = mapping(entry.get("miniPi"))
    source = mapping(entry.get("preserveData"))
    native = mapping(metadata.get("native")) or None
    if ANTHROPIC_PRESERVE_KEY in source:
        if payload := upstream_anthropic_compaction(db, session, parent, entry, mapped):
            return graph.append(db, session, parent, "compact", payload=payload)
    if not metadata and any(key in source for key in RESPONSES_PRESERVE_KEYS):
        payload = upstream_responses_compaction(db, session, parent, entry, mapped)
        return graph.append(db, session, parent, "compact" if payload else "foreign", payload=payload or entry)
    first = metadata.get("cut", entry.get("firstKeptEntryId"))
    cut = batches.get(first) if isinstance(first, str) else None
    if cut is None or (cut and graph.first_batch_at(db, session, parent, cut) != cut):
        cut = 0 if "cut" in metadata and first is None else graph.checkpoint(db, session, parent)[0]
    summary = str(metadata.get("summary", entry.get("summary")) or "")
    if native and "portable_first_kept" in native:
        portable = native["portable_first_kept"]
        start = graph.transcript_boundary(db, session, parent)[1]
        seq = batches.get(portable, start) if isinstance(portable, str) else start
        if seq < start or graph.first_batch_at(db, session, parent, seq) != seq:
            seq = start
        native = {**native, "portable_first_kept": seq}
    if native and native.get("provider") and "items" not in native:
        source_payload = mapping(source.get(RESPONSES_PRESERVE_KEYS[0]))
        items = source_payload.get("replacementHistory")
        if (source_payload.get("provider") == native["provider"] and isinstance(items, list)
                and all(isinstance(item, dict) for item in items)):
            native = {**native, "items": items, "source_fields": {
                key: value for key, value in source_payload.items() if key not in ("provider", "replacementHistory")
            }}
        else:
            native = {key: value for key, value in native.items() if key != "provider"}
            native["replay_through"] = None
    if native and "display_first_kept" in native:
        first = native["display_first_kept"]
        display = batches.get(first) if isinstance(first, str) else None
        if display and graph.first_batch_at(db, session, parent, display) == display:
            native = {**native, "display_first_kept": display}
        else:
            native.pop("display_first_kept")
    if native and "replay_through" in native:
        anchor = native["replay_through"]
        covered = None
        if isinstance(anchor, str):
            covered = covered_batch(db, session, parent, mapped.get(anchor)) if native.get("provider") else batches.get(anchor)
        if covered and graph.first_batch_at(db, session, parent, covered) == covered:
            native = {**native, "replay_through": covered}
            cut = covered + 1
        else:
            cut = native.get("portable_first_kept", cut)
            summary = str(native.get("portable_summary", summary) or "")
            archive = native.get("portable_archive")
            native = {"snapcompact": archive} if archive else native.get("portable_handoff")
    if not native and entry.get("method") == "handoff":
        native = {"method": "handoff"}
    if not native and mapping(entry.get("details")).get("kind") == "experimental-context-rollover":
        native = {"method": "experimental-context-rollover", "tokensBefore": entry.get("tokensBefore", 0)}
    if not native and "snapcompact" in source:
        archive = mapping(source["snapcompact"])
        text = archive.get("text") or "\n".join(archive.get(key) or "" for key in ("textHead", "textTail"))
        if text.strip():
            summary += "\n\n" + text.replace("█", "\n").replace("\x0e", "").replace("\x0f", "")
    if diagnostics := compaction_diagnostics(entry):
        native = {**(native or {}), **diagnostics}
    return graph.append(db, session, parent, "compact", payload=[cut, summary, native])


def restore_shake_artifacts(db, session, parent, edits):
    missing = {}
    for seq, index, edit in edits:
        if isinstance(edit, dict):
            continue
        for identifier in SHAKE_REFERENCE.findall(tool_text(edit)):
            if not db.execute("SELECT 1 FROM artifacts WHERE session = ? AND id = ?", (session, identifier)).fetchone():
                missing.setdefault((seq, index), set()).add(identifier)
    if not missing:
        return
    original = dict(graph.stored_batches(db, session, parent, min(missing)[0], max(missing)[0] + 1,
                                         apply_elisions=False))
    for (seq, index), identifiers in missing.items():
        batch = original.get(seq, ())
        if not 0 <= index < len(batch):
            continue
        message = batch[index]
        content = message.get("content")
        if not isinstance(content, (str, list)) or not valid_elision(content):
            continue
        text = tool_text(content)
        regions = (text,) if message["role"] == "tool" else (text[start:end] for start, end in block_ranges(text))
        for region in regions:
            if (identifier := artifact_id(region)) in identifiers:
                db.execute("INSERT OR IGNORE INTO artifacts VALUES (?, ?, ?)", (session, identifier, region))


def import_journal(db_path, workspace, source, *, session_id=None, replace=False, notice=None, blob_dirs=(),
                   preserve_identity=False):
    source = source.resolve(strict=True)
    header, rows, skipped = read_journal(source)
    order, missing = ordered_rows(rows)
    if notice is None and session_id is None:
        marker = next((details for entry in reversed(rows)
                       if entry["type"] == "custom_message" and entry.get("customType") == "session-transfer"
                       and (details := graph.transfer_details(entry.get("details")))), None)
        if marker and marker["kind"] == "departed":
            notice = graph.transfer_message("forked", source=str(source), to=marker["to"])
    session = session_id or str(uuid4())
    initialize(db_path)
    mapped, batches, pending = {None: None}, {None: 0}, {None: {}}
    artifacts = sidecar(source)
    state = Path(db_path).parent
    target = state / "artifacts" / session
    blob_dirs = (state / "blobs", *blob_dirs)
    blob_cache = {}
    with install_sidecar(artifacts, target, replace=replace), database(db_path) as db:
        if replace:
            graph.drop_journal(db, session)
        db.execute("INSERT INTO sessions(id,workspace,updated) VALUES (?,?,?)", (session, workspace, time.time()))
        db.executemany("INSERT INTO artifacts(session,id,content) VALUES (?,?,?)",
                       ((session, identifier, text) for identifier, text in stored_artifacts(artifacts)))
        saved_header = {key: header[key] for key in ("id", "version", "title", "titleSource", "timestamp") if key in header}
        db.execute("INSERT INTO session_imports VALUES (?,?)", (session, pack({**saved_header, "_pipyTransferIndexed": True})))
        if isinstance(title := header.get("title"), str) and title.strip():
            title_source = "auto" if header.get("titleSource") == "auto" else "user"
            db.execute("INSERT INTO session_titles VALUES (?,?,?)", (session, title, title_source))
        original_id = header.get("id")
        if isinstance(original_id, str) and original_id:
            parent_id = header.get("parentSession") if preserve_identity or session == original_id else original_id
            cache_key = header.get("providerPromptCacheKey")
            if not isinstance(cache_key, str) or not 0 < len(cache_key) <= 64:
                cache_key = original_id if session_id and len(original_id) <= 64 else session
            if isinstance(parent_id, str) or cache_key != session:
                db.execute("INSERT INTO session_origins VALUES (?,?,?)",
                           (session, parent_id if isinstance(parent_id, str) else "", cache_key))
        for entry in order:
            identifier, parent_id, kind = entry["id"], entry.get("parentId"), entry["type"]
            parent = mapped[parent_id]
            last, calls = batches[parent_id], pending[parent_id]
            node = parent
            hydrate_images(entry, blob_dirs, blob_cache)
            message = import_message(entry)
            if message and "_history_at" not in message:
                wire = mapping(entry.get("message"))
                recorded = journal_time(entry.get("timestamp"))
                when = journal_time(wire.get("timestamp"))
                if when is None:
                    when = recorded
                if when is not None:
                    message["_history_at"] = when
                if recorded is not None and recorded != when:
                    message["_journal_at"] = recorded
                if message["role"] in ("user", "tool") and isinstance(parts := wire.get("content"), list) and all(
                    isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str)
                    for part in parts
                ):
                    message["_history_text_parts"] = [part["text"] for part in parts]
            if message and message["role"] == "tool" and message["tool_call_id"] in calls:
                seq, index = calls[message["tool_call_id"]]
                node = graph.append(db, session, parent, "tool_update", payload=[seq, index, message])
                last = seq
            elif message:
                if message["role"] == "tool":
                    message = {"role": "developer", "content": "Orphaned tool result:\n" + plain_text(message["content"])}
                tool_calls = message.get("tool_calls", ())
                batch = [message, *map(placeholder, tool_calls)]
                node, last = insert_batch(db, session, parent, batch)
                if message["role"] in ("assistant", "user"):
                    calls = {call["id"]: (last, index) for index, call in enumerate(tool_calls, 1)}
            elif kind == "compaction":
                node = import_compaction(db, session, parent, entry, batches, mapped)
            elif kind == "reset_boundary":
                node = graph.append(db, session, parent, "reset", payload=last)
                calls = {}
            elif kind == "custom" and entry.get("customType") == "mini-pi.elision":
                edits = []
                for item in entry["data"] if isinstance(entry.get("data"), list) else ():
                    if (
                        isinstance(item, list) and len(item) == 3 and isinstance(item[0], str) and item[0] in batches
                        and type(item[1]) is int and valid_elision(item[2])
                    ):
                        edits.append((batches[item[0]], item[1], item[2]))
                restore_shake_artifacts(db, session, parent, edits)
                node = graph.append(db, session, parent, "elision", payload=edits)
            elif kind == "custom" and entry.get("customType") == "mini-pi.route":
                node = graph.append(db, session, parent, "route", payload=entry["data"])
            elif entry.get("customType") != "mini-pi.active-leaf":
                node = graph.append(db, session, parent, "foreign", payload=entry)
            if node != parent:
                db.execute("INSERT INTO session_entry_sources VALUES (?,?,?,?)", (
                    session, node, identifier, pack(source_metadata(entry)),
                ))
            mapped[identifier], batches[identifier], pending[identifier] = node, last, calls
        for entry in rows:
            if entry["type"] == "label":
                target = mapped.get(entry.get("targetId"))
                label = entry.get("label")
                if target is not None and isinstance(label, str) and label:
                    db.execute("INSERT OR REPLACE INTO session_labels VALUES (?,?,?)", (session, target, label))
                elif target is not None:
                    db.execute("DELETE FROM session_labels WHERE session = ? AND node = ?", (session, target))
        last = rows[-1] if rows else {}
        selected = last.get("activeLeafId", last.get("id"))
        if selected is not None and (not isinstance(selected, str) or selected not in mapped):
            selected = next((
                entry["id"] for entry in reversed(rows) if entry.get("activeLeafId", entry["id"]) == entry["id"]
            ), None)
        leaf = mapped[selected]
        if notice:
            leaf, _ = insert_batch(db, session, leaf, [notice])
        graph.set_head(db, session, leaf)
        graph.projection(db, session, *graph.checkpoint(db, session, leaf))
    return session, skipped, missing


def export_parts(content):
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    output = []
    for item in content or ():
        if item["type"] == "image_url":
            url = item["image_url"]["url"]
            prefix, separator, data = url.partition(",")
            if separator and prefix.startswith("data:") and prefix.endswith(";base64"):
                output.append({"type": "image", "mimeType": prefix[5:-7], "data": data})
            else:
                output.append({"type": "text", "text": "[External image remains in the source session.]"})
        else:
            output.append(item)
    return output


def export_message(message, call_names):
    role = message["role"]
    source = message.get("content")
    custom_content = export_parts(source) if isinstance(source, list) else source
    if kind := message.get("_goal_message") or message.get("_todo_message"):
        return {"type": "custom_message", "customType": kind, "content": custom_content,
                "display": False, "attribution": "agent"}
    if role == "developer":
        if message.get("_compaction_continue"):
            return {"type": "custom_message", "customType": "compaction-auto-continue", "content": custom_content,
                    "display": False, "attribution": "agent", "synthetic": True}
        if message.get("_thinking_loop_redirect"):
            return {"type": "custom_message", "customType": "thinking-loop-redirect", "content": custom_content,
                    "display": False, "attribution": "agent"}
        if details := message.get("_session_transfer"):
            return {"type": "custom_message", "customType": "session-transfer", "content": custom_content,
                    "display": True, "details": details}
        if message.get("_branch_summary"):
            return {"type": "branch_summary", "summary": message["content"], "fromId": ""}
        if changes := message.get("_memory_recall_changes"):
            return {"type": "custom_message", "customType": "memory-recall-changes", "content": custom_content,
                    "display": False, "attribution": "agent", "details": changes}
        if details := message.get("_async_jobs"):
            return {"type": "custom_message", "customType": "async-result", "content": custom_content,
                    "display": True, "attribution": "agent", "details": details}
        if (peers := message.get("_irc_messages")) or message.get("_parent_assignment"):
            details = {"messages": peers or []}
            if message.get("_parent_assignment"):
                details["fromParent"] = True
            return {"type": "custom_message", "customType": "irc:incoming", "content": custom_content,
                    "display": True, "attribution": "agent", "details": details}
        if services := message.get("_service_runs"):
            return {"type": "custom_message", "customType": "launch-completion", "content": custom_content,
                    "display": True, "attribution": "agent", "details": {"daemons": services}}
        if rules := message.get("_ttsr_rules"):
            if message.get("_ttsr_kind") == "tool":
                return {"type": "message", "message": {"role": "developer", "content": export_parts(message["content"]),
                                                       "timestamp": int(time.time() * 1000), "attribution": "agent"},
                        "miniPi": {"message": message}}
            return {"type": "custom_message", "customType": "ttsr-injection", "content": custom_content,
                    "display": False, "attribution": "agent", "details": {"rules": rules, "kind": message.get("_ttsr_kind")}}
        return {"type": "custom_message", "customType": "mini-pi.recovery", "content": custom_content, "display": True}
    content = custom_content if isinstance(source, list) else export_parts(source)
    if role in ("user", "tool") and isinstance(source, str):
        parts = message.get("_history_text_parts")
        if parts is not None and "\n".join(parts) == source:
            content = [{"type": "text", "text": part} for part in parts]
        elif role == "user" and parts is None and "_history_at" in message:
            content = source
    now = message.get("_history_at", int(time.time() * 1000))
    if role == "tool":
        text = plain_text(content)
        result = {"role": "toolResult", "toolCallId": message["tool_call_id"],
                  "toolName": call_names.get(message["tool_call_id"], "todo" if message.get("_todo_snapshot") else "unknown"),
                  "content": content,
                  "isError": message.get("_tool_error", is_failed(text) or text.startswith("Skipped after a new user steering message.")),
                  "timestamp": now}
        if details := message.get("_tool_details"):
            result["details"] = details
        if "_tool_pruned_at" in message:
            result["prunedAt"] = message["_tool_pruned_at"]
        if message.get("_tool_useless"):
            result["useless"] = True
    elif role == "assistant":
        if message.get("_thinking"):
            content.insert(0, {"type": "thinking", "thinking": message["_thinking"]})
        for call in message.get("tool_calls", ()):
            function = call["function"]
            call_names[call["id"]] = function["name"]
            try:
                arguments = json.loads(function["arguments"])
            except ValueError:
                arguments = {}
            content.append({"type": "toolCall", "id": call["id"], "name": function["name"], "arguments": arguments})
        native = message.get("_anthropic", {})
        native_blocks = native.get("blocks", ()) if native.get("text") == message.get("content") else ()
        if native_blocks:
            calls = {item["id"]: item for item in content if item["type"] == "toolCall"}
            content = []
            for block in native_blocks:
                kind = block["type"]
                if kind in ("text", "fallback"):
                    content.append(block)
                elif kind == "tool_use" and block["id"] in calls:
                    content.append(calls[block["id"]])
                elif kind == "thinking":
                    reasoning = {"type": "thinking", "thinking": block.get("thinking", "")}
                    if block.get("signature"):
                        reasoning["thinkingSignature"] = block["signature"]
                    content.append(reasoning)
                elif kind == "redacted_thinking":
                    content.append({"type": "redactedThinking", "data": block["data"]})
                elif kind == "server_tool_use" or kind.endswith("_tool_result"):
                    content.append({"type": "anthropicServerTool", "block": block})
        result = {
            "role": "assistant", "content": content, "timestamp": now, "api": "openai-completions",
            "provider": "openai", "model": "unknown",
            "stopReason": ("aborted" if message.get("_ttsr_aborted") else message.get("_tool_stop")
                           or ("toolUse" if message.get("tool_calls") else "stop")),
            "usage": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0,
                      "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0}},
        }
        if imported := message.get("_anthropic_import"):
            result.update({"api": "anthropic-messages", **imported})
    else:
        result = {"role": role, "content": content, "timestamp": now}
        if role == "user" and message.get("liveSteered") is True:
            result["liveSteered"] = True
        if role == "user" and message.get("_mcp_notification"):
            result["attribution"] = "agent"
    if role == "assistant" and (metadata := message.get("_rpc")):
        result.update({key: value for key, value in metadata.items() if key != "usage"})
        result["usage"].update(metadata.get("usage", {}))
    if role == "assistant" and (transformations := message.get("_input_transformations")):
        result["inputTransformations"] = transformations
    if role == "assistant" and (details := message.get("_stop_details")):
        result["stopDetails"] = details
    return {"type": "message", "message": result, "miniPi": {"message": message}}


def journal_entries(db, session, workspace, leaf, node_ids=None, *, header_id=None):
    imported = db.execute("SELECT header FROM session_imports WHERE session = ?", (session,)).fetchone()
    header = json.loads(imported[0]) if imported else {}
    header.pop("_pipyTransferIndexed", None)
    created = header.get("timestamp")
    origin = db.execute("SELECT parent,prompt_cache_key FROM session_origins WHERE session = ?", (session,)).fetchone()
    header = {**header, "type": "session", "id": header_id or session, "version": VERSION, "cwd": workspace,
              "timestamp": created if isinstance(created, str) and 0 < len(created) <= 40 else timestamp()}
    if row := db.execute("SELECT title,source FROM session_titles WHERE session=?", (session,)).fetchone():
        header["title"], header["titleSource"] = row
    if origin:
        header["providerPromptCacheKey"] = origin[1]
        if origin[0] and origin[0] != header["id"]:
            header["parentSession"] = origin[0]
    yield title_slot(header)
    yield header
    mapped, anchors, call_names = {None: None}, {0: None}, {}
    sources = {
        node: (identifier, json.loads(metadata))
        for node, identifier, metadata in db.execute(
            "SELECT node,source_id,metadata FROM session_entry_sources WHERE session = ?", (session,),
        )
    }
    used_ids = {identifier for identifier, _ in sources.values()}
    index = 0
    def make(record, parent, source=None, *, todo_entry=None):
        nonlocal index
        if source:
            identifier, metadata = source
            wire = metadata.get("message")
            if wire and isinstance(record.get("message"), dict):
                record["message"].update(wire)
            record.update({key: value for key, value in metadata.items() if key != "message"})
        elif todo_entry and todo_entry not in used_ids:
            identifier = todo_entry
            used_ids.add(identifier)
        else:
            while True:
                index += 1
                identifier = f"{index:08x}"
                if identifier not in used_ids:
                    break
        return {**record, "id": identifier, "parentId": parent, "timestamp": record.get("timestamp") or timestamp()}

    def anchor(cut, parent):
        return anchors.get(cut) if cut in anchors else anchors.get(graph.first_batch_at(db, session, parent, cut))

    def replay_anchor(seq, parent):
        for node, _, kind, batch, data in reversed(graph.nodes(db, session, parent)):
            if (kind == "batch" and batch == seq) or (kind == "tool_update" and json.loads(data)[0] == seq):
                return mapped[node]
        return None

    attempts = {}
    for identifier, parent, data in db.execute(
        "SELECT id,parent,payload FROM session_usage_attempts WHERE session = ? ORDER BY id", (session,),
    ):
        payload = json.loads(data)
        after = payload.pop("_after_node", parent or 0)
        attempts.setdefault(after, []).append((identifier, parent, payload))

    def model_usage(after):
        for identifier, parent, payload in attempts.get(after, ()):
            wire_id = f"usage-{identifier:x}"
            while wire_id in used_ids:
                wire_id += "-"
            used_ids.add(wire_id)
            yield make(payload, mapped.get(parent), (wire_id, {}))

    yield from model_usage(0)
    for identifier, parent, kind, seq, data in db.execute("""
        SELECT id,parent,kind,batch,payload FROM session_nodes WHERE session = ? ORDER BY id
    """, (session,)):
        base = mapped[parent]
        payload = json.loads(data) if data else None
        if kind == "batch":
            raw, = db.execute("SELECT messages FROM batches WHERE session = ? AND seq = ?", (session, seq)).fetchone()
            batch = json.loads(raw)
            batch = [item for item in batch if not pending_result(item)]
            for number, message in enumerate(batch):
                record = export_message(message, call_names)
                if payload and payload.get("timestamp"):
                    record["timestamp"] = payload["timestamp"]
                if record["type"] == "branch_summary":
                    record["fromId"] = mapped.get(message.get("_branch_from_node")) or base or ""
                entry = make(record, base, sources.get(identifier) if number == 0 else None,
                             todo_entry=message.get("_todo_entry_id"))
                base = entry["id"]
                if number == 0:
                    anchors[seq] = base
                if node_ids is not None:
                    node_ids[base] = identifier
                yield entry
        else:
            if kind == "compact":
                cut, summary, native = payload
                portable_cut, portable_summary = cut, summary
                preserve = None
                anthropic = None
                if native and "portable_first_kept" in native:
                    portable_cut, portable_summary = native["portable_first_kept"], native["portable_summary"]
                    native = {**native, "portable_first_kept": anchor(portable_cut, parent)}
                if native and "replay_through" in native:
                    native = {**native, "replay_through": replay_anchor(native["replay_through"], parent)}
                if native and native.get("provider") and "items" in native:
                    preserve = {**native.pop("source_fields", {}), "provider": native["provider"],
                                "replacementHistory": native.pop("items")}
                    if "display_first_kept" in native:
                        native["display_first_kept"] = anchor(native["display_first_kept"], parent)
                if native and (provider := native.get("anthropic_provider")) and "block" in native:
                    native = dict(native)
                    block = native.pop("block")
                    anthropic = {**native.pop("anthropic_metadata", {}), "provider": provider, "content": block["content"]}
                    for wire, field in (("signature", "signature"), ("encrypted_content", "encryptedContent"),
                                        ("tool_changes", "toolChanges")):
                        if wire in block:
                            anthropic[field] = block[wire]
                if native and "snapcompact" in native:
                    portable_summary += "\n\n" + native["snapcompact"]["text"].replace("█", "\n").replace("\x0e", "").replace("\x0f", "")
                elif native and "portable_archive" in native:
                    portable_summary += "\n\n" + native["portable_archive"]["text"].replace("█", "\n").replace("\x0e", "").replace("\x0f", "")
                record = {"type": "compaction", "summary": portable_summary,
                          "firstKeptEntryId": anchor(portable_cut, parent) or "", "tokensBefore": 0,
                          "miniPi": {"native": native, "cut": anchor(cut, parent), "summary": summary}}
                if preserve is not None:
                    record.update(summary=summary, firstKeptEntryId=native.get("display_first_kept") or "",
                                  providerReplayThroughEntryId=native["replay_through"],
                                  preserveData={RESPONSES_PRESERVE_KEYS[0]: preserve})
                if anthropic is not None:
                    record["preserveData"] = {ANTHROPIC_PRESERVE_KEY: anthropic}
                    if native.get("replay_through"):
                        record.update(firstKeptEntryId="", providerReplayThroughEntryId=native["replay_through"])
                    if (committed := native.get("anthropic_committed_at")) is not None:
                        record["timestamp"] = datetime.fromtimestamp(committed / 1000, timezone.utc).isoformat(timespec="milliseconds")
                if native:
                    record.update({key: native[key] for key in ("tokensBefore", "tokensAfter", "warning") if key in native})
                if native and native.get("method") == "handoff":
                    record["method"] = "handoff"
                if native and native.get("method") == "experimental-context-rollover":
                    record["details"] = {"kind": "experimental-context-rollover", "version": 1}
            elif kind == "elision":
                record = {"type": "custom", "customType": "mini-pi.elision",
                          "data": [(anchors[batch], position, value) for batch, position, value in payload]}
            elif kind == "tool_update":
                record = export_message(payload[2], call_names)
            elif kind == "reset":
                record = {"type": "reset_boundary"}
            elif kind == "foreign":
                record = {key: value for key, value in payload.items() if key != "activeLeafId"}
            else:
                record = {"type": "custom", "customType": "mini-pi.route", "data": payload}
            todo_entry = None
            if kind == "tool_update":
                todo_entry = payload[2].get("_todo_entry_id")
            elif kind == "foreign" and record.get("customType") in ("user_todo_edit", "experimental_context_notes"):
                todo_entry = record.get("id")
            entry = make(record, base, sources.get(identifier), todo_entry=todo_entry)
            base = entry["id"]
            if node_ids is not None:
                node_ids[base] = identifier
            yield entry
        mapped[identifier] = base
        yield from model_usage(identifier)
    active = mapped[leaf]
    for target, label in db.execute("SELECT node,label FROM session_labels WHERE session = ? ORDER BY node", (session,)):
        yield make({"type": "label", "targetId": mapped[target], "label": label, "activeLeafId": active}, active)
    yield make({"type": "custom", "customType": "mini-pi.active-leaf", "activeLeafId": active}, active)


def export_journal(db_path, session, workspace, leaf, output, *, header_id=None):
    temporary = None
    try:
        with NamedTemporaryFile(dir=output.parent, delete=False) as file:
            temporary = Path(file.name)
        opener = gzip.open if output.suffix.lower() == ".gz" else open
        with database(db_path) as db:
            db.execute("BEGIN")
            with opener(temporary, "wt", encoding="utf-8", newline="\n") as file:
                for record in journal_entries(db, session, workspace, leaf, header_id=header_id):
                    file.write(pack(record) + "\n")
            target = sidecar(output)
            if os.path.lexists(target):
                raise FileExistsError(f"Export artifact directory already exists: {target}")
            has_artifacts, = db.execute("SELECT EXISTS(SELECT 1 FROM artifacts WHERE session = ?)", (session,)).fetchone()
            with copy_sidecar(Path(db_path).parent / "artifacts" / session, target, create=bool(has_artifacts)):
                for identifier, content in db.execute("SELECT id,content FROM artifacts WHERE session = ?", (session,)):
                    name = f"{identifier}.pipy.log"
                    if not PIPY_ARTIFACT.fullmatch(name):
                        raise ValueError("Invalid stored artifact identifier.")
                    with (target / name).open("xb") as destination:
                        destination.write(content.encode("utf-8"))
                os.link(temporary, output)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)
