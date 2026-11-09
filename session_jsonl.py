"""Import and export oh-my-pi session journals without replaying tools."""

import gzip
import heapq
import json
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

import session_graph as graph
from runtime import database, initialize, pack
from session_artifacts import PIPY_ARTIFACT, copy_sidecar, install_sidecar, sidecar, stored_artifacts
from session_blobs import hydrate_images
from shake import artifact_id, block_ranges, tool_text
from snapcompact_inline import is_failed


MISSING_RESULT = "Interrupted before result was recorded. Check workspace state before repeating this action."
VERSION = 3
TITLE_BYTES = 256
SHAKE_REFERENCE = re.compile(r"\[shaken ~[0-9]+ tokens — recover: artifact://([a-f0-9]{64})\]")


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def mapping(value):
    return value if isinstance(value, dict) else {}


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
        return set(edit) == {"responses_prefix"} and isinstance(edit["responses_prefix"], str)
    return isinstance(edit, str) or isinstance(edit, list) and all(
        isinstance(part, dict) and (
            part.get("type") == "text" and isinstance(part.get("text"), str)
            or part.get("type") == "image_url" and isinstance(part.get("image_url"), dict)
            and isinstance(part["image_url"].get("url"), str)
        ) for part in edit
    )


def import_message(entry):
    kind = entry["type"]
    if kind == "branch_summary":
        return {"role": "developer", "content": str(entry.get("summary") or ""), "_branch_summary": True}
    if kind == "custom_message":
        message = {"role": "developer", "content": plain_text(text_and_images(entry.get("content")))}
        if entry.get("customType") == "thinking-loop-redirect":
            message["_thinking_loop_redirect"] = True
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
            rows = mapping(entry.get("details")).get("messages")
            if isinstance(rows, list):
                message["_irc_messages"] = [row for row in rows if isinstance(row, dict) and type(row.get("id")) is int]
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
        if not calls and message.get("stopReason") == "stop" and mapping(message.get("stopDetails")).get("type") == "pause_turn":
            result["_stop_details"] = {"type": "pause_turn"}
        if isinstance(message.get("inputTransformations"), list):
            result["_input_transformations"] = message["inputTransformations"]
        return result
    if role in ("custom", "hookMessage", "developer"):
        return {"role": "developer", "content": plain_text(text_and_images(message.get("content")))}
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


def import_compaction(db, session, parent, entry, batches):
    metadata = mapping(entry.get("miniPi"))
    source = mapping(entry.get("preserveData"))
    native = mapping(metadata.get("native")) or None
    if not metadata and (
        "openaiRemoteCompaction" in source or "openaiCompactionV2" in source
    ):
        return graph.append(db, session, parent, "foreign", payload=entry)
    first = metadata.get("cut", entry.get("firstKeptEntryId"))
    cut = batches.get(first) if isinstance(first, str) else None
    if cut is None or (cut and graph.first_batch_at(db, session, parent, cut) != cut):
        cut = 0 if "cut" in metadata and first is None else graph.checkpoint(db, session, parent)[0]
    summary = str(metadata.get("summary", entry.get("summary")) or "")
    if native and "portable_first_kept" in native:
        portable = native["portable_first_kept"]
        native = {**native, "portable_first_kept": batches.get(portable, 0) if isinstance(portable, str) else 0}
    if native and "replay_through" in native:
        anchor = native["replay_through"]
        covered = batches.get(anchor) if isinstance(anchor, str) else None
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
    if not native and "snapcompact" in source:
        archive = mapping(source["snapcompact"])
        text = archive.get("text") or "\n".join(archive.get(key) or "" for key in ("textHead", "textTail"))
        if text.strip():
            summary += "\n\n" + text.replace("█", "\n").replace("\x0e", "").replace("\x0f", "")
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
            if message and message["role"] == "tool" and message["tool_call_id"] in calls:
                seq, index = calls[message["tool_call_id"]]
                node = graph.append(db, session, parent, "tool_update", payload=[seq, index, message])
                last = seq
            elif message:
                if message["role"] == "tool":
                    message = {"role": "developer", "content": "Orphaned tool result:\n" + plain_text(message["content"])}
                tool_calls = message.get("tool_calls", ())
                batch = [message, *[
                    {"role": "tool", "tool_call_id": call["id"], "content": MISSING_RESULT} for call in tool_calls
                ]]
                node, last = insert_batch(db, session, parent, batch)
                if message["role"] in ("assistant", "user"):
                    calls = {call["id"]: (last, index) for index, call in enumerate(tool_calls, 1)}
            elif kind == "compaction":
                node = import_compaction(db, session, parent, entry, batches)
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
    if role == "developer":
        if message.get("_thinking_loop_redirect"):
            return {"type": "custom_message", "customType": "thinking-loop-redirect", "content": message["content"],
                    "display": False, "attribution": "agent"}
        if details := message.get("_session_transfer"):
            return {"type": "custom_message", "customType": "session-transfer", "content": message["content"],
                    "display": True, "details": details}
        if message.get("_branch_summary"):
            return {"type": "branch_summary", "summary": message["content"], "fromId": ""}
        if changes := message.get("_memory_recall_changes"):
            return {"type": "custom_message", "customType": "memory-recall-changes", "content": message["content"],
                    "display": False, "attribution": "agent", "details": changes}
        if details := message.get("_async_jobs"):
            return {"type": "custom_message", "customType": "async-result", "content": message["content"],
                    "display": True, "attribution": "agent", "details": details}
        if peers := message.get("_irc_messages"):
            return {"type": "custom_message", "customType": "irc:incoming", "content": message["content"],
                    "display": True, "attribution": "agent", "details": {"messages": peers}}
        if services := message.get("_service_runs"):
            return {"type": "custom_message", "customType": "launch-completion", "content": message["content"],
                    "display": True, "attribution": "agent", "details": {"daemons": services}}
        if rules := message.get("_ttsr_rules"):
            if message.get("_ttsr_kind") == "tool":
                return {"type": "message", "message": {"role": "developer", "content": export_parts(message["content"]),
                                                       "timestamp": int(time.time() * 1000), "attribution": "agent"},
                        "miniPi": {"message": message}}
            return {"type": "custom_message", "customType": "ttsr-injection", "content": message["content"],
                    "display": False, "attribution": "agent", "details": {"rules": rules, "kind": message.get("_ttsr_kind")}}
        return {"type": "custom_message", "customType": "mini-pi.recovery", "content": message["content"], "display": True}
    content = export_parts(message.get("content"))
    now = int(time.time() * 1000)
    if role == "tool":
        text = plain_text(content)
        result = {"role": "toolResult", "toolCallId": message["tool_call_id"],
                  "toolName": call_names.get(message["tool_call_id"], "unknown"), "content": content,
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
        if any(block["type"] == "fallback" for block in native_blocks):
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
    pending_imports = {
        json.loads(data)[0] for data, in db.execute(
            "SELECT payload FROM session_nodes WHERE session = ? AND kind = 'tool_update'", (session,),
        )
    }

    def make(record, parent, source=None):
        nonlocal index
        if source:
            identifier, metadata = source
            wire = metadata.get("message")
            if wire and isinstance(record.get("message"), dict):
                record["message"].update(wire)
            record.update({key: value for key, value in metadata.items() if key != "message"})
        else:
            while True:
                index += 1
                identifier = f"{index:08x}"
                if identifier not in used_ids:
                    break
        return {**record, "id": identifier, "parentId": parent, "timestamp": record.get("timestamp") or timestamp()}

    def anchor(cut, parent):
        return anchors.get(cut) if cut in anchors else anchors.get(graph.first_batch_at(db, session, parent, cut))

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
            if seq in pending_imports or identifier in sources:
                batch = [item for item in batch if item["role"] != "tool"]
            for number, message in enumerate(batch):
                record = export_message(message, call_names)
                if payload and payload.get("timestamp"):
                    record["timestamp"] = payload["timestamp"]
                if record["type"] == "branch_summary":
                    record["fromId"] = mapped.get(message.get("_branch_from_node")) or base or ""
                entry = make(record, base, sources.get(identifier) if number == 0 else None)
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
                if native and "portable_first_kept" in native:
                    portable_cut, portable_summary = native["portable_first_kept"], native["portable_summary"]
                    native = {**native, "portable_first_kept": anchor(portable_cut, parent)}
                if native and "replay_through" in native:
                    native = {**native, "replay_through": anchor(native["replay_through"], parent)}
                if native and "snapcompact" in native:
                    portable_summary += "\n\n" + native["snapcompact"]["text"].replace("█", "\n").replace("\x0e", "").replace("\x0f", "")
                elif native and "portable_archive" in native:
                    portable_summary += "\n\n" + native["portable_archive"]["text"].replace("█", "\n").replace("\x0e", "").replace("\x0f", "")
                record = {"type": "compaction", "summary": portable_summary,
                          "firstKeptEntryId": anchor(portable_cut, parent) or "", "tokensBefore": 0,
                          "miniPi": {"native": native, "cut": anchor(cut, parent), "summary": summary}}
                if native and native.get("method") == "handoff":
                    record["method"] = "handoff"
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
            entry = make(record, base, sources.get(identifier))
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
