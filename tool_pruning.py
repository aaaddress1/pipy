"""Cache-aware pruning of tool results already consumed by the model."""

import json
import re


SUPERSEDED = "[Superseded by a newer read of this file]"
UNEVENTFUL = "[Uneventful result elided]"
MIN_OUTPUT = 50
PROTECT_OUTPUT = 40_000
MIN_BULK_SAVINGS = 20_000
WARM_SUFFIX = 8_000
IDLE_FLUSH_SECONDS = 90 * 60
RANGE = r"L?\d+(?:(?:[-+]|\.\.)L?\d+|-|\.\.)?"
RANGES = re.compile(rf"{RANGE}(?:,{RANGE})*", re.I | re.ASCII)
LINE_RANGE = re.compile(r"L?(\d+)(?:(\.\.|[-+])(?:L?(\d+))?)?", re.I | re.ASCII)


class ReadText(str):
    def __new__(cls, content, *, complete=None, failed=False):
        result = super().__new__(cls, content)
        result.complete, result.failed = complete, failed
        return result


def split_read(path):
    base, sep, selector = path.rpartition(":")
    if not sep or not base or not (RANGES.fullmatch(selector) or selector.lower() in ("raw", "conflicts")):
        return path, None
    root, inner, previous = base.rpartition(":")
    if root and inner and (previous.lower() == "raw" and RANGES.fullmatch(selector)
                           or selector.lower() == "raw" and RANGES.fullmatch(previous)):
        return root, f"{previous}:{selector}"
    return base, selector


def selected_lines(file, selector, limit, check=None):
    chunks = selector.lower().split(":")
    ranges = next((part.split(",") for part in chunks if RANGES.fullmatch(part)), None)
    if ranges is None:
        return file.read(limit + 1)
    pinned = "raw" in chunks or len(ranges) > 1
    bounds = []
    for part in ranges:
        start, separator, end = LINE_RANGE.fullmatch(part).groups()
        first, last = int(start), int(end) if end is not None else None
        if first < 1 or separator == "+" and not last:
            raise ValueError("Line selectors use 1-based lines and positive counts.")
        if separator == "+":
            last = first + last - 1
        elif separator is None and pinned:
            last = first
        if last is not None and last < first:
            raise ValueError("Line range ends before its start.")
        bounds.append((first, float("inf") if last is None else last))
    bounds.sort()
    result, index, remaining = [], 0, limit + 1
    line_number, chunk_size = 1, 65_536
    partial_cr = selected = False
    while remaining:
        if check:
            check()
        piece = file.readline(chunk_size)
        if not piece:
            break
        if partial_cr and piece.startswith("\n"):
            if selected:
                result.append("\n")
                remaining -= 1
            piece = piece[1:]
        partial_cr = False
        if not piece:
            continue
        while index < len(bounds) and line_number > bounds[index][1]:
            index += 1
        if index == len(bounds):
            break
        selected = line_number >= bounds[index][0]
        if selected:
            result.append(piece[:remaining])
            remaining -= min(len(piece), remaining)
        if piece.endswith(("\n", "\r")):
            line_number += 1
            partial_cr = len(piece) == chunk_size and piece.endswith("\r")
    return "".join(result)


def read_target(call):
    if not call or call.get("name") != "read":
        return None
    try:
        args = json.loads(call.get("arguments", ""))
    except (ValueError, TypeError, RecursionError):
        return None
    path = args.get("path") if isinstance(args, dict) else None
    return path if isinstance(path, str) and path else None


def read_key(path):
    if not path or "://" in path:
        return None
    base, selector = split_read(path)
    if selector is None:
        return base, "", ""
    form = "alternate" if any(part in ("raw", "conflicts") for part in selector.lower().split(":")) else "range"
    return base, form, selector


def blocks(message):
    content = message.get("content")
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content.strip() else []
    return content if isinstance(content, list) else []


def complete_read(message):
    details = message.get("_tool_details") or {}
    meta = details.get("meta") or {}
    return (not details.get("summary") and not (details.get("truncation") or {}).get("truncated")
            and not meta.get("truncation") and not (meta.get("limits") or {}).get("columnTruncated")
            and ("totalLines" in details or any("image" in part.get("type", "") for part in blocks(message))))


def useless(message):
    return (message["role"] == "tool" and not message.get("_tool_error")
            and (message.get("_tool_useless") is True or (message.get("_tool_details") or {}).get("useless") is True))


def without_useless(messages):
    dropped = {message["tool_call_id"] for message in messages if useless(message)}
    if not dropped:
        return messages
    output = []
    for message in messages:
        if message["role"] == "tool" and message.get("tool_call_id") in dropped:
            continue
        calls = message.get("tool_calls", ())
        if calls and any(call["id"] in dropped for call in calls):
            kept = [call for call in calls if call["id"] not in dropped]
            if not kept and not (message.get("content") or message.get("_thinking")):
                continue
            message = {**message, "tool_calls": kept}
        output.append(message)
    return output


def cache_floor(rows, lookback):
    if lookback is None:
        return 0
    positions, floor = 6, len(rows)
    later = None
    newer_assistant = hoisted = False
    for index in range(len(rows) - 1, -1, -1):
        message = rows[index][2]
        role = message["role"]
        if role == "tool":
            if later != "result":
                positions += 1
                hoisted = False
            images = sum("image" in part.get("type", "") for part in blocks(message))
            if images and message.get("_tool_error"):
                positions += images + (not hoisted)
                hoisted = True
            later, newer_assistant = "result", False
        elif role == "assistant":
            calls = message.get("tool_calls", ())
            native = (message.get("_anthropic") or {}).get("blocks", ())
            thinking = sum(part.get("type") in ("thinking", "redacted_thinking") for part in native)
            text = sum(part.get("type") == "text" and bool(part.get("text", "").strip()) for part in blocks(message))
            if calls or text or thinking:
                positions += bool(newer_assistant) + (bool(calls) and later != "call") + text + thinking
                later, newer_assistant = (None if text or thinking else "call"), True
        else:
            count = len(blocks(message))
            if count:
                positions += count
                later, newer_assistant = None, False
        if positions >= lookback:
            return floor
        if role == "assistant":
            floor = index
    return 0


def prune_plan(batches, count_tokens, *, supersede=True, drop_useless=True, bulk=False,
               warm_limit=WARM_SUFFIX, lookback=None):
    rows, calls = [], {}
    for seq, batch in batches:
        local = {call["id"]: call["function"] for message in batch for call in message.get("tool_calls", ())}
        calls.update(local)
        rows.extend((seq, index, message, calls.get(message.get("tool_call_id"))) for index, message in enumerate(batch))
    if not any(row[2]["role"] == "tool" for row in rows):
        return {}, 0
    floor = cache_floor(rows, lookback) if warm_limit is not None else 0
    newest, edits = {}, {}
    suffix = output_tokens = saved = 0
    for position in range(len(rows) - 1, floor - 1, -1):
        if warm_limit is not None and suffix > warm_limit:
            break
        seq, index, message, call = rows[position]
        size = count_tokens(message)
        suffix += size
        if message["role"] != "tool":
            continue
        age = output_tokens
        output_tokens += size
        if not call or call.get("name") == "skill" or "_tool_pruned_at" in message:
            continue
        path = read_target(call)
        if path and path.startswith("skill://"):
            continue
        notice = None
        if supersede and (key := read_key(path)):
            failed = bool(message.get("_tool_error"))
            same = newest.get(key)
            parent = newest.get((key[0], "", "")) if key[1] == "range" else None
            if (failed and (same is not None or parent and parent[0])
                    or not failed and (same and same[0] or parent and parent[1])):
                notice = SUPERSEDED
            if same is None or not failed and not same[0]:
                newest[key] = not failed, not failed and complete_read(message)
        if notice is None and drop_useless and useless(message):
            if size > count_tokens({**message, "content": UNEVENTFUL}):
                notice = UNEVENTFUL
        if notice is None and bulk and age >= PROTECT_OUTPUT and size >= MIN_OUTPUT:
            notice = f"[Output truncated - {size} tokens]"
        if notice is not None:
            edits[seq, index] = notice
            saved += max(0, size - count_tokens({**message, "content": notice}))
    return ({}, 0) if bulk and saved < MIN_BULK_SAVINGS else (edits, saved)
