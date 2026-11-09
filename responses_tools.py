"""Track Responses function arguments across output-token truncation."""

import json
from dataclasses import dataclass


FUNCTION_EVENTS = frozenset(("response.function_call_arguments.delta", "response.function_call_arguments.done"))
CUSTOM_EVENTS = frozenset(("response.custom_tool_call_input.delta", "response.custom_tool_call_input.done"))
TOOL_EVENTS = FUNCTION_EVENTS | CUSTOM_EVENTS
ITEM_EVENTS = frozenset(("response.output_item.added", "response.output_item.done"))
OTHER_TOOLS = frozenset(("custom_tool_call", "computer_call", "local_shell_call", "shell_call", "apply_patch_call"))
TOOL_ITEMS = OTHER_TOOLS | {"function_call"}


def reject_constant(value):
    raise ValueError(f"Invalid JSON constant: {value}")


def complete_arguments(raw):
    try:
        return isinstance(json.loads(raw, parse_constant=reject_constant), dict)
    except (TypeError, ValueError, RecursionError):
        return False


@dataclass
class Draft:
    item: dict
    index: int | None
    raw: str
    arguments_done: bool = False
    authoritative: bool = False
    closed: bool = False

    def executable(self, codex):
        # Codex consumes a nonblank arguments.done buffer without latching the standard Responses proof flag.
        if not codex and self.arguments_done:
            return complete_arguments(self.raw)
        return not self.closed and not self.authoritative and complete_arguments(self.raw)


class StreamedTools:
    def __init__(self, *, codex=False):
        self.codex = codex
        self.reset()

    def reset(self):
        self.calls = {}
        self.finished = {}
        self.unsafe = False

    def lookup(self, event):
        identifier, index = event.get("item_id"), event.get("output_index")
        candidates = [draft for draft in self.calls.values() if not draft.closed]
        if identifier:
            if len(matches := [draft for draft in candidates if identifier in (draft.item.get("id"), draft.item["call_id"])]) == 1:
                draft = matches[0]
                return draft if index is None or draft.index is None or index == draft.index else None
            return None
        if index is not None:
            candidates = [draft for draft in candidates if draft.index == index]
        return candidates[0] if len(candidates) == 1 else None

    def observe(self, event):
        kind = event["type"]
        if kind in CUSTOM_EVENTS:
            self.unsafe = True
            return None
        if kind in ITEM_EVENTS:
            item = event["item"]
            index = event.get("output_index")
            if type(index) is not int or index < 0:
                index = None
            if kind == "response.output_item.done" and item["type"] != "function_call":
                self.finished[index if index is not None else len(self.finished)] = item
            if item["type"] in OTHER_TOOLS:
                self.unsafe = True
            if item["type"] != "function_call":
                return None
            identifier, name = item.get("call_id"), item.get("name")
            if not isinstance(identifier, str) or not identifier or not isinstance(name, str) or not name:
                self.unsafe = True
                return None
            raw = item.get("arguments")
            draft = self.calls.get(identifier)
            if draft is None:
                draft = self.calls[identifier] = Draft(item, index, raw if isinstance(raw, str) else "")
            elif (draft.item["name"] != name or index is not None and draft.index not in (None, index)
                  or draft.item.get("id") and item.get("id") and draft.item["id"] != item["id"]):
                self.unsafe = True
                return None
            if kind == "response.output_item.done":
                draft.item = item
                draft.closed = True
                if isinstance(raw, str) and raw.strip() and (self.codex or not draft.authoritative):
                    draft.raw = raw
            return draft
        draft = self.lookup(event)
        if draft is None:
            return None
        if kind == "response.function_call_arguments.delta":
            if draft.authoritative or not isinstance(delta := event.get("delta"), str):
                return None
            draft.raw += delta
        elif isinstance(raw := event.get("arguments"), str):
            draft.arguments_done = True
            if raw.strip():
                draft.raw, draft.authoritative = raw, True
        return draft

    def truncated(self, response, text, *, allow_execute):
        output = list(response.get("output") or ())
        if output:
            reasoning = {item.get("id") for item in output if item["type"] == "reasoning"}
            for index, item in sorted(self.finished.items()):
                if item["type"] == "reasoning" and item.get("id") and item["id"] not in reasoning:
                    output.insert(min(index, len(output)), item)
        else:
            indexed = [*self.finished.items(), *((draft.index, draft.item) for draft in self.calls.values())]
            output = [item for _, item in sorted(indexed, key=lambda entry: (entry[0] is None, entry[0] or 0))]
        # Only the stream can prove an incomplete response's calls are executable.
        # The terminal snapshot can name calls to skip and supply their replay data.
        supplied = {item.get("call_id") for item in output if item["type"] == "function_call"}
        output.extend(draft.item for identifier, draft in self.calls.items() if identifier not in supplied)
        native = []
        seen = set()
        safe = allow_execute and not self.unsafe
        for item in output:
            kind = item["type"]
            if kind != "function_call":
                native.append(item)
                safe &= kind not in OTHER_TOOLS
                continue
            identifier, name = item.get("call_id"), item.get("name")
            if (not isinstance(identifier, str) or not identifier or identifier in seen
                or not isinstance(name, str) or not name):
                safe = False
                continue
            seen.add(identifier)
            draft = self.calls.get(identifier)
            executable = bool(draft and draft.executable(self.codex))
            if draft and (draft.item["name"] != name or (
                draft.item.get("id") and item.get("id") and draft.item["id"] != item["id"]
            )):
                item, executable = draft.item, False
            safe &= executable
            raw = draft.raw if executable else item.get("arguments")
            if not complete_arguments(raw):
                raw = draft.raw if draft and complete_arguments(draft.raw) else "{}"
            native.append({**item, "status": "completed", "arguments": raw})
        if not seen:
            return None, False
        if text and not any(item["type"] == "message" and item.get("content") for item in native):
            index = next(index for index, item in enumerate(native) if item["type"] == "function_call")
            native.insert(index, {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]})
        return {**response, "output": native}, safe
