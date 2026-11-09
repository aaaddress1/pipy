"""Keep an interrupted stream's display blocks separate from its provider replay."""

import time

import model_catalog


TYPE = "interrupted-thinking"
USER_INTERRUPT = "Interrupted by user"
MIN_CHARS = 60


class StreamDraft:
    def __init__(self):
        self.blocks = []
        self.signatures = {}

    def append(self, kind, value=None, key=None):
        if kind != "other" and not value:
            return
        if not self.blocks or self.blocks[-1][:2] != (kind, key):
            self.blocks.append((kind, key, []))
        if value:
            self.blocks[-1][2].append(value)

    def snapshot(self):
        blocks = []
        for kind, key, fragments in self.blocks:
            block = {"type": kind}
            if kind != "other":
                block[kind] = "".join(fragments)
            if kind == "thinking" and key in self.signatures:
                block["thinkingSignature"] = self.signatures[key]
            blocks.append(block)
        return assistant_draft(blocks)


def assistant_draft(blocks):
    if not blocks:
        return None
    return {"content": "".join(block["text"] for block in blocks if block["type"] == "text") or None,
            "_interrupt_blocks": blocks}


def trailing_run(blocks):
    end = len(blocks)
    while end and blocks[end - 1]["type"] == "text" and not str(blocks[end - 1].get("text") or "").strip():
        end -= 1
    start = end
    while start:
        block = blocks[start - 1]
        value = block.get("thinking")
        if block["type"] != "thinking" or not isinstance(value, str) or not value.strip() or block.get("thinkingSignature"):
            break
        start -= 1
    return (start, end) if start < end else None


def reader_allows(route):
    if route is None:
        return False
    model = route.catalog_model
    if model:
        provider, name = model["provider"], model["id"]
        policy = (model_catalog.catalog_policy(provider, name, route.upstream) if model_catalog.lookup(provider, name)
                  else model_catalog.model_policy(model, route.upstream))
        return policy.get("classifierCotExtraction") is not True
    return route.api not in model_catalog.ANTHROPIC_COMPACTION_APIS and "claude" not in route.model.lower()


def note_role(route):
    if route and (route.api in model_catalog.RESPONSES_APIS or
                  ((route.catalog_model or {}).get("compat") or {}).get("supportsDeveloperRole")):
        return "developer"
    return "user"


def continuity_note(message, route):
    blocks = message.get("_interrupt_blocks") or ()
    if not reader_allows(route) or not (tail := trailing_run(blocks)):
        return None
    start, end = tail
    reasoning = "\n\n".join(block["thinking"].strip() for block in blocks[start:end])
    if len(reasoning.encode("utf-16-le", errors="surrogatepass")) < 2 * MIN_CHARS:
        return None
    now = int(time.time() * 1000)
    model = message["_rpc"]
    return {"role": "developer", "content": f"You were saying this but I interrupted you:\n```\n{reasoning}\n```",
            "_history_at": now, "_interrupted_thinking": {"interruptedAt": now, "provider": model["provider"],
                                                          "model": model["model"], "blockCount": end - start}}


def provider_assistant(message):
    blocks = message.get("_interrupt_blocks") or ()
    projected = {key: value for key, value in message.items() if key not in (
        "_interrupt_blocks", "_live_interruption", "_thinking", "_chat_thinking_route", "_anthropic", "_anthropic_import",
        "tool_calls",
    )}
    if tail := trailing_run(blocks):
        blocks = blocks[:tail[0]]
        projected["content"] = "".join(block.get("text") or "" for block in blocks if block["type"] == "text") or None
        if responses := projected.get("_responses"):
            items = [item for item in responses["items"] if item.get("role") != "assistant"
                     or not isinstance(item.get("content"), str) or item["content"].strip()]
            projected["_responses"] = {**responses, "text": projected["content"], "items": items}
    if native := message.get("_anthropic"):
        safe = [block for block in native["blocks"] if block["type"] == "redacted_thinking"
                or block["type"] == "text" and (block.get("text") or "").strip()
                or block["type"] == "thinking" and block.get("signature")]
        if any(block["type"] != "text" for block in safe):
            projected["_anthropic"] = {**native, "text": projected.get("content"), "blocks": safe}
    elif imported := message.get("_anthropic_import"):
        safe = [block for block in blocks if block["type"] in ("text", "redactedThinking")
                or block["type"] == "thinking" and block.get("thinkingSignature")]
        if any(block["type"] != "text" for block in safe):
            projected["_anthropic_import"] = {**imported, "content": safe, "stopReason": "stop"}
    if projected.get("content") or any(
        key in projected for key in ("_anthropic", "_anthropic_import", "_responses")
    ):
        projected["_interrupted_replay"] = True
        return projected
    return None
