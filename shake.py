"""Identify large context regions that can move to session artifacts."""

import json
import re
from dataclasses import dataclass
from hashlib import sha256


OPEN_XML = re.compile(r"<([a-z_-]+)(?:\s+[^>]*)?>")
CLOSE_XML = re.compile(r"</([a-z_-]+)>")


@dataclass(frozen=True)
class ShakePreset:
    protect_tokens: int
    min_savings: int = 0
    protect_artifact_reads: bool = True


AUTO = ShakePreset(16_000, 4_000)
MANUAL = ShakePreset(4_000, protect_artifact_reads=False)
RESCUE = ShakePreset(0)
FENCE_MIN_TOKENS = 400


def artifact_id(text):
    return sha256(text.encode("utf-8")).hexdigest()


def block_ranges(text):
    ranges = []
    fence_start, xml_start, offset = None, None, 0
    tags = []
    for line in text.split("\n"):
        end = offset + len(line)
        trimmed = line.lstrip()
        if trimmed.startswith(("```", "~~~")):
            if fence_start is None:
                fence_start = offset
            else:
                ranges.append((fence_start, end))
                fence_start = None
        elif fence_start is None:
            opening = OPEN_XML.fullmatch(line)
            closing = CLOSE_XML.fullmatch(trimmed)
            if opening:
                if not tags:
                    xml_start = offset
                tags.append(opening[1])
            elif closing and tags and tags[-1] == closing[1]:
                tags.pop()
                if not tags:
                    ranges.append((xml_start, end))
        offset = end + 1

    previous_end = 0
    for start, end in sorted(ranges):
        if start >= previous_end:
            yield start, end
            previous_end = end


def protected_result(call, preset):
    if not call:
        return True
    if call.get("name") == "skill":
        return True
    if call.get("name") != "read":
        return False
    try:
        args = json.loads(call.get("arguments", ""))
    except (ValueError, TypeError):
        return True
    path = args.get("path") if isinstance(args, dict) else None
    return isinstance(path, str) and (
        path.startswith("skill://") or (preset.protect_artifact_reads and path.startswith("artifact://"))
    )


def elide(content, ranges, count_tokens, artifacts):
    for start, end in reversed(ranges):
        original = content[start:end]
        count = count_tokens(original)
        identifier = artifact_id(original)
        replacement = f"[shaken ~{count} tokens — recover: artifact://{identifier}]"
        if count_tokens(replacement) >= count:
            continue
        artifacts[identifier] = original
        content = content[:start] + replacement + content[end:]
    return content


def plan(batches, count_tokens, preset):
    """Return message content overrides, their artifact bodies, and actual estimated savings."""
    replacements, artifacts = {}, {}
    recent = savings = 0
    for seq, batch in reversed(batches):
        calls = {
            call["id"]: call["function"]
            for message in batch if message["role"] == "assistant"
            for call in message.get("tool_calls", [])
        }
        for index in range(len(batch) - 1, -1, -1):
            message = batch[index]
            protected = recent < preset.protect_tokens
            recent += count_tokens(message)
            content = message.get("content")
            if protected or not isinstance(content, str) or not content.strip():
                continue
            if message["role"] == "tool":
                if content.startswith("[shaken ~") or protected_result(calls.get(message.get("tool_call_id")), preset):
                    continue
                ranges = [(0, len(content))]
            elif message["role"] in ("user", "assistant", "developer"):
                ranges = [
                    (start, end) for start, end in block_ranges(content)
                    if count_tokens(content[start:end]) >= FENCE_MIN_TOKENS
                ]
            else:
                continue
            replacement = elide(content, ranges, count_tokens, artifacts)
            if replacement != content:
                replacements[seq, index] = replacement
                savings += count_tokens(content) - count_tokens(replacement)

    if savings <= 0 or savings < preset.min_savings:
        return {}, {}, 0
    return replacements, artifacts, savings
