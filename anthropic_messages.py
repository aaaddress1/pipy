"""Anthropic Messages wire adapter for the shared Mini Pi transcript."""

import asyncio
import json
import re
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256
from urllib.parse import urlsplit

import httpx
from openai import APIStatusError

import anthropic_fallback as fallback
from cancellable_http import active_turn
from model_catalog import EFFORTS
from rate_limits import classify_rate_limit, retry_after
from recovery import (
    ClassifierRefusal, ContextWindowExceeded, EmptyTurn, IncompleteTurn, StreamInterrupted, TRANSPORT_ERRORS, retry_backoff,
)
from runtime import CompactionSummary, native_provider
from snapcompact import anthropic_content


COMPACTION_BETA = {"anthropic-beta": "compact-2026-09-04"}
LEGACY_COMPACTION_BETA = "compact-2026-01-12"
BINDING_BETA = "thinking-binding-controls-2026-08-01"
CREDIT_BETA = "fallback-credit-2026-06-01"
CREDIT_TTL = 300
THINKING_BUDGETS = dict(zip(EFFORTS, (1024, 4096, 8192, 16384, 32768, 32768)))
INVALID_SIGNATURE = re.compile(r"invalid\s+\x60?signature\x60?\s+in\s+\x60?thinking\x60?(?:\s+block)?", re.I)
MISSING_SIGNATURE = re.compile(r'thinking\.signature\b[^"\n]{0,32}\brequired\b', re.I)
PREFIX_MISMATCH = re.compile(r"bound to a different conversation|prefix_mismatch_behavior", re.I)
IMMUTABLE_THINKING = re.compile(
    r"messages\.\d+\.content\.\d+.*\b(?:thinking|redacted_thinking)\b.*\blatest assistant message cannot be modified\b",
    re.I | re.S,
)
BLOCK_PATH = re.compile(r"messages\.(\d+)\.content\.(\d+)")
SIGNING_HOST = re.compile(
    r"(?:^|\.)(?:api\.anthropic\.com|githubcopilot\.com|zenmux\.ai|(?:inference|services)\.ai\.azure\.com)$"
    r"|(?:^|\.)(?:copilot-api\.|bedrock-runtime\.|bedrock-mantle\.)",
)


def official_endpoint(url):
    return url.lower() == "https://api.anthropic.com" or url.lower().startswith("https://api.anthropic.com/")


def credit_header(header=None):
    betas = [beta.strip() for beta in (header or "").split(",") if beta.strip()]
    if not any(beta.startswith("fallback-credit-") for beta in betas):
        betas.append(CREDIT_BETA)
    return ",".join(betas)


def server_tool_block(block):
    kind = block["type"]
    return kind == "server_tool_use" or kind.endswith("_tool_result")


@dataclass(frozen=True, repr=False)
class FallbackCredit:
    token: str
    body: dict
    fingerprint: str
    beta: str | None
    expires: float
    echo: tuple
    server_tools: bool

    def live(self):
        return time.monotonic() < self.expires

    def retarget(self, model, continuation):
        body = deepcopy(self.body)
        body["model"] = model
        body["fallback_credit_token"] = self.token
        body.pop("fallbacks", None)
        if continuation:
            body["messages"].append({"role": "assistant", "content": deepcopy(continuation)})
        return body

    def headers(self):
        betas = (beta.strip() for beta in (self.beta or "").split(","))
        beta = ",".join(beta for beta in betas if not beta.startswith("server-side-fallback-"))
        return {"anthropic-beta": credit_header(beta)}


def refusal_echo(blocks):
    echoed = [
        deepcopy(block) for block in blocks
        if block["type"] in ("text", "fallback") or thinking_key(block) or server_tool_block(block)
    ]
    if echoed and echoed[-1]["type"] == "text":
        echoed[-1]["text"] = echoed[-1]["text"].rstrip()
    return tuple(echoed)


class CreditRedemption:
    def __init__(self, credit):
        self.credit = credit
        self.echo = credit.echo
        self.temporary = 0
        self.transient = 0

    def rejected(self, error):
        if error.status_code == 400:
            if "redemption temporarily unavailable" in error.message:
                if self.temporary >= 2 or not self.credit.live():
                    return None
                self.temporary += 1
                return "retry", .5
            if self.echo:
                self.echo = ()
                return "retry", 0
            if "fallback_credit_token" in error.message and not self.credit.server_tools:
                return "fresh", 0
            return None
        verdict = error.response.headers.get("x-should-retry")
        if verdict == "false" or self.transient >= 10 or not self.credit.live():
            return None
        delay = retry_after(error.response)
        if (delay is not None and delay > 60) or classify_rate_limit(error, delay) in ("quota", "billing"):
            return None
        if verdict == "true" or error.status_code in (408, 429) or error.status_code >= 500:
            self.transient += 1
            return "retry", max(delay or 0, retry_backoff(self.transient))
        return None


def credit_wait(delay, cancel):
    if delay:
        if turn := active_turn():
            turn.wait(delay)
        else:
            time.sleep(delay)
    if cancel:
        cancel()


def thinking_key(block):
    if block["type"] == "thinking" and isinstance(signature := block.get("signature"), str) and signature.strip():
        return "thinking", signature
    if block["type"] == "redacted_thinking" and isinstance(data := block.get("data"), str) and data.strip():
        return "redacted", data
    return None


@dataclass
class ThinkingReplay:
    demote_unsigned: bool = False
    drop_all: bool = False
    dropped: set = field(default_factory=set)
    reported: set = field(default_factory=set)


def record_transformations(body, replay, transformations, value, *, replace=False):
    if value is None:
        return
    if replace:
        transformations.clear()
    if not isinstance(value, list):
        return
    paths = []
    for entry in value:
        if not isinstance(entry, dict) or not isinstance(entry.get("type"), str):
            continue
        key = encode(entry)
        if key in transformations:
            continue
        transformations[key] = entry
        if entry.get("reason") != "prefix_binding_mismatch":
            continue
        path = entry.get("path")
        key = None
        if isinstance(path, str):
            match = BLOCK_PATH.fullmatch(path)
            if match:
                mi, bi = map(int, match.groups())
                if mi < len(body["messages"]) and bi < len(content := body["messages"][mi]["content"]):
                    key = thinking_key(content[bi])
            key = key or ("path", path)
            if key in replay.reported:
                continue
            replay.reported.add(key)
        paths.append(path)
    if paths:
        print(f"[Anthropic: provider dropped {len(paths)} thinking block(s) after a prefix mismatch]", file=sys.stderr)


class ThinkingRepair:
    def __init__(self, replay=None):
        self.replay = replay if replay is not None else ThinkingReplay()
        self.demote_unsigned = self.replay.demote_unsigned
        self.drop_all = self.replay.drop_all
        self.prefix_retry = False
        self.latest_retry = False
        self.drop_request = False

    def retry(self, route, body, error):
        text = str(error)
        if re.search(r"\bcompaction_[a-z_]+\b", text, re.I):
            return False
        invalid = INVALID_SIGNATURE.search(text)
        if invalid and PREFIX_MISMATCH.search(text):
            if self.prefix_retry or route.thinking_prefix_mismatch == "error":
                return False
            self.prefix_retry = True
            blocks = [(mi, bi, key) for mi, message in enumerate(body["messages"])
                      for bi, block in enumerate(message["content"]) if (key := thinking_key(block))]
            match = BLOCK_PATH.search(text)
            if match or blocks:
                start = tuple(map(int, match.groups())) if match else blocks[0][:2]
                self.replay.dropped.update(key for mi, bi, key in blocks if (mi, bi) >= start)
            else:
                self.drop_request = True
            print("[Anthropic: repaired thinking prefix for retry]", file=sys.stderr)
            return True
        if IMMUTABLE_THINKING.search(text):
            if self.latest_retry:
                return False
            keys = set()
            for message in reversed(body["messages"]):
                content = message["content"]
                if message["role"] == "assistant":
                    keys.update(key for block in content if (key := thinking_key(block)))
                elif not content or any(block["type"] != "tool_result" for block in content):
                    break
            if not keys:
                return False
            self.latest_retry = True
            self.replay.dropped.update(keys)
            print("[Anthropic: removed rejected latest-turn thinking from retry]", file=sys.stderr)
            return True
        if not (invalid or MISSING_SIGNATURE.search(text)):
            return False
        if not self.demote_unsigned:
            self.demote_unsigned = True
            self.replay.demote_unsigned = True
            print("[Anthropic: demoting unsigned thinking for retry]", file=sys.stderr)
            return True
        if not (self.drop_request or self.drop_all):
            self.drop_all = True
            self.replay.drop_all = True
            print("[Anthropic: retrying without thinking history]", file=sys.stderr)
            return True
        return False


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def endpoint(base_url):
    return base_url + ("/messages" if base_url.endswith("/v1") else "/v1/messages")


def headers(key, auth):
    credential = {"authorization": f"Bearer {key}"} if auth == "bearer" else {"x-api-key": key}
    return {"anthropic-version": "2023-06-01", **credential}


def text_block(value):
    return {"type": "text", "text": value}


class WireHistory:
    def __init__(self, system, tools, effort):
        self.messages = []
        self.digest = sha256(encode([system, tools, effort]).encode("utf-8"))

    def add(self, role, content, *, merge=True):
        if not content:
            return
        if merge and self.messages and self.messages[-1]["role"] == role:
            self.messages[-1]["content"].extend(content)
            return
        if self.messages:
            self.update(self.digest, self.messages[-1])
        self.messages.append({"role": role, "content": content if isinstance(content, str) else list(content)})

    @staticmethod
    def update(digest, message):
        digest.update(encode(message).encode("utf-8") + b"\n")

    def fingerprint(self):
        copy = self.digest.copy()
        if self.messages:
            self.update(copy, self.messages[-1])
        return copy.hexdigest()


def tool_use_block(call):
    function = call["function"]
    try:
        arguments = json.loads(function["arguments"])
    except (ValueError, TypeError):
        arguments = {}
    return {"type": "tool_use", "id": call["id"], "name": function["name"],
            "input": arguments if isinstance(arguments, dict) else {}}


def assistant_content(message, identity, fingerprint):
    text = message.get("content")
    calls = message.get("tool_calls", [])
    saved = message.get("_anthropic")
    if saved and saved["route"] == identity and saved["prefix"] == fingerprint and text == saved["text"]:
        return saved["blocks"]
    blocks = [text_block(text)] if text else []
    blocks.extend(tool_use_block(call) for call in calls)
    return blocks


def imported_assistant_content(message, provider, *, keep_thinking):
    saved = message.get("_anthropic_import")
    if not saved or saved["provider"] != provider:
        return None
    blocks = []
    calls = {call["id"]: call for call in message.get("tool_calls", ())}
    keep_thinking = keep_thinking and saved.get("stopReason") not in ("aborted", "error")
    for item in saved["content"]:
        kind = item.get("type")
        if kind == "text" and isinstance(text := item.get("text"), str) and text.strip():
            blocks.append(text_block(text))
        elif kind == "toolCall" and isinstance(identifier := item.get("id"), str) and identifier in calls:
            blocks.append(tool_use_block(calls[identifier]))
        elif kind == "thinking" and keep_thinking and isinstance(item.get("thinking"), str):
            blocks.append({"type": "thinking", "thinking": item["thinking"], "signature": item.get("thinkingSignature", "")})
        elif kind == "redactedThinking" and keep_thinking and isinstance(item.get("data"), str):
            blocks.append({"type": "redacted_thinking", "data": item["data"]})
        elif kind == "fallback":
            blocks.append(item)
        elif kind == "anthropicServerTool" and isinstance(block := item.get("block"), dict):
            if isinstance(block.get("type"), str) and server_tool_block(block):
                blocks.append(block)
    return blocks


def compaction_files(summary):
    metadata = summary.get("anthropic_metadata", {})
    files = metadata.get("filesText")
    if not metadata.get("exactTail"):
        return [], files
    retained = list(metadata.get("retainedFiles", ()))
    if files and (after := summary.get("anthropic_committed_at")) is not None:
        retained.append({"text": files, "after": after})
    return sorted(retained, key=lambda entry: entry["after"]), None


def preserved_content(message, vision):
    content = message["content"]
    parts = message.get("_history_text_parts")
    if vision and parts is not None and "\n".join(parts) == content:
        return [text_block(part) for part in parts if part.strip()]
    if isinstance(content, str):
        return content if content.strip() else ""
    blocks = anthropic_content(content)
    if blocks and all(block["type"] == "image" for block in blocks):
        blocks.insert(0, text_block("(see attached image)"))
    return blocks


def allows_unsigned(route, model, effort):
    if route.replay_unsigned_thinking is not None:
        return route.replay_unsigned_thinking
    url = urlsplit(route.endpoint())
    host = url.hostname or ""
    signing = SIGNING_HOST.search(host) or (
        host == "gateway.ai.cloudflare.com" and "/anthropic" in url.path
        or host.endswith("aiplatform.googleapis.com") and "/publishers/anthropic/" in url.path
    )
    return not signing and (effort is not None or "claude" in model.lower() or host.endswith("deepseek.com"))


def replay_content(blocks, repair, unsigned, model):
    replay = repair.replay
    signed = any(block["type"] == "thinking" and thinking_key(block) for block in blocks)
    for block in blocks:
        kind = block["type"]
        if kind not in ("thinking", "redacted_thinking"):
            yield block
        elif not (repair.drop_all or repair.drop_request):
            key = thinking_key(block)
            if key and key not in replay.dropped:
                yield block
            elif not key and kind == "thinking" and (text := block.get("thinking") or "").strip():
                if unsigned and not repair.demote_unsigned and not signed:
                    yield {"type": "thinking", "thinking": text, "signature": ""}
                else:
                    yield text_block(text if "claude" in model.lower() else f"<think>\n{text}\n</think>")


def apply_thinking(body, catalog, effort):
    if not catalog:
        if effort in ("off", "none"):
            body["thinking"] = {"type": "disabled"}
        elif effort:
            body["thinking"] = {"type": "adaptive"}
            if effort != "auto":
                body["output_config"] = {"effort": effort}
        return
    if not effort:
        return
    compat = catalog.get("compat") or {}
    thinking = catalog.get("thinking") or {}
    mode = thinking.get("mode")
    mapping = thinking.get("effortMap") or {}
    adaptive = mode == "anthropic-adaptive" and not compat.get("disableAdaptiveThinking")
    levels = thinking.get("efforts") or ()
    tag_only = adaptive and levels and all(mapping.get(level) == "adaptive" for level in levels)
    if effort in ("off", "none"):
        if compat.get("supportsBetweenToolsThinking"):
            body["thinking"] = {"type": "between_tools"}
            if pinned := compat.get("betweenToolsEffort"):
                body["output_config"] = {"effort": pinned}
        elif adaptive and not tag_only:
            body["output_config"] = {"effort": "low"}
        elif compat.get("disabledThinking") != "omit":
            body["thinking"] = {"type": "disabled"}
        return
    wire_effort = mapping.get(effort, effort)
    if wire_effort == "minimal":
        wire_effort = "low"
    if adaptive:
        body["thinking"] = {"type": "adaptive"}
        if thinking.get("supportsDisplay"):
            body["thinking"]["display"] = "summarized"
    else:
        level = thinking.get("defaultLevel") or "medium" if effort == "auto" else effort
        budget = (thinking.get("effortBudgets") or {}).get(level) or THINKING_BUDGETS.get(level, 1024)
        budget = min(budget, body["max_tokens"] - 4000)
        if budget < 1024:
            body["thinking"] = {"type": "disabled"}
            return
        body["thinking"] = {"type": "enabled", "budget_tokens": budget}
        if thinking.get("supportsDisplay") is not False:
            body["thinking"]["display"] = "summarized"
    if (adaptive or mode == "anthropic-budget-effort") and wire_effort not in ("auto", "adaptive"):
        body["output_config"] = {"effort": wire_effort}


def request_body(route, model, messages, schemas, effort, *, stream=True, max_tokens=None, repair=None, metadata=None,
                 compacting=False, files_due_before=float("inf")):
    identity = route.identity() if model == route.model else None
    repair = repair or ThinkingRepair()
    fallbacks = route.server_side_fallbacks if stream and identity else ()
    replay_handoff = bool(fallbacks and official_endpoint(route.endpoint()))
    unsigned = allows_unsigned(route, model, effort)
    system = "\n\n".join(message["content"] for message in messages if message["role"] == "system")
    tools = [
        {
            "name": schema["function"]["name"],
            "description": schema["function"]["description"],
            "input_schema": schema["function"]["parameters"],
        }
        for schema in schemas
    ]
    history = WireHistory(system, tools, effort)
    pending_files, legacy_files, tool_images = [], None, []
    preserved = False
    rewrite_at = float("-inf")
    legacy_replay = False
    provider = native_provider(route) if identity and route.native_compaction else None

    def flush_files(before):
        for entry in pending_files:
            if entry["after"] < before and entry["text"].strip():
                history.add("user", entry["text"], merge=False)
        pending_files[:] = [entry for entry in pending_files if entry["after"] >= before]

    for index, message in enumerate(messages):
        role = message["role"]
        if role == "system":
            continue
        summary = message.get("_anthropic_compaction")
        if provider and summary and not history.messages and (
            summary.get("route") == identity or summary.get("anthropic_provider") == provider
        ):
            block = summary["block"]
            legacy = not block.get("signature") and bool(block.get("encrypted_content"))
            if not (compacting and legacy):
                history.add("assistant", [block])
                legacy_replay = legacy
                preserved = "anthropic_provider" in summary
                pending_files, legacy_files = compaction_files(summary)
                if preserved and not summary.get("anthropic_metadata", {}).get("exactTail") and index + 1 < len(messages):
                    first = messages[index + 1]
                    rewrite_at = first.get("_journal_at", first.get("_history_at", float("-inf"))) - 1
                continue
        when = message.get("_history_at", float("inf"))
        if role != "tool":
            flush_files(when)
        if role == "user" and legacy_files is not None:
            if legacy_files.strip():
                history.add("user", legacy_files, merge=False)
            legacy_files = None
        if role == "assistant":
            blocks = imported_assistant_content(message, provider, keep_thinking=when > rewrite_at) if preserved else None
            if blocks is None:
                blocks = assistant_content(message, identity, history.fingerprint())
            if not replay_handoff:
                blocks = fallback.collapse_segments(blocks)
            wire = fallback.tool_calls_last(list(replay_content(blocks, repair, unsigned, model)))
            if preserved and wire and (last := history.messages[-1])["role"] == "assistant":
                if len(last["content"]) != 1 or last["content"][0]["type"] != "compaction":
                    history.add("user", "Continue.", merge=False)
            history.add(role, wire)
        elif role == "tool":
            content = preserved_content(message, route.vision) if preserved else (
                anthropic_content(message["content"]) if isinstance(message.get("content"), list) else message.get("content")
            )
            previous_tool = index > 0 and messages[index - 1]["role"] == "tool"
            error = message.get("_tool_error")
            if preserved and error and isinstance(content, list):
                tool_images.extend(block for block in content if block["type"] == "image")
                content = [block for block in content if block["type"] != "image"]
            if preserved and error and not content:
                content = "Tool failed with no output."
            result = {
                "type": "tool_result", "tool_use_id": message["tool_call_id"],
                "content": content or "",
            }
            if error or preserved and error is False:
                result["is_error"] = error
            history.add("user", [result], merge=not preserved or previous_tool)
        elif role == "user" and message.get("content"):
            history.add("user", preserved_content(message, route.vision) if preserved else anthropic_content(message["content"]),
                        merge=not preserved)
        tools_complete = role == "tool" and (index + 1 == len(messages) or messages[index + 1]["role"] != "tool")
        if tools_complete and tool_images:
            history.add("user", [text_block("Attached image(s) from the tool result(s) above:"), *tool_images])
            tool_images.clear()
        if legacy_files is not None and (tools_complete or role == "assistant" and not any(
            block["type"] == "tool_use" for block in wire
        )):
            if legacy_files.strip():
                history.add("user", legacy_files, merge=False)
            legacy_files = None
    flush_files(files_due_before)
    if not compacting and legacy_files and legacy_files.strip():
        history.add("user", legacy_files, merge=False)
    if preserved and not compacting and history.messages[-1]["role"] == "assistant":
        history.add("user", "Continue.", merge=False)
    body = {
        "model": route.wire_model(effort) if identity else model,
        "max_tokens": max_tokens or route.max_output_tokens,
        "messages": history.messages, "stream": stream,
    }
    if system:
        body["system"] = system
    if tools:
        body["tools"] = tools
    if metadata:
        body["metadata"] = metadata
    if legacy_replay:
        window = (route.catalog_model or {}).get("contextWindow") or route.window or 0
        body["context_management"] = {"edits": [{"type": "compact_20260112", "trigger": {
            "type": "input_tokens", "value": max(50_000, window),
        }}]}
    apply_thinking(body, route.catalog_model if identity else None, effort)
    if route.thinking_prefix_mismatch and not repair.prefix_retry and effort not in ("off", "none"):
        body.setdefault("thinking", {"type": "adaptive"})["block_binding"] = {
            "prefix_mismatch_behavior": route.thinking_prefix_mismatch,
        }
    if route.cache_retention != "none":
        body["cache_control"] = {"type": "ephemeral", "ttl": route.cache_retention}
    if fallbacks:
        body["fallbacks"] = list(fallbacks)
    return body, history.fingerprint()


def status_error(response, body):
    error = body.get("error", body) if isinstance(body, dict) else {}
    if not isinstance(error, dict):
        error = {}
    message = error.get("message") or f"Anthropic Messages HTTP {response.status_code}"
    return APIStatusError(str(message), response=response, body=error)


def require_success(response):
    if not response.is_success:
        response.read()
        try:
            error = response.json()
        except ValueError:
            error = {}
        raise status_error(response, error)


def compact(client, route, messages, schemas, effort, instructions, *, replay=None, meter=None, metadata=None,
            files_due_before=float("inf")):
    repair = ThinkingRepair(replay)
    while True:
        body, _ = request_body(route, route.model, messages, schemas, effort, stream=False, repair=repair,
                               metadata=metadata, compacting=True, files_due_before=files_due_before)
        body["compaction"] = {"type": "summarize", "instructions": instructions}
        if meter:
            meter.admit()
        response = client.post(endpoint(route.endpoint()), json=body, headers=replay_headers(body, route), timeout=300)
        try:
            require_success(response)
        except APIStatusError as error:
            if not repair.retry(route, body, error):
                raise
        else:
            break
    result = response.json()
    if meter:
        meter.observe(result.get("usage"))
    blocks = result.get("content") or []
    if result.get("stop_reason") != "compaction" or len(blocks) != 1:
        raise RuntimeError("Anthropic did not return one native compaction block.")
    block = blocks[0]
    if block.get("type") != "compaction" or not all(
        isinstance(block.get(field), str) and block[field].strip() for field in ("content", "signature")
    ):
        raise RuntimeError("Anthropic native compaction block is unsigned or empty.")
    native = {"route": route.identity(), "block": block}
    prior = next((message["_anthropic_compaction"] for message in messages if "_anthropic_compaction" in message), {})
    if prior.get("anthropic_provider") == native_provider(route) and prior.get("block", {}).get("signature"):
        files, _ = compaction_files(prior)
        retained = [entry for entry in files if entry["after"] >= files_due_before]
        native.update(anthropic_provider=prior["anthropic_provider"],
                      anthropic_metadata={"exactTail": True, "retainedFiles": retained, "model": route.model})
    return CompactionSummary(block["content"], native)


def event_data(lines):
    frame = []
    for line in lines:
        if not line:
            if frame:
                yield json.loads("\n".join(frame))
                frame.clear()
        elif line.startswith("data:"):
            frame.append(line[5:].lstrip(" "))


async def async_event_data(lines):
    frame = []
    async for line in lines:
        if not line:
            if frame:
                yield json.loads("\n".join(frame))
                frame.clear()
        elif line.startswith("data:"):
            frame.append(line[5:].lstrip(" "))


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cache_ttl: str | None
    served_model: str | None = None
    cost: dict | None = None

    @classmethod
    def from_wire(cls, raw, *, served_model=None, cost=None):
        reads = raw.get("cache_read_input_tokens") or 0
        writes = raw.get("cache_creation_input_tokens") or 0
        creation = raw.get("cache_creation") or {}
        tier = "1h" if creation.get("ephemeral_1h_input_tokens") else (
            "5m" if creation.get("ephemeral_5m_input_tokens") else None
        )
        return cls((raw.get("input_tokens") or 0) + reads + writes, raw.get("output_tokens") or 0, reads, writes,
                   tier, served_model, cost)


class MessageStream:
    def __init__(self, guard, show, turn=None, *, fallback_enabled=False, tracker=None, on_text=None, speculation=None):
        self.guard = guard
        self.on_text = on_text
        self.show = show
        self.turn = turn
        self.tracker = tracker
        self.speculation = speculation
        self.blocks = {}
        self.open_blocks = set()
        self.arguments = {}
        self.fragments = {}
        self.text = []
        self.started = False
        self.stopped = False
        self.reason = None
        self.stop_details = {}
        self.usage = {}
        self.transformations = {}
        self.continuation = ()
        self.fallback_enabled = fallback_enabled
        self.served_model = None

    def update_usage(self, raw):
        self.usage.update(raw)
        if self.fallback_enabled and (served := fallback.served_model(raw)):
            self.served_model = served

    def usage_snapshot(self, route, model):
        cost = None
        if self.fallback_enabled:
            first_party = route.broker_provider == "anthropic" or official_endpoint(route.endpoint())
            cost = fallback.iteration_cost(self.usage, model, route.cache_prices, first_party)
        return Usage.from_wire(self.usage, served_model=self.served_model, cost=cost)

    def unfinished(self):
        return bool(self.open_blocks & fallback.retained(self.blocks).keys())

    def interrupted(self, message):
        return StreamInterrupted(
            message, text="".join(self.text),
            tool_draft=any(block["type"] == "tool_use" for block in fallback.retained(self.blocks).values()),
            usage=Usage.from_wire(self.usage),
        )

    def visible(self, text):
        self.guard.push(None, text, False)
        self.text.append(text)
        if self.turn:
            self.turn.delta("text", text)
        if self.on_text and text:
            self.on_text(text)
        if self.show and text:
            print(text, end="", flush=True)
        if self.tracker:
            self.tracker.text("text", text)

    def inspect_call(self, index, *, final=False):
        checking = self.tracker and self.tracker.rules.stream_tools
        if not checking and not self.speculation:
            return
        block = self.blocks[index]
        if block["type"] != "tool_use":
            return
        fragments = self.arguments.get(index)
        raw = "".join(fragments) if fragments else json.dumps(block.get("input") or {})
        if checking:
            self.tracker.tool(index, block.get("name", ""), raw, block.get("id"), final=final)
        if self.speculation:
            self.speculation.observe(index, block.get("name", ""), block.get("id"), raw, final=final)

    def push(self, event, response):
        kind = event.get("type")
        if kind == "error":
            error = event.get("error", {})
            status = {
                "overloaded_error": 529, "api_error": 500, "rate_limit_error": 429,
                "authentication_error": 401, "permission_error": 403,
            }.get(error.get("type"), 400)
            if status >= 500 and self.text:
                raise self.interrupted(str(error.get("message", "Anthropic stream error.")))
            raise status_error(httpx.Response(status, headers=response.headers, request=response.request), event)
        if kind == "message_start":
            if self.started:
                raise self.interrupted("Duplicate Anthropic message start.")
            self.started = True
            self.update_usage(event["message"].get("usage") or {})
        elif kind == "content_block_start":
            index = event["index"]
            if not self.started or index in self.blocks:
                raise self.interrupted("Unexpected Anthropic content block start.")
            block = dict(event["content_block"])
            if block["type"] == "fallback":
                block = fallback.marker(block) or {"type": "_ignored_fallback"}
                if self.fallback_enabled and block["type"] == "fallback":
                    self.served_model = block["to"]["model"]
                    if self.tracker:
                        self.tracker.restart()
                    if self.speculation:
                        self.speculation.restart()
            self.blocks[index] = block
            self.open_blocks.add(index)
            if block["type"] == "text":
                self.visible(block.get("text") or "")
            elif block["type"] == "tool_use":
                self.guard.push(None, None, True)
                if self.speculation:
                    self.speculation.observe(index, block.get("name", ""), block.get("id"), None)
            elif block["type"] == "thinking":
                self.guard.push(block.get("thinking"), None, False)
                if self.tracker:
                    self.tracker.text("thinking", block.get("thinking"))
        elif kind == "content_block_delta":
            index = event["index"]
            if index not in self.open_blocks:
                raise self.interrupted("Anthropic sent a delta for a closed block.")
            delta = event["delta"]
            delta_type = delta["type"]
            if delta_type == "input_json_delta":
                self.arguments.setdefault(index, []).append(delta["partial_json"])
                self.inspect_call(index)
            elif delta_type == "text_delta":
                self.fragments.setdefault((index, "text"), []).append(delta["text"])
                self.visible(delta["text"])
            elif delta_type in ("thinking_delta", "signature_delta"):
                field = delta_type.removesuffix("_delta")
                self.fragments.setdefault((index, field), []).append(delta[field])
                if field == "thinking":
                    self.guard.push(delta[field], None, False)
                    if self.turn:
                        self.turn.delta("thinking", delta[field])
                    if self.tracker:
                        self.tracker.text("thinking", delta[field])
        elif kind == "content_block_stop":
            index = event["index"]
            if index not in self.open_blocks:
                raise self.interrupted("Anthropic stopped an unknown content block.")
            self.open_blocks.remove(index)
            block = self.blocks[index]
            if block["type"] == "tool_use":
                self.inspect_call(index, final=True)
            elif self.tracker and block["type"] in ("text", "thinking"):
                self.tracker.end_text(block["type"])
            for field in ("text", "thinking", "signature"):
                fragments = self.fragments.pop((index, field), None)
                if fragments:
                    block[field] = (block.get(field) or "") + "".join(fragments)
            if block["type"] == "server_tool_use" and index in self.arguments:
                try:
                    arguments = json.loads("".join(self.arguments.pop(index)))
                except ValueError:
                    arguments = None
                if isinstance(arguments, dict):
                    block["input"] = arguments
        elif kind == "message_delta":
            self.reason = event["delta"].get("stop_reason") or self.reason
            if isinstance(details := event["delta"].get("stop_details"), dict):
                self.stop_details = details
            self.update_usage(event.get("usage") or {})
        elif kind == "message_stop":
            self.stopped = True

    def result(self, route, model, fingerprint, *, text_only=False):
        if not self.started or not self.stopped or self.unfinished() or not self.reason:
            raise self.interrupted("Anthropic stream ended without a complete turn.")
        self.guard.finish()
        visible = "".join(self.text)
        usage = self.usage_snapshot(route, model)
        if self.reason == "model_context_window_exceeded":
            raise ContextWindowExceeded("Model reached its context window.", text=visible, usage=usage)
        if self.reason == "max_tokens":
            raise IncompleteTurn(
                "Model reached its generation limit.", text=visible,
                tool_draft=any(block["type"] == "tool_use" for block in fallback.retained(self.blocks).values()),
                usage=usage,
            )
        if self.reason not in ("end_turn", "stop_sequence", "tool_use"):
            raise RuntimeError(f"Anthropic stopped with {self.reason!r}.")
        if text_only:
            text = "\n".join(
                block["text"] for _, block in sorted(self.blocks.items()) if block["type"] == "text"
            )
            return {"role": "assistant", "content": text or None}, usage

        message = self.assistant(route, model, fingerprint)
        calls = message.get("tool_calls", [])
        if bool(calls) != (self.reason == "tool_use"):
            raise self.interrupted("Anthropic returned unmatched tool calls.")
        if not calls and not visible.strip() and not any(
            block["type"] == "thinking" and (block.get("signature") or "").strip()
            for block in (message.get("_anthropic") or {}).get("blocks", ())
        ):
            raise EmptyTurn("Model stopped without actionable output.")
        return message, usage

    def assistant(self, route, model, fingerprint):
        calls, blocks = [], list(self.continuation)
        for index, block in sorted(fallback.retained(self.blocks).items()):
            kind = block["type"]
            if kind == "tool_use":
                if index in self.arguments:
                    try:
                        block["input"] = json.loads("".join(self.arguments[index]))
                    except ValueError as error:
                        raise self.interrupted("Anthropic tool arguments are incomplete.") from error
                if not isinstance(block.get("input"), dict) or not block.get("id") or not block.get("name"):
                    raise self.interrupted("Anthropic tool call is malformed.")
                calls.append({
                    "id": block["id"], "type": "function",
                    "function": {"name": block["name"], "arguments": encode(block["input"])},
                })
            if (
                kind in ("text", "tool_use") or thinking_key(block)
                or kind == "thinking" and (block.get("thinking") or "").strip()
                or kind == "fallback" or server_tool_block(block)
            ):
                blocks.append(block)
        if len({call["id"] for call in calls}) != len(calls):
            raise self.interrupted("Anthropic returned duplicate tool calls.")
        message = {"role": "assistant", "content": "".join(self.text) or None}
        if calls:
            message["tool_calls"] = calls
        if model == route.model and (self.continuation or any(
            block["type"] in ("thinking", "redacted_thinking", "fallback") or server_tool_block(block)
            for block in blocks
        )):
            message["_anthropic"] = {
                "route": route.identity(), "prefix": fingerprint, "text": message["content"], "blocks": blocks,
            }
            if self.served_model:
                message["_anthropic"]["served_model"] = self.served_model
        if self.transformations:
            message["_input_transformations"] = list(self.transformations.values())
        return message

    def replayable_tool_turn(self, route, model, fingerprint):
        if self.unfinished() or "".join(self.text).strip() or any(
            block["type"] not in ("text", "thinking", "redacted_thinking", "tool_use", "fallback")
            for block in fallback.retained(self.blocks).values()
        ):
            return None
        try:
            assistant = self.assistant(route, model, fingerprint)
        except StreamInterrupted:
            return None
        return assistant if assistant.get("tool_calls") else None

    def refusal(self, route, model, body, fingerprint, beta):
        label = "Refusal" if self.reason == "refusal" else "Content flagged by safety filters"
        if category := self.stop_details.get("category"):
            label += f" ({category})"
        explanation = self.stop_details.get("explanation")
        blocks = tuple(block for _, block in sorted(fallback.retained(self.blocks).items()))
        unexecuted = self.replayable_tool_turn(route, model, fingerprint)
        safe = not "".join(self.text).strip() and all(
            block["type"] in ("text", "thinking", "redacted_thinking", "fallback")
            or block["type"] == "tool_use" and unexecuted is not None
            for block in blocks
        )
        token = self.stop_details.get("fallback_credit_token")
        credit = None
        if self.stop_details.get("type") == "refusal" and isinstance(token, str) and token:
            prefill = self.stop_details.get("fallback_has_prefill_claim") is not False
            credit = FallbackCredit(token, deepcopy(body), fingerprint, beta, time.monotonic() + CREDIT_TTL,
                                    refusal_echo(blocks) if prefill else (), any(map(server_tool_block, blocks)))
            if isinstance(explanation, str):
                explanation = explanation.replace(token, "[redacted]")
        message = f"{label}: {explanation.strip()}" if isinstance(explanation, str) and explanation.strip() else label
        return ClassifierRefusal(
            message, safe=safe, credit=credit, unexecuted_turn=unexecuted, refused_model=self.served_model or model,
        )


def replay_headers(body, route=None):
    content = body["messages"][0]["content"] if body["messages"] else None
    betas = []
    signed = isinstance(content, list) and content and content[0].get("type") == "compaction" and content[0].get("signature")
    if body.get("compaction") or signed:
        betas.append(COMPACTION_BETA["anthropic-beta"])
    if body.get("context_management"):
        betas.append(LEGACY_COMPACTION_BETA)
    if "block_binding" in body.get("thinking", {}):
        betas.append(BINDING_BETA)
    betas.extend(fallback.chain_betas(body.get("fallbacks")))
    if route and route.catalog_model and official_endpoint(route.endpoint()):
        if body.get("output_config", {}).get("effort"):
            betas.append("effort-2025-11-24")
        if body.get("thinking", {}).get("type") == "enabled":
            betas.append("interleaved-thinking-2025-05-14")
    return {"anthropic-beta": ",".join(dict.fromkeys(betas))} if betas else None


def stream_attempt(
    client, route, model, messages, schemas, effort, guard, show, cache_observer=None, *, tool_choice=None, cancel=None,
    turn=None, usage_observer=None, replay=None, credit=None, admit=None, tracker=None, on_text=None, speculation=None,
    metadata=None,
):
    repair = ThinkingRepair(replay)
    redemption = CreditRedemption(credit) if credit else None
    while True:
        if cancel:
            cancel()
        body, fingerprint = request_body(route, model, messages, schemas, effort, repair=repair, metadata=metadata)
        if tool_choice and schemas:
            body["tool_choice"] = ({"type": "tool", "name": tool_choice["function"]["name"]}
                                   if isinstance(tool_choice, dict) else {"type": tool_choice})
        headers = replay_headers(body, route) or {}
        if route.fallback_credit is True or (
            route.fallback_credit is None and official_endpoint(route.endpoint()) and client.headers.get("x-api-key")
        ):
            headers["anthropic-beta"] = credit_header(headers.get("anthropic-beta"))
        if redemption and (
            not credit.live() or body.get("system") != credit.body.get("system")
            or body.get("tools") != credit.body.get("tools")
        ):
            if credit.server_tools:
                raise RuntimeError("Fallback credit became unavailable after provider tools ran.")
            redemption = None
        if redemption:
            body = credit.retarget(body["model"], redemption.echo)
            fingerprint = credit.fingerprint
            headers = credit.headers()
        message = MessageStream(guard, show, turn, fallback_enabled=bool(body.get("fallbacks")),
                                tracker=tracker, on_text=on_text, speculation=speculation)
        try:
            if admit:
                admit()
            return stream_once(client, route, model, body, fingerprint, message, None if redemption else cache_observer,
                               tool_choice, cancel, usage_observer, repair.replay, headers,
                               continuation=redemption.echo if redemption else ())
        except APIStatusError as error:
            if redemption and credit.server_tools:
                error.replay_unsafe = True
            if redemption and credit.token in error.message:
                error.message = error.message.replace(credit.token, "[redacted]")
                error.args = (error.message,)
            if message.blocks:
                raise
            if redemption:
                if decision := redemption.rejected(error):
                    action, delay = decision
                    if action == "retry":
                        credit_wait(delay, cancel)
                        continue
                    redemption = None
                    repair.drop_request = True
                    continue
                if error.status_code == 400 and not credit.server_tools and (
                    "redemption temporarily unavailable" not in error.message and repair.retry(route, body, error)
                ):
                    redemption = None
                    continue
                raise
            if not repair.retry(route, body, error):
                raise
        except StreamInterrupted as error:
            if redemption and credit.server_tools:
                error.replay_unsafe = True
            raise


def stream_once(client, route, model, body, fingerprint, message, cache_observer, tool_choice, cancel, usage_observer,
                replay, headers, *, continuation=()):
    sent_at = time.monotonic()
    completed = False
    if cache_observer:
        cache_observer(sent_at, body, None, False)
    try:
        with client.stream(
            "POST", endpoint(route.endpoint()), json=body, headers=headers,
            timeout=120 if message.show else 90,
        ) as response:
            require_success(response)
            for event in event_data(response.iter_lines()):
                if cancel:
                    cancel()
                message.push(event, response)
                if event.get("type") == "message_start":
                    record_transformations(body, replay, message.transformations,
                                           event["message"].get("input_transformations"))
                    if cache_observer:
                        cache_observer(sent_at, body, message.usage_snapshot(route, model), False)
                elif event.get("type") == "message_delta":
                    record_transformations(body, replay, message.transformations, event.get("input_transformations"),
                                           replace=True)
                if message.stopped:
                    break
        # Some gateways close after the terminal delta without a message_stop.
        if message.reason and not message.unfinished():
            message.stopped = True
        if not message.stopped and tool_choice is None:
            error = message.interrupted("Anthropic stream ended without a complete turn.")
            error.unexecuted_turn = message.replayable_tool_turn(route, model, fingerprint)
            raise error
        if message.started and message.stopped and not message.unfinished() and message.reason in ("refusal", "sensitive"):
            raise message.refusal(route, model, body, fingerprint, response.request.headers.get("anthropic-beta"))
        if continuation:
            message.continuation = continuation
            message.text[:0] = [block["text"] for block in continuation if block["type"] == "text"]
        result = message.result(route, model, fingerprint, text_only=tool_choice in ("none", "auto"))
        if cache_observer:
            cache_observer(sent_at, body, result[1], True)
        completed = True
        return result
    except TRANSPORT_ERRORS as error:
        interrupted = message.interrupted(f"Anthropic stream interrupted ({type(error).__name__}).")
        if tool_choice is None:
            interrupted.unexecuted_turn = message.replayable_tool_turn(route, model, fingerprint)
        raise interrupted from error
    except (ValueError, KeyError, TypeError) as error:
        raise message.interrupted(f"Anthropic stream interrupted ({type(error).__name__}).") from error
    finally:
        if usage_observer and message.usage:
            if message.fallback_enabled:
                usage_observer(message.usage, iteration_cost=message.usage_snapshot(route, model).cost)
            else:
                usage_observer(message.usage)
        if cache_observer and not completed:
            cache_observer(sent_at, body, None, False)
        if message.show and message.text:
            print()


async def warm(url, credentials, body, *, route=None, meter=None, **client_settings):
    replay = {**body, "max_tokens": 1}
    usage = None
    try:
        async with httpx.AsyncClient(headers=credentials, timeout=25, **client_settings) as client, asyncio.timeout(25):
            if meter:
                meter.admit()
            async with client.stream("POST", url, json=replay, headers=replay_headers(replay, route)) as response:
                if not response.is_success:
                    await response.aread()
                    require_success(response)
                async for event in async_event_data(response.aiter_lines()):
                    kind = event.get("type")
                    if kind == "error":
                        raise RuntimeError("Anthropic rejected a cache refresh.")
                    if kind == "message_start":
                        usage = dict(event["message"].get("usage") or {})
                    elif kind == "message_delta" and usage is not None:
                        usage.update(event.get("usage") or {})
                    elif kind in ("content_block_delta", "message_stop"):
                        if usage is not None:
                            return Usage.from_wire(usage)
                        break
                    elif kind == "content_block_start" and usage is not None:
                        block = event["content_block"]
                        if block["type"] in ("tool_use", "redacted_thinking") or block.get("text") or block.get("thinking"):
                            return Usage.from_wire(usage)
    finally:
        if meter:
            meter.observe(usage)
    raise RuntimeError("Anthropic refresh ended before it confirmed usage.")


async def recap(client, route, messages, effort, *, replay=None, meter=None, metadata=None):
    repair = ThinkingRepair(replay)
    while True:
        body, _ = request_body(route, route.model, messages, [], effort, stream=False, max_tokens=512, repair=repair,
                               metadata=metadata)
        if meter:
            meter.admit()
        response = await client.post(endpoint(route.endpoint()), json=body, headers=replay_headers(body, route))
        try:
            require_success(response)
        except APIStatusError as error:
            if not repair.retry(route, body, error):
                raise
        else:
            break
    message = response.json()
    if meter:
        meter.observe(message.get("usage"))
    if message.get("stop_reason") in ("end_turn", "stop_sequence"):
        return "".join(block["text"] for block in message["content"] if block["type"] == "text")
    return None
