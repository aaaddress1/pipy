"""Bedrock Converse turns and signed Anthropic compaction over InvokeModel."""

import base64
import binascii
import json
import os
import re
from contextlib import contextmanager
from dataclasses import replace
from functools import lru_cache
from urllib.parse import quote, urlsplit, urlunsplit

import httpx
from openai import APIStatusError

import anthropic_messages as anthropic
from cancellable_http import turn_scope
from recovery import StreamInterrupted
from runtime import CompactionSummary, native_provider
from snapcompact import anthropic_content
from stream_watchdog import Watchdog, seconds


API = "bedrock-converse-stream"
AWS_CHAIN = "bedrock:aws-credential-chain"
TOKEN_ENVS = ("AWS_BEARER_TOKEN_BEDROCK", "AWS_BEDROCK_API_KEY")
REGIONAL_HOST = re.compile(r"bedrock-runtime\.[a-z0-9-]+\.amazonaws\.com\Z")
GEO_REGIONS = {"us": "us-east-1", "us-gov": "us-gov-west-1", "eu": "eu-west-1", "apac": "ap-southeast-1",
               "au": "ap-southeast-2", "jp": "ap-northeast-1"}
ERROR_STATUS = {"accessdeniedexception": 403, "internalserverexception": 500, "modelerrorexception": 424,
                "modelstreamerrorexception": 424, "modelnotreadyexception": 429, "modeltimeoutexception": 408,
                "resourcenotfoundexception": 404, "serviceunavailableexception": 503, "throttlingexception": 429}
STOP_REASONS = {"guardrail_intervened": "sensitive", "content_filtered": "sensitive"}
TOOL_ID = re.compile(r"[^a-zA-Z0-9_-]")
METADATA_VALUE = re.compile(r"[a-zA-Z0-9\s:_@$#=/+,\-.]{1,256}\Z", re.ASCII)
THINKING_BUDGETS = {"minimal": 1024, "low": 2048, "medium": 8192, "high": 16384, "xhigh": 32768, "max": 32768}


@lru_cache(maxsize=4)
def aws_session(profile):
    from botocore.session import Session

    return Session(profile=profile)


def profile():
    return os.getenv("AWS_PROFILE") or os.getenv("AWS_DEFAULT_PROFILE")


def ambient_credentials():
    from botocore.exceptions import BotoCoreError, ClientError

    try:
        return aws_session(profile()).get_credentials()
    except (BotoCoreError, ClientError):
        return None


def region_for(model):
    if model.startswith("arn:"):
        parts = model.split(":", 5)
        if len(parts) == 6 and parts[2] == "bedrock" and parts[3]:
            return parts[3]
    ambient = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
    if not ambient:
        from botocore.exceptions import ProfileNotFound

        try:
            ambient = aws_session(profile()).get_config_variable("region")
        except ProfileNotFound:
            pass
    geo = model.partition(".")[0]
    if geo in GEO_REGIONS:
        matches = ambient and (
            ambient == GEO_REGIONS[geo] if geo in ("jp", "au") else
            ambient.startswith("ap-" if geo == "apac" else geo + "-") and
            (geo != "us" or not ambient.startswith("us-gov-"))
        )
        return ambient if matches else GEO_REGIONS[geo]
    return ambient or "us-east-1"


def base_url(model, configured=None):
    region = region_for(model)
    standard = f"bedrock-runtime.{region}.amazonaws.com"
    url = urlsplit(configured or f"https://{standard}")
    return urlunsplit(url._replace(netloc=standard) if REGIONAL_HOST.fullmatch(url.netloc) else url).rstrip("/")


def supports_compaction(model, configured, *, override=None):
    if (model.get("compat") or {}).get("supportsServerCompaction") is not True:
        return False
    if override is not None:
        return override is True
    if not configured:
        return True
    host = urlsplit(configured).hostname or ""
    return bool(re.search(r"(?:^|\.)bedrock-runtime(?:-fips)?\.[\w.-]+\.amazonaws\.com(?:\.cn)?$", host)
                or host.startswith("vpce-") and ".bedrock-runtime." in host and host.endswith(".vpce.amazonaws.com"))


class BedrockAuth(httpx.Auth):
    requires_request_body = True

    def __init__(self, region, token=None, credentials=None):
        self.region, self.token, self.credentials = region, token, credentials

    def auth_flow(self, request):
        if self.token:
            request.headers["authorization"] = "Bearer " + self.token
        else:
            from botocore.auth import SigV4Auth
            from botocore.awsrequest import AWSRequest

            if self.credentials is None:
                raise RuntimeError("AWS credential chain did not provide Bedrock credentials.")
            wire = AWSRequest(method=request.method, url=str(request.url), data=request.content,
                              headers=dict(request.headers))
            SigV4Auth(self.credentials.get_frozen_credentials(), "bedrock", self.region).add_auth(wire)
            request.headers.update(dict(wire.headers.items()))
        yield request


def events(response):
    from botocore.eventstream import EventStreamBuffer, ParserError

    buffer = EventStreamBuffer()
    try:
        for chunk in response.iter_bytes():
            buffer.add_data(chunk)
            for frame in buffer:
                headers = frame.headers
                kind = headers.get(":message-type")
                if kind in ("error", "exception"):
                    code = headers.get(":exception-type") or headers.get(":error-code") or "BedrockStreamError"
                    try:
                        detail = json.loads(frame.payload).get("message")
                    except (ValueError, AttributeError):
                        detail = frame.payload.decode("utf-8", errors="replace")
                    detail = headers.get(":error-message") or detail or code
                    status = ERROR_STATUS.get(code.lower(), 400)
                    error = httpx.Response(status, request=response.request, headers=response.headers)
                    raise APIStatusError(f"{code}: {detail}", response=error, body={"code": code, "message": detail})
                if kind == "event":
                    yield headers.get(":event-type"), json.loads(frame.payload) if frame.payload else {}
    except (ValueError, ParserError) as error:
        raise StreamInterrupted(f"Invalid Bedrock EventStream frame ({type(error).__name__}).") from error


@contextmanager
def stream(client, route, operation, body):
    url = f"{route.endpoint()}/model/{quote(route.model, safe='')}/{operation}"
    headers = {"accept": "application/vnd.amazon.eventstream"}
    if operation == "invoke-with-response-stream":
        headers["x-amzn-bedrock-accept"] = "application/json"
    compat = (route.catalog_model or {}).get("compat") or {}
    idle = seconds(os.getenv("PI_STREAM_IDLE_TIMEOUT_MS"), seconds(compat.get("streamIdleTimeoutMs"), 300))
    default = idle if operation == "invoke-with-response-stream" else 300
    first = seconds(os.getenv("PI_STREAM_FIRST_EVENT_TIMEOUT_MS"), seconds(compat.get("streamFirstEventTimeoutMs"), default))
    watchdog = Watchdog("Bedrock", (first, idle))
    watchdog.arm("headers", first)
    timeout = httpx.Timeout(client.timeout.connect, read=max(first, idle) or None)
    with turn_scope(watchdog), client.stream(
        "POST", url, json=body, headers=headers, timeout=timeout,
    ) as response:
        yield response, watchdog


def require_success(response):
    try:
        anthropic.require_success(response)
    except httpx.TransportError as error:
        raise anthropic.status_error(response, {}) from error


def bedrock_content(value):
    blocks = []
    for part in anthropic_content(value):
        if part["type"] == "text" and part["text"].strip():
            blocks.append({"text": part["text"]})
        elif part["type"] == "image" and part["source"]["type"] == "base64":
            source = part["source"]
            blocks.append({"image": {"format": source["media_type"].split("/", 1)[-1],
                                     "source": {"bytes": source["data"]}}})
    return blocks


def tool_id(value):
    return TOOL_ID.sub("_", value)[:64]


def assistant_blocks(message, route):
    saved = message.get("_anthropic") or {}
    if saved.get("route") == route.identity() and saved.get("text") == message.get("content"):
        blocks = saved["blocks"]
    else:
        blocks = anthropic.imported_assistant_content(message, native_provider(route), keep_thinking=True)
        if blocks is None:
            blocks = anthropic.assistant_content(message, None, None)
    for block in blocks:
        kind = block["type"]
        if kind == "text" and block["text"].strip():
            yield {"text": block["text"]}
        elif kind == "tool_use":
            yield {"toolUse": {"toolUseId": tool_id(block["id"]), "name": block["name"], "input": block["input"]}}
        elif kind == "thinking":
            if block.get("signature"):
                yield {"reasoningContent": {"reasoningText": {"text": block["thinking"], "signature": block["signature"]}}}
            elif block.get("thinking", "").strip():
                text = block["thinking"]
                yield {"text": text if "claude" in route.model.lower() else f"<think>\n{text}\n</think>"}
        elif kind == "redacted_thinking":
            yield {"reasoningContent": {"redactedContent": block["data"]}}


def thinking_fields(route, effort):
    model = route.catalog_model or {}
    thinking, compat = model.get("thinking") or {}, model.get("compat") or {}
    if not effort or effort in ("off", "none"):
        return {"thinking": {"type": "disabled"}, "output_config": {"effort": "low"}} if (
            compat.get("disabledThinking") == "disabled"
        ) else {}
    if thinking.get("mode") == "effort":
        level = thinking.get("defaultLevel", "medium") if effort == "auto" else effort
        return {"reasoning": {"effort": (thinking.get("effortMap") or {}).get(level, level)}}
    if model and not model.get("reasoning"):
        return {}
    if model and thinking.get("mode") not in ("anthropic-adaptive", "adaptive"):
        level = thinking.get("defaultLevel", "medium") if effort == "auto" else effort
        budget = (thinking.get("effortBudgets") or {}).get(level) or THINKING_BUDGETS.get(level, 8192)
        return {"thinking": {"type": "enabled", "budget_tokens": budget, "display": "summarized"}}
    body = {"max_tokens": route.max_output_tokens}
    anthropic.apply_thinking(body, model, effort)
    body.pop("max_tokens")
    return body


def request_body(route, messages, schemas, effort, tool_choice=None):
    catalog = route.catalog_model or {}
    wire, system, tool_content, tool_images = [], [], [], []
    for message in [*messages, {"role": "end"}]:
        role = message["role"]
        if role == "tool":
            content = bedrock_content(message["content"])
            error = message.get("_tool_error")
            if error or catalog.get("requiresToolResultImageHoisting"):
                tool_images.extend(part for part in content if "image" in part)
                content = [{"text": "(see attached image)"} if "image" in part else part for part in content]
            tool_content.append({"toolResult": {"toolUseId": tool_id(message["tool_call_id"]),
                                                 "status": "error" if error else "success", "content": content}})
            continue
        if tool_content:
            wire.append({"role": "user", "content": tool_content + tool_images})
            tool_content, tool_images = [], []
        if role == "system" and message.get("content"):
            system.append({"text": message["content"]})
        elif role in ("user", "assistant", "developer"):
            content = list(assistant_blocks(message, route)) if role == "assistant" else bedrock_content(message["content"])
            if content:
                wire.append({"role": "assistant" if role == "assistant" else "user", "content": content})
    compat = catalog.get("compat") or {}
    if route.cache_retention != "none" and compat.get("promptCacheMode") != "automatic":
        explicit = compat.get("promptCacheMode") == "explicit"
        limit = (compat.get("promptCacheMaximumCheckpoints") or 0) if explicit else 2 if os.getenv("AWS_BEDROCK_FORCE_CACHE") else 0
        cache = {"cachePoint": {"type": "default"}}
        if route.cache_retention == "1h" and compat.get("supportsLongPromptCacheRetention"):
            cache["cachePoint"]["ttl"] = "1h"
        if limit and wire and wire[-1]["role"] == "user":
            wire[-1]["content"].append(cache)
            limit -= 1
        if limit and system:
            system.append(cache)
    body = {"messages": wire, "inferenceConfig": {"maxTokens": route.max_output_tokens}}
    if system:
        body["system"] = system
    fields = thinking_fields(route, effort)
    tools = []
    for schema in schemas:
        function = schema["function"]
        spec = {"name": function["name"], "inputSchema": {"json": function["parameters"]}}
        if function.get("description"):
            spec["description"] = function["description"]
        tools.append({"toolSpec": spec})
    if tools:
        body["toolConfig"] = {"tools": tools}
        choice = ({"tool": {"name": tool_choice["function"]["name"]}} if isinstance(tool_choice, dict) else
                  {"any": {}} if tool_choice == "required" else {"auto": {}} if tool_choice == "auto" else None)
        if choice:
            body["toolConfig"]["toolChoice"] = choice
    elif tool_choice == "none" and any("toolUse" in part or "toolResult" in part for item in wire for part in item["content"]):
        body["toolConfig"] = {"tools": [{"toolSpec": {"name": "__no_tools__", "description": "Answer with text.",
                         "inputSchema": {"json": {"type": "object", "properties": {}}}}}], "toolChoice": {"auto": {}}}
    choice = body.get("toolConfig", {}).get("toolChoice", {})
    thinking = catalog.get("thinking") or {}
    if "any" in choice or "tool" in choice:
        if catalog and (not compat.get("supportsForcedToolChoice") or thinking.get("prefixBinding")):
            body["toolConfig"]["toolChoice"] = {"auto": {}}
        elif fields.get("thinking", {}).get("type") != "disabled":
            fields = {}
    if thinking.get("prefixBinding") and compat.get("supportsThinkingBindingControls") and (
        fields.get("thinking", {}).get("type") != "disabled"
    ):
        fields.setdefault("thinking", {"type": "adaptive"})["block_binding"] = {
            "prefix_mismatch_behavior": route.thinking_prefix_mismatch or "drop_block",
        }
        fields.setdefault("anthropic_beta", []).append(anthropic.BINDING_BETA)
    if fields:
        body["additionalModelRequestFields"] = fields
    return body


def converse_usage(raw):
    return {"input_tokens": raw.get("inputTokens") or 0, "output_tokens": raw.get("outputTokens") or 0,
            "cache_read_input_tokens": raw.get("cacheReadInputTokens") or 0,
            "cache_creation_input_tokens": raw.get("cacheWriteInputTokens") or 0}


def converse_events(response):
    kinds = {}
    stopped = False
    for kind, body in events(response):
        if kind == "messageStart":
            yield {"type": "message_start", "message": {}}
        elif kind == "messageStop":
            stop = body.get("stopReason")
            yield {"type": "message_delta", "delta": {"stop_reason": STOP_REASONS.get(stop, stop)}}
            stopped = True
        elif kind == "metadata":
            yield {"type": "message_delta", "delta": {}, "usage": converse_usage(body.get("usage") or {})}
            if stopped:
                break
        elif kind in ("contentBlockStart", "contentBlockDelta", "contentBlockStop"):
            index = body["contentBlockIndex"]
            if kind == "contentBlockStop":
                if index in kinds:
                    yield {"type": "content_block_stop", "index": index}
                continue
            delta = body.get("delta") or {}
            start = body.get("start") or {}
            if index not in kinds:
                reasoning = delta.get("reasoningContent") or {}
                if tool := start.get("toolUse"):
                    block = {"type": "tool_use", "name": tool["name"], "id": tool_id(tool["toolUseId"]), "input": {}}
                elif "redactedContent" in reasoning:
                    block = {"type": "redacted_thinking", "data": reasoning["redactedContent"]}
                elif reasoning:
                    block = {"type": "thinking", "thinking": ""}
                elif delta:
                    block = {"type": "text", "text": ""}
                else:
                    continue
                kinds[index] = block["type"]
                yield {"type": "content_block_start", "index": index, "content_block": block}
            for name, field, value in converse_deltas(delta):
                yield {"type": "content_block_delta", "index": index, "delta": {"type": name, field: value}}
    if stopped:
        yield {"type": "message_stop"}


def converse_deltas(delta):
    if "text" in delta:
        yield "text_delta", "text", delta["text"]
    if "toolUse" in delta:
        yield "input_json_delta", "partial_json", delta["toolUse"]["input"]
    reasoning = delta.get("reasoningContent") or {}
    for key, field in (("text", "thinking"), ("signature", "signature")):
        if key in reasoning:
            yield field + "_delta", field, reasoning[key]


def stream_attempt(client, route, model, messages, schemas, effort, guard, show, *, tool_choice=None, cancel=None,
                   turn=None, usage_observer=None, tracker=None, on_text=None, speculation=None):
    if model != route.model:
        route = replace(route, model=model)
    body = request_body(route, messages, schemas, effort, tool_choice)
    message = anthropic.MessageStream(guard, show, turn, tracker=tracker, on_text=on_text, speculation=speculation)
    if turn:
        turn.interrupt_snapshot = lambda: message.interruption_snapshot(route, "bedrock")

    def interrupted(label, *, transport=False):
        error = message.interrupted(label, transport=transport)
        if transport and tool_choice is None:
            error.unexecuted_turn = message.replayable_tool_turn(route, model, "bedrock", allow_text=True)
        return error

    try:
        for attempt in range(2):
            try:
                with stream(client, route, "converse-stream", body) as (response, watchdog):
                    require_success(response)
                    watchdog.arm("first", watchdog.first)
                    for event in converse_events(response):
                        watchdog.observe(True)
                        if cancel:
                            cancel()
                        message.push(event, response)
                break
            except APIStatusError as error:
                if attempt or message.started or route.thinking_prefix_mismatch == "error" or not anthropic.PREFIX_MISMATCH.search(str(error)):
                    raise
                body["messages"] = [{**item, "content": content} for item in body["messages"]
                                    if (content := [part for part in item["content"] if "reasoningContent" not in part])]
        if not message.stopped:
            raise interrupted("Bedrock ConverseStream ended before messageStop.", transport=True)
        if message.reason == "sensitive":
            raise message.refusal(route, model, body, "bedrock", None)
        return message.result(route, model, "bedrock", text_only=tool_choice in ("none", "auto"))
    except APIStatusError as error:
        text = "".join(message.text)
        if (show or turn) and text.strip() or any(block["type"] == "tool_use" for block in message.blocks.values()):
            error.replay_unsafe = True
            error.text = text
        raise
    except (httpx.TransportError, KeyError, TypeError, StreamInterrupted) as error:
        if isinstance(error, StreamInterrupted) and error.transport:
            raise
        raise interrupted(f"Bedrock ConverseStream interrupted ({type(error).__name__}).",
                          transport=isinstance(error, httpx.TransportError)) from error
    finally:
        if usage_observer and message.usage:
            usage_observer(message.usage)
        if show and message.text:
            print()


def invoke_events(response):
    for kind, frame in events(response):
        if kind == "chunk":
            try:
                event = json.loads(base64.b64decode(frame["bytes"], validate=True))
            except (ValueError, binascii.Error, KeyError, TypeError) as error:
                raise StreamInterrupted("Bedrock InvokeModel returned a malformed Anthropic event.") from error
            if not isinstance(event, dict) or not isinstance(event.get("type"), str):
                raise StreamInterrupted("Bedrock InvokeModel returned a malformed Anthropic event envelope.")
            yield event
            if event["type"] == "message_stop":
                return
    raise StreamInterrupted("Bedrock InvokeModel stream ended before message_stop.")


def compact(client, route, messages, schemas, effort, instructions, *, replay=None, meter=None, metadata=None,
            files_due_before=float("inf")):
    body, _ = anthropic.request_body(route, route.model, messages, schemas, effort, compacting=True,
                                    files_due_before=files_due_before, native_history=True)
    body.pop("model")
    body.pop("stream")
    body.update(anthropic_version="bedrock-2023-05-31", compaction={"type": "summarize", "instructions": instructions})
    user = (metadata or {}).get("user_id")
    if isinstance(user, str) and not METADATA_VALUE.fullmatch(user):
        try:
            user = json.loads(user).get("session_id")
        except (ValueError, AttributeError):
            user = None
    if isinstance(user, str) and METADATA_VALUE.fullmatch(user):
        body["metadata"] = {"user_id": user}
    for field in ("thinking", "output_config"):
        body.pop(field, None)
    body.update(thinking_fields(route, effort))
    body["anthropic_beta"] = list(dict.fromkeys([
        *body.get("anthropic_beta", ()), *(anthropic.replay_headers(body, route) or {}).get("anthropic-beta", "").split(","),
    ]))
    if meter:
        meter.admit()
    blocks, open_blocks, usage, reason = {}, set(), {}, None
    started = False
    try:
        with stream(client, route, "invoke-with-response-stream", body) as (response, watchdog):
            require_success(response)
            watchdog.arm("first", watchdog.first)
            for event in invoke_events(response):
                kind = event["type"]
                watchdog.observe(kind != "ping")
                if kind == "error":
                    error = event.get("error") or {}
                    status = {"overloaded_error": 529, "rate_limit_error": 429, "api_error": 500}.get(error.get("type"), 400)
                    raise anthropic.status_error(httpx.Response(status, request=response.request), event)
                if kind == "message_start":
                    if started:
                        raise StreamInterrupted("Bedrock InvokeModel returned a duplicate message_start.")
                    started = True
                    usage.update(event["message"].get("usage") or {})
                elif kind == "content_block_start":
                    index = event["index"]
                    if not started or index in blocks:
                        raise StreamInterrupted("Bedrock InvokeModel returned an unexpected content block.")
                    blocks[index] = dict(event["content_block"])
                    open_blocks.add(index)
                elif kind == "content_block_delta":
                    delta = event["delta"]
                    index = event["index"]
                    if index not in open_blocks:
                        raise StreamInterrupted("Bedrock InvokeModel returned a delta for a closed content block.")
                    block = blocks[index]
                    for key in ("content", "signature"):
                        if isinstance(value := delta.get(key), str):
                            block[key] = (block.get(key) or "") + value
                elif kind == "content_block_stop":
                    open_blocks.discard(event["index"])
                elif kind == "message_delta":
                    usage.update(event.get("usage") or {})
                    reason = event["delta"].get("stop_reason") or reason
    finally:
        if meter and usage:
            meter.observe(usage)
    if not started or open_blocks or reason != "compaction" or len(blocks) != 1:
        raise RuntimeError("Bedrock did not return one Anthropic compaction block.")
    block = next(iter(blocks.values()))
    if block.get("type") != "compaction" or not all(isinstance(block.get(key), str) and block[key] for key in ("content", "signature")):
        raise RuntimeError("Bedrock compaction block is unsigned or empty.")
    if "tool_changes" in block and (not isinstance(block["tool_changes"], list) or
                                     not all(isinstance(change, dict) for change in block["tool_changes"])):
        raise RuntimeError("Bedrock compaction block has malformed signed tool changes.")
    prior = next((message["_anthropic_compaction"] for message in messages if "_anthropic_compaction" in message), {})
    files, _ = anthropic.compaction_files(prior) if prior.get("anthropic_provider") == native_provider(route) else ([], None)
    retained = [entry for entry in files if entry["after"] >= files_due_before]
    stats = anthropic.Usage.from_wire(usage)
    return CompactionSummary(block["content"], {"route": route.identity(), "block": block,
                                               "anthropic_provider": native_provider(route), "anthropic_metadata": {
                                                   "exactTail": True, "retainedFiles": retained, "model": route.model,
                                                   "usedTokens": stats.prompt_tokens,
                                               }})
