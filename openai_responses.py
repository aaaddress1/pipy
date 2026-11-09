"""Responses transport for the shared Mini Pi transcript."""

import json
import sys
import time
from concurrent.futures import CancelledError
from contextlib import closing, contextmanager
from dataclasses import dataclass
from hashlib import sha256
from itertools import chain
from urllib.parse import urlsplit

import httpx
from openai import APIConnectionError, APIError, APIStatusError

import model_catalog
import codex_responses
from responses_effort import supported as supports_effort_updates
from cancellable_http import active_turn
from cost_gate import CostCapError
from handoff import deadline
from recovery import ContextWindowExceeded, EmptyTurn, IncompleteTurn, StreamInterrupted, TRANSPORT_ERRORS, context_overflow
from runtime import CompactionSummary, responses_replay_matches, responses_replay_scope, tokens
from snapcompact import responses_content
from snapcompact_inline import stb_messages


CONTEXT_PREFIXES = (
    "<environment_context>", "<user_instructions>", "<additional_context>", "<skills",
    "<token_budget>", "<model_switch>", "[agent recovery]", "summary of earlier conversation.", "resume prior conversation.",
    "context replaced.", "branch-return summary:",
)
GATEWAY_FIELDS = frozenset(("provider", "providerOptions", "caching", "cache_anchor_items", "cache_ttl"))
V2_TIMEOUT = 300
STABLE_REPLAY_PREFIX = "stable:"


def sdk_body(body):
    extra = {key: value for key, value in body.items() if key in GATEWAY_FIELDS}
    if extra:
        return {**{key: value for key, value in body.items() if key not in GATEWAY_FIELDS}, "extra_body": extra}
    return body


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int

    @classmethod
    def from_response(cls, response):
        usage = response.get("usage") or {}
        return cls(usage.get("input_tokens") or 0, usage.get("output_tokens") or 0)


def append_items(items, digests, added):
    for item in added:
        encoded = json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        for digest in digests:
            digest.update(encoded)
        items.append(item)


def replay_digest(instructions, tools, effort):
    return sha256(json.dumps([instructions, tools, effort], ensure_ascii=False, sort_keys=True).encode("utf-8"))


def request_body(route, model, messages, schemas, effort, *, max_tokens=None, user_items=None):
    instructions = "\n\n".join(message["content"] for message in messages if message["role"] == "system")
    tools = [{"type": "function", "strict": False, **schema["function"]} for schema in schemas]
    identity = route.identity() if route.model == model else None
    stable = bool(identity and supports_effort_updates(route))
    prefix_tag = STABLE_REPLAY_PREFIX if stable else ""
    digest = replay_digest(instructions, tools, STABLE_REPLAY_PREFIX if stable else effort)
    # Older sessions included the effort in the replay fingerprint. Preserve
    # their native items only when the original instructions, tools and history still match.
    legacy = ()
    if stable and any(
        (saved := message.get("_responses")) and saved.get("route") == identity
        and isinstance(prefix := saved.get("prefix"), str) and not prefix.startswith(STABLE_REPLAY_PREFIX)
        for message in messages
    ):
        levels = dict.fromkeys((effort, None, "auto", "off", "none", *model_catalog.EFFORTS))
        legacy = tuple(replay_digest(instructions, tools, level) for level in levels)
    digests = (digest, *legacy)
    items = []
    for message in messages:
        role = message["role"]
        compacted = message.get("_responses_compaction")
        if role == "system":
            continue
        authored = role == "user" and not message.get("_compaction_context")
        if identity and compacted and not items and responses_replay_matches(compacted, route):
            added = stb_messages(compacted["items"], route, fields=("content", "output"))
            authored = True
        elif role == "assistant":
            content = message.get("content")
            calls = message.get("tool_calls", [])
            saved = message.get("_responses")
            if (
                identity and saved and saved["route"] == identity and saved["text"] == content and saved["calls"] == calls
                and (saved["prefix"] == prefix_tag + digest.hexdigest()
                     or any(saved["prefix"] == candidate.hexdigest() for candidate in legacy))
            ):
                added = saved["items"]
            else:
                added = [{"role": role, "content": content}] if content else []
                added += [
                    {"type": "function_call", "call_id": call["id"], **call["function"]} for call in calls
                ]
        elif role == "tool":
            added = [{
                "type": "function_call_output", "call_id": message["tool_call_id"],
                "output": responses_content(message.get("content") or ""),
            }]
        else:
            added = [{"role": role, "content": responses_content(message.get("content") or "")}]
        if user_items is not None and authored:
            user_items.extend(item for item in added if item.get("role") == "user")
        append_items(items, digests, added)
    body = {
        "model": route.wire_model(effort) if identity else model, "input": items,
        "store": False, "include": ["reasoning.encrypted_content"],
    }
    body.update(model_catalog.gateway_body(route.catalog_model, responses=True, cache_retention=route.cache_retention))
    if not (route.catalog_model or {}).get("omitMaxOutputTokens"):
        body["max_output_tokens"] = max_tokens or route.max_output_tokens
    if instructions:
        body["instructions"] = instructions
    if tools:
        body["tools"] = tools
    if effort in ("off", "none"):
        body["reasoning"] = {"effort": "none"}
    elif effort:
        body["reasoning"] = {"summary": "auto"}
        if effort != "auto":
            body["reasoning"]["effort"] = effort
    return body, prefix_tag + digest.hexdigest()


def compaction_summary(route, items):
    if not any(
        item.get("type") == "compaction_summary" or item.get("type") == "compaction" and item.get("encrypted_content")
        for item in items
    ):
        raise RuntimeError("Responses did not return a compaction item.")
    replacement_tokens = tokens([{key: value for key, value in item.items() if key != "encrypted_content"} for item in items])
    return CompactionSummary(
        "[Earlier conversation compacted by the Responses provider.]",
        {"route": route.identity(), "scope": responses_replay_scope(route), "items": items,
         "replacement_tokens": replacement_tokens},
    )


def text_size(text):
    return (len(text.encode("utf-16-le")) // 2 + 3) // 4


def truncate_text(text, budget):
    limit = budget * 4
    marker = f"…{text_size(text) - budget} tokens truncated…"
    wire = text.encode("utf-16-le")
    if limit <= len(marker) + 2:
        return wire[:limit * 2].decode("utf-16-le", errors="ignore")
    side = ((limit - len(marker)) // 2) * 2
    return wire[:side].decode("utf-16-le", errors="ignore") + marker + wire[-side:].decode("utf-16-le", errors="ignore")


def retained_users(items, budget=64_000):
    retained = []
    for item in reversed(items):
        if not budget:
            break
        if item.get("type", "message") != "message" or item.get("role") != "user":
            continue
        content = item["content"]
        parts = [{"type": "input_text", "text": content}] if isinstance(content, str) else content
        texts = [part.get("text", "") for part in parts if part["type"] in ("input_text", "output_text")]
        if any(text.lstrip().lower().startswith(CONTEXT_PREFIXES) for text in texts):
            continue
        cost = sum(text_size(text) for text in texts) + 765 * sum(part["type"] == "input_image" for part in parts)
        if max(1, cost) <= budget:
            retained.append(item)
            budget -= max(1, cost)
            continue
        clipped = []
        for part in parts:
            kind = part["type"]
            if kind == "input_image" and budget >= 765:
                clipped.append(part)
                budget -= 765
            elif kind in ("input_text", "output_text") and budget:
                text = part.get("text", "")
                size = text_size(text)
                clipped.append(part if size <= budget else {**part, "text": truncate_text(text, budget)})
                budget -= min(budget, size)
        if clipped:
            retained.append({**item, "content": clipped})
            break
    return list(reversed(retained))


def compaction_sse(lines):
    name, data = None, []
    for line in chain(lines, ("",)):
        if line.startswith("event:"):
            name = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
        elif not line:
            if data and data != ["[DONE]"]:
                try:
                    event = json.loads("\n".join(data))
                except ValueError as error:
                    raise StreamInterrupted("Responses V2 stream parse failed.") from error
                if not isinstance(event, dict):
                    raise StreamInterrupted("Responses V2 stream event must be an object.")
                yield event.get("type") or name, event
            name, data = None, []


@contextmanager
def raw_response(client, body, endpoint, headers):
    response = client.post(endpoint, body={**body, "stream": True}, options={"headers": headers},
                           cast_to=httpx.Response, stream=True)
    with closing(response):
        yield response


@contextmanager
def v2_events(client, body, endpoint, headers, identity, websocket, lite):
    if websocket:
        saved = identity.state, websocket.models_etag
        try:
            with websocket.exchange(body, headers, identity, lite=lite, compaction=True) as stream:
                buffered = list(stream) if stream is not None else None
        except httpx.RemoteProtocolError:
            identity.state, websocket.models_etag = saved
            websocket.sse_only = True
        else:
            if buffered is not None:
                for event in buffered:
                    if event["type"] == "response.metadata":
                        websocket.observe(identity, event.get("headers") or {})
                response = httpx.Response(200, request=httpx.Request("POST", endpoint))
                yield ((event["type"], event) for event in buffered), response
                return
    with raw_response(client, body, endpoint, headers) as stream:
        if identity:
            identity.observe(stream.headers)
        yield compaction_sse(stream.iter_lines()), stream


def collect_v2(client, body, endpoint, headers, meter=None, identity=None, *, websocket=None, lite=False):
    compactions = []
    completed = False
    if meter:
        meter.admit()
    with v2_events(client, body, endpoint, headers, identity, websocket, lite) as (events, stream):
        for kind, event in events:
            if kind == "response.output_item.done":
                item = event.get("item") or {}
                if item.get("type") == "compaction":
                    compactions.append(item)
            elif kind == "response.metadata" and identity:
                identity.observe(event.get("headers") or {})
            elif kind in ("response.completed", "response.done"):
                if meter and not completed:
                    meter.observe((event.get("response") or {}).get("usage"))
                completed = True
            elif kind in ("response.failed", "response.incomplete", "error"):
                response = event.get("response") or {}
                if meter:
                    meter.observe(response.get("usage"))
                error = response.get("error") or event.get("error") or event
                code = error.get("code")
                status = event.get("status") or event.get("status_code") or error.get("status")
                if code in ("context_length_exceeded", "context_window_exceeded", "context_too_large"):
                    status = status if status in (400, 422) else 400
                elif code in ("rate_limit_exceeded", "rate_limit_error", "usage_limit_reached", "insufficient_quota",
                              "quota_exceeded") and status is None:
                    status = 429
                if type(status) is int and 400 <= status <= 599:
                    raise APIStatusError(str(error.get("message") or code or "Responses V2 failed."),
                                         response=httpx.Response(status, request=stream.request, headers=stream.headers),
                                         body=error)
                if code in ("server_error", "internal_error", "overloaded"):
                    raise StreamInterrupted("Responses V2 reported a transient server error.")
                raise RuntimeError("Responses V2 did not complete.")
    if not completed:
        raise StreamInterrupted("Responses V2 stream ended before completion.")
    if len(compactions) != 1:
        raise RuntimeError("Responses V2 requires exactly one compaction output item.")
    return compactions[0]


def compaction_headers(route, model, session_id, prompt_cache_key):
    headers = {"Accept": "text/event-stream"}
    if not session_id:
        return headers
    catalog = route.catalog_model or {}
    backend = catalog.get("providerType") or catalog.get("provider") or route.provider()
    if backend == "openai" or urlsplit(route.endpoint()).hostname == "api.openai.com":
        headers.update({"session_id": session_id, "x-client-request-id": session_id})
    elif backend in ("opencode-go", "opencode-zen"):
        policy = model_catalog.model_policy({**catalog, "provider": catalog.get("provider", backend),
                                             "id": model, "api": route.api})
        shared = backend == "opencode-go" and policy.get("sharedPromptCache") is True
        headers["x-opencode-session"] = (prompt_cache_key if shared else None) or session_id
    if name := (catalog.get("compat") or {}).get("sessionHeader"):
        if name.lower() not in {key.lower() for key in headers}:
            headers[name] = session_id
    return headers


def compact_v2(client, route, body, user_items, *, retry, meter, session_id, prompt_cache_key, retained_budget,
               identity=None, operation=None, websocket=None):
    remote = remote_settings(route)
    endpoint = remote.get("v2Endpoint") or remote.get("streamingEndpoint") or "/responses"
    request = {**body, "input": [*body["input"], {"type": "compaction_trigger"}]}
    request["model"] = remote.get("model") or (route.catalog_model or {}).get("requestModelId") or route.model
    headers = compaction_headers(route, request["model"], session_id, prompt_cache_key)
    request.pop("max_output_tokens", None)
    if not request.get("reasoning"):
        request.pop("include", None)
    if key := prompt_cache_key or session_id:
        request["prompt_cache_key"] = key
    if request.get("tools"):
        request["tool_choice"] = "auto"
    lite = False
    if identity:
        endpoint = remote.get("v2Endpoint") or remote.get("streamingEndpoint") or codex_responses.endpoint(route.endpoint())
        lite = codex_responses.lite_enabled(route, compact=True)
        codex_responses.request_shape(route, request, lite=lite)
        headers = identity.decorate(request, lite=lite,
                                    compaction={**operation, "implementation": "responses_compaction_v2"})
        headers["x-codex-beta-features"] = "remote_compaction_v2"
    attempts = 3 if retry else 1
    turn = active_turn()
    for attempt in range(attempts):
        try:
            with deadline(V2_TIMEOUT, turn) as limit:
                try:
                    item = collect_v2(client, request, endpoint, headers, meter, identity, websocket=websocket, lite=lite)
                    limit.check()
                except (CancelledError, APIConnectionError) as error:
                    if turn:
                        turn.check()
                    if not limit.event.is_set():
                        raise
                    raise StreamInterrupted("Responses V2 compaction timed out.") from error
            return compaction_summary(route, [*retained_users(user_items, retained_budget), item])
        except APIStatusError as error:
            if attempt + 1 == attempts or (error.status_code not in (408, 409, 425, 429) and error.status_code < 500):
                raise
        except (StreamInterrupted, *TRANSPORT_ERRORS):
            if attempt + 1 == attempts:
                raise
        (turn.wait if turn else time.sleep)(2 ** attempt)


def remote_settings(route):
    model = route.catalog_model or {}
    remote = model.get("remoteCompaction") or {}
    if (route.endpoint() == (model.get("baseUrl") or "").rstrip("/")
        and remote.get("api", route.api) == route.api):
        return remote
    return {}


def compact(client, route, messages, schemas, effort, instructions, *, retry=True, meter=None, prefix=None,
            session_id=None, prompt_cache_key=None, v2_enabled=True, retained_budget=64_000,
            codex_session=None, automatic=False, websocket=None):
    remote = remote_settings(route)
    client = client.with_options(timeout=300)
    identity = codex_session.turn() if codex_session else None
    operation = {"trigger": "auto" if automatic else "manual",
                 "reason": "context_limit" if automatic else "user_requested", "phase": "standalone_turn"}

    def prepare(messages):
        user_items = []
        body, _ = request_body(route, route.model, messages, schemas, effort, user_items=user_items)
        return body, user_items

    whole = prepare(messages)

    def whole_then_prefix(send):
        try:
            return send(*whole)
        except APIStatusError as error:
            if prefix is None or not context_overflow(error):
                raise
            covered, messages = prefix
            return send(*prepare(messages)).through(covered)

    if v2_enabled and route.responses_compaction_v2:
        try:
            return whole_then_prefix(lambda body, users: compact_v2(
                client, route, body, users, retry=retry, meter=meter, session_id=session_id,
                prompt_cache_key=prompt_cache_key, retained_budget=retained_budget, identity=identity, operation=operation,
                websocket=websocket,
            ))
        except CostCapError:
            raise
        except (APIError, RuntimeError, ValueError, KeyError, TypeError, *TRANSPORT_ERRORS) as error:
            if not retry and (isinstance(error, (StreamInterrupted, *TRANSPORT_ERRORS))
                              or isinstance(error, APIStatusError)
                              and (context_overflow(error) or error.status_code in (401, 403, 408, 409, 413, 425, 429)
                                   or error.status_code >= 500)):
                raise
            print(f"[compaction: Responses V2 failed ({type(error).__name__}); trying V1]", file=sys.stderr)

    return whole_then_prefix(lambda body, _: compact_v1(client, route, body, instructions, remote, meter,
                                                        identity, operation, prompt_cache_key or session_id))


def compact_v1(client, route, body, instructions, remote, meter, identity, operation, cache_key):
    payload = {"model": remote.get("model") or body["model"], "input": body["input"],
               "instructions": body.get("instructions") or instructions}
    payload.update((key, value) for key, value in body.items() if key in GATEWAY_FIELDS)
    if meter:
        meter.admit()
    if identity:
        lite = codex_responses.lite_enabled(route, compact=True)
        payload["input"] = codex_responses.history_items(payload["input"])
        if cache_key:
            payload["prompt_cache_key"] = cache_key
        if lite:
            codex_responses.lite_shape(payload)
            payload["include"] = ["reasoning.encrypted_content"]
        headers = identity.decorate(payload, lite=lite, compaction={**operation, "implementation": "responses_compact"})
        headers["Accept"] = "application/json"
        headers[codex_responses.INSTALLATION_ID] = codex_responses.installation_id()
        payload.pop("client_metadata", None)
        endpoint = remote.get("endpoint") or codex_responses.endpoint(route.endpoint()) + "/compact"
        reply = client.post(endpoint, body=payload, options={"headers": headers}, cast_to=object)
    elif endpoint := remote.get("endpoint"):
        reply = client.post(endpoint, body=payload, cast_to=object)
    else:
        reply = client.responses.with_raw_response.compact(**sdk_body(payload)).http_response.json()
    if meter:
        meter.observe(reply.get("usage"))
    items = [
        item for item in reply["output"]
        if item["type"] in ("compaction", "compaction_summary")
        or item["type"] == "message" and item["role"] in ("assistant", "user")
    ]
    return compaction_summary(route, items)


def item_text(item, separator=""):
    if item["type"] == "message" and item["role"] == "assistant":
        return separator.join(
            part.get("text") or part.get("refusal") or ""
            for part in item.get("content", []) if part["type"] in ("output_text", "refusal")
        )
    return ""


def completed_message(response, route, model, fingerprint):
    output = response["output"]
    text = "".join(item_text(item) for item in output)
    calls, replay = [], []
    for item in output:
        kind = item["type"]
        if kind == "function_call":
            if item.get("status") not in (None, "completed") or not item["call_id"] or not item["name"]:
                raise ValueError("Incomplete Responses function call.")
            if not isinstance(json.loads(item["arguments"]), dict):
                raise ValueError("Responses function arguments must be an object.")
            calls.append({
                "id": item["call_id"], "type": "function",
                "function": {"name": item["name"], "arguments": item["arguments"]},
            })
            replay.append(item)
        elif kind == "message" and item["role"] == "assistant":
            replay.append(item)
        elif kind == "reasoning" and item.get("encrypted_content"):
            replay.append(item)
    if len({call["id"] for call in calls}) != len(calls):
        raise ValueError("Responses returned duplicate function calls.")
    message = {"role": "assistant", "content": text or None}
    if calls:
        message["tool_calls"] = calls
    if replay and route.model == model:
        message["_responses"] = {
            "route": route.identity(), "prefix": fingerprint, "text": message["content"],
            "calls": calls, "items": replay,
        }
    return message


@contextmanager
def response_events(client, body, websocket, *, codex_route=None, identity=None, live_source=None, effort_control=None):
    if codex_route:
        lite = codex_responses.lite_enabled(codex_route)
        codex_responses.request_shape(codex_route, body, lite=lite)
        if effort_control:
            effort_control.apply(body)
        headers = identity.decorate(body, lite=lite)
        endpoint = codex_responses.endpoint(codex_route.endpoint())
        with closing(codex_live_events(client, body, endpoint, headers, identity, websocket, lite, live_source,
                                       codex_route.vision)) as stream:
            yield stream
        return
    if effort_control:
        effort_control.apply(body)
    if websocket is not None and websocket.open():
        with closing(websocket.events(body)) as stream:
            yield stream
    else:
        with client.responses.create(**sdk_body(body), stream=True) as stream:
            yield (event.to_dict() for event in stream)


def codex_live_events(client, body, endpoint, headers, identity, websocket, lite, live_source, vision):
    if websocket:
        saved = identity.state, websocket.models_etag
        for attempt in range(codex_responses.WEBSOCKET_RETRIES + 1):
            exposed = False
            try:
                with websocket.exchange(body, headers, identity, lite=lite, live_source=live_source, vision=vision) as stream:
                    if stream is None:
                        break
                    for event in stream:
                        kind = event["type"]
                        if kind == "response.metadata":
                            websocket.observe(identity, event.get("headers") or {})
                        exposed |= kind.startswith(("response.output_text.", "response.refusal.", "response.reasoning_",
                                                    "response.function_call_arguments.")) or (
                            kind in ("response.output_item.added", "response.output_item.done")
                            and (event.get("item") or {}).get("type") in ("function_call", "custom_tool_call")
                        )
                        yield event
                return
            except (httpx.RemoteProtocolError, APIStatusError) as error:
                code = getattr(error, "code", None)
                native_lane = code == "unsupported_native_inflight_message"
                if isinstance(error, APIStatusError) and not native_lane and code != "websocket_connection_limit_reached":
                    raise
                if native_lane:
                    websocket.steering_unsupported = True
                if exposed:
                    websocket.sse_only = True
                    raise httpx.RemoteProtocolError("Codex WebSocket interrupted streamed output.") from error
                if native_lane:
                    saved = None, None
                    headers = {key: value for key, value in headers.items() if key != codex_responses.TURN_STATE}
                identity.state, websocket.models_etag = saved
                if attempt == codex_responses.WEBSOCKET_RETRIES:
                    websocket.sse_only = True
                    break
                codex_responses.websocket_backoff(attempt + 1)
    with raw_response(client, body, endpoint, headers) as response:
        identity.observe(response.headers)
        yield from codex_events(compaction_sse(response.iter_lines()), identity, response)


def codex_events(frames, identity, transport):
    output = codex_responses.OutputItems()
    for kind, event in frames:
        if not kind:
            continue
        if kind == "response.metadata":
            identity.observe(event.get("headers") or {})
        elif kind == "error":
            error = event.get("error") or event
            code = error.get("code")
            status = event.get("status") or event.get("status_code") or error.get("status")
            if type(status) is not int:
                status = 429 if code in ("rate_limit_exceeded", "usage_limit_reached", "insufficient_quota") else 400
            raise APIStatusError(str(error.get("message") or code or "Codex stream failed."),
                                 response=httpx.Response(status, headers=transport.headers, request=transport.request),
                                 body=error)
        event["type"] = kind
        yield output.normalize(event)


def stream_attempt(
    client, route, model, messages, schemas, effort, guard, show, *, tool_choice=None, cancel=None, prompt_cache_key=None,
    turn=None, usage_observer=None, tracker=None, on_text=None, speculation=None, websocket=None, codex_turn=None,
    live_source=None, effort_control=None,
):
    body, fingerprint = request_body(route, model, messages, schemas, effort)
    if tool_choice and schemas:
        body["tool_choice"] = ({"type": "function", "name": tool_choice["function"]["name"]}
                               if isinstance(tool_choice, dict) else tool_choice)
    if prompt_cache_key:
        body["prompt_cache_key"] = prompt_cache_key
    codex_route = route if route.api == codex_responses.API else None
    if codex_route:
        codex_turn = codex_turn or codex_responses.SessionIdentity(None).turn()
        body["model"] = (route.catalog_model or {}).get("requestModelId") or body["model"]
        body.setdefault("prompt_cache_key", codex_turn.session.session_id)
    visible, thinking, tool_draft = [], [], False
    stream_tools = {}
    usage = reported_usage = None
    try:
        with response_events(client, body, websocket, codex_route=codex_route, identity=codex_turn,
                             live_source=live_source, effort_control=effort_control) as stream:
            for event in stream:
                if cancel:
                    cancel()
                kind = event["type"]
                if kind == "response.created":
                    if tracker:
                        tracker.restart()
                    if speculation:
                        speculation.restart()
                    stream_tools.clear()
                elif kind in ("response.output_text.delta", "response.refusal.delta"):
                    piece = event["delta"]
                    guard.push(None, piece, tool_draft)
                    if turn:
                        turn.delta("text", piece)
                    if on_text and piece:
                        on_text(piece)
                    visible.append(piece)
                    if show:
                        print(piece, end="", flush=True)
                    if tracker:
                        tracker.text("text", piece)
                elif kind in ("response.reasoning_text.delta", "response.reasoning_summary_text.delta"):
                    guard.push(event["delta"], None, tool_draft)
                    if turn:
                        turn.delta("thinking", event["delta"])
                    thinking.append(event["delta"])
                    if tracker:
                        tracker.text("thinking", event["delta"])
                elif kind in ("response.output_text.done", "response.refusal.done") and tracker:
                    tracker.end_text("text")
                elif kind in ("response.reasoning_text.done", "response.reasoning_summary_text.done") and tracker:
                    tracker.end_text("thinking")
                elif kind in ("response.function_call_arguments.delta", "response.function_call_arguments.done") or (
                    kind in ("response.output_item.added", "response.output_item.done")
                    and event["item"]["type"] == "function_call"
                ):
                    tool_draft = True
                    guard.push(None, None, True)
                    checking = tracker and tracker.rules.stream_tools
                    if checking or speculation:
                        index = event.get("output_index")
                        if item := event.get("item"):
                            stream_tools[index] = [item.get("name", ""), item.get("call_id"), item.get("arguments", "")]
                        if index in stream_tools:
                            current = stream_tools[index]
                            if kind == "response.function_call_arguments.delta":
                                current[2] += event["delta"]
                            elif kind == "response.function_call_arguments.done":
                                current[2] = event.get("arguments", current[2])
                            final = kind in ("response.function_call_arguments.done", "response.output_item.done")
                            if checking:
                                tracker.tool(index, current[0], current[2], current[1], final=final)
                            if speculation:
                                speculation.observe(index, current[0], current[1], current[2], final=final)
                elif kind in ("response.completed", "response.incomplete", "response.failed"):
                    response = event["response"]
                    reported_usage = response.get("usage")
                    usage = Usage.from_response(response)
                    details = {"text": "".join(visible), "tool_draft": tool_draft, "usage": usage}
                    error = response.get("error") or {}
                    if error.get("code") in ("context_length_exceeded", "context_window_exceeded"):
                        raise ContextWindowExceeded("Responses context window exceeded.", **details)
                    if kind == "response.incomplete" and (response.get("incomplete_details") or {}).get(
                        "reason"
                    ) == "max_output_tokens":
                        raise IncompleteTurn("Responses reached the generation limit.", **details)
                    if kind != "response.completed" or response.get("status") != "completed":
                        raise StreamInterrupted("Responses did not complete the turn.", **details)
                    guard.finish()
                    if tool_choice in ("none", "auto"):
                        text = "\n".join(piece for item in response["output"] if (piece := item_text(item, "\n")))
                        return {"role": "assistant", "content": text or None}, usage
                    tool_draft |= any(item.get("type") == "function_call" for item in response["output"])
                    message = completed_message(response, route, model, fingerprint)
                    if thinking:
                        message["_thinking"] = "".join(thinking)
                    if (not response.get("_steered") and not message.get("tool_calls")
                        and not (message["content"] or "").strip() and not any(
                        item["type"] == "reasoning" and (item.get("encrypted_content") or "").strip()
                        for item in (message.get("_responses") or {}).get("items", ())
                    )):
                        raise EmptyTurn("Responses stopped without actionable output.", usage=usage)
                    if show and not visible and message["content"]:
                        visible.append(message["content"])
                        print(message["content"], end="", flush=True)
                    return message, usage
    except APIStatusError:
        raise
    except (APIError, ValueError, KeyError, TypeError, *TRANSPORT_ERRORS) as error:
        if getattr(error, "code", None) in ("context_length_exceeded", "context_window_exceeded"):
            raise ContextWindowExceeded(
                "Responses context window exceeded.", text="".join(visible), tool_draft=tool_draft, usage=usage,
            ) from error
        raise StreamInterrupted(
            f"Responses stream interrupted ({type(error).__name__}).",
            text="".join(visible), tool_draft=tool_draft, usage=usage,
        ) from error
    finally:
        if usage_observer and reported_usage is not None:
            usage_observer(reported_usage)
        if show and visible:
            print()
    raise StreamInterrupted(
        "Responses stream ended before completion.", text="".join(visible), tool_draft=tool_draft, usage=usage,
    )


async def recap(client, route, messages, effort, meter=None, *, effort_control=None):
    body, _ = request_body(route, route.model, messages, [], effort, max_tokens=512)
    if effort_control:
        effort_control.apply(body)
    if meter:
        meter.admit()
    result = await client.responses.create(**sdk_body(body))
    if meter:
        meter.observe(result.usage)
    return result.output_text if result.status == "completed" else None
