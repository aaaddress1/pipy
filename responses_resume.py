"""Recover an interrupted SSE run on providers that finish stored responses."""

import re
import time
from concurrent.futures import CancelledError

import httpx
from stream_watchdog import RESPONSES_PROGRESS
from openai import APIConnectionError, APIError, APIStatusError

from cancellable_http import active_turn
from handoff import deadline
from recovery import TRANSPORT_ERRORS


POLL_INTERVAL = 5
POLL_DEADLINE = 120
POLL_LIMIT = 24
REQUEST_TIMEOUT = 30
TERMINAL = frozenset(("response.completed", "response.done", "response.incomplete", "response.failed",
                      "response.cancelled", "error"))
THINKING = frozenset(("response.reasoning_text.delta", "response.reasoning_summary_text.delta"))
TEXT = frozenset(("response.output_text.delta", "response.refusal.delta"))


def poll(client, identifier):
    parent = active_turn()
    retrieve = client.with_options(max_retries=0).responses.with_raw_response.retrieve
    end = time.monotonic() + POLL_DEADLINE
    try:
        with deadline(POLL_DEADLINE, parent) as budget:
            for attempt in range(POLL_LIMIT):
                budget.check()
                if attempt:
                    if time.monotonic() + POLL_INTERVAL >= end:
                        return None
                    budget.wait(POLL_INTERVAL)
                remaining = end - time.monotonic()
                if remaining <= 0:
                    return None
                timeout = min(REQUEST_TIMEOUT, remaining)
                try:
                    with deadline(timeout, budget) as request:
                        result = retrieve(identifier, timeout=timeout)
                        request.check()
                except APIStatusError as error:
                    if error.status_code != 404 and error.status_code < 500:
                        return None
                    continue
                except (CancelledError, *TRANSPORT_ERRORS):
                    budget.check()
                    continue
                if result.status_code != 200:
                    return None
                try:
                    response = result.http_response.json()
                except ValueError:
                    return None
                if not isinstance(response, dict) or response.get("id") != identifier:
                    return None
                status, output = response.get("status"), response.get("output")
                if status in ("failed", "cancelled"):
                    return None
                if status == "completed" or status == "incomplete" and isinstance(output, list) and output:
                    budget.check()
                    return response
    except CancelledError:
        if parent:
            parent.check()
    return None


def socket_drop(error):
    if isinstance(error, APIConnectionError) and error.__cause__ is not None:
        error = error.__cause__
    return isinstance(error, (httpx.ReadError, httpx.RemoteProtocolError)) or re.search(
        r"\bsocket connection (?:was )?closed unexpectedly\b|^(?:error:\s*)?socket is closed\.?$", str(error).strip(), re.I,
    ) is not None


def item_thinking(item):
    parts = item.get("summary") or item.get("content") or ()
    return "\n".join(part["text"] for part in parts if isinstance(part, dict) and isinstance(part.get("text"), str))


def visible_part(part):
    if part.get("type") in ("output_text", "refusal"):
        return bool((part.get("text") or part.get("refusal") or "").strip())
    return True


def replay(response, message):
    thoughts = []
    for index, item in enumerate(response["output"]):
        if item["type"] == "reasoning" and (thought := item_thinking(item)):
            thoughts.append(thought)
        if item["type"] == "message" and item["role"] == "assistant":
            for part in item.get("content", ()):
                if part["type"] == "output_text" and isinstance(part.get("text"), str):
                    yield {"type": "response.output_text.delta", "delta": part["text"]}
                elif part["type"] == "refusal" and isinstance(part.get("refusal"), str):
                    yield {"type": "response.refusal.delta", "delta": part["refusal"]}
        yield {"type": "response.output_item.done", "output_index": index, "item": item}
    yield {"type": "response." + response["status"], "response": response,
           "_resumed_message": message, "_resumed_thinking": "\n".join(thoughts)}


def events(client, stream, prepare, watch):
    identifier = None
    thinking, unsafe, terminal = False, False, False
    failure = None
    try:
        with stream, watch.reading():
            for incoming in stream:
                event = incoming.to_dict()
                kind = event["type"]
                watch.observe(kind in RESPONSES_PROGRESS)
                if kind == "response.created":
                    identifier = event["response"].get("id")
                elif kind in TERMINAL:
                    terminal = True
                elif kind in THINKING:
                    thinking |= bool(event["delta"].strip())
                elif kind in TEXT:
                    unsafe |= bool(event["delta"].strip())
                elif kind in ("response.output_text.done", "response.refusal.done"):
                    unsafe |= bool((event.get("text") or event.get("refusal") or "").strip())
                elif kind in ("response.content_part.added", "response.content_part.done"):
                    unsafe |= visible_part(event["part"])
                elif kind in ("response.output_item.added", "response.output_item.done"):
                    item = event["item"]
                    if item["type"] == "reasoning":
                        thinking |= bool(item_thinking(item).strip())
                    elif item["type"] == "message" and item.get("role") == "assistant":
                        unsafe |= any(map(visible_part, item.get("content", ())))
                    else:
                        unsafe = True
                elif kind.startswith(("response.function_call_", "response.custom_tool_call_", "response.mcp_",
                                      "response.image_generation_call.", "response.web_search_call.",
                                      "response.file_search_call.", "response.code_interpreter_call.")):
                    unsafe = True
                yield event
    except (APIError, *TRANSPORT_ERRORS) as error:
        if not socket_drop(error):
            raise
        failure = error

    if (thinking and not unsafe and not terminal and isinstance(identifier, str)
        and re.fullmatch(r"[A-Za-z0-9_.-]+", identifier) and identifier not in (".", "..")):
        response = poll(client, identifier)
        if response is not None:
            try:
                if not isinstance(response.get("output"), list):
                    raise ValueError("Invalid stored Responses output.")
                if (response["status"] == "incomplete"
                    and (response.get("incomplete_details") or {}).get("reason") == "content_filter"):
                    raise ValueError("Stored Responses output was blocked.")
                recovered = tuple(replay(response, prepare(response)))
            except (ValueError, KeyError, TypeError, AttributeError):
                pass
            else:
                yield from recovered
                return
    if failure:
        raise failure
