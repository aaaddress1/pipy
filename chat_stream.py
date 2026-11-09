"""Read Chat Completions SSE without losing the provider's terminal marker."""

import json
from concurrent.futures import CancelledError

import httpx
from openai import APIError
from openai.types.chat import ChatCompletionChunk
from openai.types.completion_usage import CompletionUsage

from cancellable_http import sse_frames
from observed_usage import field
from recovery import TRANSPORT_ERRORS
from stream_watchdog import Watchdog


TERMINAL_GRACE = 2.5
FINISH_ALIASES = {"end": "stop", "max_tokens": "length", "function_call": "tool_calls"}
TRANSIENT_FINISH = frozenset(("error", "network_error", "insufficient_system_resource", "unknown", "other",
                              "unspecified", "finish_reason_unspecified"))
NATIVE_FAILURES = frozenset(("MALFORMED_RESPONSE", "MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL", "NETWORK_ERROR"))
REPETITION_REASONS = frozenset(("repetition", "repetition_detected"))
USAGE_DEFAULTS = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def finish_reason(reason):
    if not isinstance(reason, str) or not reason:
        return None
    normalized = reason.lower()
    return FINISH_ALIASES.get(normalized, normalized)


def native_error(choice):
    reason = getattr(choice, "native_finish_reason", None)
    return reason if isinstance(reason, str) and reason.upper() in NATIVE_FAILURES else None


def is_repetition(value):
    return isinstance(value, str) and value.lower() in REPETITION_REASONS


def repetition_reason(choice):
    finish, stop = field(choice, "finish_reason"), field(choice, "stop_reason")
    if not (is_repetition(finish) or is_repetition(stop)):
        return None
    return " ".join(f"{key}={value}" for key, value in (("finish_reason", finish), ("stop_reason", stop))
                    if isinstance(value, str) and value)


def has_cache_details(usage):
    values = (field(usage, "cached_tokens"), field(usage, "prompt_cache_hit_tokens"),
              field(usage, "cachedContentTokenCount"), field(field(usage, "prompt_tokens_details"), "cached_tokens"))
    return any(type(value) in (int, float) and value > 0 for value in values)


def choice_usage(choice):
    usage = getattr(choice, "usage", None)
    if isinstance(usage, dict):
        return CompletionUsage.construct(**(USAGE_DEFAULTS | usage))
    return None


def progress(data):
    choices = data.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    delta = choice.get("delta")
    native = choice.get("native_finish_reason")
    return (data.get("usage") is not None or choice.get("usage") is not None
            or bool(choice.get("finish_reason")) or is_repetition(choice.get("stop_reason"))
            or isinstance(native, str) and native.upper() in NATIVE_FAILURES
            or isinstance(delta, dict) and any(delta.get(key) for key in (
                "content", "refusal", "reasoning", "reasoning_content", "reasoning_text", "tool_calls", "function_call",
            )))


class ChatStream:
    def __init__(self, client, limits, **request):
        self.client, self.request = client, request
        self.signal = Watchdog("Chat Completions stream", limits)
        self.done = self.detached = False

    def __enter__(self):
        client = self.signal.client(self.client)
        with self.signal.opening():
            self.stream = client.chat.completions.create(stream=True, **self.request)
        self.response = self.stream.response
        self.frames = sse_frames(self.response.iter_lines())
        return self

    def mark_finished(self):
        if self.signal.phase != "terminal":
            self.signal.arm("terminal", TERMINAL_GRACE)

    def __iter__(self):
        for name, payload in self.frames:
            self.signal.check()
            if payload.strip() == "[DONE]":
                self.done = True
                return
            data = json.loads(payload)
            if not isinstance(data, dict):
                raise ValueError("Chat Completions SSE event must be an object.")
            self.signal.observe(progress(data))
            if name == "error" or data.get("error"):
                error = data.get("error") or data
                message = error.get("message") if isinstance(error, dict) else None
                raise APIError(str(message or "Chat Completions stream reported an error."),
                               request=self.response.request, body=error)
            if isinstance(usage := data.get("usage"), dict):
                data["usage"] = USAGE_DEFAULTS | usage
            yield ChatCompletionChunk.construct(**data)

    def detach(self):
        # The HTTP client keeps ownership of the remaining body until DONE or the existing grace deadline.
        transport = self.response.extensions.get("pipy_transport")
        if transport:
            transport.drain_tail(self.drain)
            self.detached = True

    def drain(self):
        try:
            for _, payload in self.frames:
                self.signal.check()
                if payload.strip() == "[DONE]":
                    break
        except (CancelledError, ValueError, httpx.HTTPError, *TRANSPORT_ERRORS):
            pass
        finally:
            self.close()

    def close(self):
        self.stream.close()

    def __exit__(self, *_):
        if not self.detached:
            self.close()
