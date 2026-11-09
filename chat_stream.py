"""Read Chat Completions SSE without losing the provider's terminal marker."""

import json
from concurrent.futures import CancelledError
from threading import Timer

import httpx
from openai import APIError
from openai.types.chat import ChatCompletionChunk
from openai.types.completion_usage import CompletionUsage

from cancellable_http import active_turn, sse_frames, turn_scope
from handoff import Cancellation
from observed_usage import field
from recovery import TRANSPORT_ERRORS


TERMINAL_GRACE = 2.5
FINISH_ALIASES = {"end": "stop", "max_tokens": "length", "function_call": "tool_calls"}
TRANSIENT_FINISH = frozenset(("error", "network_error", "insufficient_system_resource", "unknown", "other",
                              "unspecified", "finish_reason_unspecified"))
NATIVE_FAILURES = frozenset(("MALFORMED_RESPONSE", "MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL", "NETWORK_ERROR"))
USAGE_DEFAULTS = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def finish_reason(reason):
    if not isinstance(reason, str) or not reason:
        return None
    normalized = reason.lower()
    return FINISH_ALIASES.get(normalized, normalized)


def native_error(choice):
    reason = getattr(choice, "native_finish_reason", None)
    return reason if isinstance(reason, str) and reason.upper() in NATIVE_FAILURES else None


def has_cache_details(usage):
    values = (field(usage, "cached_tokens"), field(usage, "prompt_cache_hit_tokens"),
              field(usage, "cachedContentTokenCount"), field(field(usage, "prompt_tokens_details"), "cached_tokens"))
    return any(type(value) in (int, float) and value > 0 for value in values)


def choice_usage(choice):
    usage = getattr(choice, "usage", None)
    if isinstance(usage, dict):
        return CompletionUsage.construct(**(USAGE_DEFAULTS | usage))
    return None


class ChatStream:
    def __init__(self, client, **request):
        self.client, self.request = client, request
        self.parent = active_turn()
        self.signal = Cancellation(self.parent, reason="Chat Completions terminal grace elapsed.")
        self.timer = None
        self.done = self.detached = False

    def __enter__(self):
        with turn_scope(self.signal):
            self.stream = self.client.chat.completions.create(stream=True, **self.request)
        self.response = self.stream.response
        self.frames = sse_frames(self.response.iter_lines())
        return self

    def mark_finished(self):
        if self.timer is None:
            self.timer = Timer(TERMINAL_GRACE, self.signal.cancel)
            self.timer.daemon = True
            self.timer.start()

    def __iter__(self):
        for name, payload in self.frames:
            self.signal.check()
            if payload.strip() == "[DONE]":
                self.done = True
                return
            data = json.loads(payload)
            if not isinstance(data, dict):
                raise ValueError("Chat Completions SSE event must be an object.")
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
        if self.timer:
            self.timer.cancel()
            self.timer.join()
        self.stream.close()

    def __exit__(self, *_):
        if not self.detached:
            self.close()
