"""Semantic progress deadlines for OpenAI-family streaming transports."""

import math
import os
from contextlib import contextmanager
from time import monotonic

import httpx

from cancellable_http import active_turn, turn_scope
from handoff import Cancellation


RESPONSES_PROGRESS = frozenset((
    "response.created", "response.output_item.added", "response.reasoning_summary_part.added",
    "response.reasoning_summary_text.delta", "response.reasoning_summary_text.done",
    "response.reasoning_summary_part.done", "response.reasoning_text.delta", "response.content_part.added",
    "response.output_text.delta", "response.refusal.delta", "response.function_call_arguments.delta",
    "response.function_call_arguments.done", "response.custom_tool_call_input.delta",
    "response.custom_tool_call_input.done", "response.output_item.done", "response.completed",
    "response.incomplete", "response.failed", "response.done", "error",
))


class StreamDeadline(httpx.TimeoutException):
    pass


def seconds(value, fallback):
    try:
        number = float(value.strip() or "0" if isinstance(value, str) else value)
    except (TypeError, ValueError, OverflowError):
        return fallback
    return max(0, int(number)) / 1000 if math.isfinite(number) else fallback


def timing(route, *, codex_socket=False):
    if codex_socket:
        return (seconds(os.getenv("PI_CODEX_WEBSOCKET_FIRST_EVENT_TIMEOUT_MS"), 300),
                seconds(os.getenv("PI_CODEX_WEBSOCKET_IDLE_TIMEOUT_MS"), 300))
    compat = (route.catalog_model or {}).get("compat") or {} if route and route.api != "openai-codex-responses" else {}
    idle = seconds(os.getenv("PI_OPENAI_STREAM_IDLE_TIMEOUT_MS", os.getenv("PI_STREAM_IDLE_TIMEOUT_MS")),
                   seconds(compat.get("streamIdleTimeoutMs"), 300))
    first_default = seconds(compat.get("streamFirstEventTimeoutMs"), 300)
    explicit = os.getenv("PI_OPENAI_STREAM_FIRST_EVENT_TIMEOUT_MS")
    if explicit is not None:
        return seconds(explicit, first_default), idle
    first = seconds(os.getenv("PI_STREAM_FIRST_EVENT_TIMEOUT_MS"), first_default)
    return (max(first, idle) if first else 0), idle


class Watchdog(Cancellation):
    def __init__(self, label, limits):
        super().__init__(active_turn())
        self.label = label
        self.first, self.idle = limits
        self.phase, self.expires = None, None

    def arm(self, phase, duration):
        self.phase = phase
        self.expires = monotonic() + duration if duration else None

    def network_budget(self):
        return max(0, self.expires - monotonic()) if self.expires is not None else None

    def check(self):
        super().check()
        if self.expires is not None and monotonic() >= self.expires:
            raise self.deadline_error()

    def deadline_error(self):
        self.expires = monotonic()
        detail = {"headers": "timed out before response headers", "first": "timed out while waiting for the first event",
                  "idle": "stalled while waiting for the next event", "terminal": "terminal grace elapsed"}[self.phase]
        return StreamDeadline(f"{self.label} {detail}.")

    def observe(self, progress):
        self.check()
        if progress and self.phase != "terminal":
            self.arm("idle", self.idle)

    @contextmanager
    def reading(self):
        try:
            yield
        except Exception:
            self.check()
            raise

    @contextmanager
    def opening(self):
        self.arm("headers", 0)
        with turn_scope(self), self.reading():
            yield
        self.arm("first", self.first)

    def start_request(self):
        if self.phase == "headers" and self.expires is None:
            self.arm("headers", self.first)

    def client(self, client):
        timeout = client.timeout
        timeout = timeout if isinstance(timeout, httpx.Timeout) else httpx.Timeout(timeout)
        # The CLI's default socket read timeout must not undercut the semantic budgets.
        if timeout.read == 120 and max(self.first, self.idle) > timeout.read:
            timeout = httpx.Timeout(connect=timeout.connect, read=max(self.first, self.idle),
                                    write=timeout.write, pool=timeout.pool)
        return client.with_options(timeout=timeout, max_retries=0) if self.first or self.idle else client
