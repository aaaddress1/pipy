"""Normalize provider counters and batch session usage for the Auth Broker."""

import asyncio
import os
import re
import socket
from time import sleep
from uuid import uuid4

from auth_broker_cache import number


COUNTERS = ("requests", "inputTokens", "outputTokens", "cacheReadTokens", "cacheWriteTokens", "costUsd")
INSTALL_ID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)


def read_install_id(path):
    try:
        value = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None
    return value if INSTALL_ID.fullmatch(value) else None


def install_id(path):
    if current := read_install_id(path):
        return current
    value = str(uuid4())
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            contents = path.read_bytes()
            # A concurrently created file can briefly be empty before its write.
            if contents and not read_install_id(path) and path.read_bytes() == contents:
                path.unlink()
        except FileNotFoundError:
            pass
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            file.write(value + "\n")
    except FileExistsError:
        for _ in range(5):
            if current := read_install_id(path):
                return current
            sleep(0.01)
    except OSError:
        pass
    return value


def field(source, key):
    return source.get(key) if isinstance(source, dict) else getattr(source, key, None)


def count(value):
    return value if number(value) and value >= 0 else 0


def anthropic_counters(raw, retention):
    read = count(field(raw, "cache_read_input_tokens"))
    written = count(field(raw, "cache_creation_input_tokens"))
    prompt = count(field(raw, "input_tokens")) + read + written
    output = count(field(raw, "output_tokens"))
    creation = field(raw, "cache_creation")
    write_1h = min(written, count(field(creation, "ephemeral_1h_input_tokens")))
    if not creation and retention == "1h":
        write_1h = written
    return prompt, output, read, written, write_1h


def normalize_usage(api, provider, raw, prices, retention, cached_tokens_header=None, *, cost_detail=None):
    if raw is None:
        return None
    if api in ("anthropic-messages", "bedrock-converse-stream"):
        iterations = field(raw, "iterations")
        if isinstance(iterations, list) and any(field(item, "type") == "compaction" for item in iterations):
            prompt, output, read, written, write_1h = map(sum, zip(*(
                anthropic_counters(item, retention) for item in iterations
            )))
        else:
            prompt, output, read, written, write_1h = anthropic_counters(raw, retention)
    else:
        responses = api in ("openai-responses", "openai-codex-responses")
        details = field(raw, "input_tokens_details" if responses else "prompt_tokens_details")
        prompt = count(field(raw, "input_tokens" if responses else "prompt_tokens"))
        output = count(field(raw, "output_tokens" if responses else "completion_tokens"))
        hit, miss = field(raw, "prompt_cache_hit_tokens"), field(raw, "prompt_cache_miss_tokens")
        details_read = field(details, "cached_tokens")
        if responses:
            read = count(details_read if details_read is not None else hit)
            orchestration_input = count(field(details, "orchestration_input_tokens"))
            orchestration_output = count(field(field(raw, "output_tokens_details"), "orchestration_output_tokens"))
            orchestration = orchestration_input + orchestration_output
            total = field(raw, "total_tokens")
            if number(total) and orchestration and abs(total - prompt - output) <= abs(total - prompt - output - orchestration):
                prompt = max(0, prompt - orchestration_input)
                output = max(0, output - orchestration_output)
                read = max(0, read - min(orchestration_input, count(field(details, "orchestration_input_cached_tokens"))))
        else:
            candidates = (field(raw, "cached_tokens"), hit, details_read, field(raw, "cachedContentTokenCount"))
            read = next((value for value in candidates if count(value) > 0), 0)
            if not read and not any(number(value) and value == 0 for value in candidates) and cached_tokens_header:
                try:
                    read = count(float(cached_tokens_header))
                except ValueError:
                    pass
        details_write = field(details, "cache_write_tokens")
        # DeepSeek's miss counter is uncached input, except when an explicit write meter is supplied.
        written = count(details_write) if details_write is not None else 0 if number(hit) and count(miss) else count(miss)
        write_1h = 0

    input_tokens = max(0, prompt - read - written)
    cost = 0
    if callable(prices):
        prices = prices(prompt)
    if prices:
        parts = (input_tokens * prices.input, output * prices.output, read * prices.read,
                 (written - write_1h) * prices.write_5m + write_1h * prices.write_1h)
        cost = sum(parts) / 1_000_000
        if cost_detail is not None:
            cost_detail.update(zip(("input", "output", "cacheRead", "cacheWrite"),
                                   (value / 1_000_000 for value in parts)))
    if provider in ("openrouter", "cline-pass"):
        reported = field(raw, "cost")
        upstream = field(field(raw, "cost_details"), "upstream_inference_cost")
        if field(raw, "is_byok") is True and number(upstream) and upstream >= 0:
            reported = count(reported) + upstream
        if number(reported) and reported >= 0:
            cost = reported
    return dict(zip(COUNTERS, (1, input_tokens, output, read, written, count(cost))))


class ObservedUsage:
    def __init__(self, request, identity_path, interval=10):
        self.request = request
        self.identity_path = identity_path
        self.interval = interval
        self.identity = None
        self.entries = {}
        self.flush_at = 0
        self.wake = asyncio.Event()
        self.task = None
        self.closing = False
        self.unsupported = False

    def merge(self, entry):
        if not self.entries:
            self.flush_at = asyncio.get_running_loop().time() + self.interval
        key = entry["provider"], entry["model"]
        if current := self.entries.get(key):
            current["at"] = max(current["at"], entry["at"])
            for counter in COUNTERS:
                current[counter] += entry[counter]
        else:
            self.entries[key] = entry

    def record(self, provider, model, at, counters):
        if self.closing or self.unsupported:
            return
        self.merge({"provider": provider, "model": model, "at": at, **counters})
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self.run())

    async def run(self):
        while self.entries:
            if not self.closing:
                try:
                    delay = max(0, self.flush_at - asyncio.get_running_loop().time())
                    await asyncio.wait_for(self.wake.wait(), delay)
                except TimeoutError:
                    pass
            entries, self.entries = list(self.entries.values()), {}
            if self.identity is None:
                self.identity = {
                    "installId": await asyncio.to_thread(install_id, self.identity_path),
                    "hostname": socket.gethostname(),
                    "app": os.getenv("OMP_APP_NAME", "").strip() or "omp",
                }
            try:
                await self.request("POST", "/v1/usage/observed", json={**self.identity, "entries": entries})
            except OSError as error:
                if getattr(error, "status", None) in (400, 404, 501):
                    self.unsupported = True
                    self.entries.clear()
                    return
                if not self.closing:
                    for entry in entries:
                        self.merge(entry)

    async def close(self):
        self.closing = True
        self.wake.set()
        if self.task is not None:
            try:
                await asyncio.wait_for(self.task, timeout=2)
            except TimeoutError:
                pass
        self.entries.clear()
