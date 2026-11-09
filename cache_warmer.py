"""Schedule bounded Anthropic prefix replays from confirmed cache usage."""

import asyncio
import sys
import threading
import time
from dataclasses import dataclass


TTL_SECONDS = {"5m": 300, "1h": 3600}
IDLE_MAX_AGE = 1800
ACTIVE_MAX_AGE = 3600
MIN_SAVINGS = 0.05


def refresh_delay(ttl):
    return min(ttl * 0.9, ttl - 10)


@dataclass(frozen=True)
class Prices:
    input: float
    output: float
    read: float
    write_5m: float = 0
    write_1h: float = 0

    def expected_savings(self, tokens, tier, idle):
        write = self.write_1h if tier == "1h" else self.write_5m
        hit = self.read * tokens / 1_000_000
        miss = (write or self.input) * tokens / 1_000_000
        warm = hit + self.output / 1_000_000
        return (0.15 if idle else 1) * max(0, miss - hit) - warm


@dataclass
class Entry:
    tier: str
    started: float
    touched: float
    prompt_tokens: int
    prices: Prices | None
    replay: object
    confirmed: bool
    idle: bool = False
    warmable: bool = True
    valid: bool = True

    @property
    def ttl(self):
        return TTL_SECONDS[self.tier]


class CacheWarmer:
    def __init__(self, mode, clock=time.monotonic, timer_factory=threading.Timer):
        self.mode = mode
        self.clock = clock
        self.timer_factory = timer_factory
        self.lock = threading.RLock()
        self.entry = None
        self.timer = None
        self.timer_version = 0
        self.running = None
        self.worker = None

    def observe(self, configured_tier, prices, sent_at, usage, replay, *, confirmed=True):
        if configured_tier not in TTL_SECONDS:
            return
        if not (usage.cache_read_tokens or usage.cache_write_tokens):
            if confirmed:
                self.invalidate()
            return
        tier = usage.cache_ttl or configured_tier
        if tier not in TTL_SECONDS:
            return
        with self.lock:
            entry = self.entry
            if entry and entry.started == sent_at:
                entry.confirmed |= confirmed
                if entry.touched == sent_at:
                    entry.tier = tier
                    entry.prompt_tokens = usage.prompt_tokens
            else:
                self.invalidate()
                self.entry = Entry(tier, sent_at, sent_at, usage.prompt_tokens, prices, replay, confirmed)
            if not self.running:
                self.schedule()

    def settle(self):
        with self.lock:
            if self.entry:
                self.entry.idle = True
                if self.mode == "streaming":
                    self.entry.warmable = False
                    self.cancel_work()
                elif not self.running:
                    self.schedule()

    def idle_deadline(self, armed_at, fallback):
        with self.lock:
            entry = self.entry
            if entry and entry.valid and entry.confirmed and entry.tier == "1h" and self.clock() < entry.touched + entry.ttl:
                return max(armed_at, entry.touched + refresh_delay(entry.ttl) - 5)
        return armed_at + fallback

    def cancel_timer(self):
        self.timer_version += 1
        if self.timer:
            self.timer.cancel()
            self.timer = None

    def cancel_work(self):
        self.cancel_timer()
        if self.running:
            loop, task = self.running
            self.running = None
            loop.call_soon_threadsafe(task.cancel)

    def invalidate(self):
        with self.lock:
            self.entry = None
            self.cancel_work()

    def close(self):
        self.invalidate()
        worker = self.worker
        if worker and worker is not threading.current_thread():
            worker.join(timeout=2)

    def schedule(self):
        self.cancel_timer()
        entry = self.entry
        if not entry or not self.eligible(entry):
            return
        due = entry.touched + refresh_delay(entry.ttl)
        horizon = entry.started + (IDLE_MAX_AGE if entry.idle else ACTIVE_MAX_AGE)
        if due > horizon or self.clock() >= entry.touched + entry.ttl:
            return
        timer = self.timer_factory(max(0, due - self.clock()), self.refresh, args=(entry, self.timer_version))
        timer.daemon = True
        self.timer = timer
        timer.start()

    def eligible(self, entry):
        return bool(
            entry.warmable and self.mode != "off" and (not entry.idle or self.mode == "idle")
            and entry.prices and entry.prompt_tokens > 0
            and entry.prices.expected_savings(entry.prompt_tokens, entry.tier, entry.idle) >= MIN_SAVINGS
        )

    def refresh(self, entry, version):
        asyncio.run(self.replay(entry, version))

    async def replay(self, entry, version):
        current = (asyncio.get_running_loop(), asyncio.current_task())
        with self.lock:
            if self.entry is not entry or version != self.timer_version:
                return
            self.timer = None
            if self.running or not self.eligible(entry):
                return
            sent_at = self.clock()
            horizon = entry.started + (IDLE_MAX_AGE if entry.idle else ACTIVE_MAX_AGE)
            if sent_at > horizon or sent_at >= entry.touched + entry.ttl:
                return
            self.running = current
            self.worker = threading.current_thread()
        try:
            try:
                usage = await entry.replay()
            except Exception as error:
                with self.lock:
                    if self.entry is entry:
                        entry.warmable = False
                        print(f"[cache refresh failed: {type(error).__name__}]", file=sys.stderr)
                return
            with self.lock:
                if self.entry is not entry or self.running != current:
                    return
                hit = usage.cache_read_tokens > 0 and usage.cache_write_tokens == 0
                if hit or usage.cache_write_tokens:
                    entry.touched = sent_at
                    entry.prompt_tokens = usage.prompt_tokens
                    entry.tier = usage.cache_ttl or entry.tier
                if not hit:
                    entry.warmable = False
                    if not usage.cache_write_tokens:
                        entry.valid = False
                print(f"[cache refresh: {'hit' if hit else 'miss'}]", file=sys.stderr)
                if hit:
                    self.schedule()
        except asyncio.CancelledError:
            return
        finally:
            with self.lock:
                if self.running == current:
                    self.running = None
