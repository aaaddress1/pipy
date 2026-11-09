"""Track billable attempts across primary and maintenance requests in a session."""

from dataclasses import dataclass, field
from functools import partial
from threading import Lock
from time import time

from observed_usage import count, field as usage_field, normalize_usage


class CostCapError(RuntimeError):
    code = "cost_cap"


def usd(amount):
    return f"${amount:.{4 if amount < .01 else 2}f}"


def entry_usage(entry):
    usage = None
    if entry["type"] == "model_usage":
        usage = entry.get("usage")
    elif isinstance(message := entry.get("message"), dict):
        if message.get("role") == "assistant":
            usage = message.get("usage")
        elif message.get("role") == "toolResult" and message.get("toolName") == "task":
            usage = usage_field(message.get("details"), "usage")
    return usage if isinstance(usage, dict) else {}


def total_cost(usage):
    return count(usage_field(usage_field(usage, "cost"), "total"))


@dataclass
class Spend:
    amount: float
    warned: bool
    lock: object = field(default_factory=Lock)


class CostGate:
    def __init__(self, session, spent, warning, cap, notice, account=None, parent=None, on_request=None, ancestors=None):
        self.session, self.warning, self.cap = session, warning, cap
        self.notice = parent.notice if parent else notice
        self.account = parent.account if parent else account or Spend(spent, warning is not None and spent >= warning)
        self.ancestors = ancestors if ancestors is not None else ((parent.session, *parent.ancestors) if parent else ())
        self.on_request = on_request
        self.goal_usage = parent.goal_usage if parent else None

    def admit(self):
        with self.account.lock:
            if self.cap is not None and self.account.amount >= self.cap:
                raise CostCapError(
                    f"Cost cap reached: this session spent {usd(self.account.amount)} of its {usd(self.cap)} cap "
                    "(--max-cost / session.maxCost), so the next model request was not sent."
                )

    def record(self, entry):
        warning = None
        with self.account.lock:
            identifier = self.session.record_usage_attempt(entry, self.ancestors)
            self.account.amount += total_cost(entry["usage"])
            if not self.account.warned and self.warning is not None and self.account.amount >= self.warning:
                self.account.warned = True
                warning = (f"Cost warning: this session has spent {usd(self.account.amount)}, past its "
                           f"{usd(self.warning)} warning threshold (--warn-cost / session.warnCost).")
        if warning:
            self.notice("warning", warning)
        if self.goal_usage:
            self.goal_usage(entry["usage"])
        return identifier

    def meter(self, route, purpose, model=None, *, on_usage=None):
        return AttemptMeter(self, route, purpose, model or route.model, on_usage)


class AttemptMeter:
    def __init__(self, gate, route, purpose, model, on_usage=None):
        self.gate, self.route, self.purpose, self.model = gate, route, purpose, model
        self.on_usage = on_usage
        self.at = 0
        self.last = None

    def admit(self):
        self.gate.admit()
        if self.purpose == "agent" and self.gate.on_request:
            self.gate.on_request()
        self.at, self.last = int(time() * 1000), None

    def observe(self, raw, cached_tokens_header=None, *, iteration_cost=None):
        if raw is None or self.last is not None:
            return
        route = self.route
        provider = (route.catalog_model or {}).get("provider") or route.broker_provider or route.name
        api = route.api if route.api in ("chat-completions", "anthropic-messages") else "openai-responses"
        cost = dict.fromkeys(("input", "output", "cacheRead", "cacheWrite"), 0)
        prices = partial(route.prices, timestamp=self.at) if self.model == route.model else None
        counts = normalize_usage(api, provider, raw, prices, route.cache_retention, cached_tokens_header, cost_detail=cost)
        cost["total"] = counts["costUsd"]
        if iteration_cost is not None:
            cost.update(iteration_cost)
        elif api != route.api and (reported := usage_field(raw, "cost")) is not None:
            cost["total"] = count(reported)
        usage = dict(zip(("input", "output", "cacheRead", "cacheWrite"),
                         (counts[key] for key in ("inputTokens", "outputTokens", "cacheReadTokens", "cacheWriteTokens"))))
        usage.update(totalTokens=sum(usage.values()), cost=cost)
        identifier = self.gate.record({"type": "model_usage", "purpose": self.purpose, "provider": provider,
                                       "model": self.model, "stopReason": "stop", "usage": usage})
        self.last = identifier, usage
        if self.on_usage:
            self.on_usage(usage)

    def attach(self, message):
        if self.last:
            identifier, usage = self.last
            message["_usage_attempt"] = identifier
            message["_rpc"]["usage"] = usage

    def finish(self, reason):
        if self.last:
            self.gate.session.finish_usage_attempt(self.last[0], reason, self.gate.ancestors)
