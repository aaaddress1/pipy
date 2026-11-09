"""Resolve compaction triggers and the independent prompt budget."""

import math
import re
from dataclasses import dataclass, replace

from auth_broker import dotted
from auth_broker_cache import number


DEFAULT_RESERVE = 16_384


def setting(layers, key, default=None):
    return next((value for layer in reversed(layers) if (value := dotted(layer, key)) is not None), default)


def threshold_map(layers, key, *, model=False):
    entries = {}
    prefix = key + "."
    for layer in layers:
        configured = dotted(layer, key)
        if configured is not None and not isinstance(configured, dict):
            raise ValueError(f"{key} must be a mapping.")
        entries.update((name[len(prefix):], value) for name, value in layer.items()
                       if isinstance(name, str) and name.startswith(prefix))
        entries.update(configured or {})
    parsed = {}
    for name, value in entries.items():
        if not isinstance(name, str):
            raise ValueError(f"{key} must have string keys.")
        provider, _, identifier = name.partition("/")
        if model and (not provider or not identifier or "*" in name[:-1]):
            raise ValueError(f"{key} keys must name {'a provider/model or a provider/prefix*' if model else 'an agent'}.")
        if value is None:
            continue
        if number(value) and 0 < value < 2**53 and int(value) == value:
            parsed[name] = ("base" if model else "fixed", int(value))
            continue
        if isinstance(value, str):
            value = value.strip()
            if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?%", value) and 0 < (percent := float(value[:-1])) <= 100:
                parsed[name] = ("percent", percent)
                continue
            if model and re.fullmatch(r"[fF][0-9]+", value) and 0 < (tokens := int(value[1:])) < 2**53:
                parsed[name] = ("fixed", tokens)
                continue
        forms = 'a positive integer, a percentage, or "f90000"' if model else "a positive integer or a percentage"
        raise ValueError(f"{key}.{name} must be {forms}.")
    return parsed


@dataclass(frozen=True)
class Budget:
    percent: float = 70
    fixed: float = -1
    reserve: float | None = None
    keep_recent: int = 20_000
    base: int | None = None

    def budget_reserve(self, window):
        proportional = window * 15 // 100
        reserve = max(proportional, DEFAULT_RESERVE if self.reserve is None else self.reserve)
        minimum = max(1, proportional)
        if reserve >= window or self.reserve is None and reserve >= window - minimum:
            return minimum
        return reserve

    def prompt_budget(self, window):
        return max(0, min(window - 1, window - self.budget_reserve(window))) if window else None

    def trigger(self, window):
        if not window:
            return None
        if self.fixed > 0:
            return self.prompt_budget(window) if self.fixed >= window else max(1, min(window - 1, self.fixed))
        base = min(window, self.base) if self.base else window
        if self.percent <= 0:
            return self.prompt_budget(base)
        return math.floor(base * (min(99, max(1, self.percent)) / 100))

    def recent_budget(self, window):
        if not window:
            return self.keep_recent
        recovery = math.floor(self.trigger(window) * 0.8)
        floor = min(self.budget_reserve(window), DEFAULT_RESERVE if self.reserve is None else self.reserve)
        tail = recovery - floor if recovery > floor else recovery // 2
        return min(self.keep_recent, max(1, tail))

    def summary_limit(self):
        reserve = DEFAULT_RESERVE if self.reserve is None else self.reserve
        return max(1, min(DEFAULT_RESERVE, math.floor(reserve * 0.8)))


class CompactionPolicy:
    def __init__(self, layers, *, percent=None, keep_recent=None, agent=None):
        if percent is None:
            percent = setting(layers, "compaction.thresholdPercent", 70)
        if keep_recent is None:
            keep_recent = setting(layers, "compaction.keepRecentTokens", 20_000)
        fixed = setting(layers, "compaction.thresholdTokens", -1)
        reserve = setting(layers, "compaction.reserveTokens")
        self.budget = Budget(
            percent=percent if number(percent) else 70,
            fixed=fixed if number(fixed) else -1,
            reserve=reserve if number(reserve) and reserve >= 0 else None,
            keep_recent=int(keep_recent) if number(keep_recent) and keep_recent >= 0 else 20_000,
        )
        agents = threshold_map(layers, "task.agentCompactionThresholdOverrides")
        models = threshold_map(layers, "compaction.modelThresholds", model=True)
        self.agent = agents.get(agent)
        self.models = models if not self.agent and setting(layers, "compaction.modelThresholdsEnabled") is not False else {}
        self.extended = setting(layers, "extendedContext") is True

    def model_entry(self, route):
        model = route.catalog_model
        if model:
            selector = f"{model['provider']}/{model['id']}"
        else:
            defaults = {"anthropic-messages": "anthropic", "openai-codex-responses": "openai-codex"}
            selector = f"{route.broker_provider or defaults.get(route.api, 'openai')}/{route.model}"
        if selector in self.models:
            return self.models[selector]
        key = max((key for key in self.models if key.endswith("*") and selector.startswith(key[:-1])),
                  key=len, default=None)
        return self.models.get(key)

    def for_route(self, route):
        entry = self.agent or self.model_entry(route)
        if not entry:
            return self.budget
        kind, value = entry
        if kind == "base":
            return replace(self.budget, fixed=-1, base=value)
        if kind == "fixed":
            return replace(self.budget, percent=-1, fixed=value)
        return replace(self.budget, percent=value, fixed=-1)

    def fit_route(self, route):
        if not (tiers := route.context_tiers()):
            return route
        standard, extended = tiers
        kind, value = self.model_entry(route) or ("percent", 0)
        opens_extended = kind == "base" and value > standard or kind == "fixed" and value >= standard
        window = extended if self.extended or opens_extended else standard
        return replace(route, window=window) if window != route.window else route
