"""Bundled Oh My Pi model rows and catalog policy for the pinned revision."""

import json
import operator
import re
import time
from collections import defaultdict, deque
from fnmatch import fnmatchcase
from functools import cache, lru_cache
from pathlib import Path
from urllib.parse import urlsplit

from cache_warmer import Prices


REVISION = "24439066040e006eb08da7b204f0ac41c79d2de9"
DATA = Path(__file__).with_name("catalog")
APIS = {"openai-completions": "chat-completions", "openai-responses": "openai-responses",
        "anthropic-messages": "anthropic-messages", "openai-codex-responses": "openai-codex-responses",
        "bedrock-converse-stream": "bedrock-converse-stream"}
RESPONSES_APIS = frozenset(("openai-responses", "openai-codex-responses"))
ANTHROPIC_COMPACTION_APIS = frozenset(("anthropic-messages", "bedrock-converse-stream"))
JUDGE_APIS = frozenset(("typesafe", "openrouter-decisions", "openai-decisions", "workers-ai-systemone"))
EFFORTS = ("minimal", "low", "medium", "high", "xhigh", "max")
OPS = {"=": operator.eq, ">": operator.gt, "<": operator.lt, ">=": operator.ge, "<=": operator.le}
UNDEFINED = object()
UPSTREAM_SLUG = re.compile(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(?:/[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)*", re.I | re.ASCII)
GATEWAY_ROUTING = ("openRouterRouting", "vercelGatewayRouting")


def load_snapshot(file):
    data = json.loads((DATA / file).read_text(encoding="utf-8"))
    if data["version"] != 1:
        raise ValueError(f"Unsupported Oh My Pi {file} format.")
    return data


class Catalog:
    def __init__(self, data):
        self.keys = data["keys"]
        self.shapes = data["booleanShapes"]
        self.nodes = data["nodes"]
        self.providers = {provider: (common, {row[0]: row[1:] for row in rows})
                          for provider, common, rows in data["providers"]}
        self.compat = dict(data["compat"])
        self.decoded = {}

    def value(self, encoded):
        if not isinstance(encoded, list):
            return encoded
        index = encoded[0]
        if index == -1:
            return UNDEFINED
        if index not in self.decoded:
            node = self.nodes[index]
            self.decoded[index] = self.object(node) if node[0] == 0 else [
                None if (item := self.value(entry)) is UNDEFINED else item for entry in node[1]
            ]
        return self.decoded[index]

    def object(self, node):
        properties, shape, masks = node[1:]
        result = {self.keys[key]: value for key, entry in zip(properties[::2], properties[1::2])
                  if (value := self.value(entry)) is not UNDEFINED}
        result.update((self.keys[key], bool(masks[bit // 32] & (1 << (bit % 32))))
                      for bit, key in enumerate(self.shapes[shape]))
        return result

    def model(self, provider, identifier):
        if provider not in self.providers:
            return None
        common, rows = self.providers[provider]
        if identifier not in rows:
            return None
        encoded, *delta = rows[identifier]
        result = {"id": identifier, **self.value(common), **self.value(encoded)}
        if delta:
            removed, changed = delta[0]
            compat = self.value(self.compat[result["api"]]).copy()
            for key in removed:
                compat.pop(self.keys[key], None)
            compat.update(self.value(changed))
            result["compat"] = compat
        return result


@cache
def bundled():
    return Catalog(load_snapshot("models.json"))


@lru_cache(maxsize=8192)
def lookup(provider, identifier):
    return bundled().model(provider, identifier)


@cache
def rules():
    return load_snapshot("rules.json")


def provider_info(provider):
    return rules()["providers"].get(provider, {})


@cache
def local_key_placeholders():
    return frozenset(entry["login"]["emptyFallback"] for entry in rules()["auth"]["providers"]
                     if (entry.get("login") or {}).get("emptyFallback"))


def select(selector):
    provider, slash, model = selector.partition("/")
    return lookup(provider, model) if slash and model else None


def split_upstream(selector):
    base, at, tail = selector.rpartition("@")
    slug, _, suffix = tail.partition(":")
    if at and base and UPSTREAM_SLUG.fullmatch(slug):
        return base + (":" + suffix if ":" in tail else ""), slug
    return None


def gateway_key(model):
    if not model:
        return None
    provider = model.get("provider")
    host = urlsplit(model.get("baseUrl") or "").hostname or ""
    if provider == "vercel-ai-gateway" or host == "ai-gateway.vercel.sh" or host.endswith(".ai-gateway.vercel.sh"):
        return "vercelGatewayRouting"
    if provider == "openrouter" or host == "openrouter.ai" or host.endswith(".openrouter.ai"):
        return "openRouterRouting"
    return None


def with_upstream(model, slug):
    if key := gateway_key(model):
        return {**model, "compat": {**(model.get("compat") or {}), key: {"only": [slug]}}}
    return None


def select_routed(registry, selector):
    if row := registry.select(selector):
        return row
    if (pin := split_upstream(selector)) and (base := registry.select(pin[0])):
        return with_upstream(base, pin[1])
    return None


def upstream_pin(model):
    if key := gateway_key(model):
        routing = (model.get("compat") or {}).get(key)
        if isinstance(routing, dict):
            only = routing.get("only")
            if isinstance(only, list) and len(only) == 1 and isinstance(only[0], str) and only[0]:
                return only[0]
    return None


def model_identity(model):
    base = f"{model['provider']}/{model['id']}"
    slug = upstream_pin(model)
    return f"{base}@{slug}" if slug else base


def gateway_body(model, *, responses=False, cache_retention="none"):
    key = gateway_key(model)
    routing = (model.get("compat") or {}).get(key) if key else None
    if not isinstance(routing, dict):
        return {}
    if key == "openRouterRouting":
        return {"provider": routing}
    fields = ("only", "order", "caching") if not responses and cache_retention != "none" else ("only", "order")
    gateway = {name: routing[name] for name in fields if name in routing}
    body = {"providerOptions": {"gateway": gateway}} if gateway else {}
    if responses and cache_retention != "none" and routing.get("caching") == "auto":
        body["caching"] = "auto"
        if "cacheAnchorItems" in routing:
            body["cache_anchor_items"] = routing["cacheAnchorItems"]
        ttl = routing.get("cacheTtl")
        if ttl in ("5m", "1h") and (ttl != "1h" or cache_retention == "1h"):
            body["cache_ttl"] = ttl
    return body


def list_models(pattern=""):
    pattern = pattern.casefold()
    catalog = bundled()
    for provider, (_, rows) in catalog.providers.items():
        for identifier in rows:
            qualified = f"{provider}/{identifier}".casefold()
            if not pattern or pattern in qualified or fnmatchcase(qualified, pattern):
                yield lookup(provider, identifier)


MODEL_FAMILY = re.compile(r"^(claude|gemini|gpt|grok|glm|qwen|deepseek|kimi|mimo|doubao|ernie|gemma|minimax|step|command|jamba|llama|o[1345])")
LEADING_AFFIX = re.compile(r"^(?:\s*[\[【][^\]】]+[\]】]\s*)+")
TRAILING_AFFIX = re.compile(r"(?:\s*[\[【][^\]】]+[\]】]\s*)+$")


def bracket_affix_candidates(identifier):
    normalized = " ".join(identifier.split())
    lead = LEADING_AFFIX.sub("", normalized).strip()
    tail = TRAILING_AFFIX.sub("", normalized).strip()
    both = TRAILING_AFFIX.sub("", lead).strip()
    return tuple(dict.fromkeys(value for value in (both, lead, tail) if value and value != normalized))


def model_segments(identifier):
    return sorted({part for part in re.findall(r"[a-z0-9.:-]+", identifier.lower())
                   if MODEL_FAMILY.match(part) and any(char.isdigit() for char in part)}, key=lambda part: (-len(part), part))


@cache
def reference_indexes():
    exact, aliases = defaultdict(list), defaultdict(list)
    for provider, (_, rows) in bundled().providers.items():
        for identifier in rows:
            exact[identifier.strip().lower()].append((provider, identifier))
            if "/" in identifier and (segments := model_segments(identifier.rsplit("/", 1)[1])):
                aliases[segments[0]].append((provider, identifier))
    return exact, aliases


@cache
def reference_suffixes():
    taxonomy = rules()["taxonomy"]
    discovery, collapse = taxonomy["discovery"], taxonomy["collapse"]
    markers = (*discovery["trailingMarkers"], *discovery["referenceOnlyTrailingMarkers"])
    suffixes = {*discovery["billingVariantSuffixes"], *(f"{prefix}{marker}" for prefix in ("-", ":") for marker in markers),
                *(row["suffix"] for row in (*collapse["suffixes"], *collapse["routingVariants"]))}
    return sorted(suffixes, key=lambda suffix: -len(suffix))


def reference_quality(model):
    cost = model.get("cost") or {}
    return (model.get("contextWindow") or 0, model.get("maxTokens") or 0,
            bool(cost.get("cacheRead") or cost.get("cacheWrite")), model["provider"] == "openai")


def reference_candidates(identifier):
    pending, seen = deque([identifier]), set()
    while pending:
        candidate = pending.popleft().strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        yield candidate.lower()
        pending.extend(bracket_affix_candidates(candidate))
        pending.extend(model_segments(candidate))
        lowered = candidate.lower()
        if lowered.endswith((":cloud", "-cloud")):
            pending.append(candidate[:-6])
        pending.extend((candidate.rsplit("/", 1)[-1], candidate.replace(":", "-"), lowered))
        suffix = next((suffix for suffix in reference_suffixes() if lowered.endswith(suffix)), None)
        if suffix:
            pending.append(candidate[:-len(suffix)])


@lru_cache(maxsize=8192)
def reference(identifier):
    if identifier.startswith("@"):
        return {}
    for candidate in reference_candidates(identifier):
        for index in reference_indexes():
            models = (lookup(provider, key) for provider, key in index.get(candidate, ()))
            best = max((row for row in models if row["provider"] != "xai-oauth"
                        or any((row.get("cost") or {}).get(leg) for leg in ("input", "output", "cacheRead", "cacheWrite"))),
                       key=reference_quality, default=None)
            if best:
                return best
    return {}


def revision(value):
    parts = re.split(r"[.-]", value) if isinstance(value, str) else []
    if 1 <= len(parts) <= 3 and all(re.fullmatch(r"[0-9]+", part) and int(part) <= 255 for part in parts):
        return tuple(map(int, parts)) + (0,) * (3 - len(parts))
    return None


@cache
def glob(pattern):
    return re.compile(re.escape(pattern).replace(r"\*", ".*"))


def model_selector(selector, model, lower, tokens):
    kind, value = selector["kind"], selector["value"]
    if kind == "exact":
        return 2 if value == model else 0
    if kind == "glob":
        return 1 if glob(value).fullmatch(lower) else 0
    return 1 if kind == "token" and value in tokens else 0


@lru_cache(maxsize=8192)
def catalog_policy(provider, identifier, upstream=None):
    model = lookup(provider, identifier)
    if model is None:
        return {}
    return model_policy(model, upstream)


def model_policy(model, upstream=None):
    provider, identifier = model["provider"], model["id"]
    identity = model.get("identity") or {}
    name = model["id"]
    if identity.get("class") == "unknown" and (wire_id := model.get("requestModelId")):
        if (wire := lookup(provider, wire_id)) and (wire.get("identity") or {}).get("class") not in (None, "unknown"):
            name, identity = wire_id, wire["identity"]
    dimensions = {"class": identity.get("class"), "providers": model.get("providerType") or provider,
                  "apis": model["api"], "upstreams": upstream, "family": identity.get("family")}
    current_revision = revision(identity.get("revision"))
    lower = name.lower()
    tokens = set(re.split(r"[^a-z0-9]+", lower))
    winners = {}
    for rule in rules()["cascade"]["rules"]:
        if not (axes := rule.get("catalog")):
            continue
        if any(key in rule and (
            value not in rule[key] if key in ("providers", "apis", "upstreams") else value != rule[key]
        ) for key, value in dimensions.items()):
            continue
        if "revision" in rule and (current_revision is None or not all(
            OPS[term["op"]](current_revision, revision(term["revision"])) for term in rule["revision"]
        )):
            continue
        exactness = 0
        if "models" in rule:
            exactness = max((model_selector(item, name, lower, tokens) for item in rule["models"]), default=0)
            if exactness == 0:
                continue
        rank = (exactness, sum(key in rule for key in (*dimensions, "revision", "models")), rule.get("priority", 0))
        for axis, value in axes.items():
            held = winners.get(axis)
            if held and held[0] == rank:
                raise ValueError(f"Ambiguous {provider}/{identifier} catalog axis {axis}: {held[2]} and {rule['source']}.")
            if held is None or held[0] < rank:
                winners[axis] = rank, value, rule["source"]
    return {axis: entry[1] for axis, entry in winners.items()}


def thinking_effort(model, requested):
    if model is None:
        return requested
    thinking = model.get("thinking") or {}
    if not model.get("reasoning") or thinking.get("unverified"):
        return None
    levels = thinking.get("efforts") or ()
    if not levels:
        return None
    if requested in (None, "off", "none"):
        if thinking.get("requiresEffort"):
            routing, default_wire = thinking.get("effortRouting") or {}, model.get("requestModelId")
            if default_wire:
                for effort in EFFORTS:
                    if effort in levels and routing.get(effort) == default_wire:
                        return effort
            return next((effort for effort in EFFORTS if effort in levels), levels[0])
        return requested
    if requested in levels or requested == "auto":
        return requested
    mapping = thinking.get("effortMap") or {}
    alias = mapping.get(requested)
    if alias in levels and mapping.get(alias, alias) == alias:
        return alias
    if requested not in EFFORTS:
        return None
    lower = [effort for effort in EFFORTS[:EFFORTS.index(requested) + 1] if effort in levels]
    return lower[-1] if lower else levels[0]


def wire_model(model, effort):
    routing = (model.get("thinking") or {}).get("effortRouting") or {}
    return routing.get("off" if effort in (None, "none") else effort) or model.get("requestModelId") or model["id"]


def chat_options(model, options, max_tokens, cache_retention="none"):
    if not model:
        return options
    compat = model.get("compat") or {}
    thinking = model.get("thinking") or {}
    result = dict(options)
    effort = result.pop("reasoning_effort", None)
    wire_effort = (thinking.get("effortMap") or {}).get(effort, effort) if effort in EFFORTS else None
    body = dict(result.get("extra_body") or {})
    if compat.get("thinkingFormat") in ("qwen", "qwen-chat-template"):
        top_level = compat["thinkingFormat"] == "qwen"
        kwargs = dict(body.get("chat_template_kwargs") or {})
        if compat.get("qwenPreserveThinking"):
            kwargs["preserve_thinking"] = True
            if top_level:
                body["preserve_thinking"] = True
        if compat.get("reasoningDisableMode") == "none-effort":
            if effort in ("off", "none"):
                result["reasoning_effort"] = "none"
            elif compat.get("supportsReasoningEffort") and wire_effort is not None:
                result["reasoning_effort"] = wire_effort
        elif model.get("reasoning") and effort is not None:
            (body if top_level else kwargs)["enable_thinking"] = effort not in ("off", "none")
            if compat.get("qwenTemplateReasoningEffort") and wire_effort is not None:
                kwargs["reasoning_effort"] = wire_effort
                if top_level:
                    result["reasoning_effort"] = wire_effort
        if kwargs:
            body["chat_template_kwargs"] = kwargs
    elif compat.get("supportsReasoningEffort") and wire_effort is not None:
        result["reasoning_effort"] = wire_effort
    elif compat.get("reasoningDisableMode") == "none-effort" and effort in ("off", "none"):
        result["reasoning_effort"] = "none"
    field = compat.get("maxTokensField", "max_completion_tokens")
    if max_tokens is not None and not model.get("omitMaxOutputTokens") and field in ("max_tokens", "max_completion_tokens"):
        result[field] = max_tokens
    routing = gateway_body(model, cache_retention=cache_retention)
    extra = {**body, **routing, **(compat.get("extraBody") or {})}
    if upstream_pin(model):
        extra.update(routing)
    if extra:
        result["extra_body"] = extra
    return result


def token_prices(model, prompt_tokens=0, timestamp=None):
    if model is None or not isinstance(cost := model.get("cost"), dict):
        return None
    schedule = cost.get("timeBased")
    rates, multiplier = cost, 1
    if schedule:
        timestamp = int(time.time() * 1000) if timestamp is None else timestamp
        due = (item for item in schedule.get("effectiveRates", ()) if item["effectiveFrom"] <= timestamp)
        rates = max(due, key=lambda item: item["effectiveFrom"], default=cost)
        day, ms = divmod(timestamp, 86_400_000)
        minute, weekday = ms // 60_000, (day + 4) % 7
        if not any(window["startMinute"] <= minute < window["endMinute"] and weekday in window["weekdays"]
                   for window in schedule.get("peakWindows", ())):
            multiplier = schedule["offPeakMultiplier"]
    long = rates.get("longContext")
    if long and (prompt_tokens > long["inputThreshold"] or
                 long.get("inputThresholdInclusive") and prompt_tokens == long["inputThreshold"]):
        rates = long
    return Prices(*(rates[key] * multiplier for key in ("input", "output", "cacheRead", "cacheWrite")),
                  write_1h=2 * rates["input"] * multiplier)
