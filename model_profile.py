"""Oh My Pi model roles and profile retry chains."""

import os
import re
import sys
from dataclasses import replace
from fnmatch import fnmatchcase

import model_catalog
from models_config import model_target
from auth_broker import dotted
from routing import load_routes


KIND_ROLES = {"image", "web", "speech", "dictation", "judge"}
ROLE_FALLBACKS = {"tiny": "smol", "memory": "tiny", "widget": "smol", "advisor": "slow"}
ROLE_PRIORITIES = {"tiny": "smol", "memory": "smol", "widget": "smol", "advisor": "slow"}
EFFORT_NAMES = {level: level for level in (*model_catalog.EFFORTS, "auto", "off", "inherit")}
EFFORT_NAMES.update(min="minimal", med="medium", hi="high", xhi="xhigh", none="off")
PROVIDER_ORDER = (
    "openai-codex", "anthropic", "openai", "google-gemini-cli", "google", "google-vertex", "kimi-code",
    "moonshot", "qwen-portal", "alibaba-token-plan", "zai", "xai-oauth", "xai", "mistral", "deepseek", "groq",
    "fireworks", "cerebras", "baseten", "deepinfra", "openrouter", "aimlapi", "together", "alibaba-coding-plan",
    "umans", "google-antigravity", "opencode-zen", "gitlab-duo", "opencode-go", "kilo", "vercel-ai-gateway",
    "cloudflare-ai-gateway", "nanogpt", "github-copilot",
)


def patterns(value):
    items = value if isinstance(value, list) else [value]
    return [word.strip() for item in items if isinstance(item, str) for word in item.split(",") if word.strip()]


def matches(pattern, selector):
    return fnmatchcase(selector.casefold(), pattern.casefold())


def route_base(route):
    if model := route.catalog_model:
        return model_catalog.model_identity(model)
    if route.broker_provider:
        return f"{route.broker_provider}/{route.model}"
    return route.profile_base or route.name


def alias(name):
    if name == "*":
        return "default"
    if name.startswith("@"):
        return name[1:]
    if name.startswith("pi/"):
        return name[3:]
    return None


class ModelProfile:
    def __init__(self, settings, routes, route_options):
        self.routes = routes
        self.options = route_options
        self.catalog = route_options.get("configured_models")
        self.roles = self.mapping(settings, "modelRoles")
        raw = self.mapping(settings, "retry.fallbackChains")
        self.chains = {}
        for key, entries in raw.items():
            if isinstance(entries, list) and all(isinstance(entry, str) for entry in entries):
                self.chains[key] = tuple(entry.strip() for entry in entries if entry.strip())
            else:
                print(f"[retry.fallbackChains {key!r} must be a list of selectors]", file=sys.stderr)
        for role in self.roles:
            if "default" in self.chains and role not in KIND_ROLES:
                self.chains.setdefault(role, self.chains["default"])
        effort = dotted(settings, "defaultThinkingLevel")
        self.default_effort = EFFORT_NAMES.get(effort, "high") if isinstance(effort, str) else "high"
        self.revert = dotted(settings, "retry.fallbackRevertPolicy") != "never"
        configured = (provider.lower() for provider in patterns(dotted(settings, "modelProviderOrder")))
        self.provider_order = dict.fromkeys((*configured, *PROVIDER_ORDER))
        self.provider_rank = {provider: rank for rank, provider in enumerate(self.provider_order)}
        self.known = tuple(routes)
        self.role_routes = {}
        self.judge_routes = {}

    @staticmethod
    def mapping(settings, key):
        current = dotted(settings, key)
        result = {name: value for name, value in current.items() if isinstance(name, str)} if isinstance(current, dict) else {}
        prefix = key + "."
        result.update((name[len(prefix):], value) for name, value in settings.items()
                      if isinstance(name, str) and name.startswith(prefix) and name[len(prefix):] not in result)
        return result

    def split(self, selector):
        if selector in self.known or (self.catalog or model_catalog).select(selector):
            return selector, None
        base, colon, level = selector.rpartition(":")
        effort = EFFORT_NAMES.get(level.lower()) if colon else None
        return (base, effort) if effort is not None else (selector, None)

    def expand(self, value, visited=()):
        for selector in patterns(value):
            base, effort = self.split(selector)
            role = alias(base)
            if role is None:
                yield base, effort
                continue
            if role in visited:
                continue
            configured = self.roles.get(role)
            if not configured and role in ROLE_FALLBACKS:
                other = ROLE_FALLBACKS[role]
                if role != "advisor" or self.roles.get(other):
                    configured = f"@{other}"
            if not configured and role in ("smol", "slow") and self.roles.get("default"):
                configured = "@default"
            defaults = model_catalog.rules()["behavior"].get("roleDefaults", {}).get(
                ROLE_PRIORITIES.get(role, role), (),
            )
            expanded = tuple(self.expand(configured or list(defaults), (*visited, role)))
            if not expanded and configured and defaults:
                expanded = tuple(self.expand(list(defaults), (*visited, role)))
            for pattern, inner in expanded:
                yield pattern, effort if effort is not None else inner

    def exact(self, name):
        route = self.routes.get(name)
        if route:
            return route
        for candidate in tuple(self.routes.values()):
            if route_base(candidate) == name and candidate.name in self.known:
                return candidate
        if not model_catalog.select_routed(self.catalog or model_catalog, name):
            return None
        try:
            for key, candidate in load_routes(None, name, None, **self.options).items():
                self.routes.setdefault(key, candidate)
        except ValueError:
            return None
        return self.routes.get(name)

    def exact_selection(self, selector):
        base, effort = self.split(selector)
        route = self.exact(base)
        if route is None and (pin := model_catalog.split_upstream(selector)):
            model, level = self.split(pin[0])
            if level is not None:
                base, effort = f"{model}@{pin[1]}", level
                route = self.exact(base)
        return route, base, effort

    def matching(self, selector, *, partial=False):
        if route := self.exact(selector):
            yield route
            return
        registry = self.catalog or model_catalog
        provider, slash, _ = selector.partition("/")
        wildcard = any(char in selector for char in "*?[")
        if slash and not wildcard:
            known_provider = self.catalog.has_provider(provider) if self.catalog else provider in model_catalog.bundled().providers
            if known_provider:
                return
        matched_local = False
        for name in self.known:
            route = self.routes[name]
            if matches(selector, route.model) or matches(selector, route_base(route)):
                yield route
                matched_local = True
        if matched_local and not wildcard:
            return
        choices = [row for row in registry.list_models(selector) if row and
                   (matches(selector, f"{row['provider']}/{row['id']}") or matches(selector, row["id"]))]
        if not choices and partial and not wildcard:
            needle = selector.casefold()
            choices = [row for row in registry.list_models("") if row
                       and (needle in row["id"].casefold() or needle in (row.get("name") or "").casefold())]
            aliases = [row for row in choices if not re.search(r"-\d{8}$", row["id"])]
            if aliases:
                choices = aliases
            elif choices:
                newest = max(row["id"] for row in choices)
                choices = [row for row in choices if row["id"] == newest]
        choices.sort(key=lambda row: self.provider_rank.get(row["provider"], len(self.provider_rank)))
        for row in choices:
            if route := self.exact(f"{row['provider']}/{row['id']}"):
                yield route

    def pinned(self, route, slug):
        if not (model := model_catalog.with_upstream(route.catalog_model, slug)):
            return None
        name = f"{route.name}@{slug}"
        if name not in self.routes:
            self.routes.setdefault(name, replace(route, name=name, catalog_model=model, profile_base=model_catalog.model_identity(model)))
        return self.routes[name]

    def variant(self, route, effort):
        if effort in (None, "inherit") or route.reasoning_effort == effort:
            return route
        name = f"{route.name}:{effort}"
        if name not in self.routes:
            self.routes[name] = replace(route, name=name, reasoning_effort=effort, profile_base=route_base(route))
        return self.routes[name]

    def choices(self, value):
        found = {}
        for selector, effort in self.expand(value):
            direct = tuple(self.matching(selector))
            if direct:
                for route in direct:
                    found.setdefault(self.variant(route, effort).name, None)
                continue
            if (pin := model_catalog.split_upstream(selector)) and not (self.catalog or model_catalog).select(selector):
                base, inner_effort = self.split(pin[0])
                if (base_route := next(self.matching(base, partial=True), None)) and (route := self.pinned(base_route, pin[1])):
                    found.setdefault(self.variant(route, effort if effort is not None else inner_effort).name, None)
        return tuple(found)

    def primary(self, requested, window):
        first = patterns(requested)
        role = alias(self.split(first[0])[0]) if first else "default"
        if not self.known and len(first) == 1 and role is None and "/" not in first[0] and "@" not in first[0] and not any(
            char in first[0] for char in "*?["
        ):
            selected = tuple(route.name for name in self.catalog.configured_selectors()
                             if name.partition("/")[2] == first[0] and (route := self.exact(name))) if self.catalog else ()
            if not selected and self.catalog and not (os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_KEY")):
                selected = tuple(route.name for name in self.catalog.discovered_selectors(first[0])
                                 if name.partition("/")[2] == first[0] and (route := self.exact(name)))
            if not selected:
                self.routes.update(load_routes(None, first[0], window, **self.options))
                selected = (first[0],)
        else:
            selected = self.choices(requested or "@default")
        if not selected and not requested:
            selected = self.known or self.provider_defaults()
        if not selected and requested and not self.known and len(first) == 1 and role is None:
            # Raw IDs on an OpenAI-compatible endpoint remain a valid CLI entry.
            provider, _, _ = first[0].partition("/")
            known_provider = self.catalog.has_provider(provider) if self.catalog else provider in model_catalog.bundled().providers
            if not known_provider and not any(char in first[0] for char in "*?["):
                self.routes.update(load_routes(None, first[0], window, **self.options))
                selected = (first[0],)
        if not selected:
            raise ValueError(f"No routable model matches {requested or '@default'!r}.")
        if window is not None:
            for name in selected:
                route = self.routes[name]
                self.routes[name] = replace(route, window=window, metadata_pins=route.metadata_pins | {"contextWindow"})
        return selected, role

    def provider_defaults(self):
        builtins = model_catalog.rules()["providers"]
        providers = dict.fromkeys((*self.provider_order, *builtins))
        result = []
        for provider in providers:
            entry = builtins.get(provider, {})
            if entry.get("automaticDefault") is False:
                continue
            if (model := entry.get("defaultModel")) and (route := self.exact(f"{provider}/{model}")):
                result.append(route.name)
        if self.catalog:
            names = dict.fromkeys((*self.catalog.configured_selectors(), *self.catalog.discovered_selectors()))
            result.extend(route.name for name in names
                          if (route := self.exact(name)) and route.name not in result)
        return tuple(result)

    def prepare(self):
        for role in self.roles:
            if role not in KIND_ROLES:
                self.role_routes[role] = self.choices(f"@{role}")
        if "memory" not in self.role_routes and any(role in self.roles for role in ("tiny", "smol")):
            self.role_routes["memory"] = self.choices("@memory")
        for key, entries in self.chains.items():
            if "/" in key and not key.endswith("/*"):
                self.choices(key)
            for entry in entries:
                if entry.endswith("/*"):
                    if self.catalog:
                        self.catalog.discovered_rows(entry.partition("/")[0])
                else:
                    self.choices(entry)

    def judge_candidates(self):
        defaults = model_catalog.rules()["behavior"].get("roleDefaults", {}).get("judge", ())
        primary = self.roles.get("judge") or (defaults[0] if defaults else "@tiny")
        chain = self.chains.get("judge", defaults)
        candidates = {}
        registry = self.catalog or model_catalog
        for selector, explicit in ((primary, "judge" in self.roles),
                                   *((entry, "judge" in self.chains) for entry in chain)):
            for base, effort in self.expand(selector):
                row = model_catalog.select_routed(registry, base)
                if row and row.get("api") in model_catalog.JUDGE_APIS:
                    if base not in self.judge_routes:
                        try:
                            self.judge_routes[base] = load_routes(None, base, None, allow_judge=True, **self.options)[base]
                        except ValueError:
                            continue
                    name = base
                elif names := self.choices(base if effort is None else f"{base}:{effort}"):
                    name = names[0]
                else:
                    continue
                candidates[name] = explicit or candidates.get(name, False)
                break
        return tuple(candidates.items())

    def compaction(self, route):
        if not route.catalog_model:
            return ()
        model = route.catalog_model
        target = model_target(self.catalog or model_catalog, model["provider"], model.get("compactionModel"))
        candidate = self.exact(target) if target else None
        return (candidate.name,) if candidate else ()

    def match(self, route, effort, key):
        base, level = self.split(key)
        if route_base(route) != base and route.name != base:
            return 0
        if effort == level:
            return 4
        if level in (None, "auto", "inherit"):
            return 2
        if model_catalog.thinking_effort(route.catalog_model, level) == effort:
            return 3
        return 1

    def chain_key(self, route, effort, role_hint=None):
        exact = [(self.match(route, effort, key), key) for key in self.chains
                 if "/" in key and not key.endswith("/*")]
        rank, key = max(exact, key=lambda row: row[0], default=(0, None))
        if rank >= 2:
            return key
        base = route_base(route)
        wildcards = [key for key in self.chains if key.endswith("/*") and base.startswith(key[:-1])]
        if wildcards:
            return max(wildcards, key=len)
        if role_hint in self.chains:
            if role_hint not in self.roles or any(route_base(self.routes[name]) == base
                                                 for name in self.role_routes.get(role_hint, ())):
                return role_hint
        roles = []
        for key in self.chains:
            if "/" not in key and key not in KIND_ROLES:
                rank = max((self.match(route, effort, self.label(self.routes[name]))
                            for name in self.role_routes.get(key, ())), default=0)
                if rank:
                    roles.append((rank != 1, key == "default", key))
        if roles:
            return max(roles, key=lambda row: row[:2])[2]
        return "default" if self.chains.get("default") else None

    @staticmethod
    def label(route):
        base = route_base(route)
        return f"{base}:{route.reasoning_effort}" if route.reasoning_effort else base

    def candidates(self, key, route, effort):
        if key not in self.chains:
            return ()
        if key.endswith("/*"):
            primary = (route.name,)
        elif "/" in key:
            primary = self.choices(key)
        else:
            primary = self.role_routes.get(key, ()) or ((route.name,) if key == "default" else ())
        chain = list(primary[:1])
        base = route_base(route)
        model_id = (route.catalog_model or {}).get("id") or base.partition("/")[2] or route.model
        bare = model_id.rsplit("/", 1)[-1]
        for entry in self.chains[key]:
            if entry.endswith("/*"):
                provider, slash, prefix = entry[:-2].partition("/")
                names = self.choices(f"{provider}/{prefix}/{bare}" if slash else f"{provider}/{model_id}")
                if not names and not slash and bare != model_id:
                    names = self.choices(f"{provider}/{bare}")
            else:
                names = self.choices(entry)
            chain.extend(name for name in names if name not in chain)
        exact = [i for i, name in enumerate(chain) if route_base(self.routes[name]) == base
                 and self.routes[name].reasoning_effort == effort]
        matched = exact or [i for i, name in enumerate(chain) if route_base(self.routes[name]) == base]
        return tuple(chain[matched[0] + 1:] if matched else chain)
