"""Read Oh My Pi model definitions without resolving request credentials."""

import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from pathlib import Path
from threading import Lock

import yaml

import model_catalog
from auth_broker_cache import config_root
from model_discovery import (
    DISCOVERY_TYPES, DEFAULT_OUTPUT, LAZY_METADATA_TYPES, Discovery,
    implicit_llama, implicit_lm_studio, implicit_ollama, token_limit,
)


MODEL_FIELDS = {
    "name", "api", "kind", "reasoning", "thinking", "input", "imageInputDecoder", "imageCountBudget",
    "tokenizer", "supportsTools", "promptCache", "contextWindow", "maxContextWindow", "maxTokens",
    "omitMaxOutputTokens", "preferWebsockets", "contextPromotionTarget", "compactionModel", "premiumMultiplier",
}
COST = dict.fromkeys(("input", "output", "cacheRead", "cacheWrite"), 0)


def mapping(value, name):
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a mapping.")
    return value


def headers(value, name):
    values = mapping(value, name)
    if not all(key and isinstance(raw, str) for key, raw in values.items()):
        raise ValueError(f"{name} must contain string header values.")
    return tuple(values.items())


def merge(base, patch):
    result = dict(base or {})
    for key, value in patch.items():
        previous = result.get(key)
        result[key] = merge(previous, value) if isinstance(value, dict) and isinstance(previous, dict) else value
    return result


def patch_row(row, patch):
    row.update((key, patch[key]) for key in MODEL_FIELDS & patch.keys())
    if "reasoning" in patch and (row.get("thinking") or {}).get("unverified") and "thinking" not in patch:
        row.pop("thinking", None)
    for key in ("compat", "remoteCompaction"):
        if key in patch:
            row[key] = merge(row.get(key), patch[key])
    if "cost" in patch:
        cost = {**COST, **(row.get("cost") or {}), **patch["cost"]}
        cost.pop("timeBased", None)
        row["cost"] = cost


def validate_model_settings(patch, name):
    mapping(patch, name)
    headers(patch.get("headers", {}), f"{name} headers")
    for key in ("cost", "compat", "remoteCompaction", "thinking"):
        if key in patch:
            mapping(patch[key], f"{name} {key}")
    for key in ("reasoning", "supportsTools", "omitMaxOutputTokens", "preferWebsockets"):
        if key in patch and type(patch[key]) is not bool:
            raise ValueError(f"{name} has invalid {key}.")
    for key in ("contextWindow", "maxContextWindow", "maxTokens", "imageCountBudget"):
        if key in patch and (type(patch[key]) is not int or patch[key] < 1):
            raise ValueError(f"{name} has invalid {key}.")
    for key in ("name", "api", "baseUrl", "compactionModel", "contextPromotionTarget"):
        if key in patch and (not isinstance(patch[key], str) or not patch[key].strip()):
            raise ValueError(f"{name} has invalid {key}.")
    remote = patch.get("remoteCompaction", {})
    for key in ("enabled", "v2StreamingEnabled"):
        if key in remote and type(remote[key]) is not bool:
            raise ValueError(f"{name} has invalid remoteCompaction {key}.")
    for key in ("api", "endpoint", "model", "v2Endpoint", "streamingEndpoint"):
        if key in remote and (not isinstance(remote[key], str) or not remote[key].strip()):
            raise ValueError(f"{name} has invalid remoteCompaction {key}.")


def model_target(registry, provider, target):
    if not isinstance(target, str) or not (target := target.strip()):
        return None
    if model_catalog.select_routed(registry, target):
        return target
    base, _, effort = target.rpartition(":")
    if effort.lower() in (*model_catalog.EFFORTS, "auto", "off", "none") and model_catalog.select_routed(registry, base):
        return base
    qualified = f"{provider}/{target}"
    return qualified if model_catalog.select_routed(registry, qualified) else None


@dataclass(frozen=True)
class Connection:
    mode: str | None = None
    key: str | None = field(default=None, repr=False)
    auth_header: bool = False
    header_layers: tuple = field(default=(), repr=False)


class Provider:
    def __init__(self, name, config):
        self.name = name
        self.config = mapping(config, f"Provider {name!r}")
        self.models = {}
        model_defs = config.get("models", [])
        if not isinstance(model_defs, list):
            raise ValueError(f"Provider {name!r} models must be a list.")
        self.key = config.get("apiKey")
        self.mode = config.get("auth")
        self.discovery = mapping(config.get("discovery", {}), f"Provider {name!r} discovery")
        self.header_layer = headers(config.get("headers", {}), f"Provider {name!r} headers")
        if self.mode not in (None, "apiKey", "oauth", "none"):
            raise ValueError(f"Provider {name!r} has invalid auth.")
        for key in ("apiKey", "baseUrl", "api", "transport"):
            if key in config and (not isinstance(config[key], str) or not config[key].strip()):
                raise ValueError(f"Provider {name!r} has invalid {key}.")
        if "authHeader" in config and type(config["authHeader"]) is not bool:
            raise ValueError(f"Provider {name!r} authHeader must be boolean.")
        if self.discovery.get("type") in DISCOVERY_TYPES:
            if "api" not in config and self.discovery.get("type") != "proxy":
                raise ValueError(f"Provider {name!r} discovery requires an API.")
            timeout = self.discovery.get("timeoutMs", 1)
            if type(timeout) not in (int, float) or not 0 < timeout < math.inf:
                raise ValueError(f"Provider {name!r} discovery requires a positive timeoutMs.")
            for key in ("injectV1", "advertisedReasoning"):
                if key in self.discovery and type(self.discovery[key]) is not bool:
                    raise ValueError(f"Provider {name!r} discovery requires boolean {key}.")
            if "injectV1" in self.discovery and self.discovery["type"] != "openai-models-list":
                raise ValueError(f"Provider {name!r} injectV1 requires openai-models-list discovery.")
        if model_defs and (not config.get("baseUrl") or not self.key and self.mode not in ("oauth", "none")):
            raise ValueError(f"Provider {name!r} models require baseUrl and an auth configuration.")
        for model in model_defs:
            model = mapping(model, f"Provider {name!r} model")
            identifier = model.get("id")
            api = model.get("api", config.get("api"))
            if not isinstance(identifier, str) or not identifier.strip() or not isinstance(api, str) or not api:
                raise ValueError(f"Provider {name!r} models require a model id and API.")
            self.models[identifier] = model
        self.overrides = mapping(config.get("modelOverrides", {}), f"Provider {name!r} modelOverrides")
        for patch in (config, *self.models.values(), *self.overrides.values()):
            validate_model_settings(patch, f"Provider {name!r} model configuration")
        inherited_apis = {model.get("api", config.get("api")) for model in self.models.values() if "baseUrl" not in model}
        self.url_apis = inherited_apis or ({config["api"]} if not self.models and config.get("api") else set())

    def materialize(self, identifier, bundled, discovered=None):
        model = self.models.get(identifier)
        if not model and not bundled and not discovered:
            return None
        config = self.config
        override = self.overrides.get(identifier, {})
        row = dict(discovered or bundled or {})
        if discovered and self.name == "ollama" and config.get("api") == "openai-responses":
            if context := token_limit(os.getenv("OLLAMA_CONTEXT_LENGTH")):
                row.update(contextWindow=context, maxTokens=min(context, DEFAULT_OUTPUT))
        if not row:
            ref = model_catalog.reference(identifier)
            row.update(name=identifier, contextWindow=128000, maxTokens=16384, input=["text"], reasoning=False, cost=COST)
            for key in ("contextWindow", "maxTokens", "input", "reasoning", "cost", "supportsTools", "omitMaxOutputTokens"):
                if key in ref and ref[key] is not None:
                    row[key] = ref[key]
            if ref.get("provider") == self.name and ref.get("thinking"):
                row["thinking"] = ref["thinking"]
        endpoint = config.get("baseUrl")
        if endpoint and not discovered and (
            not self.url_apis or row.get("api") in self.url_apis or config.get("transport") == "pi-native"
        ):
            row["baseUrl"] = endpoint
        patch_row(row, {key: config[key] for key in ("compat", "remoteCompaction") if key in config})
        if model:
            api = model.get("api", config.get("api"))
            if api != row.get("api"):
                row.pop("compat", None)
                row.pop("requestModelId", None)
            patch_row(row, {key: config[key] for key in ("compat", "remoteCompaction") if key in config})
            row.update(api=api, baseUrl=model.get("baseUrl", endpoint))
            patch_row(row, model)
        if "api" in override and override["api"] != row.get("api"):
            row.pop("compat", None)
            row.pop("requestModelId", None)
        patch_row(row, override)
        if config.get("disableStrictTools"):
            row["compat"] = {"disableStrictTools": True, **(row.get("compat") or {})}
        compat = row.get("compat") or {}
        if row["api"] == "openai-responses" and compat.get("supportsReasoningEffort") is False:
            row.pop("thinking", None)
        elif row.get("reasoning") and not row.get("thinking") and not compat.get("trustExplicitThinkingOnly"):
            limit = 5 if row["api"] in ("anthropic-messages", "openai-responses") or (
                compat.get("thinkingFormat", "openai") == "openai" and compat.get("supportsReasoningEffort")
            ) else 4
            row["thinking"] = {"efforts": model_catalog.EFFORTS[:limit]}
        if row.get("thinking") and isinstance(compat.get("reasoningEffortMap"), dict):
            row["thinking"] = {**row["thinking"], "effortMap": merge(
                row["thinking"].get("effortMap"), compat["reasoningEffortMap"],
            )}
        if "contextWindow" in override and "maxContextWindow" not in override:
            row.pop("maxContextWindow", None)
        row.update(provider=self.name, id=identifier)
        if discovered:
            row["_discovered"] = True
        if self.discovery.get("type") in LAZY_METADATA_TYPES:
            row["_lazy_metadata"] = True
            row["_metadata_pins"] = tuple(key for key in ("contextWindow", "maxTokens", "input")
                                          if key in (model or {}) or key in override)
        if config.get("transport"):
            row["_transport"] = config["transport"]
        mode = self.mode
        if mode is None and model and row["api"] == "anthropic-messages":
            mode = "oauth"
        layers = self.header_layer, headers((model or {}).get("headers", {}), "Model headers"), headers(
            override.get("headers", {}), "Model override headers",
        )
        if mode or self.key or any(layers) or discovered:
            row["_connection"] = Connection(mode, self.key, config.get("authHeader", False), layers)
        return row


class ConfiguredModels:
    def __init__(self, providers):
        self.providers = {name: Provider(name, config) for name, config in providers.items()}
        self.rows = {}
        self.config_mtime = 0
        self.discovery = None
        self.discovery_locks = {}
        self.discovered = {}
        self.optional = set()

    def enable_discovery(self, path, secrets, *, refresh=False, disabled=()):
        for name, implicit in (("ollama", implicit_ollama), ("llama.cpp", implicit_llama), ("lm-studio", implicit_lm_studio)):
            if name not in self.providers and name not in disabled:
                config, optional = implicit()
                self.providers[name] = Provider(name, config)
                if optional:
                    self.optional.add(name)
        self.discovery = Discovery(path, secrets, self.config_mtime, refresh=refresh)
        self.discovery_locks = {name: Lock() for name, provider in self.providers.items()
                                if name not in disabled and provider.discovery.get("type") in DISCOVERY_TYPES}
        return self

    def discovered_rows(self, name):
        if name not in self.discovery_locks:
            return {}
        with self.discovery_locks[name]:
            if name not in self.discovered:
                self.discovered[name] = self.discovery.models(self.providers[name], model_catalog.reference,
                                                            optional=name in self.optional)
            return self.discovered[name]

    def runtime_metadata(self, route, client):
        model = route.catalog_model or {}
        provider = self.providers.get(model.get("provider"))
        if (self.discovery and provider and provider.name in self.discovery_locks and model.get("_lazy_metadata")
            and model.get("id") == route.model and (model.get("baseUrl") or "").rstrip("/") == route.endpoint()):
            return self.discovery.runtime(client, route, provider)
        return {}

    @classmethod
    def load(cls, path, profile):
        explicit = path is not None
        if path is None:
            root = config_root(profile) / "agent"
            if profile == "default" and (override := os.getenv("PI_CODING_AGENT_DIR")):
                root = Path(override).expanduser()
            path = next((candidate for name in ("models.yml", "models.yaml", "models.json")
                         if (candidate := root / name).is_file()), None)
        if path is None:
            return cls({})
        try:
            source = path.read_text(encoding="utf-8-sig")
            try:
                data = json.loads(source) if path.suffix.lower() == ".json" else yaml.safe_load(source)
            except (ValueError, yaml.YAMLError):
                raise ValueError(f"Invalid model configuration in {path}.") from None
            data = {} if data is None else mapping(data, "Model configuration")
            catalog = cls(mapping(data.get("providers", {}), "Model providers"))
            catalog.config_mtime = path.stat().st_mtime
            return catalog
        except (OSError, ValueError) as error:
            if explicit:
                raise
            print(f"[ignoring model configuration in {path}: {error}]", file=sys.stderr)
            return cls({})

    def configured_selectors(self):
        return (f"{name}/{identifier}" for name, provider in self.providers.items() for identifier in provider.models)

    def discovered_selectors(self, pattern=""):
        prefix, slash, _ = pattern.partition("/")
        matched = next((name for name in self.providers if name.casefold() == prefix), None)
        scoped = slash and (matched or self.has_provider(prefix))
        names = [name for name in self.discovery_locks if not scoped or name == matched]
        if len(names) == 1:
            yield from (f"{names[0]}/{identifier}" for identifier in self.discovered_rows(names[0]))
        elif names:
            with ThreadPoolExecutor(max_workers=min(4, len(names))) as pool:
                for name, rows in zip(names, pool.map(self.discovered_rows, names)):
                    yield from (f"{name}/{identifier}" for identifier in rows)

    def has_provider(self, provider):
        return provider in self.providers or provider in model_catalog.bundled().providers

    def select(self, selector):
        provider, slash, identifier = selector.partition("/")
        if not slash or not identifier:
            return None
        if provider not in self.providers:
            return model_catalog.select(selector)
        if selector not in self.rows:
            discovered = self.discovered_rows(provider).get(identifier)
            self.rows[selector] = self.providers[provider].materialize(identifier, model_catalog.select(selector), discovered)
        return self.rows[selector]

    def list_models(self, pattern=""):
        selectors = (f"{row['provider']}/{row['id']}" for row in model_catalog.list_models(pattern))
        folded = pattern.casefold()
        for selector in dict.fromkeys((*selectors, *self.configured_selectors(), *self.discovered_selectors(folded))):
            qualified = selector.casefold()
            if folded in qualified or fnmatchcase(qualified, folded):
                yield self.select(selector)
