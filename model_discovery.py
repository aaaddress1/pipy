"""Live discovery and cached model metadata for configured inference endpoints."""

import ipaddress
import json
import os
import re
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from contextvars import copy_context
from hashlib import sha256

import httpx

import model_catalog


DISCOVERY_TYPES = ("ollama", "llama.cpp", "lm-studio", "openai-models-list", "proxy")
LAZY_METADATA_TYPES = ("llama.cpp", "lm-studio")
LLAMA_DEFAULTS = ("http://127.0.0.1:8080", "http://127.0.0.1:9931")
DEFAULT_CONTEXT = 128_000
DEFAULT_OUTPUT = 32_768
MAX_TOKEN_LIMIT = 2**53 - 1
TTL = 86_400
COST = dict.fromkeys(("input", "output", "cacheRead", "cacheWrite"), 0)


def record(value):
    return value if isinstance(value, dict) else {}


def entries(value):
    return value if isinstance(value, list) else ()


def token_limit(value):
    if type(value) not in (int, float, str):
        return None
    try:
        number = float(value)
    except (ValueError, OverflowError):
        return None
    return int(number) if 0 < number <= MAX_TOKEN_LIMIT and number.is_integer() else None


def first_limit(*values):
    return next((number for value in values if (number := token_limit(value)) is not None), None)


def endpoint(value, default):
    url = httpx.URL(value or default)
    if url.scheme not in ("http", "https") or not url.host or url.userinfo:
        raise ValueError("Discovery requires an HTTP(S) URL without credentials.")
    return url.copy_with(query=None, fragment=None)


def implicit_ollama():
    url = (os.getenv("OLLAMA_BASE_URL") or "").strip()
    host = (os.getenv("OLLAMA_HOST") or "").strip()
    if not url and host:
        if host.startswith("//"):
            host = "http:" + host
        elif host.startswith(":"):
            host = "http://127.0.0.1" + host
        elif "://" not in host:
            host = "http://" + host
        try:
            parsed = endpoint(host, "")
            if parsed.scheme == "http" and parsed.port is None:
                parsed = parsed.copy_with(port=11434)
            url = str(parsed.copy_with(path="/"))
        except (ValueError, httpx.InvalidURL):
            pass
    return {
        "api": "openai-responses", "auth": "none",
        "baseUrl": url or "http://127.0.0.1:11434", "discovery": {"type": "ollama"},
    }, not bool(url)


def implicit_llama():
    url = os.getenv("LLAMA_CPP_BASE_URL")
    return {
        "api": "openai-responses", "auth": "apiKey" if os.getenv("LLAMA_CPP_API_KEY") else "none",
        "baseUrl": url or LLAMA_DEFAULTS[0], "discovery": {"type": "llama.cpp"},
    }, not bool(url)


def implicit_lm_studio():
    url = os.getenv("LM_STUDIO_BASE_URL")
    return {
        "api": "openai-completions", "auth": "none",
        "baseUrl": url or "http://127.0.0.1:1234/v1", "discovery": {"type": "lm-studio"},
    }, not bool(url)


def local_host(host):
    if host in ("localhost", "0.0.0.0"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def ollama_metadata(data):
    parameters = data.get("parameters")
    match = re.search(r"^\s*num_ctx\s+(\d+)\s*$", parameters, re.MULTILINE) if isinstance(parameters, str) else None
    info = record(data.get("model_info"))
    context = first_limit(match[1] if match else None, *(value for key, value in info.items()
                          if key == "context_length" or key.endswith(".context_length")))
    capabilities = data.get("capabilities")
    if isinstance(capabilities, dict):
        capabilities = {name for name, value in capabilities.items() if value is True}
    else:
        capabilities = {value.lower() for value in entries(capabilities) if isinstance(value, str)}
    return {
        "contextWindow": context or DEFAULT_CONTEXT, "maxTokens": min(context or DEFAULT_OUTPUT, DEFAULT_OUTPUT),
        "input": ["text", "image"] if {"vision", "image"} & capabilities else ["text"],
        "reasoning": "thinking" in capabilities, "imageInputDecoder": "stb",
    }


def modalities(*values):
    return {value.strip().lower() for candidate in values for value in entries(candidate)
            if isinstance(value, str) and value.strip()}


def llama_configured_context(status):
    args = entries(status.get("args"))
    for index, arg in enumerate(args):
        if not isinstance(arg, str):
            continue
        flag, equals, value = arg.partition("=")
        if flag in ("--ctx-size", "-c"):
            if context := token_limit(value if equals else args[index + 1] if index + 1 < len(args) else None):
                return context
    preset = status.get("preset")
    match = re.search(r"^\s*ctx-size\s*=\s*(-?\d+)\s*$", preset, re.MULTILINE) if isinstance(preset, str) else None
    return token_limit(match[1]) if match else None


def llama_metadata(item, props, *, default=None):
    meta, status = record(item.get("meta")), record(item.get("status"))
    settings = record(props.get("default_generation_settings"))
    context = first_limit(meta.get("n_ctx"), llama_configured_context(status), settings.get("n_ctx"),
                          props.get("n_ctx"), meta.get("n_ctx_train"), default)
    input_modalities = record(item.get("architecture")).get("input_modalities")
    media = props.get("modalities")
    result = {}
    if isinstance(input_modalities, list):
        result["input"] = ["text", "image"] if "image" in modalities(input_modalities) else ["text"]
    elif isinstance(media, dict):
        result["input"] = ["text", "image"] if media.get("vision") is True else ["text"]
    if context:
        unlimited = any(llama_unlimited(value)
                        for source in (record(settings.get("params")), settings, props)
                        for value in (source.get("max_tokens"), source.get("n_predict")))
        result.update(contextWindow=context, maxTokens=context if unlimited else min(context, DEFAULT_OUTPUT))
    return result


def llama_unlimited(value):
    if type(value) not in (str, int, float):
        return False
    try:
        return float(value) == -1
    except (ValueError, OverflowError):
        return False


def qwen_template_effort(identifier):
    qwen = next((name for name in model_catalog.model_segments(identifier) if name.startswith("qwen")), None)
    if qwen is None:
        return None
    generation = re.match(r"qwen[-_ ]?(\d+(?:\.\d+)*)", qwen)
    return generation is not None and tuple(map(int, generation[1].split("."))) >= (3, 8)


def llama_model(identifier, api, item, props):
    row = {"name": identifier, "api": api, "input": ["text"], "reasoning": False, "imageInputDecoder": "stb", "cost": COST,
           **llama_metadata(item, props, default=DEFAULT_CONTEXT)}
    template_effort = qwen_template_effort(identifier)
    if template_effort is not None:
        row.update(api="openai-completions", reasoning=True, thinking={
            "efforts": ["low", "medium", "xhigh"] if template_effort else ["high"],
            "requiresEffort": template_effort,
        }, compat={
            "supportsStore": False, "supportsDeveloperRole": False, "supportsNamedToolChoice": False,
            "supportsReasoningEffort": template_effort, "thinkingFormat": "qwen",
            "reasoningDisableMode": "qwen-enable-thinking-false",
            "qwenPreserveThinking": True, "qwenTemplateReasoningEffort": template_effort,
        })
    return row


def reasoning_spec(item, metadata, row, reference, discovery):
    if discovery.get("advertisedReasoning") is False:
        return
    def field(name):
        return item.get(name) if metadata.get(name) is None else metadata[name]

    raw = {effort.lower() for entry in entries(field("supported_reasoning_levels"))
           if isinstance(effort := (entry.get("effort") if isinstance(entry, dict) else entry), str)}
    levels = [effort for effort in model_catalog.EFFORTS[1:] if effort in raw]
    compat = row["compat"]
    if field("reasoning") is False:
        row["reasoning"] = False
        row.pop("thinking", None)
        compat["supportsReasoningEffort"] = False
    elif levels:
        default = field("default_reasoning_level")
        default = default.lower() if isinstance(default, str) else None
        inherited = (reference.get("thinking") or {}).get("defaultLevel")
        aliases = {source.lower(): target.lower() for source, target in record(field("reasoning_effort_map")).items()
                   if source.lower() in model_catalog.EFFORTS and isinstance(target, str)
                   and target.lower() in model_catalog.EFFORTS[1:]}
        thinking = {"efforts": levels, "defaultLevel": next((level for level in (default, inherited)
                    if level in levels), levels[0]), "mode": "budget" if row["api"] == "anthropic-messages" else "effort"}
        if aliases:
            thinking["effortMap"] = aliases
        required = field("reasoning_required")
        if type(required) is bool:
            thinking["requiresEffort"] = required
        row.update(reasoning=True, thinking=thinking)
        if row["api"] != "anthropic-messages":
            compat.pop("reasoningEffortMap", None)
            compat.pop("omitReasoningEffort", None)
            compat["supportsReasoningEffort"] = True
            if "none" in raw:
                compat["reasoningDisableMode"] = "none-effort"


def lm_studio_native_rows(v0, v1):
    result = {}
    for value in entries(record(v0).get("data")):
        item = record(value)
        identifier = item.get("id")
        if not isinstance(identifier, str) or not identifier:
            continue
        kind = str(item.get("type", "")).lower()
        decision = kind == "decision"
        vision = item.get("has_vision_adapter") is True if decision else (
            kind == "vlm" or bool({"vision", "image"} & modalities(item.get("capabilities")))
        )
        context = first_limit(item.get("loaded_context_length") if item.get("state") == "loaded" else None,
                              item.get("max_context_length"), item.get("context_length"), item.get("max_model_len"))
        result[identifier] = {"input": ["text", "image"] if vision else ["text"], "contextWindow": context,
                              "decision": decision}
    for value in entries(record(v1).get("models")):
        item = record(value)
        if str(item.get("type", "")).lower() != "decision":
            continue
        instances = [record(instance) for instance in entries(item.get("loaded_instances"))]
        context = first_limit(*(record(instance.get("config")).get("context_length") for instance in instances),
                              item.get("max_context_length"))
        inputs = ["text", "image"] if item.get("has_vision_adapter") is True else ["text"]
        model = {"input": inputs, "contextWindow": context, "decision": True}
        for identifier in (item.get("key"), *(instance.get("id") for instance in instances)):
            if isinstance(identifier, str) and identifier:
                result[identifier] = model
    return result


def openai_model(item, api, reference, discovery, provider, native=None):
    native = native or {}
    metadata, limits = record(item.get("metadata")), record(item.get("limits"))
    top, architecture = record(item.get("top_provider")), record(item.get("architecture"))
    limit_in, limit_out = token_limit(limits.get("max_input_tokens")), token_limit(limits.get("max_output_tokens"))
    context = first_limit(item.get("max_model_len"), item.get("context_length"), metadata.get("context_window"),
                          limit_in + limit_out if limit_in and limit_out else None,
                          native.get("contextWindow"), reference.get("contextWindow"))
    cap = first_limit(limit_out, item.get("max_output_tokens"), item.get("max_completion_tokens"),
                      top.get("max_completion_tokens"), item.get("max_tokens"), reference.get("maxTokens"))
    inputs = modalities(item.get("input"), item.get("input_modalities"), architecture.get("input_modalities"),
                        metadata.get("input_modalities"))
    output = modalities(item.get("output"), item.get("output_modalities"), architecture.get("output_modalities"))
    display = metadata.get("display_name")
    name = (display.strip() if isinstance(display, str) else "") or reference.get("name") or item["id"]
    inputs = native.get("input") or (
        (["text", "image"] if "image" in inputs else ["text"]) if inputs else reference.get("input") or ["text"]
    )
    if native.get("decision") or output in ({"image"}, {"embedding"}, {"embeddings"}):
        kind = "judge" if native.get("decision") else "image" if output == {"image"} else "embedding"
        return {"name": name, "input": inputs, "kind": kind, "reasoning": False, "supportsTools": False,
                "api": {"image": "openai-images", "judge": "openai-decisions", "embedding": "openai-embeddings"}[kind],
                "contextWindow": None if kind == "image" else context, "maxTokens": None}
    reference_compat = reference.get("compat") or {}
    compat = {"supportsStore": False, "supportsDeveloperRole": False,
              "supportsReasoningEffort": reference_compat.get("supportsReasoningEffort", False)}
    compat.update((key, reference_compat[key]) for key in ("reasoningEffortMap", "omitReasoningEffort")
                  if key in reference_compat)
    context = context or DEFAULT_CONTEXT
    row = {"name": name, "api": api, "contextWindow": context,
           "maxTokens": min(context, cap or (8192 if api == "anthropic-messages" else DEFAULT_OUTPUT)),
           "input": inputs, "reasoning": reference.get("reasoning") is True, "compat": compat}
    if reference.get("provider") == provider and reference.get("thinking"):
        row["thinking"] = reference["thinking"]
    if discovery["type"] == "lm-studio":
        row["imageInputDecoder"] = "stb"
        if api == "openai-completions":
            compat["supportsNamedToolChoice"] = False
            template_effort = qwen_template_effort(item["id"])
            if template_effort is not None:
                compat.update(thinkingFormat="qwen", qwenPreserveThinking=True,
                              reasoningDisableMode="qwen-enable-thinking-false")
                if provider == "lm-studio" and template_effort:
                    row.update(reasoning=True, thinking={"efforts": ["low", "medium", "xhigh"], "requiresEffort": True})
                    compat.update(supportsReasoningEffort=True, qwenTemplateReasoningEffort=True)
    reasoning_spec(item, metadata, row, reference, discovery)
    return row


def proxy_model(item, api, reference, discovery, provider):
    types = entries(item.get("supported_endpoint_types"))
    api = "anthropic-messages" if "anthropic" in types else "openai-completions" if "openai" in types else api
    identifier = item["id"]
    label = item.get("name")
    name = label.strip() if isinstance(label, str) else ""
    if not name or name == identifier:
        name = reference.get("name") or next(iter(model_catalog.bracket_affix_candidates(identifier)), identifier)
    row = {
        "name": name, "api": api, "contextWindow": first_limit(item.get("context_length"), reference.get("contextWindow"))
        or DEFAULT_CONTEXT, "maxTokens": reference.get("maxTokens") or (8192 if api == "anthropic-messages" else DEFAULT_OUTPUT),
        "input": reference.get("input") or ["text"], "reasoning": reference.get("reasoning") is True,
        "compat": {"supportsStore": False, "supportsDeveloperRole": False, "supportsReasoningEffort": False}
        if api != "anthropic-messages" else {},
    }
    if reference.get("provider") == provider and reference.get("thinking"):
        row["thinking"] = reference["thinking"]
    reasoning_spec(item, record(item.get("metadata")), row, reference, discovery)
    return row


class Discovery:
    def __init__(self, path, secrets, config_mtime, *, refresh=False):
        self.path, self.secrets = path, secrets
        self.config_mtime = config_mtime
        self.refresh = refresh

    def read(self, key):
        if not self.path.is_file():
            return 0, None
        try:
            with closing(sqlite3.connect(self.path, timeout=1)) as db:
                row = db.execute("SELECT fetched, rows FROM discovery WHERE key = ?", (key,)).fetchone()
            if row and isinstance(rows := json.loads(row[1]), dict) and all(isinstance(model, dict) for model in rows.values()):
                return row[0], rows
        except (OSError, sqlite3.Error, ValueError):
            pass
        return 0, None

    def write(self, key, fetched, rows):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with closing(sqlite3.connect(self.path, timeout=1)) as db, db:
                db.execute("CREATE TABLE IF NOT EXISTS discovery (key TEXT PRIMARY KEY, fetched REAL, rows TEXT)")
                db.execute(
                    "INSERT INTO discovery VALUES (?, ?, ?) ON CONFLICT(key) DO UPDATE SET "
                    "fetched=excluded.fetched, rows=excluded.rows WHERE excluded.fetched >= discovery.fetched",
                    (key, fetched, json.dumps(rows)),
                )
        except (OSError, sqlite3.Error):
            pass

    def request_headers(self, provider, kind, refresh=None):
        headers = httpx.Headers()
        for name, value in provider.header_layer:
            if resolved := self.secrets.resolve(value, refresh):
                headers[name] = resolved
        key = self.secrets.resolve(provider.key, refresh)
        if not key:
            key = next((value for name in model_catalog.provider_info(provider.name).get("envVars", [])
                        if (value := os.getenv(name))), None)
        if key in model_catalog.local_key_placeholders():
            key = None
        if key and kind != "ollama":
            headers["Authorization"] = "Bearer " + key
        available = key or provider.mode == "none" or "Authorization" in headers or "x-api-key" in headers
        return headers, bool(available)

    @staticmethod
    def with_endpoint(rows, base, kind):
        url = base + "/v1" if kind in ("ollama", "llama.cpp") and not base.endswith("/v1") else base
        return {identifier: {**row, "baseUrl": url} for identifier, row in rows.items()}

    def models(self, provider, reference, *, optional=False):
        config, discovery = provider.config, provider.discovery
        kind, api = discovery["type"], config.get("api", "openai-completions")
        stale = None
        try:
            default = {"ollama": "http://127.0.0.1:11434", "llama.cpp": LLAMA_DEFAULTS[0]}.get(kind, "http://127.0.0.1:1234")
            url = endpoint(config.get("baseUrl"), default)
            path = b"" if kind == "ollama" else url.raw_path.rstrip(b"/")
            if kind not in ("ollama", "llama.cpp") and discovery.get("injectV1", True) and not path.endswith(b"/v1"):
                path += b"/v1"
            base = str(url.copy_with(raw_path=path or b"/")).rstrip("/")
            headers, available = self.request_headers(provider, kind)
            key = self.cache_key(provider, base, headers)
            fetched, stale = self.read(key)
            if (stale is not None and kind != "llama.cpp" and not self.refresh
                and fetched >= self.config_mtime and time.time() - fetched < TTL):
                return self.with_endpoint(stale, base, kind)
            if not available:
                return self.with_endpoint(stale or {}, base, kind)
            timeout = discovery.get("timeoutMs")
            fetched = time.time()
            with httpx.Client(follow_redirects=False) as client:
                if kind == "ollama":
                    rows = self.ollama(client, base, api, headers, timeout, local_host(url.host))
                elif kind == "llama.cpp":
                    try:
                        rows, headers = self.llama(client, provider, base, api, headers)
                    except httpx.TransportError:
                        if not optional:
                            raise
                        primary_base, primary_cache = base, stale
                        base = LLAMA_DEFAULTS[1]
                        key = self.cache_key(provider, base, headers)
                        _, stale = self.read(key)
                        try:
                            rows, headers = self.llama(client, provider, base, api, headers)
                        except (httpx.HTTPError, ValueError):
                            if stale is None:
                                base, stale = primary_base, primary_cache
                            raise
                    key = self.cache_key(provider, base, headers)
                else:
                    rows, headers = self.openai(client, provider, base, api, headers, timeout, reference)
                    key = self.cache_key(provider, base, headers)
            self.write(key, fetched, rows)
            return self.with_endpoint(rows, base, kind)
        except (httpx.HTTPError, httpx.InvalidURL, ValueError) as error:
            if not optional:
                issue = f"HTTP {error.response.status_code}" if isinstance(error, httpx.HTTPStatusError) else type(error).__name__
                print(f"[model discovery for {provider.name}: {issue}" + ("; using cached models]" if stale is not None else "]"),
                      file=sys.stderr)
            return self.with_endpoint(stale, base, kind) if stale is not None else {}

    @staticmethod
    def cache_key(provider, base, headers):
        return sha256(json.dumps(
            [3, model_catalog.REVISION, provider.name, base, provider.config.get("api"), provider.mode, provider.discovery,
             sorted(headers.multi_items())], sort_keys=True,
        ).encode()).hexdigest()

    @staticmethod
    def ollama(client, base, api, headers, timeout, loopback):
        response = client.get(base + "/api/tags", headers=headers, timeout=timeout / 1000 if timeout else (0.25 if loopback else 10))
        response.raise_for_status()
        roster = {}
        for item in entries(record(response.json()).get("models")):
            item = record(item)
            identifier = item.get("model") or item.get("name")
            if isinstance(identifier, str) and identifier:
                name = item.get("name")
                roster[identifier] = name if isinstance(name, str) and name else identifier

        def show(identifier):
            try:
                result = client.post(base + "/api/show", headers=headers, json={"model": identifier},
                                     timeout=timeout / 1000 if timeout else (0.15 if loopback else 10))
                result.raise_for_status()
                data = record(result.json())
            except (httpx.HTTPError, ValueError):
                data = {}
            return {"name": roster[identifier], "api": api, "cost": COST,
                    **ollama_metadata(data)}

        with ThreadPoolExecutor(max_workers=8) as pool:
            return dict(zip(roster, pool.map(show, roster)))

    @staticmethod
    def llama_probe(client, models_base, timeout, headers=None):
        props_base = models_base.removesuffix("/v1")
        loopback = local_host(httpx.URL(models_base).host)
        list_timeout = timeout / 1000 if timeout else (0.25 if loopback else 10)
        props_timeout = timeout / 1000 if timeout else (0.15 if loopback else 10)

        def properties():
            try:
                response = client.get(props_base + "/props", headers=headers, timeout=props_timeout)
                response.raise_for_status()
                return record(response.json())
            except (httpx.HTTPError, ValueError):
                return {}

        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(copy_context().run, properties)
            response = client.get(models_base + "/models", headers=headers, timeout=list_timeout)
            return response, pending.result()

    def llama(self, client, provider, base, api, headers):
        started = time.monotonic()
        timeout = provider.discovery.get("timeoutMs")
        response, props = self.llama_probe(client, base, timeout, headers)
        if response.status_code == 401:
            updated, available = self.request_headers(provider, "llama.cpp", started)
            if available and updated != headers:
                headers = updated
                response, props = self.llama_probe(client, base, timeout, headers)
        response.raise_for_status()
        rows = {}
        for entry in entries(record(response.json()).get("data")):
            item = record(entry)
            if isinstance(identifier := item.get("id"), str) and identifier:
                rows[identifier] = llama_model(identifier, api, item, props)
        return rows, headers

    def llama_runtime(self, client, route, provider):
        base = route.endpoint().removesuffix("/v1")
        try:
            response, props = self.llama_probe(client, base, provider.discovery.get("timeoutMs"))
            response.raise_for_status()
            for entry in entries(record(response.json()).get("data")):
                item = record(entry)
                if item.get("id") == route.model:
                    return llama_metadata(item, props)
        except (httpx.HTTPError, ValueError):
            pass
        return {}

    @staticmethod
    def lm_studio_metadata(client, base, timeout, headers=None):
        base = base.removesuffix("/v1")
        request_headers = {"Accept": "application/json", **(headers or {})}

        def fetch(path):
            try:
                result = client.get(base + path, headers=request_headers, timeout=timeout)
                result.raise_for_status()
                return result.json()
            except (httpx.HTTPError, ValueError):
                return None

        with ThreadPoolExecutor(max_workers=1) as pool:
            old = pool.submit(copy_context().run, fetch, "/api/v0/models")
            current = fetch("/api/v1/models")
            return lm_studio_native_rows(old.result(), current)

    def runtime(self, client, route, provider):
        if provider.discovery["type"] == "llama.cpp":
            return self.llama_runtime(client, route, provider)
        base = route.endpoint()
        timeout = self.lm_studio_timeout(base, provider.discovery.get("timeoutMs"))
        row = self.lm_studio_metadata(client, base, timeout).get(route.model)
        if not row:
            return {}
        updates = {"input": row["input"]}
        if context := row.get("contextWindow"):
            updates["contextWindow"] = context
            if route.max_output_tokens is not None:
                updates["maxTokens"] = route.max_output_tokens
        return updates

    @staticmethod
    def lm_studio_timeout(base, timeout):
        return timeout / 1000 if timeout else (0.25 if local_host(httpx.URL(base).host) else 10)

    def openai(self, client, provider, base, api, headers, timeout, reference):
        kind = provider.discovery["type"]
        seconds = self.lm_studio_timeout(base, timeout) if kind == "lm-studio" else (timeout or 10_000) / 1000

        def attempt(values):
            if kind != "lm-studio":
                return client.get(base + "/models", headers=values, timeout=seconds), {}
            with ThreadPoolExecutor(max_workers=1) as pool:
                metadata = pool.submit(self.lm_studio_metadata, client, base, seconds, values)
                response = client.get(base + "/models", headers=values, timeout=seconds)
                return response, metadata.result()

        started = time.monotonic()
        response, native = attempt(headers)
        if response.status_code == 401:
            updated, available = self.request_headers(provider, kind, started)
            if available and updated != headers:
                headers = updated
                response, native = attempt(headers)
        response.raise_for_status()
        rows = {}
        for item in entries(record(response.json()).get("data")):
            item = record(item)
            identifier = item.get("id")
            if isinstance(identifier, str) and identifier:
                args = item, api, reference(identifier), provider.discovery, provider.name
                row = proxy_model(*args) if kind == "proxy" else openai_model(*args, native.get(identifier))
                rows[identifier] = {"cost": COST, **row}
        return rows, headers
