"""Prompt cache options and stable input breakpoints for standard Responses."""

import os
from dataclasses import dataclass


RETENTIONS = ("none", "short", "long")
LATEST = "latest-stable-message"
MARKER = "prompt_cache_breakpoint"
CACHEABLE = frozenset(("input_text", "input_image", "input_file"))


@dataclass(frozen=True)
class PromptCache:
    mode: str
    ttl: str | None = None
    breakpoint: str = LATEST


def resolve_retention(setting=None):
    if setting in RETENTIONS:
        return setting
    env = os.getenv("PI_CACHE_RETENTION")
    return env if env in RETENTIONS else "short"


def parse_policy(value, compat, retention):
    if (not isinstance(value, dict) or value.keys() - {"mode", "ttl", "breakpoint"}
        or value.get("mode") not in ("implicit", "explicit")
        or "ttl" in value and value["ttl"] not in ("30m", "12h")
        or value.get("breakpoint", LATEST) not in (LATEST, "none")):
        raise ValueError("Invalid Responses prompt_cache policy.")
    if retention != "none" and value["mode"] == "explicit" and compat.get("supportsPromptCacheBreakpoints") is not True:
        raise ValueError("Explicit prompt caching is unsupported for this Responses model.")
    return PromptCache(**value)


def prepare(body, route, messages, cache_key=None):
    retention = route.cache_retention
    if retention == "none":
        return body, None
    model = route.catalog_model or {}
    compat = model.get("compat") or {}
    policy = route.prompt_cache
    if cache_key:
        body = {**body, "prompt_cache_key": cache_key}
    if compat.get("cacheControlFormat") == "anthropic":
        body = {**body, "cache_control": {"type": "ephemeral", **({"ttl": "1h"} if retention == "long" else {})}}
    if not compat.get("supportsPromptCacheBreakpoints"):
        if cache_key and retention == "long" and compat.get("supportsLongPromptCacheRetention"):
            body = {**body, "prompt_cache_retention": "24h"}
        return body, None
    if not policy and retention != "long":
        return body, None
    mode = policy.mode if policy else "implicit"
    ttl = (policy.ttl if policy else None) or ("12h" if retention == "long" else compat.get("promptCacheBreakpointTtl"))
    body = {**body, "prompt_cache_options": {"mode": mode, **({"ttl": ttl} if ttl else {})}}
    if mode != "explicit":
        return body, None

    items = [
        {**item, "content": [{"type": "input_text", "text": item["content"]}]}
        if item.get("role") == "user" and isinstance(item.get("content"), str) else item
        for item in body["input"]
    ]
    if body.get("instructions") and model.get("reasoning") and compat.get("supportsDeveloperRole"):
        items[:0] = [{"role": "developer", "content": message["content"]} for message in messages
                     if message["role"] == "system" and message["content"]]
        body.pop("instructions")
    body["input"] = items
    return body, policy.breakpoint


def cacheable_parts(item):
    role, content = item.get("role"), item.get("content")
    if role not in ("assistant", "developer", "system", "user"):
        return ()
    if isinstance(content, str) and content and role in ("developer", "system"):
        return ({"type": "input_text", "text": content},)
    if isinstance(content, list) and all(isinstance(part, dict) and part.get("type") in CACHEABLE for part in content):
        return content
    return ()


def mark(body, policy, baseline=None):
    if policy != LATEST:
        return body
    items = list(body["input"])
    if baseline is not None:
        found_marker = restored = False
        for index, previous in enumerate(baseline):
            parts = cacheable_parts(previous)
            if not any(MARKER in part for part in parts):
                continue
            found_marker = True
            if index >= len(items) or items[index].get("role") != previous.get("role"):
                continue
            current = cacheable_parts(items[index])
            if list(current) == [{key: value for key, value in part.items() if key != MARKER} for part in parts]:
                items[index] = {**items[index], "content": [dict(part) for part in parts]}
                restored = True
        if restored or not found_marker:
            return {**body, "input": items}

    newest = next((index for index in range(len(items) - 1, -1, -1)
                   if items[index].get("role") in ("user", "developer")), 0)
    for index in range(newest - 1, -1, -1):
        if parts := cacheable_parts(items[index]):
            items[index] = {**items[index], "content": [*parts[:-1], {**parts[-1], MARKER: {"mode": "explicit"}}]}
            return {**body, "input": items}
    return body
