"""Local input accounting for maintenance and Responses compaction requests."""

import base64
import binascii
import json
from functools import cache, lru_cache
from io import BytesIO
from pathlib import Path

from PIL import Image

import model_catalog
from snapcompact import catalog_image_rule, image_token_cost


TABLES = Path(__file__).with_name("catalog")
UNKNOWN_IMAGE = (65535, 65535)
OPENAI_IMAGE_RULE = {"regime": "openai-patch", "auto": "original", "multiplier": 1.2,
                     "low": {"maxEdge": 512}, "high": {"maxEdge": 2048, "patchBudget": 2500},
                     "original": {"maxEdge": 6000, "patchBudget": 10000}}
ANTHROPIC_IMAGE_RULE = {"regime": "anthropic-patch", "maxEdge": 2576, "maxTokens": 4784}
CL100K_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|"
    r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
O200K_PATTERN = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
    r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def compact_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


@cache
def encoding(family):
    from tiktoken import Encoding
    from zstandard import ZstdDecompressor

    pattern = CL100K_PATTERN if family == "cl100k" else O200K_PATTERN
    table = TABLES / f"{family}_base.bin.zst"
    with table.open("rb") as file, ZstdDecompressor().stream_reader(file) as reader:
        raw = reader.read()
    if not raw.startswith(b"UTOK1\n"):
        raise ValueError(f"Invalid token table {table.name}.")
    count, offset = int.from_bytes(raw[6:10], "little"), 10
    ranks = {}
    for rank in range(count):
        length, shift = 0, 0
        while True:
            part = raw[offset]
            offset += 1
            length |= (part & 127) << shift
            if part < 128:
                break
            shift += 7
        if length:
            ranks[raw[offset:offset + length]] = rank
        offset += length
    if offset != len(raw):
        raise ValueError(f"Invalid token table size for {table.name}.")
    return Encoding(f"pipy-{family}", pat_str=pattern, mergeable_ranks=ranks, special_tokens={})


@lru_cache(maxsize=512)
def count_text(family, text):
    return len(encoding(family).encode_ordinary(text)) if family else (len(text.encode("utf-8", "replace")) + 3) // 4


@lru_cache(maxsize=64)
def count_large_text(family, text):
    return len(encoding(family).encode_ordinary(text))


@lru_cache(maxsize=128)
def image_size(header):
    try:
        with Image.open(BytesIO(base64.b64decode(header))) as image:
            return image.size
    except (OSError, ValueError, binascii.Error, Image.DecompressionBombError):
        return None


def data_dimensions(url):
    if not isinstance(url, str) or not url.startswith("data:"):
        return UNKNOWN_IMAGE
    start = url.find(",")
    if start < 0 or not url[:start].endswith(";base64"):
        return UNKNOWN_IMAGE
    start += 1
    return image_size(url[start:start + 1024]) or image_size(url[start:start + 87384]) or UNKNOWN_IMAGE


class ContextCounter:
    def __init__(self, route):
        model = route.catalog_model or {} if route else {}
        family = model.get("tokenizer")
        self.family = ("cl100k" if family == "cl100k" else "o200k") if route else None
        self.identity = route.identity() if route else None
        self.vision = route is None or route.vision
        self.default_detail = route.image_detail if route else None
        self.fixed_image = route.image_tokens if route else None
        self.schemas_in_context = model.get("toolSchemasInContext") is not False and model.get("supportsTools") is not False
        compat = model.get("compat") or {}
        self.replay_reasoning = not compat.get("filterReasoningHistory")
        api = route.api if route else None
        self.anthropic_replay = api == "anthropic-messages" and route.native_compaction
        self.provider = (route.broker_provider or model.get("provider") or (
            "anthropic" if route.endpoint() in ("https://api.anthropic.com", "https://api.anthropic.com/v1")
            else route.provider()
        )) if route else None
        self.chat_thinking = compact_json([model.get("provider"), route.model, route.endpoint(), api,
                                           *([route.upstream] if route.upstream else [])]) if (
            route and api == "chat-completions" and compat.get("qwenPreserveThinking")
        ) else None
        self.image_rule = catalog_image_rule(route) or (
            ANTHROPIC_IMAGE_RULE if api in model_catalog.ANTHROPIC_COMPACTION_APIS else OPENAI_IMAGE_RULE
        )
        self._snapshot = None

    def text(self, text):
        if len(text) <= 16384:
            return count_text(self.family, text)
        if self.family is None:
            return (len(text.encode("utf-8", "replace")) + 3) // 4
        if len(text) <= 262144:
            return count_large_text(self.family, text)
        return len(encoding(self.family).encode_ordinary(text))

    def image(self, url, detail=None):
        if self.fixed_image is not None:
            return self.fixed_image
        rule = self.image_rule
        if rule["regime"] == "fixed":
            return rule["tokens"]
        width, height = data_dimensions(url)
        return image_token_cost(rule, width, height, detail or self.default_detail)

    def content(self, content):
        if isinstance(content, str):
            return self.text(content)
        total = 0
        for block in content or ():
            kind = block.get("type")
            if kind in ("text", "input_text", "output_text"):
                total += self.text(block.get("text") or "")
            elif kind in ("image_url", "input_image"):
                image = block.get("image_url")
                url = image.get("url") if isinstance(image, dict) else image
                detail = image.get("detail") if isinstance(image, dict) else block.get("detail")
                total += self.image(url, detail) if self.vision else self.text("[Image omitted for text-only model.]")
        return total

    def message(self, message):
        if snapshot := message.get("_responses_compaction"):
            items = snapshot["items"]
            if (cached := self._snapshot) is not None and cached[0] is items:
                return cached[1]
            cost = self.responses(items)
            self._snapshot = items, cost
            return cost
        if (snapshot := message.get("_anthropic_compaction")) and (size := self.anthropic_snapshot(snapshot)) is not None:
            return size
        count = self.content(message.get("content"))
        for call in message.get("tool_calls", ()):
            function = call["function"]
            count += self.text(function["name"]) + self.text(function["arguments"])
        thinking = message.get("reasoning_content")
        if self.chat_thinking and message.get("_chat_thinking_route") == self.chat_thinking:
            thinking = message.get("_thinking") or thinking
        if thinking:
            count += self.text(thinking)
        if (saved := message.get("_anthropic")) and saved.get("route") == self.identity:
            count += sum(self.text(block.get("thinking") or "") for block in saved.get("blocks", ())
                         if block.get("type") == "thinking")
        if self.replay_reasoning and (saved := message.get("_responses")) and saved.get("route") == self.identity:
            reasoning = saved.get("reasoning_tokens")
            if type(reasoning) is int and reasoning > 0 and any(item.get("type") == "reasoning" for item in saved.get("items", ())):
                count += reasoning
        return count

    def anthropic_snapshot(self, snapshot):
        if not self.anthropic_replay or not (snapshot.get("route") == self.identity
                                            or snapshot.get("anthropic_provider") == self.provider):
            return None
        metadata = snapshot.get("anthropic_metadata") or {}
        files = metadata.get("retainedFiles", ()) if metadata.get("exactTail") else ()
        return self.text(snapshot["block"].get("content") or "") + self.text(metadata.get("filesText") or "") + sum(
            self.text(entry["text"]) for entry in files
        )

    def messages(self, messages):
        return sum(self.message(message) for message in messages)

    def tools(self, schemas):
        if not self.schemas_in_context:
            return 0
        return sum(self.text(tool["name"]) + self.text(tool.get("description") or "")
                   + self.text(compact_json(tool.get("parameters") or {}))
                   for schema in schemas for tool in (schema["function"],))

    def response_parts(self, value):
        fragments, framing = [], []
        images = 0

        def collect(item):
            nonlocal images
            if isinstance(item, str):
                if item:
                    fragments.append(item)
            elif isinstance(item, list):
                for value in item:
                    collect(value)
            elif isinstance(item, dict):
                if item.get("type") == "input_image":
                    images += self.image(item.get("image_url"), item.get("detail"))
                else:
                    for key, value in item.items():
                        if key != "encrypted_content" or not isinstance(value, str):
                            framing.append(key)
                            collect(value)
            else:
                framing.append(compact_json(item))

        collect(value)
        if framing:
            fragments.append(" ".join(framing))
        return fragments, images

    def responses(self, value):
        fragments, images = self.response_parts(value)
        return sum(map(self.text, fragments)) + images
