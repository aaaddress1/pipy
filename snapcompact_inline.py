"""Request-only Snapcompact imaging for instructions and tool results."""

import base64
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO

from PIL import Image

from snapcompact import frame_tokens, model_revision, normalize, resolve_shape, text_of
from snapcompact_render import paginate, render


SYSTEM_FRAMES = 6
MIN_TOOL_TOKENS = 3000
SAVINGS_MARGIN = 0.9
SYSTEM_STUB = (
    "Full operating instructions: PNG image(s) attached at start of first user message. "
    "Before anything else, read every frame carefully, in order; follow as authoritative system prompt."
)
SYSTEM_NOTE = "=== OPERATING INSTRUCTIONS — image(s) below: your system prompt ==="
CONTEXT_STUB = (
    "Loaded context-file instructions: PNG image(s) attached below at the first user message start. "
    "At this marker, read every frame in order; apply as if original context-file text remained here."
)
CONTEXT_NOTE = "=== CONTEXT FILE INSTRUCTIONS — read the image(s) below as the loaded context files replaced in the system prompt ==="
TOOL_NOTE = (
    "[Tool result text: compacted PNG frame(s) immediately below; read as verbatim output except labeled "
    "source-image position markers. Matching original images follow the frames. Do not rerun the call to recover this text.]"
)
CONTEXT_SECTIONS = re.compile(r"<repo-rules>\n.*?\n</repo-rules>|## Context\n<instructions>\n.*?\n</instructions>", re.S)
READ_HEADER = re.compile(r"^(?:\[[^\n]*\]\n)?\[[^\n]+#[0-9A-F]{4}\]\n")
ROW_PREFIX = re.compile(r"^\d+(?:-\d+)?[:|]")
ROW_NUMBER = re.compile(r"^(\d+)(?:-(\d+))?[:|]")
COUNT_BRACKET = re.compile(r"^(?:\d+(?:-\d+)?[:|])?\[(?:0|(?:[»·]\d+)+)\]")
INDENT = re.compile(r"^[ \t]+")


def estimate(text):
    return (len(text.encode("utf-8")) + 3) // 4


def text_part(text):
    return {"type": "text", "text": text}


def parts(content):
    return [text_part(content)] if isinstance(content, str) else (content or [])


def image_size(block):
    url = block["image_url"]["url"]
    return len(url.partition(",")[2]) if url.startswith("data:") else 0


def excludes_webp(route):
    model = route.catalog_model or {}
    return (model.get("imageInputDecoder") == "stb"
            or model.get("provider") in ("ollama", "ollama-cloud", "llama.cpp", "lm-studio", "local-server")
            or os.getenv("OMP_NO_WEBP", "").lower() in ("1", "true"))


@lru_cache(maxsize=8)
def stb_image(data, side, budget):
    try:
        with Image.open(BytesIO(base64.b64decode(data, validate=True))) as original:
            alpha = "A" in original.getbands() or "transparency" in original.info
            image = original.convert("RGBA" if alpha else "RGB")
        image.thumbnail((side, side), Image.Resampling.LANCZOS)
        encoding, mime = ("PNG", "png") if alpha else ("JPEG", "jpeg")
        while True:
            with BytesIO() as output:
                image.save(output, format=encoding, quality=80, optimize=True)
                encoded = base64.b64encode(output.getvalue()).decode("ascii")
            if len(encoded) <= budget:
                return f"data:image/{mime};base64,{encoded}"
            if image.size == (1, 1):
                return None
            image = image.resize(tuple(max(1, edge * 3 // 4) for edge in image.size), Image.Resampling.LANCZOS)
    except (OSError, ValueError, Image.DecompressionBombError):
        return None


def stb_block(block, route):
    kind = block.get("type")
    if kind not in ("image_url", "input_image"):
        return block
    image = block.get("image_url")
    url = image.get("url") if isinstance(image, dict) else image
    if not isinstance(url, str):
        return block
    mime, separator, data = url.partition(",")
    mime = mime.lower()
    if not separator or not mime.startswith("data:") or not mime.endswith(";base64"):
        return block
    try:
        signature = base64.b64decode(data[:16], validate=True)
    except ValueError:
        signature = b""
    if mime != "data:image/webp;base64" and not (signature.startswith(b"RIFF") and signature[8:12] == b"WEBP"):
        return block
    convert = stb_image if len(data) <= 4_000_000 else stb_image.__wrapped__
    converted = convert(data, route.image_max_edge or 1568, route.image_byte_budget)
    if not converted:
        return {"type": "input_text" if kind == "input_image" else "text",
                "text": "[An image was omitted because it could not be decoded for this model.]"}
    return {**block, "image_url": {**image, "url": converted} if isinstance(image, dict) else converted}


def stb_messages(messages, route, *, fields=("content",)):
    if not excludes_webp(route):
        return messages
    result = None
    for index, message in enumerate(messages):
        updates = {}
        for field in fields:
            content = message.get(field)
            if isinstance(content, list):
                normalized = [stb_block(block, route) if isinstance(block, dict) else block for block in content]
                if normalized != content:
                    updates[field] = normalized
        if updates:
            if result is None:
                result = list(messages)
            result[index] = {**message, **updates}
    return messages if result is None else result


def is_failed(text):
    if text.startswith(("Error:", "Interrupted before result")):
        return True
    match = re.match(r"Exit code: (-?\d+)\b", text)
    return bool(match and int(match[1]) != 0)


def user_authored(message):
    if message["role"] != "user" or "_responses_compaction" in message or "_anthropic_compaction" in message:
        return False
    return not text_of(message.get("content")).startswith((
        "Summary of earlier conversation.", "Context replaced.", "Resume prior conversation.", "[Agent recovery]",
        "Branch-return summary:",
    ))


def reads_as_images(route):
    if route.image_read_results is not None:
        return route.image_read_results
    model = route.model.lower()
    return "claude" in model and "opus" in model and (5, 5) <= model_revision(model) < (10, 0)


def read_payload(text, row_prefixes=None):
    header_match = READ_HEADER.match(text) if row_prefixes is not False else None
    header = header_match[0] if header_match else ""
    rest = text[len(header):]
    numbered = row_prefixes
    if numbered is None:
        lines = rest.split("\n", 3)
        if lines[0].startswith("[") and len(lines) > 1:
            lines = lines[1:]
        left = ROW_NUMBER.match(lines[0])
        gap = len(lines) > 1 and lines[1] == "…"
        index = 2 if gap else 1
        right = ROW_NUMBER.match(lines[index]) if len(lines) > index else None
        numbered = bool(header)
        if not numbered and left and right:
            difference = int(right[1]) - int(left[2] or left[1])
            numbered = difference > 1 if gap else difference == 1
    encoded = []
    for line in rest.split("\n"):
        match = ROW_PREFIX.match(line) if numbered else None
        prefix = match[0] if match else ""
        body = line[len(prefix):]
        leading = INDENT.match(body)
        if leading:
            runs = re.findall(r"\t+| +", leading[0])
            bracket = "[" + "".join(("»" if run[0] == "\t" else "·") + str(len(run)) for run in runs) + "]"
            body = bracket + body[len(leading[0]):]
        elif COUNT_BRACKET.match(body):
            body = "[0]" + body
        encoded.append(prefix + body)
    body = "\n".join(encoded)
    note = TOOL_NOTE
    if body != rest:
        note += "\nAt each line start, [»2·4] means two tabs followed by four spaces; [0] means no indentation."
    if header:
        note += "\n" + header.rstrip()
    return body, note


@lru_cache(maxsize=24)
def pages_for(source, shape):
    return tuple(paginate(normalize(source, shape), shape))


@lru_cache(maxsize=32)
def png_page(page, shape):
    png, height, _ = render(page, shape)
    return base64.b64encode(png).decode("ascii"), height


@dataclass(frozen=True)
class Swap:
    index: int
    blocks: tuple
    count: int
    size: int
    saved: int
    system: str | None = None


class InlineImages:
    def __init__(self, mode, tool_results, shape):
        self.mode, self.tool_results, self.shape = mode, tool_results, shape
        self.disabled = False

    def enabled(self, route):
        return bool(route.vision and not self.disabled and (self.mode != "none" or self.tool_results))

    def disable(self):
        if self.disabled or (self.mode == "none" and not self.tool_results):
            return False
        self.disabled = True
        return True

    def system_target(self, system):
        if self.mode == "all":
            return system, SYSTEM_STUB, SYSTEM_NOTE
        if self.mode != "agents-md":
            return None
        sections = CONTEXT_SECTIONS.findall(system)
        if sections:
            return "\n\n".join(sections), CONTEXT_SECTIONS.sub(CONTEXT_STUB, system), CONTEXT_NOTE
        return None

    def plan_swap(self, route, source, removed, count_limit, byte_limit, index, note, replacement=None):
        if count_limit < 1 or byte_limit < 1:
            return None
        shape = resolve_shape(route, self.shape, source)
        pages = pages_for(source, shape)
        cost = len(pages) * frame_tokens(route, shape.size, shape.size)
        if not 0 < len(pages) <= count_limit or cost > removed * SAVINGS_MARGIN:
            return None
        images = []
        size = 0
        for page in pages:
            data, _ = png_page(page, shape)
            size += len(data)
            if size > byte_limit:
                return None
            image = {"url": "data:image/png;base64," + data}
            if route.image_detail:
                image["detail"] = route.image_detail
            images.append({"type": "image_url", "image_url": image})
        return Swap(index, (text_part(note), *images), len(images), size, max(0, removed - cost), replacement)

    def transform(self, messages, route, *, archive_blocks=()):
        if route.vision:
            messages = stb_messages(messages, route)
        if not self.enabled(route) or not messages:
            return messages, 0
        count_limit, byte_limit = route.image_count_budget, route.image_byte_budget
        swaps, names, source_images = [], {}, {}
        count = size = swap_count = swap_size = 0
        first_user = next((i for i, message in enumerate(messages) if user_authored(message)), None)
        archive_index = next((i for i, message in enumerate(messages) if message["role"] == "user"), None)
        scanned = []
        for index, message in enumerate(messages):
            blocks = parts(message.get("content"))
            originals = [block for block in blocks if block.get("type") == "image_url"]
            if index == archive_index and archive_blocks:
                originals.extend(block for block in archive_blocks if block["type"] == "image_url")
            count += len(originals)
            size += sum(map(image_size, originals))
            scanned.append((blocks, originals, count, size))

        target = None
        if first_user is not None and messages[0]["role"] == "system":
            target = self.system_target(messages[0]["content"])
        if target:
            source, replacement, note = target
            _, _, prefix_count, prefix_size = scanned[first_user]
            removed = estimate(messages[0]["content"]) - estimate(replacement) - estimate(note)
            prompt = self.plan_swap(
                route, source, removed, min(SYSTEM_FRAMES, count_limit - prefix_count),
                byte_limit - prefix_size, first_user, note, replacement,
            )
            if prompt:
                swaps.append(prompt)
                swap_count, swap_size = prompt.count, prompt.size

        for index, message in enumerate(messages):
            for call in message.get("tool_calls", ()):
                names[call["id"]] = call["function"]["name"]
            if not self.tool_results or message["role"] != "tool":
                continue
            name = names.get(message["tool_call_id"])
            if name == "read" and not reads_as_images(route):
                continue
            blocks, originals, prefix_count, prefix_size = scanned[index]
            remaining = count_limit - prefix_count - swap_count, byte_limit - prefix_size - swap_size
            body = "\n".join(
                block["text"] if block["type"] == "text" else "[An original source image was attached here.]"
                for block in blocks if block["type"] in ("text", "image_url")
            )
            if estimate(body) < MIN_TOOL_TOKENS or message.get("is_error") or is_failed(body):
                continue
            # The current read tool returns raw file text without row numbers.
            rendered, note = read_payload(body, row_prefixes=False) if name == "read" else (body, TOOL_NOTE)
            swap = self.plan_swap(route, rendered, estimate(body) - estimate(note), *remaining, index, note)
            if swap:
                source_images[index] = originals
                swaps.append(swap)
                swap_count += swap.count
                swap_size += swap.size
        while swaps and (count + swap_count > count_limit or size + swap_size > byte_limit):
            newest = swaps.pop()
            swap_count -= newest.count
            swap_size -= newest.size
        if not swaps:
            return messages, 0
        output = list(messages)
        for swap in swaps:
            message = messages[swap.index]
            if swap.system is not None:
                output[0] = {**messages[0], "content": swap.system}
                tail = parts(message.get("content"))
            else:
                tail = [
                    part for index, image in enumerate(source_images[swap.index], 1)
                    for part in (text_part(f"[Original source image {index}; see its marker in the tool output.]"), image)
                ]
            output[swap.index] = {**message, "content": [*swap.blocks, *tail]}
        return output, sum(swap.saved for swap in swaps)


def chat_tool_images(messages):
    if not any(message["role"] == "tool" and isinstance(message.get("content"), list) for message in messages):
        return messages
    output, images = [], []
    for message in messages:
        if message["role"] != "tool" and images:
            output.append({"role": "user", "content": [text_part("Attached image(s) from tool result:"), *images]})
            images.clear()
        if message["role"] == "tool" and isinstance(message.get("content"), list):
            blocks = message["content"]
            images.extend(block for block in blocks if block["type"] == "image_url")
            message = {**message, "content": "\n".join(block["text"] for block in blocks if block["type"] == "text")}
        output.append(message)
    if images:
        output.append({"role": "user", "content": [text_part("Attached image(s) from tool result:"), *images]})
    return output
