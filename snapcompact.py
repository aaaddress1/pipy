"""Bounded, model-aware image archives for discarded conversation history."""

import base64
import json
import math
import re
import unicodedata
from dataclasses import asdict, replace

import regex

import model_catalog
from interrupted_thinking import reader_allows
from snapcompact_render import (
    DIM_OFF, DIM_ON, NEWLINE, SHAPES, Shape, bitmap_font, is_wide, paginate, render, silver_coverage,
)


MAX_FRAMES = 80
FRAME_BYTES = 170_000
DEFAULT_BYTES = 3_000_000
ANSI = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[@-_])")
DATA_URL = re.compile(r"data:[\w.+/-]*(?:;[\w=.+/-]+)*,[A-Za-z0-9%+/=_-]+", re.I)
SPACE = regex.compile(r"[\s\p{Cf}]+")
PICTOGRAPH = regex.compile(r"\p{Extended_Pictographic}")
DIM = str.maketrans({DIM_ON: None, DIM_OFF: None})
FOLDS = dict(zip("‘’‚‛“”„′″‵‶‹›‐‑‒–—―−⁄․•‣⁃∙●■▪✓✔✗✘", (
    "'", "'", "'", "'", '"', '"', '"', "'", '"', "'", '"', "<", ">",
    "-", "-", "-", "-", "-", "-", "-", "/", ".", "*", "*", "-", "*", "*", "*", "*", "v", "v", "x", "x",
)))
FOLDS.update({
    "‥": "..", "…": "...", "⋯": "...", "←": "<-", "↑": "^", "→": "->", "↓": "v", "↔": "<->",
    "⇐": "<=", "⇒": "=>", "⇔": "<=>", "✅": "[OK]", "☑": "[OK]", "✔": "[OK]", "❌": "[FAIL]",
    "❎": "[FAIL]", "✖": "[FAIL]", "⚠": "[WARN]", "🚨": "[ALERT]", "ℹ": "[INFO]", "🐛": "[BUG]",
    "💥": "[CRASH]", "🔥": "[HOT]", "🔒": "[LOCK]", "🔓": "[UNLOCK]", "📁": "[DIR]", "📂": "[DIR]",
    "📄": "[FILE]", "📝": "[NOTE]", "🧪": "[TEST]", "⏳": "[WAIT]", "⌛": "[WAIT]", "🚀": "[RUN]",
})


def elide_data(text):
    return DATA_URL.sub("[data URL elided]", text)


def truncate(text, limit):
    text = elide_data(text).translate(DIM)
    if len(text) <= limit:
        return text
    head = round(limit * 0.6)
    return f"{text[:head]} […{len(text) - limit}ch elided…] {text[-(limit - head):]}"


def text_of(content):
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in (content or []) if part.get("type") in ("text", "input_text"))


def thinking_of(message):
    return message.get("_thinking") or "".join(
        block.get("thinking", "") for block in message.get("_anthropic", {}).get("blocks", ())
        if block.get("type") == "thinking"
    )


def serialize(messages, include_thinking=True, previous_files=None):
    groups, merged = [], set()
    useless = {
        msg["tool_call_id"] for msg in messages
        if msg["role"] == "tool" and msg.get("useless") and not msg.get("is_error")
    }
    outputs = {
        msg["tool_call_id"]: text_of(msg.get("content"))
        for msg in messages if msg["role"] == "tool" and msg["tool_call_id"] not in useless
    }
    files = {name: set(paths) for name, paths in (previous_files or {}).items()}
    for name in ("read", "write"):
        files.setdefault(name, set())

    def add(label, text):
        if not text.strip():
            return
        if groups and groups[-1][0] == label:
            groups[-1][1].append(text)
        else:
            groups.append((label, [text]))

    def output(raw):
        return f"<out>\n{DIM_ON}{truncate(raw, 2000)}{DIM_OFF}\n</out>"

    for message in messages:
        role = message["role"]
        content = text_of(message.get("content")).translate(DIM)
        if "_interrupted_thinking" in message:
            if include_thinking:
                add("think", content)
        elif role in ("user", "developer"):
            add("user", content)
        elif role == "assistant":
            if include_thinking:
                add("think", thinking_of(message).translate(DIM))
            add("ai", content)
            for call in message.get("tool_calls", ()):
                if call["id"] in useless:
                    continue
                name = call["function"]["name"]
                try:
                    args = json.loads(call["function"]["arguments"])
                except (ValueError, TypeError):
                    args = {}
                if not isinstance(args, dict):
                    args = {}
                intent = args.pop("i", "")
                path = args.get("path")
                if isinstance(path, str) and path and "://" not in path:
                    if name == "read":
                        files["read"].add(path)
                    elif name in ("write", "edit"):
                        files["write"].add(path)
                rendered = ", ".join(
                    f"{key}={truncate(json.dumps(value, ensure_ascii=False, separators=(',', ':')), 500)}"
                    for key, value in args.items()
                )
                text = f"{name}({truncate(rendered, 2000)})"
                if isinstance(intent, str) and intent.strip():
                    text += "//" + " ".join(intent.translate(DIM).split())
                if outputs.get(call["id"]):
                    text += "\n" + output(outputs[call["id"]])
                    merged.add(call["id"])
                add("call", text)
        elif role == "tool" and message["tool_call_id"] not in merged and outputs.get(message["tool_call_id"]):
            add("call", "\n" + output(outputs[message["tool_call_id"]]))
    transcript = "\n\n".join("¶" + label + ":" + "\n".join(parts) for label, parts in groups)
    return elide_data(transcript), {kind: sorted(paths) for kind, paths in files.items()}


def normalize(text, shape):
    collapsed = SPACE.sub(
        lambda match: NEWLINE if any(char in match[0] for char in "\n\r\u2028\u2029")
        else (" " if any(unicodedata.category(char) != "Cf" for char in match[0]) else ""),
        ANSI.sub("", elide_data(text)),
    )
    supported = silver_coverage()
    bitmap = bitmap_font(shape.font)[1] if shape.font != "silver" else {}
    parts = []
    for char in collapsed:
        code = ord(char)
        if char in (NEWLINE, DIM_ON, DIM_OFF) or 32 <= code < 127 or 160 <= code <= 255:
            parts.append(char)
        elif char in FOLDS:
            parts.append(FOLDS[char])
        elif 0x2500 <= code <= 0x257F:
            parts.append("|" if char in "│┃" else "-" if char in "─━" else "+")
        elif not PICTOGRAPH.fullmatch(char):
            if code in bitmap or code in supported:
                parts.append(char)
            else:
                folded = unicodedata.normalize("NFKD", char)
                plain = "".join(part for part in folded if ord(part) < 127 and part.isprintable())
                if plain:
                    parts.append(plain)
                elif unicodedata.category(char) not in ("Cc", "Mn", "Me", "Cs"):
                    parts.append("?")
    return re.sub(r" +", " ", "".join(parts)).strip(" " + NEWLINE)


def includes_thinking(route):
    return route is None or reader_allows(route)


def model_revision(model):
    match = re.search(r"(?:opus|sonnet|haiku|gpt|grok|deepseek[-_.]v)[-_. ]*(\d+)(?:[-_. ](\d+))?", model)
    if not match:
        match = re.search(r"(\d+)[-_. ](\d+)[-_. ](?:opus|sonnet|haiku)", model)
    if not match:
        return (0, 0)
    major, minor = match.groups()
    if minor is None and len(major) == 2 and "claude" in model:
        major, minor = major
    return int(major), int(minor or "0")


def resolve_shape(route, name, text):
    model = route.model.lower() if route else ""
    anthropic = "claude" in model or (route and route.api == "anthropic-messages")
    high_res = anthropic and (
        "fable" in model or "mythos" in model
        or ("opus" in model and model_revision(model) >= (4, 7))
    )
    if name != "auto":
        shape = SHAPES[name]
    elif anthropic:
        shape = replace(SHAPES["11on16-bw"], size=1932 if high_res else 1568)
    elif "glm" in model:
        shape = SHAPES["8on16-bw"]
    else:
        shape = replace(SHAPES["8on22-bw"], size=2048 if "gemini" in model else 1568)
    if name == "auto":
        coverage = silver_coverage()
        wide = [char for char in text if is_wide(char)]
        if len(wide) >= max(8, len(text) * 0.25) and all(ord(char) in coverage for char in wide):
            shape = SHAPES["silver16-bw"]
    if route and route.image_max_edge:
        shape = replace(shape, size=min(route.image_max_edge, shape.size))
    return shape


def plan(source, high, max_frames):
    edge = high.capacity
    if len(source) <= 2 * edge:
        return source, [], "", source, 0
    head, middle, tail = source[:edge], source[edge:-edge], source[-edge:]
    if max_frames < 1:
        return head, [], tail, head + tail, len(middle)
    pages = paginate(middle, high)
    if len(pages) <= max_frames:
        return head, [(page, high) for page in pages], tail, source, 0
    if high.columns == 2:
        pinned, recent = pages[:1], pages[1:][-(max_frames - 1):] if max_frames > 1 else []
        dropped = sum(len(page) for page in pages[1:len(pages) - len(recent)])
        kept = pinned + recent
        return head, [(page, high) for page in kept], tail, head + " ".join(" ".join(p.splitlines()) for p in kept) + tail, dropped

    low = replace(SHAPES["8on16-bw"], size=high.size)
    if high.font == "silver" or low.capacity <= high.capacity:
        low = high
    count = min(3, (max_frames - 1) // 2)
    first, last = pages[:count], pages[-count:] if count else []
    lead, end = "".join(first), "".join(last)
    center = paginate(middle[len(lead):len(middle) - len(end)], low)
    available = max_frames - 2 * count
    omitted = max(0, len(center) - available)
    kept = center[omitted:]
    frames = [(page, high) for page in first] + [(page, low) for page in kept] + [(page, high) for page in last]
    return head, frames, tail, head + lead + "".join(kept) + end + tail, sum(map(len, center[:omitted]))


def anthropic_tokens(width, height, edge, budget):
    long, short = max(width, height), min(width, height)

    def patches(long_side, short_side):
        return math.ceil(long_side / 28) * math.ceil(short_side / 28)

    if math.ceil(long / 28) * 28 <= edge and patches(long, short) <= budget:
        return patches(long, short)
    lo, hi = 1, long
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if math.ceil(mid / 28) * 28 <= edge and patches(mid, max(1, round(mid * short / long))) <= budget:
            lo = mid
        else:
            hi = mid
    return patches(lo, max(1, round(lo * short / long)))


def openai_tokens(width, height, edge, budget, multiplier):
    shrink = min(1, edge / max(width, height))
    width, height = max(1, math.floor(width * shrink + 0.5)), max(1, math.floor(height * shrink + 0.5))
    patches = math.ceil(width / 32) * math.ceil(height / 32)
    if budget is not None and patches > budget:
        shrink = math.sqrt(1024 * budget / (width * height))
        sw, sh = width * shrink / 32, height * shrink / 32
        shrink *= min(math.floor(sw) / sw, math.floor(sh) / sh)
        width, height = math.floor(width * shrink), math.floor(height * shrink)
        patches = min(budget, math.ceil(width / 32) * math.ceil(height / 32)) if width and height else budget
    return math.ceil(patches * multiplier)


def deepseek_tokens(width, height):
    if width * height < 295_936:
        scale = math.sqrt(295_936 / (width * height))
        width, height = int(width * scale), int(height * scale)

    def span(rows, cols):
        return rows * (cols + 1) + 2

    tokens = span(math.ceil(height / 42), math.ceil(width / 42))
    if tokens <= 1024:
        return tokens
    aspect = height / width
    cols = math.sqrt(1022 / aspect + 0.25) - 0.5
    rows = cols * aspect
    if cols < 1:
        return span(511, 1)
    if rows < 1:
        return span(1, 1021)
    scale = min(math.floor(cols) * 42 / width, math.floor(rows) * 42 / height)
    return span(math.ceil(math.floor(height * scale / 14) / 3), math.ceil(math.floor(width * scale / 14) / 3))


def catalog_image_rule(route):
    model = route.catalog_model if route else None
    if not model or not all(key in model for key in ("api", "provider", "id")):
        return None
    policy = model_catalog.catalog_policy(model["provider"], model["id"], route.upstream)
    return (policy or model_catalog.model_policy(model, route.upstream)).get("imageTokenization")


def image_token_cost(rule, width, height, detail):
    regime = rule["regime"]
    if regime == "fixed":
        return rule["tokens"]
    if regime == "anthropic-patch":
        return anthropic_tokens(width, height, rule["maxEdge"], rule["maxTokens"])
    if regime == "deepseek-patch":
        return deepseek_tokens(width, height)
    sizing = rule[detail if detail in ("low", "high", "original") else rule["auto"]]
    return openai_tokens(width, height, sizing["maxEdge"], sizing.get("patchBudget"), rule["multiplier"])


def frame_tokens(route, width, height):
    if route and route.image_tokens:
        return route.image_tokens
    if rule := catalog_image_rule(route):
        return image_token_cost(rule, width, height, route.image_detail)
    model = route.model.lower() if route else ""
    if "gemini" in model:
        return 1120
    revision = model_revision(model)
    if "deepseek" in model and "flash" in model and revision == (4, 1):
        return deepseek_tokens(width, height)
    if "claude" in model or (route and route.api == "anthropic-messages"):
        standard = any(family in model and revision < floor for family, floor in (
            ("opus", (4, 7)), ("sonnet", (4, 7)), ("haiku", (4, 7)),
        ))
        return anthropic_tokens(width, height, 1568 if standard else 2576, 1568 if standard else 4784)
    if "grok" in model and (4, 6) <= revision <= (4, 20):
        return openai_tokens(width, height, 65535, 2401, 1.017)
    if not route:
        return 5024
    detail = route.image_detail or "auto"
    limits = {"low": (512, None), "high": (2048, 2500), "original": (6000, 10000)}
    if "gpt" in model and (5, 2) <= revision < (5, 4):
        limits = dict.fromkeys(limits, (2048, 6144))
    elif "gpt" in model and revision == (5, 4):
        limits["low"] = (2048, 6144)
        if detail == "auto":
            detail = "high"
    elif "gpt" in model and (5, 6) <= revision < (10, 0):
        limits["original"] = (65535, None)
        if "gpt-6-astra" in model:
            limits["high"] = (65535, 2500)
    return openai_tokens(width, height, *limits.get(detail, limits["original"]), 1.2)


def frame_budget(route, shape, room):
    byte_budget = route.image_byte_budget
    scaled = shape in (replace(SHAPES[name], size=shape.size) for name in ("8on22-bw", "11on16-bw"))
    estimated = min(FRAME_BYTES, math.ceil(FRAME_BYTES * (shape.size / 1932) ** 2)) if scaled else FRAME_BYTES
    limits = [MAX_FRAMES, route.image_count_budget, max(1, byte_budget // estimated)]
    if room is not None:
        edge_tokens = math.ceil(2 * shape.capacity * 1.15 / 4)
        limits.append(max(1, (room - edge_tokens - 2000) // frame_tokens(route, shape.size, shape.size)))
    return min(limits)


def build_archive(source, shape, max_frames, truncated=0, files=None):
    head, pages, tail, kept, dropped = plan(source, shape, max_frames)
    dim = head.rfind(DIM_ON) > head.rfind(DIM_OFF)
    frames = []
    for text, page_shape in pages:
        text = DIM_ON + text if dim else text
        dim = text.rfind(DIM_ON) > text.rfind(DIM_OFF)
        png, height, rows = render(text, page_shape)
        frames.append({
            "data": base64.b64encode(png).decode("ascii"), "width": page_shape.size, "height": height,
            "cols": page_shape.cols, "rows": rows, "chars": len(text.translate(DIM)), "shape": asdict(page_shape),
        })
    return {
        "text": kept, "head": head, "tail": tail, "frames": frames, "truncated": truncated + dropped,
        "shape": asdict(shape), "files": files or {},
    }


def without_thinking(text):
    pieces = re.split(rf"(?<![^{NEWLINE}\n])(?=¶(?:user|think|ai|call):)", text)
    return "".join(part for part in pieces if not part.startswith("¶think:"))


def plain(text):
    return text.translate(DIM).replace(NEWLINE, "\n")


def summary(archive, frames, include_thinking, dropped):
    scopes = "¶user: user, ¶ai: assistant reply, ¶call: tool call"
    if include_thinking:
        scopes += ", ¶think: assistant reasoning"
    lines = [
        "Resume prior conversation. Earlier turns are archived below, oldest→newest. Read them before continuing.",
        f"Archived scopes are {scopes}. Unprefixed lines continue the current scope.",
        "A tool call uses name(args)//intent. <out>…</out> marks tool output.",
    ]
    if frames:
        columns = archive["shape"]["columns"]
        layout = "Read the left column top to bottom, then the right." if columns == 2 else "Read each grid left to right, top to bottom."
        lines.append(f"The middle is in {len(frames)} transcript image(s). {layout} A solid black cell marks a newline.")
    if archive["text"].startswith("[Summary of earlier history]"):
        lines.append("History opens with the condensed digest of still older context.")
    if dropped:
        lines.append(f"About {dropped} characters of older middle history were omitted.")
    lines.append("When an exact detail is unclear, re-read files or rerun commands to verify it.")
    read = set(archive["files"].get("read", ()))
    written = set(archive["files"].get("write", ()))
    paths = sorted(read | written)
    if paths:
        lines.append("\nFILES\n===================")
        lines.extend(f"{path} ({'RW' if path in read and path in written else 'Read' if path in read else 'Write'})" for path in paths[:20])
        if len(paths) > 20:
            lines.append(f"[…{len(paths) - 20} files elided…]")
    return "\n".join(lines) + "\n\nHISTORY\n==================="


def reader_shape_key(route, shape_name):
    return [route.model, route.api, route.image_max_edge, shape_name]


def history(archive, route, shape_name="auto", threshold=None):
    allow_thinking = includes_thinking(route)
    source = archive["text"] if allow_thinking else without_thinking(archive["text"])
    if not route or not route.vision:
        limit = int(threshold * 0.6) * 4 if threshold is not None else None
        head, tail = source, ""
        if limit is not None and len(source.encode("utf-8")) > limit:
            raw = source.encode("utf-8")
            first = (limit + 1) // 2
            head = raw[:first].decode("utf-8", errors="ignore")
            tail = raw[-(limit - first):].decode("utf-8", errors="ignore") if limit > first else ""
        omitted = archive["truncated"] + len(source) - len(head) - len(tail)
        text = plain(head) + ("\n--- middle history omitted ---\n" + plain(tail) if tail else "")
        return summary(archive, [], allow_thinking, omitted), [{"type": "text", "text": text}], (len(text.encode()) + 3) // 4

    stored_shape = Shape(**archive["shape"])
    same_reader = archive.get("shape_reader") == reader_shape_key(route, shape_name)
    shape = stored_shape if same_reader else resolve_shape(route, shape_name, source)
    if shape != stored_shape or source != archive["text"]:
        archive = build_archive(source, shape, frame_budget(route, shape, None), archive["truncated"], archive["files"])
    accepted, omitted, bytes_left = set(), archive["truncated"], route.image_byte_budget
    for index in range(len(archive["frames"]) - 1, -1, -1):
        frame = archive["frames"][index]
        size = len(frame["data"])
        if size <= bytes_left and len(accepted) < route.image_count_budget:
            bytes_left -= size
            accepted.add(index)
        else:
            omitted += frame["chars"]
    blocks, cost = [], 0
    if archive["head"]:
        blocks.append({"type": "text", "text": plain(archive["head"]) + ("\n--- imaged middle below ---" if accepted else "")})
    gaps = 0
    for index, frame in enumerate(archive["frames"]):
        if index not in accepted:
            gaps += 1
            continue
        if gaps:
            blocks.append({"type": "text", "text": f"--- {gaps} archived image frame(s) omitted ---"})
            gaps = 0
        image = {"url": "data:image/png;base64," + frame["data"]}
        if route.image_detail:
            image["detail"] = route.image_detail
        blocks.append({"type": "image_url", "image_url": image})
        cost += frame_tokens(route, frame["width"], frame["height"])
    if gaps:
        blocks.append({"type": "text", "text": f"--- {gaps} archived image frame(s) omitted ---"})
    if archive["tail"]:
        edge = "--- imaged middle above ---\n" if accepted else "\n--- middle history omitted ---\n"
        blocks.append({"type": "text", "text": edge + plain(archive["tail"])})
    cost += sum((len(block["text"].encode()) + 3) // 4 for block in blocks if block["type"] == "text")
    return summary(archive, accepted, allow_thinking, omitted), blocks, cost


def compact(messages, previous_summary, previous_archive, route, shape_name, room):
    previous_archive = previous_archive or {}
    thinking = includes_thinking(route)
    previous = previous_archive.get("text", "")
    if not thinking:
        previous = without_thinking(previous)
    text, files = serialize(messages, thinking, previous_archive.get("files"))
    shape = resolve_shape(route, shape_name, previous + text)
    normalized = normalize(text, shape)
    if previous:
        normalized = NEWLINE.join(part for part in (previous, normalized) if part)
    elif previous_summary:
        lead = normalize(previous_summary, shape)
        normalized = f"[Summary of earlier history] {lead}{NEWLINE}[Recent conversation] {normalized}"
    max_frames = frame_budget(route, shape, room)
    truncated = previous_archive.get("truncated", 0)
    for attempt in range(3):
        archive = build_archive(normalized, shape, max_frames, truncated, files)
        archive["shape_reader"] = reader_shape_key(route, shape_name)
        frames = archive["frames"]
        size = sum(len(frame["data"]) for frame in frames)
        allowed = len(frames)
        if size > route.image_byte_budget:
            allowed = len(frames) * route.image_byte_budget // size
        if room is not None and len(frames) > 1:
            lead, _, used = history(archive, route, shape_name)
            used += (len(lead.encode("utf-8")) + 3) // 4
            if used > room:
                frame_cost = sum(frame_tokens(route, frame["width"], frame["height"]) for frame in frames) / len(frames)
                allowed = min(allowed, max(1, len(frames) - math.ceil((used - room) / frame_cost)))
        if allowed >= len(frames) or attempt == 2 or allowed < 1:
            if size > route.image_byte_budget:
                raise RuntimeError("Snapcompact archive exceeds the route's image payload budget.")
            return archive
        max_frames = allowed


def anthropic_content(content):
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    blocks = []
    for part in content or []:
        if part["type"] == "image_url":
            url = part["image_url"]["url"]
            prefix, separator, data = url.partition(",")
            source = {"type": "base64", "media_type": prefix[5:-7], "data": data} if (
                separator and prefix.startswith("data:") and prefix.endswith(";base64")
            ) else {"type": "url", "url": url}
            blocks.append({"type": "image", "source": source})
        else:
            blocks.append(part)
    return blocks


def responses_content(content):
    if isinstance(content, str):
        return content
    return [
        {"type": "input_image", "image_url": part["image_url"]["url"], "detail": part["image_url"].get("detail", "auto")}
        if part["type"] == "image_url" else {"type": "input_text", "text": part["text"]}
        for part in content or []
    ]
