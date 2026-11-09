"""Recover DeepSeek tool calls that arrive in the assistant's text channel."""

import json
import re
from pathlib import Path
from uuid import uuid4


NUDGE = (Path(__file__).with_name("prompts") / "dsml-tool-call-leak.md").read_text(encoding="utf-8").rstrip()
MAX_NUDGES = 2
TAG = re.compile(r"<\s*(/?)\s*[｜|]{1,2}\s*DSML\s*[｜|]{1,2}\s*(tool_calls|invoke|parameter|calls)\b[^<>]*>?")
TOKEN = re.compile(r"<(｜?)(tool[▁_](?:calls[▁_](?:begin|end)|call[▁_](?:begin|end)|sep))(｜?)>")
QUOTE = re.compile(r"(`{3,}|~{3,})[\s\S]*?(?:\1|$)|(`+)(?!`)[^\n]*?(?<!`)\2(?!`)")
TOOL_LINE = re.compile(r"(^|\n)[ \t]*[A-Za-z_][\w.-]*[ \t]*\r?\n[ \t]*$")
PENDING_NAME = re.compile(r"(^|\n)[ \t]*[A-Za-z_][\w.-]*[ \t]*(?:\r?\n[ \t]*)?$")
BLANK = re.compile(r"\r?\n[ \t]*\r?\n")
POSSIBLE_MARKUP = re.compile(r"<\s*/?\s*(?:[｜|]|tool[▁_]|t(?:o(?:o(?:l)?)?)?$|$)")
NAME = re.compile(r'\sname="([^"]+)"')
JSON_VALUE = re.compile(r'\sstring="false"')
SECTION_OPEN = re.compile(r"<[｜|]DSML[｜|]tool_calls>")
SECTION_CLOSE = re.compile(r"</[｜|]DSML[｜|]tool_calls>")
INVOKE = re.compile(r"<[｜|]DSML[｜|]invoke\b[^<>]*>")
INVOKE_CLOSE = re.compile(r"</[｜|]DSML[｜|]invoke>")
PARAM = re.compile(r"<[｜|]DSML[｜|]parameter\b[^<>]*>")
PARAM_CLOSE = re.compile(r"</[｜|]DSML[｜|]parameter>")
SPACE = re.compile(r"\s*")


def targeted(model, route):
    catalog = (route.catalog_model or {}) if route else {}
    identity = catalog.get("identity") or {}
    compat = catalog.get("compat") or {}
    return (isinstance(identity, dict) and identity.get("class") == "deepseek" or compat.get("streamMarkupHealingPattern") == "dsml"
            or any(re.search(r"(?:^|[/_.:-])deepseek(?:$|[/_.:-]|\d)", value, re.I)
                   for value in (model, catalog.get("id") or "")))


def token_name(match):
    return match[2].replace("▁", "_") if match and (match[1] or match[3]) else None


def skip_space(text, index):
    return SPACE.match(text, index).end()


def reject_constant(value):
    raise ValueError(f"Invalid JSON constant: {value}")


DECODER = json.JSONDecoder(parse_constant=reject_constant)


def envelope(text, start):
    index, calls = start, []
    while invoke := INVOKE.match(text, skip_space(text, index)):
        name = NAME.search(invoke[0])
        if not name:
            return None
        index, arguments = invoke.end(), {}
        while param := PARAM.match(text, skip_space(text, index)):
            key, close = NAME.search(param[0]), PARAM_CLOSE.search(text, param.end())
            if not key or not close:
                return None
            value = text[param.end():close.start()]
            if JSON_VALUE.search(param[0]):
                try:
                    value = DECODER.decode(value.strip())
                except (ValueError, RecursionError):
                    pass
            arguments[key[1]] = value
            index = close.end()
        close = INVOKE_CLOSE.match(text, skip_space(text, index))
        if not close:
            return None
        calls.append((name[1], arguments))
        index = close.end()
    close = SECTION_CLOSE.match(text, skip_space(text, index))
    if close and calls:
        return close.end(), calls
    # A complete invocation may precede an EOF within the outer closing tag.
    tail = text[index:].lstrip()
    if calls and any(tag.startswith(tail) for tag in ("</｜DSML｜tool_calls>", "</|DSML|tool_calls>")):
        return len(text), calls
    return None


def legacy_call(text, start):
    separator = TOKEN.search(text, start)
    name = text[start:separator.start()].strip() if separator else ""
    if not separator or token_name(separator) != "tool_sep" or not name:
        return None
    index = skip_space(text, separator.end())
    fenced = name == "function"
    if fenced:
        fence = text.find("```json", index)
        if fence < 0:
            return None
        name, index = text[index:fence].strip(), skip_space(text, fence + len("```json"))
    if not re.fullmatch(r"[\w.:-]+", name):
        return None
    try:
        args, index = DECODER.raw_decode(text, index)
    except (ValueError, RecursionError):
        return None
    index = skip_space(text, index)
    if fenced and text.startswith("```", index):
        index = skip_space(text, index + 3)
    close = TOKEN.match(text, index)
    key = token_name(close)
    if not isinstance(args, dict) or key not in ("tool_call_end", "tool_calls_end", "tool_call_begin"):
        return None
    return (index if key == "tool_call_begin" else close.end()), (name, args)


def unquoted_markup(text):
    tags = [(match.start(), match.end(), bool(match[1])) for match in TAG.finditer(text)]
    tags.extend((match.start(), match.end(), name.endswith("_end")) for match in TOKEN.finditer(text)
                if (name := token_name(match)))
    if not tags:
        return text
    start = min(tags)[0]
    _, end, closed = max(tags)
    if name := TOOL_LINE.search(text[:start]):
        start = name.start() + len(name[1])
    if not closed:
        blank = BLANK.search(text, end)
        end = blank.start() if blank else len(text)
    before, after = text[:start].rstrip(), text[end:].lstrip()
    return before + ("\n\n" if before and after else "") + after


def strip_markup(text):
    if not POSSIBLE_MARKUP.search(text):
        return text
    output, cursor = [], 0
    for quoted in QUOTE.finditer(text):
        output.extend((unquoted_markup(text[cursor:quoted.start()]), quoted[0]))
        cursor = quoted.end()
    output.append(unquoted_markup(text[cursor:]))
    return "".join(output)


def heal(text):
    output, calls, cursor = [], [], 0
    broken = False
    tokens = re.compile(r"[<`~]")
    for match in tokens.finditer(text):
        start = match.start()
        if start < cursor:
            continue
        if quoted := QUOTE.match(text, start):
            output.append(text[cursor:quoted.end()])
            cursor = quoted.end()
        elif not broken and (opening := SECTION_OPEN.match(text, start)) and (parsed := envelope(text, opening.end())):
            end, recovered = parsed
            output.append(text[cursor:start])
            calls.extend(recovered)
            cursor = end
        elif tag := TAG.match(text, start):
            if tag[1]:
                if not broken and any(pattern.match(text, start) for pattern in (SECTION_CLOSE, INVOKE_CLOSE, PARAM_CLOSE)):
                    output.append(text[cursor:start])
                    cursor = tag.end()
                if tag[2] in ("tool_calls", "calls"):
                    broken = False
            else:
                broken = True
        elif (token := TOKEN.match(text, start)) and (key := token_name(token)):
            if key == "tool_call_begin":
                if not broken and (parsed := legacy_call(text, token.end())):
                    cursor_end, recovered = parsed
                    output.append(text[cursor:start])
                    calls.append(recovered)
                    cursor = cursor_end
                else:
                    broken = True
            elif not broken and key in ("tool_calls_begin", "tool_calls_end"):
                output.append(text[cursor:start])
                cursor = token.end()
            if key == "tool_calls_end":
                broken = False
    output.append(text[cursor:])
    return "".join(output), calls


def make_call(name, arguments):
    return {"id": "call_" + uuid4().hex[:24], "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}}


class TextBuffer:
    """Stream ordinary prose, keeping candidate markup and adjacent layout until finalization."""

    def __init__(self):
        self.parts = []
        self.pending = ""
        self.held = False
        self.emitted = 0

    def feed(self, text):
        self.parts.append(text)
        if self.held:
            return ""
        pending = self.pending + text
        if POSSIBLE_MARKUP.search(pending):
            self.held = True
            return ""
        name = PENDING_NAME.search(pending)
        end = name.start() if name else len(pending.rstrip())
        visible, self.pending = pending[:end], pending[end:]
        self.emitted += len(visible)
        return visible

    def finish(self):
        raw = "".join(self.parts)
        healed, calls = heal(raw) if self.held else (raw, [])
        clean = strip_markup(healed)
        tail = clean[self.emitted:] if clean.startswith(raw[:self.emitted]) else ""
        return clean, tail, calls, clean != healed


def clean_message(message):
    changed = False
    for field in ("content", "_thinking"):
        original = message.get(field)
        if not isinstance(original, str):
            continue
        clean = strip_markup(original)
        if clean != original:
            message[field] = clean if clean.strip() else None
            changed = True
    anthropic = (message.get("_anthropic") or {}).get("blocks", ())
    responses = (message.get("_responses") or {}).get("items", ())
    native_thoughts = [block.get("thinking") for block in anthropic if block.get("type") == "thinking"]
    native_thoughts.extend(part.get("text") for item in responses if item.get("type") == "reasoning"
                           for part in [*(item.get("summary") or ()), *(item.get("content") or ())]
                           if isinstance(part, dict))
    if any(isinstance(thought, str) and strip_markup(thought) != thought for thought in native_thoughts):
        changed = True
    if changed:
        message.pop("_responses", None)
        message.pop("_anthropic", None)
    return changed
