"""Resolve a concrete effort for each operator prompt in auto mode."""

import json
import re

from judgment import PROMPTS
from model_catalog import EFFORTS


QUESTIONS = json.loads((PROMPTS / "auto-thinking.json").read_text(encoding="utf-8"))
FENCE = re.compile(r"```+.*?(?:```+|\Z)", re.DOTALL)
XML = re.compile(r"<([a-zA-Z][\w-]*)(?:\s[^>]*)?>.*?</\1>", re.DOTALL)
MARKUP = re.compile(r"<!--.*?(?:-->|\Z)|<(/?)([A-Za-z][A-Za-z0-9-]*)(?:[^<>\x22']|\"[^\"]*\"|'[^']*')*>", re.DOTALL)
MARKDOWN_FENCE = re.compile(r"^ {0,3}([`~]{3,})(.*)$")


def supported(model):
    thinking = (model or {}).get("thinking") or {}
    if not (model and model.get("reasoning")) or thinking.get("unverified"):
        return ()
    return tuple(effort for effort in EFFORTS if effort in (thinking.get("efforts") or ()))


def eligible(model, ceiling="max"):
    levels = supported(model)
    pool = tuple(level for level in levels if level != "minimal") or levels
    return tuple(level for level in pool if EFFORTS.index(level) <= EFFORTS.index(ceiling))


def clamp(model, preferred, ceiling="max"):
    pool = eligible(model, ceiling)
    if not pool:
        return None
    return next((level for level in reversed(pool) if EFFORTS.index(level) <= EFFORTS.index(preferred)), pool[0])


def provisional(model):
    preferred = ((model or {}).get("thinking") or {}).get("defaultLevel")
    return clamp(model, preferred if preferred in EFFORTS else "high", "xhigh")


def edges(value, limit):
    return value if len(value) <= 2 * limit else value[:limit] + "\n…\n" + value[-limit:]


def collapse(value):
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", value)).strip()


def preprocess(value):
    value = re.sub(r"\x1b\[[0-9;]*m", "", edges(value, 262_144))
    stripped = collapse(FENCE.sub(" ", value))
    value = XML.sub(" ", edges(stripped if len(stripped) >= 12 else value, 16_384))
    value = collapse(re.sub(r"\b[0-9a-fA-F]{12,}\b", lambda match: match[0][:7], value))
    if len(value) <= 2000:
        return value
    omitted = len(value) - 2000
    while True:
        marker = f"\n[… {omitted} chars omitted …]\n"
        kept = 2000 - len(marker)
        corrected = len(value) - kept
        if omitted == corrected:
            break
        omitted = corrected
    head = (2 * kept + 2) // 3
    return value[:head] + marker + value[-(kept - head):]


def without_fences(text):
    lines = []
    fence = None
    for line in text.split("\n"):
        match = MARKDOWN_FENCE.match(line)
        if fence:
            if match and match[1][0] == fence[0] and len(match[1]) >= len(fence) and not match[2].strip():
                fence = None
            lines.append("")
        elif match and (match[1][0] != "`" or "`" not in match[2]):
            fence = match[1]
            lines.append("")
        else:
            lines.append(line)
    return "\n".join(lines)


def mask_ranges(text, ranges):
    pieces = []
    offset = 0
    for start, stop in sorted(ranges):
        if stop <= offset:
            continue
        if start > offset:
            pieces.append(text[offset:start])
        pieces.append(" ")
        offset = stop
    pieces.append(text[offset:])
    return "".join(pieces)


def prose_only(text):
    text = without_fences(text)
    ticks = list(re.finditer(r"`+", text))
    next_tick, closing = {}, {}
    for index in reversed(range(len(ticks))):
        length = len(ticks[index][0])
        if length in next_tick:
            closing[index] = next_tick[length]
        next_tick[length] = index
    ranges = []
    index = 0
    while index < len(ticks):
        if (end := closing.get(index)) is not None:
            ranges.append((ticks[index].start(), ticks[end].end()))
            index = end + 1
        else:
            index += 1
    text = mask_ranges(text, ranges)
    ranges = []
    opened = {}
    for tag in MARKUP.finditer(text):
        ranges.append(tag.span())
        if tag[2] is None or tag[0].endswith("/>"):
            continue
        stack = opened.setdefault(tag[2].lower(), [])
        if not tag[1]:
            stack.append(tag.start())
        elif stack:
            ranges.append((stack.pop(), tag.end()))
    return mask_ranges(text, ranges)


def keyword(text, name):
    if name not in text:
        return False
    return re.search(rf"(?<![\w./\\-])(?<!::){name}(?![\w/\\-])(?!\.[\w-])(?!\()", prose_only(text)) is not None


def resolve(judge, model, request, ceiling, *, solution_space=None, ultrathink=True, chillax=True):
    if not (levels := supported(model)):
        return None
    if chillax and (model.get("compat") or {}).get("supportsPerMessageEffort") and keyword(request, "chillax"):
        return levels[0]
    if ultrathink and keyword(request, "ultrathink"):
        return clamp(model, "max")
    pool = tuple(level for level in eligible(model, ceiling) if level != "minimal")
    if not pool:
        return clamp(model, "low", "xhigh")
    if len(pool) == 1:
        return pool[0] if pool[0] != "max" or next(judge().available(), None) is not None else None
    solution_space = (solution_space or "").strip()
    state = {"solution_space" if solution_space else "request": preprocess(solution_space or request)}
    rubric = QUESTIONS["level"]
    instruction = QUESTIONS["solutionSpace" if solution_space else "level"]["instructions"]
    instruction = re.sub(r"{{#if withMax}}(.*?){{/if}}", r"\1" if "max" in pool else "", instruction)
    question = {"type": "choice", "instructions": instruction, "criteria": {level: rubric["criteria"][level] for level in pool}}
    return judge().ask("level", state, question, purpose="auto-thinking", seconds=4)
