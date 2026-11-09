"""Thinking-loop detector ported from Oh My Pi (see LICENSE.oh-my-pi)."""

import os
from collections import deque
from pathlib import Path

import regex


REDIRECT = (Path(__file__).with_name("prompts") / "thinking-loop-redirect.md").read_text(encoding="utf-8").rstrip()
TAIL = 4096
MAX_PERIOD = 1024
SCAN_STRIDE = 128
SEGMENT_CAP = 700
PARAGRAPH = regex.compile(r"\n\s*\n")
CODE = regex.compile(
    r"^(?:[ \t]*`{3}.*|(?: {2,}|\t)(?![ \t]*(?:[-*+]|\d+[.)])[ \t]).*|.*[{}\[\](;,>][ \t]*)$",
    regex.M | regex.ASCII,
)
TITLE = regex.compile(r"^[ \t]*(?:#{1,6}[ \t].*|\*{2,3}.+?\*{2,3}[ \t]*)$", regex.M)
ANCHOR = regex.compile(
    r"\x60[^\x60]+\x60|\b\w{2,}\.[a-zA-Z]\w{0,4}\b|[\w-]+(?:/[\w-]+){2,}"
    r"|\b\w+_\w+\b|\b[a-z]+[A-Z]\w*\b|\b[A-Z][a-z]+[A-Z]\w*\b",
    regex.ASCII,
)
LETTER_OR_EMOJI = regex.compile(r"\p{L}|\p{Extended_Pictographic}")
NUMERIC_OR_SPACE = regex.compile(r"[\p{N}\s]*")


def utf16_units(text):
    """Count and compare the same UTF-16 code units as the TypeScript detector."""
    raw = text.encode("utf-16-le", "surrogatepass")
    return "".join(chr(low | high << 8) for low, high in zip(raw[::2], raw[1::2]))


def unicode_text(units):
    return units.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "surrogatepass")


def exact_cycle(text):
    if len(text) < 180:
        return None
    reversed_text = text[::-1]
    max_period = min(MAX_PERIOD, len(text) // 3)
    z = [0] * (max_period + 1)
    left, right = 0, 0
    for index in range(1, max_period + 1):
        match = min(right - index + 1, z[index - left]) if index <= right else 0
        while index + match < len(text) and reversed_text[match] == reversed_text[index + match]:
            match += 1
        z[index] = match
        if index + match - 1 > right:
            left, right = index, index + match - 1

    for period in range(2, max_period + 1):
        repeats = 1 + z[period] // period
        short = period <= 60
        if repeats < (4 if short else 3):
            continue
        unit = unicode_text(text[-period:])
        if LETTER_OR_EMOJI.search(unit):
            minimum = 180 if short and "\n" not in unit else 1024
        elif NUMERIC_OR_SPACE.fullmatch(unit):
            continue
        else:
            minimum = 1024
        if period * repeats >= minimum:
            return f"repeated an exact {period}-character cycle {repeats}× back-to-back"
    return None


class Detector:
    def __init__(self, semantic):
        self.semantic = semantic
        self.tail = ""
        self.since_scan = 0
        self.pending = ""
        self.segments = deque(maxlen=16)
        self.vocabulary = deque(maxlen=8)
        self.anchors = deque(maxlen=8)
        self.count = 0
        self.stall_run = 0

    def push(self, delta):
        if not delta:
            return None
        delta = utf16_units(delta)
        self.tail = (self.tail + delta)[-TAIL:]
        self.since_scan += len(delta)
        if self.since_scan >= SCAN_STRIDE:
            self.since_scan = 0
            reason = exact_cycle(self.tail)
            if reason:
                return reason
        if not self.semantic:
            return None

        self.pending += delta
        while self.pending:
            boundary = PARAGRAPH.search(self.pending)
            if boundary:
                segment, self.pending = self.pending[:boundary.start()], self.pending[boundary.end():]
            elif len(self.pending) > SEGMENT_CAP:
                segment, self.pending = self.pending[:SEGMENT_CAP], self.pending[SEGMENT_CAP:]
            else:
                break
            reason = self.consume_chunks(segment)
            if reason:
                return reason
        return None

    def flush(self):
        reason = exact_cycle(self.tail)
        if reason or not self.semantic:
            return reason
        segment, self.pending = self.pending, ""
        return self.consume_chunks(segment)

    def consume_chunks(self, segment):
        for start in range(0, len(segment), SEGMENT_CAP):
            reason = self.consume(unicode_text(segment[start:start + SEGMENT_CAP]))
            if reason:
                return reason
        return None

    def consume(self, segment):
        segment = TITLE.sub("", CODE.sub("", segment))
        words = [
            word for word in regex.findall(r"[a-z0-9]+", segment.lower())
            if regex.search("[a-z]", word)
        ]
        if len(" ".join(words)) < 60:
            return None
        fingerprint = set(zip(words, words[1:], words[2:])) if len(words) >= 3 else {tuple(words)}
        cluster = 1 + sum(
            len(fingerprint & previous) / len(fingerprint | previous) >= 0.8
            for previous in self.segments
        )
        unique_words = set(words)
        previous_words = set().union(*self.vocabulary)
        novelty = len(unique_words - previous_words) / len(unique_words) if previous_words else 1
        anchors = {anchor.replace("`", "").lower() for anchor in ANCHOR.findall(segment)}
        if novelty <= 0.2 and not anchors.difference(*self.anchors):
            self.stall_run += 1
        else:
            self.stall_run = 0

        self.segments.append(fingerprint)
        self.vocabulary.append(unique_words)
        self.anchors.append(anchors)
        self.count += 1
        if self.count >= 8:
            if cluster >= 4:
                return f"{cluster} near-identical segments within the last 16"
            if self.stall_run >= 8:
                return f"{self.stall_run} low-information segments recycling recent wording"
        return None


class ThinkingLoop(RuntimeError):
    error_id = 0x00010000


class StreamGuard:
    def __init__(self, level, check_text):
        self.reasoning = Detector(level == "semantic") if level != "off" else None
        self.text = Detector(level == "semantic") if self.reasoning and check_text else None

    def push(self, reasoning, text, has_tool_call):
        if self.reasoning and reasoning:
            self.check(self.reasoning.push(reasoning))
        if text:
            if self.reasoning:
                self.check(self.reasoning.flush())
                self.reasoning = None
            if self.text:
                self.check(self.text.push(text))
        if has_tool_call:
            self.text = None

    def finish(self):
        if self.reasoning:
            self.check(self.reasoning.flush())
        if self.text:
            self.check(self.text.flush())

    @staticmethod
    def check(detail):
        if detail:
            raise ThinkingLoop(f"Thinking loop detected ({detail}).")


def loop_level(model, configured):
    if os.getenv("PI_NO_THINKING_LOOP_GUARD") == "1":
        return "off"
    if configured != "auto":
        return configured
    identity = model.lower()
    if regex.search(r"(?:^|[/_.:-])claude(?:$|[/_.:-]|\d)", identity):
        return "off"
    if regex.search(r"(?:^|[/_.:-])(?:gemini|deepseek|grok)(?:$|[/_.:-]|\d)", identity):
        return "semantic"
    return "exact"
