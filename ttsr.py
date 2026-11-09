"""Stream rules that interrupt or annotate a model response."""

import json
import re
import sys
import time
from concurrent.futures import CancelledError, wait
from dataclasses import dataclass
from html import escape
from pathlib import Path
from threading import Lock

import regex
from ast_grep_py import SgRoot
from pydantic_core import from_json

from ttsr_rules import MODES, discover, extension_roots, glob_matches, values


TOOL_SCOPE = re.compile(r"(?:tool(?::([\w-]+))?|([\w-]+))(?:\(([^)]+)\))?\Z", re.I)
INLINE_FLAGS = re.compile(r"^\(\?[ims]+\)")
LANGUAGES = {".ts": "typescript", ".tsx": "tsx", ".js": "javascript", ".jsx": "javascript",
             ".mjs": "javascript", ".cjs": "javascript", ".py": "python", ".pyi": "python",
             ".go": "go", ".rs": "rust", ".java": "java", ".kt": "kotlin", ".c": "c", ".h": "c",
             ".cpp": "cpp", ".cc": "cpp", ".hpp": "cpp", ".cs": "csharp", ".rb": "ruby",
             ".swift": "swift", ".lua": "lua", ".sh": "bash", ".bash": "bash", ".php": "php",
             ".html": "html", ".css": "css", ".json": "json", ".yaml": "yaml", ".yml": "yaml"}
HARNESS = re.compile(r"<(?=\s*/?\s*system-(?:interrupt|reminder)\b)", re.I)


def tool_name(name):
    name = name.lower()
    return "shell" if name in ("bash", "exec") else name


def line_local(pattern):
    flags = INLINE_FLAGS.match(pattern)
    body = pattern[flags.end():] if flags else pattern
    return not ((flags and "s" in flags[0]) or "\n" in body or "\r" in body
                or re.search(r"\\[nrsWDSuUxpcR]|\[\^|\(\?", body))


@dataclass(frozen=True)
class Compiled:
    rule: object
    patterns: tuple
    prose: frozenset
    tools: tuple

    def accepts(self, source, name, paths):
        if self.rule.paths and not any(glob_matches(pattern, path) for pattern in self.rule.paths for path in paths):
            return False
        if source != "tool":
            return source in self.prose
        return any((not tool or tool == name) and (not glob or any(glob_matches(glob, path) for path in paths))
                   for tool, glob in self.tools)


def compile_rule(rule):
    patterns = []
    for source in rule.conditions:
        try:
            patterns.append((regex.compile(source), line_local(source)))
        except regex.error:
            print(f"[TTSR: ignored invalid condition in {rule.name}]", file=sys.stderr)
    prose, tools = set(), []
    for token in rule.scopes or ("text", "tool"):
        lowered = token.lower()
        if lowered in ("text", "thinking"):
            prose.add(lowered)
        elif lowered in ("tool", "toolcall"):
            tools.append(("", None))
        elif match := TOOL_SCOPE.fullmatch(token):
            tools.append((tool_name(match[1] or match[2] or ""), match[3]))
    if (patterns or rule.ast or rule.question) and (prose or tools):
        return Compiled(rule, tuple(patterns), frozenset(prose), tuple(tools))


@dataclass(frozen=True)
class Match:
    rule: object
    source: str
    excerpt: str = ""


class RuleInterrupt(RuntimeError):
    def __init__(self, hits, partial):
        super().__init__("Stream interrupted by a workspace rule.")
        self.hits, self.partial = hits, partial


class Rules:
    def __init__(self, workspace, user, settings, agent, restored=(), plugins=()):
        self.workspace, self.user, self.agent, self.plugins = workspace, user, agent, plugins
        self.age, self.injected = 0, dict.fromkeys(restored, 0)
        self.last = None
        self.response = None
        self.judgments, self.judge_lock = {}, Lock()
        self.configure(settings)
        self.refresh()

    def configure(self, settings):
        self.settings = settings
        self.enabled = settings.get("enabled") is not False
        self.builtin = settings.get("builtinRules") is not False
        self.disabled = set(values(settings.get("disabledRules")))
        self.interrupt = settings.get("interruptMode")
        if self.interrupt not in MODES:
            self.interrupt = "always"
        self.keep = settings.get("contextMode") == "keep"
        self.repeat = settings.get("repeatMode") == "after-gap"
        self.gap = settings.get("repeatGap", 10)
        if type(self.gap) is not int or self.gap < 1:
            self.gap = 10
        self.judge_mode = settings.get("judge", "auto")
        if isinstance(self.judge_mode, bool):
            self.judge_mode = "on" if self.judge_mode else "off"

    def refresh(self, settings=None, plugins=None):
        changed = settings is not None and (settings != self.settings or plugins != self.plugins)
        if changed:
            self.configure(settings)
            self.plugins = plugins
            if not self.enabled or self.judge_mode == "off":
                self.close_judgments()
        loaded = tuple(discover(self.workspace, self.user, extension_roots(self.workspace, self.user, self.plugins)))
        if not changed and loaded == self.last:
            return False
        self.last = loaded
        rules, always, book, triggers = {}, [], [], []
        for rule in loaded:
            if rule.name in self.disabled or rule.source == "builtin-defaults" and not self.builtin:
                continue
            if not applies_to_agent(rule, self.agent):
                continue
            trigger = compile_rule(rule) if self.enabled else None
            if trigger:
                triggers.append(trigger)
            elif rule.always:
                always.append(rule.content)
            elif rule.description:
                book.append(f"- rule://{rule.name} — {rule.description}")
            else:
                continue
            rules[rule.name] = rule
        if self.enabled:
            previous = {item.rule.name for item in getattr(self, "triggers", ())}
            removed = previous - {item.rule.name for item in triggers}
            for name in removed:
                self.injected.pop(name, None)
        self.rules, self.triggers = rules, tuple(triggers)
        self.prose_sources = {kind for item in triggers if item.patterns and not item.rule.question for kind in item.prose}
        self.stream_tools = any(item.tools and item.patterns and not item.rule.question for item in triggers)
        self.judge_enabled = self.judge_mode != "off" and any(item.rule.question for item in triggers)
        self.prompt = "\n\n".join(filter(None, [
            "<sticky-rules>\n" + "\n\n".join(always) + "\n</sticky-rules>" if always else "",
            "## Rulebook\nRead applicable rules using their rule:// references.\n" + "\n".join(book) if book else "",
        ]))
        return True

    def eligible(self, rule):
        return rule.name not in self.injected or self.repeat and self.age - self.injected[rule.name] >= self.gap

    def claim(self, hits):
        for hit in hits:
            self.injected[hit.rule.name] = self.age

    def message(self, hits, kind):
        self.claim(hits)
        tag = "system-interrupt" if kind == "interrupt" else "system-reminder"
        blocks = []
        for hit in hits:
            rule = hit.rule
            attributes = f' reason="rule_violation" rule="{escape(rule.name)}" path="{escape(rule.display_path(self.workspace))}"'
            if kind == "interrupt":
                lead = "Output interrupted by a workspace rule. Follow this instruction on retry."
            elif kind == "judged":
                lead = f"The rule judge flagged your {hit.source} output. Check the rule and correct a real violation."
            elif kind == "tool":
                lead = ("A built-in style rule matched. Check its exceptions and correct real violations."
                        if rule.source == "builtin-defaults" else "The tool ran. Follow this rule in subsequent calls and responses.")
            else:
                lead = "A workspace rule matched the response. Correct the output if the rule applies."
            excerpt = f"\nMatched excerpt (quoted output): {escape(hit.excerpt)}" if hit.excerpt and kind == "interrupt" else ""
            blocks.append(f"<{tag}{attributes}>\n{lead}{excerpt}\n\n{rule.content}\n</{tag}>")
        return {"role": "developer", "content": "\n\n".join(blocks), "_ttsr_rules": [hit.rule.name for hit in hits],
                "_ttsr_kind": kind}

    def watch_judgment(self, future, cancellation, wake):
        with self.judge_lock:
            self.judgments[future] = cancellation
        if wake:
            future.add_done_callback(lambda _: wake())

    def judgment_ready(self):
        with self.judge_lock:
            return any(future.done() for future in self.judgments)

    def judgment_pending(self):
        with self.judge_lock:
            return bool(self.judgments)

    def close_judgments(self):
        with self.judge_lock:
            pending = self.judgments
            self.judgments = {}
        for future, cancellation in pending.items():
            cancellation.cancel()
            future.cancel()

    def take_judgments(self, seconds=0, cancellation=None):
        if seconds:
            end = time.monotonic() + seconds
            with self.judge_lock:
                pending = {future for future in self.judgments if not future.done()}
            while pending and (remaining := end - time.monotonic()) > 0:
                if cancellation:
                    cancellation.check()
                _, pending = wait(pending, timeout=min(.1, remaining))
        with self.judge_lock:
            ready = tuple(future for future in self.judgments if future.done())
            for future in ready:
                del self.judgments[future]
        hits = {}
        for future in ready:
            try:
                for hit in future.result():
                    if self.rules.get(hit.rule.name) == hit.rule and self.eligible(hit.rule):
                        hits.setdefault(hit.rule.name, hit)
            except CancelledError:
                continue
            except Exception as error:
                print(f"[TTSR: judge failed ({type(error).__name__})]", file=sys.stderr)
        if hits:
            return self.message(tuple(hits.values()), "judged")


def arguments(raw, final):
    try:
        parsed = json.loads(raw) if final else from_json(raw, allow_partial="trailing-strings")
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, ValueError):
        return {}


def tool_output(name, raw, final=False):
    params = arguments(raw, final)
    path = params.get("path")
    paths = (path,) if isinstance(path, str) and path else ()
    if isinstance(other := params.get("paths"), list):
        paths += tuple(item for item in other if isinstance(item, str))
    fields = ("content", "text") if name == "write" else ("new", "new_string") if name == "edit" else ()
    if fields:
        source = next((params[key] for key in fields if isinstance(params.get(key), str)), "")
        return source, paths, True
    return raw, paths, False


def ast_excerpt(pattern, content, paths):
    for path in paths:
        language = LANGUAGES.get(Path(path.replace("\\", "/")).suffix.lower())
        if not language:
            continue
        try:
            node = SgRoot(content, language).root().find(pattern=pattern)
            if node is not None:
                return node.text()[:240] or pattern[:240]
        except (TypeError, ValueError, RuntimeError):
            continue


def applies_to_agent(rule, agent):
    return agent is None or not rule.agents or any(glob_matches(pattern, agent.lower()) for pattern in rule.agents)


def matching_patterns(item, content, paths=(), *, ast=False):
    found = []
    for pattern, _ in item.patterns:
        try:
            if pattern.search(content, timeout=.03):
                found.append(pattern.pattern)
        except TimeoutError:
            continue
    return {"regex": found, "ast": [pattern for pattern in item.rule.ast
                                    if ast and ast_excerpt(pattern, content, paths) is not None]}


def assistant_outputs(reply, buffers=None):
    buffers = buffers or {}
    for source, field in (("text", "content"), ("thinking", "_thinking")):
        content = buffers.get(source) or reply.get(field)
        if source == "thinking" and not content:
            content = "".join(block.get("thinking", "") for block in reply.get("_anthropic", {}).get("blocks", ())
                              if block.get("type") == "thinking")
        if isinstance(content, str) and content:
            yield source, "", content, (), False
    for call in reply.get("tool_calls", ()) or ():
        function = call["function"]
        name = tool_name(function["name"])
        content, paths, ast = tool_output(name, function["arguments"], final=True)
        if content:
            yield "tool", name, content, paths, ast


def output_state(source, name, content):
    return {"output": name if source == "tool" else source,
            "content": content.encode("utf-8")[:96_000].decode("utf-8", errors="ignore")}


def judge_question(rule):
    return {"type": "noul", "instructions": rule.question,
            "criteria": {"true": "The output violates the rule.", "false": "The output is consistent with the rule."}}


class Attempt:
    def __init__(self, rules):
        self.rules = rules
        self.buffers, self.scans, self.bound = {}, {}, {}
        self.prose_hits, self.tool_hits = {}, {}

    def restart(self):
        self.buffers.clear()
        self.scans.clear()
        self.bound.clear()
        self.prose_hits.clear()
        self.tool_hits.clear()

    def text(self, kind, piece):
        if not piece or (kind not in self.rules.prose_sources and not self.rules.keep):
            return
        merged = self.buffers.get(kind, "") + piece
        self.buffers[kind] = merged
        if kind in self.rules.prose_sources:
            self.scan(merged, kind, kind)

    def end_text(self, kind):
        if kind in self.rules.prose_sources and (content := self.buffers.get(kind)):
            self.scan(content, kind, kind, final=True)

    def tool(self, slot, name, raw, identifier=None, *, final=False, ast=False):
        if not name or not raw:
            return
        if identifier:
            old = self.bound.get(slot, slot)
            if old != identifier and old in self.tool_hits:
                self.tool_hits.setdefault(identifier, {}).update(self.tool_hits.pop(old))
            self.bound[slot] = identifier
        name = tool_name(name)
        data, paths, is_source = tool_output(name, raw, final)
        if data or is_source:
            self.scan(data, "tool", slot, name, paths, final=final, ast=ast and is_source,
                      tool_key=self.bound.get(slot, slot), replace=is_source)

    def scan(self, content, source, key, name="", paths=(), *, final=False, ast=False, tool_key=None, replace=False):
        hits = []
        for item in self.rules.triggers:
            rule = item.rule
            if rule.question or not self.rules.eligible(rule) or not item.accepts(source, name, paths):
                continue
            match = self.regex_match(item, content, key, final, replace)
            if match is None and ast:
                match = next((value for pattern in rule.ast if (value := ast_excerpt(pattern, content, paths)) is not None), None)
            if match is not None:
                hits.append(Match(rule, source, match))
        if not hits:
            return
        if any(self.interrupts(hit) for hit in hits):
            combined = {**self.prose_hits, **{hit.rule.name: hit for hit in hits}}
            partial = {"role": "assistant", "content": self.buffers.get("text") or None}
            if thinking := self.buffers.get("thinking"):
                partial["_thinking"] = thinking
            raise RuleInterrupt(tuple(combined.values()), partial)
        if source == "tool":
            self.rules.claim(hits)
            self.tool_hits.setdefault(tool_key, {}).update((hit.rule.name, hit) for hit in hits)
        else:
            self.prose_hits.update((hit.rule.name, hit) for hit in hits)

    def regex_match(self, item, content, key, final, replace):
        for index, (pattern, local) in enumerate(item.patterns):
            tracker = (key, item.rule.name, index)
            previous = self.scans.get(tracker, 0)
            length = len(content)
            if not final and not local and length > 4096 and previous and length < previous * 1.25 and length >= previous:
                continue
            start = 0
            if local and not final and not replace and previous and length >= previous:
                start = content.rfind("\n", 0, previous) + 1
            self.scans[tracker] = length
            try:
                if found := pattern.search(content, pos=start, timeout=.03):
                    return found.group(0)[:240]
            except TimeoutError:
                continue

    def interrupts(self, hit):
        mode = hit.rule.mode or self.rules.interrupt
        return mode == "always" or mode == "tool-only" and hit.source == "tool" or (
            mode == "prose-only" and hit.source != "tool")

    def finish(self, reply):
        for kind, field in (("text", "content"), ("thinking", "_thinking")):
            content = reply.get(field) or self.buffers.get(kind) or ""
            if kind == "thinking" and not content:
                content = "".join(block.get("thinking", "") for block in reply.get("_anthropic", {}).get("blocks", ())
                                  if block.get("type") == "thinking")
            if content:
                self.buffers[kind] = content
                self.scan(content, kind, kind, final=True)
        for call in reply.get("tool_calls", ()):
            func = call["function"]
            self.tool(call["id"], func["name"], func["arguments"], call["id"], final=True, ast=True)
        return self

    def judged(self, reply):
        for index, (source, name, content, paths, ast_source) in enumerate(assistant_outputs(reply, self.buffers)):
            hits = []
            for item in self.rules.triggers:
                rule = item.rule
                if not rule.question or not self.rules.eligible(rule) or not item.accepts(source, name, paths):
                    continue
                if item.patterns or rule.ast:
                    match = self.regex_match(item, content, ("judge", index), True, False) is not None
                    if not match and not (ast_source and any(ast_excerpt(pattern, content, paths) is not None for pattern in rule.ast)):
                        continue
                hits.append(Match(rule, source))
            if hits:
                yield output_state(source, name, content), tuple(hits)
