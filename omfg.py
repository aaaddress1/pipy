"""Generate, replay, and save a Time Traveling Stream Rule."""

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import regex

from cancellable_http import active_turn
from ttsr import compile_rule, matching_patterns, output_state
from ttsr_rules import parse_rule, values


PROMPT = (Path(__file__).with_name("prompts") / "omfg-user.md").read_text(encoding="utf-8")
ATTEMPTS = 3
JUDGED_OUTPUTS = 8


@dataclass(frozen=True)
class Generate:
    prompt: str
    attempt: int


@dataclass(frozen=True)
class Validate:
    item: object
    content: str
    attempt: int


@dataclass(frozen=True)
class Ask:
    prompt: str
    content: str
    status: str


def prompt(complaint, feedback, previous):
    prefix, _, suffix = PROMPT.partition("{{#if feedback}}")
    followup, _, ending = suffix.partition("{{/if}}")
    if feedback:
        followup = re.sub(r"{{(feedback|previousRule)}}",
                          lambda match: {"feedback": "\n\n".join(feedback), "previousRule": previous}[match[1]],
                          followup)
    return prefix.replace("{{complaint}}", complaint) + (followup if feedback else "") + ending


def candidate_json(text):
    fence = re.search(r"\x60{3}(?:json)?\s*(.*?)\x60{3}", text, re.I | re.S)
    for source in (fence[1], text) if fence else (text,):
        start = source.find("{")
        if start < 0:
            continue
        depth, in_string, escaped = 0, False, False
        for index in range(start, len(source)):
            char = source[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return source[start:index + 1]
    raise ValueError("Missing generated rule JSON object.")


def candidate(data):
    def field(key):
        value = data.get(key)
        return value.strip() if isinstance(value, str) else ""

    raw_name = re.sub(r"['\"\x60]", "", field("name").lower())
    name = re.sub(r"-+", "-", re.sub(r"[^a-z0-9_-]+", "-", raw_name)).strip("-_")
    description, body = field("description") or field("desc"), field("body")
    scope = values(data.get("scope"))
    if not (name and description and body and scope):
        raise ValueError("Expected a valid name, description, scope and body.")
    patterns = values(data.get("condition")) or values(data.get("cond"))
    ast, question = values(data.get("astCondition")), field("question")
    if not (patterns or ast or question):
        raise ValueError("Expected condition, astCondition or question.")
    normalized = []
    for pattern in patterns:
        try:
            regex.compile(pattern)
        except regex.error:
            pattern = pattern.replace("\\\\", "\\")
            try:
                regex.compile(pattern)
            except regex.error as error:
                raise ValueError(f"Invalid regex condition {pattern!r}: {error}") from None
        normalized.append(pattern)
    return render(name, description, body, scope, normalized, ast, question)


def render(name, description, body, scope, patterns, ast, question):
    def scalar_or_array(items):
        return items[0] if len(items) == 1 else list(items)

    meta = {"name": name, "description": description, "condition": scalar_or_array(patterns),
            "astCondition": scalar_or_array(ast), "question": question, "scope": scalar_or_array(scope)}
    lines = ["---", *(f"{key}: {json.dumps(value, ensure_ascii=False)}" for key, value in meta.items() if value),
             "---", "", body.replace("\r\n", "\n").replace("\r", "\n"), ""]
    content = "\n".join(lines)
    item = compile_rule(parse_rule(content, Path(name + ".md"), "native"))
    if item is None:
        raise ValueError("Rule has no valid trigger or reachable scope.")
    return item, content


def output_subject(source, tool, paths):
    if source == "tool":
        return f"{tool} call on {paths[0]}" if paths else f"{tool} call"
    return "reasoning" if source == "thinking" else "reply"


def feedback_excerpt(content, hints):
    excerpt = re.sub(r"\s+", " ", content)
    if len(excerpt) <= 260:
        return excerpt
    lower = excerpt.lower()
    index = min((offset for hint in hints if (offset := lower.find(hint)) >= 0), default=-1)
    if index < 0:
        return excerpt[:260] + "…"
    start, end = max(0, index - 120), index + 140
    return ("…" if start else "") + excerpt[start:end] + ("…" if end < len(excerpt) else "")


def no_match_feedback(rule, outputs):
    triggers = {"condition": rule.conditions, "astCondition": rule.ast, "question": rule.question}
    trigger = " / ".join(f"{key} {json.dumps(value)}" for key, value in triggers.items() if value)
    scope = json.dumps(rule.scopes)
    lead = (f"No assistant output within scope {scope} passed {trigger}. The question was never asked."
            if rule.question else f"No assistant output matched {trigger} within scope {scope}.")
    if not outputs:
        return lead + "\nNo assistant replies, reasoning, or tool calls were available to check."
    hints = {word.lower() for pattern in (*rule.conditions, *rule.ast)
             for word in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", pattern)}
    hints -= {"tool", "text", "any", "true", "false"}
    samples = [f"- {output_subject(source, tool, paths)}: {json.dumps(feedback_excerpt(content, hints))}"
               for source, tool, content, paths, _ in outputs[:5]]
    if len(outputs) > 5:
        samples.append(f"- … {len(outputs) - 5} more output(s)")
    return "\n".join([lead, "Checked outputs:", *samples,
                      "Edit and write calls are checked as source text. Other tool calls use serialized JSON arguments.",
                      "If the trigger looks right, fix the scope so it reaches the offending tool and file glob."])


def replay(item, outputs, judge):
    rule = item.rule
    signal = active_turn()
    matches = []
    for source, tool, content, paths, ast in outputs:
        if signal:
            signal.check()
        if not item.accepts(source, tool, paths):
            continue
        patterns = matching_patterns(item, content, paths, ast=ast)
        if any(patterns.values()) or rule.question and not (item.patterns or rule.ast):
            matches.append((source, tool, content, paths, ast))
    if not matches:
        return False, False, no_match_feedback(rule, outputs)
    if rule.question:
        recent = matches[-JUDGED_OUTPUTS:]
        with ThreadPoolExecutor(max_workers=len(recent), thread_name_prefix="mini-pi-omfg") as workers:
            pending = [workers.submit(copy_context().run, judge, output_state(source, tool, content), rule)
                       for source, tool, content, _, _ in recent]
            results = [future.result() for future in pending]
        if signal:
            signal.check()
        if any(result is None for result in results):
            return False, True, "The rule judge is unavailable."
        if not any(results):
            subjects = ", ".join(output_subject(source, tool, paths) for source, tool, _, paths, _ in recent)
            return False, False, (f"The judge answered no to {rule.question!r} for each checked output ({subjects}). "
                                  "Rephrase the question or use a condition if a literal or structural pattern can catch the output.")
        return True, False, ""
    if {"text", "tool", "toolcall"} & {token.lower() for token in rule.scopes} or not rule.scopes:
        for source, tool, _, paths, _ in matches:
            if source != "tool":
                continue
            for path in paths:
                extension = Path(path.replace("\\", "/")).suffix.lower()
                if extension:
                    return False, False, f"The trigger matched {tool} on {path}. Narrow its scope to tool:{tool}(*{extension})."
    return True, False, ""


def validate(item, content, outputs, judge):
    if callable(outputs):
        outputs = outputs()
    matched, unavailable, feedback = replay(item, outputs, judge)
    rule = item.rule
    repaired = tuple(pattern.replace("\\\\", "\\") for pattern in rule.conditions)
    if not matched and repaired != rule.conditions:
        try:
            alternate, alternate_content = render(rule.name, rule.description, rule.content, rule.scopes,
                                                   repaired, rule.ast, rule.question)
        except ValueError:
            pass
        else:
            if len(alternate.patterns) == len(repaired) and replay(alternate, outputs, judge)[0]:
                return alternate, alternate_content, True, False, ""
    return item, content, matched, unavailable, feedback


def generate(complaint, amendment=None, previous=""):
    failures = [f"User requested this amendment before saving:\n{amendment}"] if amendment else []
    last = None
    for attempt in range(1, ATTEMPTS + 1):
        raw = yield Generate(prompt(complaint, failures, previous), attempt)
        previous = raw.strip()
        try:
            previous = candidate_json(raw)
            item, content = candidate(json.loads(previous))
        except ValueError as error:
            failures.append(f"Attempt {attempt} returned an invalid rule ({error}).\nFailed candidate:\n{previous}")
            continue
        item, content, matched, unavailable, feedback = yield Validate(item, content, attempt)
        last = item.rule.name, content, matched
        if matched or unavailable:
            return last
        previous = content
        failures.append(f"Attempt {attempt} failed validation:\n{feedback}\nFailed candidate:\n{content}")
    if last is None:
        raise ValueError("The model did not return a valid TTSR rule.")
    return last


def save(target, content, overwrite):
    target.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=target.parent, delete=False) as file:
        temporary = Path(file.name)
        try:
            file.write(content)
            file.close()
            if overwrite:
                os.replace(temporary, target)
            elif os.name == "nt":
                os.rename(temporary, target)
            else:
                os.link(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)


def workflow(complaint, workspace, user):
    amendment, previous = None, ""
    while True:
        name, content, matched = yield from generate(complaint, amendment, previous)
        if not matched:
            choice = yield Ask("Couldn't confirm a conversation match. Save anyway? [y/N] ", content, "Confirm validation")
            if choice.strip().lower() not in ("y", "yes"):
                return
        while True:
            selection = yield Ask("Save rule to [p]roject, [g]lobal, [a]mend, or [Enter] to cancel? ", content, "Choose where to save")
            selection = selection.strip().lower()
            if not selection:
                return
            if selection in ("a", "amend"):
                amendment = yield Ask("Describe the amendment: ", content, "Amend the rule")
                amendment = amendment.strip()
                if amendment:
                    previous = content
                    break
            if selection in ("p", "project", "g", "global"):
                directory = workspace / ".omp" if selection in ("p", "project") else user
                target = directory / "rules" / (name + ".md")
                overwrite = target.exists()
                if overwrite:
                    choice = yield Ask(f"Overwrite {target}? [y/N] ", content, "Confirm overwrite")
                    if choice.strip().lower() not in ("y", "yes"):
                        return
                save(target, content, overwrite)
                return target


def run(complaint, workspace, user, ask_model, outputs, judge, ask):
    if not complaint:
        print("Usage: /omfg <complaint>")
        return
    flow = workflow(complaint, workspace, user)
    result, shown = None, None
    try:
        while True:
            action = flow.send(result)
            if isinstance(action, Generate):
                print(f"/omfg attempt {action.attempt}/{ATTEMPTS} — generating…")
                shown = None
                result = ask_model(action.prompt)
            elif isinstance(action, Validate):
                result = validate(action.item, action.content, outputs, judge)
            else:
                if action.content != shown:
                    shown = action.content
                    print(shown, end="")
                result = ask(action.prompt)
    except StopIteration as done:
        if done.value:
            print(f"Saved TTSR rule to {done.value}")
        return done.value
    finally:
        flow.close()
