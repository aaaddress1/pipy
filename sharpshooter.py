"""Friction-gated project decision memory, using Oh My Pi's file and tool contracts."""

import json
import os
import re
import time
from concurrent.futures import CancelledError, TimeoutError
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, RLock, Thread
from uuid import uuid4

import regex

from handoff import Cancellation
from memory import redact_memory
from memory_files import atomic_text
from mnemopi_store import base36, project_bank
from reset_fence import try_lock
from runtime import pack
from session_jsonl import plain_text


PROMPTS = Path(__file__).with_name("prompts")
FILES = ("architecture.md", "product.md", "style.md")
KINDS = ("architecture_decision", "product_decision", "style_decision", "constraint", "rejected_approach", "correction")
SOURCES = ("explicit_user", "contextual_resolution")
FRICTION = ("corrective", "regression", "subtle")
GRAPHEME = regex.compile(r"\X")
SESSION_DIR = re.compile(r"[A-Za-z0-9_-]+\Z")


def object_schema(properties, required):
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


def array_schema(name, item):
    return object_schema({name: {"type": "array", "items": item}}, [name])


STRING = {"type": "string"}
EXTRACT = array_schema("deltas", object_schema({
    "kind": {"type": "string", "enum": KINDS}, "statement": STRING,
    "rejectedAlternative": STRING, "rationale": STRING,
    "source": {"type": "string", "enum": SOURCES}, "evidence": STRING,
    "friction": object_schema(dict.fromkeys(FRICTION, {"type": "boolean"}), FRICTION),
}, ("kind", "statement", "source", "evidence", "friction")))
CONSOLIDATE = array_schema("files", object_schema({
    "name": {"type": "string", "enum": FILES}, "content": STRING,
}, ("name", "content")))


def milliseconds():
    return time.time_ns() // 1_000_000


def render(name, **variables):
    template = (PROMPTS / ("sharpshooter-" + name + ".md")).read_text("utf-8")
    template = re.sub(r"\{\{#if (\w+)\}\}\n?(.*?)\{\{/if\}\}\n?",
                      lambda match: match[2] if variables.get(match[1]) else "", template, flags=re.S)
    return re.sub(r"\{\{(\w+)\}\}", lambda match: str(variables[match[1]]), template)


def token_clip(text, tokens):
    limit = max(0, tokens * 4)
    return text if len(text) <= limit else text[:max(0, limit - 1)].rstrip() + "…"


def read_text(file):
    try:
        return file.read_text("utf-8") if not file.is_symlink() else ""
    except (OSError, UnicodeError):
        return ""


def context_text(message, limit):
    content = plain_text(message.get("content"))
    return " ".join(re.sub(r"(```|~~~).*?\1", "[code omitted]", content, flags=re.S).split())[:limit]


def admitted_delta(value, prompt, session_id):
    if not isinstance(value, dict):
        return None
    statement, evidence = value.get("statement"), value.get("evidence")
    kind, source, friction = value.get("kind"), value.get("source"), value.get("friction")
    if (not isinstance(statement, str) or not statement.strip() or not isinstance(evidence, str)
            or not evidence or evidence not in prompt or kind not in KINDS or source not in SOURCES
            or not isinstance(friction, dict) or any(type(friction.get(key)) is not bool for key in FRICTION)):
        return None
    optional = {key: redact_memory(value[key].strip()) for key in ("rejectedAlternative", "rationale")
                if isinstance(value.get(key), str) and value[key].strip()}
    return {"v": 1, "kind": kind, "statement": redact_memory(statement.strip()), **optional, "source": source,
            "evidence": redact_memory(evidence), "friction": {key: friction[key] for key in FRICTION},
            "sessionId": session_id, "ts": milliseconds()}


def replacement_files(calls, current):
    if len(calls) != 1 or calls[0][0] != "replace_memory_files":
        raise ValueError("Sharpshooter requires exactly one replace_memory_files call.")
    values = calls[0][1]
    if not isinstance(values, dict) or not isinstance(values.get("files"), list):
        raise ValueError("replace_memory_files requires a files array.")
    files = {}
    for item in values["files"]:
        if not isinstance(item, dict) or item.get("name") not in FILES or not isinstance(item.get("content"), str):
            raise ValueError("Invalid Sharpshooter memory file.")
        name, content = item["name"], redact_memory(item["content"])
        if name in files:
            raise ValueError(f"Duplicate Sharpshooter memory file {name}.")
        if content and content[:-1].count("\n") >= 120:
            raise ValueError(f"{name} exceeds the 120-line limit.")
        files[name] = content
    if not any(text.strip() for text in files.values()) and any(text.strip() for text in current.values()):
        raise ValueError("Refusing all-empty output over existing Sharpshooter memory.")
    return files


class Sharpshooter:
    def __init__(self, agent_dir, workspace, session, setting, request, executor, notice, *, primary=True):
        self.workspace, self.session = workspace, session
        self.setting, self.request, self.executor, self.notice = setting, request, executor, notice
        self.bank = agent_dir / "memories" / "sharpshooter" / project_bank(workspace)
        self.queue = self.bank / "queue"
        self.lock_path = self.bank / "consolidate.lock"
        self.write_lock = RLock()
        self.generation = 0
        self.stopping = Event()
        self.extraction_signal = Cancellation(reason="Sharpshooter extraction cancelled.")
        self.scheduler_signal = Cancellation(reason="Sharpshooter scheduler cancelled.")
        self.extraction = self.scheduler = None
        if primary:
            self.scheduler = Thread(target=self.schedule, name="mini-pi-sharpshooter", daemon=True)
            self.scheduler.start()

    def number(self, key, default):
        result = self.setting("sharpshooter." + key)
        return result if type(result) is int else default

    def read_files(self):
        return {name: read_text(self.bank / name) for name in FILES}

    def instructions(self):
        sections = [f"## {name[:-3]}\n\n{content.strip()}" for name, content in self.read_files().items() if content.strip()]
        if not sections:
            return ""
        heading = "Project decision memory (sharpshooter). These are friction-earned decisions; follow them unless the user overrides."
        return token_clip("\n\n".join([heading, *sections]), self.number("injectionTokenLimit", 15000))

    def state(self):
        try:
            value = json.loads(read_text(self.bank / "state.json"))
            if isinstance(value, dict) and value.get("v") == 1 and type(value.get("lastConsolidatedAt")) is int:
                return value
        except ValueError:
            pass
        return {"v": 1, "lastConsolidatedAt": 0}

    def groups(self):
        if not self.queue.is_dir() or self.queue.is_symlink():
            return []
        groups = []
        for directory in self.queue.iterdir():
            if not SESSION_DIR.fullmatch(directory.name) or not directory.is_dir() or directory.is_symlink():
                continue
            deltas = []
            for file in directory.glob("*.json"):
                try:
                    value = json.loads(read_text(file))
                    if (isinstance(value, dict) and value.get("v") == 1 and isinstance(value.get("statement"), str)
                            and type(value.get("ts")) is int and isinstance(value.get("friction"), dict)):
                        deltas.append((file, value))
                except ValueError:
                    continue
            if deltas:
                groups.append((directory.name, sorted(deltas, key=lambda entry: (entry[1]["ts"], entry[0].name))))
        return sorted(groups, key=lambda group: group[1][0][1]["ts"])

    def on_input(self, content):
        if not self.scheduler or self.stopping.is_set() or self.extraction and not self.extraction.done():
            return
        if isinstance(content, dict):
            content = content.get("content")
        prompt = plain_text(content).strip()
        minimum = max(2, self.number("minPromptLength", 16))
        if not prompt or prompt.startswith("/") or sum(1 for _ in GRAPHEME.finditer(prompt)) < minimum:
            return
        previous, assistant = None, None
        for _, batch in reversed(self.session.batches[:-1]):
            for message in reversed(batch):
                if message["role"] == "user" and previous is None:
                    previous = context_text(message, 400)
                if message["role"] == "assistant" and assistant is None:
                    assistant = context_text(message, 800)
            if previous is not None and assistant is not None:
                break
        self.extraction = self.executor.submit(self.extract, prompt, previous, assistant, self.generation)

    def extract(self, prompt, previous, assistant, generation):
        try:
            envelope = render("extract-input", prompt=redact_memory(prompt),
                              previousHuman=redact_memory(previous or ""), assistantContext=redact_memory(assistant or ""))
            response = self.request("record_deltas", "Record durable project decision deltas.", EXTRACT,
                                    render("extract-system"), envelope, "low", 2048, self.extraction_signal)
            if response is None:
                return
            _, calls = response
            for name, values in calls:
                if name != "record_deltas" or not isinstance(values, dict) or not isinstance(values.get("deltas"), list):
                    continue
                for value in values["deltas"]:
                    if not (delta := admitted_delta(value, prompt, self.session.id)):
                        continue
                    with self.write_lock:
                        if generation != self.generation or self.extraction_signal.is_cancelled():
                            return
                        identifier = re.sub(r"[^A-Za-z0-9_-]+", "-", self.session.id).strip("-") or "session"
                        folder = self.queue / identifier
                        folder.mkdir(parents=True, exist_ok=True)
                        file = folder / f"{base36(delta['ts']):0>10}-{uuid4().hex[:8]}.json"
                        atomic_text(file, pack(delta), durable=True)
        except CancelledError:
            return
        except Exception as error:
            self.notice(f"Sharpshooter extraction failed ({type(error).__name__}).")

    def schedule(self):
        while not self.stopping.is_set():
            self.consolidate(self.scheduler_signal)
            self.stopping.wait(60)

    def consolidate(self, signal, *, force=False):
        try:
            self.bank.mkdir(parents=True, exist_ok=True)
            release = try_lock(self.lock_path)
            if not release:
                return "Sharpshooter is already consolidating."
        except OSError as error:
            self.notice(f"Sharpshooter lock failed ({type(error).__name__}).")
            return "Could not lock Sharpshooter memory."
        state = self.state()
        try:
            interval = max(0, self.number("intervalMinutes", 5)) * 60_000
            if not force and milliseconds() - state["lastConsolidatedAt"] < interval:
                return "Sharpshooter consolidation is not due."
            groups = self.groups()
            if not groups:
                atomic_text(self.bank / "state.json", pack({**state, "lastConsolidatedAt": milliseconds()}))
                return "Sharpshooter queue is empty."
            current = self.read_files()
            docs = []
            for name in ("AGENTS.md", "CLAUDE.md"):
                if text := read_text(self.workspace / name).strip():
                    docs.append(f"--- {name} ---\n{text}")
            sessions = "\n\n".join("\n".join([f"### session {session}", *(f"- {pack(delta)}" for _, delta in group)])
                                   for session, group in groups)
            prompt = render("consolidate-input", architecture=current["architecture.md"], product=current["product.md"],
                            style=current["style.md"], projectDocs=token_clip("\n\n".join(docs), 6000), sessions=sessions)
            reply = self.request("replace_memory_files", "Replace complete Sharpshooter decision files.", CONSOLIDATE,
                                 render("consolidate-system", maxFileLines=120), redact_memory(prompt), "medium", 8192, signal)
            if reply is None:
                return "Sharpshooter has no available model."
            model, calls = reply
            files = replacement_files(calls, current)
            signal.check()
            with self.write_lock:
                with TemporaryDirectory(prefix=".sharp-", dir=self.bank) as temporary:
                    staging = Path(temporary)
                    for name, content in files.items():
                        atomic_text(staging / name, content, durable=True)
                    for name in files:
                        (staging / name).replace(self.bank / name)
                consumed = [file for _, group in groups for file, _ in group]
                for file in consumed:
                    file.unlink(missing_ok=True)
                for directory, _ in groups:
                    try:
                        (self.queue / directory).rmdir()
                    except OSError:
                        pass
                now = milliseconds()
                atomic_text(self.bank / "state.json", pack({"v": 1, "lastConsolidatedAt": now, "lastResult": {
                    "at": now, "sessions": len(groups), "deltas": len(consumed), "model": model,
                }}))
            return f"Consolidated {len(consumed)} Sharpshooter decision deltas."
        except CancelledError:
            return "Sharpshooter consolidation cancelled."
        except Exception as error:
            message = redact_memory(str(error))[:500] if isinstance(error, ValueError) else type(error).__name__
            try:
                atomic_text(self.bank / "state.json", pack({**state, "lastError": {"at": milliseconds(), "message": message}}))
            except OSError:
                pass
            self.notice(f"Sharpshooter consolidation failed ({message}).")
            return "Sharpshooter consolidation failed. Pending decisions were retained."
        finally:
            release()

    def command(self, argument, cancellation=None):
        verb, _, query = argument.strip().partition(" ")
        verb = verb.lower() or "view"
        if verb == "view":
            return self.instructions() or "Memory payload is empty."
        if verb in ("enqueue", "rebuild", "sync"):
            return self.consolidate(cancellation or Cancellation(), force=True)
        if verb in ("stats", "diagnose"):
            state = self.state()
            if verb == "stats":
                lines = [f"{name}: {len(text.splitlines())} lines, {len(text.encode('utf-8'))} bytes"
                         for name, text in self.read_files().items()]
                lines += [f"Pending: {sum(len(group) for _, group in self.groups())}"]
            else:
                interval = max(0, self.number("intervalMinutes", 5))
                due = max(0, (state["lastConsolidatedAt"] + interval * 60_000 - milliseconds() + 999) // 1000)
                lines = [f"Interval: {interval} minutes", f"Due in: {due}s", f"Lock: {self.lock_path}"]
            for label, key in (("Last result", "lastResult"), ("Last error", "lastError")):
                lines.append(f"{label}: {pack(state[key])}" if key in state else f"{label}: none")
            return "\n".join(lines)
        if verb == "queue":
            return "\n".join(f"{session}: {delta.get('kind', '')} {delta['statement']}"
                             for session, group in self.groups() for _, delta in group) or "Queue is empty."
        if verb == "search":
            needle = query.strip().casefold()
            return "\n".join(f"{name}: {line}" for name, text in self.read_files().items()
                             for line in text.splitlines() if needle and needle in line.casefold()) or "No matches."
        if verb in ("clear", "reset"):
            return self.clear()
        raise ValueError("Use /memory view, stats, diagnose, queue, search, sync, enqueue, rebuild, or clear.")

    def clear(self):
        self.bank.mkdir(parents=True, exist_ok=True)
        release = try_lock(self.lock_path)
        if not release:
            return "Sharpshooter is already consolidating."
        try:
            with self.write_lock:
                self.generation += 1
                for name in (*FILES, "state.json"):
                    (self.bank / name).unlink(missing_ok=True)
                if self.queue.is_dir() and not self.queue.is_symlink():
                    for directory in self.queue.iterdir():
                        if not directory.is_dir() or directory.is_symlink() or not SESSION_DIR.fullmatch(directory.name):
                            continue
                        for file in directory.glob("*.json"):
                            file.unlink(missing_ok=True)
                        try:
                            directory.rmdir()
                        except OSError:
                            pass
            return "Cleared this project's Sharpshooter memory."
        finally:
            release()

    def close(self):
        self.stopping.set()
        self.scheduler_signal.cancel()
        if self.scheduler:
            self.scheduler.join()
        if self.extraction:
            try:
                self.extraction.result(timeout=5)
            except TimeoutError:
                self.extraction_signal.cancel()
