"""Durable side conversations shared by terminal and RPC sessions."""

import hashlib
import json
import os
import re
import stat
from concurrent.futures import CancelledError
from contextvars import copy_context
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import RLock
from time import time
from uuid import uuid4

from filelock import FileLock

from cancellable_http import turn_scope
from handoff import Cancellation


PROMPTS = Path(__file__).with_name("prompts")
PROMPT = (PROMPTS / "btw-user.md").read_text(encoding="utf-8")
NO_TOOLS = (PROMPTS / "side-channel-no-tools.md").read_text(encoding="utf-8").strip()
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
STATES = {"running", "complete", "cancelled", "error", "interrupted"}
TURN_FIELDS = {"question", "images", "answer", "status", "createdAt", "updatedAt", "error"}
ANSI = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))|[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def timestamp():
    return int(time() * 1000)


def turns(record):
    return (record, *record.get("followUps", ()))


def latest(record):
    return turns(record)[-1]


def copy_text(record):
    return next((turn["answer"] for turn in reversed(turns(record)) if turn["answer"].strip()), "")


def split_inject(answer):
    closing = answer.rfind("</inject>")
    opening = answer.rfind("<inject>", 0, closing) if closing >= 0 else -1
    if opening < 0:
        return None
    start = opening + len("<inject>")
    end = answer.index("</inject>", start)
    return answer[:opening], ANSI.sub("", answer[start:end]).expandtabs(4).strip(), answer[end + len("</inject>"):]


def inject_text(answer):
    parts = split_inject(answer)
    return parts[1] if parts else ""


def turn_key(record, index=None):
    return record["id"], len(record.get("followUps", ())) if index is None else index


def patch_latest(record, **fields):
    if followups := record.get("followUps"):
        return {**record, "followUps": [*followups[:-1], {**followups[-1], **fields}]}
    return {**record, **fields}


def recover(record):
    def turn(item):
        return {**item, "status": "interrupted"} if item["status"] == "running" else item

    updated = turn(record)
    if followups := updated.get("followUps"):
        updated = {**updated, "followUps": [turn(item) for item in followups]}
    return updated


def wire_record(record):
    return {key: value for key, value in record.items() if key != "title"}


def valid_turn(item):
    if not isinstance(item, dict):
        return False
    if not all(isinstance(item.get(key), str) for key in ("question", "answer", "status")):
        return False
    if item["status"] not in STATES or "error" in item and not isinstance(item["error"], str):
        return False
    if not all(type(item.get(key)) in (int, float) and 0 <= item[key] <= 8_640_000_000_000_000
               for key in ("createdAt", "updatedAt")):
        return False
    images = item.get("images", [])
    return isinstance(images, list) and all(
        isinstance(image, dict) and image.get("type") == "image"
        and isinstance(image.get("data"), str) and isinstance(image.get("mimeType"), str)
        for image in images
    )


def validate_record(record):
    if (not valid_turn(record) or not isinstance(record.get("id"), str) or not IDENTIFIER.fullmatch(record["id"])
            or "leafId" not in record or record["leafId"] is not None and not isinstance(record["leafId"], str)
            or set(record) - TURN_FIELDS - {"id", "leafId", "followUps"}):
        raise ValueError("Invalid /btw history record.")
    followups = record.get("followUps", [])
    if not isinstance(followups, list) or not all(valid_turn(item) and not set(item) - TURN_FIELDS for item in followups):
        raise ValueError("Invalid /btw follow-up record.")


def atomic_write(path, data):
    with NamedTemporaryFile(dir=path.parent, delete=False) as output:
        temporary = Path(output.name)
        try:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
            output.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


class HistoryConflict(RuntimeError):
    pass


class History:
    def __init__(self, directory):
        self.directory = directory

    def path(self, identifier, suffix="json"):
        if not isinstance(identifier, str) or not IDENTIFIER.fullmatch(identifier):
            raise ValueError("Invalid /btw topic ID.")
        return self.directory / f"entry-{identifier}.{suffix}"

    def read(self, identifier):
        path = self.path(identifier)
        try:
            if not stat.S_ISREG(path.lstat().st_mode):
                raise ValueError(f"Expected a regular /btw history file at {path}.")
            raw = path.read_bytes()
        except FileNotFoundError:
            return None, None
        record = json.loads(raw)
        validate_record(record)
        if record["id"] != identifier:
            raise ValueError(f"Topic ID does not match {path.name}.")
        title = self.path(identifier, "title")
        try:
            if stat.S_ISREG(title.lstat().st_mode):
                record["title"] = title.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            pass
        return recover(record), hashlib.sha256(raw).digest()

    def records(self):
        records = []
        for path in sorted(self.directory.glob("entry-*.json")):
            record, _ = self.read(path.stem.removeprefix("entry-"))
            if record is not None:
                records.append(record)
        return records

    def lease(self, identifier):
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if os.name != "nt":
            self.directory.chmod(0o700)
        lock = FileLock(str(self.path(identifier)) + ".lock", thread_local=False)
        lock.acquire(timeout=5)
        return lock

    def checkpoint(self, record, revision):
        identifier = record["id"]
        previous, current = self.read(identifier)
        if current != revision:
            raise HistoryConflict(f"BTW history conflict for {identifier}; reopen history before retrying.")
        fields = wire_record(record)
        validate_record(fields)
        if previous is None:
            title = self.path(identifier, "title")
            if "title" in record:
                atomic_write(title, record["title"].encode("utf-8"))
            else:
                title.unlink(missing_ok=True)
        data = (json.dumps(fields, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        atomic_write(self.path(identifier), data)
        return hashlib.sha256(data).digest()


def dedupe(answer):
    result = []
    for line, run in groupby(answer.strip().split("\n")):
        count = sum(1 for _ in run)
        result.extend([line, f"[…{count}×]"] if count > 3 else [line] * count)
    return "\n".join(result)


def prompt(question, inject=False):
    prefix, _, suffix = PROMPT.partition("{{#if inject}}")
    instructions, _, ending = suffix.partition("{{/if}}")
    return (prefix + (instructions if inject else "") + ending).replace("{{question}}", question)


def question_content(question, images=()):
    if not images:
        return question
    return [{"type": "text", "text": question}, *[
        {"type": "image_url", "image_url": {"url": f"data:{image['mimeType']};base64,{image['data']}"}}
        for image in images
    ]]


def messages(history, question, images, inject):
    replay = []
    for turn in history:
        replay.append({"role": "user", "content": question_content(prompt(turn["question"], inject), turn.get("images", ()))})
        if turn["answer"]:
            replay.append({"role": "assistant", "content": turn["answer"]})
    replay.append({"role": "user", "content": question_content(prompt(question, inject), images)})
    return replay


@dataclass
class Request:
    record: dict
    revision: bytes
    lease: FileLock
    signal: Cancellation
    leaf: int | None
    inject: bool
    assistant: dict | None = None


class Btw:
    def __init__(self, history, prepare, workers, budget=None, emit=None):
        self.history, self.prepare, self.workers = history, prepare, workers
        self.budget, self.emit = budget, emit or (lambda *_, **__: None)
        self.lock = RLock()
        self.running = None
        self.completed = None
        self.unsaved = {}
        self.injections = {}
        self.sending = set()

    def records(self):
        with self.lock:
            records = {record["id"]: record for record in self.history.records()}
            records.update((key, request.record) for key, request in self.unsaved.items())
            if self.running:
                records[self.running.record["id"]] = self.running.record
            return sorted(records.values(), key=lambda record: (-record["createdAt"], record["id"]))

    def begin(self, question, identifier=None, *, inject=False, images=(), title=None):
        if not isinstance(question, str) or not (question := question.strip()):
            raise ValueError("btw requires a non-empty question")
        with self.lock:
            if self.running:
                raise RuntimeError("A /btw question is still running; cancel it first")
            self.flush()
            previous, revision = self.history.read(identifier) if identifier is not None else (None, None)
            if identifier is not None and previous is None:
                raise ValueError(f"Unknown /btw topic: {identifier}")
            leaf, send = self.prepare()
            now = timestamp()
            turn = {"question": question, "answer": "", "status": "running", "createdAt": now, "updatedAt": now}
            if images:
                turn["images"] = list(images)
            if previous:
                record = {**previous, "followUps": [*previous.get("followUps", ()), turn]}
            else:
                record = {**turn, "id": uuid4().hex, "leafId": str(leaf) if leaf is not None else None}
                if title:
                    record["title"] = title
            lease = self.history.lease(record["id"])
            try:
                revision = self.history.checkpoint(record, revision)
            except BaseException:
                lease.release()
                raise
            request = Request(record, revision, lease, Cancellation(self.budget, reason="/btw question cancelled."), leaf, inject)
            self.running = request
            self.completed = None
            self.emit("btw_record", record=wire_record(record))

            def launch():
                replay = messages(turns(previous) if previous else (), question, images, inject)
                return self.workers.submit(copy_context().run, self.run, request, send, replay)

            return record, launch

    def run(self, request, send, replay):
        try:
            with turn_scope(request.signal):
                request.signal.check()
                reply = send(replay, request.signal, lambda text: self.delta(request, text))
                request.signal.check()
            answer = dedupe(reply.get("content") or "")
            request.assistant = {"role": "assistant", "content": answer}
            thinking = reply.get("_thinking") or "\n\n".join(
                block.get("thinking", "") for block in reply.get("_anthropic", {}).get("blocks", ())
                if block.get("type") == "thinking"
            )
            if thinking:
                request.assistant["_thinking"] = thinking
            self.finish(request, "complete", answer=answer)
        except CancelledError:
            self.finish(request, "cancelled")
        except Exception as error:
            self.finish(request, "error", error=str(error))

    def delta(self, request, text):
        with self.lock:
            if self.running is not request or request.signal.is_cancelled():
                return
            answer = latest(request.record)["answer"] + text if text is not None else ""
            request.record = patch_latest(request.record, answer=answer, updatedAt=timestamp())
            if text:
                self.emit("btw_delta", recordId=request.record["id"], delta=text)

    def finish(self, request, status, **fields):
        with self.lock:
            if self.running is not request:
                return False
            self.running = None
            request.record = patch_latest(request.record, status=status, updatedAt=timestamp(), **fields)
            if status == "complete":
                self.completed = request
                if request.inject and (text := inject_text(fields.get("answer", ""))):
                    self.injections[turn_key(request.record)] = text
            try:
                self.history.checkpoint(request.record, request.revision)
            except (OSError, ValueError, RuntimeError) as error:
                self.unsaved[request.record["id"]] = request
                self.emit("notice", source="btw-history", level="error", message=f"Could not save /btw history: {error}")
            finally:
                request.lease.release()
            self.emit("btw_record", record=wire_record(request.record))
            return True

    def cancel(self, identifier=None):
        with self.lock:
            request = self.running
            if request is None or identifier is not None and request.record["id"] != identifier:
                return False
            request.signal.cancel()
            return self.finish(request, "cancelled")

    def flush(self):
        with self.lock:
            for identifier, request in tuple(self.unsaved.items()):
                try:
                    lease = self.history.lease(identifier)
                    try:
                        self.history.checkpoint(request.record, request.revision)
                    finally:
                        lease.release()
                except HistoryConflict as error:
                    self.emit("notice", source="btw-history", level="error", message=str(error))
                except (OSError, ValueError, RuntimeError) as error:
                    raise RuntimeError(f"/btw history could not be saved. Unsaved answers remain in this session. {error}") from error
                del self.unsaved[identifier]

    def close(self):
        self.cancel()
        self.flush()
        self.completed = None

    def injection(self, record):
        with self.lock:
            key = turn_key(record)
            return (self.injections.get(key) or "") if latest(record)["status"] == "complete" and key not in self.sending else ""

    def reserve_injection(self, record):
        with self.lock:
            text = self.injection(record)
            if text:
                self.sending.add(turn_key(record))
            return text

    def release_injection(self, record):
        with self.lock:
            self.sending.discard(turn_key(record))

    def injection_state(self, record, index):
        with self.lock:
            key = turn_key(record, index)
            if key in self.injections:
                return "ready" if self.injections[key] else "sent"

    def injected(self, record):
        with self.lock:
            key = turn_key(record)
            self.sending.discard(key)
            self.injections[key] = None

    def promotion(self, identifier, leaf):
        with self.lock:
            request = self.completed
            if self.running or request is None or request.record["id"] != identifier:
                raise ValueError("No completed /btw answer to branch.")
            if request.record.get("followUps"):
                raise ValueError("Multi-turn side conversations remain in BTW history.")
            if leaf is None or request.leaf != leaf:
                raise ValueError("The session changed since /btw started.")
            if not copy_text(request.record):
                raise ValueError("The /btw answer is empty.")
            self.flush()
            record = request.record
            return [{"role": "user", "content": question_content(record["question"], record.get("images", ()))},
                    request.assistant]
