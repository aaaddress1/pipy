"""Handoff prompt and bounded side-request policy."""

import json
import re
import sys
import time
from concurrent.futures import CancelledError
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Lock, Thread

from openai import APIStatusError

from cancellable_http import turn_scope
from recovery import EmptyTurn, StreamInterrupted, TRANSPORT_ERRORS, retry_after


PROMPTS = Path(__file__).with_name("prompts")
DOCUMENT = (PROMPTS / "handoff-document.md").read_text(encoding="utf-8")
CONTEXT = (PROMPTS / "handoff-summary-context.md").read_text(encoding="utf-8").rstrip()
AUTO_FOCUS = "Threshold-triggered maintenance: preserve critical implementation state and immediate next actions."


class Cancellation:
    def __init__(self, parent=None, *, automatic=False, reason="Context maintenance cancelled."):
        self.event = Event()
        self.parent = parent
        self.automatic = automatic
        self.reason = reason
        self.lock = Lock()
        self.futures = set()

    def is_cancelled(self):
        return self.event.is_set() or (self.parent is not None and self.parent.is_cancelled())

    def check(self):
        if self.event.is_set():
            raise CancelledError(self.reason)
        if self.parent:
            self.parent.check()

    def cancel(self):
        self.event.set()
        with self.lock:
            pending = tuple(self.futures)
        for future in pending:
            future.cancel()

    def attach(self, parent):
        with self.lock:
            self.parent = parent
            pending = tuple(self.futures)
        if parent:
            for future in pending:
                binding = parent.network_call(future)
                binding.__enter__()
                future.add_done_callback(lambda _, binding=binding: binding.__exit__(None, None, None))

    def wait(self, delay):
        end = time.monotonic() + delay
        while True:
            self.check()
            remaining = end - time.monotonic()
            if remaining <= 0:
                return
            self.event.wait(min(remaining, .1) if self.parent else remaining)

    @contextmanager
    def network_call(self, future):
        with self.lock:
            self.futures.add(future)
        if self.event.is_set():
            future.cancel()
        try:
            with self.parent.network_call(future) if self.parent else nullcontext():
                yield
        finally:
            with self.lock:
                self.futures.discard(future)


@contextmanager
def deadline(seconds, parent=None):
    cancellation = Cancellation(parent, reason="Deadline exceeded.")
    stopped = Event()
    end = time.monotonic() + seconds

    def expire():
        while (remaining := end - time.monotonic()) > 0:
            if stopped.wait(min(remaining, 3600)):
                return
        if not stopped.is_set():
            cancellation.cancel()

    timer = Thread(target=expire, name="mini-pi-deadline", daemon=True)
    timer.start()
    try:
        with turn_scope(cancellation):
            yield cancellation
    finally:
        stopped.set()
        timer.join()


def prompt(focus):
    base, _, optional = DOCUMENT.partition("{{#if additionalFocus}}")
    if not focus:
        return base.rstrip()
    return base + optional.partition("{{/if}}")[0].strip().replace("{{additionalFocus}}", focus)


def auto_only(error):
    if not isinstance(error, APIStatusError) or error.status_code != 400:
        return False
    return all(re.search(rf"\b{word}\b", error.message, re.I) for word in ("tool_choice", "auto", "supported"))


def transient(error):
    if not isinstance(error, APIStatusError):
        return True
    retry = error.response.headers.get("x-should-retry")
    return retry != "false" and (retry == "true" or error.status_code in (408, 409, 429) or error.status_code >= 500)


def oneshot(send, choice, cancellation):
    for attempt in range(3):
        cancellation.check()
        try:
            return send(choice, cancellation.check)
        except (APIStatusError, StreamInterrupted, *TRANSPORT_ERRORS) as error:
            cancellation.check()
            if attempt == 2 or not transient(error):
                raise
            delay = retry_after(error.response) if isinstance(error, APIStatusError) else None
            if delay is not None and delay > 300:
                raise
            cancellation.wait(delay if delay is not None else 0.5 * 2 ** attempt)


def generate(send, cancellation, *, retry=True):
    def request(choice):
        return oneshot(send, choice, cancellation) if retry else send(choice, cancellation.check)

    try:
        try:
            reply, _ = request("none")
        except APIStatusError as error:
            if not auto_only(error):
                raise
            reply, _ = request("auto")
    except EmptyTurn:
        cancellation.check()
        return ""
    cancellation.check()
    return (reply.get("content") or "").strip()


def file_operations(messages):
    files = {"read": set(), "write": set()}
    for message in messages:
        for call in message.get("tool_calls", ()):
            function = call["function"]
            name = function["name"]
            if name not in ("read", "edit", "write"):
                continue
            try:
                args = json.loads(function["arguments"])
            except (TypeError, ValueError):
                continue
            path = args.get("path") if isinstance(args, dict) else None
            if isinstance(path, str) and path and "://" not in path:
                files["read" if name == "read" else "write"].add(path)
    return {kind: sorted(paths) for kind, paths in files.items()}


def with_files(document, files):
    if not document:
        return document
    document = re.sub(r"(?is)\n*<files>.*?</files>", "", document).strip()
    read, written = set(files["read"]), set(files["write"])
    paths = sorted(read | written)
    if not paths:
        return document
    lines = [
        f"{path} ({'RW' if path in read and path in written else 'Read' if path in read else 'Write'})"
        for path in paths[:20]
    ]
    if len(paths) > 20:
        lines.append(f"[…{len(paths) - 20} files elided…]")
    return document + "\n\n<files>\n" + "\n".join(lines) + "\n</files>"


def save_document(directory, document):
    name = datetime.now(timezone.utc).strftime("handoff-%Y-%m-%dT%H-%M-%S.%fZ.md")
    path = directory / name
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="\n") as output:
            output.write(document + "\n")
        return path
    except OSError as error:
        print(f"[handoff: artifact could not be saved ({type(error).__name__})]", file=sys.stderr)
        return None
