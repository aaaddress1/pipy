"""Session-owned jobs and one-time result delivery."""

import re
import subprocess
import time
from concurrent.futures import CancelledError
from dataclasses import dataclass
from itertools import count
from logging import getLogger
from pathlib import Path
from sqlite3 import Error as SQLiteError
from tempfile import TemporaryDirectory
from threading import Condition, Thread

from handoff import Cancellation
from mcp_tools import ExternalToolResult


HARNESS_TAG = re.compile(r"<(?=\s*/?\s*(?:irc|system-[a-z][a-z-]*)(?![\w-]))", re.I)
WAIT_WINDOWS = (5, 10, 30, 60, 300)
logger = getLogger(__name__)


def tool_result(text, details, failed=False):
    return ExternalToolResult([{"type": "text", "text": text}], details, failed)


@dataclass
class Job:
    id: str
    command: str
    path: Path
    timeout: float
    cancel: Cancellation
    background: bool
    started: float
    worker: Thread = None
    status: str = "running"
    text: str = ""
    duration_ms: int = 0
    delivered_at: float | None = None
    exit_code: int | None = None
    timed_out: bool = False
    kind: str = "bash"
    details: dict | None = None
    agent_id: str | None = None
    queued: bool = False
    suppress_delivery: bool = False

    def summary(self):
        return {"jobId": self.id, "type": self.kind, "status": self.status, "label": self.command,
                **({"result": {key: value for key, value in self.details.items() if key != "report"}} if self.details else {}),
                "durationMs": round((time.monotonic() - self.started) * 1000)
                if self.status == "running" else self.duration_ms}

    def result(self):
        details = {"wallTimeMs": self.duration_ms,
                   **({"timeoutSeconds": self.timeout} if self.timeout else {"timeoutDisabled": True})}
        if self.exit_code is not None:
            details["exitCode"] = self.exit_code
        if self.timed_out:
            details["timedOut"] = True
        return tool_result(self.text, details, self.status != "completed")


class AsyncJobs:
    def __init__(self, execute, archive, limit, budget=None):
        self.execute, self.archive, self.limit, self.budget = execute, archive, limit, budget
        self.condition = Condition()
        self.claimed_peer = set()
        self.jobs = {}
        self.ids = count(1)
        self.directory = None
        self.wake = None
        self.peer = None
        self.services = None
        self.rules = None
        self.rule_notice = None
        self.closed = False
        self.wait_index, self.last_wait = 0, 0

    def available(self):
        return sum(job.status == "running" and not job.queued for job in self.jobs.values()) < self.limit

    def prune(self):
        now = time.monotonic()
        for identifier, job in list(self.jobs.items()):
            if job.status != "running" and job.delivered_at is not None and now - job.delivered_at >= 30:
                self.jobs.pop(identifier)

    def start(self, command, cwd, timeout, requested, threshold, turn):
        if turn:
            turn.check()
        wait_seconds = threshold if not timeout else min(threshold, max(0, timeout - 1))
        background = requested or wait_seconds == 0
        with self.condition:
            self.prune()
            if self.closed or not self.available():
                if requested:
                    raise ValueError("Too many background jobs are running.")
                return None
            if self.directory is None:
                self.directory = TemporaryDirectory(prefix="mini-pi-jobs-", ignore_cleanup_errors=True)
            identifier = f"bg_{next(self.ids)}"
            job = Job(identifier, command, Path(self.directory.name) / identifier, timeout,
                      Cancellation(self.budget, reason="Background job cancelled."), background, time.monotonic())
            job.worker = Thread(target=self.run, args=(job, cwd), name=f"mini-pi-{identifier}", daemon=True)
            self.jobs[identifier] = job
            job.worker.start()
            if not background:
                end = job.started + wait_seconds
                try:
                    while job.status == "running":
                        if turn:
                            turn.check()
                        if time.monotonic() >= end or self.steered(turn, soft=True):
                            break
                        self.condition.wait(min(0.1, max(0, end - time.monotonic())))
                    if turn:
                        turn.check()
                except CancelledError:
                    job.cancel.cancel()
                    job.delivered_at = time.monotonic()
                    raise
                if job.status != "running":
                    self.jobs.pop(identifier)
                    return job.result()
                job.background = True
        deadline = f" (killed once it has run {timeout:g}s in total)" if timeout else " (no deadline)"
        return tool_result(
            f"Backgrounded as job {identifier}{deadline}; its output is injected into the conversation as a follow-up "
            "the moment it finishes. Do NOT poll for it. Do other work, or end your reply and wait to be woken.",
            {"async": {"state": "running", "jobId": identifier, "type": "bash"},
             **({"timeoutSeconds": timeout} if timeout else {"timeoutDisabled": True})},
        )

    def spawn(self, items, reserve, *, kind="task", queued=False, suppress_delivery=False):
        """Register a batch before any member starts executing."""
        with self.condition:
            self.prune()
            running = sum(job.status == "running" and not job.queued for job in self.jobs.values())
            if self.closed or not queued and running + len(items) > self.limit:
                raise ValueError("Too many background jobs are running.")
            identifiers = {item[0] for item in items}
            if len(identifiers) != len(items) or identifiers & self.jobs.keys():
                raise ValueError("A background job with this id already exists.")
            reserve()
            spawned = []
            for identifier, label, agent_id, execute in items:
                job = Job(identifier, label, None, 0, Cancellation(self.budget, reason=f"{kind.title()} cancelled."),
                          True, time.monotonic(), kind=kind, agent_id=agent_id, queued=queued,
                          suppress_delivery=suppress_delivery)
                job.worker = Thread(target=self.run_registered, args=(job, execute), name=f"mini-pi-{kind}-{identifier}", daemon=True)
                self.jobs[identifier] = job
                spawned.append(job)
            for job in spawned:
                job.worker.start()
            return [job.summary() for job in spawned]

    def start_eval(self, label, timeout, threshold, turn, execute, on_update):
        """Keep an eval on the same kernel when its foreground wait ends."""
        if turn:
            turn.check()
        latest, result = None, None
        wait_seconds = min(threshold, max(0, timeout - 1)) if timeout else threshold

        def progress(value):
            nonlocal latest
            with self.condition:
                latest = value
                visible = not job.background and job.delivered_at is None
            if visible and on_update:
                on_update(value)

        def run():
            nonlocal result
            try:
                result = execute(job.cancel, progress)
            except Exception as error:
                result = tool_result(str(error) or "Eval cell aborted.", {}, True)
            job.details = result.details
            text = "\n".join(block["text"] for block in result.content if block["type"] == "text")
            status = "cancelled" if job.cancel.is_cancelled() else "failed" if result.is_error else "completed"
            self.finish(job, status, text)

        with self.condition:
            self.prune()
            if self.closed or not self.available():
                return None
            identifier = f"bg_{next(self.ids)}"
            job = Job(identifier, label, None, timeout, Cancellation(self.budget, reason="Eval job cancelled."),
                      wait_seconds == 0, time.monotonic(), kind="eval")
            self.jobs[identifier] = job
            job.worker = Thread(target=run, name=f"mini-pi-eval-{identifier}", daemon=True)
            job.worker.start()
            end = job.started + wait_seconds
            steered = False
            try:
                while not job.background and job.status == "running":
                    if turn:
                        turn.check()
                    steered = self.steered(turn, soft=True)
                    remaining = end - time.monotonic()
                    if steered or remaining <= 0:
                        break
                    self.condition.wait(min(.1, remaining))
                if turn:
                    turn.check()
            except (CancelledError, KeyboardInterrupt):
                job.delivered_at = time.monotonic()
                job.cancel.cancel()
                raise
            if job.status != "running" and not job.background:
                self.jobs.pop(identifier)
                return result
            job.background = True
            details = dict(latest.details) if latest else {}
            preview = latest.content[0]["text"].strip() if latest else ""
        deadline = f"killed once it has run {timeout:g}s in total" if timeout else "no deadline"
        notice = "Backgrounded early to handle an incoming message. " if steered else ""
        details["async"] = {"state": "running", "jobId": identifier, "type": "eval"}
        text = f"{notice}Backgrounded as job {identifier} ({deadline}). Its result will be delivered automatically."
        return tool_result((preview + "\n\n" if preview else "") + text, details)

    def progress(self, identifier, details, text):
        with self.condition:
            if (job := self.jobs.get(identifier)) and job.status == "running":
                job.details, job.text = details, text

    def run_registered(self, job, execute):
        discard = False
        try:
            result = execute(job.cancel)
            job.details = result
            status, text = result["status"], result["report"]
            discard = result.get("suppressDelivery", False)
        except CancelledError:
            status, text = "cancelled", f"{job.kind.title()} {job.id} was cancelled."
        except Exception as error:
            status, text = "failed", f"{job.kind.title()} {job.id} failed: {error}"
        self.finish(job, status, text, discard=discard or job.suppress_delivery)

    def run(self, job, cwd):
        status = "completed"
        code, timed_out = None, False
        try:
            with job.path.open("w+b") as output:
                try:
                    code = self.execute(job.command, cwd, output, job.cancel, job.timeout)
                    status = "completed" if code == 0 else "failed"
                    header = f"Exit code: {code}"
                except subprocess.TimeoutExpired:
                    status, header = "failed", f"Command timed out after {job.timeout:g}s."
                    timed_out = True
                except CancelledError:
                    status, header = "cancelled", "Background job cancelled."
                except OSError as error:
                    status, header = "failed", f"Could not run command: {error}"
                text = self.archive(output, header, 12_000, 4_000)
        except Exception as error:
            status, text = "failed", f"Could not collect background job output: {error}"
        job.exit_code, job.timed_out = code, timed_out
        self.finish(job, status, text)

    def finish(self, job, status, text, *, discard=False):
        if self.peer and job.agent_id:
            try:
                self.peer.bus.record_job_reply(job.agent_id, self.peer.sid)
            except SQLiteError:
                logger.exception("Unable to record the reply from job %s", job.id)
        with self.condition:
            job.text, job.status = text, status
            if discard:
                job.delivered_at = time.monotonic()
            job.duration_ms = round((time.monotonic() - job.started) * 1000)
            try:
                if job.path:
                    job.path.unlink(missing_ok=True)
            except OSError:
                pass
            wake = self.wake if job.background and not job.suppress_delivery and not self.closed else None
            self.condition.notify_all()
        if self.peer and job.kind == "task":
            self.peer.bus.recover_deferred()
        if wake:
            wake(self)

    def notify(self):
        with self.condition:
            wake = self.wake if not self.closed else None
            self.condition.notify_all()
        if wake:
            wake(self)

    def pending(self, include_peer=True):
        if include_peer and self.peer and self.peer.ready():
            return True
        if include_peer and self.services and self.services.peek():
            return True
        if include_peer and self.rules and (self.rule_notice or self.rules.judgment_pending()):
            return True
        with self.condition:
            return any(job.background and not job.suppress_delivery and job.delivered_at is None for job in self.jobs.values())

    def ready(self):
        if self.peer and self.peer.ready():
            return True
        if self.services and self.services.peek():
            return True
        if self.rules and (self.rule_notice or self.rules.judgment_ready()):
            return True
        with self.condition:
            return any(job.background and not job.suppress_delivery and job.delivered_at is None and job.status != "running"
                       for job in self.jobs.values())

    def take(self):
        with self.condition:
            ready = [job for job in self.jobs.values() if job.background and not job.suppress_delivery
                     and job.delivered_at is None and job.status != "running"]
            for job in ready:
                self.consume(job)
        if not ready:
            return None
        jobs = [job.summary() for job in ready]
        labels = [job.command.replace("\r", r"\r").replace("\n", r"\n") for job in ready]
        labels = [label if len(label) <= 400 else label[:198] + " ... " + label[-197:] for label in labels]
        labels = [HARNESS_TAG.sub("&lt;", label) for label in labels]
        if len(ready) == 1:
            header = f"Background job {ready[0].id} ({labels[0]}) has completed. Resume your work using the result below."
            body = HARNESS_TAG.sub("&lt;", ready[0].text)
        else:
            header = f"{len(ready)} background jobs have completed. Resume your work using the results below."
            body = "\n\n".join(f"── Job {job.id} ({label}) ──\n" + HARNESS_TAG.sub("&lt;", job.text)
                                 for job, label in zip(ready, labels))
        return {"role": "developer", "content": f"<system-notice>\n{header}\n\n{body}\n</system-notice>",
                "_async_jobs": {"jobs": jobs}}

    def consume(self, job):
        with self.condition:
            if job.delivered_at is None:
                job.delivered_at = time.monotonic()

    def take_background(self):
        delivery = ((self.peer.peek() if self.peer else None) or self.take()
                    or (self.services.peek() if self.services else None))
        if delivery:
            return delivery
        with self.condition:
            warning, self.rule_notice = self.rule_notice, None
        return warning or (self.rules.take_judgments() if self.rules else None)

    @staticmethod
    def steered(turn, *, soft=False):
        return bool(turn and (not soft or getattr(turn, "soft_steering", True))
                    and (queued := getattr(turn, "has_steering", None)) and queued())

    def wait(self, turn=None, *, drain=False):
        try:
            return self.wait_loop(turn, drain)
        finally:
            if self.peer:
                self.peer.wait_for(set())

    def wait_loop(self, turn, drain):
        if self.services:
            self.services.attach(existing_only=True)
        with self.condition:
            if not drain and time.monotonic() - self.last_wait >= 60:
                self.wait_index = 0
            window = WAIT_WINDOWS[self.wait_index]
            end = None if drain else time.monotonic() + window
            while True:
                if turn:
                    turn.check()
                if self.steered(turn):
                    self.wait_index = 0
                    return tool_result("Wait interrupted by message.", {"op": "wait", "interrupted": True})
                running = [job for job in self.jobs.values() if job.background and not job.suppress_delivery and job.status == "running"
                           and job.delivered_at is None]
                service_running = self.services and not drain and self.services.active()
                if self.peer and (message := self.peer.peek(waiting=not drain, watch_replies=not running and not service_running,
                                                           exclude=self.claimed_peer)):
                    self.wait_index = 0
                    self.last_wait = time.monotonic()
                    if drain:
                        return message
                    self.claimed_peer.update(row["id"] for row in message["_irc_messages"])
                    return tool_result(message["content"], {"op": "wait", "waited": message["_irc_messages"]})
                if delivery := self.take():
                    self.wait_index = 0
                    self.last_wait = time.monotonic()
                    if drain:
                        return delivery
                    return tool_result(delivery["content"], {"op": "wait", **delivery["_async_jobs"]})
                if self.services and (completion := self.services.peek()):
                    if drain:
                        return completion
                    self.wait_index = 0
                    self.last_wait = time.monotonic()
                    if running:
                        text, details = self.inspect("")
                        return tool_result(text, {"op": "wait", "jobs": details})
                    return tool_result("A supervised service finished. Read proc:// for status and output.",
                                       {"op": "wait", "services": completion["_service_runs"]})
                if self.rules and (warning := self.rule_notice or self.rules.take_judgments()):
                    if drain:
                        self.rule_notice = None
                        return warning
                    if not self.rule_notice:
                        self.rule_notice = warning
                        return tool_result("Wait interrupted by a rule judgment.", {"op": "wait", "interrupted": True})
                if self.peer:
                    agents = [job.agent_id for job in running] if not service_running and all(job.agent_id for job in running) else []
                    self.peer.wait_for(agents, interruptible=True)
                if not running and not service_running:
                    if drain:
                        if self.rules and self.rules.judgment_pending():
                            self.condition.wait(.1)
                            continue
                        return None
                    if not self.peer:
                        return tool_result("Nothing to wait for.", {"op": "wait", "jobs": []})
                    replies, blocked, on_user, stopped = self.peer.awaiting()
                    if not replies:
                        return tool_result(self.peer.no_reply(blocked, on_user, stopped), {"op": "wait", "jobs": []})
                if end is not None and time.monotonic() >= end:
                    self.last_wait = time.monotonic()
                    self.wait_index = min(self.wait_index + 1, len(WAIT_WINDOWS) - 1)
                    if running:
                        text, details = self.inspect("")
                    else:
                        details = []
                        text = (f"No service finished within {window}s. Read proc:// for status."
                                if service_running else f"No reply within {window}s.")
                    if window > 5:
                        text += f"\n<system-notice>Waited {window}s. Work on anything that does not require these results.</system-notice>"
                    return tool_result(text, {"op": "wait", "useless": True, "jobs": details})
                self.condition.wait(0.1)

    def release_tool_delivery(self, result):
        if rows := result.get("_irc_messages"):
            with self.condition:
                self.claimed_peer.difference_update(row["id"] for row in rows)
                self.condition.notify_all()

    def inspect(self, identifier):
        with self.condition:
            self.prune()
            jobs = [job for job in self.jobs.values() if job.background and (not identifier or job.id == identifier)]
            if identifier and not jobs:
                raise ValueError(f"Unknown job {identifier!r}.")
            text = []
            for job in jobs:
                row = f"{job.id} [{job.kind}] {job.status} — {job.command}"
                if identifier:
                    if job.status != "running" or job.text:
                        row += "\n" + job.text
                    elif job.path and job.path.is_file():
                        with job.path.open("rb") as output:
                            size = output.seek(0, 2)
                            output.seek(max(0, size - 4000))
                            row += "\n" + output.read(4000).decode("utf-8", errors="replace")
                text.append(row)
            return "\n".join(text) or "No background jobs.", [job.summary() for job in jobs]

    def kill(self, identifier):
        with self.condition:
            job = self.jobs.get(identifier)
            if not job or not job.background:
                raise ValueError(f"Unknown job {identifier!r}.")
            if job.status != "running":
                return f"Job {identifier} has already settled."
            job.cancel.cancel()
            self.condition.notify_all()
            return f"Cancelled job {identifier}."

    def close(self):
        with self.condition:
            self.closed = True
            self.wake = None
            jobs = list(self.jobs.values())
            for job in jobs:
                if job.status == "running":
                    job.cancel.cancel()
            self.condition.notify_all()
        for job in jobs:
            job.worker.join()
        if self.rules:
            self.rules.close_judgments()
        if self.services:
            self.services.close()
        if self.directory:
            self.directory.cleanup()

    def discard(self):
        with self.condition:
            for job in self.jobs.values():
                job.background = False
                job.delivered_at = time.monotonic()
                if job.status == "running":
                    job.cancel.cancel()
            self.condition.notify_all()
