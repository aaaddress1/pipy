"""Session-owned bulk eval judgments and their shared result cursor."""

import logging
import math
from concurrent.futures import ThreadPoolExecutor
from threading import Condition, Event, Lock, Thread, Timer
from time import monotonic
from uuid import uuid4

from cancellable_http import turn_scope
from cost_gate import total_cost, usd
from eval_completion import REQUEST_LIMIT, request_slot
from eval_judgment import parse_questions, parse_state
from handoff import Cancellation
from judgment_images import load_images


NAME = "__judge_batch__"
EVENT = "eval:judgment-batch-progress"
DRAIN_COALESCE = .1
EVENT_INTERVAL = .25
MISSING = object()
logger = logging.getLogger(__name__)


def invalid(detail):
    return ValueError(f"judge_batch() received invalid arguments: {detail}")


def number(value):
    return type(value) is int or type(value) is float and math.isfinite(value)


def count(args, key, default, maximum=None):
    value = args.get(key, default)
    if not number(value) or value < 0 or type(value) is float and not value.is_integer():
        raise invalid(f"{key} must be a non-negative integer")
    return int(min(value, maximum) if maximum is not None else value)


def inputs(args):
    if "images" in args:
        raise invalid("images must be provided on each item")
    items = args.get("items")
    if not isinstance(items, list) or not items:
        raise invalid("items must be a nonempty array of { key, state }")
    seen, parsed = set(), []
    for index, item in enumerate(items):
        key = item.get("key") if isinstance(item, dict) else None
        if not (isinstance(key, str) and key or number(key)):
            raise invalid(f"item {index} needs a string or number key")
        if key in seen:
            raise invalid(f"duplicate item key {key!r}")
        seen.add(key)
        parsed.append((key, parse_state(item.get("state")), item.get("images", MISSING)))
    return parsed


def key_string(key):
    return str(int(key)) if type(key) is float and key.is_integer() else str(key)


class Batch:
    def __init__(self, bridge, jobs, budget, items, questions, concurrency, retries, minimum, intent, emit, forget):
        self.bridge, self.jobs, self.items, self.questions = bridge, jobs, items, questions
        self.concurrency, self.retries, self.minimum, self.intent = concurrency, retries, minimum, intent
        self.emit, self.forget = emit, forget
        self.id = "jdgb-" + uuid4().hex
        self.signal = Cancellation(budget, reason="judge_batch cancelled")
        self.condition, self.finished = Condition(), Event()
        self.settled = []
        self.cursor, self.failures, self.cost = 0, 0, 0
        self.model = self.error = None
        self.running = True
        self.started = monotonic()
        self.last_event = self.last_report = -math.inf
        self.timer = None

    def start(self):
        with self.condition:
            snapshot = self.snapshot()
            self.publish()
        if self.jobs:
            try:
                self.jobs.spawn([(self.id, self.intent, None, self.run)], lambda: None, kind="eval")
                return snapshot
            except ValueError:
                pass
        Thread(target=self.run, name=f"mini-pi-{self.id}", daemon=True).start()
        return snapshot

    def snapshot(self):
        return {"id": self.id, "intent": self.intent, "total": len(self.items), "done": len(self.settled),
                "failed": self.failures, "cost": self.cost, "running": self.running,
                "elapsedS": math.floor((monotonic() - self.started) * 10 + .5) / 10,
                **({"model": self.model} if self.model is not None else {}),
                **({"error": self.error} if self.error is not None else {})}

    def status(self):
        with self.condition:
            return self.snapshot()

    def publish(self, *, final=False):
        if not self.emit:
            return
        now = monotonic()
        elapsed = now - self.last_event
        if not final and elapsed < EVENT_INTERVAL:
            if self.timer is None:
                self.timer = Timer(EVENT_INTERVAL - elapsed, self.publish_pending)
                self.timer.daemon = True
                self.timer.start()
            return
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None
        self.last_event = now
        status = self.snapshot()
        status.pop("model", None)
        status.pop("elapsedS", None)
        try:
            self.emit(status)
        except Exception:
            logger.debug("Could not emit judgment batch progress", exc_info=True)

    def publish_pending(self):
        with self.condition:
            if self.timer is not None and self.running:
                self.timer = None
                self.publish()

    def add_usage(self, usage):
        with self.condition:
            self.cost += total_cost(usage)

    def cancel(self):
        with self.condition:
            if not self.running or self.signal.is_cancelled():
                return False
            self.signal.cancel()
            self.condition.notify_all()
            return True

    def drain(self, cancellation, milliseconds):
        end = math.inf if milliseconds is None else monotonic() + min(milliseconds, 10**20) / 1000
        with self.condition:
            if self.cursor == len(self.settled) and self.running:
                previous = len(self.settled)
                while previous == len(self.settled) and self.running and (remaining := end - monotonic()) > 0:
                    cancellation.check()
                    self.condition.wait(min(.05, remaining))
                if self.cursor < len(self.settled):
                    linger = min(end, monotonic() + DRAIN_COALESCE)
                    while self.running and (remaining := linger - monotonic()) > 0:
                        cancellation.check()
                        self.condition.wait(min(.05, remaining))
                cancellation.check()
            if self.cursor < len(self.settled):
                items = self.settled[self.cursor:]
                self.cursor = len(self.settled)
                return items
            if not self.running and self.error is not None:
                raise ValueError(self.error)
            return []

    def collected(self, field):
        with self.condition:
            return {key_string(item["key"]): item[field] for item in self.settled if field in item}

    def summary(self, snapshot):
        phase = "judging" if snapshot["running"] else "cancelled" if self.signal.is_cancelled() else "judged"
        values = [f"{phase} {snapshot['done']}/{snapshot['total']}"]
        if snapshot["failed"]:
            values.append(f"{snapshot['failed']} failed")
        if snapshot["cost"] > 0:
            values.append(usd(snapshot["cost"]))
        if "model" in snapshot:
            values.append(snapshot["model"])
        values.append(f"{snapshot['elapsedS']:g}s")
        return " · ".join(values)

    def judge_item(self, key, state, sources):
        if self.signal.is_cancelled():
            return {"key": key, "error": "cancelled"}
        if sources is not MISSING and not isinstance(sources, list):
            return {"key": key, "error": f"images for input {key_string(key)!r} must be an array of paths or image attachments"}
        try:
            with request_slot(self.signal), turn_scope(self.signal):
                images = [] if sources is MISSING else load_images(sources, self.bridge.cwd, self.signal)
                for attempt in range(self.retries + 1):
                    self.signal.check()
                    try:
                        answer = self.bridge.classify(state, self.questions, images, self.signal,
                                                      "judge_batch", on_usage=self.add_usage)
                        self.signal.check()
                        return {"key": key, **answer}
                    except Exception:
                        if attempt == self.retries or self.signal.is_cancelled():
                            raise
        except Exception as error:
            return {"key": key, "error": "cancelled" if self.signal.is_cancelled()
                    else str(error) or type(error).__name__}

    def run(self, job_signal=None):
        if job_signal is not None:
            self.signal.attach(job_signal)
        items = iter(self.items)

        def worker():
            while True:
                with self.condition:
                    item = next(items, None)
                if item is None:
                    return
                settled = self.judge_item(*item)
                report = None
                with self.condition:
                    self.settled.append(settled)
                    if "error" in settled:
                        self.failures += 1
                    if "model" in settled:
                        self.model = settled["model"]
                    if self.jobs and monotonic() - self.last_report >= 1:
                        self.last_report = monotonic()
                        report = self.snapshot()
                    self.publish()
                    self.condition.notify_all()
                if report:
                    self.jobs.progress(self.id, {"judgmentBatch": report}, self.summary(report))

        failure = None
        try:
            size = min(self.concurrency, len(self.items))
            with ThreadPoolExecutor(max_workers=size, thread_name_prefix=f"mini-pi-{self.id}") as workers:
                futures = [workers.submit(worker) for _ in range(size)]
                for future in futures:
                    future.result()
        except Exception as error:
            failure = str(error) or type(error).__name__
        finally:
            with self.condition:
                cancelled = self.signal.is_cancelled()
                ok = len(self.settled) - self.failures
                self.error = failure
                if cancelled:
                    self.error = "judge_batch cancelled"
                elif self.error is None and ok < self.minimum:
                    last = next((item["error"] for item in reversed(self.settled)
                                 if "error" in item and item["error"] != "cancelled"), "unknown")
                    self.error = (f"judge_batch: only {ok}/{len(self.items)} item(s) judged "
                                  f"(min_ok={self.minimum}); last error: {last}")
                self.running = False
                snapshot = self.snapshot()
                self.publish(final=True)
                self.condition.notify_all()
            self.finished.set()
            self.forget(self)
        status = "cancelled" if cancelled else "failed" if self.error is not None else "completed"
        report = self.summary(snapshot)
        if self.error:
            report += "\n" + self.error
        return {"status": status, "report": report, "judgmentBatch": snapshot}


class JudgmentBatches:
    def __init__(self, bridge, jobs, budget, emit=None):
        self.bridge, self.jobs, self.budget, self.emit = bridge, jobs, budget, emit
        self.lock = Lock()
        self.batches, self.active = {}, set()
        self.closed = False

    def forget(self, batch):
        with self.lock:
            self.active.discard(batch)

    def create(self, args, cancellation):
        items, questions = inputs(args), parse_questions(args.get("questions"))
        intent = args.get("intent", "Judging")
        if not isinstance(intent, str) or not intent.strip():
            raise invalid("intent must be a non-empty string")
        concurrency = count(args, "concurrency", REQUEST_LIMIT, REQUEST_LIMIT) or REQUEST_LIMIT
        retries, minimum = count(args, "retries", 1), count(args, "minOk", 1, len(items))
        cancellation.check()
        batch = Batch(self.bridge, self.jobs, self.budget, items, questions, concurrency, retries, minimum,
                      intent, self.emit, self.forget)
        with self.lock:
            if self.closed:
                raise ValueError("The eval session has closed.")
            self.batches[batch.id] = batch
            self.active.add(batch)
            try:
                return batch.start()
            except Exception:
                self.batches.pop(batch.id, None)
                self.active.discard(batch)
                raise

    def dispatch(self, args, cancellation):
        if not isinstance(args, dict) or not isinstance(op := args.get("op"), str):
            raise invalid("expected { op, ... }")
        if op == "create":
            return self.create(args, cancellation)
        identifier = args.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise invalid("op requires a batch id")
        with self.lock:
            batch = self.batches.get(identifier)
            if batch and op == "close":
                self.batches.pop(identifier)
        if batch is None:
            raise ValueError(f'unknown judge_batch "{identifier}"')
        if op in ("attach", "status"):
            return batch.status()
        if op == "drain":
            timeout = args.get("timeoutMs")
            if "timeoutMs" in args and (not number(timeout) or timeout < 0):
                raise invalid("timeoutMs must be a non-negative finite number")
            return {"items": batch.drain(cancellation, timeout)}
        if op == "results":
            return {"results": batch.collected("answers")}
        if op == "failed":
            return {"failed": batch.collected("error")}
        if op == "cancel":
            return {"cancelled": batch.cancel()}
        if op == "close":
            batch.cancel()
            return {"closed": True}
        raise invalid(f'unknown op "{op}"')

    def close(self):
        with self.lock:
            self.closed = True
            pending = tuple(self.active)
            self.batches.clear()
        for batch in pending:
            batch.cancel()
        for batch in pending:
            batch.finished.wait()
