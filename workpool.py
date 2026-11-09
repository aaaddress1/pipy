"""Process-local work pools over retained child sessions."""

import logging
import unicodedata
from collections import Counter, deque
from dataclasses import dataclass, field, replace
from time import monotonic, time

from cancellable_http import turn_scope


NAME = "__workpool__"
STATES = ("queued", "running", "completed", "failed", "cancelled")
logger = logging.getLogger(__name__)


def optional_text(args, key):
    if key in args and (not isinstance(args[key], str) or not args[key].strip()):
        raise ValueError(f"workpool {key} must be a non-empty string")
    return args[key].strip() if key in args else None


def label(text, limit=80):
    line = " ".join("".join(" " if unicodedata.category(char) in ("Cc", "Cf") else char for char in text).split())
    return line if len(line) <= limit else line[:limit - 1] + "…"


def batch_message(pool, batch):
    safe = lambda text: text.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")
    lines = [
        f'<workpool pool="{safe(pool)}" batch="{safe(batch.id)}">',
        f"You are a worker in pool `{pool}`. Complete every item below in order. "
        "After EACH item, call \x60yield\x60 once as \x60{ key, data }\x60 or \x60{ key, error }\x60, "
        "where \x60key\x60 is the item's 1-based number and \x60data\x60 is its self-contained outcome/evidence value. "
        "The tool tells you which keys remain; the final key ends the turn automatically. NEVER combine several items into one yield.",
    ]
    for index, item in enumerate(batch.items, 1):
        lines.extend((f"## Item {index}", item.text))
    return "\n".join([*lines, "</workpool>"])


@dataclass(eq=False)
class Item:
    id: str
    text: str
    status: str = "queued"
    queued_notice: bool = False


@dataclass(eq=False)
class Worker:
    task: object
    turns: int = 0
    queue: deque = field(default_factory=deque)
    current: str | None = None
    tokens: int | None = None
    window: int | None = None

    def load(self):
        tokens = self.tokens or 0
        return tokens / self.window if self.window and self.window > 0 else tokens

    def status(self):
        return {"id": self.task.name, "state": "running" if self.current else "idle",
                "queued": len(self.queue), "turns": self.turns,
                **({"contextTokens": self.tokens} if self.tokens is not None else {}),
                **({"contextWindow": self.window} if self.window is not None else {}),
                **({"current": self.current} if self.current else {})}


@dataclass
class Batch:
    id: str
    worker: Worker
    items: tuple
    started: float = field(default_factory=monotonic)
    job: object = None
    status: str = "running"
    output: str | None = None

    def snapshot(self):
        return {"id": self.id, "agent": self.worker.task.name, "items": [item.id for item in self.items],
                "status": self.status, **({"output": self.output} if self.output is not None else {})}


class WorkerBusy(RuntimeError):
    pass


class WorkPool:
    def __init__(self, bridge, name, agent, model, context, custom, limit, fresh, observe):
        self.bridge, self.name, self.agent, self.model = bridge, name, agent, model
        self.context, self.custom, self.limit, self.fresh = context, custom, limit, fresh
        self.observe = observe
        self.tasks = bridge.tasks
        self.jobs = self.tasks.jobs
        self.condition = self.jobs.condition
        self.closed = self.started = False
        self.items, self.batches, self.workers = [], [], []
        self.unassigned = deque()
        self.active = {}
        self.next_worker = 1
        self.rr = self.last_card = 0

    def push(self, texts):
        with self.condition:
            if self.closed:
                raise ValueError(f"workpool {self.name} is closed")
            if not texts:
                return []
            if not self.started:
                if self.name in self.jobs.jobs:
                    raise ValueError(f'workpool job id "{self.name}" already exists')
                self.jobs.spawn([(self.name, self.name, None, self.aggregate)], lambda: None, queued=True)
                self.started = True
            first = len(self.items) + 1
            added = [Item(f"{self.name}#{seq}", text) for seq, text in enumerate(texts, first)]
            self.items.extend(added)
            self.unassigned.extend(added)
            self.condition.notify_all()
            return [item.id for item in added]

    def status(self):
        with self.condition:
            counts = Counter(item.status for item in self.items)
            return {"name": self.name, "agent": self.agent, "limit": self.limit(), "closed": self.closed,
                    "freshAgents": self.fresh, "agents": [worker.status() for worker in self.workers],
                    "items": {state: counts[state] for state in STATES}, "batches": len(self.batches)}

    def peek(self):
        with self.condition:
            return {"batches": [batch.snapshot() for batch in self.batches], "pending": self.pending()}

    def pending(self):
        return sum(item.status in ("queued", "running") for item in self.items)

    def close(self):
        with self.condition:
            self.closed = True
            dropped = []
            for item in self.items:
                if item.status == "queued":
                    item.status = "cancelled"
                    dropped.append(item.id)
            self.unassigned.clear()
            for worker in self.workers:
                worker.queue.clear()
            self.condition.notify_all()
            return {"dropped": dropped}

    def card(self, mode, agent, body):
        if not self.observe:
            return
        self.last_card = max(round(time() * 1000), self.last_card + 1)
        try:
            self.observe({"role": "custom", "customType": "irc:workpool", "content": f"[pool {self.name} → {agent}]\n\n{body}",
                          "display": True, "details": {"pool": self.name, "from": f"pool:{self.name}", "to": agent,
                                                        "body": body, "mode": mode},
                          "attribution": "agent", "timestamp": self.last_card})
        except Exception:
            logger.debug("Could not publish workpool progress", exc_info=True)

    def busy(self, worker):
        peer = self.tasks.peer
        if peer:
            with peer.bus.lock:
                return peer.bus.active(worker.task.sid)
        return False

    def reap(self):
        for identifier, batch in tuple(self.active.items()):
            job = batch.job
            if job.status == "running":
                continue
            del self.active[identifier]
            details = job.details or {}
            batch.status = job.status
            batch.output = details.get("workpoolOutput") or job.text
            errors = set(details.get("workpoolItemErrors", ()))
            for item in batch.items:
                item.status = "failed" if batch.status == "completed" and item.id in errors else batch.status
            worker = batch.worker
            worker.turns += 1
            worker.current = None
            worker.tokens = details.get("contextTokens", worker.tokens)
            worker.window = details.get("contextWindow", worker.window)
            if self.fresh or not details.get("workpoolReusable") or batch.status == "cancelled":
                self.workers.remove(worker)
                self.unassigned.extend(worker.queue)
                worker.queue.clear()

    def assign(self):
        # A retained worker drains its accumulated queue as one next turn.
        for worker in self.workers:
            if worker.queue and not worker.current and not self.busy(worker) and self.jobs.available():
                self.start_batch(worker, tuple(worker.queue))
                worker.queue.clear()
                return True, None
        if not self.unassigned:
            return False, None
        ceiling = self.limit()
        if not self.fresh:
            idle = min((worker for worker in self.workers if not worker.current and not worker.queue and not self.busy(worker)),
                       key=Worker.load, default=None)
            if idle and self.jobs.available():
                item = self.unassigned[0]
                self.start_batch(idle, (item,))
                self.unassigned.popleft()
                self.card("dispatched", idle.task.name, f"[{item.id}] {item.text}")
                return True, None
        if ceiling is None or len(self.workers) < ceiling:
            return (True, self.unassigned.popleft()) if self.jobs.available() else (False, None)
        if self.fresh:
            for item in self.unassigned:
                if not item.queued_notice:
                    item.queued_notice = True
                    self.card("queued", self.name, f"[{item.id}] {item.text}")
            return False, None
        for offset in range(len(self.workers)):
            index = (self.rr + offset) % len(self.workers)
            worker = self.workers[index]
            if worker.current:
                self.rr = (index + 1) % len(self.workers)
                item = self.unassigned.popleft()
                worker.queue.append(item)
                self.card("queued", worker.task.name, f"[{item.id}] {item.text}")
                return True, None
        return False, None

    def spawn_worker(self, item, signal):
        with self.tasks.dispatch_lock:
            try:
                with turn_scope(signal):
                    signal.check()
                    values = {"agent": self.agent, "label": f"{self.name}-{self.next_worker}", "prompt": item.text}
                    if self.model is not None:
                        values["model"] = self.model
                    task = self.bridge.prepare(values)
                if self.custom:
                    task = replace(task, eval_tools=self.custom, eval_tool_owner=self.tasks.eval_tools.id)
                task = replace(task, context=self.context, workpool=True)
                worker = Worker(task)
                with self.condition:
                    if self.closed or signal.is_cancelled() or self.jobs.closed:
                        item.status = "cancelled"
                    elif not self.jobs.available():
                        self.unassigned.appendleft(item)
                    else:
                        self.start_batch(worker, (item,), first=True)
                        self.next_worker += 1
                        self.workers.append(worker)
                        self.card("spawned", task.name, f"[{item.id}] {item.text}")
            except Exception as error:
                with self.condition:
                    item.status = "cancelled" if self.closed or signal.is_cancelled() else "failed"
                if item.status == "failed":
                    logger.warning("Workpool item %s could not be dispatched: %s", item.id, error)

    def start_batch(self, worker, items, *, first=False):
        batch = Batch(f"{worker.task.name}-b{worker.turns + 1}", worker, items)
        identifier, suffix = batch.id, 2
        while identifier in self.jobs.jobs:
            identifier = f"{batch.id}-{suffix}"
            suffix += 1
        peer = self.tasks.peer

        def reserve():
            if first:
                self.tasks.reserve([worker.task])
            elif peer:
                with peer.bus.lock:
                    if peer.bus.active(worker.task.sid):
                        raise WorkerBusy()
                    peer.bus.mark(worker.task.sid)

        self.jobs.spawn([(identifier, identifier, worker.task.sid, lambda cancel: self.run_batch(batch, cancel))],
                        reserve, suppress_delivery=True)
        batch.job = self.jobs.jobs[identifier]
        for item in items:
            item.status = "running"
        worker.current = identifier
        self.batches.append(batch)
        self.active[identifier] = batch
        if worker.turns:
            self.card("batch", worker.task.name, batch_message(self.name, batch))

    def run_batch(self, batch, signal):
        worker = batch.worker
        keys = tuple(item.id for item in batch.items)
        schema = {"type": "object", "properties": dict.fromkeys(keys, {}), "required": list(keys), "additionalProperties": False}
        message = batch_message(self.name, batch)
        task = replace(worker.task, assignment=message, structured=True, schema=schema, schema_mode="strict", eval_schema_source="caller")

        def progress(runtime):
            with self.condition:
                worker.tokens, worker.window = runtime.context_tokens, runtime.context_window
                self.jobs.progress(batch.job.id, {
                    "totalDurationMs": round((monotonic() - batch.started) * 1000),
                    "progress": [{"id": task.name, "agent": task.agent.name, "requests": runtime.count,
                                  "contextTokens": worker.tokens, "contextWindow": worker.window}],
                }, f"Running agent {task.name}...")

        result, value = self.tasks.run_task(task, signal, False, progress=progress, pool_items=keys, pool_followup=bool(worker.turns))
        result = self.bridge.finish(task, result, value)
        text = (result.get("evalResult", {}).get("text") or result.get("error") or "(no output)")
        report = self.render_batch(batch, result, text)
        return {**result, "report": report, "workpoolOutput": text}

    def render_batch(self, batch, result, text):
        status, errors = result["status"], set(result.get("workpoolItemErrors", ()))
        with self.condition:
            remaining = self.pending() - len(batch.items)
        output = text.strip()
        if len(output) > 6000:
            output = output[:6000] + "\n[output truncated to 6000 characters]"
        statuses = [f"- [{item.id}] {'failed' if status == 'completed' and item.id in errors else status} — {label(item.text)}"
                    for item in batch.items]
        count = len(batch.items)
        suffix = f"{remaining} item(s) still queued or running in this pool." if remaining else "Pool queue drained."
        return "\n".join([f"Pool \x60{self.name}\x60 · agent \x60{batch.worker.task.name}\x60 · batch \x60{batch.id}\x60 "
                          f"{status} ({count} item{'s' if count != 1 else ''}):", *statuses, output,
                          f"{suffix} Transcript: history://{batch.worker.task.name} · full output: agent://{batch.worker.task.name}"])

    def aggregate(self, signal):
        while True:
            candidate = None
            with self.condition:
                self.reap()
                if signal.is_cancelled() or self.jobs.closed:
                    self.close()
                    for batch in self.active.values():
                        batch.job.cancel.cancel()
                if not self.active and not self.unassigned and not any(worker.queue for worker in self.workers):
                    self.closed = True
                    status = "cancelled" if signal.is_cancelled() else "completed"
                    summary = f"Pool \x60{self.name}\x60 drained: {len(self.items)} item(s), {len(self.batches)} batch(es)."
                    self.card(status, self.tasks.session.id, summary)
                    return {"status": status, "report": self.render_aggregate(), "workpool": self.status()}
                if not self.closed:
                    try:
                        changed, candidate = self.assign()
                    except WorkerBusy:
                        changed = False
                    if changed and candidate is None:
                        continue
                if candidate is None:
                    self.condition.wait(.1)
                    continue
            self.spawn_worker(candidate, signal)

    def render_aggregate(self):
        lines = [f"Pool \x60{self.name}\x60 completed ({len(self.items)} item(s), {len(self.batches)} batch(es))."]
        for batch in self.batches:
            agent = batch.worker.task.name
            lines.extend(("", f"## {batch.id} · agent \x60{agent}\x60 · {batch.status}",
                          *(f"- [{item.id}] {item.status} — {label(item.text)}" for item in batch.items)))
            if batch.output:
                lines.extend(("", batch.output.strip()))
            lines.append(f"Transcript: history://{agent} · full output: agent://{agent}")
        lines.extend(("", "Pool queue drained."))
        return "\n".join(lines)


class WorkPools:
    def __init__(self, bridge, limit, fresh, observe=None):
        self.bridge, self.limit, self.fresh, self.observe = bridge, limit, fresh, observe
        self.pools = {}

    def create(self, args, cancellation):
        agent, name, context, model = (optional_text(args, key) for key in ("agent", "name", "context", "model"))
        names = args.get("tools", [])
        if not isinstance(names, list) or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("workpool tools must be an array of non-empty strings")
        tasks = self.bridge.tasks
        if not tasks:
            raise ValueError(self.bridge.unavailable)
        request = {"prompt": f"Create workpool {name or agent or 'worker'}"}
        if agent:
            request["agent"] = agent
        if model:
            request["model"] = model
        with turn_scope(cancellation):
            cancellation.check()
            policy = self.bridge.prepare(request)
        custom = self.bridge.custom_tools(names, cancellation)
        if not tasks.jobs or not tasks.background:
            raise ValueError("workpool() needs the session's async job manager; unavailable here")
        with tasks.jobs.condition:
            cancellation.check()
            if tasks.jobs.closed:
                raise ValueError("The eval session has closed.")
            tasks.jobs.prune()
            if name is None:
                base, suffix = policy.agent.name + "-pool", 2
                name = base
                while name in self.pools or name in tasks.jobs.jobs:
                    name = f"{base}-{suffix}"
                    suffix += 1
            if name in self.pools:
                raise ValueError(f'workpool "{name}" already exists')
            if name in tasks.jobs.jobs:
                raise ValueError(f'workpool job id "{name}" already exists')
            pool = WorkPool(self.bridge, name, policy.agent.name, model, context or "", custom,
                            self.limit, self.fresh(), self.observe)
            self.pools[name] = pool
            return {"name": name, "agent": pool.agent, "limit": pool.limit()}

    def dispatch(self, args, cancellation):
        if not isinstance(args, dict) or not isinstance(op := args.get("op"), str):
            raise ValueError("workpool() requires an op")
        cancellation.check()
        if op == "create":
            return self.create(args, cancellation)
        name = optional_text(args, "name")
        if name is None:
            raise ValueError("workpool operation requires a non-empty name")
        pool = self.pools.get(name)
        if pool is None:
            raise ValueError(f'unknown workpool "{name}"')
        if op == "push":
            items = args.get("items")
            if not isinstance(items, list) or any(not isinstance(item, str) for item in items):
                raise ValueError("workpool push requires an items string array")
            return {"ids": pool.push(items)}
        if op == "status":
            return pool.status()
        if op == "peek":
            return pool.peek()
        if op == "close":
            return pool.close()
        raise ValueError(f'unknown workpool operation "{op}"')

    def close(self):
        for pool in self.pools.values():
            pool.close()
        self.pools.clear()
