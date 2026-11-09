"""Process-local eval completion handles with session ownership."""

import json
import math
import re
from concurrent.futures import CancelledError, ThreadPoolExecutor
from contextlib import contextmanager, nullcontext, suppress
from dataclasses import dataclass, field
from threading import BoundedSemaphore, Condition, Lock, Timer
from time import monotonic
from uuid import uuid4

from cancellable_http import turn_scope
from cost_gate import CostCapError
from handoff import Cancellation
import model_catalog
from model_profile import route_base


NAMES = frozenset(("__completion__", "__wait__", "__status__", "__cancel__"))
RETENTION = 30 * 60
REQUEST_LIMIT = 32
WORKERS = ThreadPoolExecutor(max_workers=REQUEST_LIMIT, thread_name_prefix="mini-pi-eval-completion")
REQUEST_SLOTS = BoundedSemaphore(REQUEST_LIMIT)
PAYLOAD = re.compile(r"\{[\s\S]*\}")
TIER_DEFAULT = object()


@contextmanager
def request_slot(cancellation):
    while not REQUEST_SLOTS.acquire(timeout=.05):
        cancellation.check()
    try:
        cancellation.check()
        yield
    finally:
        REQUEST_SLOTS.release()


class CellCancellation(Cancellation):
    def __init__(self, parent=None):
        super().__init__(parent, reason="Eval cell aborted.")
        self.handles = Cancellation(parent, reason="Eval cell aborted.")
        self.waits = Condition()
        self.agent_waits = 0
        self.waits_closed = False

    def cancel(self):
        self.handles.cancel()
        super().cancel()

    def finish(self):
        # A settled cell drops nested calls while retained handles can finish later.
        super().cancel()

    @contextmanager
    def wait_scope(self):
        with self.waits:
            if self.waits_closed:
                self.check()
            self.agent_waits += 1
        try:
            yield
        finally:
            with self.waits:
                self.agent_waits -= 1
                self.waits.notify_all()

    def settle_waits(self):
        with self.waits:
            self.waits_closed = True
            self.waits.wait_for(lambda: not self.agent_waits)


def format_model(route):
    model = route.catalog_model or {}
    if model.get("provider"):
        if model_catalog.upstream_pin(model):
            return route_base(route)
        return f"{model['provider']}/{route.model}"
    return f"{route.broker_provider}/{route.model}" if route.broker_provider else route.model


def effective_effort(route, tier, explicit, parent=TIER_DEFAULT):
    if explicit in ("off", "none"):
        return "off"
    model = route.catalog_model
    if explicit not in (None, "inherit", "auto"):
        return model_catalog.thinking_effort(model, explicit)
    if parent is not TIER_DEFAULT:
        return "off" if parent == "off" else model_catalog.thinking_effort(model, parent) if parent else None
    if tier != "slow" or not model or not model.get("reasoning"):
        return None
    levels = (model.get("thinking") or {}).get("efforts") or ()
    if not levels:
        return None
    return model_catalog.thinking_effort(model, "high" if "high" in levels else levels[-1])


def candidates(router, profile, tier, fallback):
    if tier == "default" and router.route:
        primary = router.route
    elif profile and (names := profile.choices("@" + tier)):
        primary = router.routes[names[0]]
    else:
        return ()

    def explicit(route):
        return profile.split(route.name)[1] if profile else None

    def key(route, effort):
        return route_base(route), route.endpoint(), route.api, effort

    selected = explicit(primary) if tier == "slow" else None
    first = effective_effort(primary, tier, selected)
    result = [(primary, first)]
    seen = {key(primary, first)}
    expanded = set()

    def expand(route, effort, inherit, hint=None):
        visit = route.name, effort, inherit
        if visit in expanded:
            return
        expanded.add(visit)
        names = []
        if profile and (chain := profile.chain_key(route, effort, hint)):
            names.extend(profile.candidates(chain, route, effort))
        names.extend(route.fallbacks)
        for name in names:
            child = router.routes[name]
            value = effective_effort(child, tier, explicit(child), effort if inherit else TIER_DEFAULT)
            identity = key(child, value)
            if identity in seen:
                continue
            seen.add(identity)
            result.append((child, value))
            expand(child, value, True)

    if fallback:
        expand(primary, first, selected not in (None, "auto", "inherit"), tier)
    return tuple(result)


def completion_result(reply, schema):
    text = (reply.get("content") or "").strip()
    if schema is None:
        if not text:
            raise ValueError("completion() returned no text output.")
        return {"text": text}
    call = next((item["function"] for item in reply.get("tool_calls", ()) if item["function"]["name"] == "respond"), None)
    if not call and not text:
        raise ValueError("completion() returned no structured response.")
    try:
        if call:
            data = json.loads(call["arguments"])
        elif match := PAYLOAD.search(text):
            data = json.loads(match[0])
        else:
            raise ValueError()
        if not isinstance(data, dict):
            raise ValueError()
        return {"text": json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)}
    except (ValueError, TypeError, RecursionError):
        raise ValueError("completion() did not return a structured response matching the schema.") from None


@dataclass
class Completion:
    identifier: str
    signal: Cancellation
    future: object = None
    expires: float | None = None
    attempts: list = field(default_factory=list)
    lock: object = field(default_factory=Lock)

    def attempt(self, route, effort):
        value = {"model": format_model(route), "effort": effort, "outcome": "running"}
        with self.lock:
            self.attempts.append(value)
        return value

    def outcome(self, attempt, state, error=None):
        with self.lock:
            attempt["outcome"] = state
            if error is not None:
                attempt["error"] = str(error) or type(error).__name__

    agent_id = None

    def settled(self):
        return self.future.done()

    def join(self):
        with suppress(Exception):
            self.future.result()

    def snapshot(self, *, attempts=False, consume=False):
        value = {"kind": "completion", "id": self.identifier, "status": "running"}
        if self.future.done():
            try:
                value.update(self.future.result(), status="completed")
            except Exception as error:
                state = "cancelled" if isinstance(error, CancelledError) else "failed"
                value.update(status=state, error=str(error) or "Completion cancelled.")
        if attempts:
            with self.lock:
                value["attempts"] = [dict(attempt) for attempt in self.attempts]
        return value

    def cancel(self):
        if self.future.done():
            return False
        self.signal.cancel()
        self.future.cancel()
        return True


class CompletionBridge:
    def __init__(self, select, credential, send, admit, retries):
        self.select, self.credential, self.send, self.admit = select, credential, send, admit
        self.retries = retries
        self.handles = {}
        self.lock = Lock()
        self.reaper = None
        self.closed = False
        self.agents = None

    def start(self, args, cancel):
        if not isinstance(args, dict):
            raise ValueError("completion() requires an argument object.")
        prompt, system, schema = (args.get(key) for key in ("prompt", "system", "schema"))
        tier = args.get("model", "default")
        if (not isinstance(prompt, str) or not prompt or tier not in ("smol", "default", "slow")
                or ("system" in args and not isinstance(system, str))
                or ("schema" in args and not isinstance(schema, dict))):
            raise ValueError("Invalid completion() arguments. Provide a prompt, a model tier and optional system text or schema.")
        cancel.check()
        routes = self.select(tier)
        if not routes:
            raise ValueError(f"completion() could not resolve a model for the {tier!r} tier.")
        identifier = "cmp-" + uuid4().hex
        handle = Completion(identifier, Cancellation(cancel.handles, reason="Cancelled by eval completion handle."))
        with self.lock:
            if self.closed:
                raise ValueError("The eval session has closed.")
            handle.future = WORKERS.submit(self.run, handle, routes, prompt, system, schema)
            self.handles[identifier] = handle
        handle.future.add_done_callback(lambda _: self.settled(handle))
        return {"id": identifier}

    def run(self, handle, routes, prompt, system, schema):
        last_error = None
        used = 0
        with request_slot(handle.signal), turn_scope(handle.signal):
            for index, (route, effort) in enumerate(routes):
                handle.signal.check()
                if index and used >= self.retries:
                    break
                self.admit()
                attempt = handle.attempt(route, effort)
                try:
                    credential = self.credential(route)
                    handle.signal.check()
                    if credential is None:
                        handle.outcome(attempt, "no-credentials")
                        last_error = ValueError(f"completion() has no API key for {format_model(route)}.")
                        continue
                    if index:
                        used += 1
                    reply = self.send(route, credential, effort, prompt, system, schema, handle.signal)
                    handle.signal.check()
                except Exception as error:
                    state = "cancelled" if handle.signal.is_cancelled() or isinstance(error, CancelledError) else "failed"
                    handle.outcome(attempt, state, error)
                    if state == "cancelled":
                        raise CancelledError("completion() request aborted.") from error
                    if isinstance(error, CostCapError):
                        raise
                    last_error = error
                    continue
                try:
                    value = completion_result(reply, schema)
                except ValueError as error:
                    handle.outcome(attempt, "failed", error)
                    raise
                handle.outcome(attempt, "succeeded")
                return value
        raise last_error or ValueError("completion() request failed.")

    def settled(self, handle):
        with self.lock:
            if self.closed:
                return
            handle.expires = monotonic() + RETENTION
            self.arm()

    def arm(self):
        if self.reaper is None:
            expires = min((handle.expires for handle in self.handles.values() if handle.expires is not None), default=None)
            if expires is not None:
                self.reaper = Timer(max(.001, expires - monotonic()), self.expire)
                self.reaper.daemon = True
                self.reaper.start()

    def expire(self):
        with self.lock:
            self.reaper = None
            if not self.closed:
                now = monotonic()
                self.handles = {identifier: handle for identifier, handle in self.handles.items()
                                if handle.expires is None or handle.expires > now}
                self.arm()

    def resolve(self, item):
        if (not isinstance(item, dict) or item.get("kind") not in ("agent", "completion")
                or not isinstance(item.get("id"), str) or not item["id"]):
            raise ValueError("Expected an agent or completion handle from this session.")
        identifier = item["id"]
        if item["kind"] == "agent":
            if self.agents:
                return self.agents.resolve(identifier)
            raise ValueError(f"Unknown agent handle {identifier!r}.")
        with self.lock:
            handle = self.handles.get(identifier)
            if handle and (handle.expires is None or handle.expires > monotonic()):
                return handle
        raise ValueError(f"Unknown completion handle {identifier!r}. Its result may have expired.")

    def wait(self, args, cancel, progress):
        items = args.get("items") if isinstance(args, dict) else None
        if not isinstance(items, list):
            raise ValueError("wait() requires an items array.")
        handles = [self.resolve(item) for item in items]
        milliseconds = args.get("timeoutMs", 0)
        if "timeoutMs" in args and (type(milliseconds) not in (int, float) or milliseconds < 0
                                    or (type(milliseconds) is float and not math.isfinite(milliseconds))):
            raise ValueError("wait() timeoutMs must be a nonnegative finite number.")
        end = monotonic() + min(milliseconds, 10**20) / 1000 if "timeoutMs" in args else None
        peers = self.agents.tasks.peer if self.agents and self.agents.tasks else None
        agents = [handle for handle in handles if handle.agent_id]
        previous = {}

        def emit():
            if progress and agents:
                changes = []
                for handle in agents:
                    update = handle.progress()
                    if update and previous.get(handle.agent_id) != update:
                        previous[handle.agent_id] = update
                        changes.append(update)
                if changes:
                    progress(changes)

        with cancel.wait_scope() if agents else nullcontext():
            hold = peers and agents and end is None
            if hold:
                peers.wait_for({handle.agent_id for handle in agents if not handle.settled()})
            try:
                while True:
                    cancel.check()
                    emit()
                    if all(handle.settled() for handle in handles):
                        break
                    remaining = end - monotonic() if end is not None else .05
                    if remaining <= 0:
                        break
                    cancel.wait(min(.05, remaining))
            except CancelledError:
                for handle in handles:
                    handle.cancel()
                for handle in handles:
                    handle.join()
                raise
            finally:
                if hold:
                    peers.wait_for(set())
            return {"items": [handle.snapshot(consume=True) for handle in handles]}

    def dispatch(self, name, args, cancel, progress=None):
        if name == "__completion__":
            return self.start(args, cancel)
        if name == "__wait__":
            return self.wait(args, cancel, progress)
        handle = self.resolve(args.get("item") if isinstance(args, dict) else None)
        return handle.snapshot(attempts=True) if name == "__status__" else {"cancelled": handle.cancel()}

    def close(self):
        with self.lock:
            self.closed = True
            if self.reaper:
                self.reaper.cancel()
                self.reaper = None
            handles = tuple(self.handles.values())
            self.handles.clear()
        for handle in handles:
            handle.cancel()
        for handle in handles:
            handle.join()
