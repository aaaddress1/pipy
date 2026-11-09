"""Claim user prompts during a Codex exchange and reconcile its successor."""

from time import monotonic

import httpx
from websockets.exceptions import WebSocketException

from snapcompact import responses_content


class LiveQueue:
    def __init__(self, turn, queue, mode, changed):
        self.turn, self.queue, self.mode, self.changed = turn, queue, mode, changed
        self.pending, self.accepted, self.deferred = [], [], []

    def claim(self):
        with self.turn.server.lock:
            self.turn.check()
            assert not self.pending
            while self.queue:
                work = self.queue[0]
                if (not isinstance(work.content, (str, list)) or getattr(work, "operation", None)
                    or getattr(work, "background", False) or getattr(work, "on_drop", None)):
                    break
                self.pending.append(self.queue.popleft())
                if self.mode() != "all":
                    break
            if self.pending:
                self.changed()
            return [work.content for work in self.pending]

    def settle(self, accepted):
        with self.turn.server.lock:
            if self.turn.is_cancelled() or not self.pending:
                return
            if accepted:
                self.accepted.extend(self.pending)
            elif self.accepted:
                self.deferred.extend(self.pending)
            else:
                self.queue.extendleft(reversed(self.pending))
            self.pending.clear()
            self.changed()

    def drain(self):
        accepted, self.accepted = self.accepted, []
        deferred, self.deferred = self.deferred, []
        return accepted, deferred

    def take_all(self):
        pending, self.pending = self.pending, []
        accepted, deferred = self.drain()
        return accepted + deferred + pending

    def rollback(self):
        with self.turn.server.lock:
            if self.turn.is_cancelled():
                return
            if works := self.take_all():
                self.queue.extendleft(reversed(works))
                self.changed()


def steer_item(content, lite, vision):
    if not vision and isinstance(content, list):
        content = "\n".join(part["text"] if part["type"] == "text" else "[Image omitted for text-only model.]"
                            for part in content)
    content = responses_content(content)
    if lite and isinstance(content, list):
        content = [{key: value for key, value in part.items() if key != "detail"} for part in content]
    return {"type": "message", "role": "user", "content": content}


def steer_key(item):
    if item.get("role") != "user":
        return None
    content = item.get("content")
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]
    if isinstance(content, list):
        return [{key: value for key, value in part.items() if key != "detail"} for part in content]
    return content


def plan_successor(delta, accepted):
    if delta is None:
        return "discard", None
    keys = iter(steer_key(item) for _, items in accepted for item in items)
    missing = next(keys, None)
    rest = []
    for item in delta:
        if missing is not None and steer_key(item) == missing:
            missing = next(keys, None)
        else:
            rest.append(item)
    if missing is not None:
        return "discard", None
    if not rest:
        return "attach", None
    if any((item.get("type") or "").endswith("_output") for item in rest):
        return "create", rest
    return "discard", None


class SteerExchange:
    def __init__(self, socket, source, lite, vision, reopen):
        self.socket, self.source, self.lite, self.reopen = socket, source, lite, reopen
        self.vision = vision
        self.response_id = self.pending = None
        self.accepted = []
        self.carried = set()
        self.stopped = self.uncertain = self.finished = False

    def start(self, request, wire, delta):
        if not self.socket.accepted_steers:
            return wire, delta
        accepted, self.socket.accepted_steers = self.socket.accepted_steers, []
        plan, remaining = plan_successor(delta, accepted)
        if plan == "discard":
            self.socket.close()
            self.reopen()
            return request, None
        self.carried = {identifier for identifier, _ in accepted}
        if plan == "attach":
            return None, delta
        return {**wire, "input": remaining}, delta

    def submit(self, turn):
        if self.pending is not None:
            if monotonic() >= self.ack_deadline:
                self.refuse(uncertain=True)
            return
        if self.stopped or self.source is None or self.response_id is None or self.socket.steering_unsupported:
            return
        if contents := self.source.claim():
            self.pending = [steer_item(content, self.lite, self.vision) for content in contents]
            self.ack_deadline = monotonic() + 10
            self.socket.send({"type": "response.steer", "previous_response_id": self.response_id,
                              "input": self.pending}, turn)

    def refuse(self, *, uncertain=False):
        self.source.settle(False)
        self.pending = None
        self.stopped = True
        self.uncertain |= uncertain

    def receive(self, turn, timeout):
        deadline = monotonic() + timeout if timeout is not None else None
        while True:
            self.submit(turn)
            if self.socket.steer_frames:
                return self.socket.steer_frames.popleft()
            remaining = deadline - monotonic() if deadline is not None else None
            if remaining is not None and remaining <= 0:
                raise TimeoutError("Codex response timed out.")
            polling = self.pending is not None or (self.source is not None and self.response_id is not None
                                                    and not self.stopped and not self.socket.steering_unsupported)
            try:
                wait = (min(0.1, remaining) if remaining is not None else 0.1) if polling else remaining
                return self.socket.receive(turn, wait)
            except TimeoutError:
                if not polling:
                    raise

    def control(self, event):
        kind = event["type"]
        if kind == "response.created":
            response_id = (event.get("response") or {}).get("id")
            if isinstance(response_id, str):
                self.response_id = response_id
        if not kind.startswith("response.steer."):
            return False
        steer = event.get("steer") or {}
        identifier = steer.get("id")
        if kind == "response.steer.failed" and (
            identifier in self.carried or any(identifier == known for known, _ in self.accepted)
        ):
            raise httpx.RemoteProtocolError("Codex rejected previously accepted steering.")
        if self.pending is None or steer.get("previous_response_id") != self.response_id:
            return True
        if kind == "response.steer.accepted" and identifier:
            self.accepted.append((identifier, self.pending))
            self.pending = None
            self.source.settle(True)
        elif kind == "response.steer.failed":
            if (event.get("error") or {}).get("code") == "unsupported_native_inflight_message":
                self.socket.steering_unsupported = True
            self.refuse()
        return True

    def terminal(self, baseline, turn):
        while self.pending is not None:
            try:
                event = self.socket.receive(turn, max(0, self.ack_deadline - monotonic()))
            except (OSError, TimeoutError, WebSocketException, httpx.RemoteProtocolError):
                self.refuse(uncertain=True)
                break
            if not event["type"].startswith("response.steer."):
                if len(self.socket.steer_frames) >= 4096:
                    self.refuse(uncertain=True)
                    break
                self.socket.steer_frames.append(event)
            else:
                self.control(event)
        self.finished = True
        if self.uncertain:
            return False
        if self.accepted:
            if baseline is None or baseline[0] != self.response_id:
                return False
            self.socket.accepted_steers = self.accepted
        return True

    def abort(self):
        if not self.finished and self.source:
            self.source.rollback()
