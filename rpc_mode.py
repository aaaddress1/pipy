"""JSONL RPC for driving the existing agent loop from a host process."""

import base64
import json
import sys
from collections import deque
from concurrent.futures import CancelledError
from contextlib import contextmanager
from dataclasses import dataclass
from tempfile import SpooledTemporaryFile
from threading import Condition, RLock, Thread
from time import time

from openai import APIStatusError

from cost_gate import CostCapError
from handoff import Cancellation
from live_steering import LiveQueue
from session_jsonl import export_message
from thinking_loop import ThinkingLoop


MAX_FRAME = 1024 * 1024
MAX_LOGICAL = 64 * MAX_FRAME
CHUNK_BYTES = 720 * 1024
MAX_QUEUE = 256
API_NAMES = {"chat-completions": "openai-completions", "openai-responses": "openai-responses",
             "anthropic-messages": "anthropic-messages", "openai-codex-responses": "openai-codex-responses",
             "bedrock-converse-stream": "bedrock-converse-stream"}


def model_metadata(route, usage):
    model = getattr(usage, "served_model", None) or route.model
    provider = (route.catalog_model or {}).get("provider") or route.provider() if route.api == "bedrock-converse-stream" else route.name
    metadata = {"api": API_NAMES[route.api], "provider": provider, "model": model,
                "timestamp": int(time() * 1000)}
    if usage:
        inputs = getattr(usage, "prompt_tokens", 0) or 0
        outputs = getattr(usage, "completion_tokens", 0) or 0
        details = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(usage, "cache_read_tokens", 0) or getattr(details, "cached_tokens", 0) or 0
        written = getattr(usage, "cache_write_tokens", 0) or 0
        metadata["usage"] = {"input": max(0, inputs - cached - written), "output": outputs, "cacheRead": cached,
                             "cacheWrite": written, "totalTokens": inputs + outputs}
        if (cost := getattr(usage, "cost", None)) is not None:
            metadata["usage"]["cost"] = cost
    return metadata


def encoded(value):
    if value.get("type") == "response" and isinstance(value.get("data"), dict) and "tree" in value["data"]:
        tree = value["data"]["tree"]
        payload = {key: item for key, item in value.items() if key != "data"}
        raw = bytearray(encoded(payload)[:-1] + b',"data":{')
        metadata = {key: item for key, item in value["data"].items() if key != "tree"}
        if metadata:
            raw.extend(encoded(metadata)[1:-1] + b",")
        raw.extend(b'"tree":[')
        # Session trees can be deeper than Python's recursive JSON encoder.
        stack = [(iter(tree), True)]
        while stack:
            siblings, first = stack[-1]
            node = next(siblings, None)
            if node is None:
                stack.pop()
                raw.extend(b"]}" if stack else b"]}}")
                continue
            if not first:
                raw.extend(b",")
            stack[-1] = (siblings, False)
            raw.extend(encoded({key: item for key, item in node.items() if key != "children"})[:-1])
            raw.extend(b',"children":[')
            stack.append((iter(node["children"]), True))
        return bytes(raw)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8", errors="backslashreplace")


class RpcOutput:
    """Serialize all frames into one drain, spilling queued bytes under backpressure."""

    def __init__(self, stream, *, chunked=True):
        self.stream = stream
        self.chunked = chunked
        self.version = 1
        self.chunk_id = 0
        self.spool = SpooledTemporaryFile(max_size=512 * 1024)
        self.condition = Condition()
        self.read_pos = self.write_pos = 0
        self.closed = False
        self.failure = None
        self.thread = Thread(target=self.drain, name="mini-pi-rpc-output", daemon=True)
        self.thread.start()

    def write(self, frame):
        with self.condition:
            if self.closed or self.failure:
                return
            data = encoded(frame)
            if not self.chunked:
                self.append(data)
                self.condition.notify()
                return
            if self.version == 1 and len(data) + 1 > MAX_FRAME and frame["type"] == "agent_end":
                frame = {**frame, "messageCount": len(frame["messages"]), "messages": []}
                data = encoded(frame)
            if len(data) + 1 <= MAX_FRAME:
                self.append(data)
            elif self.version == 2 and len(data) <= MAX_LOGICAL:
                self.chunk_id += 1
                count = (len(data) + CHUNK_BYTES - 1) // CHUNK_BYTES
                for index in range(count):
                    chunk = data[index * CHUNK_BYTES:(index + 1) * CHUNK_BYTES]
                    self.append(encoded({
                        "type": "rpc_chunk", "chunkId": f"rpc-{self.chunk_id}", "index": index,
                        "count": count, "byteLength": len(data), "data": base64.b64encode(chunk).decode("ascii"),
                    }))
            elif frame["type"] == "response":
                replacement = {key: frame[key] for key in ("type", "id", "command") if key in frame}
                self.append(encoded({**replacement, "success": False, "error": "RPC response exceeded the transport limit"}))
            else:
                self.append(encoded({"type": "rpc_frame_error", "originalType": frame["type"], "error": "RPC frame exceeded the transport limit"}))
            self.condition.notify()

    def append(self, data):
        self.spool.seek(self.write_pos)
        self.spool.write(data + b"\n")
        self.write_pos = self.spool.tell()

    def drain(self):
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: self.closed or self.read_pos < self.write_pos)
                    if self.read_pos == self.write_pos:
                        return
                    self.spool.seek(self.read_pos)
                    data = self.spool.read(min(self.write_pos - self.read_pos, 64 * 1024))
                    self.read_pos += len(data)
                    if self.read_pos == self.write_pos:
                        self.spool.seek(0)
                        self.spool.truncate()
                        self.read_pos = self.write_pos = 0
                self.stream.write(data)
                self.stream.flush()
        except OSError as error:
            self.failure = error

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify()
        self.thread.join()
        self.spool.close()
        if self.failure:
            raise self.failure


class RpcFailure(RuntimeError):
    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code


def wire_message(message, call_names):
    entry = export_message(message, call_names)
    if entry["type"] == "message":
        return entry["message"]
    if message.get("_goal_message") or message.get("_todo_message") or entry.get("customType") in (
        "async-result", "irc:incoming", "launch-completion", "ttsr-injection", "thinking-loop-redirect", "compaction-auto-continue",
    ):
        return {"role": "custom", **{key: value for key, value in entry.items() if key != "type"},
                "timestamp": int(time() * 1000)}
    return {"role": "custom", "customType": entry["type"], "content": entry.get("content", message["content"]), "display": True}


@dataclass
class Work:
    command: dict
    content: object
    ack_pending: bool = False
    live_steered: bool = False

    @property
    def text(self):
        return self.command.get("message", "")

    @property
    def reports_result(self):
        return self.command["type"] in ("prompt", "abort_and_prompt", "continue")

    @property
    def goal_steer(self):
        return self.command["type"] == "goal-steer"

    def restored(self):
        item = {"text": self.text}
        if self.command.get("images"):
            item["images"] = self.command["images"]
        return item


class RpcTurn(Cancellation):
    def __init__(self, server, work, budget=None, *, live_steering=True):
        super().__init__(budget, reason="Turn aborted.")
        self.server = server
        self.works = work if isinstance(work, list) else [work]
        self.background = self.works[0].command["type"] == "background"
        self.messages = []
        self.call_names = {}
        self.draft = None
        self.message_id = None
        self.withdrawn = []
        self.claimed = []
        self.interrupt_responses = []
        self.goal_next = None
        self.compaction_next = None
        self.compaction_blocked = False
        if live_steering:
            self.live = LiveQueue(self, server.steering, lambda: server.steering_mode,
                                  lambda: server.event("queue_update", **server.queues()))

    def publish(self, message):
        wire = wire_message(message, self.call_names)
        identifier = self.server.next_message()
        self.server.event("message_start", message=wire, messageId=identifier)
        self.server.event("message_end", message=wire, messageId=identifier)
        self.messages.append(wire)
        return wire

    def request(self):
        self.check()
        self.server.event("turn_start")
        self.message_id = self.server.next_message()
        self.draft = wire_message({"role": "assistant", "content": None}, self.call_names)
        self.server.event("message_start", message=self.draft, messageId=self.message_id)

    def delta(self, kind, value):
        if not value or self.draft is None:
            return
        blocks = self.draft["content"]
        if not blocks or blocks[-1]["type"] != kind:
            blocks.append({"type": kind, kind: ""})
        blocks[-1][kind] += value
        message = self.draft
        index = len(blocks) - 1
        if self.server.message_updates == "delta":
            message = {**message, "content": [{"type": kind, kind: value}]}
            index = 0
        self.server.event("message_update", message=message, messageId=self.message_id, assistantMessageEvent={
            "type": kind + "_delta", "contentIndex": index, "delta": value, "partial": message,
        })

    def response(self, message):
        wire = wire_message(message, self.call_names)
        if self.draft is not None:
            self.server.event("message_end", message=wire, messageId=self.message_id)
            self.messages.append(wire)
            self.draft = None
        return wire

    def interrupted(self, error, aborted=False):
        if self.draft is not None:
            wire = {**self.draft, "stopReason": "aborted" if aborted else "error", "errorMessage": str(error)}
            if isinstance(error, ThinkingLoop):
                wire.update(content=[], errorId=error.error_id)
            self.server.event("message_end", message=wire, messageId=self.message_id)
            self.server.event("turn_end", message=wire, toolResults=[])
            self.messages.append(wire)
            self.draft = None

    def steering(self):
        with self.server.lock:
            self.check()
            queue = self.server.steering
            count = len(queue) if self.server.steering_mode == "all" else min(1, len(queue))
            accepted, deferred = self.live.drain()
            work = accepted + deferred if accepted else [queue.popleft() for _ in range(count)]
            for entry in accepted:
                entry.live_steered = True
            self.claimed.extend(work)
            if work:
                self.server.event("queue_update", **self.server.queues())
        return work

    @contextmanager
    def admission(self, work):
        with self.server.lock:
            self.check()
            yield
            self.claimed.remove(work)
            self.works.append(work)

    def take_uncommitted_steering(self):
        claimed, self.claimed = self.claimed, []
        return claimed + self.live.take_all()

    def has_steering(self):
        return bool(self.live.accepted or self.server.steering)

    def tool_start(self, call):
        function = call["function"]
        try:
            args = json.loads(function["arguments"])
        except ValueError:
            args = {}
        self.server.event("tool_execution_start", toolCallId=call["id"], toolName=function["name"], args=args)

    def tool_finished(self, call, result):
        wire = wire_message(result, self.call_names)
        output = {"content": wire["content"]}
        if "details" in wire:
            output["details"] = wire["details"]
        self.server.event("tool_execution_end", toolCallId=call["id"], toolName=call["function"]["name"],
                          result=output, isError=wire["isError"])

    def allow_tool(self, call):
        return True


class RpcServer:
    def __init__(self, budget=None):
        self.budget = budget
        self.output = RpcOutput(sys.stdout.buffer)
        self.lock = RLock()
        self.worker = None
        self.turn = None
        self.steering = deque()
        self.follow_up = deque()
        self.steering_mode = self.follow_up_mode = "one-at-a-time"
        self.event_filter = None
        self.message_updates = "full"
        self.message_counter = 0
        self.transition = None
        self.commands = {}
        self.ready = False
        self.mcp = None
        self.jobs = None
        self.awaiting_jobs = False
        self.idle_command = False
        self.background_suppressed = False

    def next_message(self):
        self.message_counter += 1
        return f"mini-pi-{self.message_counter}"

    def side_draft(self):
        turn = self.turn
        if not turn or not (draft := turn.draft):
            return None
        blocks = tuple(draft["content"])
        text = "".join(block.get("text", "") for block in blocks)
        thinking = "".join(block.get("thinking", "") for block in blocks)
        if turn.draft is not draft or not (text or thinking):
            return None
        result = {"role": "assistant", "content": text or None}
        if thinking:
            result["_thinking"] = thinking
        return result

    def event(self, kind, *, unfiltered=False, **fields):
        if unfiltered or self.event_filter is None or kind in self.event_filter or kind == "session_settled":
            self.output.write({"type": kind, **fields})

    def response_frame(self, command, data=None, error=None):
        frame = {"type": "response", "command": command.get("type", "parse"), "success": error is None}
        if "id" in command:
            frame["id"] = command["id"]
        if error is not None:
            frame["error"] = str(error)
            if isinstance(error, RpcFailure) and error.code:
                frame["code"] = error.code
        elif data is not None:
            frame["data"] = data
        return frame

    def response(self, command, data=None, error=None):
        self.output.write(self.response_frame(command, data, error))

    def response_limit(self):
        return MAX_LOGICAL if self.output.version == 2 else MAX_FRAME - 1

    def restored_queue(self, command):
        turn = self.turn
        claimed = [*turn.claimed, *turn.live.accepted, *turn.live.deferred, *turn.live.pending] if turn else []
        queues = {"steering": [*claimed, *self.steering],
                  "followUp": self.follow_up}
        payload = {key: [work.restored() for work in queue if not work.goal_steer] for key, queue in queues.items()}
        limit = self.response_limit()
        empty = {"steering": [], "followUp": []}

        def size(fields):
            return len(encoded(self.response_frame(command, fields)))

        entries_size = sum(len(encoded(item)) for queue in payload.values() for item in queue)
        commas = sum(max(0, len(queue) - 1) for queue in queues.values())
        if size(empty) + entries_size + commas <= limit:
            return payload
        if any("images" in item for queue in payload.values() for item in queue):
            empty["imagesDropped"] = True
            for queue in payload.values():
                for item in queue:
                    item.pop("images", None)
            entries_size = sum(len(encoded(item)) for queue in payload.values() for item in queue)
            if size(empty) + entries_size + commas <= limit:
                return {**empty, **payload}
        empty["truncated"] = True
        remaining = limit - size(empty)
        for key, queue in payload.items():
            for item in queue:
                cost = len(encoded(item)) + bool(empty[key])
                if cost > remaining:
                    return empty
                empty[key].append(item)
                remaining -= cost
        return empty

    def result(self, work, status, settled, error=None, *, invoked=True):
        if not work.reports_result:
            return
        frame = {"type": "prompt_result", "agentInvoked": invoked, "status": status, "sessionSettled": settled}
        if "id" in work.command:
            frame["id"] = work.command["id"]
        if error is not None:
            detail = {"message": str(error), "retryable": False}
            if isinstance(error, CostCapError):
                detail["code"] = error.code
            if isinstance(error, ThinkingLoop):
                detail["errorId"] = error.error_id
            if isinstance(error, APIStatusError):
                detail.update(httpStatus=error.status_code, retryable=error.status_code == 429 or error.status_code >= 500)
            frame["error"] = detail
        self.output.write(frame)

    def parse_prompt(self, command):
        if command["type"] == "continue":
            return Work(command, None)
        message = command.get("message")
        images = command.get("images", [])
        if not isinstance(message, str) or not isinstance(images, list) or (not message.strip() and not images):
            raise RpcFailure("A prompt requires a message or an image.")
        if not images:
            return Work(command, message)
        content = [{"type": "text", "text": message}] if message else []
        for image in images:
            if not isinstance(image, dict) or image.get("mimeType") not in ("image/png", "image/jpeg", "image/gif", "image/webp"):
                raise RpcFailure("Invalid prompt image.")
            data = image.get("data")
            if not isinstance(data, str):
                raise RpcFailure("Prompt images must contain base64 data.")
            try:
                base64.b64decode(data, validate=True)
            except ValueError as error:
                raise RpcFailure("Invalid prompt image data.") from error
            content.append({"type": "image_url", "image_url": {"url": f"data:{image['mimeType']};base64,{data}"}})
        return Work(command, content)

    def submit(self, command):
        work = self.parse_prompt(command)
        with self.lock:
            retry = isinstance(work.content, str) and work.content.strip() == "/retry"
            if retry:
                if self.turn:
                    raise RpcFailure("Wait for the current response to finish or abort it before retrying.", "session_busy")
                if getattr(self, "retry_ready", None) and not self.retry_ready():
                    self.response(command, {"message": "Nothing to retry."})
                    self.result(work, "completed", True, invoked=False)
                    return
            if self.turn and isinstance(work.content, str):
                words = work.content.split(maxsplit=1)
                if words and words[0] in ("/clear", "/fresh"):
                    raise RpcFailure(f"Wait for the current response to finish or abort it before running {words[0]}.", "session_busy")
            kind = command["type"]
            claimed = (len(self.turn.claimed) + len(self.turn.live.pending) + len(self.turn.live.accepted)
                       + len(self.turn.live.deferred)
                       if self.turn else 0)
            if kind == "abort_and_prompt":
                self.abort()
            elif claimed + len(self.steering) + len(self.follow_up) >= MAX_QUEUE:
                raise RpcFailure("The prompt queue is full.", "session_busy")
            if self.turn:
                if kind != "abort_and_prompt" and self.mcp and self.mcp.is_prompt_command(work.content):
                    raise RpcFailure("MCP prompt commands require a new turn.", "session_busy")
                behavior = command.get("streamingBehavior")
                if kind == "abort_and_prompt":
                    queue = self.follow_up
                    work.ack_pending = True
                elif kind == "steer" or behavior == "steer":
                    queue = self.steering
                elif kind in ("follow_up", "abort_and_prompt") or behavior == "followUp":
                    queue = self.follow_up
                else:
                    raise RpcFailure("A turn is running; use steer or followUp.", "session_busy")
                self.background_suppressed = False
                queue.append(work)
                if not work.ack_pending:
                    self.response(command)
                self.event("queue_update", **self.queues())
                return
            self.background_suppressed = False
            self.response(command)
            self.start_turn(work)

    def start_turn(self, work):
        self.turn = RpcTurn(self, work, self.budget)
        self.worker = Thread(target=self.run, args=(self.turn,), name="mini-pi-rpc-turn")
        self.worker.start()

    def start_goal(self, next_goal):
        with self.lock:
            if not self.turn and not self.transition and not (self.budget and self.budget.is_cancelled()):
                if content := next_goal():
                    self.start_turn(Work({"type": "goal-continuation"}, content))

    def next_background(self):
        if self.budget and self.budget.is_cancelled():
            return None
        if self.jobs and (delivery := self.jobs.take_background()):
            if not self.background_suppressed or delivery.get("_irc_messages"):
                return Work({"type": "background"}, delivery)
            self.record_input(delivery)
            message = wire_message(delivery, {})
            identifier = self.next_message()
            self.event("message_start", message=message, messageId=identifier)
            self.event("message_end", message=message, messageId=identifier)
        if self.mcp and not self.background_suppressed and (content := self.mcp.take_updates()):
            return Work({"type": "background"}, content)
        return None

    def wake_background(self, manager):
        with self.lock:
            if manager not in (self.mcp, self.jobs) or self.turn or self.idle_command or self.transition:
                return
            if work := self.next_background():
                self.start_turn(work)
            elif self.awaiting_jobs and not self.has_pending_jobs():
                self.awaiting_jobs = False
                self.event("session_settled")

    def has_pending_jobs(self):
        return self.jobs and not (self.budget and self.budget.is_cancelled()) and self.jobs.pending()

    def refresh_commands(self, manager):
        with self.lock:
            if self.mcp is manager:
                handler, _ = self.commands["get_available_commands"]
                self.output.write({"type": "available_commands_update", **handler({})})

    def run(self, turn):
        while turn:
            error = None
            status = "completed"
            self.event("agent_start")
            try:
                self.run_task(turn.works[0].content, turn)
                turn.check()
            except CancelledError as exc:
                status = "aborted"
                turn.interrupted(exc, aborted=True)
            except Exception as exc:
                error, status = exc, "error"
                turn.interrupted(exc)
            with self.lock:
                if status == "completed" and turn.is_cancelled():
                    status = "aborted"
                for work in turn.withdrawn:
                    try:
                        self.record_input(work.content)
                    except Exception as exc:
                        error, status = exc, "error"
                        break
                self.steering.extendleft(reversed(turn.take_uncommitted_steering()))
                queued = self.steering or self.follow_up
                background = self.next_background() if not queued else None
                self.awaiting_jobs = bool(self.has_pending_jobs())
                continuation = None
                if status == "completed" and not queued and not background and not self.awaiting_jobs:
                    next_turn = turn.compaction_next or turn.goal_next
                    if next_turn and (content := next_turn()):
                        source = "compaction-continuation" if turn.compaction_next else "goal-continuation"
                        continuation = Work({"type": source}, content)
                settled = not queued and not background and not continuation and not self.awaiting_jobs
                guard = {"automaticContinuationBlocked": True} if turn.compaction_blocked else {}
                self.event("agent_end", messages=turn.messages, yielded=True, isTerminal=settled, **guard)
                for work in turn.works:
                    self.result(work, status, settled, error)
                for command, data in turn.interrupt_responses:
                    self.response(command, data)
                next_work = [work for work in (background, continuation) if work]
                if queued:
                    limit = len(queued) if queued is self.follow_up and self.follow_up_mode == "all" else 1
                    next_work = [queued.popleft() for _ in range(limit)]
                    for work in next_work:
                        if work.ack_pending:
                            self.response(work.command)
                    self.event("queue_update", **self.queues())
                turn = self.turn = RpcTurn(self, next_work, self.budget) if next_work else None
                if settled:
                    self.event("session_settled")

    def abort(self, command=None, restore=False):
        with self.lock:
            if self.turn or self.steering or self.follow_up:
                self.background_suppressed = True
            if self.turn:
                self.turn.cancel()
            data = self.restored_queue(command) if restore else None
            claimed = self.turn.take_uncommitted_steering() if self.turn else []
            dropped = [*claimed, *self.steering, *self.follow_up]
            self.steering.clear()
            self.follow_up.clear()
            if dropped:
                self.event("queue_update", **self.queues())
            for work in dropped:
                if work.ack_pending:
                    self.response(work.command)
                self.result(work, "aborted", not self.turn, invoked=False)
            if self.turn and not restore:
                self.turn.withdrawn.extend(work for work in dropped if work.content is not None and not work.goal_steer)
            if command:
                if self.turn:
                    self.turn.interrupt_responses.append((command, data))
                else:
                    self.response(command, data)

    def queues(self):
        with self.lock:
            accepted = self.turn.live.accepted if self.turn else ()
            return {"steering": [work.text for work in (*accepted, *self.steering) if not work.goal_steer],
                    "followUp": [work.text for work in self.follow_up], "liveSteered": len(accepted)}

    def require_idle(self):
        if self.turn:
            raise RpcFailure("Session is busy.", "session_busy")

    def dispatch(self, command):
        kind = command["type"]
        if kind in ("prompt", "steer", "follow_up", "abort_and_prompt", "continue"):
            self.submit(command)
        elif kind == "negotiate_protocol":
            if command.get("protocolVersion") not in (1, 2):
                raise RpcFailure("Unsupported RPC protocol version.")
            self.response(command, {"protocolVersion": command["protocolVersion"], "maxFrameBytes": MAX_FRAME,
                                    "maxReassembledFrameBytes": MAX_LOGICAL})
            self.output.version = command["protocolVersion"]
        elif kind in ("abort", "abort_and_restore_queue"):
            self.abort(command, restore=kind == "abort_and_restore_queue")
        elif kind in ("set_steering_mode", "set_follow_up_mode"):
            mode = command.get("mode")
            if mode not in ("all", "one-at-a-time"):
                raise RpcFailure("Invalid queue mode.")
            setattr(self, kind.removeprefix("set_"), mode)
            self.response(command)
        elif kind == "set_event_filter":
            events = command.get("events")
            updates = command.get("messageUpdates", "full")
            if (events is not None and (not isinstance(events, list) or not all(isinstance(item, str) and item for item in events))) or updates not in ("full", "delta"):
                raise RpcFailure("Invalid event filter.")
            self.event_filter = set(events) if events is not None else None
            self.message_updates = updates
            self.response(command, {"events": list(dict.fromkeys(events)) if events is not None else None,
                                    "messageUpdates": updates})
        elif kind in ("remove_queued_message", "promote_queued_message"):
            if not isinstance(command.get("message"), str):
                raise RpcFailure("message must be a string.")
            with self.lock:
                if kind == "promote_queued_message":
                    queue = self.follow_up
                else:
                    queue = {"steering": self.steering, "followUp": self.follow_up}.get(command.get("queue"))
                if queue is None:
                    raise RpcFailure("Invalid queue.")
                work = next((item for item in queue if not item.goal_steer and item.text == command.get("message")), None)
                if work:
                    queue.remove(work)
                    if kind == "promote_queued_message":
                        self.steering.append(work)
                    self.event("queue_update", **self.queues())
                if kind == "promote_queued_message":
                    self.response(command, {"promoted": work is not None})
                else:
                    data = {"removed": work is not None}
                    if work and work.command.get("images"):
                        data["images"] = work.command["images"]
                        if len(encoded(self.response_frame(command, data))) > self.response_limit():
                            data = {"removed": True, "imagesDropped": True}
                    self.response(command, data)
                    if work:
                        if work.ack_pending:
                            self.response(work.command)
                        self.result(work, "aborted", not self.turn, invoked=False)
        elif kind in self.commands:
            handler, idle = self.commands[kind]
            if callable(idle):
                idle = idle(command)
            with self.lock:
                if idle:
                    self.require_idle()
                    self.idle_command = True
            try:
                result = handler(command)
                if isinstance(result, Transition):
                    with self.lock:
                        self.transition = (command, result)
                    return result.session
                if isinstance(result, AfterResponse):
                    self.response(command, result.data)
                    result.launch()
                else:
                    self.response(command, result)
            finally:
                if idle:
                    with self.lock:
                        self.idle_command = False
                        manager = self.jobs or self.mcp
                        if manager:
                            self.wake_background(manager)
        else:
            raise RpcFailure(f"Unknown command: {kind}")
        return None

    def serve(self, run_task, record_input, commands, mcp=None, jobs=None, *, retry_ready=None):
        self.run_task, self.record_input, self.commands = run_task, record_input, commands
        self.retry_ready = retry_ready
        if not self.ready:
            self.ready = True
            self.output.write({
                "type": "ready", "protocolVersion": 1, "supportedProtocolVersions": [1, 2],
                "maxFrameBytes": MAX_FRAME, "maxReassembledFrameBytes": MAX_LOGICAL,
            })
        if self.transition:
            command, result = self.transition
            self.transition = None
            self.response(command, result.data)
        self.mcp = mcp
        self.jobs, self.awaiting_jobs = jobs, False
        self.background_suppressed = False
        if jobs:
            jobs.wake = self.wake_background
        if mcp:
            mcp.activate(lambda: self.refresh_commands(mcp), lambda: self.wake_background(mcp))
        if jobs:
            jobs.notify()
        source = sys.stdin.buffer
        try:
            return self.read_commands(source)
        except KeyboardInterrupt:
            self.abort()
            raise
        finally:
            with self.lock:
                self.mcp = None
                self.jobs = None
            if self.worker:
                self.worker.join()

    def read_commands(self, source):
        while not self.output.failure:
            line = source.readline(MAX_FRAME + 1)
            if not line:
                break
            if len(line) > MAX_FRAME:
                while line and not line.endswith(b"\n"):
                    line = source.readline(MAX_FRAME + 1)
                self.response({}, error="RPC command exceeded the transport limit")
                continue
            try:
                command = json.loads(line)
                if not isinstance(command, dict) or not isinstance(command.get("type"), str) or (
                    "id" in command and not isinstance(command["id"], str)
                ):
                    raise ValueError("Invalid RPC command.")
            except (ValueError, RecursionError) as error:
                self.response({}, error=error)
                continue
            try:
                next_session = self.dispatch(command)
                if next_session:
                    return next_session
            except Exception as error:
                self.response(command, error=error)
        return None


@dataclass(frozen=True)
class Transition:
    session: str
    data: dict


@dataclass(frozen=True)
class AfterResponse:
    data: dict
    launch: object
