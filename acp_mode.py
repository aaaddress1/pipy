"""Agent Client Protocol over stdio, backed by the same persistent agent loop."""

import base64
import json
import os
import re
import sys
from collections import deque
from concurrent.futures import CancelledError, Future, TimeoutError
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import partial
from itertools import count
from pathlib import Path
from queue import Queue
from threading import Event, Lock, RLock, Thread
from time import monotonic
from uuid import uuid4

from filelock import Timeout
from openai import APIStatusError

from broker_usage import UsagePreflightBlocked
from cancellable_http import turn_scope
from cost_gate import CostCapError
from mcp_oauth import OAuthVault
from mcp_tools import McpSession, parse_servers
from recovery import IncompleteTurn, StepLimitReached, StreamInterrupted
from rpc_mode import MAX_LOGICAL, RpcOutput, RpcTurn, Work
from runtime import database
from session_graph import SPECULATIVE_TASK_TREE


PAGE_SIZE = 50
DEFAULT_MODE = {"id": "default", "name": "Default", "description": "Standard ACP headless mode"}
COMMANDS = [
    {"name": "clear", "description": "Clear conversation context in place"},
    {"name": "fresh", "description": "Reset provider stream state"},
    {"name": "compact", "description": "Compact session context", "input": {"hint": "Optional summary focus"}},
    {"name": "handoff", "description": "Generate a handoff and retain recent context", "input": {"hint": "Optional handoff focus"}},
    {"name": "continue", "description": "Resume an unfinished turn"},
    {"name": "retry", "description": "Retry the last failed agent turn"},
    {"name": "mcp", "description": "Manage the session's MCP servers",
     "input": {"hint": "list | resources | prompts | notifications | reconnect | reauth | unauth"}},
]
PERMISSION_OPTIONS = [
    {"optionId": key, "kind": key, "name": name}
    for key, name in (("allow_once", "Allow once"), ("allow_always", "Always allow"),
                      ("reject_once", "Reject"), ("reject_always", "Always reject"))
]
DEFERRED = object()
EOF = object()


class AcpError(RuntimeError):
    def __init__(self, message, code=-32602):
        super().__init__(message)
        self.code = code


def text_block(text):
    return {"type": "text", "text": text}


def prompt_content(blocks):
    if not isinstance(blocks, list):
        raise AcpError("prompt must be an array of content blocks.")
    texts, images = [], []

    def image(data, mime):
        if not isinstance(mime, str) or not re.fullmatch(r"image/[a-zA-Z0-9.+-]+", mime) or not isinstance(data, str):
            raise AcpError("Invalid image block.")
        try:
            if not base64.b64decode(data, validate=True):
                raise ValueError
        except ValueError as error:
            raise AcpError("Invalid image data.") from error
        images.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}})

    for block in blocks:
        if not isinstance(block, dict):
            raise AcpError("Invalid content block.")
        kind = block.get("type")
        if kind == "text" and isinstance(block.get("text"), str):
            texts.append(block["text"])
        elif kind == "image":
            image(block.get("data"), block.get("mimeType"))
        elif kind == "resource" and isinstance(block.get("resource"), dict):
            resource = block["resource"]
            if not isinstance(resource.get("uri"), str):
                raise AcpError("Embedded resources require a URI.")
            mime = resource.get("mimeType")
            if isinstance(resource.get("text"), str):
                texts.append(resource["text"])
            elif isinstance(mime, str) and mime.startswith("image/") and "blob" in resource:
                image(resource["blob"], mime)
            else:
                texts.append(f"[embedded resource: {resource['uri']}]")
        elif kind == "resource_link" and isinstance(block.get("uri"), str):
            texts.append(next((block[key] for key in ("title", "name") if isinstance(block.get(key), str)), block["uri"]))
        elif kind == "audio" and isinstance(block.get("data"), str):
            texts.append("[audio omitted]")
        else:
            raise AcpError(f"Invalid ACP content type: {kind}.")
    text = "\n\n".join(texts).strip()
    if not (text or images):
        raise AcpError("Prompt is empty.")
    return ([text_block(text)] if text else []) + images if images else text


def tool_update(cwd, identifier, name, args, status="pending"):
    args = args if isinstance(args, dict) else {}
    path = args.get("path")
    command = args.get("command")
    title = command if isinstance(command, str) else f"{name.capitalize()} {path}" if isinstance(path, str) else name
    proc_write = name == "write" and isinstance(path, str) and path.startswith("proc://")
    update = {
        "sessionUpdate": "tool_call", "toolCallId": identifier, "name": name,
        "title": title[:160], "status": status, "rawInput": args,
        "kind": "execute" if proc_write or name in ("shell", "eval") else {"read": "read", "write": "edit", "edit": "edit"}.get(name, "other"),
    }
    if isinstance(path, str) and not path.startswith(("artifact://", "proc://")):
        try:
            location = (cwd / path).resolve()
        except (OSError, ValueError, RuntimeError):
            location = None
        if location and location.is_relative_to(cwd):
            update["locations"] = [{"path": str(location)}]
    return update


def display_blocks(content):
    if isinstance(content, str):
        return [text_block(content)] if content else []
    return [block for block in content or () if block.get("type") in ("text", "image")]


class AcpBridge:
    def __init__(self, turn, capabilities):
        self.turn = turn
        fs = capabilities.get("fs") or {}
        self.read_enabled = fs.get("readTextFile") is True
        self.write_enabled = fs.get("writeTextFile") is True
        self.terminal_enabled = capabilities.get("terminal") is True

    def request(self, method, *, cancellable=True, timeout=60, **params):
        record = self.turn.record
        return record.server.client_request(
            method, {"sessionId": record.id, **params}, self.turn if cancellable else None, timeout,
        )

    def read(self, path):
        result = self.request("fs/read_text_file", path=str(path))
        content = result.get("content")
        if not isinstance(content, str):
            raise OSError("The ACP client returned invalid file content.")
        return content

    def write(self, path, content):
        self.request("fs/write_text_file", path=str(path), content=content)

    def shell(self, command, byte_limit, cwd, timeout):
        if os.name == "nt":
            executable, args = os.environ.get("ComSpec", "cmd.exe"), ["/d", "/s", "/c", command]
        else:
            executable, args = "/bin/sh", ["-c", command]
        self.turn.check()
        result = self.request("terminal/create", cancellable=False, timeout=15, command=executable,
                              args=args, cwd=str(cwd), outputByteLimit=byte_limit)
        terminal = result.get("terminalId")
        if not isinstance(terminal, str) or not terminal:
            raise OSError("The ACP client returned an invalid terminal ID.")
        try:
            with self.turn.record.lock:
                self.turn.check()
                self.turn.record.update("tool_call_update", toolCallId=self.turn.active_tool.get(),
                                        status="in_progress", content=[{"type": "terminal", "terminalId": terminal}])
            status = self.request("terminal/wait_for_exit", terminalId=terminal, timeout=timeout or float("inf"))
            output = self.request("terminal/output", terminalId=terminal)
            text = output.get("output")
            if not isinstance(text, str):
                raise OSError("The ACP client returned invalid terminal output.")
            exit_code = status.get("exitCode")
            return exit_code if type(exit_code) is int else 128, text, output.get("truncated") is True
        except (CancelledError, OSError):
            self.cleanup_terminal("terminal/kill", terminal)
            raise
        finally:
            self.cleanup_terminal("terminal/release", terminal)

    def cleanup_terminal(self, method, identifier):
        try:
            self.request(method, cancellable=False, timeout=2, terminalId=identifier)
        except OSError:
            pass


class AcpTurn(RpcTurn):
    def __init__(self, record, request_id, content, background=False):
        super().__init__(record, Work({"type": "background" if background else "prompt"}, content), record.server.budget,
                         live_steering=False)
        self.record, self.request_id, self.content = record, request_id, content
        self.bridge = AcpBridge(self, record.server.capabilities)
        self.active_tool = ContextVar("acp_active_tool", default=None)
        self.permission_lock = Lock()
        self.tool_updates = {}
        self.emitted = set()
        self.replied = False

    def steering(self):
        with self.record.lock:
            self.check()
            return [self.claim(self.record.steers)] if self.has_steering() else []

    def has_steering(self):
        return not self.background and bool(self.record.steers)

    def claim(self, queue):
        work = queue.popleft()
        self.claimed.append(work)
        return work

    @contextmanager
    def admission(self, work):
        with self.record.lock:
            self.check()
            yield
            self.claimed.remove(work)

    def admit(self, work):
        with self.record.lock:
            if work in self.claimed:
                self.claimed.remove(work)

    def allow_tool(self, call):
        self.check()
        function = call["function"]
        name = function["name"]
        self.active_tool.set(call["id"])
        update = self.tool_updates[call["id"]]
        if update["kind"] != "execute":
            return True
        params = {key: value for key, value in update.items() if key not in ("sessionUpdate", "name")}
        command = update["rawInput"].get("command")
        if isinstance(command, str):
            params["content"] = [{"type": "content", "content": text_block(f"$ {command}")}]
        peer = self.record.handle.peer
        with self.permission_lock:
            self.check()
            if name in self.record.permissions:
                return self.record.permissions[name]
            try:
                with peer.user_approval() if peer else nullcontext():
                    response = self.record.server.client_request(
                        "session/request_permission", {"sessionId": self.record.id, "toolCall": params, "options": PERMISSION_OPTIONS},
                        self, timeout=300, interrupt_on_steer=True,
                    )
            except (PermissionInterrupted, OSError):
                return False
            outcome = response.get("outcome")
            if not isinstance(outcome, dict) or outcome.get("outcome") != "selected":
                return False
            selected = outcome.get("optionId")
            if selected in ("allow_always", "reject_always"):
                self.record.permissions[name] = selected == "allow_always"
            return selected in ("allow_once", "allow_always")

    def usage(self):
        keys = {"input": "inputTokens", "output": "outputTokens", "cacheRead": "cachedReadTokens",
                "cacheWrite": "cachedWriteTokens", "totalTokens": "totalTokens"}
        totals = {wire: sum(message.get("usage", {}).get(key, 0) for message in self.messages)
                  for key, wire in keys.items()}
        if not totals["totalTokens"]:
            return None
        return {key: value for key, value in totals.items() if value or key in ("inputTokens", "outputTokens", "totalTokens")}


class PermissionInterrupted(Exception):
    pass


class AcpSession:
    message_updates = "delta"

    def __init__(self, server, identifier, cwd, *, keep_protocol_id=False):
        self.server, self.id, self.cwd = server, identifier, cwd
        self.lock = RLock()
        self.turn = self.worker = None
        self.prompts = deque()
        self.steers, self.follow_ups = deque(), deque()
        self.permissions = {}
        self.closing = False
        self.mcp = None
        self.mcp_notifications = server.mcp_notifications
        self.reconfiguring = False
        self.background_suppressed = False
        self.usage_paused = False
        self.context = server.open_session(identifier, cwd, self)
        self.handle = self.context.__enter__()
        if not keep_protocol_id:
            self.id = self.handle.session.id

    @property
    def ready(self):
        return hasattr(self, "handle") and not self.closing

    def next_message(self):
        return str(uuid4())

    def update(self, kind, **fields):
        self.server.notify("session/update", {"sessionId": self.id, "update": {"sessionUpdate": kind, **fields}})

    def chunk(self, kind, block, message_id):
        self.update(kind, content=block, messageId=message_id)

    def notice(self, text, *, action_required=False):
        self.chunk("agent_message_chunk", text_block(text), self.next_message())

    def configure_mcp(self, configs):
        with self.lock:
            self.require_idle()
            self.reconfiguring = True
            if self.mcp:
                self.mcp_notifications = self.mcp.notifications_enabled
        try:
            self.close_mcp()
            if configs:
                self.mcp = McpSession(
                    self.server.loop, configs, self.cwd, self.notice, notifications=self.mcp_notifications,
                    debounce_ms=self.server.mcp_debounce_ms, oauth_vault=self.server.oauth_vault,
                )
        finally:
            with self.lock:
                self.reconfiguring = False

    def close_mcp(self):
        manager, self.mcp = self.mcp, None
        if manager:
            self.server.loop.call(manager.close())

    def external_tools(self):
        return self.mcp.tools() if self.mcp else {}

    def external_instructions(self):
        return self.mcp.instructions() if self.mcp else ""

    def bootstrap(self):
        if self.handle.jobs:
            self.handle.jobs.wake = self.wake_background
        if self.mcp:
            manager = self.mcp
            manager.activate(self.update_commands, lambda: self.wake_background(manager))
        else:
            self.update_commands()
        if self.handle.jobs:
            self.handle.jobs.notify()

    def update_commands(self):
        prompts = self.mcp.prompt_commands() if self.mcp else []
        commands = [*COMMANDS, *[{key: value for key, value in command.items() if key != "source"} for command in prompts]]
        self.update("available_commands_update", availableCommands=commands)

    def next_background(self):
        if self.closing or self.reconfiguring or self.usage_paused or (
            self.server.budget and self.server.budget.is_cancelled()
        ):
            return None
        if self.handle.jobs and (delivery := self.handle.jobs.take_background()):
            if not self.background_suppressed or delivery.get("_irc_messages"):
                return AcpTurn(self, None, delivery, background=True)
            self.handle.session.append([delivery])
            self.chunk("user_message_chunk", text_block(delivery["content"]), self.next_message())
        if self.mcp and not self.background_suppressed and (content := self.mcp.take_updates()):
            return AcpTurn(self, None, content, background=True)
        return None

    def wake_background(self, manager):
        with self.lock:
            if manager in (self.mcp, self.handle.jobs) and not self.turn and (turn := self.next_background()):
                self.start_turn(turn)

    def event(self, kind, **fields):
        with self.lock:
            self.forward_event(kind, fields)

    def forward_event(self, kind, fields):
        turn = self.turn
        if kind in ("model_changed", "thinking_level_changed"):
            self.update("config_option_update", configOptions=self.config())
        if not turn:
            return
        if turn.is_cancelled() and (kind != "tool_execution_end" or fields["toolCallId"] not in turn.tool_updates):
            return
        if kind == "message_update":
            event = fields["assistantMessageEvent"]
            variant = "agent_thought_chunk" if event["type"] == "thinking_delta" else "agent_message_chunk"
            message_id = fields["messageId"]
            turn.emitted.add((message_id, variant))
            self.chunk(variant, text_block(event["delta"]), message_id)
        elif kind == "message_end" and fields["message"]["role"] == "assistant":
            message_id = fields["messageId"]
            for block in fields["message"]["content"]:
                if block["type"] == "thinking" and (message_id, "agent_thought_chunk") not in turn.emitted:
                    self.chunk("agent_thought_chunk", text_block(block["thinking"]), message_id)
                elif block["type"] in ("text", "image") and (message_id, "agent_message_chunk") not in turn.emitted:
                    self.chunk("agent_message_chunk", block, message_id)
        elif kind == "message_end" and fields["message"].get("customType") in ("async-result", "irc:incoming", "launch-completion"):
            self.chunk("user_message_chunk", text_block(fields["message"]["content"]), fields["messageId"])
        elif kind == "ttsr_triggered":
            self.notice("TTSR applied: " + ", ".join(rule["name"] for rule in fields["rules"]))
        elif kind == "notice" and fields.get("source") == "compaction":
            self.notice(fields["message"])
        elif kind == "tool_execution_start":
            identifier = fields["toolCallId"]
            update = tool_update(self.cwd, identifier, fields["toolName"], fields["args"])
            turn.tool_updates[identifier] = update
            self.server.notify("session/update", {"sessionId": self.id, "update": update})
        elif kind == "tool_execution_end":
            identifier = fields["toolCallId"]
            if identifier not in turn.tool_updates:
                self.event("tool_execution_start", toolCallId=identifier, toolName=fields["toolName"], args={})
            result = fields["result"]
            self.tool_end(identifier, result.get("content"), fields["isError"], result.get("details"))
        elif kind == "tool_execution_update":
            self.update("tool_call_update", toolCallId=fields["toolCallId"], status="in_progress",
                        rawOutput=fields["partialResult"])

    def tool_end(self, identifier, content, failed, details=None):
        content = display_blocks(content)
        output = {"content": content}
        if details is not None:
            output["details"] = details
        self.update("tool_call_update", toolCallId=identifier, status="failed" if failed else "completed",
                    rawOutput=output, content=[{"type": "content", "content": block} for block in content])

    def replay(self):
        starts = set()
        for message in self.call("get_messages")["messages"]:
            if message.get("customType") == "interrupted-thinking":
                continue
            role, message_id = message["role"], self.next_message()
            if role == "assistant":
                for block in message["content"]:
                    if block["type"] in ("text", "image"):
                        self.chunk("agent_message_chunk", block, message_id)
                    elif block["type"] == "thinking" and block["thinking"].strip():
                        self.chunk("agent_thought_chunk", text_block(block["thinking"]), message_id)
                    elif block["type"] == "toolCall":
                        identifier = block["id"]
                        starts.add(identifier)
                        update = tool_update(self.cwd, identifier, block["name"], block["arguments"])
                        self.server.notify("session/update", {"sessionId": self.id, "update": update})
            elif role == "toolResult":
                identifier = message["toolCallId"]
                if identifier not in starts:
                    update = tool_update(self.cwd, identifier, message["toolName"], {})
                    self.server.notify("session/update", {"sessionId": self.id, "update": update})
                self.tool_end(identifier, message["content"], message["isError"], message.get("details"))
            elif role in ("user", "custom", "compactionSummary"):
                for block in display_blocks(message.get("content", message.get("summary"))):
                    self.chunk("user_message_chunk", block, message_id)

    def call(self, name, **params):
        handler, _ = self.handle.commands[name]
        return handler(params)

    def config(self):
        model = self.handle.route()
        return [
            {"id": "mode", "name": "Mode", "category": "mode", "type": "select", "currentValue": "default",
             "options": [{"value": "default", "name": DEFAULT_MODE["name"], "description": DEFAULT_MODE["description"]}]},
            {"id": "model", "name": "Model", "category": "model", "type": "select",
             "currentValue": f"{model.name}/{model.model}",
             "options": [{"value": f"{route.name}/{route.model}", "name": route.name, "description": route.model}
                         for route in self.server.routes.values()]},
            {"id": "thinking", "name": "Thinking", "category": "thought_level", "type": "select",
             "currentValue": self.handle.thinking_level(),
             "options": [{"value": level, "name": level.capitalize()} for level in (
                 (self.handle.thinking_level(),) if model.reasoning_effort else (
                     "off", "auto", "minimal", "low", "medium", "high", "xhigh", "max"
                 )
             )]},
        ]

    def state(self):
        return {"configOptions": self.config(), "modes": {"availableModes": [DEFAULT_MODE], "currentModeId": "default"}}

    def set_config(self, name, value):
        with self.lock:
            self.require_idle()
            if name == "mode" and value == "default":
                self.update("current_mode_update", currentModeId="default")
                self.update("config_option_update", configOptions=self.config())
            elif name == "model":
                route = next((route for route in self.server.routes.values() if f"{route.name}/{route.model}" == value), None)
                if not route:
                    raise AcpError(f"Unknown ACP model: {value}.")
                self.call("set_model", provider=route.name, modelId=route.model)
            elif name == "thinking":
                self.call("set_thinking_level", level=value)
            else:
                raise AcpError(f"Unsupported config option: {name}={value}.")
            return {"configOptions": self.config()}

    def require_idle(self):
        if self.turn or self.closing:
            raise AcpError("The session is busy.", -32000)

    def prompt(self, identifier, content):
        with self.lock:
            if self.closing:
                raise AcpError("The session is closing.")
            if self.turn and isinstance(content, str):
                words = content.split(maxsplit=1)
                if words and words[0] in ("/clear", "/fresh", "/retry"):
                    raise AcpError(f"Wait for the current response to finish or cancel it before running {words[0]}.")
            self.background_suppressed = False
            turn = AcpTurn(self, identifier, content)
            if self.turn:
                self.cancel_turn(self.turn, user=True)
                self.prompts.append(turn)
            else:
                self.start_turn(turn)

    def start_turn(self, turn):
        if not turn.background:
            self.usage_paused = False
        self.turn = turn
        self.worker = Thread(target=self.run, args=(turn,), name=f"mini-pi-acp-{self.id}")
        self.worker.start()

    def run(self, turn):
        while turn:
            stop, error = "end_turn", None
            queued = None
            try:
                content = turn.content
                while True:
                    turn.check()
                    admit = partial(turn.admit, queued) if queued is not None else None
                    with turn_scope(turn):
                        self.execute(content, turn, on_admitted=admit)
                    with self.lock:
                        if admit:
                            admit()
                        turn.check()
                        queue = (self.steers or self.follow_ups) if not turn.background else None
                        queued = None
                        if queue:
                            queued = turn.claim(queue)
                            content = queued.content
                        elif (not self.prompts and turn.compaction_next
                              and (continuation := turn.compaction_next())):
                            content = continuation
                        else:
                            # Finalize under the same lock that admits steering.
                            turn = self.finish(turn, stop)
                            break
                continue
            except CancelledError as exc:
                stop = "cancelled"
                turn.interrupted(exc, aborted=True)
            except IncompleteTurn as exc:
                if isinstance(exc, StreamInterrupted):
                    error = exc
                else:
                    stop = "max_tokens"
            except StepLimitReached:
                stop = "max_turn_requests"
            except CostCapError:
                stop = "end_turn"  # The cost-gate notice has already been sent.
            except Exception as exc:
                if isinstance(exc, APIStatusError) and re.search(r"content[_ ]?filter|refus(al|ed)", str(exc), re.I):
                    stop = "refusal"
                else:
                    error = exc
            with self.lock:
                if turn.is_cancelled():
                    stop, error = "cancelled", None
                elif isinstance(error, UsagePreflightBlocked):
                    self.usage_paused = True
                    if turn.message_id or turn.background:
                        self.notice(str(error))
                        error = None
                    else:
                        error = AcpError(str(error), -32000)
                elif error and turn.message_id:
                    self.chunk("agent_message_chunk", text_block(str(error)), turn.message_id)
                    error = None
                turn = self.finish(turn, stop, error)

    def execute(self, content, turn, *, on_admitted=None):
        if isinstance(content, str):
            command, _, focus = content.partition(" ")
            if command in ("/compact", "/handoff"):
                self.call(command[1:], customInstructions=focus or None)
                self.chunk("agent_message_chunk", text_block("Context compacted." if command == "/compact" else
                           "Context handed off."), self.next_message())
                return
            if content == "/continue":
                content = None
        self.handle.run_task(content, turn, on_admitted=on_admitted)

    def finish(self, turn, stop, error=None):
        for work in reversed(turn.claimed):
            queue = self.follow_ups if work.command["type"] == "followUp" else self.steers
            queue.appendleft(work)
        turn.claimed.clear()
        self.reply_to_prompt(turn, stop, error)
        self.turn = self.prompts.popleft() if self.prompts else self.next_background()
        if self.turn and not self.turn.background:
            self.usage_paused = False
        return self.turn

    def reply_to_prompt(self, turn, stop, error=None):
        if turn.replied or turn.background:
            return
        turn.replied = True
        if error:
            self.server.respond(turn.request_id, error=error)
        else:
            result = {"stopReason": stop}
            if usage := turn.usage():
                result["usage"] = usage
            self.server.respond(turn.request_id, result=result)

    def cancel_turn(self, turn, *, user=False):
        if user:
            turn.user_interrupt = True
        turn.cancel()
        self.reply_to_prompt(turn, "cancelled")

    def steer(self, content, queue):
        with self.lock:
            turn = self.turn
            if not turn or turn.background or turn.is_cancelled() or self.closing:
                return {"outcome": "promptRequired", "queue": queue}
            if isinstance(content, str):
                words = content.split(maxsplit=1)
                if words and words[0] in ("/clear", "/fresh"):
                    return {"outcome": "failed", "reason": f"{words[0]} requires an idle session."}
            if self.mcp and self.mcp.is_prompt_command(content):
                return {"outcome": "failed", "reason": "MCP prompt commands require a new turn."}
            (self.follow_ups if queue == "followUp" else self.steers).append(Work({"type": queue}, content))
            return {"outcome": "injected", "queue": queue}

    def cancel(self, *, user=True):
        with self.lock:
            if self.turn or self.prompts or self.steers or self.follow_ups:
                self.background_suppressed = True
            if self.turn:
                self.cancel_turn(self.turn, user=user)
            for prompt in self.prompts:
                prompt.cancel()

    def close(self):
        with self.lock:
            self.closing = True
            self.cancel(user=False)
        if self.worker:
            self.worker.join()
        try:
            self.close_mcp()
        finally:
            self.context.__exit__(None, None, None)


class AcpServer:
    def __init__(self, state, open_session, fork_session, routes, loop, initial=None, *,
                 mcp_notifications=False, mcp_debounce_ms=500, oauth_vault=None, budget=None):
        self.path = state / "sessions.sqlite3"
        self.open_session, self.fork_session, self.routes = open_session, fork_session, routes
        self.initial = initial
        self.loop = loop
        self.budget = budget
        self.owns_oauth_vault = oauth_vault is None
        self.oauth_vault = oauth_vault or OAuthVault.configured(state / "mcp-oauth.sqlite3")
        self.mcp_notifications = mcp_notifications
        self.mcp_debounce_ms = mcp_debounce_ms
        self.output = RpcOutput(sys.stdout.buffer, chunked=False)
        self.inbox = Queue()
        self.disconnected = Event()
        self.capabilities = {}
        self.initialized = False
        self.sessions = {}
        self.client_calls = {}
        self.calls_lock = Lock()
        self.call_ids = count(1)

    def respond(self, identifier, *, result=None, error=None):
        frame = {"jsonrpc": "2.0", "id": identifier}
        if error is None:
            frame["result"] = result if result is not None else {}
        else:
            code = error.code if isinstance(error, AcpError) else -32000 if isinstance(error, Timeout) else -32603
            frame["error"] = {"code": code, "message": str(error) or type(error).__name__}
        self.output.write(frame)

    def notify(self, method, params):
        self.output.write({"jsonrpc": "2.0", "method": method, "params": params})

    def client_request(self, method, params, turn, timeout, interrupt_on_steer=False):
        if turn:
            turn.check()
        if self.disconnected.is_set():
            raise OSError("The ACP client disconnected.")
        future = Future()
        with self.calls_lock:
            identifier = f"mini-pi-client-{next(self.call_ids)}"
            self.client_calls[identifier] = future
        self.output.write({"jsonrpc": "2.0", "id": identifier, "method": method, "params": params})
        deadline = monotonic() + timeout
        try:
            while True:
                if turn:
                    turn.check()
                    if interrupt_on_steer:
                        with turn.record.lock:
                            if turn.has_steering():
                                raise PermissionInterrupted()
                if self.disconnected.is_set():
                    raise OSError("The ACP client disconnected.")
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise OSError(f"The ACP client timed out in {method}.")
                try:
                    response = future.result(timeout=min(remaining, 0.1))
                    break
                except TimeoutError:
                    pass
            if "error" in response:
                detail = response["error"]
                message = detail.get("message") if isinstance(detail, dict) else None
                raise OSError(message or f"The ACP client rejected {method}.")
            result = response.get("result")
            if result is not None and not isinstance(result, dict):
                raise OSError(f"The ACP client returned invalid data for {method}.")
            return result or {}
        finally:
            with self.calls_lock:
                self.client_calls.pop(identifier, None)
            if not future.done() and not self.disconnected.is_set():
                self.notify("$/cancel_request", {"id": identifier})

    def read(self):
        source = sys.stdin.buffer
        try:
            while line := source.readline(MAX_LOGICAL + 1):
                if len(line) > MAX_LOGICAL:
                    while line and not line.endswith(b"\n"):
                        line = source.readline(MAX_LOGICAL + 1)
                    self.respond(None, error=AcpError("ACP message is too large.", -32700))
                    continue
                try:
                    frame = json.loads(line)
                except (ValueError, RecursionError):
                    self.respond(None, error=AcpError("Parse error.", -32700))
                    continue
                if isinstance(frame, dict) and frame.get("jsonrpc") == "2.0" and "method" not in frame and (
                    "result" in frame or "error" in frame
                ):
                    identifier = frame.get("id")
                    if isinstance(identifier, str):
                        with self.calls_lock:
                            future = self.client_calls.get(identifier)
                            if future and not future.done():
                                future.set_result(frame)
                    continue
                self.inbox.put(frame)
        finally:
            self.disconnected.set()
            self.inbox.put(EOF)

    def serve(self):
        Thread(target=self.read, name="mini-pi-acp-input", daemon=True).start()
        try:
            while (frame := self.inbox.get()) is not EOF:
                self.dispatch_frame(frame)
        finally:
            self.disconnected.set()
            records = list(self.sessions.values())
            for record in records:
                record.cancel()
            try:
                for record in records:
                    record.close()
            finally:
                try:
                    if self.owns_oauth_vault:
                        self.loop.call(self.oauth_vault.close())
                finally:
                    self.output.close()

    def dispatch_frame(self, frame):
        if not isinstance(frame, dict) or frame.get("jsonrpc") != "2.0" or not isinstance(frame.get("method"), str):
            self.respond(None, error=AcpError("Invalid request.", -32600))
            return
        identifier = frame.get("id")
        if "id" in frame and identifier is not None and type(identifier) not in (int, str):
            self.respond(None, error=AcpError("Invalid request ID.", -32600))
            return
        try:
            params = frame.get("params", {})
            if not isinstance(params, dict):
                raise AcpError("params must be an object.")
            method = frame["method"]
            if method == "session/prompt" and "id" not in frame:
                return
            result, bootstrap = self.dispatch(method, params, identifier)
            if "id" in frame and result is not DEFERRED:
                self.respond(identifier, result=result)
            if bootstrap:
                bootstrap.bootstrap()
        except Exception as error:
            if "id" in frame:
                self.respond(identifier, error=error)

    def workspace(self, value):
        if not isinstance(value, str) or not value or not Path(value).is_absolute():
            raise AcpError("cwd must be an absolute path.")
        path = Path(value).resolve()
        if not path.is_dir():
            raise AcpError("cwd must be an existing directory.")
        return path

    def stored_workspace(self, identifier):
        with database(self.path) as db:
            row = db.execute("SELECT workspace FROM sessions WHERE id = ?", (identifier,)).fetchone()
        if row is None:
            raise AcpError(f"Unknown ACP session: {identifier}.")
        return Path(row[0])

    def record(self, params):
        identifier = params.get("sessionId")
        if not isinstance(identifier, str) or not identifier:
            raise AcpError("sessionId is required.")
        record = self.sessions.get(identifier)
        if not record:
            raise AcpError(f"Unknown ACP session: {identifier}.")
        return record

    def list_sessions(self, params, limit=PAGE_SIZE):
        cursor = params.get("cursor")
        if cursor is not None and (not isinstance(cursor, str) or not re.fullmatch(r"[0-9]{1,18}", cursor)):
            raise AcpError("Invalid session cursor.")
        offset = int(cursor or "0")
        cwd = params.get("cwd")
        filters, bindings = "", []
        if cwd is not None:
            if not isinstance(cwd, str) or not Path(cwd).is_absolute():
                raise AcpError("cwd must be an absolute path.")
            filters, bindings = "AND s.workspace = ?", [os.path.normcase(str(Path(cwd).resolve()))]
        with database(self.path) as db:
            rows = db.execute(SPECULATIVE_TASK_TREE + f"""
                SELECT s.id, s.workspace, s.updated,
                    COALESCE(SUM(json_array_length(b.messages)), 0),
                    COALESCE(SUM(length(CAST(b.messages AS BLOB))), 0)
                FROM sessions s LEFT JOIN batches b ON s.id = b.session
                WHERE s.id NOT IN (SELECT session FROM speculative_tasks)
                {filters}
                GROUP BY s.id ORDER BY s.updated DESC, s.id
                LIMIT ? OFFSET ?
            """, (*bindings, limit + 1, offset)).fetchall()
        result = {"sessions": [
            {"sessionId": identifier, "cwd": workspace, "updatedAt": datetime.fromtimestamp(updated, timezone.utc).isoformat(),
             "_meta": {"messageCount": messages, "size": size}}
            for identifier, workspace, updated, messages, size in rows[:limit]
        ]}
        if len(rows) > limit:
            result["nextCursor"] = str(offset + limit)
        return result

    def open_record(self, method, params):
        cwd = self.workspace(params.get("cwd"))
        try:
            configs = parse_servers(params.get("mcpServers", []))
        except ValueError as error:
            raise AcpError(str(error)) from error
        if method == "session/new":
            if self.initial and self.initial[1] == cwd:
                identifier = self.initial[0]
                self.initial = None
            else:
                identifier = str(uuid4())
            record = self.sessions.get(identifier)
        else:
            identifier = params.get("sessionId")
            if not isinstance(identifier, str) or not identifier:
                raise AcpError("sessionId is required.")
            record = self.sessions.get(identifier)
            if method == "session/fork":
                source = self.stored_workspace(identifier)
                if record:
                    with record.lock:
                        record.require_idle()
                        identifier = self.fork_session(record.handle.session, workspace=cwd)
                else:
                    with self.open_session(identifier, source, protect_moved=False) as handle:
                        identifier = self.fork_session(handle.session, workspace=cwd)
                record = None
            elif self.stored_workspace(identifier) != cwd:
                raise AcpError("The session belongs to a different workspace.")
        if not record:
            record = AcpSession(self, identifier, cwd, keep_protocol_id=method in ("session/load", "session/resume"))
            try:
                record.configure_mcp(configs)
            except Exception:
                record.close()
                raise
            self.sessions[record.id] = record
        else:
            record.configure_mcp(configs)
        if method == "session/load":
            with record.lock:
                record.require_idle()
                record.replay()
        result = record.state()
        if method in ("session/new", "session/fork"):
            result["sessionId"] = record.id
        return result, record

    def dispatch(self, method, params, identifier):
        if method == "initialize":
            if type(params.get("protocolVersion")) is not int:
                raise AcpError("protocolVersion must be an integer.")
            capabilities = params.get("clientCapabilities") or {}
            if not isinstance(capabilities, dict) or not isinstance(capabilities.get("fs", {}), dict):
                raise AcpError("Invalid clientCapabilities.")
            if self.initialized:
                raise AcpError("The ACP connection is already initialized.")
            self.capabilities = capabilities
            self.initialized = True
            return {
                "protocolVersion": 1,
                "agentInfo": {"name": "mini-pi", "title": "Mini Pi", "version": "0.1"},
                "authMethods": [{"id": "agent", "name": "Use existing local credentials",
                                 "description": "Use configured provider routes and agent credentials."}],
                "agentCapabilities": {
                    "loadSession": True, "mcpCapabilities": {"http": True, "sse": True},
                    "promptCapabilities": {"embeddedContext": True, "image": any(route.vision for route in self.routes.values())},
                    "sessionCapabilities": {"list": {}, "fork": {}, "resume": {}, "close": {}},
                },
                "_meta": {"steering": {"supported": True}},
            }, None
        if not self.initialized:
            raise AcpError("Initialize the ACP connection first.", -32000)
        if method == "authenticate":
            if params.get("methodId") != "agent":
                raise AcpError("Unknown ACP authentication method.")
            return {}, None
        if method in ("session/new", "session/load", "session/resume", "session/fork", "unstable_session/fork"):
            return self.open_record("session/fork" if method == "unstable_session/fork" else method, params)
        if method == "session/list":
            return self.list_sessions(params), None
        if method in ("_omp/sessions/listAll", "_omp/chats/byCwd"):
            limit = params.get("limit", 100 if method.endswith("byCwd") else 1000)
            if type(limit) is not int:
                raise AcpError("limit must be an integer.")
            if method.endswith("byCwd") and not params.get("cwd"):
                raise AcpError("cwd is required.")
            result = self.list_sessions(params, max(1, min(limit, 500 if method.endswith("byCwd") else 5000)))
            result.pop("nextCursor", None)
            if method.endswith("listAll"):
                result["total"] = len(result["sessions"])
            return result, None
        if method == "session/close":
            session_id = params.get("sessionId")
            if not isinstance(session_id, str):
                raise AcpError("sessionId is required.")
            record = self.sessions.get(session_id)
            if record:
                record.close()
                del self.sessions[record.id]
            return {}, None
        if method == "session/prompt":
            self.record(params).prompt(identifier, prompt_content(params.get("prompt")))
            return DEFERRED, None
        if method == "session/cancel":
            self.record(params).cancel()
            return {}, None
        if method == "_session/steering":
            record = self.record(params)
            metadata = params.get("_meta")
            steering = metadata.get("steering") if isinstance(metadata, dict) else None
            queue = "followUp" if isinstance(steering, dict) and steering.get("queue") == "followUp" else "steer"
            return record.steer(prompt_content(params.get("prompt")), queue), None
        if method == "session/set_mode":
            self.record(params).set_config("mode", params.get("modeId"))
            return {}, None
        if method == "session/set_config_option":
            config, value = params.get("configId"), params.get("value")
            if not isinstance(config, str) or not isinstance(value, str):
                raise AcpError("A configId and string value are required.")
            return self.record(params).set_config(config, value), None
        if method == "$/cancel_request":
            return {}, None
        raise AcpError(f"Method not found: {method}.", -32601)
