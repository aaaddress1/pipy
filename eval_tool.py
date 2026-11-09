"""Session-owned host for the upstream Python cell runner."""

import ctypes
import json
import math
import os
import re
import secrets
import shlex
import shutil
import signal
import socket
import subprocess
import sys
from concurrent.futures import CancelledError
from contextlib import contextmanager, suppress
from copy import deepcopy
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock, Thread
from time import monotonic
from urllib.parse import unquote, urlsplit
from uuid import uuid4

from auth_broker import dotted
from eval_completion import CellCancellation
from eval_js import JsBackend, prepare as prepare_js
from handoff import Cancellation
from mcp_tools import ExternalToolResult
from runtime import database
from service_runtime import stop_tree
from session_graph import ANCESTORS


SOURCE = Path(__file__).with_name("eval_kernel")
PRELUDE = (SOURCE / "prelude.py").read_text(encoding="utf-8")
ENV_NAMES = frozenset("""
    PATH HOME USERPROFILE HOMEDRIVE HOMEPATH USER LOGNAME USERNAME
    LANG LANGUAGE TZ TERM COLORTERM TMPDIR TMP TEMP SYSTEMROOT WINDIR COMSPEC PATHEXT
    APPDATA LOCALAPPDATA PROGRAMDATA VIRTUAL_ENV CONDA_PREFIX
    PYTHONPATH PYTHONHOME PYTHONUTF8 PYTHONNOUSERSITE PYTHONUSERBASE
""".split())
MAX_TEXT = 50 * 1024
MAX_CAPTURE = 16 * 1024 * 1024
MAX_LINE = 16 * 1024 * 1024
IMAGE_TYPE = re.compile(r"image/(?:png|jpeg|gif|webp)\Z")
LOAD = re.compile(r"^%load(?:[ \t]+([\s\S]*))?$")
WINDOWS_INTERRUPT = """
import signal as __omp_signal
def __omp_break(signum, frame):
    handler = __omp_signal.getsignal(__omp_signal.SIGINT)
    if callable(handler):
        handler(__omp_signal.SIGINT, frame)
if hasattr(__omp_signal, "SIGBREAK"):
    __omp_signal.signal(__omp_signal.SIGBREAK, __omp_break)
"""


def settings(layers):
    resolved = {}
    for layer in layers:
        for key in ("eval.py", "eval.js", "eval.autoProvision", "python.interpreter", "python.kernelMode",
                    "eval.autoBackground.enabled", "eval.autoBackground.thresholdMs"):
            value = dotted(layer, key)
            if value is not None:
                resolved[key] = value
    def enabled(language):
        override = os.environ.get("PI_" + language.upper(), "").strip().lower()
        if override in ("0", "false", "off", "no"):
            return False
        if override in ("1", "true", "on", "yes"):
            return True
        return resolved.get("eval." + language) is not False
    threshold = resolved.get("eval.autoBackground.thresholdMs", 60_000)
    if type(threshold) is int or (type(threshold) is float and math.isfinite(threshold)):
        threshold = max(0, min(10**15, math.floor(threshold)))
    else:
        threshold = 60_000
    background = threshold / 1000 if resolved.get("eval.autoBackground.enabled") is True else None
    return (enabled("py"), enabled("js"), resolved.get("python.interpreter"),
            resolved.get("python.kernelMode") == "per-call", background, resolved.get("eval.autoProvision") is not False)


def interpreters(workspace, requested):
    if isinstance(requested, str) and requested.strip():
        yield workspace / Path(requested).expanduser()
        return
    binary = Path("Scripts/python.exe" if os.name == "nt" else "bin/python")
    roots = [Path(os.environ[key]) for key in ("VIRTUAL_ENV", "CONDA_PREFIX") if os.environ.get(key)]
    roots += [workspace / ".venv", workspace / "venv", Path.home() / ".omp" / "python-env"]
    for root in roots:
        yield root / binary
    for name in ("python", "python3"):
        if path := shutil.which(name):
            yield Path(path)
    yield Path(sys.executable)


def kernel_environment(executable):
    env = {key: value for key, value in os.environ.items()
           if key.upper() in ENV_NAMES or key.upper().startswith(("LC_", "XDG_"))}
    env.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
    if executable.parent.name.lower() in ("scripts", "bin") and (executable.parent.parent / "pyvenv.cfg").is_file():
        env["PATH"] = str(executable.parent) + os.pathsep + env.get("PATH", "")
        env["VIRTUAL_ENV"] = str(executable.parent.parent)
    return env


def command_options():
    if os.name != "nt":
        return {"start_new_session": True}
    attached = ctypes.WinDLL("kernel32").GetConsoleWindow() or any(
        stream and stream.isatty() for stream in (sys.stdin, sys.stdout, sys.stderr)
    )
    return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | (0 if attached else subprocess.CREATE_NO_WINDOW)}


class CellClock:
    def __init__(self, seconds):
        self.seconds, self.end = seconds, monotonic() + seconds
        self.paused = 0
        self.lock = Lock()

    def expired(self):
        with self.lock:
            return self.seconds and not self.paused and monotonic() >= self.end

    def startup_window(self):
        with self.lock:
            return min(10, max(.001, self.end - monotonic())) if self.seconds and not self.paused else 10

    def pause(self):
        with self.lock:
            self.paused += 1

    def resume(self):
        with self.lock:
            if self.paused:
                self.paused -= 1
                if not self.paused:
                    self.end = monotonic() + self.seconds


class Kernel:
    def __init__(self, executable, workspace):
        self.executable = executable
        self.frames = Queue()
        self.eof = Event()
        self.tool_frames = {}
        self.tool_lock, self.send_lock = Lock(), Lock()
        self.tool_socket_lock = Lock()
        self.tool_listener = self.tool_socket = None
        env = kernel_environment(executable)
        if os.name == "nt":
            self.tool_token = secrets.token_urlsafe(32)
            self.tool_listener = socket.socket()
            self.tool_listener.bind(("127.0.0.1", 0))
            self.tool_listener.listen()
            self.tool_listener.settimeout(.05)
            env.update(MINI_PI_PY_TOOL_PORT=str(self.tool_listener.getsockname()[1]), MINI_PI_PY_TOOL_TOKEN=self.tool_token)
        self.options = command_options()
        try:
            self.process = subprocess.Popen(
                [str(executable), "-u", str(SOURCE / "runner.py")],
                cwd=workspace, env=env, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, **self.options,
            )
        except BaseException:
            if self.tool_listener:
                self.tool_listener.close()
            raise
        self.reader = Thread(target=self.read, name="mini-pi-eval-frames", daemon=True)
        self.reader.start()

    def read(self):
        try:
            while line := self.process.stdout.readline(MAX_LINE + 1):
                if len(line) > MAX_LINE:
                    while line and not line.endswith(b"\n"):
                        line = self.process.stdout.readline(MAX_LINE + 1)
                    continue
                with suppress(ValueError, RecursionError):
                    frame = json.loads(line)
                    if isinstance(frame, dict):
                        identifier = frame.get("id")
                        if isinstance(identifier, str) and identifier.startswith("tool-"):
                            with self.tool_lock:
                                if (queue := self.tool_frames.get(identifier)) is not None:
                                    queue.put(frame)
                        else:
                            self.frames.put(frame)
        finally:
            self.eof.set()
            self.frames.put(None)
            with self.tool_lock:
                for queue in self.tool_frames.values():
                    queue.put(None)

    def alive(self):
        return not self.eof.is_set() and self.process.poll() is None

    def send(self, request):
        data = json.dumps(request).encode("utf-8") + b"\n"
        with self.send_lock:
            self.process.stdin.write(data)
            self.process.stdin.flush()

    def send_tool(self, request, cancellation=None):
        data = json.dumps(request).encode("utf-8") + b"\n"
        if len(data) > MAX_LINE:
            raise ValueError("Python eval tool request exceeds the transport limit.")
        if os.name != "nt":
            with self.send_lock:
                self.process.stdin.write(data)
                self.process.stdin.flush()
            return
        with self.tool_socket_lock:
            deadline = monotonic() + 5
            while self.tool_socket is None and self.alive() and self.tool_listener and monotonic() < deadline:
                if cancellation:
                    cancellation.check()
                try:
                    channel, _ = self.tool_listener.accept()
                except TimeoutError:
                    continue
                expected = (self.tool_token + "\n").encode()
                received = bytearray()
                channel.settimeout(.2)
                try:
                    while len(received) < len(expected):
                        part = channel.recv(len(expected) - len(received))
                        if not part:
                            break
                        received.extend(part)
                except OSError:
                    pass
                if secrets.compare_digest(received, expected):
                    self.tool_socket = channel
                    self.tool_socket.settimeout(5)
                    self.tool_listener.close()
                    self.tool_listener = None
                else:
                    channel.close()
            if self.tool_socket is None:
                raise OSError("Python kernel did not open its tool transport.")
            self.tool_socket.sendall(data)

    def invoke_tool(self, identifier, request, env, cancellation):
        frames = Queue()
        sent = False
        with self.tool_lock:
            if not self.alive():
                raise ValueError("Python kernel is not running.")
            self.tool_frames[identifier] = frames
        try:
            cancellation.check()
            self.send_tool({"type": "tool", "id": identifier, "env": env, **request}, cancellation)
            sent = True
            reply, error = None, None
            while True:
                cancellation.check()
                try:
                    frame = frames.get(timeout=.05)
                except Empty:
                    continue
                if frame is None:
                    raise ValueError("Python kernel is not running.")
                if frame.get("type") == "reply":
                    reply = frame.get("value")
                elif frame.get("type") == "error":
                    error = frame.get("evalue") or "\n".join(frame.get("traceback") or ())
                elif frame.get("type") == "done":
                    cancellation.check()
                    return reply if frame.get("status") == "ok" else {"ok": False, "error": error or "Python tool failed."}
        except CancelledError:
            if sent:
                with suppress(OSError, ValueError):
                    self.send_tool({"type": "tool_cancel", "id": identifier})
            raise
        finally:
            with self.tool_lock:
                self.tool_frames.pop(identifier, None)

    def interrupt(self, identifier):
        if os.name == "nt":
            if self.options["creationflags"] & subprocess.CREATE_NO_WINDOW:
                self.close(force=True)
                return False
            with suppress(OSError):
                self.process.send_signal(signal.CTRL_BREAK_EVENT)
                return True
        else:
            with suppress(OSError):
                self.send({"type": "interrupt", "id": identifier})
                return True
        self.close(force=True)
        return False

    def execute(self, request, clock, cancel, consume):
        identifier = request["id"]
        cancel.check()
        if clock.expired():
            return {"interrupted": "timeout"}
        if not self.alive():
            return {"dead": True}
        self.send(request)
        reason, grace = None, None
        while True:
            if reason is None and (cancel.is_cancelled() or clock.expired()):
                reason = "cancelled" if cancel.is_cancelled() else "timeout"
                cancel.cancel()
                cancel.settle_waits()
                if not self.interrupt(identifier):
                    return {"interrupted": reason, "dead": True}
                grace = monotonic() + 5
            if grace is not None and monotonic() >= grace:
                self.close(force=True)
                return {"interrupted": reason, "dead": True}
            try:
                frame = self.frames.get(timeout=.05)
            except Empty:
                continue
            if frame is None:
                return {"dead": True, "interrupted": reason}
            if frame.get("id") != identifier:
                continue
            if frame.get("type") == "done":
                return {**frame, "interrupted": reason}
            try:
                consume(frame)
            except BaseException:
                cancel.cancel()
                self.close(force=True)
                raise

    def shadow_plan(self, code, cancel):
        identifier = uuid4().hex
        cancel.check()
        if not self.alive():
            return None
        self.send({"id": identifier, "type": "shadow_plan", "code": code})
        end = monotonic() + 5
        while monotonic() < end:
            cancel.check()
            try:
                frame = self.frames.get(timeout=.025)
            except Empty:
                continue
            if frame is None:
                return None
            if frame.get("id") == identifier and frame.get("type") == "shadow_plan":
                if frame.get("eligible") is True and type(frame.get("revision")) is int and isinstance(frame.get("digest"), str):
                    return frame
                return None
        return None

    def close(self, *, force=False):
        if self.process.poll() is None:
            if not force:
                with suppress(OSError, subprocess.TimeoutExpired):
                    self.send({"type": "exit"})
                    self.process.wait(timeout=.5)
            if self.process.poll() is None:
                stop_tree(self.process, grace=0)
        self.reader.join(timeout=.5)
        with self.tool_socket_lock:
            for channel in (self.tool_socket, self.tool_listener):
                if channel:
                    channel.close()
            self.tool_socket = self.tool_listener = None
        with self.send_lock:
            self.process.stdin.close()
        if not self.reader.is_alive():
            self.process.stdout.close()


class Bridge(ThreadingHTTPServer):
    def __init__(self, session):
        super().__init__(("127.0.0.1", 0), BridgeRequest)
        self.session = session
        self.active = {}
        self.lock = Lock()
        self.thread = Thread(target=self.serve_forever, kwargs={"poll_interval": .05}, daemon=True, name="mini-pi-eval-bridge")
        self.thread.start()

    @contextmanager
    def bind(self, identifier, invoke, cancel, clock, shadow=None):
        token = secrets.token_urlsafe(32)
        with self.lock:
            self.active[identifier] = token, invoke, cancel, clock, shadow
        try:
            yield {
                "PI_TOOL_BRIDGE_URL": f"http://127.0.0.1:{self.server_port}",
                "PI_TOOL_BRIDGE_TOKEN": token, "PI_TOOL_BRIDGE_SESSION": self.session,
            }
        finally:
            with self.lock:
                self.active.pop(identifier, None)

    def close(self):
        self.shutdown()
        self.server_close()
        self.thread.join()


class BridgeRequest(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        try:
            self.connection.settimeout(5)
            value = self.call()
            response = {"ok": True, "value": value}
        except Exception as error:
            response = {"ok": False, "error": str(error) or type(error).__name__}
        try:
            data = json.dumps(response, allow_nan=False).encode()
        except (TypeError, ValueError, RecursionError):
            data = b'{"ok":false,"error":"Tool result cannot be encoded as JSON."}'
        with suppress(OSError):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    def call(self):
        count = int(self.headers.get("Content-Length", 0))
        if self.path != "/v1/tool" or not 0 < count <= MAX_LINE:
            raise ValueError("Invalid tool bridge request.")
        request = json.loads(self.rfile.read(count))
        if not isinstance(request, dict) or request.get("session") != self.server.session:
            raise ValueError("Unknown tool bridge session.")
        identifier, name, args = (request.get(key) for key in ("run", "name", "args"))
        if not isinstance(identifier, str) or not isinstance(name, str) or not isinstance(args, dict):
            raise ValueError("Malformed tool bridge call.")
        with self.server.lock:
            active = self.server.active.get(identifier)
        if not active or not secrets.compare_digest(self.headers.get("Authorization", ""), "Bearer " + active[0]):
            raise ValueError("Tool bridge call is unauthorized or its cell ended.")
        _, invoke, cancel, clock, shadow = active
        cancel.check()
        pause = name in ("task", "wait", "__wait__", "__judge__") or name == "__judge_batch__" and args.get("op") == "drain"
        if pause:
            clock.pause()
        try:
            if shadow and name == "read" and isinstance(identity := request.get("identity"), dict):
                return invoke(name, args, cancel, (shadow, identity, clock))
            return invoke(name, args, cancel)
        finally:
            if pause:
                clock.resume()


def prepare(code, workspace):
    match = LOAD.fullmatch(code.strip())
    if not match:
        return code, None
    arguments = match[1] or ""
    if "\n" in arguments or "\r" in arguments:
        raise ValueError("Use %load as a standalone cell.")
    # Preserve literal path separators while allowing single and double quoted filenames.
    parser = shlex.shlex(arguments, posix=True)
    parser.whitespace_split, parser.commenters, parser.escape = True, "", ""
    files = list(parser)
    if len(files) != 1 or not files[0]:
        raise ValueError("Usage: %load <script path>. Quote paths containing spaces.")
    path = files[0]
    if path.startswith("file://"):
        parsed = urlsplit(path)
        if parsed.netloc not in ("", "localhost"):
            raise ValueError("Eval scripts must be local.")
        path = unquote(parsed.path)
        if os.name == "nt" and re.match(r"^/[a-zA-Z]:", path):
            path = path[1:]
    elif path.startswith("local://"):
        path = path.removeprefix("local://")
    elif "://" in path:
        raise ValueError("Eval scripts must be local.")
    filename = (workspace / Path(path).expanduser()).resolve(strict=True)
    return filename.read_text(encoding="utf-8-sig"), str(filename)


class CellOutput:
    def __init__(self, args, clock, progress):
        self.clock, self.progress = clock, progress
        self.text = bytearray()
        self.size = 0
        self.last_update = 0
        self.images = []
        self.image_size = 0
        language = "python" if args["language"] == "py" else "js"
        self.runtime = "Python kernel" if language == "python" else "JS worker"
        self.cell = {"index": 0, "code": args["code"], "language": language, "output": "", "status": "running"}
        if "title" in args:
            self.cell["title"] = args["title"]
        self.details = {"language": language, "languages": [language], "cells": [self.cell]}

    def append(self, text):
        data = text.encode("utf-8", errors="replace")
        self.size += len(data)
        self.text.extend(data[:max(0, MAX_CAPTURE - len(self.text))])

    def event(self, frame):
        kind = frame.get("type")
        if kind in ("stdout", "stderr") and isinstance(frame.get("data"), str):
            self.append(frame["data"])
        elif kind == "error":
            lines = frame.get("traceback") or [f"{frame.get('ename', 'Error')}: {frame.get('evalue', '')}"]
            self.append("\n".join(lines) + "\n")
        elif kind == "status" and isinstance(event := frame.get("event"), dict):
            op = event.get("op")
            if op == "timeout-pause":
                self.clock.pause()
            elif op == "timeout-resume":
                self.clock.resume()
            else:
                events = self.cell.setdefault("statusEvents", [])
                if event not in events:
                    events[:] = events[-127:] + [event]
        elif kind == "view" and isinstance(view := frame.get("view"), dict):
            views = self.cell.setdefault("views", [])
            if len(views) < 16 and len(json.dumps(view)) <= 32 * 1024:
                views.append({"text": frame.get("text", ""), "view": view})
        elif kind == "image" and isinstance(image := frame.get("image"), dict):
            mime, data = image.get("mimeType"), image.get("data")
            if isinstance(mime, str) and IMAGE_TYPE.fullmatch(mime) and isinstance(data, str):
                if self.image_size + len(data) <= MAX_CAPTURE:
                    self.image_size += len(data)
                    self.images.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}})
        if self.progress and monotonic() - self.last_update >= .05:
            self.last_update = monotonic()
            self.cell["output"] = bytes(self.text[-MAX_TEXT:]).decode("utf-8", errors="replace")
            self.progress(ExternalToolResult([{"type": "text", "text": self.cell["output"]}], deepcopy(self.details), False))

    def finish(self, session, result, start, timeout):
        interruption = result.get("interrupted")
        status = "error" if result.get("dead") or result.get("status") == "error" or interruption else "complete"
        if interruption == "timeout":
            note = f"Eval cell timed out after {timeout:g}s."
            note += " Kernel stopped and its state was lost." if result.get("dead") else " Kernel interrupted but remains running."
            self.append("\n" + note)
        elif interruption or result.get("cancelled"):
            self.append("\nCommand aborted.")
        elif result.get("dead"):
            self.append(f"\nThe {self.runtime} exited during this cell. Its state is unavailable.")
        elif status == "error":
            self.append("\nCommand exited with code 1.")
        raw = bytes(self.text)
        if len(raw) > MAX_TEXT or raw.count(b"\n") > 3000:
            full = raw.decode("utf-8", errors="replace")
            if self.size > len(raw):
                full += f"\n[Capture exceeded {MAX_CAPTURE} bytes.]"
            artifact = session.save_artifact(full)
            self.details["meta"] = {"truncated": True, "artifact": f"artifact://{artifact}"}
            preview = raw[:MAX_TEXT].decode("utf-8", errors="replace").splitlines(keepends=True)[:3000]
            text = "".join(preview).rstrip() + f"\n[Output truncated. Full capture: artifact://{artifact}]"
        else:
            text = raw.decode("utf-8", errors="replace").strip()
        if not text:
            text = f"(displayed {len(self.images)} image(s); no text output)" if self.images else "(no output)"
        self.cell.update(output=text, status=status, exitCode=int(status == "error"),
                         durationMs=round((monotonic() - start) * 1000))
        self.details["isError"] = status == "error"
        notice = self.details.get("notice")
        text = f"{notice}\n\n{text}" if notice else text
        return ExternalToolResult([{"type": "text", "text": text}, *self.images], self.details, status == "error")


class EvalTool:
    concurrency = "exclusive"
    PARAMS = {
        "type": "object", "properties": {
            "language": {"type": "string"},
            "code": {"type": "string"}, "title": {"type": "string"},
            "timeout": {"type": "number", "description": "Seconds, default 30. Zero disables the watchdog."},
            "reset": {"type": "boolean"},
        },
        "required": ["language", "code"], "additionalProperties": False,
    }

    def __init__(self, workspace, session, requested, per_call, max_timeout, jobs=None, background=None,
                 *, python=True, javascript_profile=None, auto_provision=True):
        self.workspace, self.session = workspace, session
        self.requested, self.per_call, self.max_timeout = requested, per_call, max_timeout
        self.jobs, self.background = jobs, background
        self.completion = None
        self.agents = None
        self.workpools = None
        self.custom_tools = None
        self.judgment = None
        self.judgment_batches = None
        self.python = python
        self.js = (JsBackend(workspace, session.id, javascript_profile, auto_provision, kernel_environment, command_options)
                   if javascript_profile else None)
        self.kernel = self.bridge = None
        self.lock = Lock()
        self.dead = False
        self.loaded = {}
        self.used = None
        self.failed_probe = None
        self.bridge_lock = Lock()

    def tool_bridge(self):
        with self.bridge_lock:
            if self.bridge is None:
                self.bridge = Bridge(self.session.id)
            return self.bridge

    def invoke_kernel_tool(self, language, identifier, request, invoke, cancellation):
        backend = self.js if language == "js" else None
        kernel = backend.kernel if backend else self.kernel if language == "python" else None
        if kernel is None or not kernel.alive():
            if request["op"] == "describe":
                return {"ok": True, "tools": [], "missing": request["names"]}
            raise ValueError(f"{'JavaScript' if language == 'js' else 'Python'} kernel is not running.")
        try:
            if backend:
                return kernel.invoke_tool(identifier, request, cancellation, backend.callback_workers, invoke)
            with self.tool_bridge().bind(identifier, invoke, cancellation, CellClock(0)) as env:
                return kernel.invoke_tool(identifier, request, env, cancellation)
        except (OSError, ValueError):
            if request["op"] == "describe" and not kernel.alive():
                return {"ok": True, "tools": [], "missing": request["names"]}
            raise

    def runtime_inputs(self):
        candidates = tuple(dict.fromkeys(interpreters(self.workspace, self.requested)))
        return tuple((candidate, candidate.is_file()) for candidate in candidates)

    def languages(self):
        languages = []
        if self.python and (self.failed_probe is None or self.runtime_inputs() != self.failed_probe):
            languages.append("py")
        if self.js and self.js.available():
            languages.append("js")
        return languages

    def available(self):
        return bool(self.languages())

    @property
    def parameters(self):
        return {**self.PARAMS, "properties": {**self.PARAMS["properties"], "language": {
            "type": "string", "enum": self.languages(),
        }}}

    @property
    def description(self):
        languages = self.languages()
        common = ("Execute one cell in a retained session runtime. Separate cells of the same language reuse variables and imports. "
                  "Supports top-level await, trailing-expression output, %load, and await tool.<name>({...}). "
                  "Use reset to recreate only this language's runtime.")
        python = " Python (py) supports %pip." if "py" in languages else ""
        javascript = " JavaScript (js) runs in Bun and supports %bun add and %environment managed|project." if "js" in languages else ""
        completion = (" completion(prompt) starts a stateless model request and returns a handle. "
                      "Use handle.wait() or wait(handles) to collect results. Options include model (smol, default, slow), "
                      "system and JSON schema.") if self.completion else ""
        judgment = (" Use await judge(state, questions, images=...) for typed bool, choice, and score judgments."
                    if self.judgment else "")
        batches = (" Use judge_batch() in Python or await judgeBatch() in JavaScript for many states. "
                   "Drain results across cells or attach by batch ID after a reset." if self.judgment_batches else "")
        agents = (" agent(prompt) starts a child agent and returns a handle. Use wait(handles) to collect its result."
                  if self.agents and self.agents.tasks and self.agents.tasks.background else "")
        custom = (" Define tools with @tool in Python or tool(fn, options) in JavaScript and pass their names to agent(tools=...)."
                  if agents and self.custom_tools and self.custom_tools.enabled() else "")
        pools = (" Use workpool() to queue items for retained workers and wait() to collect the pool's aggregate result."
                 if agents and self.workpools else "")
        return common + python + javascript + completion + judgment + batches + agents + custom + pools

    def validate(self, args):
        if not isinstance(args, dict) or not {"language", "code"} <= args.keys() <= self.PARAMS["properties"].keys():
            raise ValueError("Eval takes one cell with language and code, plus optional title, timeout and reset.")
        if not (args["language"] == "py" and self.python or args["language"] == "js" and self.js):
            raise ValueError("Requested eval backend is disabled or unavailable.")
        if not isinstance(args["code"], str) or "title" in args and not isinstance(args["title"], str):
            raise ValueError("Eval code and title must be strings.")
        if "reset" in args and type(args["reset"]) is not bool:
            raise ValueError("Eval reset must be a boolean.")
        timeout = args.get("timeout", 30)
        if type(timeout) not in (int, float) or (type(timeout) is float and not math.isfinite(timeout)):
            raise ValueError("Eval timeout must be a finite number.")
        ceiling = min(3600, self.max_timeout) if self.max_timeout else 3600
        return min(ceiling, max(1, timeout)) if timeout else 0

    def start(self, clock, cancel):
        startup = CellClock(clock.startup_window())
        candidates = self.runtime_inputs()
        for executable, exists in candidates:
            cancel.check()
            if startup.expired():
                break
            if not exists:
                continue
            kernel = None
            ready = False
            try:
                kernel = Kernel(executable, self.workspace)
                request = {"id": uuid4().hex, "code": PRELUDE + (WINDOWS_INTERRUPT if os.name == "nt" else ""),
                           "cwd": str(self.workspace), "silent": True, "storeHistory": False}
                init = kernel.execute(request, startup, CellCancellation(cancel), lambda _: None)
                ready = init.get("status") == "ok" and not init.get("interrupted")
                if ready:
                    self.failed_probe = None
                    return kernel
            except OSError:
                pass
            finally:
                if kernel and not ready:
                    kernel.close()
        cancel.check()
        if startup.expired():
            raise ValueError("Python kernel startup timed out.")
        self.failed_probe = candidates
        raise ValueError("Python kernel is unavailable. Set python.interpreter to a working Python executable.")

    def execute(self, args, invoke, turn, progress=None, *, shadow=None):
        self.used = True
        timeout = self.validate(args)
        if args["language"] == "js":
            cell = prepare_js(args["code"], self.workspace, prepare)

            def run(cancel, update):
                start = monotonic()
                clock = CellClock(timeout)
                output = CellOutput(args, clock, update)
                self.notice_timeout(args, timeout, output)
                try:
                    result = self.js.run(cell, args, clock, CellCancellation(cancel),
                                         output, invoke, shadow=shadow)
                except (OSError, ValueError) as error:
                    output.append(str(error))
                    result = {"status": "error"}
                return output.finish(self.session, result, start, timeout)
        else:
            code, filename = prepare(args["code"], self.workspace)

            def run(cancel, update):
                return self.run(args, code, filename, timeout, invoke, cancel, update, shadow)

        if self.jobs and self.background is not None:
            label = args.get("title") or ("Python eval" if args["language"] == "py" else "JavaScript eval")
            result = self.jobs.start_eval(label, timeout, self.background, turn, run, progress)
            if result is not None:
                if "async" in result.details and "cells" not in result.details:
                    result.details.update(CellOutput(args, None, None).details)
                return result
        return run(turn, progress)

    @staticmethod
    def notice_timeout(args, timeout, output):
        if "timeout" in args and timeout != args["timeout"]:
            notice = f"Eval timeout clamped from {args['timeout']}s to {timeout:g}s."
            output.details["notice"] = " ".join(filter(None, (notice, output.details.get("notice"))))

    def run(self, args, code, filename, timeout, invoke, turn, progress, shadow):
        cancel = CellCancellation(turn)
        while not self.lock.acquire(timeout=.05):
            cancel.check()
        start = monotonic()
        clock = CellClock(timeout)
        output = CellOutput(args, clock, progress)
        try:
            cancel.check()
            if args.get("reset"):
                self.reset()
                self.dead = False
            if self.kernel and not self.kernel.alive():
                self.reset()
                self.dead = True
            if not self.kernel:
                self.kernel = self.start(clock, cancel)
            notices = []
            if self.dead:
                notices.append("Kernel restarted: the previous Python kernel is dead; state is unavailable.")
                self.dead = False
            if notices:
                output.details["notice"] = " ".join(notices)
            self.notice_timeout(args, timeout, output)
            def execute_cell(token=None):
                identifier = uuid4().hex
                with self.tool_bridge().bind(identifier, invoke, cancel, clock, shadow if token else None) as env:
                    env["PI_EVAL_LOCAL_ROOTS"] = json.dumps({"local": str(self.workspace)})
                    if self.agents and self.agents.artifacts:
                        env["PI_ARTIFACTS_DIR"] = str(self.agents.artifacts)
                    request = {"id": identifier, "code": code, "cwd": str(self.workspace), "filename": filename, "env": env}
                    if token:
                        request.update(expectedShadowRevision=token[0], expectedShadowDigest=token[1])
                    return self.kernel.execute(request, clock, cancel, output.event)

            token = shadow.execution_token(self.kernel, args) if shadow and not filename and not self.per_call else None
            result = execute_cell(token)
            if token and result.get("admissionRejected") and not result.get("interrupted"):
                shadow.invalidate()
                result = execute_cell()
            if result.get("dead"):
                self.reset()
                self.dead = not self.per_call
            elif filename and result.get("status") == "ok" and not result.get("interrupted"):
                self.loaded[filename] = None
            return output.finish(self.session, result, start, timeout)
        finally:
            cancel.finish()
            if self.per_call:
                self.reset()
            self.lock.release()

    def reset(self):
        if self.kernel:
            self.kernel.close()
            self.kernel = None
        self.loaded.clear()

    def reset_context(self):
        with self.lock:
            self.reset()
            self.dead, self.used = False, None
        if self.js:
            with self.js.lock:
                self.js.reset()

    def shadow_plan(self, code, cancel, language):
        if self.background is not None:
            return None
        if language == "js":
            return self.js.shadow_plan(code, cancel) if self.js else None
        if self.per_call or not self.lock.acquire(blocking=False):
            return None
        try:
            kernel = self.kernel
            if kernel and (frame := kernel.shadow_plan(code, cancel)):
                return kernel, frame
            return None
        finally:
            self.lock.release()

    def summary(self):
        def line(label, executable, scripts):
            paths = [path[:300] for path in list(scripts)[-12:]]
            prefix = f"Retained {label} is live ({escape(str(executable))})."
            return prefix + (" Loaded scripts: " + escape(json.dumps(paths, ensure_ascii=False)) if paths else "")

        active = []
        if (kernel := self.kernel) and kernel.alive() and not self.per_call:
            active.append(line("Python eval kernel", kernel.executable, self.loaded))
        if self.js and (kernel := self.js.kernel) and kernel.alive():
            active.append(line("JavaScript eval worker", kernel.bun, self.js.loaded))
        return "\n".join(active) or None

    def context(self):
        summary = self.summary()
        if summary is None:
            if self.used is None:
                with database(self.session.path) as db:
                    self.used = bool(db.execute(ANCESTORS + """
                        SELECT 1 FROM ancestry a JOIN session_nodes n ON n.id=a.id
                        JOIN batches b ON b.seq=n.batch, json_each(b.messages) m, json_each(m.value, '$.tool_calls') c
                        WHERE json_extract(c.value, '$.function.name') = 'eval' LIMIT 1
                    """, (self.session.leaf, self.session.id, self.session.id)).fetchone())
            if not self.used:
                return None
            summary = "No retained eval kernel is live. Re-run the setup needed by later cells."
        return f"<system-reminder>\n{summary}\nEval variable values cannot be inferred from this snapshot.\n</system-reminder>"

    def close(self):
        if self.custom_tools:
            self.custom_tools.close()
        with self.lock:
            self.reset()
            with self.bridge_lock:
                if self.bridge:
                    self.bridge.close()
                    self.bridge = None
        if self.js:
            self.js.close()
