"""Python transport and package lifecycle for the retained Bun cell worker."""

import json
import os
import re
import secrets
import shlex
import shutil
import socket
import subprocess
from concurrent.futures import CancelledError, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from tempfile import TemporaryFile
from threading import Event, Lock, Thread
from time import monotonic
from uuid import uuid4

from filelock import FileLock, Timeout as LockTimeout

from handoff import Cancellation
from service_runtime import stop_tree


WORKER = Path(__file__).with_name("eval_kernel") / "js"
FRAME_LIMIT = 16 * 1024 * 1024
MAGIC = re.compile(r"^%(load|pip|bun|environment)(?:[ \t]+([\s\S]*))?$")


def executable():
    if explicit := os.environ.get("PI_BUN"):
        path = Path(explicit).expanduser()
        return path.resolve() if path.is_file() else None
    binary = "bun.exe" if os.name == "nt" else "bun"
    if found := shutil.which(binary):
        return Path(found)
    default = Path(os.environ.get("BUN_INSTALL", str(Path.home() / ".bun"))) / "bin" / binary
    return default if default.is_file() else None


@dataclass(frozen=True)
class JsCell:
    code: str
    filename: str | None = None
    environment: str | None = None
    packages: tuple = ()


def wire_frames(stream):
    try:
        while line := stream.readline(FRAME_LIMIT + 1):
            if len(line) > FRAME_LIMIT:
                break
            frame = json.loads(line)
            if isinstance(frame, dict):
                yield frame
    except (OSError, ValueError, RecursionError):
        pass


def prepare(code, workspace, load):
    match = MAGIC.fullmatch(code.strip())
    if not match:
        return JsCell(code)
    command, raw = match[1], match[2] or ""
    if "\n" in raw or "\r" in raw:
        raise ValueError("Use %load, %bun and %environment as standalone cells.")
    if command == "load":
        return JsCell(*load(code, workspace))
    if command == "pip":
        raise ValueError("Use %bun add for JavaScript dependencies.")
    parser = shlex.shlex(raw, posix=True)
    parser.whitespace_split, parser.commenters, parser.escape = True, "", ""
    args = list(parser)
    if command == "environment":
        if len(args) != 1 or args[0] not in ("managed", "project"):
            raise ValueError("Usage: %environment managed|project.")
        return JsCell("", environment=args[0])
    if len(args) < 2 or args[0] != "add":
        raise ValueError("Usage: %bun add <package requirements...>.")
    requirements = tuple(dict.fromkeys(item.strip() for item in args[1:]))
    if any(not item or item.startswith("-") or "\0" in item for item in requirements):
        raise ValueError("Eval package requirements must be nonempty and cannot be installer flags.")
    return JsCell("", packages=requirements)


class JsKernel:
    def __init__(self, bun, workspace, env, options, clock, cancel, *, threaded=False):
        self.bun = bun
        self.threaded = threaded
        self.frames = Queue()
        self.eof = Event()
        self.closing = Event()
        self.runs, self.controls = {}, {}
        self.run_lock, self.send_lock, self.close_lock = Lock(), Lock(), Lock()
        self.reader = self.channel = self.input = None
        token = secrets.token_urlsafe(32)
        end = monotonic() + clock.startup_window()
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            listener.settimeout(.05)
            env = {**env, "MINI_PI_EVAL_TOKEN": token, "MINI_PI_EVAL_PORT": str(listener.getsockname()[1]),
                   "MINI_PI_EVAL_THREAD": "1" if threaded else "0"}
            self.process = subprocess.Popen(
                [str(bun), str(WORKER / "entry.ts")], cwd=workspace, env=env, stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **options,
            )
            try:
                while monotonic() < end and self.process.poll() is None:
                    cancel.check()
                    try:
                        channel, _ = listener.accept()
                    except TimeoutError:
                        continue
                    channel.settimeout(min(.1, max(.001, end - monotonic())))
                    expected = (token + "\n").encode()
                    received = bytearray()
                    try:
                        while len(received) < len(expected):
                            part = channel.recv(len(expected) - len(received))
                            if not part:
                                break
                            received.extend(part)
                    except OSError:
                        pass
                    if secrets.compare_digest(received, expected):
                        self.channel = channel
                        break
                    channel.close()
                if self.channel is None:
                    raise ValueError("Bun eval worker failed to establish its local transport.")
                self.channel.settimeout(None)
                self.channel.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.input = self.channel.makefile("rb")
                self.reader = Thread(target=self.read, daemon=True, name="mini-pi-js-frames")
                self.reader.start()
            except BaseException:
                self.close(force=True)
                raise

    def read(self):
        try:
            for frame in wire_frames(self.input):
                run_id, control_id = frame.get("runId"), frame.get("id")
                if isinstance(run_id, str) or isinstance(control_id, str):
                    with self.run_lock:
                        queue = self.runs.get(run_id) if isinstance(run_id, str) else self.controls.get(control_id)
                        if queue is not None:
                            queue.put(frame)
                else:
                    self.frames.put(frame)
        finally:
            self.eof.set()
            self.frames.put(None)
            with self.run_lock:
                for queue in self.runs.values():
                    queue.put(None)
                for queue in self.controls.values():
                    queue.put(None)

    def alive(self):
        return not self.closing.is_set() and not self.eof.is_set() and self.process.poll() is None

    def send(self, frame):
        data = json.dumps(frame, allow_nan=False).encode() + b"\n"
        with self.send_lock:
            channel = self.channel
            if channel is None:
                raise OSError("The Bun worker is closed.")
            channel.sendall(data)

    def initialize(self, snapshot, clock, cancel):
        self.send({"type": "init", "snapshot": snapshot})
        end = monotonic() + clock.startup_window()
        while monotonic() < end:
            cancel.check()
            try:
                frame = self.frames.get(timeout=.05)
            except Empty:
                continue
            if frame is None:
                break
            if frame.get("type") == "ready":
                return
            if frame.get("type") == "init-failed":
                raise ValueError(frame.get("error", {}).get("message") or "Bun eval worker initialization failed.")
        raise ValueError("Bun eval worker did not become ready.")

    def shadow_snapshot(self, snapshot, cancel):
        identifier = uuid4().hex
        frames = Queue()
        with self.run_lock:
            if not self.alive() or self.runs:
                return None
            self.controls[identifier] = frames
        try:
            cancel.check()
            self.send({"type": "shadow-snapshot", "id": identifier, "snapshot": snapshot})
            end = monotonic() + 5
            while monotonic() < end:
                cancel.check()
                try:
                    frame = frames.get(timeout=.025)
                except Empty:
                    continue
                if frame and frame.get("type") == "shadow-snapshot" and frame.get("eligible") is True:
                    return frame.get("snapshot")
                return None
            return None
        finally:
            with self.run_lock:
                self.controls.pop(identifier, None)

    def invoke_tool(self, identifier, request, cancellation, workers, invoke):
        frames, pending = Queue(), {}
        completed = False
        with self.run_lock:
            if not self.alive():
                raise ValueError("JavaScript kernel is not running.")
            self.runs[identifier] = frames

        def call(frame):
            try:
                name, args = frame.get("name"), frame.get("args")
                if not isinstance(name, str) or not isinstance(args, dict):
                    raise ValueError("Eval tool arguments must be an object.")
                cancellation.check()
                value = invoke(name, args, cancellation)
                json.dumps(value, allow_nan=False)
                return {"ok": True, "value": value}
            except Exception as error:
                return {"ok": False, "error": {"message": str(error) or type(error).__name__, "isToolError": True}}

        try:
            cancellation.check()
            self.send({"type": "tool", "runId": identifier, **request})
            while True:
                cancellation.check()
                for key, future in tuple(pending.items()):
                    if future.done():
                        self.send({"type": "tool-reply", "id": key, "reply": future.result()})
                        del pending[key]
                try:
                    frame = frames.get(timeout=.05)
                except Empty:
                    continue
                if frame is None:
                    raise ValueError("JavaScript kernel is not running.")
                if frame.get("type") == "result":
                    cancellation.check()
                    completed = True
                    return frame.get("value") if frame.get("ok") else {
                        "ok": False, "error": (frame.get("error") or {}).get("message") or "JavaScript tool failed.",
                    }
                if frame.get("type") == "tool-call" and isinstance(key := frame.get("id"), str) and key not in pending:
                    pending[key] = workers.submit(call, frame)
        finally:
            if not completed:
                cancellation.cancel()
                with suppress(OSError):
                    self.send({"type": "tool-cancel", "runId": identifier})
            with self.run_lock:
                self.runs.pop(identifier, None)
            for future in pending.values():
                future.cancel()

    def execute(self, cell, snapshot, clock, cancel, output, workers, invoke, *, shadow=None, token=None):
        identifier = uuid4().hex
        control = uuid4().hex if token else None
        pending = {}
        frames = Queue()
        request = {"type": "run", "runId": identifier, "code": cell.code, "filename": cell.filename or f"eval-{identifier}.js",
                   "snapshot": snapshot}
        if control:
            request.update(type="run-if-snapshot-matches", id=control, expectedRevision=token[0], expectedDigest=token[1])

        def call(frame):
            name, args = frame.get("name"), frame.get("args")
            if not isinstance(name, str) or not isinstance(args, dict):
                return {"ok": False, "error": {"message": "Eval tool arguments must be an object.", "isToolError": True}}
            pause = name in ("task", "wait", "__wait__", "__judge__") or name == "__judge_batch__" and args.get("op") == "drain"
            if pause:
                clock.pause()
            try:
                cancel.check()
                identity = frame.get("identity")
                claim = (shadow, identity, clock) if token and name == "read" and isinstance(identity, dict) else None
                value = invoke(name, args, cancel, claim) if claim else invoke(name, args, cancel)
                json.dumps(value, allow_nan=False)
                return {"ok": True, "value": value}
            except Exception as error:
                return {"ok": False, "error": {"message": str(error) or type(error).__name__, "isToolError": True}}
            finally:
                if pause:
                    clock.resume()

        try:
            if not self.alive():
                return {"dead": True}
            if clock.expired():
                return {"interrupted": "timeout"}
            cancel.check()
            with self.run_lock:
                if not self.alive():
                    return {"dead": True}
                self.runs[identifier] = frames
                if control:
                    self.controls[control] = frames
            self.send(request)
            while True:
                if cancel.is_cancelled() or clock.expired():
                    reason = "cancelled" if cancel.is_cancelled() else "timeout"
                    cancel.cancel()
                    cancel.settle_waits()
                    self.close(force=True)
                    return {"interrupted": reason, "dead": True}
                for key, future in tuple(pending.items()):
                    if future.done():
                        self.send({"type": "tool-reply", "id": key, "reply": future.result()})
                        del pending[key]
                try:
                    frame = frames.get(timeout=.025)
                except Empty:
                    if not self.alive():
                        return {"dead": True}
                    continue
                if frame is None:
                    return {"dead": True}
                if control and frame.get("type") == "shadow-run" and frame.get("id") == control:
                    if frame.get("eligible") is not True:
                        return {"admissionRejected": True}
                    continue
                if frame.get("runId") != identifier:
                    continue
                kind = frame.get("type")
                if kind == "result":
                    if frame.get("ok"):
                        return {"status": "ok"}
                    error = frame.get("error") or {}
                    output.event({"type": "error", "traceback": [
                        error.get("stack") or error.get("message") or "JavaScript cell failed."
                    ]})
                    return {"status": "error"}
                if kind == "text":
                    output.event({"type": "stdout", "data": frame.get("chunk")})
                elif kind == "display" and isinstance(display := frame.get("output"), dict):
                    output.event({"type": "image", "image": display} if display.get("type") == "image" else display)
                elif kind == "tool-call" and isinstance(key := frame.get("id"), str) and key not in pending:
                    pending[key] = workers.submit(call, frame)
        except OSError:
            return {"dead": True}
        except BaseException:
            cancel.cancel()
            self.close(force=True)
            raise
        finally:
            with self.run_lock:
                self.runs.pop(identifier, None)
                if control:
                    self.controls.pop(control, None)
            for future in pending.values():
                future.cancel()

    def close(self, *, force=False):
        with self.close_lock:
            if self.closing.is_set():
                return
            self.closing.set()
            self.shutdown(force)

    def shutdown(self, force):
        if self.process.poll() is None and self.channel and not force:
            with suppress(OSError):
                self.send({"type": "close"})
                end = monotonic() + .5
                while monotonic() < end:
                    try:
                        frame = self.frames.get(timeout=.05)
                    except Empty:
                        continue
                    if frame is None or frame.get("type") == "closed":
                        break
        if self.channel:
            with suppress(OSError):
                self.channel.shutdown(socket.SHUT_RDWR)
            self.channel.close()
            self.channel = None
        if self.process.poll() is None:
            stop_tree(self.process, grace=0)
        if self.reader:
            self.reader.join(timeout=.5)
        if self.input and (not self.reader or not self.reader.is_alive()):
            self.input.close()


class JsPlanner:
    """Isolate source projection from mutable globals in the retained worker."""

    def __init__(self, bun, env, options):
        self.bun = bun
        self.process = subprocess.Popen(
            [str(bun), str(WORKER / "shadow-entry.ts")], cwd=WORKER, env=env, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, **options,
        )
        self.frames = Queue()
        self.reader = Thread(target=self.read, name="mini-pi-js-shadow-frames", daemon=True)
        self.reader.start()

    def read(self):
        for frame in wire_frames(self.process.stdout):
            self.frames.put(frame)
        self.frames.put(None)

    def request(self, payload, cancel):
        identifier = uuid4().hex
        data = json.dumps({**payload, "id": identifier}, ensure_ascii=False, allow_nan=False).encode(
            "utf-8", errors="backslashreplace",
        ) + b"\n"
        if len(data) > FRAME_LIMIT:
            return None
        cancel.check()
        self.process.stdin.write(data)
        self.process.stdin.flush()
        end = monotonic() + 5
        while monotonic() < end:
            cancel.check()
            try:
                frame = self.frames.get(timeout=.025)
            except Empty:
                continue
            if frame is None:
                raise OSError("The JavaScript planner exited.")
            if frame.get("id") == identifier:
                return frame if not frame.get("failed") else None
        raise TimeoutError("The JavaScript planner timed out.")

    def close(self):
        with suppress(OSError):
            self.process.stdin.close()
        with suppress(subprocess.TimeoutExpired):
            self.process.wait(timeout=.15)
        if self.process.poll() is None:
            stop_tree(self.process, grace=0)
        self.reader.join(timeout=.5)
        if not self.reader.is_alive():
            self.process.stdout.close()


class JsBackend:
    def __init__(self, workspace, session_id, profile, auto_provision, env, options):
        self.workspace, self.session_id = workspace, session_id
        self.cache = profile / "cache" / "eval-js"
        self.auto_provision, self.env, self.options = auto_provision, env, options
        self.mode = "managed"
        self.project_key = None
        self.lock = Lock()
        self.kernel = None
        self.snapshot = None
        self.planner_lock = Lock()
        self.planner = None
        self.dead = False
        self.loaded = {}
        self.workers = ThreadPoolExecutor(max_workers=32, thread_name_prefix="mini-pi-js-tool")
        self.callback_workers = ThreadPoolExecutor(max_workers=32, thread_name_prefix="mini-pi-js-callback")

    def available(self):
        return executable() is not None

    def command(self, bun, arguments, cwd, cancel, seconds, extra_env=None):
        env = self.env(bun)
        env.update(extra_env or {})
        end = monotonic() + seconds
        with TemporaryFile() as output, subprocess.Popen(
            [str(bun), *arguments], cwd=cwd, env=env, stdin=subprocess.DEVNULL,
            stdout=output, stderr=subprocess.STDOUT, **self.options(),
        ) as process:
            try:
                while process.poll() is None:
                    cancel.check()
                    remaining = end - monotonic()
                    if remaining <= 0:
                        raise ValueError("Bun command timed out.")
                    try:
                        process.wait(timeout=min(.1, remaining))
                    except subprocess.TimeoutExpired:
                        pass
            except BaseException:
                stop_tree(process, grace=0)
                raise
            size = output.seek(0, os.SEEK_END)
            output.seek(max(0, size - 32768))
            return process.returncode, output.read().decode("utf-8", errors="replace").strip()

    def package_root(self, bun, mode, cancel, clock):
        if mode == "project":
            return self.workspace
        if self.project_key is None:
            source = 'process.stdout.write(Bun.hash(process.env.MINI_PI_EVAL_PROJECT).toString(16).padStart(16,"0"))'
            status, value = self.command(bun, ["-e", source], self.workspace, cancel, clock.startup_window(),
                                         {"MINI_PI_EVAL_PROJECT": str(self.workspace)})
            if status or not re.fullmatch("[0-9a-f]{16}", value):
                raise ValueError("Could not identify the managed Bun environment.")
            self.project_key = value
        return self.cache / "environments" / self.project_key

    def install(self, bun, cell, mode, root, cancel):
        lock_path = root.with_name(root.name + ".install") if mode == "managed" else self.cache / "project-locks" / self.project_key
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(lock_path))
        end = monotonic() + 600
        while True:
            cancel.check()
            if monotonic() >= end:
                raise ValueError("Timed out waiting for the Bun package environment.")
            try:
                lock.acquire(timeout=.1)
                break
            except LockTimeout:
                continue
        try:
            manifest = root / "package.json"
            if mode == "managed" and not manifest.exists():
                if not self.auto_provision:
                    raise ValueError("Managed Bun environment is unprovisioned. Enable eval.autoProvision or select %environment project.")
                root.mkdir(parents=True, exist_ok=True)
                with manifest.open("x", encoding="utf-8") as file:
                    json.dump({"private": True, "dependencies": {}}, file)
            cancel.check()
            status, output = self.command(bun, ["add", "--cwd", str(root), "--ignore-scripts", *cell.packages],
                                          root, cancel, max(.001, end - monotonic()))
            if status:
                raise ValueError(f"Bun could not update {root} (exit {status}). The retained worker was left unchanged.\n{output}")
            try:
                dependencies = sorted(json.loads(manifest.read_text(encoding="utf-8")).get("dependencies", {}))
                names = ", ".join(dependencies[:32]) or "none recorded"
                if len(dependencies) > 32:
                    names += f" (+{len(dependencies) - 32} more)"
            except (OSError, ValueError, TypeError, AttributeError):
                names = "unavailable"
            return (f"Bun reconciled {mode} environment at {root}. Current declared dependencies: {names}. "
                    "Lifecycle scripts were disabled. Already imported modules stay cached until reset.")
        finally:
            lock.release()

    def run(self, cell, args, clock, cancel, output, invoke, *, shadow=None):
        try:
            while not self.lock.acquire(timeout=.05):
                cancel.check()
            try:
                kernel, snapshot = self.get_kernel(cell, args, clock, cancel, output)
            finally:
                self.lock.release()
            token = shadow.execution_token(kernel, args) if shadow and not (cell.filename or cell.environment or cell.packages) else None
            result = kernel.execute(cell, snapshot, clock, cancel, output, self.workers, invoke, shadow=shadow, token=token)
            if token and result.get("admissionRejected"):
                shadow.invalidate()
                result = kernel.execute(cell, snapshot, clock, cancel, output, self.workers, invoke)
            with self.lock:
                if self.kernel is kernel:
                    if result.get("dead"):
                        self.reset()
                        self.dead = True
                    elif result.get("status") == "ok":
                        if cell.filename:
                            self.loaded[cell.filename] = None
                        if cell.environment:
                            self.mode = cell.environment
                            output.append(f"Eval environment: {self.mode}.")
            return result
        finally:
            cancel.finish()

    def get_kernel(self, cell, args, clock, cancel, output):
        cancel.check()
        bun = executable()
        if not bun:
            raise ValueError("JavaScript eval requires a working Bun executable on PATH or at PI_BUN.")
        if not (WORKER / "node_modules" / "@babel" / "parser" / "package.json").is_file():
            raise ValueError(f"Install the bundled worker dependencies in {WORKER} with bun install --frozen-lockfile --ignore-scripts.")
        mode = cell.environment or self.mode
        root = self.package_root(bun, mode, cancel, clock)
        description = f"{'OMP-managed' if mode == 'managed' else 'project'} environment at {root}"
        if cell.packages:
            clock.pause()
            try:
                if mode == "project":
                    self.package_root(bun, "managed", cancel, clock)
                installed = self.install(bun, cell, mode, root, cancel)
                output.details["notice"] = " ".join(filter(None, (output.details.get("notice"), installed)))
            finally:
                clock.resume()
        if args.get("reset"):
            self.reset()
            self.dead = False
        if self.kernel and (not self.kernel.alive() or self.kernel.bun != bun):
            self.reset()
            self.dead = True
        snapshot = {"cwd": str(self.workspace), "sessionId": "js:" + self.session_id,
                    "localRoots": {"local": str(self.workspace)}, "packageRoot": str(root), "packageEnvironment": description}
        if not self.kernel:
            self.kernel = self.start_kernel(bun, snapshot, clock, cancel)
        if self.dead:
            notice = "Kernel restarted: the previous JS worker is dead; state is unavailable."
            output.details["notice"] = " ".join(filter(None, (output.details.get("notice"), notice)))
            self.dead = False
        self.snapshot = snapshot
        return self.kernel, snapshot

    def start_kernel(self, bun, snapshot, clock, cancel):
        errors = []
        for threaded in (False, True):
            cancel.check()
            if clock.expired():
                break
            kernel = None
            ready = False
            try:
                kernel = JsKernel(bun, self.workspace, self.env(bun), self.options(), clock, cancel, threaded=threaded)
                kernel.initialize(snapshot, clock, cancel)
                ready = True
                return kernel
            except (OSError, RuntimeError, ValueError) as error:
                label = "Worker thread" if threaded else "subprocess"
                errors.append(f"{label}: {error}")
            finally:
                if kernel and not ready:
                    kernel.close(force=True)
        cancel.check()
        raise ValueError("Could not start JavaScript eval. " + ("; ".join(errors) or "Startup timed out."))

    def plan_request(self, bun, payload, cancel):
        while not self.planner_lock.acquire(timeout=.025):
            cancel.check()
        try:
            cancel.check()
            if self.planner and (self.planner.bun != bun or self.planner.process.poll() is not None):
                self.close_planner()
            if self.planner is None:
                self.planner = JsPlanner(bun, self.env(bun), self.options())
            return self.planner.request(payload, cancel)
        except (OSError, CancelledError):
            self.close_planner()
            raise
        finally:
            self.planner_lock.release()

    def shadow_plan(self, code, cancel):
        if not self.lock.acquire(blocking=False):
            return None
        try:
            kernel = self.kernel
            if not kernel or not self.snapshot or not (snapshot := kernel.shadow_snapshot(self.snapshot, cancel)):
                return None
        finally:
            self.lock.release()
        result = self.plan_request(kernel.bun, {"type": "plan", "code": code, "snapshot": snapshot}, cancel)
        return (kernel, {**result, "values": snapshot["values"]}) if result else None

    def shadow_arguments(self, kernel, values, results, calls, cancel):
        result = self.plan_request(kernel.bun, {"type": "arguments", "values": values, "results": results, "calls": calls}, cancel)
        return dict(result["args"]) if result else {}

    def reset(self):
        if self.kernel:
            self.kernel.close()
            self.kernel = None
        self.loaded.clear()
        self.snapshot = None

    def close(self):
        with self.lock:
            self.reset()
        with self.planner_lock:
            self.close_planner()
        self.workers.shutdown(wait=True, cancel_futures=True)
        self.callback_workers.shutdown(wait=True, cancel_futures=True)

    def close_planner(self):
        planner, self.planner = self.planner, None
        if planner:
            planner.close()
