"""Terminal /loop limits and exit-code continuation conditions."""

import os
import re
import shlex
import signal
import subprocess
from concurrent.futures import CancelledError
from contextlib import suppress
from pathlib import Path
from tempfile import TemporaryDirectory, TemporaryFile
from threading import Lock
from time import monotonic, sleep

import psutil

from btw import ANSI
from service_runtime import stop_tree
from service_shell import shell_environment, shell_plan, windows_arguments


UNITS = {unit: seconds for units, seconds in (
    ("s sec secs second seconds", 1), ("m min mins minute minutes", 60), ("h hr hrs hour hours", 3600),
) for unit in units.split()}
USAGE = "Usage: /loop [count|duration] [--while|--until '<command>'] [prompt]."
LIMIT_REACHED = "Loop limit reached. Loop mode disabled."
TIME_REACHED = "Loop time limit reached. Loop mode disabled."
WORD = re.compile(r'''(?:'[^']*'|"(?:\\.|[^"\\])*"|\\.|[^\s'"\\]|\\$)+''', re.DOTALL)
POSIX_SHELLS = {"bash", "sh", "dash", "zsh", "ksh", "ash"}
OUTPUT_PREVIEW_BYTES = 8192


def positive(value, message):
    digits = value.lstrip("0")
    if not digits or len(digits) > 16 or (amount := int(digits)) > 2**53 - 1:
        raise ValueError(message)
    return amount


def duration(segments):
    seconds = 0
    for digits, unit in segments:
        if unit not in UNITS:
            raise ValueError("Loop duration unit must be seconds, minutes, or hours.")
        seconds += positive(digits, "Loop duration must be positive.") * UNITS[unit]
    return seconds


def shell_word(text, flag):
    if not (match := WORD.match(text)) or (match.end() < len(text) and not text[match.end()].isspace()):
        raise ValueError(f"{flag} has an unterminated quote.")
    chunks = re.findall(r'''('[^']*'|"(?:\\.|[^"\\])*"|\\.|[^'"\\]+|\\$)''', match[0], re.DOTALL)
    command = "".join(
        chunk[1:-1] if chunk.startswith("'") else
        re.sub(r'\\(["\\$`])', r'\1', chunk[1:-1]) if chunk.startswith('"') else
        chunk[1:] if chunk.startswith("\\") and len(chunk) > 1 else chunk
        for chunk in chunks
    ).strip()
    return command, text[match.end():].strip()


def parse(arguments):
    rest = arguments.strip()
    parts = rest.split(maxsplit=1)
    token = parts[0].lower() if parts else ""
    count = seconds = condition = None
    if re.match(r"[+-]?[0-9]", token):
        rest = parts[1] if len(parts) > 1 else ""
        tail = rest.split(maxsplit=1)
        if re.fullmatch(r"[0-9]+", token):
            if tail and tail[0].lower() in UNITS:
                seconds = duration([(token, tail[0].lower())])
                rest = tail[1] if len(tail) > 1 else ""
            else:
                count = positive(token, "Loop count must be a positive integer.")
        elif re.fullmatch(r"(?:[0-9]+[a-z]+)+", token):
            seconds = duration(re.findall(r"([0-9]+)([a-z]+)", token))
        else:
            raise ValueError(USAGE)
    while rest.startswith("--"):
        flag = re.match(r"--[a-z][a-z-]*(?=[\s=]|$)", rest)
        name = flag[0] if flag else rest.split(maxsplit=1)[0]
        if name not in ("--while", "--until"):
            raise ValueError(f"Unknown /loop flag {name}. {USAGE}")
        if condition:
            raise ValueError("Use only one of --while or --until.")
        value = rest[len(name):].removeprefix("=").lstrip()
        command, rest = shell_word(value, name) if value and not value.startswith("-") else ("", rest)
        if not command:
            raise ValueError(f"{name} needs a shell command. Quote it when it contains spaces.")
        condition = (name == "--until", command)
    return count, seconds, condition, rest


def preview(text):
    clean = "".join(char if char.isprintable() else " " for char in ANSI.sub("", text)).strip()
    return clean if len(clean) <= 60 else clean[:59] + "…"


def condition_stop(condition, runner, timeout, cancel):
    until, command = condition
    label = f"Loop condition `{preview(command)}`"
    try:
        with TemporaryFile() as output:
            code = runner(command, output, cancel, timeout)
            if code not in (0, 1):
                output.seek(0)
                lines = output.read(OUTPUT_PREVIEW_BYTES).decode(errors="replace").splitlines()
                first = next((line.strip() for line in lines if line.strip()), "")
                detail = f": {preview(first)}" if first else ""
                return f"{label} failed (exit {code}){detail}. Loop mode disabled."
    except subprocess.TimeoutExpired:
        return f"{label} timed out after {timeout:g}s. Loop mode disabled."
    except (OSError, ValueError) as error:
        return f"{label} could not run: {preview(str(error))}. Loop mode disabled."
    if (code == 0) == until:
        verdict = "is now satisfied" if until else "no longer holds"
        return f"{label} {verdict}. Loop mode disabled."
    return None


class ConditionShell:
    """Keep the terminal session's condition state separate from agent tools."""

    def __init__(self, workspace, configured, environment, run_once):
        self.workspace, self.configured, self.environment = Path(workspace), configured, environment
        self.run_once = run_once
        self.process = None
        self.lock = Lock()

    def __call__(self, command, output, cancel, timeout):
        cancel.check()
        argv = shell_plan(command, self.configured, self.environment)
        if Path(argv[0]).stem.lower() not in POSIX_SHELLS:
            environment = shell_environment(argv, self.environment)
            if os.name == "nt":
                argv = subprocess.list2cmdline(argv[:1]) + " " + windows_arguments(argv)
            return self.run_once(argv, self.workspace, output, cancel, timeout, env=environment, shell=False)

        deadline = monotonic() + timeout if timeout else None

        def remaining():
            cancel.check()
            if deadline is None:
                return .05
            seconds = deadline - monotonic()
            if seconds <= 0:
                raise subprocess.TimeoutExpired(command, timeout)
            return min(.05, seconds)

        while not self.lock.acquire(timeout=remaining()):
            pass
        try:
            remaining()
            return self.evaluate(argv, output, remaining)
        finally:
            self.lock.release()

    def start(self, argv):
        if self.process and self.process.poll() is None:
            return self.process
        self.close()
        group = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
                 if os.name == "nt" else {"start_new_session": True})
        self.process = subprocess.Popen(
            [*argv[:-2], "-s"], cwd=self.workspace, env=shell_environment(argv, self.environment),
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, bufsize=0, **group,
        )
        return self.process

    def evaluate(self, argv, output, remaining):
        with TemporaryDirectory(prefix="pipy-condition-", ignore_cleanup_errors=True) as directory:
            script, transcript, status = (Path(directory, name) for name in ("script", "output", "status"))
            script.write_text(argv[-1], encoding="utf-8", newline="\n")
            cwd, source, capture, result = (shlex.quote(path.as_posix()) for path in
                                            (self.workspace, script, transcript, status))
            # Separate files keep stdout, including binary output, out of the control channel.
            control = (
                f'{{ if ! command test . -ef {cwd}; then command cd -- {cwd} || exit 2; fi; '
                f'command eval "\n$(command -p cat {source})\n"; }} </dev/null >{capture} 2>&1\n'
                f'command printf "%s\\n" "$?" >{result}\n'
            )
            process = self.start(argv)
            try:
                remaining()
                try:
                    process.stdin.write(control.encode())
                except BrokenPipeError:
                    # A login script or an earlier background trap may have exited the shell.
                    pass
                while True:
                    wait = remaining()
                    try:
                        code = status.read_bytes()
                    except FileNotFoundError:
                        code = b""
                    if code.endswith(b"\n"):
                        code = int(code)
                        break
                    if (code := process.poll()) is not None:
                        self.close()
                        break
                    sleep(wait)
                if code not in (0, 1) and transcript.is_file():
                    with transcript.open("rb") as captured:
                        output.write(captured.read(OUTPUT_PREVIEW_BYTES))
                return code
            except (CancelledError, OSError, KeyboardInterrupt, ValueError, subprocess.TimeoutExpired):
                self.close()
                raise

    def close(self):
        process, self.process = self.process, None
        if process:
            try:
                if process.poll() is None:
                    # Stop shells before killing their children so an interrupted wait cannot run the next command.
                    if os.name == "nt":
                        with suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                            root = psutil.Process(process.pid)
                            root.suspend()
                            for child in root.children(recursive=True):
                                with suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                                    child.suspend()
                    else:
                        with suppress(ProcessLookupError):
                            os.killpg(process.pid, signal.SIGSTOP)
                stop_tree(process, grace=0)
            finally:
                process.stdin.close()


class Loop:
    def __init__(self, count, seconds, condition):
        self.remaining = self.initial = count
        self.deadline = monotonic() + seconds if seconds else None
        self.condition = condition
        self.prompt = None
        self.paused = False
        self.revision = 0

    def capture(self, prompt):
        self.prompt = prompt
        self.paused = prompt is None
        self.revision += 1

    def expired(self):
        return self.deadline is not None and monotonic() >= self.deadline

    def exhausted(self):
        return self.remaining == 0 or self.expired()

    def consume(self):
        if self.exhausted():
            return False
        if self.remaining is not None:
            self.remaining -= 1
        return True

    def status(self):
        state = "paused" if self.paused else "running" if self.prompt else "waiting for prompt"
        budget = ""
        if self.remaining is not None:
            budget = f" · {self.remaining}/{self.initial} repeats left"
        elif self.deadline is not None:
            budget = f" · {max(0, self.deadline - monotonic()):.0f}s left"
        if self.condition:
            until, command = self.condition
            budget += f" · {'until' if until else 'while'} `{preview(command)}` succeeds"
        return f"Loop {state}{budget}"


class LoopReset:
    """An admitted loop iteration must reopen the interactive session."""

    def __init__(self, loop):
        self.loop = loop
