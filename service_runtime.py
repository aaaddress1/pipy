"""Project-scoped services, supervised outside the agent process."""

import codecs
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from sqlite3 import Row, connect
from threading import Event, Lock, Thread

import psutil
from filelock import FileLock, Timeout

from service_shell import shell_environment, shell_plan, windows_arguments


SCHEMA = """
CREATE TABLE IF NOT EXISTS service_runs (
    id TEXT PRIMARY KEY, project TEXT NOT NULL, name TEXT NOT NULL, owner TEXT NOT NULL,
    spec TEXT NOT NULL, mode TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'starting',
    heartbeat REAL NOT NULL, pid INTEGER, ready REAL, finished REAL, exit_code INTEGER,
    error TEXT, stop INTEGER NOT NULL DEFAULT 0, consumed INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS service_project ON service_runs(project,name);
CREATE INDEX IF NOT EXISTS service_owner ON service_runs(project,owner,consumed,finished);
CREATE INDEX IF NOT EXISTS service_recovery ON service_runs(project,heartbeat) WHERE finished IS NULL;
CREATE TABLE IF NOT EXISTS service_clients (
    project TEXT NOT NULL, token TEXT NOT NULL, seen REAL NOT NULL, PRIMARY KEY(project,token)
);
CREATE TABLE IF NOT EXISTS service_input (
    id INTEGER PRIMARY KEY, run TEXT NOT NULL, body TEXT NOT NULL, result TEXT
);
CREATE TABLE IF NOT EXISTS service_processes (
    run TEXT PRIMARY KEY, supervisor_pid INTEGER NOT NULL, supervisor_birth REAL NOT NULL, process_birth REAL
);
CREATE TABLE IF NOT EXISTS service_generations (
    run TEXT PRIMARY KEY, created REAL NOT NULL, started REAL NOT NULL, restarts INTEGER NOT NULL
);
"""
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,47}\Z")
LEASE = 4
STALE = 12
TAIL_BYTES = 64 * 1024
MAX_LOG_BYTES = 25 * 1024 * 1024
TERMINAL_ESCAPE = re.compile(r"\x1b(?:\][^\x07\x1b]*(?:\x07|\x1b\\)|\[[0-?]*[ -/]*[@-~]|[@-_])")


@contextmanager
def connection(path):
    db = connect(path, timeout=5)
    db.row_factory = Row
    try:
        with db:
            yield db
    finally:
        db.close()


def log_path(path, identifier):
    return path.parent / "services" / (identifier + ".log")


def tail_bytes(path, limit):
    if not limit:
        return b""
    try:
        with path.open("rb") as output:
            output.seek(max(0, output.seek(0, os.SEEK_END) - limit))
            return output.read(limit)
    except FileNotFoundError:
        return b""


def previous_log(path):
    return path.with_suffix(".previous.log")


def tail(path):
    current = tail_bytes(path, TAIL_BYTES)
    previous = tail_bytes(previous_log(path), TAIL_BYTES - len(current))
    return (previous + current).decode("utf-8", errors="replace")


class ServiceLog:
    def __init__(self, path, pattern):
        self.path = path
        self.pattern = re.compile(pattern) if pattern else None
        self.matched = Event()
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.window = ""
        self.lock = Lock()

    def __enter__(self):
        self.output = self.path.open("ab", buffering=0)
        self.size = self.output.tell()
        return self

    def __exit__(self, *_):
        with self.lock:
            self.output.close()
            self.output = None

    def write(self, data):
        with self.lock:
            if self.output is None:
                return
            if self.size and self.size + len(data) > MAX_LOG_BYTES:
                self.rotate()
            self.output.write(data)
            self.size += len(data)
            if self.pattern and not self.matched.is_set():
                self.window = (self.window + self.decoder.decode(data))[-TAIL_BYTES:]
                if self.pattern.search(TERMINAL_ESCAPE.sub("", self.window)):
                    self.matched.set()

    def rotate(self):
        self.output.close()
        try:
            self.path.replace(previous_log(self.path))
            self.size = 0
        except PermissionError:
            # A concurrent reader can briefly prevent rename on Windows.
            pass
        self.output = self.path.open("ab", buffering=0)


def snapshot(row):
    state = "exited" if row["state"] == "stopped" else row["state"]
    return {"id": row["id"], "name": row["name"], "state": state, "pid": row["pid"] if row["finished"] is None else None,
            "ready": row["ready"] is not None, "exitCode": row["exit_code"], "mode": row["mode"]}


def status(row):
    state = "exited" if row["state"] == "stopped" else row["state"]
    text = f"{row['name']} [service] — {state} — {row['mode']}"
    if row["pid"] is not None and row["finished"] is None:
        text += f" — pid {row['pid']}"
    if row["exit_code"] is not None:
        text += f" — exit code {row['exit_code']}"
    return text + ("\n" + row["error"] if row["error"] else "")


def verified_process(pid, birth):
    if pid is None or birth is None:
        return None
    try:
        process = psutil.Process(pid)
        if process.create_time() == birth and process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
            return process
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        pass
    return None


def terminate_recovered(process, grace=.5):
    if not process or not process.is_running():
        return
    try:
        children = process.children(recursive=True)
    except psutil.NoSuchProcess:
        return
    targets = [*reversed(children), process]
    for target in targets:
        try:
            target.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(targets, timeout=grace)
    for target in alive:
        try:
            target.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(alive, timeout=.5)


def finish_run(db, identifier, state, code=None, error=None, *, clear_pid=False):
    stopped = state == "stopped"
    db.execute("""UPDATE service_runs SET state=?,exit_code=?,error=?,finished=?,pid=CASE WHEN ? THEN NULL ELSE pid END,
                  consumed=CASE WHEN stop>0 OR ? THEN 1 ELSE consumed END
                  WHERE id=? AND finished IS NULL""",
               ("exited" if stopped else state, code, error, time.time(), clear_pid, stopped, identifier))
    db.execute("UPDATE service_input SET body='',result='The service stopped.' WHERE run=? AND result IS NULL",
               (identifier,))


def stop_tree(process, grace=2):
    if process.poll() is not None:
        return
    if os.name == "nt":
        if grace > 0:
            try:
                if isinstance(process, WindowsTerminal):
                    process.send(b"\x03")
                else:
                    process.send_signal(signal.CTRL_BREAK_EVENT)
                until = time.monotonic() + grace
                while process.poll() is None and time.monotonic() < until:
                    time.sleep(.05)
                if process.poll() is not None:
                    return
            except OSError:
                pass
        taskkill = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "taskkill.exe"
        try:
            subprocess.run([str(taskkill), "/PID", str(process.pid), "/T", "/F"], check=False, timeout=3,
                           stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           creationflags=subprocess.CREATE_NO_WINDOW)
        except (OSError, subprocess.TimeoutExpired):
            pass
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM if grace else signal.SIGKILL)
            process.wait(timeout=max(.1, grace))
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
    if process.poll() is None:
        process.kill()
    process.wait()


class WindowsTerminal:
    def __init__(self, argv, cwd, env):
        from winpty import PTY, WinptyError

        self.error = WinptyError
        block = "\0".join(f"{key}={value}" for key, value in sorted(env.items(), key=lambda item: item[0].upper())) + "\0"
        self.pty = PTY(100, 30)
        if not self.pty.spawn(argv[0], cmdline=" " + windows_arguments(argv), cwd=cwd, env=block):
            raise OSError("Unable to spawn the Windows terminal.")
        self.pid = self.pty.pid

    @property
    def returncode(self):
        return self.pty.get_exitstatus()

    def poll(self):
        return None if self.pty.isalive() else self.returncode

    def wait(self):
        while self.pty.isalive():
            time.sleep(.05)
        return self.returncode

    def kill(self):
        os.kill(self.pid, signal.SIGTERM)

    def read(self, output):
        try:
            while not self.pty.iseof():
                if text := self.pty.read(blocking=True):
                    output.write(text.encode("utf-8"))
                else:
                    time.sleep(.03)
        except (self.error, OSError, EOFError):
            pass

    def send(self, data):
        try:
            self.pty.write(data.decode("utf-8"))
        except self.error as error:
            raise OSError(str(error)) from error


def pump_pty(master, output):
    try:
        while data := os.read(master, 8192):
            output.write(data)
    except OSError:
        pass


def pump_pipe(stream, output):
    try:
        with stream:
            while data := stream.read1(8192):
                output.write(data)
    except OSError:
        pass


def controlling_terminal():
    import fcntl
    import termios

    os.setsid()
    fcntl.ioctl(0, termios.TIOCSCTTY, 0)


def relay_input(path, identifier, process, master, closed):
    while not closed.is_set():
        with connection(path) as db:
            item = db.execute("SELECT id,body FROM service_input WHERE run=? AND result IS NULL ORDER BY id LIMIT 1",
                              (identifier,)).fetchone()
        if not item:
            closed.wait(.1)
            continue
        error = ""
        try:
            data = item["body"].encode("utf-8")
            terminal = master is not None or isinstance(process, WindowsTerminal)
            if not data.endswith((b"\r", b"\n")):
                data += b"\r" if terminal else b"\n"
            if isinstance(process, WindowsTerminal):
                process.send(data)
            elif master is None:
                process.stdin.write(data)
                process.stdin.flush()
            else:
                while data:
                    data = data[os.write(master, data):]
        except OSError as exc:
            error = str(exc)
        with connection(path) as db:
            db.execute("UPDATE service_input SET body='',result=? WHERE id=?", (error, item["id"]))


def probe_port(host, port):
    try:
        with socket.create_connection((host, port), timeout=.15):
            return True
    except OSError:
        return False


def supervise(path, identifier, recovered=False):
    supervisor = psutil.Process()
    with connection(path) as db:
        row = db.execute("""SELECT r.*, p.process_birth FROM service_runs r LEFT JOIN service_processes p ON p.run=r.id
                            WHERE r.id=? AND r.finished IS NULL""", (identifier,)).fetchone()
        if row:
            db.execute("""INSERT INTO service_processes(run,supervisor_pid,supervisor_birth) VALUES (?,?,?)
                          ON CONFLICT(run) DO UPDATE SET supervisor_pid=excluded.supervisor_pid,
                          supervisor_birth=excluded.supervisor_birth""",
                       (identifier, supervisor.pid, supervisor.create_time()))
            db.execute("UPDATE service_runs SET heartbeat=? WHERE id=?", (time.time(), identifier))
    if row is None:
        return
    if recovered:
        if row["mode"] != "detached":
            return
        process = verified_process(row["pid"], row["process_birth"])
        state, code = monitor(path, row, process, recovered=True) if process else ("exited", None)
        with connection(path) as db:
            finish_run(db, identifier, state, code, clear_pid=True)
        return
    spec = json.loads(row["spec"])
    process = master = slave = reader = writer = None
    closed = Event()
    state, error, code = "failed", None, None
    output_path = log_path(path, identifier)
    detached = row["mode"] == "detached"
    log_regex = (spec["ready"] or {}).get("log")
    try:
        if spec["pty"] and os.name != "nt":
            import pty
            master, slave = pty.openpty()
        with (output_path.open("ab", buffering=0) if detached else ServiceLog(output_path, log_regex)) as output:
            try:
                flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
                argv = spec.get("argv") or shell_plan(spec["command"], None, os.environ)
                env = shell_environment(argv, os.environ)
                env.update(PI_SESSION_ID=row["owner"], PI_SESSION_FILE=str(path), PI_TOOL_CALL_ID=spec.get("call", ""))
                if spec["pty"] and os.name == "nt":
                    process = WindowsTerminal(argv, spec["cwd"], env)
                    reader = Thread(target=process.read, args=(output,), daemon=True)
                    reader.start()
                else:
                    command = subprocess.list2cmdline(argv[:1]) + " " + windows_arguments(argv) if os.name == "nt" else argv
                    stdin = subprocess.DEVNULL if detached else subprocess.PIPE if slave is None else slave
                    stdout = output if detached else subprocess.PIPE if slave is None else slave
                    process = subprocess.Popen(command, cwd=spec["cwd"], env=env, stdin=stdin,
                                               stdout=stdout, stderr=subprocess.STDOUT,
                                               start_new_session=os.name != "nt" and slave is None, creationflags=flags,
                                               preexec_fn=controlling_terminal if slave is not None else None)
                if slave is not None:
                    os.close(slave)
                    slave = None
                    reader = Thread(target=pump_pty, args=(master, output), daemon=True)
                    reader.start()
                elif not detached and not reader:
                    reader = Thread(target=pump_pipe, args=(process.stdout, output), daemon=True)
                    reader.start()
                if not detached:
                    writer = Thread(target=relay_input, args=(path, identifier, process, master, closed), daemon=True)
                    writer.start()
                state = "starting" if spec["ready"] else "running"
                ready_at = None if spec["ready"] else time.time()
                with connection(path) as db:
                    try:
                        birth = psutil.Process(process.pid).create_time()
                    except psutil.NoSuchProcess:
                        birth = None
                    db.execute("UPDATE service_processes SET process_birth=? WHERE run=?", (birth, identifier))
                    db.execute("UPDATE service_runs SET pid=?,state=?,ready=?,heartbeat=? WHERE id=? AND finished IS NULL",
                               (process.pid, state, ready_at, time.time(), identifier))
                state, code = monitor(path, {**dict(row), "state": state, "ready": ready_at}, process,
                                      log_match=None if detached else output.matched)
            finally:
                closed.set()
                if process:
                    stop_tree(process)
                if writer:
                    writer.join(timeout=1)
                if reader:
                    reader.join(timeout=1)
    except Exception as exc:
        error, state = str(exc), "failed"
    finally:
        for fd in (master, slave):
            if fd is not None:
                os.close(fd)
        with connection(path) as db:
            finish_run(db, identifier, state, code, error)


def monitor(path, row, process, *, recovered=False, log_match=None):
    identifier = row["id"]
    readiness = json.loads(row["spec"])["ready"]
    pattern = re.compile(readiness["log"]) if readiness and readiness.get("log") else None
    port = readiness.get("port") if readiness else None
    host = readiness.get("host", "127.0.0.1") if readiness else None
    log_ready, port_ready = pattern is None, port is None
    state, ready_at = row["state"], row["ready"]
    absent, heartbeat = None, time.time()
    while True:
        if recovered:
            if not verified_process(process.pid, row["process_birth"]):
                return "exited", None
        elif (code := process.poll()) is not None:
            return ("exited" if code == 0 else "failed"), code
        changed = False
        if readiness and ready_at is None:
            if not log_ready:
                log_ready = (log_match.is_set() if log_match is not None
                             else bool(pattern.search(TERMINAL_ESCAPE.sub("", tail(log_path(path, identifier))))))
            port_ready = port_ready or probe_port(host, port)
            if log_ready and port_ready:
                state, ready_at = "ready", time.time()
                changed = True
        now = time.time()
        with connection(path) as db:
            current = db.execute("SELECT mode,stop,finished FROM service_runs WHERE id=?", (identifier,)).fetchone()
            seen = db.execute("SELECT 1 FROM service_clients WHERE project=? AND seen>? LIMIT 1",
                              (row["project"], now - LEASE)).fetchone()
            if changed or now - heartbeat >= .7:
                db.execute("UPDATE service_runs SET state=?,ready=?,heartbeat=? WHERE id=? AND finished IS NULL",
                           (state, ready_at, now, identifier))
                heartbeat = now
        absent = (absent or now) if not seen else None
        if not current or current["finished"] or current["stop"] or (
            current["mode"] == "session" and absent is not None and now - absent >= 3
        ):
            stop = current["stop"] if current else 0
            grace = (stop - 1) / 1000 if stop else 2
            if recovered:
                terminate_recovered(process, grace)
                return "stopped", None
            stop_tree(process, grace)
            return "stopped", process.returncode
        time.sleep(.15)


if __name__ == "__main__":
    path, identifier = Path(sys.argv[1]), sys.argv[2]
    try:
        with FileLock(path.parent / "services" / (identifier + ".worker.lock"), timeout=0):
            supervise(path, identifier, recovered=len(sys.argv) > 3)
    except Timeout:
        pass
