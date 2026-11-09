"""Session client for shared project service supervisors."""

import json
import math
import os
import re
import subprocess
import sys
import time
from hashlib import sha256
from pathlib import Path
from sqlite3 import Error as SQLiteError
from threading import Event, Lock, Thread
from uuid import uuid4

import psutil
from filelock import FileLock, Timeout

from async_jobs import HARNESS_TAG, tool_result
from service_runtime import (LEASE, NAME, STALE, TERMINAL_ESCAPE, connection, finish_run, log_path, snapshot,
                             status, tail, terminate_recovered, verified_process)
from service_shell import shell_plan


def normalize_readiness(ready):
    if ready is None:
        return None
    if not isinstance(ready, dict) or ready.keys() - {"log", "port", "host", "timeout"}:
        raise ValueError("ready accepts log, port, host and timeout.")
    clean = {}
    for key in ("log", "host"):
        value = ready.get(key)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"ready.{key} must be a string.")
        if value and value.strip():
            clean[key] = value.strip()
    for key in ("port", "timeout"):
        value = ready.get(key)
        if value is not None and (type(value) not in (int, float) or type(value) is float and not math.isfinite(value)):
            raise ValueError(f"ready.{key} must be a finite number.")
        if value is not None and (key != "port" or value != 0):
            clean[key] = value
    return clean or None


class Services:
    def __init__(self, path, session_id, workspace, wake, env=None, shell=None, *, scope=None):
        self.path, self.owner = path, session_id
        self.project = scope if scope is not None else os.path.normcase(str(workspace.resolve()))
        self.wake, self.env = wake, env
        self.shell = shell
        self.token, self.closed = uuid4().hex, Event()
        self.lock, self.worker = Lock(), None
        self.owned = False

    def reap(self):
        cutoff = time.time() - 2
        with connection(self.path) as db:
            candidates = db.execute("SELECT id FROM service_runs WHERE project=? AND finished IS NULL AND heartbeat<?",
                                    (self.project, cutoff)).fetchall()
        for candidate in candidates:
            identifier = candidate["id"]
            lock = self.path.parent / "services" / (identifier + ".recovery.lock")
            try:
                with FileLock(lock, timeout=0):
                    self.recover(identifier, cutoff)
            except (Timeout, psutil.AccessDenied):
                continue

    def recover(self, identifier, cutoff):
        with connection(self.path) as db:
            row = db.execute("""SELECT r.*,p.supervisor_pid,p.supervisor_birth,p.process_birth FROM service_runs r
                                LEFT JOIN service_processes p ON p.run=r.id
                                WHERE r.id=? AND r.finished IS NULL AND r.heartbeat<?""", (identifier, cutoff)).fetchone()
        if not row or verified_process(row["supervisor_pid"], row["supervisor_birth"]):
            return
        if row["supervisor_pid"] is None and time.time() - row["heartbeat"] < STALE:
            return
        service = verified_process(row["pid"], row["process_birth"])
        if service and row["mode"] == "detached":
            with connection(self.path) as db:
                db.execute("UPDATE service_runs SET heartbeat=? WHERE id=? AND finished IS NULL", (time.time(), identifier))
            self.launch_worker(identifier, recovered=True)
            return
        terminate_recovered(service)
        with connection(self.path) as db:
            finish_run(db, identifier, "exited", error="Previous service supervisor exited.", clear_pid=True)

    def lease(self, db):
        now = time.time()
        db.execute("DELETE FROM service_clients WHERE project=? AND seen<?", (self.project, now - LEASE))
        db.execute("INSERT OR REPLACE INTO service_clients VALUES (?,?,?)", (self.project, self.token, now))

    def attach(self, *, existing_only=False):
        with self.lock:
            if self.worker or self.closed.is_set():
                return
            with connection(self.path) as db:
                self.owned = db.execute(
                    "SELECT 1 FROM service_runs WHERE project=? AND owner=? AND consumed=0 LIMIT 1",
                    (self.project, self.owner),
                ).fetchone() is not None
                if existing_only and not db.execute(
                    """SELECT 1 FROM service_runs WHERE project=? AND
                       (finished IS NULL OR (owner=? AND consumed=0)) LIMIT 1""", (self.project, self.owner)
                ).fetchone():
                    return
                self.lease(db)
            self.worker = Thread(target=self.watch, name="mini-pi-services", daemon=True)
            self.worker.start()

    def watch(self):
        seen = set()
        renew_at = 0
        while not self.closed.is_set():
            try:
                now = time.monotonic()
                if now >= renew_at:
                    with connection(self.path) as db:
                        self.lease(db)
                    self.reap()
                    renew_at = now + .8
                with connection(self.path) as db:
                    current = {row["id"] for row in self.completed(db)} if self.owned else set()
                if current - seen:
                    self.wake()
                seen = current
            except (SQLiteError, OSError):
                pass
            self.closed.wait(.15 if self.owned else .6)

    def close(self):
        with self.lock:
            self.closed.set()
            worker = self.worker
        if worker:
            worker.join()
            with connection(self.path) as db:
                db.execute("DELETE FROM service_clients WHERE project=? AND token=?", (self.project, self.token))

    def completed(self, db):
        return db.execute("""SELECT * FROM service_runs WHERE project=? AND owner=?
                             AND consumed=0 AND finished IS NOT NULL AND state<>'stopped' ORDER BY finished,id""",
                          (self.project, self.owner)).fetchall()

    def active(self):
        if not self.owned:
            return False
        with connection(self.path) as db:
            return db.execute("SELECT 1 FROM service_runs WHERE project=? AND owner=? AND finished IS NULL LIMIT 1",
                              (self.project, self.owner)).fetchone() is not None

    def peek(self):
        if not self.owned:
            return None
        with connection(self.path) as db:
            rows = self.completed(db)
        if not rows:
            return None
        lines = [f"Supervised process {row['name']} {row['state']}" +
                 (f" with exit code {row['exit_code']}." if row["exit_code"] is not None else " without an exit code.")
                 for row in rows]
        return {"role": "developer", "content": "\n".join(lines),
                "_service_runs": [snapshot(row) for row in rows]}

    def lookup(self, name=None):
        with connection(self.path) as db:
            if name is not None:
                return db.execute("SELECT * FROM service_runs WHERE project=? AND name=? ORDER BY rowid DESC LIMIT 1",
                                  (self.project, name)).fetchone()
            rows = db.execute("""SELECT * FROM service_runs WHERE rowid IN (
                                 SELECT MAX(rowid) FROM service_runs WHERE project=? GROUP BY name)
                                 ORDER BY finished IS NOT NULL,finished DESC,rowid""", (self.project,)).fetchall()
            return [row for row in rows if row["finished"] is None] + [row for row in rows if row["finished"] is not None][:10]

    def by_id(self, identifier):
        with connection(self.path) as db:
            return db.execute("SELECT * FROM service_runs WHERE id=? AND project=?", (identifier, self.project)).fetchone()

    def inspect(self, name=None):
        self.attach(existing_only=True)
        records = self.lookup(name)
        if name is None:
            return "\n".join(status(row) for row in records)
        if records is not None:
            logs = self.logs(records["id"])
            return status(records) + ("\n" + logs if logs else "")

    def logs(self, identifier):
        text = tail(log_path(self.path, identifier))
        return HARNESS_TAG.sub("&lt;", TERMINAL_ESCAPE.sub("", text)).replace("\r\n", "\n")

    def name_lock(self, name):
        directory = self.path.parent / "services"
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        key = sha256(os.fsencode(self.project) + b"\0" + name.encode()).hexdigest()
        return FileLock(directory / (key + ".lock"), timeout=9)

    def spawn(self, name, spec, owner, mode, *, predecessor=None):
        identifier = uuid4().hex
        with connection(self.path) as db:
            now = time.time()
            db.execute("INSERT INTO service_runs(id,project,name,owner,spec,mode,heartbeat) VALUES (?,?,?,?,?,?,?)",
                       (identifier, self.project, name, owner, json.dumps(spec), mode, now))
            created, restarts = now, 0
            if predecessor:
                previous = db.execute("SELECT created,restarts FROM service_generations WHERE run=?",
                                      (predecessor["id"],)).fetchone()
                created = previous["created"] if previous else predecessor["heartbeat"]
                restarts = previous["restarts"] + 1 if previous else 1
            db.execute("INSERT INTO service_generations VALUES (?,?,?,?)",
                       (identifier, created, now, restarts))
        if owner == self.owner:
            self.owned = True
        try:
            self.launch_worker(identifier)
        except OSError as error:
            with connection(self.path) as db:
                finish_run(db, identifier, "failed", error=str(error))
        return identifier

    def launch_worker(self, identifier, *, recovered=False):
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        command = [sys.executable, str(Path(__file__).with_name("service_runtime.py")), str(self.path), identifier]
        if recovered:
            command.append("recover")
        process = subprocess.Popen(command, env=self.env, cwd=Path(__file__).resolve().parent,
                                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   start_new_session=os.name != "nt", creationflags=flags)
        Thread(target=process.wait, name="mini-pi-service-reaper", daemon=True).start()

    def poll(self, identifier, predicate, timeout, turn=None, interrupt=None):
        end = time.monotonic() + timeout
        while True:
            if turn:
                turn.check()
            row = self.by_id(identifier)
            if predicate(row):
                return row, False, False
            if interrupt and interrupt():
                return row, False, True
            remaining = end - time.monotonic()
            if remaining <= 0:
                return row, True, False
            time.sleep(min(.1, remaining))

    def start(self, name, command, directory, pty, ready, turn, interrupt, call):
        if not NAME.fullmatch(name):
            raise ValueError("Service name must contain 1–48 letters, digits, dots, underscores or hyphens and start with a letter or digit.")
        if ready is not None:
            port, log = ready.get("port"), ready.get("log")
            if port is not None:
                if not 1 <= port <= 65535 or port != int(port):
                    raise ValueError("ready.port must be an integer from 1 to 65535.")
                ready["port"] = int(port)
            if not log and port is None:
                raise ValueError("ready requires a log regex or a TCP port.")
            try:
                if log:
                    re.compile(log)
            except re.error as error:
                raise ValueError(f"Invalid readiness regex: {error}") from error
            timeout = ready.get("timeout", 30)
            timeout = max(.05, min(3600, timeout))
        else:
            timeout = 5
        if turn:
            turn.check()
        argv = shell_plan(command, self.shell, self.env if self.env is not None else os.environ)
        self.attach()
        with self.name_lock(name):
            if prior := self.lookup(name):
                self.stop(prior, turn)
            spec = {"command": command, "argv": argv, "cwd": str(directory), "pty": pty, "ready": ready, "call": call}
            identifier = self.spawn(name, spec, self.owner, "session")
        row, timed_out, interrupted = self.poll(
            identifier, lambda row: row["finished"] is not None or row["ready"] is not None, timeout, turn,
            interrupt if ready else None,
        )
        text = status(row)
        if timed_out:
            text += "\nReadiness timed out. The service can continue starting."
        if interrupted:
            text += "\nReadiness wait interrupted. The service can continue starting."
        if logs := self.logs(identifier):
            text += "\n" + logs[-4000:]
        return tool_result(text, {"service": {**snapshot(row), "timedOut": timed_out, "interrupted": interrupted}})

    def stop(self, row, turn, *, grace=5):
        if row["finished"] is not None:
            return row
        encoded = round(grace * 1000) + 1
        with connection(self.path) as db:
            db.execute("UPDATE service_runs SET stop=CASE WHEN stop=0 THEN ? ELSE MIN(stop,?) END WHERE id=?",
                       (encoded, encoded, row["id"]))
        row, timed_out, _ = self.poll(row["id"], lambda item: item["finished"] is not None, grace + 6, turn)
        if timed_out:
            raise ValueError("The service has not stopped yet.")
        return row

    def manage(self, name, action, *, grace=5, turn=None):
        self.attach(existing_only=True)
        with self.name_lock(name):
            row = self.lookup(name)
            if not row:
                raise ValueError(f"Unknown service {name!r}.")
            if action in ("stop", "kill"):
                return self.stop(row, turn, grace=.1 if action == "kill" else grace)
            if action != "restart":
                raise ValueError(f"Unknown service action {action!r}.")
            self.attach()
            spec = json.loads(row["spec"])
            self.stop(row, turn, grace=2)
            identifier = self.spawn(name, spec, row["owner"], row["mode"], predecessor=row)
            launched, timed_out, _ = self.poll(identifier, lambda item: item["pid"] is not None or item["finished"], 6, turn)
            if timed_out:
                raise ValueError("The replacement service has not started yet.")
            return launched

    def control(self, name, action, content, turn=None):
        self.attach(existing_only=True)
        with self.name_lock(name):
            row = self.lookup(name)
            if not row:
                raise ValueError(f"Unknown service {name!r}.")
            if action == "kill":
                return status(self.stop(row, turn))
            if action == "mode":
                if row["finished"] is not None:
                    raise ValueError(f"The service is {row['state']}.")
                mode = (content or "").strip()
                if mode not in ("persist", "session", "detached"):
                    raise ValueError("Service mode must be persist, session, or detached.")
                if row["mode"] == "detached":
                    if mode == "session":
                        raise ValueError("Detached services must remain persistent.")
                    return status(row)
                if mode == "detached":
                    self.attach()
                    self.stop(row, turn)
                    spec = {**json.loads(row["spec"]), "pty": False}
                    identifier = self.spawn(name, spec, row["owner"], mode, predecessor=row)
                    launched, _, _ = self.poll(identifier, lambda item: item["pid"] is not None or item["finished"], 5, turn)
                    return status(launched)
                with connection(self.path) as db:
                    db.execute("UPDATE service_runs SET mode=? WHERE id=?", (mode, row["id"]))
                return status(self.by_id(row["id"]))
            if content is None or len(content.encode("utf-8")) > 65_536:
                raise ValueError("Service input requires text of at most 64 KiB.")
            if row["finished"] is not None:
                raise ValueError("The service has already stopped.")
            if row["mode"] == "detached":
                raise ValueError("Detached service stdin is unavailable.")
            with connection(self.path) as db:
                request = db.execute("INSERT INTO service_input(run,body) VALUES (?,?)", (row["id"], content)).lastrowid
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if turn:
                turn.check()
            with connection(self.path) as db:
                result = db.execute("SELECT result FROM service_input WHERE id=?", (request,)).fetchone()[0]
                if result is not None:
                    db.execute("DELETE FROM service_input WHERE id=?", (request,))
            if result is not None:
                if result:
                    raise ValueError(result)
                return f"Sent input to {name}."
            time.sleep(.1)
        raise ValueError("Service input is still queued.")
