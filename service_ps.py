"""Inspect and control Mini Pi service supervisors without opening an agent."""

import argparse
import json
import os
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from sqlite3 import Error as SQLiteError

import psutil

from service_logs import LogFollower, clean, log_lines
from service_runtime import NAME, connection, log_path, snapshot, verified_process
from services import Services


GLOBAL = "global:"
TABLE = ("NAME", "STATE", "PID", "UPTIME", "RESTARTS", "FLAGS", "COMMAND")


def command_args(argv, command="ps"):
    index = 0
    while index < len(argv):
        option, equal, _ = argv[index].partition("=")
        if option not in ("--cwd", "--state-dir"):
            break
        index += 1 if equal else 2
    if index < len(argv) and argv[index] == command:
        return [*argv[:index], *argv[index + 1:]]


def parse(argv):
    parser = argparse.ArgumentParser(prog="mini_pi.py ps", description=__doc__)
    parser.add_argument("action", nargs="?", default="list", choices=("list", "info", "logs", "stop", "kill", "restart"))
    parser.add_argument("name", nargs="?", help="service name for an action")
    parser.add_argument("-a", "--all", action="store_true", help="include other project and global scopes")
    parser.add_argument("-j", "--json", action="store_true", help="emit JSON snapshots")
    parser.add_argument("--plain", action="store_true", help="use a static listing")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--dir", type=Path, help="select another project directory")
    target.add_argument("--global", dest="global_scope", help="select a global service scope")
    parser.add_argument("--cwd", type=Path, default=Path.cwd(), help="project directory (default: current directory)")
    parser.add_argument("--state-dir", type=Path, default=Path.home() / ".mini-pi")
    parser.add_argument("-f", "--follow", action="store_true", help="follow new log output")
    parser.add_argument("--head", action="store_true", help="show the beginning of the log")
    parser.add_argument("-n", "--lines", type=int, default=100, help="number of log lines, from 1 to 1000")
    parser.add_argument("--grep", help="filter log lines by regular expression")
    parser.add_argument("--timeout", type=int, default=5, help="stop grace period in seconds")
    args = parser.parse_intermixed_args(argv)
    if args.action != "list" and (not args.name or not NAME.fullmatch(args.name)):
        parser.error(f"{args.action} requires a valid service name")
    if args.action == "list" and args.name:
        parser.error("list does not take a service name")
    if args.global_scope is not None and not NAME.fullmatch(args.global_scope):
        parser.error("invalid global scope")
    if not 0 <= args.timeout <= 3600:
        parser.error("--timeout must be between 0 and 3600 seconds")
    return args


def target_scope(args):
    return (GLOBAL + args.global_scope if args.global_scope is not None
            else os.path.normcase(str((args.dir or args.cwd).resolve())))


def rows(path, scope=None, name=None, *, identifier=None):
    if not path.is_file():
        return []
    with connection(path) as db:
        tables = {table for table, in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "service_runs" not in tables:
            return []
        process = "p.supervisor_pid,p.supervisor_birth,p.process_birth" if "service_processes" in tables else (
            "NULL AS supervisor_pid,NULL AS supervisor_birth,NULL AS process_birth")
        process_join = "LEFT JOIN service_processes p ON p.run=r.id" if "service_processes" in tables else ""
        generation = "g.created,g.started,g.restarts" if "service_generations" in tables else (
            "NULL AS created,NULL AS started,NULL AS restarts")
        generation_join = "LEFT JOIN service_generations g ON g.run=r.id" if "service_generations" in tables else ""
        return db.execute(f"""SELECT r.*,{process},{generation} FROM service_runs r
            {process_join} {generation_join}
            WHERE r.rowid IN (
                SELECT MAX(rowid) FROM service_runs
                WHERE (? IS NULL OR project=?) AND (? IS NULL OR name=?) AND (? IS NULL OR id=?)
                GROUP BY project,name
            ) ORDER BY r.project,r.finished IS NOT NULL,r.finished DESC,r.rowid
            """, (scope, scope, name, name, identifier, identifier)).fetchall()


def alive(pid, birth):
    try:
        return verified_process(pid, birth) is not None
    except psutil.AccessDenied:
        return False


def describe(row, *, spec=False):
    daemon = snapshot(row)
    launch = json.loads(row["spec"])
    running = row["finished"] is None
    supervisor = row["supervisor_pid"]
    supervised = running and alive(supervisor, row["supervisor_birth"])
    terminal_reason = row["error"]
    if running and not supervised and supervisor is not None:
        if row["mode"] != "detached" or not alive(row["pid"], row["process_birth"]):
            daemon.update(state="exited", pid=None)
            terminal_reason = terminal_reason or "Previous service supervisor exited."
    elif running and supervised and row["stop"]:
        daemon["state"] = "stopping"
    started = row["process_birth"] or row["started"] or row["heartbeat"]
    daemon.update(
        owner=row["owner"], createdAt=round((row["created"] or started) * 1000), startedAt=round(started * 1000),
        exitedAt=round(row["finished"] * 1000) if row["finished"] is not None else None,
        exitReason=terminal_reason, restartCount=row["restarts"] or 0, persist=row["mode"] != "session",
        detached=row["mode"] == "detached", command=launch.get("command"), cwd=launch.get("cwd"),
        supervised=supervised, supervisorPid=supervisor if supervised else None,
    )
    if spec:
        argv = launch.get("argv") or []
        daemon["spec"] = {"name": row["name"], "application": argv[0] if argv else None,
                          "args": argv[1:], "cwd": launch.get("cwd"), "pty": bool(launch.get("pty")),
                          "persist": daemon["persist"], "detached": daemon["detached"],
                          "ready": launch.get("ready"), "restart": "no"}
    return daemon


def terminal(daemon):
    return daemon["state"] in ("exited", "stopped", "failed")


def reports(path, scope, *, all_scopes=False, explicit=False):
    reports = {}
    if not all_scopes:
        reports[scope] = []
    target = None if all_scopes or not explicit else scope
    for row in rows(path, target):
        key = row["project"]
        if key != scope and not all_scopes and (explicit or not key.startswith(GLOBAL)):
            continue
        daemon = describe(row)
        if key != scope and not all_scopes and terminal(daemon):
            continue
        reports.setdefault(key, []).append(daemon)
    return reports


def scope_title(scope):
    return f"global {scope[len(GLOBAL):]}" if scope.startswith(GLOBAL) else f"project {scope}"


def cells(daemon):
    state = daemon["state"]
    if terminal(daemon) and daemon["exitCode"] is not None:
        state += f"({daemon['exitCode']})"
    age = max(0, int(time.time() - daemon["startedAt"] / 1000))
    flags = [daemon["mode"]] if daemon["persist"] else []
    if not terminal(daemon) and not daemon["supervised"]:
        flags.append("unsupervised")
    return (daemon["name"], state, str(daemon["pid"]) if daemon["pid"] and not terminal(daemon) else "-",
            "-" if terminal(daemon) else f"{age}s", str(daemon["restartCount"]),
            ",".join(flags), " ".join(clean(daemon["command"] or "").split()))


def table(daemons):
    values = [TABLE, *map(cells, daemons)]
    widths = [max(map(len, column)) for column in zip(*values)]
    return ["  ".join(value.ljust(width) for value, width in zip(row, widths)).rstrip() for row in values]


def list_output(path, data, as_json):
    if as_json:
        result = []
        for scope, daemons in data.items():
            kind = "global" if scope.startswith(GLOBAL) else "project"
            label = {"service": scope[len(GLOBAL):]} if kind == "global" else {"projectDir": scope}
            result.append({"kind": kind, **label, "runtimeDir": str(path.parent / "services"), "daemons": daemons})
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if not data:
        print("No service scopes found.")
    for scope, daemons in data.items():
        print(scope_title(scope))
        for line in table(daemons) if daemons else ["no processes"]:
            print("  " + line)


@contextmanager
def operator(path, scope, workspace):
    # Named operations can migrate a state directory that predates generation metadata.
    from runtime import initialize

    initialize(path)
    client = Services(path, "", workspace, lambda: None, scope=scope)
    try:
        client.attach(existing_only=True)
        if client.worker:
            client.reap()
        yield client
    finally:
        client.close()


def named(path, scope, name):
    matches = rows(path, scope, name)
    if not matches:
        raise ValueError(f"Unknown service {name!r} in {scope_title(scope)}.")
    return matches[0]


def detail_text(daemon):
    spec = daemon["spec"]
    lines = [f"{daemon['name']} [{daemon['state']}]",
             f"  command  {clean(daemon['command'] or '')}", f"  cwd      {clean(daemon['cwd'] or '')}",
             f"  pid      {daemon['pid'] or '-'}", f"  restarts {daemon['restartCount']}",
             f"  pty {spec['pty']}  persist {spec['persist']}  detached {spec['detached']}  owner {daemon['owner'] or '-'}"]
    if daemon["exitReason"]:
        lines.append("  exit     " + clean(daemon["exitReason"]))
    return "\n".join(lines)


def print_lines(lines):
    if lines:
        print("\n".join(lines), flush=True)


def stream_logs(path, scope, name, args):
    pattern = re.compile(args.grep) if args.grep is not None else None
    count = max(1, min(1000, args.lines))
    row = named(path, scope, name)
    identifier = row["id"]
    logfile = log_path(path, identifier)
    if not args.follow:
        print_lines(log_lines(logfile, count, head=args.head, pattern=pattern))
        print(f"[{name}: {describe(row)['state']}]")
        return
    reader = LogFollower(pattern)
    print_lines(reader.start(logfile, count, args.head))
    while True:
        latest = named(path, scope, name)
        lines, more = reader.poll(logfile)
        print_lines(lines)
        if more:
            continue
        if latest["id"] != identifier:
            print_lines(reader.finish())
            identifier = latest["id"]
            logfile = log_path(path, identifier)
            reader = LogFollower(pattern)
        state = describe(latest)["state"]
        if terminal({"state": state}) and latest["id"] == row["id"]:
            print_lines(reader.finish())
            print(f"[{name}: {state}]")
            return
        row = latest
        time.sleep(.15)


def action(path, scope, workspace, name, verb, grace=5):
    with operator(path, scope, workspace) as client:
        row = client.manage(name, verb, grace=grace)
        return describe(rows(path, scope, name, identifier=row["id"])[0])


def run(argv):
    args = parse(argv)
    if not sys.stdout.isatty() and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    path = args.state_dir.resolve() / "sessions.sqlite3"
    scope = target_scope(args)
    explicit = args.dir is not None or args.global_scope is not None
    try:
        if args.action == "list":
            if not args.plain and not args.json and sys.stdin.isatty() and sys.stdout.isatty():
                from service_monitor import monitor

                monitor(path, scope, args.cwd, args.all, explicit)
            else:
                list_output(path, reports(path, scope, all_scopes=args.all, explicit=explicit), args.json)
            return 0
        named(path, scope, args.name)
        if args.action in ("info", "logs"):
            with operator(path, scope, args.cwd):
                if args.action == "logs":
                    stream_logs(path, scope, args.name, args)
                else:
                    daemon = describe(named(path, scope, args.name), spec=True)
                    print(json.dumps(daemon, ensure_ascii=False, indent=2) if args.json else detail_text(daemon))
        else:
            daemon = action(path, scope, args.cwd, args.name, args.action, args.timeout)
            verb = {"stop": "Stopped", "kill": "Killed", "restart": "Restarted"}[args.action]
            if args.json:
                print(json.dumps(daemon, ensure_ascii=False, indent=2))
            else:
                print(f"{verb} {daemon['name']} [{daemon['state']}]")
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, SQLiteError, re.error) as error:
        print(f"mini-pi ps: {error}", file=sys.stderr)
        return 1
    return 0
