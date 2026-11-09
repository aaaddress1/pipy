"""Prelaunch closed task objects and hand their runs to normal tool dispatch."""

import json
from concurrent.futures import ThreadPoolExecutor
from sqlite3 import Error as SQLiteError
from threading import Lock

from jsonschema import Draft202012Validator

from auth_broker import dotted
def enabled(layers):
    launch, mode, approval, restricted = True, "yolo", None, None
    for layer in layers:
        values = [dotted(layer, key) for key in (
            "task.speculativeLaunch", "tools.approvalMode", "tools.approval.task", "tools.restrictedFolders",
        )]
        launch, mode, approval, restricted = [
            current if value is None else value for current, value in zip((launch, mode, approval, restricted), values)
        ]
    return launch is not False and not restricted and (approval == "allow" if approval is not None else mode == "yolo")


class BatchArguments:
    """Scan only the appended bytes, retaining closed context and task objects."""

    def __init__(self):
        self.raw = ""
        self.stack = []
        self.string = None
        self.escape = False
        self.key = None
        self.expect_key = False
        self.task_array = False
        self.item = None
        self.context = None
        self.items = []
        self.invalid = False

    def feed(self, raw):
        if not raw.startswith(self.raw):
            self.__init__()
        start, self.raw = len(self.raw), raw
        for pos in range(start, len(raw)):
            if self.invalid:
                break
            char = raw[pos]
            try:
                self.scan(char, pos)
            except ValueError:
                self.invalid = True

    def scan(self, char, pos):
        depth = len(self.stack)
        if self.string is not None:
            if self.escape:
                self.escape = False
            elif char == "\\":
                self.escape = True
            elif char == '"':
                if depth == 1:
                    value = json.loads(self.raw[self.string:pos + 1])
                    if self.expect_key:
                        self.key = value
                    elif self.key == "context":
                        self.context = value
                self.string = None
            return
        at_item = self.task_array and depth == 2
        top_tasks = depth == 1 and not self.expect_key and self.key == "tasks"
        if char == '"':
            if at_item or top_tasks:
                raise ValueError()
            self.string = pos
        elif char in "{[":
            if (depth == 0 and char != "{") or (at_item and char != "{") or (top_tasks and char != "["):
                raise ValueError()
            self.stack.append(char)
            if top_tasks:
                self.task_array = True
            if at_item:
                self.item = pos
            if depth == 0:
                self.expect_key = True
        elif char in "}]":
            if not self.stack or self.stack.pop() != ("{" if char == "}" else "["):
                raise ValueError()
            if self.task_array and depth == 3:
                self.items.append(json.loads(self.raw[self.item:pos + 1]))
                self.item = None
            elif at_item:
                self.task_array = False
        elif char == ":" and depth == 1:
            self.expect_key = False
        elif char == "," and depth == 1:
            self.expect_key = True
        elif char not in " \t\r\n," and (at_item or top_tasks or depth == 0):
            raise ValueError()


class TaskCall:
    def __init__(self, tasks, schema, executor):
        self.tasks, self.schema, self.executor = tasks, schema, executor
        self.scanner = BatchArguments()
        self.runs = []
        self.halted = False
        self.closed = False

    def update(self, raw, final=False):
        if self.closed or (self.halted and not final) or not isinstance(raw, str):
            return
        if final:
            try:
                args = json.loads(raw)
            except (ValueError, RecursionError):
                self.discard()
                return
        else:
            self.scanner.feed(raw)
            if self.scanner.invalid or not self.scanner.context or not self.scanner.items:
                return
            args = {"context": self.scanner.context, "tasks": self.scanner.items}
        if not isinstance(args, dict) or args.keys() != {"context", "tasks"} or not isinstance(args["tasks"], list):
            self.discard()
            return
        items = args["tasks"]
        if final:
            validator = Draft202012Validator(self.schema)
            if not items or not all(validator.is_valid(item) for item in items):
                self.discard()
                return
            plan = self.prepare(args, len(self.runs)) if self.runs else []
            if plan is None or not self.matches(plan):
                self.discard()
                return
        for index in range(len(self.runs), len(items)):
            if self.halted or not isinstance(items[index], dict) or "taskFile" in items[index] or items[index].get("tools"):
                self.halted = True
                return
            plan = self.prepare(args, index + 1)
            if plan is None:
                self.halted = True
                return
            try:
                self.runs.append(self.tasks.prelaunch(plan[index], self.executor))
            except (OSError, RuntimeError, SQLiteError):
                self.halted = True
                return

    def prepare(self, args, count):
        try:
            items = args["tasks"][:count]
            if len(items) == count and all(isinstance(item, dict) and "taskFile" not in item and not item.get("tools") for item in items):
                return self.tasks.prepare({**args, "tasks": items}, self.schema)
        except (ValueError, OSError, RecursionError, SQLiteError):
            pass

    def matches(self, plan):
        return len(self.runs) <= len(plan) and all(
            run.matches(task) for run, task in zip(self.runs, plan)
        )

    def discard(self):
        self.closed = True
        for run in self.runs:
            run.cancel.cancel()
            run.decided.set()
        for run in self.runs:
            run.discard()
        self.runs.clear()


class TaskSpeculation:
    def __init__(self, tasks, schema):
        self.tasks, self.schema = tasks, schema
        self.executor = ThreadPoolExecutor(max_workers=tasks.options.concurrency or 32, thread_name_prefix="mini-pi-task-preview")
        self.pending, self.claims = {}, {}
        self.lock = Lock()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.discard()
        self.executor.shutdown(wait=False)

    def begin(self, _parent):
        self.discard()

    def restart(self):
        self.discard()

    def observe(self, index, name, identifier, raw, *, final=False):
        if name != "task" or not identifier:
            return
        entry = self.pending.get(index)
        if entry is None:
            entry = self.pending[index] = (identifier, TaskCall(self.tasks, self.schema, self.executor))
        if entry[0] == identifier:
            entry[1].update(raw, final)

    def seal(self, calls):
        final = {call["id"]: call["function"] for call in calls if call["function"]["name"] == "task"}
        for identifier, entry in self.pending.values():
            if identifier not in final:
                entry.discard()
                continue
            entry.update(final[identifier]["arguments"], True)
            if not entry.closed:
                self.claims[identifier] = entry

    def take_task(self, call):
        with self.lock:
            return self.claims.pop(call["id"], None)

    def discard(self):
        for _, entry in self.pending.values():
            entry.discard()
        self.pending.clear()
        self.claims.clear()


class StreamSpeculation:
    def __init__(self, read, task, shadow=None):
        self.read, self.task, self.shadow = read, task, shadow
        self.active = ()

    def configure(self, *, read, task, shadow=False):
        streams = ((self.read, read), (self.task, task), (self.shadow, read and shadow))
        self.active = tuple(stream for stream, enabled in streams if stream and enabled)
        return self if self.active else None

    def begin(self, parent):
        for stream in self.active:
            stream.begin(parent)

    def restart(self):
        for stream in self.active:
            stream.restart()

    def observe(self, *args, final=False):
        for stream in self.active:
            stream.observe(*args, final=final)

    def seal(self, calls):
        for stream in self.active:
            stream.seal(calls)

    def take(self, call):
        return self.read.take(call) if self.read in self.active else None

    def take_task(self, call):
        return self.task.take_task(call) if self.task in self.active else None

    def take_eval(self, call):
        return self.shadow.take(call) if self.shadow in self.active else None

    def discard(self):
        for stream in (self.read, self.shadow):
            if stream in self.active:
                stream.cancel()
        for stream in self.active:
            stream.discard()
