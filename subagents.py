"""Task dispatch, limits and durable child session links."""

import json
import os
import re
from concurrent.futures import CancelledError, ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from logging import getLogger
from pathlib import Path
from sqlite3 import Error as SQLiteError
from threading import Event, Lock, Semaphore
from time import monotonic
from uuid import uuid4

import yaml
from filelock import Timeout
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from async_jobs import tool_result
from auth_broker import dotted
from cancellable_http import active_turn
from handoff import Cancellation, deadline
from runtime import Session, database, pack
from session_graph import delete_task_subtree, task_subtree
from session_jsonl import journal_entries, plain_text, text_and_images
from session_leases import upstream_lease
from session_transfer import archive_session_id, lease
import task_isolation
from task_yield import YieldState, yield_error


NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,31}\Z")
MAX_YIELD_ERRORS = 6
PROMPTS = Path(__file__).with_name("prompts") / "agents"
logger = getLogger(__name__)


@dataclass
class Tool:
    description: str
    parameters: dict
    execute: object
    task_dispatch: object = None
    call_aware: bool = False


class PrelaunchedTask:
    def __init__(self, tasks, task, executor):
        self.tasks, self.task = tasks, task
        self.cancel = Cancellation(reason="Speculative task was discarded.")
        self.decided = Event()
        self.adopted = False
        with database(tasks.session.path) as db:
            db.execute("INSERT INTO speculative_task_roots(session) VALUES (?)", (task.sid,))
        self.future = executor.submit(tasks.run_task, task, self.cancel, False, self)

    def matches(self, task):
        return self.task == replace(task, sid=self.task.sid)

    def attach(self, owner):
        self.cancel.attach(owner)
        self.adopted = True
        self.decided.set()
        try:
            result, data = self.future.result()
            return self.tasks.complete(self.task, result, data)
        except KeyboardInterrupt:
            self.cancel.cancel()
            self.tasks.observe(self.task, "failed")
            raise
        except Exception:
            self.tasks.observe(self.task, "failed")
            raise

    def discard(self):
        self.cancel.cancel()
        self.decided.set()
        try:
            if not self.future.cancel():
                self.future.result()
        except Exception:
            logger.exception("Discarded speculative task %s failed", self.task.name)
        try:
            path = self.tasks.session.path
            with database(path) as db:
                children = task_subtree(db, self.task.sid)
            if self.tasks.peer:
                self.tasks.peer.bus.discard_speculation(children)
            with database(path) as db:
                delete_task_subtree(db, children)
            artifacts = path.parent / "artifacts"
            for sid in children:
                directory = artifacts / sid
                if directory.is_dir() and directory.parent == artifacts:
                    task_isolation.remove_tree(directory, artifacts)
        except (OSError, SQLiteError):
            logger.exception("Unable to clean up discarded speculative task %s", self.task.name)


@dataclass(frozen=True)
class Agent:
    name: str
    description: str
    prompt: str
    tools: tuple | None = None
    spawns: tuple = ()
    model: object = None
    effort: str | None = None
    blocking: bool = False
    structured: bool = False
    output: object = None
    source: str = "bundled"


def names(value):
    values = value.split(",") if isinstance(value, str) else value if isinstance(value, list) else ()
    return tuple(item.strip() for item in values if isinstance(item, str) and item.strip())


def agents(user, workspace):
    result = {"task": Agent("task", "General-purpose worker", (PROMPTS / "task.md").read_text(encoding="utf-8"), spawns=("*",))}
    for source, directory in (("bundled", PROMPTS), ("user", user), ("project", workspace / ".omp" / "agents")):
        for path in sorted(directory.glob("*.md")):
            try:
                raw = path.read_text(encoding="utf-8-sig")
            except (OSError, UnicodeError):
                continue
            if not raw.startswith("---\n"):
                continue
            header, separator, prompt = raw[4:].partition("\n---\n")
            if not separator:
                continue
            try:
                meta = yaml.safe_load(header)
            except yaml.YAMLError:
                continue
            if not isinstance(meta, dict) or not isinstance(name := meta.get("name"), str) or not NAME.fullmatch(name):
                continue
            description = meta.get("description")
            if name.lower() in ("main", "sub", "seance") or not isinstance(description, str) or not description.strip():
                continue
            tools = None if meta.get("tools") is None else names(meta["tools"])
            spawns = tuple(item.lower() for item in names(meta.get("spawns"))) if "spawns" in meta else ()
            if "spawns" not in meta and "task" in (tools or ()):
                spawns = ("*",)
            result[name.lower()] = Agent(name, description, prompt.strip(), tools, spawns, meta.get("model"),
                                         meta.get("thinking-level", meta.get("thinking")), meta.get("blocking") is True,
                                         meta.get("structured") is True, meta.get("output"), source)
    return result


def jtd_schema(value):
    if not isinstance(value, dict):
        return value
    primitives = dict.fromkeys(("int8", "uint8", "int16", "uint16", "int32", "uint32"), "integer")
    primitives.update(timestamp="string", float32="number", float64="number", string="string", boolean="boolean")
    maps = ("properties", "optionalProperties", "definitions", "mapping")
    jtd_keys = set(maps) | {"metadata", "nullable", "ref", "type", "enum", "elements", "additionalProperties", "values", "discriminator"}

    def jtd(node):
        return (isinstance(node, dict) and node.keys() <= jtd_keys
                and ("type" not in node or isinstance(node["type"], str) and node["type"] in primitives)
                and all(isinstance(node[key], str) for key in ("ref", "discriminator") if key in node)
                and all(type(node[key]) is bool for key in ("nullable", "additionalProperties") if key in node)
                and ("enum" not in node or isinstance(node["enum"], list) and all(isinstance(item, str) for item in node["enum"]))
                and all(jtd(node[key]) for key in ("elements", "values") if key in node)
                and all(isinstance(node[key], dict) and all(jtd(child) for child in node[key].values())
                        for key in maps if key in node))

    def convert(node):
        if "enum" in node:
            result = {"enum": node["enum"]}
        elif "elements" in node:
            result = {"type": "array", "items": convert(node["elements"])}
        elif "type" in node:
            result = {"type": primitives[node["type"]]}
        elif "values" in node:
            result = {"type": "object", "additionalProperties": convert(node["values"])}
        elif "properties" in node or "optionalProperties" in node:
            required = node.get("properties", {})
            properties = {name: convert(child) for name, child in {**required, **node.get("optionalProperties", {})}.items()}
            result = {"type": "object", "properties": properties, "required": list(required), "additionalProperties": False}
        elif "discriminator" in node and "mapping" in node:
            tag = node["discriminator"]
            result = {"oneOf": []}
            for name, child in node["mapping"].items():
                variant = convert(child)
                variant.setdefault("properties", {})[tag] = {"const": name}
                required = variant.setdefault("required", [])
                if tag not in required:
                    required.append(tag)
                result["oneOf"].append(variant)
        elif "ref" in node:
            reference = node["ref"].replace("~", "~0").replace("/", "~1")
            result = {"$ref": "#/$defs/" + reference}
        else:
            result = {}
        return {"anyOf": [result, {"type": "null"}]} if node.get("nullable") is True else result

    kind = value.get("type")
    recognizable = (bool(value.keys() & {"elements", "values", "optionalProperties", "discriminator", "ref"})
                    or isinstance(kind, str) and kind in primitives and kind not in ("string", "boolean")
                    or "properties" in value and "type" not in value)
    if recognizable and jtd(value):
        result = convert(value)
        if value.get("definitions"):
            result["$defs"] = {name: convert(child) for name, child in value["definitions"].items()}
        return result
    return value


def local_references(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in ("$ref", "$dynamicRef") and isinstance(child, str) and not child.startswith("#"):
                raise ValueError("Task output schemas must use local references.")
            local_references(child)
    elif isinstance(value, list):
        for child in value:
            local_references(child)


@dataclass(frozen=True)
class TaskOptions:
    batch: bool
    files: bool
    structured: bool
    effort: bool
    depth: int
    concurrency: int
    timeout: float
    requests: int
    wrap_notice: bool
    request_overrides: dict
    isolation: bool
    isolation_apply: bool
    isolation_merge: str
    isolation_backend: str
    idle_ttl: float

    @classmethod
    def load(cls, settings):
        def value(key, default):
            item = dotted(settings, "task." + key)
            return item if type(item) is type(default) else default

        merge = value("isolation.merge", "patch")
        if merge not in ("patch", "branch"):
            raise ValueError("task.isolation.merge must be patch or branch.")
        backend = dotted(settings, "isolation.backend") or "auto"
        if backend not in ("auto", "apfs", "btrfs", "zfs", "reflink", "overlayfs", "projfs", "block-clone", "rcopy"):
            raise ValueError("Unsupported isolation.backend setting.")
        return cls(value("batch", True), value("file", False), value("structured", False), value("enableEffort", False),
                   value("maxRecursionDepth", 2), max(0, value("maxConcurrency", 32)), max(0, value("maxRuntimeMs", 0)) / 1000,
                   max(0, value("softRequestBudget", 200)), value("softRequestBudgetNotice", True),
                   value("agentSoftRequestBudget", {}), value("isolation.enabled", False), value("isolation.apply", True),
                   merge, backend, max(0, value("agentIdleTtlMs", 420_000)) / 1000)


@dataclass(frozen=True)
class Task:
    name: str
    sid: str
    assignment: str
    solution_space: str
    context: str
    agent: Agent
    routes: tuple
    effort: str | None
    structured: bool
    schema: object
    schema_mode: str
    file: str | None = None
    file_hash: str | None = None
    repo_root: Path | None = None
    depth: int = 1
    workspace: str | None = None
    eval_schema_source: str | None = None
    eval_label: str | None = None
    isolation_apply: bool | None = None
    isolation_merge: str | None = None
    per_call_model: str | None = None
    parent_model: str | None = None
    eval_tool_owner: str | None = None
    eval_tools: tuple = ()
    workpool: bool = False

    def spec(self):
        data = asdict(self)
        data["repo_root"] = str(self.repo_root) if self.repo_root else None
        return data

    @classmethod
    def restore(cls, data):
        data["agent"] = Agent(**data["agent"])
        if data.get("repo_root"):
            data["repo_root"] = Path(data["repo_root"])
        return cls(**data)


class TaskLimitReached(RuntimeError):
    pass


class TaskRun:
    def __init__(self, task, options, *, provisional=False, progress=None, pool_items=(), pool_followup=False, restore_plan=None):
        self.task = task
        self.provisional = provisional
        self.restore_plan = restore_plan
        self.count = 0
        override = options.request_overrides.get(task.agent.name)
        self.limit = override if type(override) is int and override >= 0 else options.requests
        if type(override) is not int:
            self.limit = min(self.limit, {"scout": 100, "sonic": 100, "default": 200}.get(task.agent.name, self.limit))
        self.notice = options.wrap_notice
        self.warned = False
        self.budget_wrap = False
        self.submitted = self.parked_yield = None
        self.yields = YieldState(task.schema, task.schema_mode) if task.structured and not pool_items else None
        self.force_yield = False
        self.yield_aborted = False
        self.yield_errors = 0
        self.schema_warning = None
        self.suppress_delivery = False
        self.progress = progress
        self.model = None
        self.pool_items, self.pool_followup = pool_items, pool_followup
        self.pool_results, self.pool_errors = {}, set()
        self.pool_reusable = False
        self.context_tokens = self.context_window = None

    def request(self):
        if self.limit and self.count >= self.budget_threshold + 5:
            raise TaskLimitReached("Subagent request budget reached. Returning partial findings.")
        self.count += 1
        if self.progress:
            self.progress(self)

    @property
    def budget_threshold(self):
        return (3 * self.limit + 1) // 2

    def budget_stopped(self):
        return self.limit > 0 and self.count >= self.budget_threshold

    def reminder(self):
        if self.budget_stopped() and not self.budget_wrap:
            self.budget_wrap = True
            instruction = ("Call yield now with your best report from completed work." if self.task.structured
                           else "Reply now with your best report from completed work. Tools are disabled.")
            return f"<system-reminder>Request budget crossed. {instruction} Mark gaps as incomplete.</system-reminder>"
        if self.limit and self.notice and not self.warned and self.count >= self.limit:
            self.warned = True
            return (f"[budget notice] Used {self.count} requests (soft budget: {self.limit}). "
                    f"Wrap up now. At {self.budget_threshold} requests you will be asked to submit your best result.")

    def yield_reminder(self, retry):
        if self.pool_items:
            remaining = [i for i, key in enumerate(self.pool_items, 1) if key not in self.pool_results]
            text = f"Use the yield tool once per remaining workpool key: {remaining}."
        else:
            text = ("Call the next tool if work remains. Submit an incremental section with type as an array, "
                    "a complete report with data, or an error for a concrete blocker.")
        return f"<system-reminder>Turn ended without a tool call. Reminder {retry} of 3. {text}</system-reminder>"
        return None

    def yield_tool(self):
        if self.pool_items:
            return self.pool_yield_tool()
        if self.yields is None:
            self.yields = YieldState(None, self.task.schema_mode)

        def submit(args):
            result = self.yields.submit(args, force=self.force_yield)
            if self.yields.terminal is not None:
                self.finish_yield()
            return result

        description = (
            'Submit {"data": ...} for a complete result or {"error": "..."} to abort. '
            'Use type: ["section"] to add an incremental section without ending the turn. '
            'Use type: "result" without data to finalize the collected sections.'
        )
        if self.yields.schema.validator:
            description += " A complete result must match the declared schema."
        else:
            description += " With no sections, a data-less terminal yield uses your final report text."
        return Tool(description, self.yields.schema.parameters, submit)

    def finish_yield(self):
        data, error, self.schema_warning = self.yields.finalize()
        self.yield_aborted = (self.yields.terminal or {}).get("status") == "aborted"
        self.submitted = {"data": data, "error": error}

    def yield_context(self, assistant, session, seq):
        previous = next((message for old_seq, group in reversed(session.batches) if old_seq != seq
                         for message in reversed(group) if not (
                             message["role"] == "developer" and plain_text(message.get("content")).startswith("<system-reminder>")
                         )), None)
        self.yields.note_assistant(assistant, previous)

    def yield_result(self, result):
        if self.submitted is not None:
            return
        if result.get("_tool_error"):
            self.yield_errors += 1
            if self.yield_errors >= MAX_YIELD_ERRORS:
                error = plain_text(result["content"])
                partial = self.pool_results if self.pool_items else self.yields.sections if self.yields else None
                self.submitted = {"data": partial or None, "error":
                                  f"Subagent submitted invalid yield results {self.yield_errors} times. Last yield error: {error}"}
        elif result.get("_tool_details", {}).get("status") == "success":
            self.yield_errors = 0

    def invalidate_yield(self):
        submission = self.submitted or self.parked_yield
        if self.yields and submission:
            self.yields.invalidate(submission)
        self.submitted = self.parked_yield = None
        self.yield_aborted = False
        self.schema_warning = None

    def pool_yield_tool(self):
        keys = list(range(1, len(self.pool_items) + 1))

        def submit(args):
            if not isinstance(args, dict) or any(value is not None and name not in ("key", "data", "error")
                                                 for name, value in args.items()):
                raise ValueError("Submit only key, data, or error for a workpool item.")
            key = args.get("key")
            if type(key) not in (int, float) or key not in keys:
                raise ValueError("key must be one of: " + ", ".join(map(str, keys)))
            key = int(key)
            identifier = self.pool_items[key - 1]
            if identifier in self.pool_results:
                raise ValueError(f"workpool item {key} was already submitted")
            data = args.get("data")
            error = yield_error(data, args.get("error"))
            if (data is None) == (not error):
                raise ValueError("workpool yield requires exactly one of data or error")
            if error:
                self.pool_errors.add(identifier)
                data = {"error": error}
            self.pool_results[identifier] = data
            remaining = [index for index, item in enumerate(self.pool_items, 1) if item not in self.pool_results]
            details = {"status": "success", "type": [identifier], "data": data}
            if error:
                details["error"] = error
            if remaining:
                text = f"Item {key} submitted. Remaining item(s): {', '.join(map(str, remaining))}."
            else:
                self.submitted = {"data": dict(self.pool_results), "error": None}
                details["complete"] = True
                text = f"Item {key} submitted. All workpool items are complete; ending this turn."
            return tool_result(text, details)

        parameters = {"type": "object", "additionalProperties": False, "description": "submit one workpool item outcome",
                      "required": ["key"], "properties": {
                          "key": {"enum": keys, "description": "1-based workpool item number"},
                          "data": {"description": "Self-contained outcome and evidence for this item"},
                          "error": {"type": "string", "description": "Failure reason for this item"},
                      }}
        return Tool("Submit ONE workpool item at a time as { key, data } or { key, error }. The final key ends this turn.",
                    parameters, submit)


class Tasks:
    def __init__(self, session, workspace, options, definitions, allowed, resolve, run, jobs, peer=None,
                 *, background=True, depth=0):
        self.session, self.workspace, self.options, self.definitions = session, workspace, options, definitions
        self.allowed, self.resolve, self.run, self.jobs = allowed, resolve, run, jobs
        self.peer, self.background, self.depth = peer, background, depth
        self.default = "task" if "*" in allowed or "task" in allowed else next(name for name in allowed if name in definitions)
        self.slots = Semaphore(options.concurrency) if options.concurrency else None
        self.dispatch_lock = Lock()
        self.resume_eval = None
        self.eval_tools = None
        self.observer = None

    def tool(self, turn):
        string = {"type": "string"}
        properties = {key: string for key in ("name", "agent", "model", "task", "solutionSpace")}
        if self.options.files:
            properties["taskFile"] = string
        if self.options.effort:
            properties["effort"] = {"type": "string", "enum": ["lo", "med", "hi"]}
        if self.options.isolation:
            properties["isolated"] = {"type": "boolean", "description": "Run in a separate Git checkout and capture task changes."}
        if self.eval_tools and self.eval_tools.enabled():
            properties["tools"] = {"type": "array", "items": string,
                                   "description": "Named tools defined in this session's retained eval kernels."}
        if self.options.structured or any(agent.structured for agent in self.definitions.values()):
            properties.update(outputSchema={}, schemaMode={"type": "string", "enum": ["permissive", "strict"]})
        item = {"type": "object", "properties": properties, "required": ["solutionSpace"] + ([] if self.options.files else ["task"]),
                "additionalProperties": False}
        parameters = {"type": "object", "properties": {"context": string, "tasks": {"type": "array", "items": item, "minItems": 1}},
                      "required": ["context", "tasks"], "additionalProperties": False} if self.options.batch else item
        description = "Delegate work to child agents. Describe how open-ended each task is in solutionSpace. Available agents: "
        description += ", ".join(agent.name for key, agent in self.definitions.items() if "*" in self.allowed or key in self.allowed)
        if self.jobs and self.background:
            description += ". Background results deliver automatically. Use wait only when you have no other useful work."
        def dispatch(args, staged=None):
            return self.dispatch(args, item, active_turn() or turn, staged)

        return Tool(description, parameters, dispatch, dispatch if self.options.batch else None)

    def prelaunch(self, task, executor):
        return PrelaunchedTask(self, task, executor)

    def file_text(self, path):
        if path.startswith("local://"):
            path = str(self.workspace / path.removeprefix("local://"))
        target = Path(path)
        if not target.is_absolute() or not target.resolve().is_relative_to(self.workspace.resolve()):
            raise ValueError("Task file must be an absolute path within the workspace or a local:// URL.")
        target = target.resolve()
        if not target.is_file() or target.stat().st_size > 4 * 1024 * 1024:
            raise ValueError("Task file must be a regular file of at most 4 MiB.")
        with target.open("rb") as file:
            data = file.read(4 * 1024 * 1024 + 1)
        if len(data) > 4 * 1024 * 1024:
            raise ValueError("Task file exceeds 4 MiB.")
        return data.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n").strip(), sha256(data).hexdigest()

    def prepare(self, args, schema):
        if not isinstance(args, dict):
            raise ValueError("Task parameters must be an object.")
        context, items = "", [args]
        if "tasks" in args or "context" in args:
            if not self.options.batch or args.keys() != {"context", "tasks"}:
                raise ValueError("Batch tasks require only context and tasks.")
            context, items = args["context"], args["tasks"]
            if not isinstance(context, str) or not context.strip() or not isinstance(items, list) or not items:
                raise ValueError("Provide nonempty context and at least one task.")
        prepared, requested = [], set()
        repo_root = None
        with database(self.session.path) as db:
            existing = {name.lower() for name, in db.execute("SELECT name FROM subagent_links WHERE owner = ?", (self.session.id,))}
        for item in items:
            if invalid := next(Draft202012Validator(schema).iter_errors(item), None):
                raise ValueError(f"Invalid task: {invalid.message}")
            name = item.get("name", "").strip()
            if name:
                if not NAME.fullmatch(name) or name.lower().startswith("bg_") or name.lower() in requested:
                    raise ValueError(f"Invalid or duplicate task name: {name!r}.")
                requested.add(name.lower())
            base = name or "Task"
            suffix = 1
            while not name or name.lower() in existing:
                tag = str(suffix)
                name = base[:32 - len(tag)] + tag
                suffix += 1
            existing.add(name.lower())
            key = item.get("agent", self.default).strip().lower()
            if "*" not in self.allowed and key not in self.allowed or key not in self.definitions:
                raise ValueError(f"Agent {key!r} is unavailable for this task.")
            agent = self.definitions[key]
            file = item.get("taskFile")
            assignment = item.get("task", "").strip()
            file_hash = None
            if file is not None:
                if "task" in item or not file.strip():
                    raise ValueError("Specify exactly one of task and taskFile.")
                assignment, file_hash = self.file_text(file)
            solution = item["solutionSpace"].strip()
            if not assignment or not solution:
                raise ValueError("Each task needs nonempty instructions and solutionSpace.")
            routes, effort, selector, parent = self.resolve(agent, item)
            structured = self.options.structured or agent.structured
            if not structured and {"outputSchema", "schemaMode"} & item.keys():
                raise ValueError("Structured output is disabled for this task.")
            output = None
            if structured:
                output = item["outputSchema"] if "outputSchema" in item else jtd_schema(agent.output)
            if isinstance(output, str):
                try:
                    output = json.loads(output)
                except json.JSONDecodeError as error:
                    raise ValueError("outputSchema must contain a JSON Schema.") from error
            if "outputSchema" in item:
                output = jtd_schema(output)
            if output is not None:
                local_references(output)
                try:
                    Draft202012Validator.check_schema(output)
                except SchemaError as error:
                    raise ValueError(f"Invalid output schema: {error.message}") from error
            if item.get("isolated") and repo_root is None:
                try:
                    repo_root = task_isolation.repository(self.workspace)
                except (task_isolation.GitError, OSError) as error:
                    raise ValueError("Isolated tasks require a Git checkout.") from error
            custom = self.eval_tools.describe(item["tools"]) if item.get("tools") and self.eval_tools else ()
            prepared.append(Task(name, str(uuid4()), assignment, solution, context.strip(), agent, routes, effort,
                                 structured, output, item.get("schemaMode", "permissive"), file, file_hash,
                                 repo_root if item.get("isolated") else None, self.depth + 1, str(self.workspace),
                                 per_call_model=selector, parent_model=parent,
                                 eval_tool_owner=self.eval_tools.id if custom else None, eval_tools=custom))
        return prepared

    def dispatch(self, args, schema, turn, staged=None):
        try:
            start = monotonic()
            with self.dispatch_lock:
                if turn:
                    turn.check()
                tasks = self.prepare(args, schema)
                if turn:
                    turn.check()
                adopted = {}
                if staged:
                    if staged.matches(tasks):
                        for index, run in enumerate(staged.runs):
                            tasks[index] = run.task
                            adopted[run.task.sid] = run
                    else:
                        staged.discard()
                background = [task for task in tasks if self.jobs and self.background and not task.agent.blocking]
                inline = [task for task in tasks if task not in background]

                def reserve():
                    self.reserve(tasks, adopted)
                    if adopted:
                        staged.runs.clear()

                def execute(task, cancel):
                    if run := adopted.get(task.sid):
                        return run.attach(cancel)
                    return self.execute(task, cancel)

                if background:
                    rows = self.jobs.spawn([
                        (task.name, task.assignment, task.sid, lambda cancel, spec=task: execute(spec, cancel))
                        for task in background
                    ], reserve)
                else:
                    reserve()
                    rows = []
            results = []
            if inline:
                signals = [Cancellation(turn) for _ in inline]
                with ThreadPoolExecutor(max_workers=min(len(inline), self.options.concurrency or len(inline))) as workers:
                    if self.peer:
                        self.peer.wait_for({task.sid for task in inline})
                    futures = [workers.submit(execute, task, signal) for task, signal in zip(inline, signals)]
                    try:
                        results = [future.result() for future in futures]
                    except KeyboardInterrupt:
                        for signal in signals:
                            signal.cancel()
                        raise
                    finally:
                        if self.peer:
                            self.peer.wait_for(set())
                if turn:
                    turn.check()
            text = []
            if background:
                text.append("\n".join(f"Spawned agent `{task.name}` (job `{task.name}`)." for task in background))
            text.extend(result["report"] for result in results)
            details = {"results": [{k: v for k, v in result.items() if k != "report"} for result in results],
                       "progress": rows, "totalDurationMs": round((monotonic() - start) * 1000)}
            if rows:
                details["async"] = {"state": "running", "jobId": rows[0]["jobId"], "type": "task"}
            return tool_result("\n\n".join(text), details, bool(results) and not rows and all(r["status"] != "completed" for r in results))
        except (OSError, ValueError) as error:
            return tool_result(str(error), {"results": [], "progress": []}, True)
        finally:
            if staged and staged.runs:
                staged.discard()

    def reserve(self, tasks, adopted=()):
        with database(self.session.path) as db:
            db.executemany("INSERT INTO subagent_links(owner,name,child) VALUES (?,?,?)",
                           ((self.session.id, task.name, task.sid) for task in tasks))
            db.executemany("INSERT INTO subagent_specs(child,data) VALUES (?,?)",
                           ((task.sid, pack(task.spec())) for task in tasks))
            db.executemany("DELETE FROM speculative_task_roots WHERE session = ?", ((identifier,) for identifier in adopted))
        if self.peer:
            self.peer.bus.mark(*(task.sid for task in tasks))
        for task in tasks:
            self.observe(task, "running")

    def observe(self, task, status):
        if observer := self.observer:
            try:
                observer(task, status)
            except (OSError, SQLiteError):
                logger.exception("Unable to update the task display for %s", task.name)

    def restoration_plan(self):
        plan = {}
        owners = [self.session.id]
        visited = {self.session.id}
        path = self.session.path
        for owner in owners:
            with database(path) as db:
                candidates = [(*row, archive_session_id(db, row[0])) for row in db.execute("""
                    SELECT l.child,l.name,s.workspace,spec.data,
                           dead.child IS NOT NULL OR coalesce(json_extract(l.result,'$.status'),'')='cancelled'
                           OR EXISTS(SELECT 1 FROM subagent_deliveries d WHERE d.child=l.child AND d.settled_leaf=head.leaf)
                    FROM subagent_links l LEFT JOIN subagent_specs spec ON spec.child=l.child
                    LEFT JOIN sessions s ON s.id=l.child
                    LEFT JOIN session_heads head ON head.session=l.child
                    LEFT JOIN subagent_tombstones dead ON dead.child=l.child
                    WHERE l.owner=? ORDER BY l.rowid
                """, (owner,))]
            for sid, name, scope, data, killed, upstream_id in candidates:
                if sid in visited:
                    continue
                visited.add(sid)
                try:
                    with lease(path.parent, sid), upstream_lease(upstream_id):
                        owners.append(sid)
                        if killed or not data or not scope or not Path(scope).is_dir():
                            continue
                        task = Task.restore(json.loads(data))
                        if task.sid != sid or task.name != name or task.repo_root or task.workpool:
                            continue
                        child = Session(path, sid, scope)
                        if not child.has_restorable_turn():
                            continue
                    plan.setdefault(owner, []).append(replace(task, workspace=scope))
                except Timeout:
                    continue
                except (KeyError, OSError, SQLiteError, TypeError, ValueError):
                    logger.warning("Unable to read the interrupted task %s", name, exc_info=True)
        return plan

    def restore_detached(self, plan):
        resumed = {self.session.id, *(task.sid for tasks in plan.values() for task in tasks)}
        for owner in plan:
            if owner not in resumed:
                self.restore_children(plan, delivery_owner=owner)

    def restore_children(self, plan, *, delivery_owner=None):
        owner = delivery_owner or self.session.id
        tasks = plan.get(owner)
        if not tasks or not self.jobs:
            return

        def reserve():
            if self.peer:
                self.peer.bus.mark(*(task.sid for task in tasks))
            if not delivery_owner:
                for task in tasks:
                    self.observe(task, "running")

        def release_unclaimed(owner):
            if self.peer:
                for child in plan.get(owner, ()):
                    if self.peer.bus.release_restoration(child.sid):
                        release_unclaimed(child.sid)

        def resume(task, cancel):
            try:
                if task.eval_schema_source is not None and self.resume_eval:
                    identifier = f"{owner}/{task.name}" if delivery_owner else None
                    return self.resume_eval(task, cancel, restore_plan=plan, owner=delivery_owner, job_id=identifier)
                return self.execute(task, cancel, restore_plan=plan, delivery_owner=delivery_owner)
            finally:
                release_unclaimed(task.sid)

        try:
            self.jobs.spawn([(task.name, task.eval_label or task.assignment, task.sid, lambda cancel, task=task: resume(task, cancel))
                             for task in tasks], reserve, resuming=True, delivery_owner=delivery_owner)
        except ValueError:
            release_unclaimed(owner)
            raise

    def revive(self, target):
        if not self.jobs or self.jobs.closed:
            raise ValueError("The parent session is closing.")
        with database(self.session.path) as db:
            row = db.execute("SELECT data FROM subagent_specs WHERE child=?", (target.sid,)).fetchone()
            previous = db.execute("SELECT 1 FROM batches WHERE session=? LIMIT 1", (target.sid,)).fetchone()
        task = Task.restore(json.loads(row[0])) if row else self.restore_legacy(target)
        if task.repo_root and target.sid not in self.peer.bus.isolations:
            raise ValueError("This isolated agent's checkout has been released.")
        if not task.repo_root and task.workspace and not Path(task.workspace).is_dir():
            raise ValueError("The saved agent's workspace is unavailable.")
        identifier = f"{task.name[:18]}-reply-{uuid4().hex[:7]}"
        def run(cancel):
            if task.eval_schema_source is not None and self.resume_eval:
                return self.resume_eval(task, cancel, revived=bool(previous), job_id=identifier, owner=target.owner)
            return self.execute(task, cancel, revived=bool(previous))

        self.jobs.spawn([(identifier, f"Message follow-up to {target.name}", task.sid, run)], lambda: None)

    def restore_legacy(self, target):
        with database(self.session.path) as db:
            link = db.execute("SELECT name,result FROM subagent_links WHERE owner=? AND child=?", (target.owner, target.sid)).fetchone()
            scope = db.execute("SELECT workspace FROM sessions WHERE id=?", (target.sid,)).fetchone()
            history = [message for data, in db.execute("SELECT messages FROM batches WHERE session=? ORDER BY seq", (target.sid,))
                       for message in json.loads(data)]
            owner_history = [message for data, in db.execute("SELECT messages FROM batches WHERE session=? ORDER BY seq", (target.owner,))
                             for message in json.loads(data)]
        if not link or not scope or scope[0] != os.path.normcase(str(self.workspace)):
            raise ValueError("The saved agent's workspace is unavailable.")
        name, data = link
        result = json.loads(data) if data else {}
        agent = self.definitions.get(result.get("agent", "task").lower())
        assignment = next((plain_text(message.get("content")) for message in history if message["role"] == "user"), "")
        if not assignment or not agent or result.get("isolation"):
            raise ValueError("The saved agent has no recoverable task definition.")
        context, item = "", {}
        for message in owner_history:
            for call in message.get("tool_calls", ()):
                if call["function"]["name"] != "task":
                    continue
                try:
                    params = json.loads(call["function"]["arguments"])
                except ValueError:
                    continue
                if not isinstance(params, dict):
                    continue
                entries = params.get("tasks", [params])
                if not isinstance(entries, list):
                    continue
                matches = [entry for entry in entries if isinstance(entry, dict)
                           and (entry.get("name") == name or entry.get("task") == assignment)]
                if matches:
                    item = matches[0]
                    context = params["context"] if isinstance(params.get("context"), str) else ""
                    break
            if item:
                break
        routes, effort, _, _ = self.resolve(agent, {})
        saved_route = next((message["_rpc"]["provider"] for message in reversed(history)
                            if isinstance(message.get("_rpc", {}).get("provider"), str)), None)
        if saved_route and saved_route not in ("openai", "unknown"):
            routes = (saved_route, *[route for route in routes if route != saved_route])
        structured = "outputSchema" in item or any(call["function"]["name"] == "yield" for message in history
                                                   for call in message.get("tool_calls", ()))
        schema = item.get("outputSchema", jtd_schema(agent.output)) if structured else None
        if isinstance(schema, str):
            schema = json.loads(schema)
        if schema is not None:
            local_references(schema)
            try:
                Draft202012Validator.check_schema(schema)
            except SchemaError as error:
                raise ValueError("The saved agent has an invalid output schema.") from error
        task = Task(name, target.sid, assignment, item.get("solutionSpace") or assignment, context, agent, routes, effort,
                    structured, schema, item.get("schemaMode", "permissive"), depth=len(self.peer.bus.ancestors(target.sid)))
        with database(self.session.path) as db:
            db.execute("INSERT OR IGNORE INTO subagent_specs(child,data) VALUES (?,?)", (task.sid, pack(task.spec())))
        return task

    def execute(self, task, cancel, *, revived=False, restore_plan=None, delivery_owner=None):
        if revived:
            self.observe(task, "running")
        try:
            result, data = self.run_task(task, cancel, revived, restore_plan=restore_plan)
        except BaseException:
            self.observe(task, "failed")
            raise
        return self.complete(task, result, data, delivery_owner=delivery_owner)

    def complete(self, task, result, data, *, delivery_owner=None):
        completed = False
        try:
            with database(self.session.path) as db:
                db.execute("UPDATE subagent_links SET result = ? WHERE child = ?", (pack({**result, "data": data}), task.sid))
                if delivery_owner:
                    summary = {"jobId": task.name, "type": "task", "status": result["status"],
                               "label": task.eval_label or task.assignment, "durationMs": result["durationMs"],
                               "result": {key: value for key, value in result.items() if key != "report"}}
                    db.execute("DELETE FROM subagent_deliveries WHERE child=? AND consumed=1", (task.sid,))
                    db.execute("""INSERT INTO subagent_deliveries(owner,child,summary,report,settled_leaf)
                                  SELECT l.owner,l.child,?,?,head.leaf FROM subagent_links l
                                  LEFT JOIN session_heads head ON head.session=l.child WHERE l.owner=? AND l.child=?""",
                               (pack(summary), result["report"], delivery_owner, task.sid))
            completed = result["status"] == "completed"
            if not delivery_owner:
                self.observe(task, result["status"])
            return result
        finally:
            if self.peer:
                self.peer.bus.finish(task.sid, retry=completed)
                self.peer.bus.recover_deferred()

    def run_task(self, task, cancel, revived, staged=None, *, progress=None, pool_items=(), pool_followup=False, restore_plan=None):
        runtime = TaskRun(task, self.options, provisional=staged is not None, progress=progress,
                          pool_items=pool_items, pool_followup=pool_followup, restore_plan=restore_plan)
        acquired = False
        start = monotonic()
        sandbox, isolation, opened = None, None, False
        bus = self.peer.bus if self.peer else None
        saved = bus.isolations.get(task.sid) if bus else None
        status = "failed"
        try:
            if self.slots and not pool_items:
                while not acquired:
                    cancel.check()
                    acquired = self.slots.acquire(timeout=.1)
            cancel.check()
            workspace = Path(task.workspace) if task.workspace else self.workspace
            if saved:
                workspace = saved[0].directory
                with database(self.session.path) as db:
                    prior = db.execute("SELECT result FROM subagent_links WHERE child=?", (task.sid,)).fetchone()
                isolation = json.loads(prior[0]).get("isolation") if prior and prior[0] else None
            elif task.repo_root:
                sandbox = task_isolation.Workspace(task.repo_root, workspace, self.session.path.parent, self.session, task,
                                                   self.options.isolation_backend, task.isolation_merge or self.options.isolation_merge,
                                                   self.options.isolation_apply if task.isolation_apply is None else task.isolation_apply)
                workspace = sandbox.open(cancel)
                opened = True
            if self.options.timeout:
                with deadline(self.options.timeout, cancel) as child_cancel:
                    data, error = self.run(runtime, child_cancel, workspace, revived)
                    child_cancel.check()
            else:
                data, error = self.run(runtime, cancel, workspace, revived)
                cancel.check()
            status = "cancelled" if runtime.yield_aborted else "failed" if error else "completed"
        except CancelledError as exc:
            status, data, error = "cancelled", None, str(exc) or "Task cancelled."
        except Exception as exc:
            status, data, error = "failed", None, str(exc)
        finally:
            try:
                if sandbox:
                    try:
                        if opened and staged:
                            staged.decided.wait()
                        if opened and (not staged or staged.adopted):
                            isolation = sandbox.finish(status == "completed" and not cancel.is_cancelled())
                    finally:
                        if (bus and opened and (not staged or staged.adopted)
                                and task.sid in bus.live and not sandbox.retain):
                            with bus.lock:
                                bus.isolations[task.sid] = (sandbox, lambda: self.release_isolation(task, sandbox))
                        else:
                            try:
                                sandbox.close()
                            except OSError as cleanup_error:
                                if isolation is None:
                                    isolation = {"backend": "git-clone", "patches": []}
                                isolation.update(cleanupError=str(cleanup_error), retainedWorkspace=str(sandbox.checkout))
            finally:
                if acquired:
                    self.slots.release()
        text = data if isinstance(data, str) else pack(data) if data is not None else ""
        body = text if len(text) <= 11_500 else text[:5_000] + "\n[Output preview]"
        report = f"Task {task.name} [{task.agent.name}] {status}."
        if error:
            report += "\n" + error
        if runtime.schema_warning:
            report += "\nOutput schema warning: " + runtime.schema_warning
        if body:
            report += "\n\n" + body
        if isolation:
            report += "\n\n" + task_isolation.report(isolation)
        report += f"\n\nAgent: agent://{task.name}, history://{task.name}"
        result = {"id": task.name, "agent": task.agent.name, "sessionId": task.sid, "status": status, "error": error,
                  "requests": runtime.count, "durationMs": round((monotonic() - start) * 1000), "report": report}
        if task.file:
            result.update(taskFile=task.file, taskFileSha256=task.file_hash)
        if runtime.schema_warning:
            result["schemaWarning"] = runtime.schema_warning
        if runtime.model:
            result["model"] = runtime.model
        if pool_items:
            result.update(workpoolItemErrors=[key for key in pool_items if key in runtime.pool_errors],
                          workpoolReusable=runtime.pool_reusable, contextTokens=runtime.context_tokens,
                          contextWindow=runtime.context_window)
        if isolation:
            result["isolation"] = isolation
        if runtime.suppress_delivery:
            result["suppressDelivery"] = True
        return result, data

    def release_isolation(self, task, sandbox):
        isolation = sandbox.finish(False, release=True)
        try:
            sandbox.close()
        except OSError as error:
            isolation = isolation or {"backend": "git-clone", "patches": []}
            isolation.update(cleanupError=str(error), retainedWorkspace=str(sandbox.checkout))
        if isolation is None:
            return
        with database(self.session.path) as db:
            row = db.execute("SELECT result FROM subagent_links WHERE child=?", (task.sid,)).fetchone()
            if row and row[0]:
                result = json.loads(row[0])
                prior = result.get("isolation")
                if prior:
                    result["report"] = result["report"].replace("\n\n" + task_isolation.report(prior), "", 1)
                result["isolation"] = isolation
                result["report"] += "\n\n" + task_isolation.report(isolation)
                db.execute("UPDATE subagent_links SET result=? WHERE child=?", (pack(result), task.sid))


def read_task(session, uri, peer=None):
    scheme, path = uri.split("://", 1)
    name, *keys = path.split("/")
    target = peer.bus.target(name) if peer and name else None
    with database(session.path) as db:
        if not name:
            if peer:
                roster = peer.bus.roster()
                with peer.bus.lock:
                    return "\n".join(f"{scheme}://{item.name} [{peer.bus.status(item.sid)}]"
                                     for item in roster.values())
            return "\n".join(f"{scheme}://{item}" for item, in db.execute(
                "SELECT name FROM subagent_links WHERE owner = ? ORDER BY name", (session.id,))) or "No subagents."
        if target:
            owner = target.sid
            row = db.execute("SELECT result FROM subagent_links WHERE owner=? AND child=?", (target.owner, owner)).fetchone()
            result = row[0] if row else None
        else:
            owner = session.id
            for part in name.split("."):
                row = db.execute("SELECT child,result FROM subagent_links WHERE owner = ? AND name = ?", (owner, part)).fetchone()
                if row is None:
                    raise ValueError(f"Unknown subagent {name!r}.")
                owner, result = row
        if scheme == "agent":
            if not result:
                if target and not target.owner:
                    return "Main has no task result."
                return f"Task {name} is still running."
            saved = json.loads(result)
            value = saved.get("data")
            if not keys and value is None:
                return saved["report"]
            try:
                for key in keys:
                    value = value[int(key)] if isinstance(value, list) and key.isdecimal() else value[key]
            except (IndexError, KeyError, TypeError) as error:
                raise ValueError(f"No output at {uri}.") from error
            return value if isinstance(value, str) else pack(value)
        row = db.execute("SELECT s.workspace,h.leaf FROM sessions s JOIN session_heads h ON h.session=s.id WHERE s.id=?",
                         (owner,)).fetchone()
        if row is None:
            return f"Task {name} has not opened its session."
        lines = []
        for entry in journal_entries(db, owner, *row):
            if message := entry.get("message"):
                content = plain_text(text_and_images(message.get("content")))
                calls = [block["name"] for block in message.get("content") or ()
                         if isinstance(block, dict) and block.get("type") == "toolCall"]
                if content or calls:
                    lines.append(f"{message['role']}: {content}" + (f" [tools: {', '.join(calls)}]" if calls else ""))
        return "\n\n".join(lines) or "No messages yet."
