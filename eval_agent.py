"""Eval subagent handles backed by the ordinary child session and job manager."""

import json
import os
import re
from dataclasses import replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from runtime import database
from subagents import Task, jtd_schema, local_references
import task_isolation


NAME = "__agent__"
MISSING = object()
ARTIFACT_PREVIEW = re.compile(r"agent://([A-Za-z0-9_-]+):raw:1-1")
UNSAFE_ID = re.compile(r"[^A-Za-z0-9_-]+")
NOTIFICATION = re.compile(r"</?system-notification>")


def output_schema(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            raise ValueError("Invalid output schema: expected a JSON schema.") from None
    value = jtd_schema(value)
    if value is False:
        raise ValueError("Invalid output schema: a false schema rejects all results.")
    if value is not None:
        local_references(value)
        try:
            Draft202012Validator.check_schema(value)
        except SchemaError as error:
            raise ValueError(f"Invalid output schema: {error.message}") from error
    return value


def fallback_output(text, schema):
    if isinstance(text, str):
        source = text.strip()
        if fence := re.fullmatch(r"\x60{3}(?:json)?\s*\n(.*?)\n\x60{3}", source, re.S | re.I):
            source = fence[1].strip()
        try:
            value = json.loads(source)
        except ValueError:
            value = MISSING
        if isinstance(value, dict) and "data" in value:
            value = value["data"]
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                pass
        if value is not MISSING and (schema is None or Draft202012Validator(schema).is_valid(value)):
            return value, None
    if schema is None and text:
        return text, None
    return text, "Subagent exited without calling the yield tool after 3 reminders."


def clean_error(message):
    return NOTIFICATION.sub("", message).strip()


def recovery_hint(isolation):
    paths = [entry["patchPath"] for entry in isolation.get("patches", ()) if entry.get("patchPath")]
    if workspace := isolation.get("retainedWorkspace"):
        paths.append(workspace)
    return "\nRecovery artifacts:\n" + "\n".join(paths) if paths else ""


class AgentJobHandle:
    def __init__(self, jobs, job):
        self.jobs, self.job = jobs, job

    @property
    def agent_id(self):
        return self.job.agent_id

    def settled(self):
        with self.jobs.condition:
            return self.job.status != "running"

    def join(self):
        self.job.worker.join()

    def cancel(self):
        with self.jobs.condition:
            if self.job.status != "running" or self.job.cancel.is_cancelled():
                return False
            self.job.cancel.cancel()
            return True

    def progress(self):
        with self.jobs.condition:
            return (self.job.details or {}).get("progress")

    def snapshot(self, *, attempts=False, consume=False):
        with self.jobs.condition:
            status = self.job.status
            data = {"kind": "agent", "id": self.job.id, "status": status}
            if status == "completed":
                result = (self.job.details or {}).get("evalResult", {})
                data["text"] = result.get("text", self.job.text)
                if "data" in result:
                    data["data"] = result["data"]
            elif status != "running":
                data["error"] = (self.job.details or {}).get("error") or self.job.text or f"Agent {status}."
            if consume and status != "running":
                self.jobs.consume(self.job)
            return data


class AgentBridge:
    def __init__(self, tasks, resolve, *, disabled, unavailable, isolation, schema=None, schema_mode="permissive"):
        self.tasks, self.resolve_route, self.disabled = tasks, resolve, disabled
        self.unavailable, self.schema, self.schema_mode = unavailable, schema, schema_mode
        self.isolation = isolation
        self.artifacts = tasks.session.path.parent / "artifacts" / tasks.session.id if tasks else None
        if tasks:
            tasks.resume_eval = self.run

    def parse(self, args):
        if not isinstance(args, dict) or not isinstance(args.get("prompt"), str) or not args["prompt"]:
            raise ValueError("agent() requires a nonempty prompt.")
        if any(key in args and not isinstance(args[key], str) for key in ("agent", "label", "model")):
            raise ValueError("agent(), label and model values must be strings.")
        if "agent" in args and not args["agent"]:
            raise ValueError("agent() requires a nonempty agent name.")
        if any(key in args and type(args[key]) is not bool for key in ("isolated", "apply", "merge")):
            raise ValueError("agent() isolation controls must be boolean.")
        if args.get("schemaMode", self.schema_mode) not in ("strict", "permissive"):
            raise ValueError("agent() schemaMode must be strict or permissive.")
        if "tools" in args and (not isinstance(args["tools"], list) or any(not isinstance(tool, str) for tool in args["tools"])):
            raise ValueError("agent() tools must be a list of tool names.")

    def prepare(self, args):
        tasks = self.tasks
        default = "task" if "*" in tasks.allowed else tasks.allowed[0]
        name = (args.get("agent", "").strip() or default).lower()
        if "*" not in tasks.allowed and name not in tasks.allowed:
            raise ValueError(f"Cannot spawn {name!r}. Allowed: {', '.join(tasks.allowed)}.")
        agent = tasks.definitions.get(name)
        if agent is None:
            raise ValueError(f"Unknown agent {name!r}. Available: {', '.join(tasks.definitions)}.")
        if name in self.disabled() or name == os.getenv("PI_BLOCKED_AGENT", "").lower():
            raise ValueError(f"Agent {name!r} is disabled for this session.")
        routes, effort, selector, parent = self.resolve_route(agent, args)
        mode = args.get("schemaMode", self.schema_mode)
        raw_schema, source = (args["schema"], "caller") if "schema" in args else (
            (agent.output, "agent") if agent.output is not None else
            (self.schema, "session") if self.schema is not None else (None, "none")
        )
        try:
            schema = output_schema(raw_schema)
        except ValueError:
            if source == "caller" or mode == "strict":
                raise
            schema = None
        root, merge = None, None
        if args.get("isolated"):
            enabled, merge = self.isolation()
            if not enabled:
                raise ValueError("Subagent isolated execution requires task.isolation.enabled; it is currently false.")
            if merge not in ("patch", "branch"):
                raise ValueError("task.isolation.merge must be patch or branch.")
            try:
                root = task_isolation.repository(tasks.workspace)
            except (OSError, task_isolation.GitError) as error:
                raise ValueError(f"Isolated subagent execution could not be prepared: {error}") from error
        label = args.get("label", "").strip()
        base = UNSAFE_ID.sub("", label)[:48] or "Task" + uuid4().hex[:10]
        if base.lower().startswith("bg_"):
            base = "Agent-" + base[:42]
        with database(tasks.session.path) as db:
            taken = {key.lower() for key, in db.execute("SELECT name FROM subagent_links WHERE owner = ?", (tasks.session.id,))}
        if self.artifacts.is_dir():
            taken.update(path.stem.lower() for path in self.artifacts.glob("*.md"))
        taken.update(("__advisor", "main", "all"))
        identifier, suffix = base, 2
        while identifier.lower() in taken:
            identifier = f"{base}-{suffix}"
            suffix += 1
        return Task(identifier, str(uuid4()), args["prompt"].strip(), "", "", agent, routes, effort, True, schema, mode,
                    repo_root=root, depth=tasks.depth + 1, workspace=str(tasks.workspace), eval_schema_source=source,
                    eval_label=label or None, isolation_apply=args.get("apply", True),
                    isolation_merge="patch" if args.get("merge") is False else merge, per_call_model=selector, parent_model=parent)

    def custom_tools(self, names, cancellation):
        if not names:
            return ()
        if not self.tasks or not self.tasks.eval_tools:
            raise ValueError("Eval-defined tools are unavailable in this session.")
        return self.tasks.eval_tools.describe(names, cancellation)

    def start(self, args, cancellation):
        self.parse(args)
        if not self.tasks:
            raise ValueError(self.unavailable)
        tasks = self.tasks
        custom = self.custom_tools(args.get("tools"), cancellation)
        with tasks.dispatch_lock:
            cancellation.check()
            task = self.prepare(args)
            if custom:
                task = replace(task, eval_tools=custom, eval_tool_owner=tasks.eval_tools.id)
            cancellation.check()
            if not tasks.jobs or not tasks.background:
                raise ValueError("agent() needs the session's async job manager; unavailable here.")
            tasks.jobs.spawn(
                [(task.name, task.eval_label or task.name, task.sid, lambda cancel: self.run(task, cancel))],
                lambda: tasks.reserve([task]),
            )
        return {"id": task.name, "agent": task.agent.name}

    def run(self, task, cancellation, *, revived=False, job_id=None, owner=None, restore_plan=None):
        tasks = self.tasks

        def progress(runtime):
            details = {"id": task.name, "agent": task.agent.name, "status": "running", "requests": runtime.count,
                       "taskPreview": task.assignment.split("\n", 1)[0][:120]}
            if runtime.model:
                details["model"] = runtime.model
            tasks.jobs.progress(job_id or task.job_id, {"progress": details}, f"Running agent {task.name}...")

        result, value = tasks.run_task(task, cancellation, revived, progress=progress, restore_plan=restore_plan)
        return self.finish(task, result, value, owner, delivery_owner=owner if restore_plan is not None else None)

    def finish(self, task, result, value, owner=None, *, delivery_owner=None):
        error = result.get("error")
        isolation = result.get("isolation", {})
        if result["status"] == "completed" and isolation and task.isolation_apply:
            conflicts = [patch for patch in isolation["patches"] if patch.get("status") == "conflict"]
            if isolation.get("error") or conflicts:
                error = f"agent() isolated apply failed for {task.name}: {task_isolation.report(isolation)}"
                result.update(status="failed", error=error)
        if result["status"] != "completed":
            error = clean_error(error or f"agent() subagent {task.agent.name!r} failed.") + recovery_hint(isolation)
            result.update(error=error, report=error)
        structured = task.eval_schema_source != "none"
        text = json.dumps(value, indent=2, ensure_ascii=False) if structured or value is not None and not isinstance(value, str) else value or ""
        if not structured and isolation and result["status"] == "completed":
            text += "\n\n" + task_isolation.report(isolation)
        if result["status"] == "completed":
            result["evalResult"] = {"text": text, **({"data": value} if structured else {})}
            result["report"] = text
        try:
            artifacts = self.artifacts.parent / owner if owner else self.artifacts
            self.save(artifacts, task.name, text if result["status"] == "completed" else result["report"])
        except OSError as error:
            report = f"Could not save the agent output: {error}"
            result.update(status="failed", error=report, report=report)
        return self.tasks.complete(task, result, value, delivery_owner=delivery_owner)

    @staticmethod
    def save(artifacts, name, text):
        artifacts.mkdir(parents=True, exist_ok=True)
        target = artifacts / f"{name}.md"
        with NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=artifacts, delete=False) as file:
            path = Path(file.name)
            try:
                file.write(text)
            except BaseException:
                file.close()
                path.unlink(missing_ok=True)
                raise
        try:
            path.replace(target)
        finally:
            path.unlink(missing_ok=True)

    def read_preview(self, resource):
        if self.artifacts and (match := ARTIFACT_PREVIEW.fullmatch(resource or "")):
            path = self.artifacts / f"{match[1]}.md"
            if path.is_file():
                with path.open(encoding="utf-8") as file:
                    return {"text": file.readline(120).rstrip("\r\n"), "details": {"resolvedPath": str(path)}}
            return {"text": "", "details": {}}
        return None

    def resolve(self, identifier):
        jobs = self.tasks.jobs if self.tasks else None
        if jobs:
            with jobs.condition:
                jobs.prune()
                job = jobs.jobs.get(identifier)
                if job and job.kind == "task" and job.agent_id:
                    return AgentJobHandle(jobs, job)
        raise ValueError(f"Unknown agent handle {identifier!r}; result expired. Read agent://{identifier}.")
