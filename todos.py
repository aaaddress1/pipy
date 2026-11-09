"""Phased todos, durable edits, and reconciliation at agent boundaries."""

import json
import re
import shlex
from copy import deepcopy
from html import escape
from pathlib import Path
from threading import RLock

from jsonschema import Draft202012Validator

from async_jobs import tool_result


EDIT = "user_todo_edit"
OPEN = ("in_progress", "pending")
CLOSED = ("completed", "abandoned")
MARKERS = dict(zip((*OPEN, *CLOSED, "blocked"), ("/", " ", "x", "-", "!")))
STATUSES = {**{mark: state for state, mark in MARKERS.items()},
            "": "pending", "X": "completed", ">": "in_progress", "~": "abandoned"}
OPS = ("init", "start", "done", "rm", "drop", "block", "unblock", "append", "view")
MUTATING = {"bash", "shell", "eval", "edit", "write", "ast_edit"}
STRING = {"type": "string"}
ITEMS = {"type": "array", "items": STRING}
SCHEMA = {
    "type": "object", "required": ["op"], "additionalProperties": False,
    "properties": {
        "op": {"type": "string", "enum": list(OPS)},
        "list": {"type": "array", "items": {
            "type": "object", "required": ["phase", "items"], "additionalProperties": False,
            "properties": {"phase": STRING, "items": {**ITEMS, "minItems": 1}},
        }},
        "task": STRING, "phase": STRING, "items": ITEMS, "reason": STRING,
    },
}
VALIDATOR = Draft202012Validator(SCHEMA)


def tasks(phases):
    return (task for phase in phases for task in phase["tasks"])


def valid_phases(phases):
    return isinstance(phases, list) and all(
        isinstance(phase, dict) and isinstance(phase.get("name"), str) and isinstance(phase.get("tasks"), list)
        and all(isinstance(task, dict) and isinstance(task.get("content"), str)
                and isinstance(task.get("status"), str) and task["status"] in MARKERS
                and ("blocker" not in task or isinstance(task["blocker"], str)) for task in phase["tasks"])
        for phase in phases
    )


def committed_phases(details):
    if isinstance(details, dict) and details.get("op") != "view" and valid_phases(details.get("phases")):
        return details["phases"]
    return None


def actionable(phases):
    pending = None
    for task in tasks(phases):
        if task["status"] == "in_progress":
            return task
        if pending is None and task["status"] == "pending":
            pending = task
    return pending


def normalize(phases):
    active = actionable(phases)
    for task in tasks(phases):
        if task is active:
            task["status"] = "in_progress"
        elif task["status"] == "in_progress":
            task["status"] = "pending"


def arguments(raw, has_phases):
    if isinstance(raw, dict) and "op" not in raw:
        if isinstance(raw.get("list"), list) and raw["list"]:
            raw = {**raw, "op": "init"}
        elif isinstance(raw.get("items"), list) and raw["items"]:
            if isinstance(raw.get("phase"), str) and raw["phase"]:
                raw = {**raw, "op": "append"}
            elif not has_phases:
                raw = {**raw, "op": "init"}
    if error := next(VALIDATOR.iter_errors(raw), None):
        raise ValueError(f"Invalid todo arguments: {error.message}")
    return raw


def new_tasks(items):
    return [{"content": item, "status": "pending"} for item in items]


def apply(phases, args):
    op, phase_name, target = args["op"], args.get("phase"), args.get("task")
    if op == "init":
        rows = args.get("list")
        if not rows and args.get("items"):
            rows = [{"phase": args.get("phase", "Tasks"), "items": args["items"]}]
        if rows is None:
            raise ValueError("Missing list for init operation")
        names, contents = set(), set()
        for row in rows:
            if row["phase"] in names:
                raise ValueError(f'Duplicate phase "{row["phase"]}" in init list')
            names.add(row["phase"])
            for item in row["items"]:
                if item in contents:
                    raise ValueError(f'Duplicate task "{item}" in init list')
                contents.add(item)
        return [{"name": row["phase"], "tasks": new_tasks(row["items"])} for row in rows]

    phase = next((phase for phase in phases if phase["name"] == phase_name), None)
    if op == "append":
        if not phase_name:
            raise ValueError("Missing phase name for append operation")
        if not (items := args.get("items")):
            raise ValueError("Missing items for append operation")
        contents = {task["content"] for task in tasks(phases)}
        for item in items:
            if item in contents:
                raise ValueError(f'Task "{item}" already exists')
            contents.add(item)
        if phase is None:
            phase = {"name": phase_name, "tasks": []}
            phases.append(phase)
        phase["tasks"].extend(new_tasks(items))
        return phases
    if op == "view":
        return phases

    if target:
        hit = next(((phase, task) for phase in phases for task in phase["tasks"] if task["content"] == target), None)
        if hit is None:
            if re.fullmatch(r"task-[0-9]+", target):
                raise ValueError(f'Task "{target}" not found. Pass the full task text, not an ID.')
            hint = " (todo list is empty)" if next(tasks(phases), None) is None else ""
            raise ValueError(f'Task "{target}" not found{hint}')
        phase, selected = hit
        selected_tasks = [selected]
    elif op == "start":
        raise ValueError("Missing task content")
    elif phase_name:
        if phase is None:
            raise ValueError(f'Phase "{phase_name}" not found')
        selected_tasks = list(phase["tasks"])
    elif op in ("block", "unblock"):
        raise ValueError(f"{op} requires a task or phase target")
    else:
        selected_tasks = list(tasks(phases))

    if op == "start":
        for task in tasks(phases):
            if task["status"] == "in_progress":
                task["status"] = "pending"
        selected_tasks[0]["status"] = "in_progress"
    elif op in ("done", "drop"):
        for task in selected_tasks:
            task["status"] = "completed" if op == "done" else "abandoned"
    elif op == "block":
        reason = " ".join(args.get("reason", "").split())
        for task in selected_tasks:
            if task["status"] not in CLOSED:
                task["status"] = "blocked"
                if reason:
                    task["blocker"] = reason
                else:
                    task.pop("blocker", None)
    elif op == "unblock":
        for task in selected_tasks:
            if task["status"] == "blocked":
                task["status"] = "pending"
                task.pop("blocker", None)
    elif op == "rm":
        if target:
            phase["tasks"].remove(selected_tasks[0])
        else:
            for item in ([phase] if phase is not None else phases):
                item["tasks"] = []
    return phases


def markdown(phases):
    lines = []
    for phase in phases:
        if lines:
            lines.append("")
        lines.append("# " + phase["name"])
        for task in phase["tasks"]:
            status = task["status"]
            blocker = f" <!-- blocker: {task['blocker']} -->" if status == "blocked" and task.get("blocker") else ""
            lines.append(f"- [{MARKERS[status]}] {task['content']}{blocker}")
    return "\n".join(lines or ["# Todos"]) + "\n"


def parse_markdown(text):
    phases, errors = [], []
    for number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        if heading := re.fullmatch(r"#{1,6}\s+(.+)", line):
            phases.append({"name": heading[1].strip(), "tasks": []})
            continue
        if not (item := re.fullmatch(r"[-*+]\s*\\?\[(.?)\\?\]\s+(.+)", line)):
            errors.append(f'Line {number}: unrecognized syntax "{line}"')
            continue
        mark, content = item.groups()
        if mark not in STATUSES:
            errors.append(f'Line {number}: unknown status marker "[{mark}]"')
            continue
        if not phases:
            phases.append({"name": "Todos", "tasks": []})
        task = {"content": content, "status": STATUSES[mark]}
        if task["status"] == "blocked" and (blocker := re.fullmatch(r"(.*?)\s*<!--\s*blocker:\s*(.*?)\s*-->", content)):
            task.update(content=blocker[1].strip(), blocker=blocker[2].strip())
        phases[-1]["tasks"].append(task)
    if errors:
        raise ValueError("\n".join(errors))
    normalize(phases)
    return phases


def summary(phases, *, read_only=False):
    items = list(tasks(phases))
    if not items:
        return "Todo list is empty." if read_only else "Todo list cleared."
    remaining = [(phase["name"], task) for phase in phases for task in phase["tasks"] if task["status"] in OPEN]
    if remaining:
        lines = [f"Remaining items ({len(remaining)}):"]
        lines.extend(f"  - {task['content']} [{task['status']}] ({phase})" for phase, task in remaining)
    else:
        lines = ["Remaining items: none."]
    closed = sum(task["status"] in CLOSED for task in items)
    blocked = sum(task["status"] == "blocked" for task in items)
    blockers = f", {blocked} blocked" if blocked else ""
    lines.append(f"Overall: {closed}/{len(items)} done, {len(remaining)} open{blockers}.")
    index = next((index for index, phase in enumerate(phases)
                  if any(task["status"] in OPEN for task in phase["tasks"])), len(phases) - 1)
    current = phases[index]
    done = sum(task["status"] in CLOSED for task in current["tasks"])
    ahead = any(task["status"] in CLOSED for task in tasks(phases[index + 1:]))
    explanation = " Later phases have closed items. The active pointer follows the earliest phase with open work." if ahead else ""
    lines.append(f'Active phase {index + 1}/{len(phases)} "{current["name"]}" ({done}/{len(current["tasks"])}).{explanation}')
    tags = {"in_progress": " (in progress)", "abandoned": " (dropped)", "blocked": " (blocked)"}
    for phase in phases:
        lines.append(f"  {phase['name']}:")
        for task in phase["tasks"]:
            status = task["status"]
            checkbox = "[X]" if status == "completed" else "[ ]"
            tag = tags.get(status, "")
            if status == "blocked" and task.get("blocker"):
                tag = f" (blocked: {task['blocker']})"
            lines.append(f"    - {checkbox} {task['content']}{tag}")
    return "\n".join(lines)


def notice(kind, text):
    return {"role": "developer", "content": text, "_todo_message": kind}


def goal_label(text):
    line = text.replace("\r\n", "\n").replace("\n", r"\n").replace("\r", r"\r").replace("\t", r"\t")
    return escape(re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u2028\u2029]", " ", line), quote=False)


OPTION = re.compile(r"^(?:[-*+]|\d+[.)])\s+\S")
LABEL = re.compile(r"^(?:q(?:uestion)?|ask)\s*\d*\s*[:.)-]\s*", re.I)
QUESTION = re.compile(
    r"^(?:what|which|when|where|why|how|who|whom|whose|do|does|did|can|could|would|will|should|is|are|am|may|shall)\b", re.I
)
CHOICE = re.compile(r"^(?:go|proceed|continue|stick)\s+with\b", re.I)
USER = re.compile(r"\b(?:you|your|we|our)\b", re.I)
CUE = re.compile(
    r"^(?:please\s+)?(?:confirm|reply|choose|pick|decide|advise|answer|approve)\b"
    r"|^(?:please\s+)?(?:let\s+me\s+know|(?:tell|message|ping)\s+me)\b"
    r"|^(?:if|once|when)\s+you\s+(?:confirm|approve|agree|allow|okay|ok|give)\b", re.I
)
CONNECTIVE = re.compile(
    r"^(?:and|or|so|then|also|otherwise|alternatively|separately|finally)\b,?\s*"
    r"(?=(?:please\b|let\s+me\s+know\b|(?:tell|message|ping)\s+me\b|(?:if|once|when)\s+you\b))", re.I
)
RECOMMEND = re.compile(r"^(?:i(?:'d| would)?\s+recommend|(?:my\s+)?recommendation)\b", re.I)


def prompt_line(text):
    text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text.strip())
    text = re.sub(r"^(?:>\s*)?(?:(?:[-*+]|\d+[.)])\s+)*", "", text).strip()
    label = LABEL.match(text)
    if label:
        text = text[label.end():].strip()
    wrapped = re.fullmatch(r"(\*\*\*|\*\*|\*|___|__|_)([\s\S]+)\1", text)
    return (wrapped[2].strip() if wrapped else text), bool(label)


def question(text, labelled):
    return text.endswith(("?", "？")) and (
        labelled or QUESTION.search(text) or CHOICE.search(text) or USER.search(text) or not text.isascii()
    )


def closing_ask(paragraph):
    last = paragraph.rsplit("\n", 1)[-1]
    sentences = []
    for line in reversed(paragraph.splitlines()):
        if OPTION.match(line.strip()):
            break
        pieces = re.split(r"(?<=[.!?…])\s+|(?<=[。！？])", line.strip())
        sentences = ([piece for piece in pieces if piece] + sentences)[-2:]
        if len(sentences) >= 2:
            break
    found_question = False
    for span in (last, *sentences[-1:]):
        text, label = prompt_line(span)
        if CUE.search(CONNECTIVE.sub("", text.rstrip(".!?。！？"))):
            return "cue"
        found_question |= bool(question(text, label))
    if len(sentences) == 2:
        text, label = prompt_line(sentences[0])
        found_question |= not text.isascii() and bool(question(text, label))
    return "question" if found_question else None


def awaiting_user(answer):
    if not answer or not answer.strip():
        return False
    paragraphs = re.split(r"\n\s*\n", answer.strip())
    last = paragraphs[-1].strip()
    if closing_ask(last):
        return True
    if len(paragraphs) > 1 and "\n" not in last and not OPTION.match(last) and closing_ask(paragraphs[-2]) == "cue":
        return True
    choices, recommended = 0, False
    for line in reversed(answer.splitlines()):
        if not (line := line.strip()):
            continue
        if OPTION.match(line):
            choices += 1
            continue
        text, label = prompt_line(line)
        if choices:
            return choices >= 2 and bool(question(text, label) and (recommended or label or USER.search(text) or CHOICE.search(text)))
        if RECOMMEND.search(text):
            recommended = True
            continue
        return False
    return False


def find_task(phases, query):
    query = query.casefold()
    items = list(tasks(phases))
    if exact := next((item for item in items if item["content"].casefold() == query), None):
        return exact
    matches = [item for item in items if query in item["content"].casefold()]
    if len(matches) == 1:
        return matches[0]
    opened = [item for item in matches if item["status"] in OPEN]
    return opened[0] if len(opened) == 1 else None


def find_phase(phases, query):
    query = query.casefold()
    if exact := next((item for item in phases if item["name"].casefold() == query), None):
        return exact
    for match in (lambda name: name.startswith(query), lambda name: query in name):
        matches = [item for item in phases if match(item["name"].casefold())]
        if len(matches) == 1:
            return matches[0]
    return None


def file_path(raw, workspace):
    name = raw.strip().strip('"').strip() or "TODO.md"
    if "://" in name:
        raise ValueError("Expected a filesystem path.")
    return (workspace / Path(name).expanduser()).resolve()


class Todos:
    description = (
        "Track phased work. Use init for multi-step work, identify tasks by their exact text, and mark work done "
        "alongside the next real tool call. Block work that needs an external response and unblock it when actionable."
    )
    parameters = SCHEMA
    concurrency = "exclusive"

    def __init__(self, session, *, enabled, reminders, eager, emit, mid_reminders=True):
        self.session, self.enabled, self.reminders, self.eager, self.emit = session, enabled, reminders, eager, emit
        self.mid_reminders = mid_reminders
        self.lock = RLock()
        self.sync()

    def sync(self):
        self.phases = next((phases for phases in self.session.todo_snapshots() if valid_phases(phases)), [])
        self.failures = []
        self.begin()

    def begin(self):
        self.reminded, self.mutations, self.nudges = 0, 0, 0
        self.awaiting_progress = False

    def snapshot(self):
        return deepcopy(self.phases)

    def set(self, phases):
        self.phases = deepcopy(phases)
        self.nudges = 0

    def edit(self, phases):
        if not valid_phases(phases):
            raise ValueError("Expected a list of phases with valid todo tasks.")
        with self.lock:
            self.session.record_metadata({"type": "custom", "customType": EDIT, "data": {"phases": phases}})
            self.set(phases)

    def execute(self, raw, *, nested=False):
        with self.lock:
            return self.execute_locked(raw, nested)

    def execute_locked(self, raw, nested):
        previous = self.snapshot()
        details = {"phases": previous, "storage": "session"}
        try:
            args = arguments(raw, bool(previous))
        except ValueError as error:
            return tool_result(str(error), details, True)
        op = details["op"] = args["op"]
        if op == "view":
            return tool_result(summary(previous, read_only=True), details)
        try:
            updated = apply(deepcopy(previous), args)
        except ValueError as error:
            return tool_result(f"Errors: {error}\n" + summary(previous, read_only=True), details, True)
        normalize(updated)
        before = {(phase["name"], task["content"]): task["status"] for phase in previous for task in phase["tasks"]}
        completed = [{"phase": phase["name"], "content": task["content"]}
                     for phase in updated for task in phase["tasks"] if task["status"] == "completed"
                     and before.get((phase["name"], task["content"])) not in (None, "completed")]
        details["phases"] = updated
        if completed:
            details["completedTasks"] = completed
        if nested:
            self.edit(updated)
        return tool_result(summary(updated), details)

    def tool_finished(self, call, result):
        name = call["function"]["name"]
        if name == "todo":
            self.mutations = 0
            if result.get("_todo_snapshot"):
                self.set(result["_tool_details"]["phases"])
            self.reconcile_error(call, result)
        elif name in MUTATING and not result.get("_tool_error"):
            self.mutations += 1
        self.awaiting_progress = False

    def reconcile_error(self, call, result):
        try:
            args = json.loads(call["function"]["arguments"])
        except ValueError:
            args = None
        if not isinstance(args, dict):
            args = None
        op = (result.get("_tool_details") or {}).get("op") or (args or {}).get("op")
        if result.get("_tool_error"):
            content = result["content"]
            if isinstance(content, list):
                content = next((block.get("text") for block in content if block.get("type") == "text"), "")
            reason = (content or "Todo returned an error.").split("\n", 1)[0]
            self.failures.append((args, op, reason))
            return
        if op == "view" or args is None:
            return
        exact = [failure for failure in self.failures if failure[0] == args or (
            failure[0] and failure[1] == "append" and "phase" not in failure[0] and isinstance(args.get("phase"), str)
            and failure[0] == {key: value for key, value in args.items() if key != "phase"}
        )]
        if not exact:
            match = next((failure for failure in self.failures if failure[1] in (None, op)), None)
            exact = [match] if match else []
        self.failures = [failure for failure in self.failures if failure not in exact]

    def take_errors(self):
        failures, self.failures = self.failures, []
        return [notice("todo-error-reminder", f"Todo update failed: {reason}\nCorrect the tool arguments and try again.")
                for _, _, reason in failures]

    def blocked_only(self):
        phases = self.phases
        return actionable(phases) is None and any(task["status"] == "blocked" for task in tasks(phases))

    def goal_context(self):
        phases = self.phases
        items = list(tasks(phases))
        if not self.enabled or not items:
            return ""
        closed = sum(task["status"] in CLOSED for task in items)
        lines = ["Use the persisted todos as the current plan. Reconcile stale items before substantial work.",
                 f"Overall: {closed}/{len(items)} done, {len(items) - closed} open."]
        lines.extend(f"- {goal_label(phase['name'])}\n" + "\n".join(
            f"  - [{task['status']}] {goal_label(task['content'])}" for task in phase["tasks"]
        ) for phase in phases if phase["tasks"])
        return "<todo_context>\n" + "\n".join(lines) + "\n</todo_context>"

    def prelude(self, task, batches, active):
        if not active or self.eager not in ("always", "preferred") or self.phases:
            return None
        if isinstance(task, dict) and task.get("_goal_message") == "guided-goal-interview":
            return None
        if isinstance(task, str) and task.rstrip().endswith(("?", "!")):
            return None
        if any(message["role"] == "user" and not message.get("_goal_message") for _, batch in batches for message in batch):
            return None
        return notice("eager-todo-prelude", "Create a phased todo list covering every requested deliverable before substantial work.")

    def mid_run(self, active):
        if active and self.mid_reminders and self.mutations >= 12 and self.nudges < 2 and actionable(self.phases):
            self.mutations, self.nudges = 0, self.nudges + 1
            return notice("mid-run-todo-nudge", "Reconcile todo progress with completed work and select the next actionable item.")
        return None

    def stop(self, answer, active):
        if not active or self.awaiting_progress or self.reminded >= self.reminders or awaiting_user(answer):
            return None
        incomplete = [{"phase": phase["name"], **task} for phase in self.phases
                      for task in phase["tasks"] if task["status"] in OPEN]
        if not incomplete:
            return None
        self.reminded += 1
        self.mutations, self.awaiting_progress = 0, True
        self.emit("todo_reminder", todos=[{key: task[key] for key in ("content", "status")} for task in incomplete],
                  attempt=self.reminded, maxAttempts=self.reminders)
        lines = "\n".join(f"- {item['phase']}: {item['content']}" for item in incomplete)
        return notice("todo-reminder", f"You stopped with unfinished todos. Continue or mark verified items complete.\n{lines}")

    def command(self, command, workspace, *, editor=None):
        args = command.strip().split(maxsplit=1)
        if not args or args[0] in ("view", "copy"):
            return markdown(self.phases).rstrip() if self.phases else "No todos."
        op, rest = args[0].lower(), args[1] if len(args) == 2 else ""
        if op in ("help", "?"):
            return "/todo [view|copy|edit|export|import|append|start|done|drop|rm|block|unblock]"
        if op == "edit":
            if not editor:
                raise ValueError("/todo edit requires an interactive editor. Use /todo export and /todo import.")
            if (updated := editor(markdown(self.phases))) is not None:
                self.edit(parse_markdown(updated))
            return summary(self.phases, read_only=True)
        if op in ("import", "export"):
            path = file_path(rest, workspace)
            if op == "import":
                self.edit(parse_markdown(path.read_text(encoding="utf-8-sig")))
                return f"Imported todos from {path}"
            if not self.phases:
                return "No todos to export."
            path.write_text(markdown(self.phases), encoding="utf-8")
            return f"Wrote todos to {path}"
        if op == "rm" and not rest:
            self.edit([])
            return "Cleared all todos."
        if op == "append":
            tokens = shlex.split(rest)
            if not tokens:
                raise ValueError("Usage: /todo append [phase] task")
            if len(tokens) == 1:
                phase = self.phases[-1]["name"] if self.phases else "Todos"
            else:
                name = tokens.pop(0)
                match = find_phase(self.phases, name)
                phase = match["name"] if match else " ".join(word[:1].upper() + word[1:] for word in name.split())
            content = " ".join(tokens).strip()
            if not content:
                raise ValueError("Task text is required.")
            params = {"op": "append", "phase": phase, "items": [content[:1].upper() + content[1:]]}
        elif op in ("start", "done", "drop", "rm", "block", "unblock"):
            params = {"op": op}
            if rest:
                target = " ".join(shlex.split(rest))
                task = find_task(self.phases, target) if target else None
                if task:
                    params["task"] = task["content"]
                elif op != "start" and target and (phase := find_phase(self.phases, target)):
                    params["phase"] = phase["name"]
                else:
                    raise ValueError(f'No task or phase matched "{target}".')
        else:
            raise ValueError("Unknown /todo subcommand. Use /todo help.")
        result = self.execute(params, nested=True)
        if result.is_error:
            raise ValueError(result.content[0]["text"])
        return result.content[0]["text"]
