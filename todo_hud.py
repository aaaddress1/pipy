"""Anchored todo progress and branch-bound visibility for the terminal."""

import json
import logging
import re
from functools import lru_cache
from sqlite3 import Error as SQLiteError
from threading import Timer

from prompt_toolkit.utils import get_cwidth

from todos import CLOSED, OPEN, actionable, tasks


STATE = "todo_hud_state"
COMPACT_ROWS = 18
TASK_CAP = 5
FOLLOWING_PHASE_CAP = 4
MARKERS = {
    "pending": ("dim", "□"),
    "in_progress": ("fg:ansicyan", "◉"),
    "completed": ("fg:ansigreen", "✓"),
    "abandoned": ("dim", "×"),
    "blocked": ("fg:ansiyellow", "!"),
}
ESCAPES = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[ -/]*[@-Z\\-_])")
CONTROLS = re.compile(r"[\x00-\x1f\x7f-\x9f\u2028\u2029]")
ROMAN = ((1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"),
         (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"))
logger = logging.getLogger(__name__)


def fingerprint(phases):
    canonical = [{"name": phase["name"], "tasks": [
        {key: task[key] for key in ("content", "status", "blocker") if key in task}
        for task in phase["tasks"]
    ]} for phase in phases]
    return json.dumps(canonical, ensure_ascii=False, separators=(",", ":"))


def display(text):
    return " ".join(CONTROLS.sub(" ", ESCAPES.sub("", text)).split())


def truncate(text, width):
    if get_cwidth(text) <= width:
        return text
    if width <= 1:
        return "…" if width else ""
    used = 0
    for index, char in enumerate(text):
        used += get_cwidth(char)
        if used >= width:
            return text[:index] + "…"
    return text


def roman(index):
    numeral = []
    for value, symbol in ROMAN:
        count, index = divmod(index, value)
        numeral.append(symbol * count)
    return "".join(numeral)


@lru_cache(maxsize=1024)
def normalized(text):
    return " ".join("".join(char if char.isalnum() else " " for char in text.lower()).split())


def matches(text, descriptions):
    target = normalized(text)
    return bool(target) and any(
        target == candidate or len(target) >= 6 and target in candidate or len(candidate) >= 6 and candidate in target
        for candidate in descriptions if candidate
    )


def more(count, noun):
    return f"… {count} more {noun}{'s' if count != 1 else ''}"


def select_tasks(items, is_matched, cap=TASK_CAP):
    opened = [task for task in items if task["status"] not in CLOSED]
    closed = [task for task in items if task["status"] in CLOSED]
    base = opened or items
    lead = closed[-1:] if opened else []
    if len(base) <= cap + 1:
        return lead + base, ""
    active = [index for index, task in enumerate(base)
              if task["status"] == "in_progress" or task["status"] == "pending" and is_matched(task)]
    if len(active) > cap:
        return lead + [base[index] for index in active[:cap]], more(len(active) - cap, "active todo")
    active_indices = set(active)
    start = active[0] if active else 0
    fill = [index for index in range(start, len(base)) if index not in active_indices][:cap - len(active)]
    selected = active + fill
    return lead + [base[index] for index in selected], more(len(base) - len(selected), "todo")


def render_task(task, active):
    style, marker = MARKERS[task["status"]]
    if active and task["status"] == "pending":
        style, marker = MARKERS["in_progress"]
    label = display(task["content"])
    if task["status"] == "blocked" and task.get("blocker"):
        label += " · " + display(task["blocker"])
    return style, f"{marker} {label}"


def rows(phases, *, expanded, descriptions, width):
    phases = [phase for phase in phases if phase["tasks"]]
    if not phases:
        return []
    active = next((index for index, phase in enumerate(phases)
                   if any(task["status"] in OPEN for task in phase["tasks"])), len(phases) - 1)
    start, end = (0, len(phases)) if expanded else (active, min(len(phases), active + 1 + FOLLOWING_PHASE_CAP))
    all_tasks = list(tasks(phases))
    done = sum(task["status"] in CLOSED for task in all_tasks)
    result = [("bold fg:ansicyan", f"TODO · {done}/{len(all_tasks)}")]
    for index in range(start, end):
        phase = phases[index]
        items = phase["tasks"]
        name = display(phase["name"])
        label = f"{roman(index + 1)}. {name}" if len(phases) > 1 else name
        closed = sum(task["status"] in CLOSED for task in items)
        result.append(("fg:ansicyan" if index == active else "dim", f" ├─ {label} · {closed}/{len(items)}"))
        if expanded or index == active:
            visible, overflow = (items, "") if expanded else select_tasks(
                items, lambda task: matches(task["content"], descriptions)
            )
            for task in visible:
                style, text = render_task(task, matches(task["content"], descriptions))
                result.append((style, f" │  {text}"))
            if overflow:
                result.append(("dim", f" │  {overflow}"))
    if end < len(phases):
        result.append(("dim", f" ├─ {more(len(phases) - end, 'stage')}"))
    rail = len(result) - 1
    tail = "└─────"
    filled = int((rail + len(tail)) * done / len(all_tasks) + .5)
    if done:
        filled = max(1, filled)
    if done < len(all_tasks):
        filled = min(rail + len(tail) - 1, filled)
    fragments = [(result[0][0], truncate(result[0][1], width) + "\n")]
    for index, (style, text) in enumerate(result[1:]):
        text = truncate(text, width)
        fragments.extend((("fg:ansicyan" if index < filled else "dim", text[:4]), (style, text[4:] + "\n")))
    bend = max(0, filled - rail)
    fragments.extend((("fg:ansicyan", " " + tail[:bend]), ("dim", tail[bend:] + "\n")))
    return fragments


def compact(phases, width, descriptions=()):
    items = list(tasks(phases))
    if not items:
        return ""
    closed = sum(task["status"] in CLOSED for task in items)
    task = actionable(phases)
    if task:
        label = render_task(task, matches(task["content"], descriptions))[1]
    elif blocked := sum(task["status"] == "blocked" for task in items):
        label = f"! {blocked} blocked"
    else:
        label = "✓ done"
    return truncate(f"TODO {closed}/{len(items)} · {label}", width)


class TodoHUD:
    def __init__(self, todo, delay, repaint):
        self.todo, self.delay, self.repaint = todo, delay, repaint
        self.expanded = self.hidden = self.closed = False
        self.timer = None
        self.running = {}
        self.ancestry_epoch, self.ancestry = None, set()
        with todo.lock:
            todo.changed = self.sync
            self.sync()

    def cancel(self):
        if self.timer:
            self.timer.cancel()
            self.timer = None

    def close(self):
        with self.todo.lock:
            self.closed = True
            self.cancel()
            self.todo.changed = None

    def sync(self):
        with self.todo.lock:
            self.cancel()
            entry, phases = self.todo.entry, self.todo.phases
            state = self.todo.session.todo_hud_visibility(entry, fingerprint(phases)) if entry else None
            self.hidden = state == "dismissed"
            items = list(tasks(phases))
            if not self.closed and not state and self.delay >= 0 and items and all(task["status"] in CLOSED for task in items):
                self.schedule(phases)
        self.repaint()

    def schedule(self, phases):
        def dismiss():
            with self.todo.lock:
                if self.closed or self.timer is not timer or self.todo.phases is not phases:
                    return
                self.timer = None
                try:
                    self.persist("dismissed")
                except (OSError, SQLiteError):
                    logger.exception("Could not save todo HUD dismissal")
                    return
                self.hidden = True
            self.repaint()

        timer = self.timer = Timer(self.delay, dismiss)
        timer.daemon = True
        timer.start()

    def persist(self, state):
        if entry := self.todo.identify():
            self.todo.session.record_metadata({"type": "custom", "customType": STATE, "data": {
                "sourceEntryId": entry, "fingerprint": fingerprint(self.todo.phases), "visibility": state,
            }})

    def expand(self, expanded):
        with self.todo.lock:
            if expanded:
                self.persist("revealed")
                self.cancel()
                self.hidden = False
            self.expanded = expanded
        self.repaint()

    def observe(self, task, status):
        with self.todo.lock:
            if self.closed:
                return
            description = normalized(task.assignment.strip() or task.name)
            if status == "running":
                session = self.todo.session
                self.running[task.sid] = (description, session.leaf, session.navigation_epoch)
            else:
                observed = self.running.pop(task.sid, None)
                if status == "completed" and (observed is None or self.relevant(observed)):
                    phases = self.todo.snapshot()
                    changed = False
                    for item in tasks(phases):
                        if item["status"] not in CLOSED and matches(item["content"], (description,)):
                            item["status"] = "completed"
                            item.pop("blocker", None)
                            changed = True
                    if changed:
                        self.todo.edit(phases)
        self.repaint()

    def relevant(self, observed):
        _, leaf, epoch = observed
        session = self.todo.session
        if epoch == session.navigation_epoch:
            return True
        if self.ancestry_epoch != session.navigation_epoch:
            self.ancestry_epoch = session.navigation_epoch
            self.ancestry = session.ancestor_ids()
        return leaf is not None and leaf in self.ancestry

    def descriptions(self):
        return tuple(observed[0] for observed in self.running.values() if self.relevant(observed))

    def full(self, width):
        with self.todo.lock:
            return [] if self.hidden else rows(
                self.todo.phases, expanded=self.expanded, descriptions=self.descriptions(), width=width,
            )

    def short(self, width):
        with self.todo.lock:
            return "" if self.hidden else compact(self.todo.phases, width, self.descriptions())
