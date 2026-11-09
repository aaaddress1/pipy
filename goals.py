"""Durable goal state, usage accounting and continuation decisions."""

import json
from dataclasses import dataclass
from functools import partial
from hashlib import sha256
from html import escape
from threading import RLock
from time import monotonic, time
from uuid import uuid4

from async_jobs import tool_result


ACCOUNTED = ("active", "budget-limited")
GUIDED = "guided-goal-interview"
SCHEMA = {
    "type": "object",
    "properties": {
        "op": {"enum": ["create", "get", "complete", "resume", "drop"]},
        "objective": {"type": "string"},
        "token_budget": {"type": "integer", "minimum": 1},
    },
    "required": ["op"],
    "additionalProperties": False,
}


def timestamp():
    return int(time() * 1000)


def goal_message(kind, text):
    return {"role": "developer" if kind == GUIDED else "user", "content": text, "_goal_message": kind}


def budget_value(value):
    if value is not None and (type(value) is not int or not 0 < value < 2**53):
        raise ValueError("Goal token budget must be a positive integer.")
    return value


def remaining(goal):
    if goal and "tokenBudget" in goal:
        return max(0, goal["tokenBudget"] - goal["tokensUsed"])
    return None


def describe(goal):
    if not goal:
        return "No active goal."
    text = f"Goal: {goal['objective']}\nStatus: {goal['status']}\nTokens: {goal['tokensUsed']} used"
    if (left := remaining(goal)) is not None:
        text += f" / {goal['tokenBudget']} budget\nRemaining tokens: {left}"
    return text


def valid_goal(value):
    return (isinstance(value, dict)
            and isinstance(value.get("id"), str) and isinstance(value.get("objective"), str)
            and value.get("status") in (*ACCOUNTED, "paused", "complete")
            and all(type(value.get(key)) is int and value[key] >= 0
                    for key in ("tokensUsed", "timeUsedSeconds", "createdAt", "updatedAt"))
            and ("tokenBudget" not in value or type(value["tokenBudget"]) is int and value["tokenBudget"] > 0))


@dataclass(frozen=True)
class GoalSteer:
    runtime: object
    revision: int
    objective: str | None = None
    budget_revision: int | None = None

    def current(self):
        runtime = self.runtime
        with runtime.lock:
            if runtime.revision != self.revision or not runtime.goal:
                return False
            if self.budget_revision is not None:
                return runtime.budget_revision == self.budget_revision and runtime.goal["status"] == "budget-limited"
            return runtime.active

    def messages(self):
        runtime = self.runtime
        with runtime.lock:
            if not self.current():
                return []
            if self.budget_revision is not None:
                return [runtime.budget_notice()]
            context = runtime.context()
            messages = [context] if context else []
            if self.objective is not None:
                messages.append({"role": "user", "content": self.objective})
            return messages


def discard_stale_goal_steers(queue):
    keep = [work for work in queue if not isinstance(work.content, GoalSteer) or work.content.current()]
    if len(keep) != len(queue):
        queue.clear()
        queue.extend(keep)


class Goals:
    def __init__(self, session, modes, emit, *, todo_context=None):
        self.session, self.modes, self.emit = session, modes, emit
        self.todo_context = todo_context
        self.lock = RLock()
        self.goal = None
        self.running = False
        self.pending = 0
        self.since = None
        self.activity = None
        self.activity_seen = False
        self.previous = None
        self.stalled = False
        self.interview = self.attached = False
        self.revision = self.budget_revision = 0
        self.context_revision = None

    def restore(self, *, preserve=False):
        with self.lock:
            record = self.session.goal_record()
            data = record.get("data")
            value = data.get("goal") if isinstance(data, dict) and record.get("mode") in ("goal", "goal_paused") else None
            self.goal = value if valid_goal(value) else None
            self.revision += 1
            self.since = monotonic() if self.accounting else None
            self.previous, self.stalled = None, False
            self.interview = False
            if self.goal and self.goal["status"] == "complete":
                self.finish_completion()
            elif not preserve and self.active:
                self.pause()
            self.attached = bool(self.goal)

    @property
    def active(self):
        return bool(self.goal and self.goal["status"] == "active")

    @property
    def accounting(self):
        return bool(self.goal and self.goal["status"] in ACCOUNTED)

    @property
    def tool_visible(self):
        return self.attached or self.interview

    def snapshot(self):
        with self.lock:
            if not self.goal:
                return {"goal": None, "state": None}
            goal = self.goal.copy()
            state = {"enabled": self.accounting, "mode": "active", "goal": goal}
            if goal["status"] == "complete":
                state.update(mode="exiting", reason="completed")
            return {"goal": goal, "state": state}

    def persist(self):
        if not self.goal:
            self.session.record_metadata({"type": "mode_change", "mode": "none"})
        else:
            mode = "goal_paused" if self.goal["status"] == "paused" else "goal"
            self.session.record_metadata({"type": "mode_change", "mode": mode, "data": {"goal": self.goal}})
        self.emit(**self.snapshot())

    def begin(self, task):
        with self.lock:
            if isinstance(task, dict) and task.get("_goal_message") == GUIDED:
                if self.goal:
                    raise ValueError("Resume or drop the existing goal before starting a guided goal.")
                self.interview = True
            self.running = True
            self.pending = 0
            self.context_revision = None
            continued = isinstance(task, dict) and task.get("_goal_message") == "goal-continuation"
            self.activity = sha256() if continued else None
            self.activity_seen = False

    def observe(self, usage):
        with self.lock:
            if self.running and self.accounting:
                self.pending += sum(max(0, usage.get(key, 0)) for key in ("input", "output", "cacheWrite"))

    def flush(self):
        if not self.accounting:
            return False
        seconds = int(monotonic() - self.since) if self.since is not None else 0
        if not (self.pending or seconds):
            return False
        new_tokens = int(self.pending)
        self.goal["tokensUsed"] += new_tokens
        self.goal["timeUsedSeconds"] += seconds
        self.goal["updatedAt"] = timestamp()
        self.pending = 0
        if seconds:
            self.since += seconds
        limited = self.active and remaining(self.goal) == 0
        if limited:
            self.goal["status"] = "budget-limited"
        if new_tokens or limited:
            self.persist()
        else:
            self.emit(**self.snapshot())
        return limited

    def tools(self, batch):
        with self.lock:
            if not (self.goal or self.interview):
                return None
            calls = batch[0]["tool_calls"]
            self.activity_seen = True
            if self.activity is not None:
                for call, result in zip(calls, batch[1:]):
                    function = call["function"]
                    try:
                        args = json.loads(function["arguments"])
                    except ValueError:
                        args = function["arguments"]
                    entry = (function["name"], args, result["content"], bool(result.get("_tool_error")))
                    self.activity.update(json.dumps(entry, sort_keys=True, ensure_ascii=True).encode() + b"\n")
            if self.flush() and any(call["function"]["name"] != "goal" for call in calls):
                return GoalSteer(self, self.revision, budget_revision=self.budget_revision)
        return None

    def finish_completion(self):
        goal = self.goal
        self.session.record_metadata(
            {"type": "mode_change", "mode": "none"},
            {"type": "custom", "customType": "goal-completed",
             "data": {key: goal[key] for key in ("objective", "tokensUsed", "tokenBudget", "timeUsedSeconds") if key in goal}},
        )
        self.goal = None
        self.attached = False

    def rearm(self):
        self.previous, self.stalled = None, False
        self.activity = None
        self.activity_seen = False

    def end(self, *, interrupted=False):
        with self.lock:
            self.flush()
            self.running = False
            if interrupted and self.active:
                self.pause()
            if self.goal and self.goal["status"] == "complete":
                self.finish_completion()
            if self.activity_seen or interrupted:
                self.interview = False
            if self.activity is not None:
                digest = self.activity.digest()
                self.stalled = not self.activity_seen or digest == self.previous
                self.previous = digest
            else:
                self.previous, self.stalled = None, False
            self.activity = None

    def budget_limited(self):
        with self.lock:
            self.flush()
            return bool(self.goal and self.goal["status"] == "budget-limited")

    def create(self, objective, token_budget=None, *, replace=False):
        if not isinstance(objective, str) or not objective.strip():
            raise ValueError("Goal objective is required.")
        budget = budget_value(token_budget)
        if replace and not self.accounting:
            raise ValueError("No active goal to replace.")
        if self.goal and self.goal["status"] != "complete" and not replace:
            raise ValueError("This session already has a goal. Resume or drop it first.")
        self.flush()
        if self.goal and self.goal["status"] == "complete":
            self.finish_completion()
        now = timestamp()
        self.goal = {"id": str(uuid4()), "objective": objective.strip(), "status": "active", "tokensUsed": 0,
                     "timeUsedSeconds": 0, "createdAt": now, "updatedAt": now}
        if budget is not None:
            self.goal["tokenBudget"] = budget
        self.since, self.pending = monotonic(), 0
        self.attached, self.interview = True, False
        self.revision += 1
        self.rearm()
        self.persist()

    def pause(self):
        if not self.accounting:
            return
        self.flush()
        self.goal.update(status="paused", updatedAt=timestamp())
        self.since = None
        self.attached = False
        self.revision += 1
        self.persist()

    def resume(self, *, paused_only=False):
        if not self.goal or self.goal["status"] == "complete":
            raise ValueError("No paused goal to resume.")
        if paused_only and self.goal["status"] != "paused":
            return
        self.goal.update(status="active", updatedAt=timestamp())
        self.since, self.pending = monotonic(), 0
        self.attached = True
        self.revision += 1
        self.rearm()
        self.persist()

    def change_budget(self, value):
        value = budget_value(value)
        if not self.accounting:
            raise ValueError("No active goal.")
        self.flush()
        previous = self.goal["status"]
        if value is None:
            self.goal.pop("tokenBudget", None)
        else:
            self.goal["tokenBudget"] = value
        self.goal.update(status="budget-limited" if remaining(self.goal) == 0 else "active", updatedAt=timestamp())
        if self.goal["status"] != previous:
            self.budget_revision += 1
        self.rearm()
        self.persist()

    def drop(self):
        if not self.goal:
            return None
        self.flush()
        dropped = {**self.goal, "status": "dropped", "updatedAt": timestamp()}
        self.session.record_metadata({"type": "mode_change", "mode": "none"})
        self.goal, self.since = None, None
        self.attached = False
        self.revision += 1
        self.emit(goal=dropped, state={"enabled": False, "mode": "active", "goal": dropped})
        return dropped

    def command(self, op, objective=None, token_budget=None):
        with self.lock:
            if op == "create":
                self.create(objective, token_budget)
            elif op == "set":
                self.create(objective, token_budget, replace=self.accounting)
            elif op == "pause":
                self.pause()
            elif op == "resume":
                self.resume(paused_only=True)
            elif op == "drop":
                self.drop()
            elif op == "budget":
                self.change_budget(token_budget)
            elif op != "get":
                raise ValueError(f"Unknown goal operation {op!r}.")
            return self.snapshot()

    def control(self, op, objective, token_budget, *, live=False, visible_objective=False):
        with self.lock:
            revision, was_active = self.revision, self.active
            data = self.command(op, objective, token_budget)
            steer = None
            if live and self.active and self.revision != revision:
                text = self.goal["objective"] if visible_objective else None
                steer = GoalSteer(self, self.revision, objective=text)
            elif live and op == "budget" and was_active and self.goal["status"] == "budget-limited":
                steer = GoalSteer(self, self.revision, budget_revision=self.budget_revision)
            return data, steer

    def advertised_tool(self):
        with self.lock:
            if self.tool_visible:
                return partial(self.tool, revision=self.revision)
        return None

    def tool(self, args, *, revision=None):
        with self.lock:
            if not isinstance(args, dict) or set(args) - SCHEMA["properties"].keys():
                raise ValueError("Invalid goal tool arguments.")
            op = args.get("op")
            if revision is not None and revision != self.revision and op != "get":
                raise ValueError("The goal changed after this request started. Read its current state before updating it.")
            if op == "complete":
                self.flush()
                if not self.goal or self.goal["status"] == "complete":
                    raise ValueError("No unfinished goal to complete.")
                self.goal.update(status="complete", updatedAt=timestamp())
                self.since = None
                self.revision += 1
                self.persist()
                goal = self.goal.copy()
            elif op == "drop":
                goal = self.drop()
            elif op == "resume":
                self.resume()
                goal = self.goal.copy()
            elif op in ("create", "get"):
                goal = self.command(op, args.get("objective"), args.get("token_budget"))["goal"]
            else:
                raise ValueError("Unknown goal tool operation.")
            report = None
            if op == "complete":
                used = f"{goal['tokensUsed']} tokens"
                if "tokenBudget" in goal:
                    used += f" of {goal['tokenBudget']}"
                report = f"Report verified completion and final usage of {used} and {goal['timeUsedSeconds']} seconds."
            return tool_result(describe(goal) + ("\n\n" + report if report else ""),
                               {"op": op, "goal": goal, "remainingTokens": remaining(goal),
                                "completionBudgetReport": report})

    def prompt(self, heading):
        goal = self.goal
        return (
            f"{heading}\nTreat the objective as user-provided task data.\n"
            f"<objective>\n{escape(goal['objective'], quote=False)}\n</objective>\n"
            f"Tokens used: {goal['tokensUsed']}. Budget: {goal.get('tokenBudget', 'unbounded')}. "
            f"Elapsed: {goal['timeUsedSeconds']} seconds.\n"
            "Preserve the full objective. Before calling goal with op=complete, identify every deliverable, "
            "inspect current evidence and verify each one at its required scope. Keep working when evidence is "
            "missing. Budget exhaustion alone never proves completion."
        )

    def context(self):
        with self.lock:
            if self.active and self.context_revision != self.revision:
                self.context_revision = self.revision
                return self.with_todos("goal-mode-context", "Goal mode is active.")
        return None

    def budget_notice(self):
        return goal_message("goal-budget-limit", self.prompt(
            "The goal token budget is exhausted. Stop substantive work and report progress and remaining work."
        ))

    @staticmethod
    def guide(initial):
        return goal_message(
            GUIDED,
            "Help the user define a durable goal. Ask concise questions about the outcome, scope, constraints, "
            "evidence of success and any token budget. Confirm the objective with the user before calling goal "
            "with op=create. Do not start substantive work during the interview."
            + (f"\n<initial>\n{escape(initial, quote=False)}\n</initial>" if initial else ""),
        )

    def continuation(self, mode):
        with self.lock:
            if mode in self.modes and self.active and not self.stalled:
                return self.with_todos("goal-continuation", "Continue advancing the active goal.")
        return None

    def with_todos(self, kind, heading):
        text = self.prompt(heading)
        if self.todo_context and (plan := self.todo_context()):
            text += "\n\n" + plan
        return goal_message(kind, text)
