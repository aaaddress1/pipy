"""Shared tool waves separated by exclusive calls."""

import json
from concurrent.futures import CancelledError, as_completed, wait
from contextvars import copy_context

from cancellable_http import turn_scope
from handoff import Cancellation
from mcp_tools import ExternalToolResult


class BatchSignal(Cancellation):
    def __init__(self, cancellation, turn):
        super().__init__(cancellation, reason="Tool batch interrupted.")
        self.turn = turn
        self.soft_steering = getattr(turn, "soft_steering", True)

    def has_steering(self):
        return bool(self.turn and self.turn.has_steering())


def exclusive(call, tools):
    function = call["function"]
    name = function["name"]
    if name in ("write", "edit", "yield", "goal"):
        return True
    policy = getattr(tools.get(name), "concurrency", None)
    if name != "shell" and not callable(policy):
        return policy == "exclusive"
    try:
        args = json.loads(function["arguments"])
        if not isinstance(args, dict):
            return True
        return args.get("pty") is True if name == "shell" else policy(args) == "exclusive"
    except Exception:
        return True


def dispatch(calls, tools, executor, signal, invoke, finish):
    """Commit ready results as they arrive, then cross the next exclusive barrier."""
    def wave(indices):
        futures = {executor.submit(copy_context().run, invoke, index): index for index in indices}
        try:
            for future in as_completed(futures):
                finish(futures[future], future.result())
                del futures[future]
        except KeyboardInterrupt:
            signal.cancel()
            for future in as_completed(futures):
                finish(futures[future], future.result())
            raise
        except BaseException:
            signal.cancel()
            wait(futures)
            raise

    shared = []
    for index, call in enumerate(calls):
        if exclusive(call, tools):
            if shared:
                wave(shared)
                shared = []
            wave([index])
        else:
            shared.append(index)
    if shared:
        wave(shared)


def execute(session, seq, batch, tools, executor, call_tool, *, turn=None, cancellation=None, child=None, jobs=None):
    """Run one assistant batch and publish tool messages in call order."""
    calls = batch[0]["tool_calls"]
    signal = BatchSignal(cancellation, turn)
    results, completed, finished = [], set(), set()
    next_result = 0

    def invoke(index):
        call = calls[index]
        result = {"role": "tool", "tool_call_id": call["id"]}
        ran = False
        with turn_scope(signal):
            try:
                if turn:
                    turn.tool_start(call)
                signal.check()
                if child and child.submitted is not None:
                    result["content"] = "Skipped after the task result was submitted."
                elif turn and not turn.allow_tool(call):
                    result["content"] = "Error: Tool execution was declined by the client."
                else:
                    output = call_tool(call)
                    ran = True
                    if isinstance(output, ExternalToolResult):
                        result.update(content=output.content, _tool_details=output.details, _tool_error=output.is_error)
                        if output.details.get("op") == "wait" and (waited := output.details.get("waited")):
                            result["_irc_messages"] = waited
                    else:
                        result["content"] = output
            except CancelledError as error:
                result.update(content=str(error) or "Tool execution was interrupted.", _tool_error=True)
            except Exception as error:
                result.update(content=f"Error: {error}", _tool_error=True)
        return result, ran

    def finish(index, outcome):
        nonlocal next_result
        if index in finished:
            return
        result, ran = outcome
        try:
            session.update_tool_result(seq, index + 1, result)
        finally:
            if jobs:
                jobs.release_tool_delivery(result)
        batch[index + 1] = result
        if child and calls[index]["function"]["name"] == "yield":
            child.yield_result(result)
        finished.add(index)
        if ran:
            completed.add(index)
        if turn:
            turn.tool_finished(calls[index], result)
            while next_result in finished:
                results.append(turn.publish(batch[next_result + 1]))
                next_result += 1

    dispatch(calls, tools, executor, signal, invoke, finish)
    return results, completed
