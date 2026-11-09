"""Journal records for work left unfinished when a session process exits."""

import json
from datetime import datetime, timezone

from interrupted_thinking import provider_assistant

TOOL_START = "tool_execution_start"
SESSION_EXIT = "session_exit"
MISSING_RESULT = "Interrupted before result was recorded. Check workspace state before repeating this action."
UNKNOWN_RESULT = "Previous OMP process exited before this tool returned; its outcome is unknown."
INTERRUPTED_TURN = "Previous OMP process exited before completing the turn."
DELIVERY_MARKERS = ("_goal_message", "_async_jobs", "_mcp_notification", "_service_runs", "_irc_messages")


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def custom(kind, data):
    return {"type": "custom", "customType": kind, "data": data}


def argument_summary(args):
    if not isinstance(args, dict):
        return {}
    return {key: value[:200] + ("…" if len(value) > 200 else "") for key in ("command", "path")
            if isinstance(value := args.get(key), str) and value}


def tool_start(call):
    function = call["function"]
    result = {"toolCallId": call["id"], "toolName": function["name"], "startedAt": timestamp()}
    try:
        args = json.loads(function["arguments"])
    except (TypeError, ValueError):
        args = None
    if summary := argument_summary(args):
        result["args"] = summary
    if isinstance(args, dict) and isinstance(args.get("intent"), str):
        result["intent"] = args["intent"]
    return result


def placeholder(call):
    return {"role": "tool", "tool_call_id": call["id"], "content": MISSING_RESULT,
            "_tool_error": True, "_tool_pending": True}


def pending_result(message):
    return message.get("role") == "tool" and (
        message.get("_tool_pending") is True or message.get("content") == MISSING_RESULT
        and message.get("_tool_error") is not False and "_tool_details" not in message
    )


def stop_reason(message):
    return (message.get("_rpc") or {}).get("stopReason") or message.get("_tool_stop") or (
        "toolUse" if message.get("tool_calls") else "stop"
    )


def failed_turn(message):
    return message["role"] == "assistant" and stop_reason(message) in ("aborted", "error")


def context_delivery(message):
    return (message["role"] == "developer" and not message.get("userInitiated")
            or message["role"] == "user" and any(key in message for key in DELIVERY_MARKERS))


def synthetic_result(message):
    return message["role"] == "tool" and (message.get("_tool_details") or {}).get("__synthetic") is True


def provider_batch(batch, following=None):
    failed_calls = set()
    for index, message in enumerate(batch):
        if failed_turn(message):
            failed_calls.update(call["id"] for call in message.get("tool_calls", ()))
            neighbor = batch[index + 1] if index + 1 < len(batch) else following
            if not message.get("_retry_omit") and stop_reason(message) == "aborted" and (
                message.get("_live_interruption") or neighbor and "_interrupted_thinking" in neighbor
            ) and (projected := provider_assistant(message)):
                yield projected
        elif not cutoff(message) and (message["role"] != "tool" or message.get("tool_call_id") not in failed_calls):
            yield message


def provider_history(batches):
    for index, (_, batch) in enumerate(batches):
        following = batches[index + 1][1][0] if index + 1 < len(batches) and batches[index + 1][1] else None
        yield from provider_batch(batch, following)


def cutoff(message):
    return (message["role"] == "assistant" and stop_reason(message) in ("aborted", "error", "length")
            and not message.get("tool_calls") and not str(message.get("content") or "").strip())


def runnable_calls(assistant):
    calls = assistant.get("tool_calls", ())
    if not calls or stop_reason(assistant) not in ("stop", "toolUse", "tool_calls"):
        return ()
    try:
        if any(not isinstance(json.loads(call["function"]["arguments"]), dict) for call in calls):
            return ()
    except (KeyError, TypeError, ValueError):
        return ()
    return calls if len({call["id"] for call in calls}) == len(calls) else ()


def interrupted(message):
    return cutoff(message) and (message.get("_rpc") or {}).get("errorMessage") == INTERRUPTED_TURN


def failure_message(model, error, *, reason="error", draft=None):
    message = {"role": "assistant", "content": None, **(draft or {})}
    message["_rpc"] = {**model, **(message.get("_rpc") or {}), "stopReason": reason, "errorMessage": error}
    return message


def abort_message(recorded_at, previous=None):
    rpc = (previous or {}).get("_rpc", {})
    details = {key: rpc[key] for key in ("api", "provider", "model") if key in rpc}
    details.update(stopReason="aborted", errorMessage=INTERRUPTED_TURN)
    try:
        details["timestamp"] = int(datetime.fromisoformat(recorded_at).timestamp() * 1000)
    except (TypeError, ValueError, OverflowError):
        pass
    return {"role": "assistant", "content": None, "_rpc": details}
