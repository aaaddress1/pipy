"""Journal records for work left unfinished when a session process exits."""

import json
from datetime import datetime, timezone


TOOL_START = "tool_execution_start"
SESSION_EXIT = "session_exit"
MISSING_RESULT = "Interrupted before result was recorded. Check workspace state before repeating this action."
UNKNOWN_RESULT = "Previous OMP process exited before this tool returned; its outcome is unknown."
INTERRUPTED_TURN = "Previous OMP process exited before completing the turn."


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


def abort_message(recorded_at, previous=None):
    rpc = (previous or {}).get("_rpc", {})
    details = {key: rpc[key] for key in ("api", "provider", "model") if key in rpc}
    details.update(stopReason="aborted", errorMessage=INTERRUPTED_TURN)
    try:
        details["timestamp"] = int(datetime.fromisoformat(recorded_at).timestamp() * 1000)
    except (TypeError, ValueError, OverflowError):
        pass
    return {"role": "assistant", "content": None, "_rpc": details}
