"""Codex OAuth wire format, request identity, and Responses replay."""

import json
import os
import re
from collections import deque
from contextlib import closing, contextmanager
from contextvars import ContextVar
from functools import cache
from hashlib import sha256
from threading import Lock
from time import sleep, time
from uuid import UUID, uuid4

import httpx

import codex_attestation
from auth_broker_cache import config_root
from cancellable_http import active_turn
from local_oauth import jwt_payload
from live_steering import SteerExchange
from responses_websocket import ResponsesSocket, delta_input


API = "openai-codex-responses"
BASE_URL = "https://chatgpt.com/backend-api"
VERSION = "0.159.0"
TURN_STATE = "x-codex-turn-state"
TURN_METADATA = "x-codex-turn-metadata"
LITE_HEADER = "x-openai-internal-codex-responses-lite"
INSTALLATION_ID = "x-codex-installation-id"
WINDOW_ID = "x-codex-window-id"
MODELS_ETAG = "x-models-etag"
WEBSOCKET_RETRIES = 5
TOOL_CALLS = {"function_call": "function_call_output", "custom_tool_call": "custom_tool_call_output",
              "computer_call": "computer_call_output"}
TOOL_OUTPUTS = frozenset(TOOL_CALLS.values())
_turn = ContextVar("codex_logical_turn", default=None)


def endpoint(base):
    if base.endswith("/codex/responses"):
        return base
    return base + ("/responses" if base.endswith("/codex") else "/codex/responses")


def account_headers(headers, token):
    claims = jwt_payload(token).get("https://api.openai.com/auth")
    if not isinstance(claims, dict):
        return
    account = claims.get("chatgpt_account_id")
    if isinstance(account, str) and account and account.isascii() and account.isprintable():
        headers["chatgpt-account-id"] = account
    residency = next((value.strip() for key in ("chatgpt_data_residency", "chatgpt_compute_residency")
                      if isinstance(value := claims.get(key), str) and value.strip()), "")
    if (residency and residency.isascii() and residency.isprintable() and residency.lower() != "no_constraint"
        and "x-openai-internal-codex-residency" not in headers):
        headers["x-openai-internal-codex-residency"] = residency


@cache
def installation_id():
    try:
        return str(UUID((config_root("default") / "install-id").read_text(encoding="utf-8").strip()))
    except (OSError, ValueError):
        return str(uuid4())


@contextmanager
def logical_turn():
    token = _turn.set({})
    try:
        yield
    finally:
        _turn.reset(token)


class SessionIdentity:
    def __init__(self, session_id):
        self.session_id = session_id or str(uuid4())
        self.thread_id, self.window_id = str(uuid4()), str(uuid4())

    def turn(self, lane=None):
        current = _turn.get()
        if current is None or lane is None:
            return RequestIdentity(self)
        key = self, lane
        if key not in current:
            current[key] = RequestIdentity(self)
        return current[key]


class RequestIdentity:
    def __init__(self, session):
        self.session = session
        self.id = str(uuid4())
        self.started = int(time() * 1000)
        self.state = None

    def observe(self, headers):
        if isinstance(headers, dict):
            headers = {key.lower(): value for key, value in headers.items()}
        if self.state is None and isinstance(value := headers.get(TURN_STATE), str) and value:
            self.state = value

    def decorate(self, body, *, lite=False, compaction=None):
        session = self.session
        metadata = {"installation_id": installation_id(), "session_id": session.session_id,
                    "thread_id": session.thread_id, "turn_id": self.id, "window_id": session.window_id,
                    "request_kind": "compaction" if compaction else "turn", "turn_started_at_unix_ms": self.started}
        if compaction:
            metadata["compaction"] = compaction
        encoded = json.dumps(metadata, ensure_ascii=True, separators=(",", ":"))
        body["client_metadata"] = {
            INSTALLATION_ID: installation_id(), "session_id": session.session_id, "thread_id": session.thread_id,
            WINDOW_ID: session.window_id, "turn_id": self.id, TURN_METADATA: encoded,
        }
        headers = {"originator": "omp", "version": VERSION, "Accept": "text/event-stream",
                   "session_id": session.session_id, "conversation_id": session.session_id,
                   "x-client-request-id": session.session_id, "session-id": session.session_id,
                   "thread-id": session.thread_id, WINDOW_ID: session.window_id, TURN_METADATA: encoded,
                   "x-codex-routing-hint": f"model={body['model']}"}
        if tier := body.get("service_tier"):
            headers["x-codex-routing-hint"] += f";tier={tier}"
        if lite:
            headers[LITE_HEADER] = "true"
        if self.state:
            headers[TURN_STATE] = self.state
        return headers


def sanitize_call_id(value):
    base = re.split(r"[\n|]", value, maxsplit=1)[0] or value.lstrip("\n|")
    clean = re.sub(r"[^a-zA-Z0-9_-]", "_", base).rstrip("_")
    if clean and clean == base and len(clean) <= 64:
        return clean
    digest = sha256((base or "empty").encode()).hexdigest()[:16]
    return (clean or "call")[:47] + "_" + digest


def replay_items(items):
    cleaned = []
    for item in items:
        kind = item.get("type")
        if kind == "item_reference":
            continue
        item = {key: value for key, value in item.items()
                if key != "internal_chat_message_metadata_passthrough" and (key != "id" or kind == "computer_call")}
        if isinstance(call_id := item.get("call_id"), str):
            item["call_id"] = sanitize_call_id(call_id)
        if kind in ("message", "function_call", "custom_tool_call"):
            item.pop("status", None)
        cleaned.append(item)
    return cleaned


def history_items(items):
    cleaned = replay_items(items)
    calls = {item["call_id"]: TOOL_CALLS[item["type"]] for item in cleaned
             if item.get("type") in TOOL_CALLS and "call_id" in item}
    outputs = {(item.get("call_id"), item["type"]) for item in cleaned if item.get("type") in TOOL_OUTPUTS}
    repaired = []
    for item in cleaned:
        kind, call_id = item.get("type"), item.get("call_id")
        if kind in TOOL_OUTPUTS and call_id is not None and calls.get(call_id) != kind:
            output = item.get("output")
            text = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)
            if len(text) > 16_000:
                text = text[:16_000] + "\n...[truncated]"
            repaired.append({"type": "message", "role": "assistant",
                             "content": f"[Previous {item.get('name') or 'tool'} result; call_id={call_id}]: {text}"})
        elif kind == "computer_call" and call_id is not None and (call_id, TOOL_CALLS[kind]) not in outputs:
            repaired.append({"type": "message", "role": "assistant",
                             "content": f"[Computer call interrupted before a screenshot was recorded; call_id={call_id}]"})
        else:
            repaired.append(item)
            if kind in TOOL_CALLS and call_id is not None and (call_id, TOOL_CALLS[kind]) not in outputs:
                repaired.append({"type": TOOL_CALLS[kind], "call_id": call_id,
                                 "output": "[No tool output recorded: the tool call was interrupted before it produced a result.]"})
    return repaired


def lite_enabled(route, *, compact=False):
    if compact:
        return (route.catalog_model or {}).get("useResponsesLite") is True
    return os.getenv("PI_CODEX_RESPONSES_LITE", "").strip().lower() in ("1", "true")


def websocket_enabled(catalog, preference):
    override = os.getenv("PI_CODEX_WEBSOCKET")
    return catalog.get("preferWebsockets") is not False and (
        preference if override is None else override.strip().lower() in ("1", "true", "on", "yes")
    )


def websocket_backoff(attempt):
    turn = active_turn()
    (turn.wait if turn else sleep)(0.5 * attempt)


def lite_shape(body):
    tools = body.pop("tools", [])
    choice = body.get("tool_choice")
    if isinstance(choice, dict):
        selected = next((tool for tool in tools if tool["type"] == choice.get("type") and (
            tool["type"] == "computer" or tool["type"] == "function" and tool["name"] == choice.get("name")
        )), None)
        if selected:
            tools, choice = [selected], "required"
    prefix = [{"type": "additional_tools", "role": "developer", "tools": tools}]
    if instructions := body.pop("instructions", None):
        prefix.append({"type": "message", "role": "developer",
                       "content": [{"type": "input_text", "text": instructions}]})
    for item in body["input"]:
        for key in ("content", "output"):
            if isinstance(parts := item.get(key), list):
                item[key] = [{k: v for k, v in part.items() if k != "detail"}
                             if isinstance(part, dict) and part.get("type") == "input_image" else part for part in parts]
    body["input"] = prefix + body["input"]
    body["parallel_tool_calls"] = False
    body["tool_choice"] = choice if choice in ("none", "required") else "auto"
    body.setdefault("reasoning", {})["context"] = "all_turns"


def request_shape(route, body, *, lite=False):
    model = route.catalog_model or {}
    body["input"] = history_items(body["input"])
    body["store"] = False
    body.pop("max_output_tokens", None)
    body.pop("max_completion_tokens", None)
    if reasoning := body.get("reasoning"):
        reasoning = body["reasoning"] = reasoning.copy()
        if model and not (model.get("compat") or {}).get("supportsReasoningSummary"):
            reasoning.pop("summary", None)
        if not reasoning:
            del body["reasoning"]
    if model.get("reasoningMode") and (body.get("reasoning") or {}).get("effort") != "none":
        body.setdefault("reasoning", {})["mode"] = model["reasoningMode"]
    if lite:
        lite_shape(body)
    body["include"] = ["reasoning.encrypted_content"]


class OutputItems:
    def __init__(self):
        self.items = {}

    def normalize(self, event):
        kind = event["type"]
        if kind == "response.output_item.done":
            index = event.get("output_index", len(self.items))
            self.items[index] = event["item"]
        elif kind in ("response.done", "response.completed", "response.incomplete", "response.failed"):
            response = event.setdefault("response", {})
            status = response.get("status") or {
                "response.done": "completed", "response.completed": "completed",
                "response.incomplete": "incomplete", "response.failed": "failed",
            }[kind]
            response["status"] = status
            if status == "incomplete" and (response.get("incomplete_details") or {}).get("reason") == "steered":
                response["status"] = status = "completed"
                response["_steered"] = True
                event["type"] = "response.completed"
            if not response.get("output"):
                response["output"] = [self.items[index] for index in sorted(self.items)]
            if kind == "response.done":
                event["type"] = f"response.{status}"
        return event


class CodexSocket(ResponsesSocket):
    receive_seconds = 300
    ping_interval = 10
    ping_timeout = 60
    connect_failure_limit = WEBSOCKET_RETRIES + 1
    stale_anchor_limit = float("inf")

    def __init__(self, loop, base_url, headers):
        super().__init__(loop, endpoint(base_url).removesuffix("/responses"), headers)
        self.lock = Lock()
        self.models_etag = None
        self.metadata_credentials = None
        self.steering_unsupported = False
        self.accepted_steers = []
        self.steer_frames = deque()

    async def disconnect(self, *, abort=False):
        self.accepted_steers.clear()
        self.steer_frames.clear()
        await super().disconnect(abort=abort)

    async def upgrade_headers(self, headers):
        return await codex_attestation.upgrade_headers(headers, self.http_url)

    def append_input(self, baseline, request):
        delta = delta_input(baseline, request, codex=True)
        if baseline is not None and delta is None:
            self.models_etag = None
        return delta

    def replay_output(self, output):
        replay = replay_items(output)
        return replay if len(replay) == len(output) else None

    def starts_response(self, kind):
        return kind not in ("response.created", "response.in_progress", "response.queued", "response.metadata")

    def rejects_anchor(self, error):
        if error.get("code") == "codex_previous_response_stale" or super().rejects_anchor(error):
            self.models_etag = None
            return True
        return False

    def observe(self, identity, headers):
        headers = httpx.Headers(headers)
        identity.observe(headers)
        if value := headers.get(MODELS_ETAG):
            self.models_etag = value

    @contextmanager
    def exchange(self, body, headers, identity, *, lite=False, compaction=False, live_source=None, vision=True, watch=None):
        if self.sse_only or not self.lock.acquire(blocking=False):
            yield None
            return
        try:
            credentials = httpx.Headers(self.headers())
            upgrade = credentials.copy()
            upgrade.update(headers)
            if self.metadata_credentials is not None and credentials != self.metadata_credentials:
                self.models_etag = identity.state = None
                upgrade.pop(TURN_STATE, None)
            self.metadata_credentials = credentials
            for key in ("accept", "content-type", "x-codex-beta-features"):
                upgrade.pop(key, None)
            upgrade["OpenAI-Beta"] = "responses_websockets=2026-02-06"
            if self.models_etag:
                upgrade[MODELS_ETAG] = self.models_etag
            connection = self.connection
            for attempt in range(self.connect_failure_limit):
                if self.open(headers=list(upgrade.items()), connection_key=credentials, watch=watch):
                    break
                if self.sse_only:
                    yield None
                    return
                websocket_backoff(attempt + 1)
            else:
                yield None
                return
            if self.connection is not connection:
                self.observe(identity, list(self.connection.response.headers.raw_items()))
            metadata = body["client_metadata"].copy()
            if lite:
                metadata["ws_request_header_x_openai_internal_codex_responses_lite"] = "true"
            if identity.state:
                metadata[TURN_STATE] = identity.state
            wire = {**body, "stream": True, "client_metadata": metadata}

            def reopen():
                if not self.open(headers=list(upgrade.items()), connection_key=credentials, watch=watch):
                    raise httpx.RemoteProtocolError("Could not reopen the Codex steering socket.")

            steering = (SteerExchange(self, live_source, lite, vision, reopen)
                        if live_source is not None or self.accepted_steers else None)
            with closing(self.events(wire, normalize=OutputItems().normalize, retain_baseline=not compaction,
                                     steering=steering, watch=watch)) as stream:
                yield stream
        finally:
            self.lock.release()
