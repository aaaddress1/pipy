"""One socket-local Responses continuation per route and credential."""

import asyncio
import json
import re
from copy import deepcopy
from urllib.parse import urlsplit, urlunsplit

import httpx
from openai import APIStatusError
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus, WebSocketException
from websockets.protocol import State

from cancellable_http import active_turn


UNSUPPORTED_UPGRADE = frozenset((404, 405, 426, 501))
TERMINAL = frozenset(("response.completed", "response.done", "response.failed", "response.incomplete"))
IDLE_SECONDS = 30


class NoRedirectConnect(connect):
    def process_redirect(self, error):
        return error


def replay_item(item):
    ignore = {"status"}
    if item.get("type") in ("message", "function_call", "custom_tool_call"):
        ignore.add("id")
    return {key: value for key, value in item.items() if key not in ignore}


def delta_input(baseline, request):
    if baseline is None:
        return None
    _, previous, output = baseline
    excluded = {"input", "client_metadata"}
    if ({key: item for key, item in previous.items() if key not in excluded}
        != {key: item for key, item in request.items() if key not in excluded}):
        return None
    prefix = previous["input"] + output
    current = request["input"]
    if len(current) > len(prefix) and all(replay_item(left) == replay_item(right) for left, right in zip(prefix, current)):
        return current[len(prefix):]
    return None


def stale_anchor(error):
    message = error.get("message") or ""
    return error.get("code") == "previous_response_not_found" or (
        isinstance(message, str) and re.search(r"previous[ _]?response", message, re.I)
        and re.search(r"not[ _]?found|invalid|expired|stale|unsupported", message, re.I)
    )


class ResponsesSocket:
    def __init__(self, loop, base_url, headers, *, chain=True):
        url = urlsplit(base_url)
        if url.scheme not in ("http", "https") or not url.hostname or url.username is not None or url.query or url.fragment:
            raise ValueError("Responses WebSocket requires an HTTP(S) endpoint without userinfo, query, or fragment.")
        path = url.path.rstrip("/") + "/responses"
        self.http_url = urlunsplit((url.scheme, url.netloc, path, "", ""))
        self.url = urlunsplit(("wss" if url.scheme == "https" else "ws", url.netloc, path, "", ""))
        self.loop, self.headers, self.chain = loop, headers, chain
        self.connection = self.connected_headers = self.baseline = self.timer = None
        self.connect_failures = self.stale_anchors = 0
        self.sse_only = False

    async def disconnect(self):
        if self.timer:
            self.timer.cancel()
            self.timer = None
        connection, self.connection = self.connection, None
        self.connected_headers = self.baseline = None
        if connection:
            await connection.close()

    def close(self):
        self.loop.call(self.disconnect())

    async def connect(self, headers):
        if self.timer:
            self.timer.cancel()
            self.timer = None
        if self.connection and self.connection.state is State.OPEN and headers == self.connected_headers:
            return
        await self.disconnect()
        self.connection = await NoRedirectConnect(self.url, additional_headers=headers, open_timeout=10,
                                                   close_timeout=1, max_size=16 * 1024 * 1024)
        self.connected_headers = headers

    def status_error(self, status, body=None, headers=None):
        body = body or {}
        response = httpx.Response(status, headers=headers, request=httpx.Request("POST", self.http_url))
        return APIStatusError(body.get("message") or f"Responses WebSocket HTTP {status}.", response=response, body=body)

    def open(self):
        if self.sse_only:
            return False
        headers = self.headers()
        try:
            self.loop.call(self.connect(headers), active_turn())
        except InvalidStatus as error:
            status = error.response.status_code
            if status not in UNSUPPORTED_UPGRADE:
                raise self.status_error(status, headers=list(error.response.headers.raw_items())) from None
            self.sse_only = True
            return False
        except (OSError, TimeoutError, WebSocketException):
            self.connect_failures += 1
            self.sse_only = self.connect_failures >= 2
            return False
        return True

    async def arm_idle(self):
        self.timer = asyncio.get_running_loop().call_later(IDLE_SECONDS, lambda: asyncio.create_task(self.disconnect()))

    def events(self, body):
        turn = active_turn()
        request = {**body, "type": "response.create"}
        baseline, self.baseline = self.baseline, None
        delta = delta_input(baseline, request) if self.chain and self.stale_anchors < 3 else None
        wire = {**request, "previous_response_id": baseline[0], "input": delta} if delta is not None else request
        ended = False
        try:
            self.loop.call(self.connection.send(json.dumps(wire)), turn)
            first = True
            while True:
                raw = self.loop.call(asyncio.wait_for(self.connection.recv(), 120), turn)
                event = json.loads(raw)
                kind = event["type"]
                response = event.get("response") or {}
                error = (response.get("error") or event.get("error") or event) if kind in ("error", "response.failed") else {}
                status = event.get("status") or event.get("status_code") or error.get("status") or error.get("status_code")
                usage = response.get("usage") or {}
                generated = response.get("output") or any(
                    isinstance(usage.get(key), (int, float)) and usage[key] > 0
                    for key in ("input_tokens", "output_tokens", "total_tokens")
                )
                if first and delta is not None and not generated and status in (None, 400, 422) and stale_anchor(error):
                    self.stale_anchors += 1
                    delta = None
                    self.loop.call(self.connection.send(json.dumps(request)), turn)
                    continue
                first = False
                if kind == "error":
                    if not isinstance(status, int) or not 400 <= status <= 599:
                        code = error.get("code") or ""
                        status = 429 if code in ("rate_limit_exceeded", "insufficient_quota") else 400
                    raise self.status_error(status, error)
                if kind in TERMINAL:
                    if kind == "response.done" and response.get("status") in ("completed", "incomplete", "failed"):
                        event["type"] = f"response.{response['status']}"
                    ended = kind in ("response.completed", "response.done", "response.incomplete")
                    if ended:
                        self.connect_failures = 0
                        if delta is not None:
                            self.stale_anchors = 0
                    if event["type"] == "response.completed" and response.get("status") == "completed":
                        if isinstance(response.get("id"), str) and isinstance(response.get("output"), list):
                            self.baseline = (response["id"], deepcopy(request), deepcopy(response["output"]))
                    yield event
                    return
                yield event
        except (OSError, TimeoutError, WebSocketException) as error:
            raise httpx.RemoteProtocolError("Responses WebSocket closed before completion.") from error
        finally:
            if ended:
                self.loop.call(self.arm_idle())
            else:
                self.close()
