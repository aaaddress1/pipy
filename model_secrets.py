"""Resolve configured model credentials and confine them to their configured origin."""

import asyncio
import os
import signal
import subprocess
import time
from threading import Lock

import httpx
import codex_responses

from auth_broker import ModelCredentialError
from model_catalog import local_key_placeholders
from models_config import Connection


CONFIG_KEY = object()
KEYLESS = object()


class ConfigSecrets:
    def __init__(self, workspace):
        self.workspace = workspace
        self.values = {}
        self.locks = {}

    def resolve(self, raw, refresh=None):
        if not raw:
            return None
        if not raw.startswith("!"):
            return os.getenv(raw) or raw
        command = raw[1:].strip()
        if not command:
            return None
        with self.locks.setdefault(command, Lock()):
            previous = self.values.get(command)
            now = time.monotonic()
            if previous and previous[1] > now and (refresh is None or previous[0] >= refresh):
                return previous[2]
            value = self.execute(command)
            self.values[command] = now, float("inf") if value else now + 30, value
            return value

    def execute(self, command):
        platform = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
        try:
            with subprocess.Popen(command, shell=True, cwd=self.workspace, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, **platform) as process:
                try:
                    output, _ = process.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        if os.name == "nt":
                            subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], capture_output=True,
                                           timeout=5, creationflags=subprocess.CREATE_NO_WINDOW)
                        else:
                            os.killpg(process.pid, signal.SIGKILL)
                    except (ProcessLookupError, subprocess.TimeoutExpired):
                        pass
                    finally:
                        process.kill()
                    return None
                if process.returncode == 0:
                    return output.decode("utf-8", errors="replace").strip() or None
        except (OSError, subprocess.SubprocessError):
            return None

    def headers(self, route, configured_key, refresh=None):
        connection = route.connection or Connection()
        resolved = httpx.Headers()
        for index, layer in enumerate(connection.header_layers):
            if index == 2 and connection.auth_header and configured_key:
                resolved["Authorization"] = "Bearer " + configured_key
            for header, raw in layer:
                if value := self.resolve(raw, refresh):
                    resolved[header] = value
        return resolved


def origin(url):
    return url.scheme, url.host, url.port


class ConfigRequestAuth(httpx.Auth):
    requires_request_body = True

    def __init__(self, route, credential, secrets):
        self.route, self.credential, self.secrets = route, credential, secrets
        self.origin = origin(httpx.URL(route.endpoint()))

    def resolved_headers(self, refresh=None):
        route = self.route
        configured_key = None if route.key_override else self.secrets.resolve(route.connection.key, refresh)
        if route.connection.mode == "none" and configured_key in local_key_placeholders():
            configured_key = None
        if self.credential is CONFIG_KEY:
            key = configured_key
        else:
            key = None if self.credential is KEYLESS else os.getenv(self.credential)
        if not key and self.credential is not KEYLESS:
            raise ModelCredentialError(f"No credential is available for route {route.name!r}.")
        headers = httpx.Headers()
        if key:
            if route.api == "anthropic-messages" and route.auth == "api-key":
                headers["x-api-key"] = key
            else:
                headers["Authorization"] = "Bearer " + key
        headers.update(self.secrets.headers(route, key if route.key_override else configured_key, refresh))
        if route.api == codex_responses.API and (authorization := headers.get("authorization", "")):
            codex_responses.account_headers(headers, authorization.removeprefix("Bearer "))
        return headers

    @staticmethod
    def authorize(request, original, headers):
        request.headers = original.copy()
        request.headers.update(headers)

    def check(self, request):
        if origin(request.url) != self.origin:
            raise ModelCredentialError("Configured model request changed origin.")
        original = request.headers.copy()
        original.pop("Authorization", None)
        original.pop("x-api-key", None)
        return original

    def sync_auth_flow(self, request):
        request.read()
        original = self.check(request)
        headers = self.resolved_headers()
        started = time.monotonic()
        self.authorize(request, original, headers)
        response = yield request
        if response.status_code == 401:
            updated = self.resolved_headers(started)
            if updated != headers:
                response.read()
                self.authorize(request, original, updated)
                yield request

    async def async_auth_flow(self, request):
        await request.aread()
        original = self.check(request)
        headers = await asyncio.to_thread(self.resolved_headers)
        started = time.monotonic()
        self.authorize(request, original, headers)
        response = yield request
        if response.status_code == 401:
            updated = await asyncio.to_thread(self.resolved_headers, started)
            if updated != headers:
                await response.aread()
                self.authorize(request, original, updated)
                yield request
