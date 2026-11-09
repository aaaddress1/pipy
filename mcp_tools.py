"""Session-scoped MCP connections and tools."""

import asyncio
import json
import logging
import math
import os
import re
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from sqlite3 import Error as SQLiteError
from threading import Lock
from time import monotonic, time
from urllib.parse import urlsplit

import httpx2
from anyio import ClosedResourceError, EndOfStream
from mcp import Client, MCPError, StdioServerParameters, types
from mcp.client.auth import OAuthFlowError
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

from mcp_catalog import (
    McpCatalog, ResourceNotFound, list_pages, prompt_args, render_prompt, render_resource, resolve_resource, resource_path,
)
from mcp_oauth import ManagedOAuth, OAuthOptions, authorize
from runtime import database


logger = logging.getLogger(__name__)


def timeout_seconds(name, default):
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return value / 1000 if math.isfinite(value) and value >= 0 else default / 1000


@dataclass(frozen=True)
class ServerConfig:
    name: str
    transport: str
    address: str
    args: tuple
    options: tuple
    cwd: str | None = None
    timeout: float = 30_000
    instructions: bool = True
    oauth: OAuthOptions = OAuthOptions()


def parse_servers(servers):
    if not isinstance(servers, list):
        raise ValueError("mcpServers must be an array.")
    configs = {}
    for server in servers:
        if not isinstance(server, dict) or not isinstance(name := server.get("name"), str) or not name.strip():
            raise ValueError("Every MCP server requires a name.")
        transport = server.get("type", "stdio")
        if transport not in ("stdio", "http", "sse"):
            raise ValueError(f"Unsupported MCP transport for {name}.")
        if transport == "stdio":
            address, args = server.get("command"), server.get("args", [])
            option_key = "env"
            if not isinstance(address, str) or not address.strip() or not isinstance(args, list) or not all(
                isinstance(arg, str) for arg in args
            ):
                raise ValueError(f"Invalid MCP command for {name}.")
        else:
            address, args = server.get("url"), []
            option_key = "headers"
            parsed = urlsplit(address) if isinstance(address, str) else None
            if not parsed or parsed.scheme not in ("http", "https") or not parsed.hostname or (
                parsed.username is not None or parsed.fragment
            ):
                raise ValueError(f"Invalid MCP URL for {name}.")
        options = server.get(option_key, [])
        if not isinstance(options, list) or not all(
            isinstance(entry, dict) and isinstance(entry.get("name"), str) and entry["name"]
            and isinstance(entry.get("value"), str) for entry in options
        ):
            raise ValueError(f"Invalid MCP {option_key} for {name}.")
        pairs = {entry["name"]: entry["value"] for entry in options}
        cwd = server.get("cwd")
        timeout = server.get("timeout", 30_000)
        instructions = server.get("instructions", True)
        try:
            valid_timeout = type(timeout) in (int, float) and math.isfinite(timeout) and timeout >= 0
        except OverflowError:
            valid_timeout = False
        configs[name] = ServerConfig(
            name, transport, address, tuple(args), tuple(sorted(pairs.items())),
            cwd if isinstance(cwd, str) and cwd else None,
            timeout if valid_timeout else 30_000, instructions is not False, OAuthOptions.parse(server),
        )
    return tuple(configs.values())


def discover_servers(workspace, config_path, warn):
    root = Path(os.environ.get("PI_CONFIG_DIR") or Path.home() / ".omp").expanduser()
    profile = os.environ.get("OMP_PROFILE") or os.environ.get("PI_PROFILE")
    if profile and re.fullmatch(r"[A-Za-z0-9_-]+", profile) and profile != "default":
        user_dir = root / "profiles" / profile / "agent"
    else:
        user_dir = Path(os.environ.get("PI_CODING_AGENT_DIR") or root / "agent").expanduser()
    filenames = ("mcp.json", ".mcp.json")
    scopes = [(workspace / ".omp", False), (user_dir, True), (workspace, False)]
    files = [(directory / name, user) for directory, user in scopes for name in filenames]
    if config_path:
        files.insert(0, (config_path, True))
    configured, allowed, denied = {}, set(), set()
    for path, user in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(data, dict) or not isinstance(data.get("mcpServers", {}), dict):
                raise ValueError("mcpServers must be an object.")
        except FileNotFoundError:
            if path != config_path:
                continue
            raise
        except (ValueError, OSError) as error:
            if path == config_path:
                raise
            warn(f"MCP config {path}: {error}")
            continue
        if user:
            for key, values in (("enabledServers", allowed), ("disabledServers", denied)):
                if isinstance(items := data.get(key), list):
                    values.update(item for item in items if isinstance(item, str))
        for name, server in data.get("mcpServers", {}).items():
            if isinstance(server, dict):
                configured.setdefault(name, server)

    def expand(value):
        if isinstance(value, str):
            return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", lambda match: os.getenv(match[1], match[0]), value)
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item) for item in value]
        return value

    entries = []
    for name, raw in configured.items():
        if name in denied or raw.get("enabled") is False and name not in allowed:
            continue
        server = {**expand(raw), "name": name}
        server.setdefault("type", "http" if "url" in server and "command" not in server else "stdio")
        for key in ("env", "headers"):
            if isinstance(options := server.get(key), dict):
                server[key] = [{"name": key, "value": value} for key, value in options.items()]
        try:
            entries.extend(parse_servers([server]))
        except ValueError as error:
            warn(f"MCP config {name}: {error}")
    return tuple(entries)


def sanitize(value, fallback=""):
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or fallback


def capped_name(name):
    if len(name) <= 64:
        return name
    # Bun.hash uses seed-zero Wyhash. Only the long-input path is needed here.
    data = name.encode("ascii")
    a, b, c, d = (0xa0761d6478bd642f, 0xe7037ed1a0b428db, 0x8ebc6af09c88c6e3, 0x589965cc75374cc3)
    mask = (1 << 64) - 1

    def mix(left, right):
        product = left * right
        return (product & mask) ^ (product >> 64)

    def word(offset):
        return int.from_bytes(data[offset:offset + 8], "little")

    lanes = [mix(a, b)] * 3
    offset = 0
    while offset + 48 < len(data):
        for index, key in enumerate((b, c, d)):
            start = offset + 16 * index
            lanes[index] = mix(word(start) ^ key, word(start + 8) ^ lanes[index])
        offset += 48
    state = lanes[0] ^ lanes[1] ^ lanes[2]
    for start in range(offset, len(data) - 16, 16):
        state = mix(word(start) ^ b, word(start + 8) ^ state)
    product = (word(len(data) - 16) ^ b) * (word(len(data) - 8) ^ state)
    number = mix((product & mask) ^ a ^ len(data), (product >> 64) ^ b)
    suffix = ""
    while number:
        number, digit = divmod(number, 36)
        suffix = "0123456789abcdefghijklmnopqrstuvwxyz"[digit] + suffix
    suffix = (suffix or "0")[:8]
    return f"{name[:63 - len(suffix)]}_{suffix}"


def tool_name(server, name):
    server, name = sanitize(server, "server"), sanitize(name, "tool")
    return capped_name(f"mcp__{server}_{name.removeprefix(server + '_')}")


def tool_alias(name, registry):
    if name in registry or not name.startswith("mcp__"):
        return name
    suffix = name[5:]
    candidates = {
        tool_name(suffix[:match.start()], suffix[match.start() + 2:])
        for match in re.finditer(r"(?=__)", suffix) if 0 < match.start() < len(suffix) - 2
    }
    if sanitized := sanitize(suffix):
        candidates.add(capped_name("mcp__" + sanitized))
    matches = candidates.intersection(registry)
    return next(iter(matches)) if len(matches) == 1 else name


@dataclass(frozen=True)
class ExternalToolResult:
    content: list
    details: dict
    is_error: bool


def format_result(result, server, tool):
    blocks, texts = [], []
    structured = result.structured_content
    duplicated = False

    def flush():
        if texts:
            blocks.append({"type": "text", "text": "\n\n".join(texts)})
            texts.clear()

    for item in result.content:
        if item.type == "text":
            texts.append(item.text)
            if structured is not None:
                try:
                    duplicated |= json.loads(item.text) == structured
                except ValueError:
                    pass
        elif item.type == "image":
            flush()
            blocks.append({"type": "image_url", "image_url": {"url": f"data:{item.mime_type};base64,{item.data}"}})
        elif item.type == "resource":
            resource = item.resource
            text = getattr(resource, "text", "")
            texts.append(f"[Resource: {resource.uri}]" + ("\n" + text if text else ""))
    flush()
    if not blocks:
        blocks.append({"type": "text", "text": ""})
    if result.is_error:
        if blocks[0]["type"] == "text":
            blocks[0]["text"] = "Error: " + blocks[0]["text"]
        else:
            blocks.insert(0, {"type": "text", "text": "Error:"})
    details = {"serverName": server, "mcpToolName": tool, "isError": result.is_error}
    if structured is not None:
        details["structuredContent"] = structured
        if not duplicated:
            rendered = json.dumps(structured, ensure_ascii=False, indent=2)
            blocks.append({"type": "text", "text": f"```json\n{rendered}\n```"})
    if result.meta is not None:
        details["mcpMeta"] = result.meta
    return ExternalToolResult(blocks, details, result.is_error)


@dataclass(frozen=True)
class McpTool:
    connection: object
    name: str
    description: str
    parameters: dict

    def invoke(self, args, turn, progress=None):
        properties = self.parameters.get("properties") or {}
        required = set(self.parameters.get("required") or [])
        args = {key: value for key, value in args.items() if not (
            key == "i" and key not in properties and key not in required
            or key in properties and key not in required and (value == "" or value == {})
        )}
        return self.connection.loop.call(self.connection.call(self.name, args, progress), turn)


class WatchedStream:
    def __init__(self, stream, on_close):
        self.stream, self.on_close = stream, on_close

    def __getattr__(self, name):
        return getattr(self.stream, name)

    async def __aenter__(self):
        await self.stream.__aenter__()
        return self

    async def __aexit__(self, *exc):
        return await self.stream.__aexit__(*exc)

    async def receive(self):
        try:
            return await self.stream.receive()
        except (EndOfStream, ClosedResourceError):
            self.on_close()
            raise

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return await self.receive()
        except EndOfStream:
            raise StopAsyncIteration from None


class ToolCache:
    def __init__(self, path, cwd):
        self.path, self.scope = path, os.path.normcase(str(cwd))

    def key(self, config):
        return sha256(json.dumps(asdict(config), sort_keys=True, ensure_ascii=True).encode("ascii")).hexdigest()

    def read(self, config):
        try:
            with database(self.path) as db:
                row = db.execute(
                    "SELECT tools FROM mcp_tool_cache WHERE workspace = ? AND config = ? AND expires > ?",
                    (self.scope, self.key(config), time()),
                ).fetchone()
            tools = json.loads(row[0]) if row else []
            if not isinstance(tools, list):
                return []
            return [
                entry for entry in tools
                if isinstance(entry, list) and len(entry) == 3
                and isinstance(entry[0], str) and isinstance(entry[1], str) and isinstance(entry[2], dict)
            ]
        except (SQLiteError, ValueError):
            return []

    def write(self, config, tools):
        try:
            with database(self.path) as db:
                db.execute(
                    "INSERT OR REPLACE INTO mcp_tool_cache VALUES (?, ?, ?, ?)",
                    (self.scope, self.key(config), json.dumps(tools), time() + 30 * 24 * 60 * 60),
                )
                db.execute("DELETE FROM mcp_tool_cache WHERE expires <= ?", (time(),))
        except (SQLiteError, ValueError):
            pass


class McpConnection:
    def __init__(self, manager, config, cwd, cache=None):
        loop, report = manager.loop, manager.report
        self.loop, self.config, self.cwd, self.report = loop, config, cwd, report
        self.notify = manager.notify
        self.cache = cache
        self.vault = manager.oauth_vault
        self.has_auth_header = any(key.lower() == "authorization" for key, _ in config.options)
        self.oauth = (
            ManagedOAuth(self.vault, config.address, config.oauth, self.has_auth_header)
            if self.vault and config.transport != "stdio" else None
        )
        self.timeout = timeout_seconds("OMP_MCP_TIMEOUT_MS", config.timeout) or None
        self.first_attempt = asyncio.Event()
        self.connected = asyncio.Event()
        self.changed = asyncio.Event()
        self.restart = False
        self.restarts = deque()
        self.suspended = False
        self.peer_closed = False
        self.connecting = False
        self.attempt_finished = asyncio.Event()
        self.attempt_finished.set()
        self.client = None
        self.auth_recovery = asyncio.Lock()
        self.last_challenge = ()
        self.tools = {
            tool_name(config.name, name): McpTool(self, name, description, parameters)
            for name, description, parameters in (cache.read(config) if cache else ())
        }
        self.instructions = ""
        self.problem = None
        self.catalog = McpCatalog(
            manager.commands_changed, lambda uri: manager.resource_updated(config.name, uri),
            self.catalog_diagnostic, lambda: manager.notifications_enabled,
        )

    @asynccontextmanager
    async def transport(self, *, auth=None, hooks=None):
        config = self.config
        options = dict(config.options)
        if auth is not None:
            options = {key: value for key, value in options.items() if key.lower() != "authorization"}
        else:
            auth = self.oauth
        if config.transport == "stdio":
            cwd = (self.cwd / config.cwd).resolve() if config.cwd else self.cwd
            yield stdio_client(StdioServerParameters(command=config.address, args=list(config.args), env=options, cwd=cwd))
        elif config.transport == "sse":
            def client_factory(**kwargs):
                return httpx2.AsyncClient(event_hooks=hooks, follow_redirects=False, **kwargs)

            yield sse_client(
                config.address, headers=options, sse_read_timeout=None, auth=auth, httpx_client_factory=client_factory,
            )
        else:
            options = {key: value for key, value in options.items() if key.lower() not in ("mcp-protocol-version", "mcp-session-id")}
            async with httpx2.AsyncClient(
                headers=options, timeout=httpx2.Timeout(5, read=None), auth=auth, event_hooks=hooks,
            ) as client:
                yield streamable_http_client(config.address, http_client=client)

    @asynccontextmanager
    async def watched(self, transport):
        async with transport as (read, write):
            yield WatchedStream(read, self.lost), write

    def lost(self):
        self.peer_closed = True
        if self.connected.is_set():
            self.connected.clear()
            self.reconnect()

    async def on_message(self, message):
        if isinstance(message, types.ToolListChangedNotification):
            self.changed.set()
        elif isinstance(message, types.ResourceListChangedNotification):
            self.catalog.mark("resources")
        elif isinstance(message, types.PromptListChangedNotification):
            self.catalog.mark("prompts")
        elif isinstance(message, types.ResourceUpdatedNotification):
            self.catalog.update(str(message.params.uri))
        elif isinstance(message, (OSError, httpx2.TransportError)):
            self.lost()

    async def roots(self, _):
        return types.ListRootsResult(roots=[types.Root(uri=self.cwd.as_uri(), name=self.cwd.name)])

    async def refresh(self):
        tools = {}
        if self.client.server_capabilities.tools is not None:
            for tool in await list_pages(self.client.list_tools, "tools"):
                name = tool_name(self.config.name, tool.name)
                if name not in tools or tool.name < tools[name].name:
                    tools[name] = McpTool(self, tool.name, tool.description or "", tool.input_schema)
        if self.tools != tools:
            self.tools = tools
        if self.cache:
            definitions = [(tool.name, tool.description, tool.parameters) for tool in tools.values()]
            await asyncio.to_thread(self.cache.write, self.config, definitions)

    def reconnect(self, manual=False):
        if manual:
            self.restarts.clear()
            self.suspended = False
        elif self.restart or self.suspended:
            return
        else:
            now = monotonic()
            while self.restarts and now - self.restarts[0] > 30:
                self.restarts.popleft()
            self.restarts.append(now)
            if len(self.restarts) > 5:
                self.suspended = True
                self.report(self.config.name, "Automatic reconnection suspended; use /mcp reconnect.")
        self.restart = True
        self.connected.clear()
        self.changed.set()

    async def run(self):
        delay = 1
        while True:
            while self.suspended:
                self.changed.clear()
                await self.changed.wait()
            self.restart = False
            self.peer_closed = False
            self.changed.clear()
            self.connecting = True
            self.attempt_finished.clear()
            try:
                async with self.transport() as transport:
                    async with Client(
                        self.watched(transport), mode="legacy", read_timeout_seconds=self.timeout,
                        list_roots_callback=self.roots, message_handler=self.on_message,
                    ) as self.client:
                        await self.refresh()
                        if self.peer_closed:
                            raise OSError("MCP connection closed during startup.")
                        if self.config.instructions:
                            self.instructions = self.client.instructions or ""
                        self.problem = None
                        self.catalog.start(self.client)
                        self.connected.set()
                        self.first_attempt.set()
                        self.connecting = False
                        self.attempt_finished.set()
                        delay = 1
                        try:
                            while not self.restart:
                                await self.changed.wait()
                                self.changed.clear()
                                if not self.restart:
                                    await self.refresh()
                        finally:
                            self.connected.clear()
                            self.instructions = ""
                            await self.catalog.close()
            except asyncio.CancelledError:
                raise
            except Exception as error:
                if not self.first_attempt.is_set():
                    self.tools = {}
                while isinstance(error, BaseExceptionGroup):
                    error = error.exceptions[0]
                text = self.error_text(error)
                if self.problem != text:
                    self.problem = text
                    self.report(self.config.name, text)
            finally:
                self.client = None
                self.connected.clear()
                self.first_attempt.set()
                self.connecting = False
                self.attempt_finished.set()
            if self.restart:
                continue
            try:
                await asyncio.wait_for(self.changed.wait(), delay)
            except TimeoutError:
                pass
            delay = min(30, delay * 2)

    def error_text(self, error):
        text = str(error) or type(error).__name__
        if self.config.transport != "stdio":
            text = text.replace(self.config.address, urlsplit(self.config.address)._replace(query="").geturl())
        for _, value in self.config.options:
            if value:
                text = text.replace(value, "[redacted]")
        if self.config.oauth.client_secret:
            text = text.replace(self.config.oauth.client_secret, "[redacted]")
        if self.oauth and (
            isinstance(error, httpx2.HTTPStatusError) and error.response.status_code == 401
            or isinstance(error, MCPError) and re.search(r"\b(?:HTTP\s+)?401\b", text, re.I)
        ):
            text += f" (/mcp reauth {self.config.name})"
        return text

    def catalog_diagnostic(self, stage, error):
        logger.debug("MCP %s %s: %s", self.config.name, stage, self.error_text(error))

    async def wait_catalog(self, name):
        await self.first_attempt.wait()
        if self.connecting:
            await self.attempt_finished.wait()
        if self.connected.is_set():
            await self.catalog.ready[name].wait()

    async def request(self, operation, send):
        if self.suspended:
            raise OSError(f"MCP server {self.config.name} needs /mcp reconnect.")
        client = None
        try:
            await asyncio.wait_for(self.connected.wait(), self.timeout)
            client = self.client
            return await send(client)
        except (MCPError, OSError, httpx2.HTTPError, OAuthFlowError) as error:
            connection_lost = (
                isinstance(error, (OSError, httpx2.TransportError))
                or isinstance(error, httpx2.HTTPStatusError) and error.response.status_code in (404, 502, 503)
                or isinstance(error, MCPError) and re.search(r"connection closed|session terminated|stream closed", str(error), re.I)
            )
            if client and self.client is client and connection_lost:
                self.reconnect()
            raise OSError(f"MCP {operation} on {self.config.name} failed: {self.error_text(error)}") from error

    async def call(self, name, args, progress):
        async def invoke(client):
            return await client.call_tool(name, args, progress_callback=progress)

        previous = self.last_challenge
        result = await self.request(f"tool {name}", invoke)
        challenges = (result.meta or {}).get("mcp/www_authenticate")
        if result.is_error and self.can_authorize and isinstance(challenges, list):
            challenge = tuple(value.strip() for value in challenges if isinstance(value, str) and value.strip()
                              and value.isascii() and not re.search(r"[\x00-\x08\x0a-\x1f\x7f]", value))
            if challenge:
                async with self.auth_recovery:
                    # A concurrent tool call may have already satisfied the same challenge.
                    if previous is self.last_challenge or challenge != self.last_challenge:
                        try:
                            await self.reauth(self.notify, challenges=challenge)
                        except (OSError, ValueError) as error:
                            self.report(self.config.name, self.error_text(error))
                            return format_result(result, self.config.name, name)
                        self.last_challenge = challenge
                result = await self.request(f"tool {name}", invoke)
        return format_result(result, self.config.name, name)

    async def oauth_handshake(self, auth, hooks):
        async with self.transport(auth=auth, hooks=hooks) as transport:
            async with Client(transport, mode="legacy", read_timeout_seconds=240):
                pass

    @property
    def can_authorize(self):
        return bool(self.oauth and self.config.oauth.auth_type != "apikey"
                    and (not self.has_auth_header or self.oauth.explicit_id))

    async def reauth(self, notify, *, challenges=()):
        if not self.oauth or self.config.oauth.auth_type == "apikey":
            raise ValueError("This MCP server is not configured for OAuth.")
        if self.has_auth_header and not self.oauth.explicit_id:
            raise ValueError("Set auth.credentialId for the active profile to use managed OAuth with an explicit Authorization header.")
        try:
            await authorize(self.vault, self.config.address, self.config.oauth, self.oauth_handshake, notify,
                            challenges=challenges)
        except OAuthFlowError as error:
            raise OSError(self.error_text(error)) from error
        except ExceptionGroup as error:
            leaf = error
            while isinstance(leaf, BaseExceptionGroup):
                leaf = leaf.exceptions[0]
            raise OSError(self.error_text(leaf)) from error
        self.reconnect(manual=True)
        await asyncio.wait_for(self.connected.wait(), self.timeout)

    async def unauth(self):
        if not self.oauth or self.config.oauth.auth_type == "apikey":
            raise ValueError("This MCP server is not configured for OAuth.")
        key = self.oauth.key
        async with self.vault.lock(key + "-login"), self.vault.lock(key):
            await self.vault.forget(self.config.address, self.oauth.explicit_id)
        self.reconnect(manual=True)


class McpSession:
    def __init__(self, loop, configs, cwd, notify, *, cache=None, startup_ms=250, notifications=False, debounce_ms=500,
                 oauth_vault=None):
        self.loop = loop
        self.notify = notify
        self.oauth_vault = oauth_vault
        self.warnings = []
        self.lock = Lock()
        self.active = False
        self.on_commands = self.on_updates = None
        self.notifications_enabled = notifications
        self.debounce = debounce_ms / 1000
        self.update_timers = {}
        self.updates = {}
        self.wake_scheduled = False
        self.startup = timeout_seconds("OMP_MCP_STARTUP_TIMEOUT_MS", startup_ms) or None
        self.connections = {config.name: McpConnection(self, config, cwd, cache) for config in configs}
        self.tasks = []
        loop.call(self.start())

    async def start(self):
        self.tasks = [asyncio.create_task(connection.run()) for connection in self.connections.values()]
        await self.wait_ready(self.startup)

    async def wait_ready(self, timeout):
        try:
            await asyncio.wait_for(
                asyncio.gather(*(connection.first_attempt.wait() for connection in self.connections.values())), timeout,
            )
        except TimeoutError:
            pass
        return [key for key, connection in self.connections.items() if not connection.connected.is_set()]

    def instructions(self):
        return "\n\n".join(
            f"### {name}\n{connection.instructions.strip()}"
            for name, connection in sorted(self.connections.items()) if connection.instructions.strip()
        )

    def tools(self):
        result = {}
        for server in sorted(self.connections):
            connection = self.connections[server]
            for key, tool in connection.tools.items():
                result.setdefault(key, tool)
        return dict(sorted(result.items()))

    def prompt_commands(self):
        return [
            {"name": f"{name}:{prompt.name}", "description": prompt.description or f"MCP prompt from {name}",
             "input": {"hint": "arguments"}, "source": "mcp_prompt"}
            for name, connection in self.connections.items() if connection.connected.is_set()
            for prompt in connection.catalog.prompts
        ]

    def is_prompt_command(self, content):
        if isinstance(content, list):
            content = next((block["text"] for block in content if block["type"] == "text"), "")
        if not isinstance(content, str) or not content.startswith("/"):
            return False
        name = content[1:].partition(" ")[0]
        return any(command["name"] == name for command in self.prompt_commands())

    async def find_catalog(self, family, lookup, candidates):
        result = lookup()
        if result is not None:
            return result
        waiting = {asyncio.create_task(connection.wait_catalog(family)) for connection in candidates}
        try:
            while waiting:
                _, waiting = await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
                result = lookup()
                if result is not None:
                    return result
            return None
        finally:
            for task in waiting:
                task.cancel()
            await asyncio.gather(*waiting, return_exceptions=True)

    async def read(self, path):
        uri = resource_path(path)
        if not uri:
            raise ResourceNotFound("mcp:// requires a resource URI.")
        connections = tuple(self.connections.values())
        target = await self.find_catalog("resources", lambda: resolve_resource(connections, uri), connections)
        if target is None:
            available = self.describe_catalog("resources")
            raise ResourceNotFound(f'No MCP server has resource "{uri}".\n\nAvailable resources:\n{available}')
        result = await target.request("resource read", lambda client: client.read_resource(uri, cache_mode="bypass"))
        return render_resource(result)

    async def expand_prompt(self, text):
        command, _, arguments = text.partition(" ")
        if not command.startswith("/"):
            return None
        name = command[1:]
        candidates = [connection for server, connection in self.connections.items() if name.startswith(server + ":")]
        if not candidates:
            return None

        def lookup():
            for connection in candidates:
                if connection.connected.is_set():
                    for prompt in connection.catalog.prompts:
                        if f"{connection.config.name}:{prompt.name}" == name:
                            return connection, prompt.name
            return None

        target = await self.find_catalog("prompts", lookup, candidates)
        if target is None:
            raise ValueError(f"Unknown MCP prompt: {name}.")
        connection, prompt = target
        args = prompt_args(arguments)
        result = await connection.request("prompt", lambda client: client.get_prompt(prompt, args))
        return render_prompt(result)

    def describe_catalog(self, family, server=None):
        lines = []
        for name, connection in self.connections.items():
            if server and name != server or not connection.connected.is_set():
                continue
            catalog = connection.catalog
            if family == "resources":
                lines.extend(f"{resource.uri} ({name})" for resource in catalog.resources)
                lines.extend(f"{template.uri_template} ({name}, template)" for template in catalog.templates)
            else:
                lines.extend(f"/{name}:{prompt.name}" + (f" — {prompt.description}" if prompt.description else "")
                             for prompt in catalog.prompts)
        return "\n".join(lines) or f"No MCP {family} available."

    async def catalog_listing(self, family, server):
        if server and server not in self.connections:
            raise ValueError(f"Unknown MCP server: {server}.")
        connections = [connection for name, connection in self.connections.items() if not server or name == server]
        await asyncio.gather(*(connection.wait_catalog(family) for connection in connections))
        return self.describe_catalog(family, server)

    def command(self, arguments, turn=None):
        action, _, name = arguments.strip().partition(" ")
        name = name.strip()
        if action in ("", "list"):
            return self.status()
        if action in ("resources", "prompts"):
            return self.loop.call(self.catalog_listing(action, name), turn)
        if action == "reconnect":
            self.loop.call(self.reconnect(name), turn)
            return "MCP reconnection scheduled."
        if action in ("reauth", "unauth"):
            connection = self.connections.get(name)
            if not connection:
                raise ValueError(f"Unknown MCP server: {name or '(missing name)'}.")
            if action == "reauth":
                self.loop.call(connection.reauth(self.notify), turn)
                return f"MCP server {name} authorized and reconnected."
            self.loop.call(connection.unauth(), turn)
            return f"Removed the stored OAuth credential for {name}."
        if action == "notifications" and name in ("", "on", "off"):
            if name:
                self.loop.call(self.set_notifications(name == "on"), turn)
            return "MCP resource notifications " + ("enabled." if self.notifications_enabled else "disabled.")
        raise ValueError("Usage: /mcp list, resources, prompts, notifications, reconnect, reauth, or unauth.")

    async def set_notifications(self, enabled):
        self.notifications_enabled = enabled
        if not enabled:
            self.clear_updates()
        for connection in self.connections.values():
            if connection.connected.is_set() and connection.client.server_capabilities.resources is not None:
                connection.catalog.mark("resources")

    def clear_updates(self):
        for timer in self.update_timers.values():
            timer.cancel()
        self.update_timers.clear()
        with self.lock:
            self.updates.clear()

    def resource_updated(self, name, uri):
        key = name, uri
        if timer := self.update_timers.pop(key, None):
            timer.cancel()
        self.update_timers[key] = asyncio.get_running_loop().call_later(self.debounce, self.queue_update, key)

    def queue_update(self, key):
        self.update_timers.pop(key, None)
        with self.lock:
            if not self.notifications_enabled:
                return
            self.updates[key] = None
            if self.active and self.on_updates and not self.wake_scheduled:
                self.wake_scheduled = True
                asyncio.get_running_loop().call_soon(self.wake_for_updates)

    def wake_for_updates(self):
        with self.lock:
            self.wake_scheduled = False
            wake = self.on_updates if self.active and self.updates else None
        if wake:
            wake()

    def take_updates(self):
        with self.lock:
            if not self.updates:
                return None
            updates = tuple(self.updates)
            self.updates.clear()
        lines = [f"[MCP notification] {len(updates)} resource(s) updated:"]
        lines.extend(f'- server="{server}" uri={uri}' for server, uri in updates)
        lines.append('Use read(path="mcp://<uri>") to inspect if relevant.')
        return "\n".join(lines)

    def has_updates(self):
        with self.lock:
            return bool(self.updates)

    def commands_changed(self):
        with self.lock:
            notify = self.on_commands if self.active else None
        if notify:
            notify()

    def status(self):
        if not self.connections:
            return "No MCP servers configured."
        return "\n".join(
            f"{name}: " + (f"connected ({len(connection.tools)} tools)" if connection.connected.is_set()
                           else "reconnection suspended" if connection.suspended
                           else "reconnecting" if connection.problem else "connecting")
            for name, connection in self.connections.items()
        )

    async def reconnect(self, name):
        if name and name not in self.connections:
            raise ValueError(f"Unknown MCP server: {name}.")
        for key, connection in self.connections.items():
            if not name or key == name:
                connection.reconnect(manual=True)

    def report(self, server, message):
        text = f"MCP server {server}: {message}"
        with self.lock:
            if self.active:
                self.notify(text)
            else:
                self.warnings.append(text)

    def activate(self, on_commands=None, on_updates=None):
        with self.lock:
            for text in self.warnings:
                self.notify(text)
            self.warnings.clear()
            self.active = True
            self.on_commands, self.on_updates = on_commands, on_updates
            pending = bool(self.updates)
        if on_commands:
            on_commands()
        if pending and on_updates:
            on_updates()

    async def close(self):
        with self.lock:
            self.active = False
            self.on_commands = self.on_updates = None
        self.clear_updates()
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
