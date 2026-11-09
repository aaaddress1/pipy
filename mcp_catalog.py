"""Discovery and routing for the MCP resource and prompt catalogs."""

import asyncio
import re

from mcp import MCPError, types


URI_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
DRIVE_PATH = re.compile(r"^[A-Za-z]:")
URI_EXPRESSION = re.compile(r"\{[^}]+\}")


class ResourceNotFound(OSError):
    pass


def resource_path(path):
    if path[:6].lower() == "mcp://":
        return path[6:].strip()
    if URI_SCHEME.match(path) and not DRIVE_PATH.match(path):
        return path
    return None


def template_score(uri, template):
    pieces = URI_EXPRESSION.split(template)
    pattern = ".*?".join(re.escape(piece) for piece in pieces)
    if re.fullmatch(pattern, uri):
        return sum(map(len, pieces)), 1 - len(pieces)
    return None


def resolve_resource(connections, uri):
    connected = [connection for connection in connections if connection.connected.is_set()]
    for connection in connected:
        if any(str(resource.uri) == uri for resource in connection.catalog.resources):
            return connection
    winner, best = None, None
    for connection in connected:
        for template in connection.catalog.templates:
            score = template_score(uri, template.uri_template)
            if score is not None and (best is None or score > best):
                winner, best = connection, score
    return winner


async def list_pages(method, field):
    items, cursors = [], set()
    cursor = None
    while True:
        page = await method(cursor=cursor, cache_mode="bypass")
        items.extend(getattr(page, field))
        cursor = page.next_cursor
        if not cursor:
            return tuple(items)
        if cursor in cursors:
            raise ValueError(f"MCP {method.__name__} repeated its cursor.")
        cursors.add(cursor)


def prompt_args(text):
    args, token = {}, []
    index = 0
    started = False
    while index <= len(text):
        char = text[index] if index < len(text) else " "
        if char in " \t":
            key, equals, value = "".join(token).partition("=")
            if key and equals:
                args[key] = value
            token.clear()
            started = False
        else:
            if char in ("'", '"') and (not started or token[-1:] == ["="]):
                closing = re.search(re.escape(char) + r"(?=[ \t]|$)", text[index + 1:])
                if closing:
                    end = index + 1 + closing.start()
                    token.append(text[index + 1:end])
                    index = end
                else:
                    token.append(char)
            else:
                token.append(char)
            started = True
        index += 1
    return args


def render_prompt(result):
    parts = []
    for message in result.messages:
        for item in message.content if isinstance(message.content, list) else (message.content,):
            if item.type == "text":
                parts.append(item.text)
            elif item.type == "resource" and (text := getattr(item.resource, "text", "")):
                parts.append(text)
    return "\n\n".join(parts)


def render_resource(result):
    parts = []
    for item in result.contents:
        if hasattr(item, "text"):
            parts.append(item.text)
        elif item.blob:
            mime = item.mime_type or "unknown"
            parts.append(f"[Binary content: {mime}, base64 length {len(item.blob)}]")
    return "\n---\n".join(parts) if parts else "(empty resource)"


class McpCatalog:
    def __init__(self, prompts_changed, updated, diagnostic, notifications):
        self.prompts_changed, self.updated, self.diagnostic = prompts_changed, updated, diagnostic
        self.notifications = notifications
        self.resources = self.templates = self.prompts = ()
        self.subscribed = set()
        self.dirty = {name: asyncio.Event() for name in ("resources", "prompts")}
        self.ready = {name: asyncio.Event() for name in self.dirty}
        self.tasks = []

    def start(self, client):
        capabilities = client.server_capabilities
        self.tasks = []
        for name, refresh in (("resources", self.refresh_resources), ("prompts", self.refresh_prompts)):
            self.ready[name].clear()
            if getattr(capabilities, name) is None:
                self.ready[name].set()
            else:
                self.mark(name)
                self.tasks.append(asyncio.create_task(self.watch(client, name, refresh)))

    def mark(self, name):
        self.ready[name].clear()
        self.dirty[name].set()

    async def watch(self, client, name, refresh):
        wake = self.dirty[name]
        while True:
            await wake.wait()
            wake.clear()
            try:
                await refresh(client)
            except Exception as error:
                self.diagnostic(name, error)
            finally:
                if not wake.is_set():
                    self.ready[name].set()

    async def refresh_resources(self, client):
        resources, templates = await asyncio.gather(
            list_pages(client.list_resources, "resources"),
            list_pages(client.list_resource_templates, "resource_templates"),
            return_exceptions=True,
        )
        if isinstance(resources, BaseException):
            self.diagnostic("resources", resources)
        else:
            self.resources = resources
        if isinstance(templates, BaseException):
            self.diagnostic("resource templates", templates)
        else:
            self.templates = templates
        if not self.dirty["resources"].is_set():
            self.ready["resources"].set()
        if client.server_capabilities.resources.subscribe:
            await self.reconcile_subscriptions(client)

    async def reconcile_subscriptions(self, client):
        desired = {str(resource.uri) for resource in self.resources} if self.notifications() else set()
        for uri in sorted(self.subscribed - desired):
            try:
                request = types.UnsubscribeRequest(params=types.UnsubscribeRequestParams(uri=uri))
                await client.session.send_request(request, types.EmptyResult)
            except (MCPError, OSError, ValueError) as error:
                self.diagnostic("resource unsubscribe", error)
            finally:
                self.subscribed.discard(uri)
        for uri in sorted(desired - self.subscribed):
            try:
                request = types.SubscribeRequest(params=types.SubscribeRequestParams(uri=uri))
                await client.session.send_request(request, types.EmptyResult)
            except (MCPError, OSError, ValueError) as error:
                self.diagnostic("resource subscribe", error)
            else:
                self.subscribed.add(uri)

    async def refresh_prompts(self, client):
        prompts = await list_pages(client.list_prompts, "prompts")
        if prompts != self.prompts:
            self.prompts = prompts
            self.prompts_changed()

    def update(self, uri):
        if self.notifications() and uri in self.subscribed:
            self.updated(uri)

    async def close(self):
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        for ready in self.ready.values():
            ready.set()
        self.subscribed.clear()
        self.resources = self.templates = ()
        if self.prompts:
            self.prompts = ()
            self.prompts_changed()
