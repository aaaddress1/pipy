"""Explicit MCP OAuth login and profile-scoped credential refresh."""

import asyncio
import ipaddress
import json
import os
import re
import secrets
from contextlib import aclosing, asynccontextmanager, closing
from dataclasses import dataclass, field
from hashlib import sha256
from sqlite3 import connect
from time import time
from urllib.parse import parse_qs, parse_qsl, quote_plus, urlencode, urljoin, urlsplit, urlunsplit

import httpx2
from anyio import BrokenResourceError, ClosedResourceError, DelimiterNotFound, EndOfStream, create_tcp_listener
from anyio.abc import SocketAttribute
from anyio.streams.buffered import BufferedByteReceiveStream
from filelock import AsyncFileLock
from mcp.client.auth import AuthorizationCodeResult, OAuthClientProvider, OAuthFlowError
from mcp.client.auth.utils import credentials_match_issuer, issuers_match
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthMetadata, OAuthToken, ProtectedResourceMetadata
from mcp.shared.auth_utils import check_resource_allowed, resource_url_from_server_url

from auth_broker import AuthBroker, AuthBrokerError, broker_config
from auth_broker_cache import SnapshotCache


def active_profile():
    profile = os.environ.get("OMP_PROFILE", os.environ.get("PI_PROFILE", "default"))
    return profile.strip().lower() or "default"


def loopback(host):
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def oauth_url(url):
    parsed = urlsplit(url)
    if not parsed.hostname or parsed.username is not None or parsed.fragment or (
        parsed.scheme != "https" and not (parsed.scheme == "http" and loopback(parsed.hostname))
    ):
        raise OAuthFlowError("MCP OAuth requires HTTPS endpoints or HTTP loopback.")
    return parsed


def origin(url):
    parsed = urlsplit(url)
    return parsed.scheme, parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)


async def check_oauth_request(request):
    oauth_url(str(request.url))


@dataclass(frozen=True)
class OAuthOptions:
    auth_type: str | None = None
    credential_id: str | None = None
    token_url: str | None = None
    resource: str | None = None
    client_id: str | None = None
    client_secret: str | None = field(default=None, repr=False)
    scope: str | None = None
    redirect_uri: str | None = None
    callback_port: int | None = None
    callback_path: str | None = None
    prompt: str | None = None
    client_metadata_url: str | None = None

    @classmethod
    def parse(cls, server):
        auth, oauth = server.get("auth"), server.get("oauth")
        if auth is None and oauth is None:
            return cls()
        if auth is not None and not isinstance(auth, dict) or oauth is not None and not isinstance(oauth, dict):
            raise ValueError("MCP auth and oauth must be objects.")
        auth, oauth = auth or {}, oauth or {}
        auth_type = auth.get("type")
        if auth and auth_type not in ("oauth", "apikey"):
            raise ValueError("MCP auth.type must be oauth or apikey.")

        def string(options, name):
            value = options.get(name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"MCP OAuth {name} must be a string.")
            return value

        callback_port = oauth.get("callbackPort")
        if callback_port is not None and (type(callback_port) is not int or not 0 <= callback_port <= 65535):
            raise ValueError("MCP OAuth callbackPort must be a valid port.")
        return cls(
            auth_type, string(auth, "credentialId"), string(auth, "tokenUrl"), string(auth, "resource"),
            string(oauth, "clientId") or string(auth, "clientId"),
            string(oauth, "clientSecret") or string(auth, "clientSecret"), string(oauth, "scope"),
            string(oauth, "redirectUri"), callback_port, string(oauth, "callbackPath"), string(oauth, "prompt"),
            string(oauth, "clientMetadataUrl"),
        )


class OAuthVault:
    def __init__(self, path, profile=None, broker=None):
        self.path = path
        self.profile = profile or active_profile()
        self.broker = broker

    @classmethod
    def configured(cls, path, model_providers=()):
        profile = active_profile()
        config = broker_config(profile)
        if not config:
            return cls(path, profile)
        url, token = config
        return cls(path, profile, AuthBroker(url, token, profile, SnapshotCache.configured(token, url, profile), model_providers))

    async def close(self):
        if self.broker:
            await self.broker.close()

    def key(self, url):
        return sha256((self.profile + "\0" + url).encode("utf-8")).hexdigest()

    def managed_id(self, url):
        return f"mcp_oauth:profile:{self.profile}:{url}"

    def lock(self, key):
        return AsyncFileLock(str(self.path.with_name(f"mcp-oauth-{key}.lock")), timeout=300)

    def providers(self, address, explicit):
        managed = self.managed_id(address)
        if explicit and explicit.startswith(("mcp_oauth:", "mcp_oauth_")) and (
            not explicit.startswith("mcp_oauth:profile:") or explicit == managed
        ):
            return tuple(dict.fromkeys((explicit, managed)))
        return (managed,)

    async def lookup(self, address, explicit=None):
        if self.broker:
            record = await self.broker.lookup(self.providers(address, explicit))
            if record and explicit == record["provider"]:
                record = {**record, "alias": explicit}
            return record
        return await asyncio.to_thread(self.read, self.key(address))

    async def save(self, address, record):
        if self.broker:
            await self.broker.upload(self.managed_id(address), record)
        else:
            await asyncio.to_thread(self.write, self.key(address), record)

    async def forget(self, address, explicit=None):
        if self.broker:
            await self.broker.disable(self.providers(address, explicit))
        else:
            await asyncio.to_thread(self.remove, self.key(address))

    @staticmethod
    def protect(data):
        if os.name != "nt":
            return data
        from win32crypt import CryptProtectData

        return CryptProtectData(data, "mini-pi MCP OAuth", None, None, None, 1)

    @staticmethod
    def unprotect(data):
        if os.name != "nt":
            return data
        from win32crypt import CryptUnprotectData

        return CryptUnprotectData(data, None, None, None, 1)[1]

    def read(self, key):
        if not self.path.is_file():
            return None
        with closing(connect(self.path, timeout=5)) as db:
            if not db.execute("SELECT 1 FROM sqlite_master WHERE name = 'credentials'").fetchone():
                return None
            row = db.execute("SELECT value FROM credentials WHERE binding = ?", (key,)).fetchone()
        return json.loads(self.unprotect(row[0])) if row else None

    def write(self, key, record):
        if not self.path.exists():
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                pass
            else:
                os.close(fd)
        value = self.protect(json.dumps(record, ensure_ascii=True).encode("ascii"))
        with closing(connect(self.path, timeout=5)) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS credentials(binding TEXT PRIMARY KEY, value BLOB NOT NULL)")
            db.execute("INSERT OR REPLACE INTO credentials VALUES (?, ?)", (key, value))

    def remove(self, key):
        if not self.path.is_file():
            return
        with closing(connect(self.path, timeout=5)) as db, db:
            if db.execute("SELECT 1 FROM sqlite_master WHERE name = 'credentials'").fetchone():
                db.execute("DELETE FROM credentials WHERE binding = ?", (key,))


def record_tokens(record, token):
    now = time()
    ttl = token.expires_in
    return {
        **record, "tokens": token.model_dump(mode="json", exclude_none=True),
        "expires": now + ttl if ttl is not None else None, "issued": now,
        "reauth": False, "retry_at": 0,
    }


class ManagedOAuth(httpx2.Auth):
    """Add stored bearer credentials and refresh under a shared binding lock."""

    def __init__(self, vault, address, options, has_header):
        self.vault, self.address, self.options, self.has_header = vault, address, options, has_header
        self.key = vault.key(address)
        self.origin = origin(address)

    @property
    def explicit_id(self):
        wanted = self.options.credential_id
        if wanted and wanted.startswith("mcp_oauth:profile:") and wanted != self.vault.managed_id(self.address):
            return None
        return wanted

    def usable(self, record):
        if not record or self.options.auth_type == "apikey":
            return False
        wanted = self.explicit_id
        explicit = bool(wanted and wanted in (record.get("alias"), record.get("provider", self.vault.managed_id(self.address))))
        return explicit or not self.has_header

    async def async_auth_flow(self, request):
        if origin(str(request.url)) != self.origin or self.options.auth_type == "apikey":
            yield request
            return
        record = await self.vault.lookup(self.address, self.explicit_id)
        if not self.usable(record):
            yield request
            return
        oauth_url(str(request.url))
        record = await self.refresh(record)
        if record.get("reauth") or expired(record):
            raise OAuthFlowError("MCP OAuth credential needs /mcp reauth.")
        request.headers["Authorization"] = "Bearer " + record["tokens"]["access_token"]
        response = yield request
        if response.status_code != 401:
            return
        replacement = await self.refresh(record, forced=True)
        if self.changed(replacement, record) and not replacement.get("reauth") and not expired(replacement):
            request.headers["Authorization"] = "Bearer " + replacement["tokens"]["access_token"]
            yield request

    def changed(self, current, observed):
        if self.vault.broker:
            return current["tokens"]["access_token"] != observed["tokens"]["access_token"]
        return current.get("issued") != observed.get("issued")

    async def refresh(self, observed, forced=False):
        if not forced and not refresh_due(observed):
            return observed
        async with self.vault.lock(self.key):
            current = await self.vault.lookup(self.address, self.explicit_id)
            if not self.usable(current):
                raise OAuthFlowError("MCP OAuth credential was removed.")
            if forced and self.changed(current, observed):
                return current
            if not forced and not refresh_due(current):
                return current
            if current.get("reauth") or current.get("retry_at", 0) > time():
                return current
            if self.vault.broker:
                try:
                    return await self.vault.broker.refresh(current, forced) or {**current, "reauth": True}
                except AuthBrokerError as error:
                    if error.status == 404:
                        return {**current, "reauth": True}
                    if expired(current):
                        raise OAuthFlowError(str(error)) from None
                    return current
            token = current["tokens"]
            token_url = current.get("token_url") or self.options.token_url
            if not token.get("refresh_token") or not token_url:
                return current
            try:
                refreshed = await refresh_token(self.address, current, self.options, token_url)
            except OAuthRefreshRejected:
                refreshed = {**current, "reauth": True}
            except (httpx2.HTTPError, OAuthFlowError, ValueError):
                refreshed = {**current, "retry_at": time() + 15}
            await asyncio.to_thread(self.vault.write, self.key, refreshed)
            return refreshed


def expired(record):
    expires = record.get("expires")
    return expires is not None and expires <= time()


def refresh_due(record):
    expires = record.get("expires")
    if expires is None or record.get("reauth"):
        return False
    if "broker_id" in record:
        return expires - 300 <= time()
    lifetime = max(0, expires - record["issued"])
    return expires - min(300, lifetime / 10) <= time()


class OAuthRefreshRejected(OAuthFlowError):
    pass


async def refresh_token(address, record, options, token_url):
    oauth_url(token_url)
    tokens = record["tokens"]
    client = record.get("client", {})
    client_id = client.get("client_id") or options.client_id
    client_secret = client.get("client_secret") or options.client_secret
    form = {"grant_type": "refresh_token", "refresh_token": tokens["refresh_token"]}
    resource = record.get("resource") or options.resource or resource_url_from_server_url(address)
    if resource:
        if not check_resource_allowed(requested_resource=resource_url_from_server_url(address), configured_resource=resource):
            raise OAuthFlowError("Stored OAuth resource does not cover this MCP URL.")
        form["resource"] = resource
    method = client.get("token_endpoint_auth_method") or ("client_secret_post" if client_secret else "none")
    basic = None
    if method == "client_secret_basic" and client_id and client_secret:
        basic = httpx2.BasicAuth(quote_plus(client_id), quote_plus(client_secret))
    elif method in ("none", "client_secret_post"):
        if client_id:
            form["client_id"] = client_id
        if method == "client_secret_post" and client_secret:
            form["client_secret"] = client_secret
    else:
        raise OAuthRefreshRejected("Unsupported token endpoint authentication method.")
    async with httpx2.AsyncClient(timeout=15, follow_redirects=False) as client:
        response = await client.post(token_url, data=form, auth=basic, headers={"Accept": "application/json"})
    if response.status_code in (400, 401, 403):
        try:
            code = response.json().get("error")
        except (ValueError, AttributeError):
            code = None
        if code in ("invalid_grant", "invalid_client", "unauthorized_client"):
            raise OAuthRefreshRejected("OAuth refresh was rejected.")
    response.raise_for_status()
    token = OAuthToken.model_validate(response.json())
    if not token.refresh_token:
        token.refresh_token = tokens["refresh_token"]
    if token.scope is None:
        token.scope = tokens.get("scope")
    return record_tokens(record, token)


class LoginStorage:
    def __init__(self, client):
        self.client, self.tokens = client, None

    async def get_tokens(self):
        return self.tokens

    async def set_tokens(self, tokens):
        self.tokens = tokens

    async def get_client_info(self):
        return self.client

    async def set_client_info(self, client):
        self.client = client


class LoopbackCallback:
    def __init__(self, options):
        self.options = options
        self.redirect = options.redirect_uri
        parsed = oauth_url(self.redirect) if self.redirect else None
        if parsed and (not loopback(parsed.hostname) or parsed.query):
            raise ValueError("OAuth redirectUri must be a loopback URL without a query.")
        self.host = "::1" if parsed and parsed.hostname == "::1" else "127.0.0.1"
        self.path = options.callback_path or (parsed.path if parsed else "") or "/callback"
        if not self.path.startswith("/") or self.path.startswith("//") or re.search(r"[\s?#]", self.path):
            raise ValueError("MCP OAuth callbackPath must be an absolute URL path.")
        self.port = options.callback_port
        self.fallback = self.port is None and not parsed
        if self.port is None:
            self.port = (parsed.port or (443 if parsed.scheme == "https" else 80)) if parsed else 3000
        if parsed and parsed.scheme == "https" and (
            options.callback_port is None or self.port == (parsed.port or 443)
        ):
            raise ValueError("HTTPS OAuth redirects require a separate local callbackPort.")
        self.state = None
        self.result = asyncio.get_running_loop().create_future()

    @asynccontextmanager
    async def listen(self):
        try:
            listener = await create_tcp_listener(local_host=self.host, local_port=self.port)
        except OSError:
            if not self.fallback:
                raise
            listener = await create_tcp_listener(local_host=self.host)
        if not self.redirect:
            port = listener.extra(SocketAttribute.local_port)
            self.redirect = f"http://127.0.0.1:{port}{self.path}"
        async with listener, asyncio.TaskGroup() as group:
            serving = group.create_task(listener.serve(self.handle))
            try:
                yield self
            finally:
                serving.cancel()

    async def handle(self, stream):
        async with stream:
            await self.respond(stream)

    async def respond(self, stream):
        try:
            reader = BufferedByteReceiveStream(stream)
            request = await asyncio.wait_for(reader.receive_until(b"\r\n\r\n", 8192), 5)
            method, target, version = request.split(b"\r\n", 1)[0].decode("ascii").split(" ", 2)
            parsed = urlsplit(target)
            fields = parse_qs(parsed.query, max_num_fields=32)
            states = fields.get("state", [])
            match = (
                method == "GET" and version in ("HTTP/1.0", "HTTP/1.1")
                and target.startswith("/") and not target.startswith("//") and not parsed.fragment
                and parsed.path == self.path and self.state and len(states) == 1
                and states[0].isascii() and secrets.compare_digest(states[0], self.state) and not self.result.done()
            )
            if match and "error" in fields:
                self.result.set_exception(OAuthFlowError("The authorization server declined login."))
            elif match and len(fields.get("code", [])) == 1 and len(fields.get("iss", [])) <= 1:
                self.result.set_result(AuthorizationCodeResult(
                    code=fields["code"][0], state=states[0], iss=fields.get("iss", [None])[0],
                ))
            else:
                match = False
            body = b"Authorization response received. Return to Mini Pi." if match else b"Invalid OAuth callback."
            status = b"200 OK" if match else b"400 Bad Request"
            await stream.send(b"HTTP/1.1 " + status + b"\r\nConnection: close\r\nContent-Type: text/plain\r\n"
                              b"Cache-Control: no-store\r\nContent-Length: " + str(len(body)).encode("ascii") + b"\r\n\r\n" + body)
        except (ValueError, OSError, BrokenResourceError, ClosedResourceError, DelimiterNotFound, EndOfStream):
            pass


class DiscoveryAuth(httpx2.Auth):
    """Resolve one challenge before letting the SDK perform the grant."""

    def __init__(self, provider, discover, *, start=False, challenges=()):
        self.provider, self.discover = provider, discover
        self.start, self.challenges = start, challenges

    async def async_auth_flow(self, request):
        async with aclosing(self.provider.async_auth_flow(request)) as flow:
            try:
                outgoing = await anext(flow)
                if self.discover and self.challenges:
                    headers = [("WWW-Authenticate", value) for value in self.challenges]
                    response = httpx2.Response(401, headers=headers, content=b"", request=request)
                elif self.discover and self.start:
                    response = None
                else:
                    response = yield outgoing
                needs_grant = response is None or response.status_code == 401 or (
                    response.status_code == 403 and challenge_fields(response).get("error") == "insufficient_scope"
                )
                if needs_grant:
                    if self.discover is None:
                        return
                    discover, self.discover = self.discover, None
                    await discover(response)
                    response = httpx2.Response(403, headers={"WWW-Authenticate": 'Bearer error="insufficient_scope"'},
                                               request=request)
                while True:
                    outgoing = await flow.asend(response)
                    response = yield outgoing
            except StopAsyncIteration:
                return


RESOURCE_METADATA = "/.well-known/oauth-protected-resource"
ISSUER_METADATA = "/.well-known/oauth-authorization-server"
OIDC_METADATA = "/.well-known/openid-configuration"
LEGACY_METADATA = ("/oauth/metadata", "/.mcp/auth", "/authorize")


def metadata_urls(address, route, *, issuer=False):
    parsed = oauth_url(address)
    base = urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
    path = parsed.path.rstrip("/")
    parent = path.rpartition("/")[0] or path
    gateway = base + parent + route
    inserted, appended = base + route + path, base + path + route
    if not route.startswith("/.well-known/"):
        paths = (gateway,)
    elif not issuer:
        paths = (inserted, gateway)
    elif route == OIDC_METADATA:
        paths = (appended, gateway, inserted)
    else:
        paths = (inserted, gateway, appended)
    yield from dict.fromkeys((*paths, base + route))


async def metadata_documents(client, address, routes, *, issuer=False, resource_hint=None):
    candidates = [(route, url) for route in routes for url in metadata_urls(address, route, issuer=issuer)]
    if resource_hint:
        candidates.insert(0, (RESOURCE_METADATA, resource_hint))
    for route, url in dict.fromkeys(candidates):
        try:
            async with client.stream("GET", url, headers={"Accept": "application/json"}) as response:
                if response.status_code != 200:
                    continue
                data = bytearray()
                async for chunk in response.aiter_bytes():
                    data.extend(chunk)
                    if len(data) > 1024 * 1024:
                        break
            if len(data) <= 1024 * 1024:
                document = json.loads(data)
                if isinstance(document, dict):
                    yield route, document
        except (httpx2.HTTPError, httpx2.InvalidURL, OAuthFlowError, ValueError):
            continue


def challenge_fields(response):
    fields = {}
    for header in response.headers.get_list("WWW-Authenticate"):
        for name, quoted, bare in re.findall(r'([\w-]+)\s*=\s*(?:"((?:[^"\\]|\\.)*)"|([^,\s]+))', header):
            fields.setdefault(name.lower(), re.sub(r'\\(["\\])', r'\1', quoted) or bare)
    issuer_hint = response.headers.get("Mcp-Auth-Server")
    if not issuer_hint:
        match = re.search(r"Mcp-Auth-Server:\s*([^;\]\s,]+)", response.headers.get("WWW-Authenticate", ""), re.I)
        issuer_hint = match[1] if match else None
    if issuer_hint:
        fields["mcp-auth-server"] = issuer_hint
    return fields


def metadata_string(document, *names):
    return next((value.strip() for name in names if isinstance(value := document.get(name), str) and value.strip()), None)


def discovery_hint(address, value):
    if value:
        try:
            url = urljoin(address, value)
            oauth_url(url)
            return url
        except (OAuthFlowError, ValueError):
            pass
    return None


def metadata_scope(document, *, resource=False):
    supported = document.get("scopes_supported")
    if resource and isinstance(supported, list):
        scope = " ".join(value.strip() for value in supported if isinstance(value, str) and value.strip())
        if scope:
            return scope
    return metadata_string(document, "scopes", "scope")


def metadata_endpoints(document, default_issuer):
    nested = next((document[name] for name in ("oauth", "authorization", "auth") if isinstance(document.get(name), dict)), {})
    fields = {**document, **nested}
    authorization = metadata_string(fields, "authorization_endpoint", "authorizationEndpoint", "authorization_url",
                                    "authorizationUrl", "authorization_uri", "authorizationUri", "authorize_url")
    token = metadata_string(fields, "token_endpoint", "tokenEndpoint", "token_url", "tokenUrl", "token_uri", "tokenUri")
    if not authorization or not token:
        return None, fields
    registration = metadata_string(fields, "registration_endpoint", "registrationEndpoint", "registration_url",
                                   "registrationUrl", "registration_uri", "registrationUri")
    try:
        parsed = oauth_url(authorization)
        issuer = metadata_string(fields, "issuer", "issuer_url", "issuerUrl") or default_issuer or (
            urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
        )
        for url in (issuer, authorization, token, registration):
            if url:
                oauth_url(url)
        return OAuthMetadata.model_validate({**fields, "issuer": issuer, "authorization_endpoint": authorization,
                                            "token_endpoint": token, "registration_endpoint": registration}), fields
    except (OAuthFlowError, ValueError):
        return None, fields


async def issuer_metadata(client, issuers, routes, *, authoritative=False):
    mismatch = None
    for issuer in dict.fromkeys(value.strip() for value in issuers if isinstance(value, str) and value.strip()):
        try:
            oauth_url(issuer)
        except (OAuthFlowError, ValueError):
            continue
        async for _, document in metadata_documents(client, issuer, routes, issuer=True):
            endpoints, fields = metadata_endpoints(document, issuer)
            if endpoints and issuers_match(str(endpoints.issuer), issuer):
                return endpoints, fields
            if endpoints:
                mismatch = issuer
    if authoritative:
        if mismatch:
            raise OAuthFlowError(f"Authorization server metadata issuer mismatch for {mismatch}.")
        raise OAuthFlowError("The advertised MCP authorization servers did not publish usable OAuth metadata.")
    return None, {}


async def discover_oauth(address, challenge=None):
    """Resolve legacy hints and protected-resource metadata for every login path."""
    canonical_resource = resource_url_from_server_url(address)
    parsed = oauth_url(address)
    base = urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
    headers = challenge_fields(challenge) if challenge is not None else {}
    required_scope = metadata_scope(headers)
    resource_hint = discovery_hint(address, metadata_string(headers, "resource_metadata"))
    issuer_hint = discovery_hint(address, metadata_string(headers, "mcp-auth-server"))
    direct, fields = None, {}
    if challenge is not None:
        # The SDK/httpx drains a challenged response before retrying anyway.
        body = await challenge.aread()
        if len(body) <= 1024 * 1024:
            try:
                document = json.loads(body)
                if isinstance(document, dict):
                    direct, fields = metadata_endpoints(document, issuer_hint)
            except ValueError:
                pass
        if not direct:
            direct, fields = metadata_endpoints({"authorization_url": headers.get("realm"), **headers}, issuer_hint)
    direct_scope = None
    if direct:
        query = dict(parse_qsl(oauth_url(str(direct.authorization_endpoint)).query))
        direct_scope = metadata_scope(fields) or metadata_scope(fields, resource=True) or query.get("scope") or required_scope

    def audience(document):
        return metadata_string(document, "resource", "resource_uri", "resourceUri")

    def bind(resource, endpoints):
        if not resource:
            return None
        if not check_resource_allowed(requested_resource=canonical_resource, configured_resource=resource):
            raise OAuthFlowError("Advertised OAuth resource does not cover this MCP URL.")
        return ProtectedResourceMetadata(resource=resource, authorization_servers=[endpoints.issuer])

    async with httpx2.AsyncClient(
        timeout=5, follow_redirects=True, max_redirects=4, event_hooks={"request": [check_oauth_request]},
    ) as client:
        routes = (ISSUER_METADATA, OIDC_METADATA, *LEGACY_METADATA)
        protected_documents = metadata_documents(client, address, (RESOURCE_METADATA,), resource_hint=resource_hint)
        async for _, document in protected_documents:
            resource = audience(document)
            if resource and not check_resource_allowed(requested_resource=canonical_resource, configured_resource=resource):
                continue
            servers = document.get("authorization_servers")
            resource_scope = metadata_scope(document, resource=True)
            if direct:
                return direct, bind(audience(fields) or resource, direct), direct_scope or resource_scope, fields
            if isinstance(servers, list) and servers:
                metadata, fields = await issuer_metadata(client, servers, routes, authoritative=True)
                scope = required_scope or resource_scope or metadata_scope(fields)
                return metadata, bind(resource or canonical_resource, metadata), scope, fields
            metadata, fields = metadata_endpoints(document, base)
            if metadata:
                return metadata, bind(resource or audience(fields), metadata), required_scope or resource_scope, fields

        if direct:
            return direct, bind(audience(fields), direct), direct_scope, fields
        if issuer_hint:
            metadata, fields = await issuer_metadata(client, (issuer_hint,), routes)
            if metadata:
                return metadata, None, required_scope or metadata_scope(fields), fields
        async for route, document in metadata_documents(client, address, routes):
            metadata, fields = metadata_endpoints(document, base)
            if metadata:
                resource = audience(fields)
                resource_scope = route in ("/oauth/metadata", "/.mcp/auth")
                scope = metadata_scope(document, resource=resource_scope) or metadata_scope(fields, resource=resource_scope)
                return metadata, bind(resource, metadata), required_scope or scope, fields
    if challenge is not None:
        raise OAuthFlowError("Could not discover MCP OAuth endpoints from the authorization challenge.")
    raise OAuthFlowError("MCP initialized anonymously and did not publish OAuth metadata.")


async def authorize(vault, address, options, handshake, notify, *, challenges=()):
    oauth_url(address)
    key = vault.key(address)
    async with vault.lock(key + "-login"):
        old = await vault.lookup(address, options.credential_id)
        async with LoopbackCallback(options).listen() as callback:
            metadata = OAuthClientMetadata(
                client_name="Mini Pi", redirect_uris=[callback.redirect],
                grant_types=["authorization_code", "refresh_token"], response_types=["code"],
                token_endpoint_auth_method="client_secret_post" if options.client_secret else "none",
            )
            client = None
            if options.client_id:
                client = OAuthClientInformationFull(
                    client_id=options.client_id, client_secret=options.client_secret,
                    **metadata.model_dump(exclude_none=True),
                )
            elif old and old.get("client", {}).get("client_id"):
                prior = OAuthClientInformationFull.model_validate(old["client"])
                if callback.redirect in [str(uri) for uri in prior.redirect_uris or ()]:
                    client = prior
            storage = LoginStorage(client)
            endpoint_params = {}
            scope = None
            endpoint_prompt = None

            async def redirect(url):
                parsed = oauth_url(url)
                params = {**endpoint_params, **dict(parse_qsl(parsed.query, keep_blank_values=True))}
                callback.state = params["state"]
                provider.context.client_metadata.scope = scope
                params.pop("scope", None)
                params.pop("prompt", None)
                if scope:
                    params["scope"] = scope
                prompt = endpoint_prompt or options.prompt
                if prompt is None and scope and "offline_access" in scope.split():
                    prompt = "consent"
                if prompt:
                    params["prompt"] = prompt
                if parsed.hostname == "accounts.google.com" or parsed.hostname.endswith(".accounts.google.com"):
                    params.setdefault("access_type", "offline")
                notify("Open this URL to authorize the MCP server:\n" + urlunsplit(parsed._replace(query=urlencode(params))),
                       action_required=True)

            async def receive():
                return await callback.result

            provider = OAuthClientProvider(
                address, metadata, storage, redirect_handler=redirect, callback_handler=receive,
                client_metadata_url=options.client_metadata_url,
            )

            async def prepare(challenge):
                nonlocal endpoint_params, endpoint_prompt, scope
                endpoints, protected, required, fields = await discover_oauth(address, challenge)
                authorization = oauth_url(str(endpoints.authorization_endpoint))
                endpoint_params = dict(parse_qsl(authorization.query, keep_blank_values=True))
                endpoint_prompt = endpoint_params.get("prompt")
                scope = required or endpoint_params.get("scope") or options.scope
                resource = options.resource or endpoint_params.get("resource")
                if not protected and resource:
                    if not check_resource_allowed(requested_resource=resource_url_from_server_url(address),
                                                  configured_resource=resource):
                        raise OAuthFlowError("OAuth resource does not cover this MCP URL.")
                    protected = ProtectedResourceMetadata(resource=resource, authorization_servers=[endpoints.issuer])
                public_id = metadata_string(fields, "client_id", "clientId", "default_client_id", "public_client_id")
                public_id = public_id or endpoint_params.get("client_id")
                reserved = ("state", "code_challenge", "code_challenge_method", "redirect_uri", "client_id",
                            "response_type", "scope", "resource", "prompt")
                endpoint_params = {key: value for key, value in endpoint_params.items() if key not in reserved}

                context = provider.context
                # The SDK appends its own query. A server's scope catalogue is not a grant request.
                context.oauth_metadata = OAuthMetadata.model_validate({
                    **endpoints.model_dump(mode="json"), "authorization_endpoint": urlunsplit(authorization._replace(query="")),
                    "scopes_supported": None,
                })
                context.client_metadata.scope = scope
                context.protected_resource_metadata = protected
                context.auth_server_url = str(endpoints.issuer)
                if context.client_info and not credentials_match_issuer(
                    context.client_info, context.auth_server_url, options.client_metadata_url,
                ):
                    context.client_info = storage.client = None
                if not context.client_info and public_id and not endpoints.registration_endpoint:
                    context.client_info = storage.client = OAuthClientInformationFull(
                        client_id=public_id, issuer=context.auth_server_url, **metadata.model_dump(exclude_none=True),
                    )
                methods = endpoints.token_endpoint_auth_methods_supported or ()
                if (context.client_info and options.client_id == context.client_info.client_id and options.client_secret
                    and "client_secret_basic" in methods and "client_secret_post" not in methods):
                    context.client_info.token_endpoint_auth_method = "client_secret_basic"

            hooks = {"request": [check_oauth_request]}
            async with asyncio.timeout(240):
                await handshake(DiscoveryAuth(provider, prepare, challenges=challenges), hooks)
                if not challenges and storage.tokens is None:
                    await handshake(DiscoveryAuth(provider, prepare, start=True), hooks)
            if storage.tokens is None or storage.client is None:
                raise OAuthFlowError("The MCP server did not complete an OAuth grant.")
            auth_metadata = provider.context.oauth_metadata
            token_url = str(auth_metadata.token_endpoint) if auth_metadata and auth_metadata.token_endpoint else (
                options.token_url or urljoin(address, "/token")
            )
            oauth_url(token_url)
            record = record_tokens({
                "token_url": token_url, "resource": provider.context.get_resource_url(),
                "authorization_url": str(auth_metadata.authorization_endpoint) if auth_metadata else None,
                "client": storage.client.model_dump(mode="json", exclude_none=True),
                "alias": options.credential_id,
            }, storage.tokens)
            async with vault.lock(key):
                await vault.save(address, record)
