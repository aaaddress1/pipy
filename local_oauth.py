"""Provider OAuth grants and the shared agent.db refresh lease."""

import asyncio
import base64
import hashlib
import ipaddress
import json
import os
import re
import sqlite3
from time import monotonic, time
from urllib.parse import quote_plus, urlsplit
from uuid import uuid4

import httpx2

from auth_broker import AuthBrokerError
from auth_broker_cache import REMOTE_REFRESH, number


GRANTS = {
    "anthropic": ("https://api.anthropic.com/v1/oauth/token", "9d1c250a-e61b-44d9-88ed-5944d1962f5e"),
    "openai-codex": ("https://auth.openai.com/oauth/token", "app_EMoamEEZ73f0CkXaXp7hrann"),
}
ORG_SCOPED = frozenset(("anthropic", "openai-codex", "factory-droid"))
CLAUDE_BOOTSTRAP = "https://api.anthropic.com/api/claude_cli/bootstrap"
CLAUDE_BETA = "oauth-2025-04-20"
DEAD_GRANT = re.compile(
    r"invalid_grant|invalid_token|unauthorized_client|\brevoked\b|refresh[\s_]?token.*expired|refresh_token_reused", re.I,
)
LEASE_MS = 15_000
RENEW_SECONDS = 5
CALLER_SECONDS = 10
EXCHANGE_SECONDS = 60
SAVE_SECONDS = 60
STALLED_MS = (EXCHANGE_SECONDS + SAVE_SECONDS) * 1000


class DeadGrant(AuthBrokerError):
    pass


def object_or_empty(value):
    return value if isinstance(value, dict) else {}


def parse_object(raw):
    try:
        return object_or_empty(json.loads(raw))
    except (TypeError, ValueError, UnicodeError):
        return {}


def text(value):
    return value.strip() if isinstance(value, str) and value.strip() else None


def can_refresh(provider, credential):
    return provider in GRANTS or (
        provider.startswith(("mcp_oauth:", "mcp_oauth_")) and bool(text(credential.get("tokenUrl")))
    )


def jwt_payload(token):
    if not isinstance(token, str) or token.count(".") != 2:
        return {}
    payload = token.split(".")[1]
    try:
        return parse_object(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    except (ValueError, UnicodeError):
        return {}


def fingerprint(token):
    return hashlib.sha256(token.encode()).digest()


def busy(error):
    return (getattr(error, "sqlite_errorcode", 0) & 0xff) in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED)


def credential_identifiers(credential):
    sources = [jwt_payload(credential.get(key)) for key in ("access", "refresh")]
    email = text(credential.get("email"))
    account = text(credential.get("accountId"))
    project, org = text(credential.get("projectId")), text(credential.get("orgId"))
    for claims in sources:
        profile = object_or_empty(claims.get("https://api.openai.com/profile"))
        auth = object_or_empty(claims.get("https://api.openai.com/auth"))
        email = email or text(claims.get("email")) or text(profile.get("email"))
        account = account or next((value for key in ("account_id", "accountId", "user_id", "sub")
                                   if (value := text(claims.get(key)))), None) or text(auth.get("chatgpt_account_id"))
    return {key: value for key, value in (("email", email.lower() if email else None), ("account", account),
                                          ("project", project), ("org", org)) if value}


def credential_identity(provider, credential):
    identifiers = credential_identifiers(credential)
    fields = ("email", "account", "project") if provider in ORG_SCOPED else ("account", "email", "project")
    base = next((f"{key}:{identifiers[key]}" for key in fields if key in identifiers), None)
    org = identifiers.get("org") if provider in ORG_SCOPED else None
    if org:
        return f"{base}|org:{org}" if base else f"org:{org}"
    return base


async def anthropic_identity(client, body, access):
    account = object_or_empty(body.get("account"))
    account_id, email = text(account.get("uuid")), text(account.get("email_address"))
    if account_id and email:
        return account_id, email
    try:
        response = await client.get(CLAUDE_BOOTSTRAP, params={"entrypoint": "cli", "model": "claude-opus-4-8"},
                                    headers={"Authorization": "Bearer " + access, "Anthropic-Beta": CLAUDE_BETA,
                                             "Accept": "application/json, text/plain, */*", "Content-Type": "application/json",
                                             "User-Agent": "claude-code/" + (os.getenv("PI_AI_CLAUDE_CODE_VERSION") or "2.1.280")},
                                    timeout=30)
        if response.is_success:
            bootstrap = object_or_empty(parse_object(response.content).get("oauth_account"))
            account_id = account_id or text(bootstrap.get("account_uuid"))
            email = email or text(bootstrap.get("account_email"))
    except httpx2.HTTPError:
        pass
    return account_id, email


def oauth_origin(value, *, secure=False):
    try:
        url = urlsplit(value)
        host = url.hostname
        if not host or url.username is not None or url.fragment:
            return None
        if url.scheme == "https":
            return url.scheme, host, url.port or 443
        if url.scheme == "http" and (not secure or host == "localhost" or ipaddress.ip_address(host).is_loopback):
            return url.scheme, host, url.port or 80
    except ValueError:
        pass
    return None


async def mcp_grant(provider, current):
    endpoint = current["tokenUrl"]
    if not oauth_origin(endpoint, secure=True):
        raise AuthBrokerError("MCP OAuth token endpoint must use HTTPS or HTTP loopback.")
    form = {"grant_type": "refresh_token", "refresh_token": current["refresh"]}
    client_id, secret = text(current.get("clientId")), text(current.get("clientSecret"))
    method = current.get("clientAuthMethod") or ("client_secret_post" if secret else "none")
    auth = None
    if method == "client_secret_basic" and client_id and secret:
        auth = httpx2.BasicAuth(quote_plus(client_id), quote_plus(secret))
    elif method in ("none", "client_secret_post"):
        if client_id:
            form["client_id"] = client_id
        if method == "client_secret_post" and secret:
            form["client_secret"] = secret
    else:
        raise AuthBrokerError("Unsupported MCP OAuth client authentication method.")

    fallback = provider.removeprefix("mcp_oauth:") if provider.startswith("mcp_oauth:") else ""
    if fallback.startswith("profile:"):
        fallback = fallback.removeprefix("profile:").partition(":")[2]
    resource = text(current.get("resource"))
    anchor = oauth_origin(text(current.get("authorizationUrl")) or endpoint)
    if resource and not oauth_origin(resource):
        raise AuthBrokerError("MCP OAuth resource must be an HTTP or HTTPS URL.")
    if resource or (fallback and oauth_origin(fallback) and oauth_origin(fallback) != anchor):
        form["resource"] = resource or fallback
    try:
        async with httpx2.AsyncClient(timeout=15, follow_redirects=False) as client:
            response = await client.post(endpoint, data=form, auth=auth, headers={"Accept": "application/json"})
    except httpx2.HTTPError:
        raise AuthBrokerError("MCP OAuth token endpoint is unavailable.") from None
    body = parse_object(response.content)
    if not response.is_success:
        detail = " ".join(str(body.get(key, "")) for key in ("error", "error_description", "message")).strip()
        if 400 <= response.status_code < 500 and DEAD_GRANT.search(detail or response.text):
            raise DeadGrant("MCP OAuth provider rejected the refresh grant.", response.status_code)
        raise AuthBrokerError(f"MCP OAuth token endpoint returned HTTP {response.status_code}.", response.status_code)
    access, seconds = text(body.get("access_token")), body.get("expires_in", 3600)
    if not access or not number(seconds) or not 0 < seconds < 10**10:
        raise AuthBrokerError("MCP OAuth token response is missing an access token or valid expiry.")
    renewed = {**current, "access": access, "refresh": text(body.get("refresh_token")) or current["refresh"],
               "expires": int((time() + seconds) * 1000)}
    if scope := text(body.get("scope")):
        renewed["scope"] = scope
    return renewed


async def refresh_grant(provider, current):
    if provider not in GRANTS:
        return await mcp_grant(provider, current)
    endpoint, client_id = GRANTS[provider]
    form = {"grant_type": "refresh_token", "refresh_token": current["refresh"], "client_id": client_id}
    if organization := text(current.get("activeOrganizationId")):
        form["organization_id"] = organization
    anthropic = provider == "anthropic"
    options = ({"json": form, "headers": {"Anthropic-Beta": CLAUDE_BETA,
                                         "User-Agent": "anthropic-sdk-typescript/0.112.1 userOAuthProvider"}}
               if anthropic else {"data": form})
    try:
        async with httpx2.AsyncClient(timeout=15, follow_redirects=False) as client:
            response = await client.post(endpoint, **options)
            body = parse_object(response.content)
            if not response.is_success:
                detail = " ".join(str(body.get(key, "")) for key in ("error", "error_description", "message")).strip()
                if 400 <= response.status_code < 500 and DEAD_GRANT.search(detail or response.text):
                    raise DeadGrant("OAuth provider rejected the refresh grant.", response.status_code)
                raise AuthBrokerError(f"OAuth token endpoint returned HTTP {response.status_code}.", response.status_code)
            access, seconds = text(body.get("access_token")), body.get("expires_in")
            if not access or not number(seconds) or not 0 < seconds < 10**10:
                raise AuthBrokerError("OAuth token response is missing an access token or valid expiry.")
            expires = int((time() + seconds - (300 if anthropic else 0)) * 1000)
            if anthropic:
                account_id, email = await anthropic_identity(client, body, access)
            else:
                access_claims, id_claims = jwt_payload(access), jwt_payload(body.get("id_token"))
                auth = [object_or_empty(claims.get("https://api.openai.com/auth")) for claims in (access_claims, id_claims)]
                profile = [object_or_empty(claims.get("https://api.openai.com/profile")) for claims in (access_claims, id_claims)]
                account_id = next((value for claims in auth if (value := text(claims.get("chatgpt_account_id")))), None)
                email = next((value.lower() for claims in profile if (value := text(claims.get("email")))), None)
    except httpx2.HTTPError:
        raise AuthBrokerError("OAuth token endpoint is unavailable.") from None
    renewed = {**current, "access": access, "refresh": text(body.get("refresh_token")) or current["refresh"], "expires": expires}
    if account_id:
        renewed["accountId"] = account_id
    if email:
        renewed["email"] = email
    return renewed


class LocalOAuth:
    def __init__(self, database, skew):
        self.database, self.skew = database, skew
        self.tasks = {}
        self.waiting = {}
        self.mints = {}
        self.failures = {}
        self.stalled = {}
        self.closing = False

    async def refresh(self, row, forced, *, reuse_recent=True, disable_dead=True):
        if self.closing:
            raise AuthBrokerError("OAuth refresh is shutting down.")
        identifier = row["broker_id"]
        key = identifier, fingerprint(row["tokens"]["access_token"])
        now = monotonic()
        self.failures = {key: until for key, until in self.failures.items() if until > now}
        if key in self.failures and (not forced or reuse_recent):
            raise AuthBrokerError("OAuth refresh is temporarily unavailable.")
        task = self.tasks.get(identifier)
        if task is None or task.done():
            stop_waiting = self.waiting[identifier] = asyncio.Event()
            task = self.tasks[identifier] = asyncio.create_task(
                self.run(row, forced, stop_waiting, reuse_recent, disable_dead),
            )

            def finished(done):
                if self.tasks.get(identifier) is done:
                    self.tasks.pop(identifier)
                    self.waiting.pop(identifier)
                if not done.cancelled() and done.exception() is not None:
                    self.failures[key] = monotonic() + 15

            task.add_done_callback(finished)
        else:
            stop_waiting = self.waiting[identifier]
            stop_waiting.clear()
        try:
            await asyncio.wait_for(asyncio.shield(task), CALLER_SECONDS)
        except asyncio.CancelledError:
            stop_waiting.set()
            raise
        except TimeoutError:
            stop_waiting.set()
            raise AuthBrokerError("Timed out waiting for local OAuth refresh.") from None
        except sqlite3.Error:
            raise AuthBrokerError("Oh My Pi OAuth credential storage is unavailable.") from None

    async def run(self, row, forced, stop_waiting, reuse_recent, disable_dead):
        identifier, provider = row["broker_id"], row["provider"]
        rejected = row["tokens"]["access_token"]
        observed_revision = row.get("credential_revision")
        owner = str(uuid4())
        deadline = monotonic() + CALLER_SECONDS
        db = self.database
        while not self.closing and not stop_waiting.is_set() and monotonic() < deadline:
            raw = await asyncio.to_thread(db.oauth, identifier, provider)
            current = parse_object(raw)
            access, expires = text(current.get("access")), current.get("expires")
            if not access or not number(expires):
                return
            now = time()
            recent = self.mints.get(identifier)
            newer = access != rejected or (observed_revision is not None and fingerprint(raw) != observed_revision)
            newly_minted = reuse_recent and recent and recent[0] == fingerprint(access) and recent[1] > monotonic()
            if expires > (now + self.skew) * 1000 and (
                not forced or newer or newly_minted
            ):
                return
            refresh = text(current.get("refresh"))
            if not can_refresh(provider, current) or not refresh or refresh == REMOTE_REFRESH:
                raise AuthBrokerError("This stored OAuth credential has no supported local refresh grant.")
            digest = fingerprint(refresh)
            stalled = self.stalled.get(identifier)
            if stalled and stalled[0] == digest and stalled[1] > now * 1000:
                await asyncio.sleep(.25)
                continue
            try:
                claimed, prior = await asyncio.to_thread(db.claim_refresh, identifier, owner, LEASE_MS)
            except sqlite3.Error as error:
                if not busy(error):
                    raise
                claimed, prior = False, None
            if not claimed:
                wait = .05 if prior is None else max(.05, min(.25, prior[1] / 1000 - time()))
                await asyncio.sleep(min(wait, max(0, deadline - monotonic())))
                continue
            recent_holder = prior if prior and prior[0] != owner and time() * 1000 - prior[1] < STALLED_MS else None
            handback = recent_holder
            try:
                if await asyncio.to_thread(db.oauth, identifier, provider) != raw:
                    continue
                if self.closing or stop_waiting.is_set():
                    return
                if monotonic() >= deadline:
                    break
                handback = None
                renew = asyncio.create_task(self.renew(identifier, owner))
                try:
                    async with asyncio.timeout(EXCHANGE_SECONDS):
                        renewed = await refresh_grant(provider, current)
                except TimeoutError:
                    raise AuthBrokerError("OAuth token exchange exceeded its deadline.") from None
                except DeadGrant:
                    if not disable_dead:
                        raise
                    if recent_holder:
                        handback = recent_holder
                        self.stalled[identifier] = digest, recent_holder[1] + STALLED_MS
                        raise AuthBrokerError("A previous OAuth refresher may still save a replacement grant.") from None
                    await self.persist(identifier, provider, owner, raw, disabled="oauth refresh failed: rejected grant")
                    return
                finally:
                    renew.cancel()
                    await asyncio.gather(renew, return_exceptions=True)
                if await self.persist(identifier, provider, owner, raw, renewed):
                    self.mints[identifier] = fingerprint(renewed["access"]), monotonic() + 300
                    self.stalled.pop(identifier, None)
                return
            finally:
                await self.leave(identifier, owner, handback)
        if not self.closing and not stop_waiting.is_set():
            raise AuthBrokerError("Timed out waiting for the shared OAuth refresh lease.")

    async def renew(self, identifier, owner):
        delay = RENEW_SECONDS
        while True:
            await asyncio.sleep(delay)
            try:
                if not await asyncio.to_thread(self.database.renew_refresh, identifier, owner, LEASE_MS):
                    return
                delay = RENEW_SECONDS
            except sqlite3.Error as error:
                if not busy(error):
                    raise
                delay = .05

    async def persist(self, identifier, provider, owner, raw, renewed=None, *, disabled=None):
        db = self.database
        replacement = json.dumps(renewed, ensure_ascii=False, separators=(",", ":")) if renewed is not None else None
        identity = credential_identity(provider, renewed) if renewed is not None else None
        delay, deadline = .05, monotonic() + SAVE_SECONDS
        while monotonic() < deadline:
            try:
                if await asyncio.to_thread(db.oauth, identifier, provider) != raw:
                    return False
                if not await asyncio.to_thread(db.renew_refresh, identifier, owner, LEASE_MS) and renewed is not None:
                    await asyncio.to_thread(db.claim_refresh, identifier, owner, LEASE_MS)
                if await asyncio.to_thread(db.commit_refresh, identifier, provider, owner, raw,
                                           replacement=replacement, identity=identity, disabled=disabled):
                    return True
                if renewed is None:
                    return False
            except sqlite3.Error as error:
                if not busy(error):
                    raise
            await asyncio.sleep(delay)
            delay = min(1, delay * 2)
        raise AuthBrokerError("OAuth refresh could not be saved before its deadline.")

    async def leave(self, identifier, owner, prior):
        delay, deadline = .05, monotonic() + SAVE_SECONDS
        while monotonic() < deadline:
            try:
                await asyncio.to_thread(self.database.leave_refresh, identifier, owner, prior)
                return
            except sqlite3.Error as error:
                if not busy(error) or asyncio.current_task().cancelling():
                    return
            await asyncio.sleep(delay)
            delay = min(1, delay * 2)

    async def close(self):
        self.closing = True
        if not self.tasks:
            return
        _, pending = await asyncio.wait(tuple(self.tasks.values()), timeout=EXCHANGE_SECONDS + SAVE_SECONDS + 5)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
