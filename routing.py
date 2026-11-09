"""Configured model routes, credentials, and session-scoped switching."""

import asyncio
import json
import math
import os
import sys
import time
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from functools import partial
from urllib.parse import urlsplit

import httpx
from mcp.client.auth import OAuthFlowError
from openai import OpenAI
from xxhash import xxh32_intdigest

from anthropic_fallback import validate_chain
from anthropic_messages import credit_header, headers as anthropic_headers, official_endpoint
from auth_broker import AuthBrokerError, ModelCredentialError, ModelQuotaError
from broker_usage import quota_scope, response_limits
from observed_usage import normalize_usage
from cache_warmer import Prices
from cancellable_http import CancellableTransport, active_turn
import codex_responses
from mcp_oauth import oauth_url
import model_catalog
from model_secrets import CONFIG_KEY, KEYLESS, ConfigRequestAuth, ConfigSecrets, origin
from responses_websocket import ResponsesSocket
from models_config import Connection, model_target
from rate_limits import UsageBlock


NO_INHERITED_EFFORT = object()


@dataclass(frozen=True)
class Route:
    name: str
    model: str
    window: int | None = None
    base_url: str | None = None
    key_envs: tuple[str, ...] = ("OPENAI_API_KEY",)
    promote_to: str | None = None
    fallbacks: tuple[str, ...] = ()
    reasoning_effort: str | None = None
    api: str = "chat-completions"
    auth: str = "api-key"
    max_output_tokens: int | None = 8192
    cache_retention: str = "none"
    native_compaction: bool = False
    cache_prices: Prices | None = None
    responses_compaction_v2: bool = False
    vision: bool = False
    image_max_edge: int | None = None
    image_count_budget: int = 5
    image_byte_budget: int = 3_000_000
    image_tokens: int | None = None
    image_detail: str | None = None
    image_read_results: bool | None = None
    broker_provider: str | None = None
    broker_accounts: frozenset | None = None
    broker_scope: str = ""
    key_override: bool = False
    replay_unsigned_thinking: bool | None = None
    thinking_prefix_mismatch: str | None = None
    fallback_credit: bool | None = None
    server_side_fallbacks: tuple[dict, ...] = field(default=(), hash=False)
    catalog_model: dict | None = field(default=None, repr=False, compare=False, hash=False)
    profile_base: str | None = field(default=None, repr=False, compare=False)
    connection: Connection | None = field(default=None, repr=False)
    metadata_pins: frozenset = field(default=frozenset(), repr=False, compare=False)
    prefer_websockets: bool = False

    def endpoint(self):
        if self.api == "anthropic-messages":
            return (self.base_url or os.getenv("ANTHROPIC_BASE_URL") or "https://api.anthropic.com").rstrip("/")
        if self.api == codex_responses.API:
            return (self.base_url or codex_responses.BASE_URL).rstrip("/")
        return (self.base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")

    @property
    def upstream(self):
        return model_catalog.upstream_pin(self.catalog_model)

    def identity(self):
        parts = [self.name, self.model, self.endpoint()]
        if self.api != "chat-completions":
            parts.extend((self.api, self.auth))
        if upstream := self.upstream:
            parts.append("@" + upstream)
        return json.dumps(parts, separators=(",", ":"))

    def quota_scope(self):
        return quota_scope(self.broker_provider, self.model, self.broker_scope)

    def provider(self):
        return self.broker_provider or self.endpoint()

    def effort(self, requested):
        effort = requested if self.reasoning_effort in (None, "auto") else self.reasoning_effort
        return model_catalog.thinking_effort(self.catalog_model, effort)

    def wire_model(self, effort):
        if self.catalog_model and self.model == self.catalog_model["id"]:
            return model_catalog.wire_model(self.catalog_model, effort)
        return self.model

    def prices(self, prompt=0, timestamp=None):
        return self.cache_prices or model_catalog.token_prices(self.catalog_model, prompt, timestamp)

    def fallback_credit_targets(self, model=None):
        if (
            self.api != "anthropic-messages" or self.fallback_credit is False
            or self.broker_provider != "anthropic" and not official_endpoint(self.endpoint())
        ):
            return ()
        policy = model_catalog.catalog_policy("anthropic", model or self.model)
        return tuple(policy.get("fallbackCreditTargets", ()))

    def can_redeem_credit(self, target, targets):
        return (
            target.api == "anthropic-messages" and target.provider() == self.provider()
            and target.endpoint() == self.endpoint() and target.model in targets and target.fallback_credit is not False
        )


def load_prices(name, value):
    required = {"input", "output", "read"}
    if not isinstance(value, dict) or not required <= value.keys() or value.keys() - (
        required | {"write_5m", "write_1h"}
    ):
        raise ValueError(f"Route {name!r} requires cache prices for input, output, and read.")
    try:
        valid = all(type(price) in (int, float) and price >= 0 and math.isfinite(price) for price in value.values())
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"Route {name!r} has an invalid cache price.")
    return Prices(**value)


def load_routes(path, primary, window, *, anthropic_server_side_fallback=False, context_promotion=False,
                allow_unmatched=False, configured_models=None, extended_context=False, allow_judge=False,
                responses_websockets=False, codex_websockets=None):
    registry = configured_models or model_catalog
    if path is None:
        if not model_catalog.select_routed(registry, primary):
            return {primary: Route(primary, primary, window)}
        raw = {primary: {"catalog": primary}}
    else:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict) or primary is not None and primary not in raw and not allow_unmatched:
        raise ValueError("Route configuration must contain the alias selected with --model.")
    routes = {}
    for name, fields in raw.items():
        if not isinstance(fields, dict) or set(fields) - {
            "model", "catalog", "context_window", "base_url", "api_key_env", "promote_to", "fallbacks", "reasoning_effort",
            "api", "auth", "max_output_tokens", "cache_retention", "native_compaction", "cache_prices",
            "responses_compaction_v2", "prefer_websockets",
            "vision", "image_max_edge", "image_count_budget", "image_byte_budget", "image_tokens", "image_detail",
            "image_read_results",
            "broker_provider", "broker_accounts", "broker_scope",
            "replay_unsigned_thinking", "thinking_prefix_mismatch", "fallback_credit", "server_side_fallbacks",
        }:
            raise ValueError(f"Invalid fields for route {name!r}.")
        catalog = {}
        if "catalog" in fields:
            selector = fields["catalog"]
            catalog = model_catalog.select_routed(registry, selector) if isinstance(selector, str) else None
            if catalog is None:
                raise ValueError(f"Route {name!r} references an unknown provider/model catalog row.")
        if (allow_judge and catalog.get("provider") == "typesafe" and catalog.get("api") == "typesafe"
            and (catalog.get("baseUrl") or "").rstrip("/") == "https://api.typesafe.ai"
            and (url := os.getenv("TYPESAFE_BASE_URL"))):
            catalog = {**catalog, "baseUrl": url}
        compat = catalog.get("compat") or {}
        policy = (model_catalog.catalog_policy(catalog["provider"], catalog["id"], model_catalog.upstream_pin(catalog))
                  if catalog else {})
        model = fields.get("model", catalog.get("id"))
        if transport := catalog.get("_transport"):
            raise ValueError(f"Route {name!r} requires the unsupported {transport!r} transport.")
        standard = catalog.get("contextWindow")
        maximum = catalog.get("maxContextWindow")
        if extended_context and type(standard) is int and type(maximum) is int:
            standard = max(standard, maximum)
        context_window = (window if name == primary and window is not None
                          else fields.get("context_window", standard))
        endpoint = fields.get("base_url", catalog.get("baseUrl"))
        catalog_api = catalog.get("api")
        default_api = model_catalog.APIS.get(catalog_api) if catalog else "chat-completions"
        if allow_judge and catalog_api in model_catalog.JUDGE_APIS:
            default_api = catalog_api
        api = fields.get("api", default_api)
        if api is None:
            raise ValueError(f"Route {name!r} requires the unsupported {catalog['api']!r} adapter.")
        catalog_url = catalog.get("baseUrl") or ""
        same_endpoint = isinstance(endpoint, str) and endpoint.rstrip("/") == catalog_url.rstrip("/")
        connection = catalog.get("_connection") if same_endpoint else None
        mode = connection.mode if connection else None
        auth = fields.get("auth", "bearer" if mode == "oauth" or api == codex_responses.API else "api-key")
        default_limit = catalog.get("maxTokens") if catalog and api == "chat-completions" else catalog.get("maxTokens") or 8192
        if api == "chat-completions" and catalog.get("omitMaxOutputTokens"):
            default_limit = None
        max_tokens = fields.get("max_output_tokens", default_limit)
        cache_retention = fields.get("cache_retention", "none")
        remote = (catalog.get("remoteCompaction") or {}) if same_endpoint else {}
        remote_enabled = remote.get("enabled")
        responses_remote = api in model_catalog.RESPONSES_APIS and remote.get("api", api) == api
        native_anthropic = api == "anthropic-messages" and compat.get("supportsServerCompaction") is True and (
            isinstance(endpoint, str) and official_endpoint(endpoint)
        )
        native_responses = responses_remote and (remote_enabled is True or remote.get("v2StreamingEnabled") is True)
        native_compaction = fields.get("native_compaction", remote_enabled is not False and (
            native_anthropic or native_responses
        ))
        responses_compaction_v2 = fields.get("responses_compaction_v2", native_compaction and responses_remote
                                            and remote.get("v2StreamingEnabled") is True)
        websocket_preference = catalog.get("preferWebsockets")
        prefer_websockets = fields.get("prefer_websockets", same_endpoint and api == "openai-responses"
                                       and policy.get("supportsWebsockets") is True and websocket_preference is not False
                                       and (responses_websockets or websocket_preference is True))
        if api == codex_responses.API:
            configured = fields.get("prefer_websockets", codex_websockets != "off")
            prefer_websockets = (codex_responses.websocket_enabled(catalog, configured)
                                 if type(configured) is bool else configured)
        vision = fields.get("vision", "image" in (catalog.get("input") or ()) and not compat.get("stripImageInput"))
        image_max_edge = fields.get("image_max_edge", compat.get("maxImageDimension"))
        official = api == "anthropic-messages" and isinstance(endpoint, str) and official_endpoint(endpoint)
        default_count = policy.get("officialImageCountBudget") if official else None
        image_count = fields.get("image_count_budget", catalog.get("imageCountBudget") or default_count
                                 or policy.get("imageCountBudget", 5))
        image_bytes = fields.get("image_byte_budget", (policy.get("inlineImageByteBudget") if official else None)
                                 or compat.get("maxImagePayloadBytes") or 3_000_000)
        image_tokens = fields.get("image_tokens")
        image_detail = fields.get("image_detail")
        image_reads = fields.get("image_read_results")
        cache_prices = load_prices(name, fields["cache_prices"]) if "cache_prices" in fields else None
        default_env = "OPENAI_API_KEY"
        if api == "anthropic-messages":
            default_env = "ANTHROPIC_AUTH_TOKEN" if auth == "bearer" else "ANTHROPIC_API_KEY"
        elif api == codex_responses.API:
            default_env = "OPENAI_CODEX_OAUTH_TOKEN"
        provider = fields.get("broker_provider")
        if provider is None and mode == "oauth" and not (connection.key or fields.get("api_key_env")):
            provider = catalog["provider"]
        if (provider is None and api == codex_responses.API and same_endpoint and not connection
            and catalog.get("provider") == "openai-codex" and "api_key_env" not in fields):
            provider = "openai-codex"
        accounts = fields.get("broker_accounts")
        scope = fields.get("broker_scope", "")
        catalog_envs = model_catalog.provider_info(catalog["provider"]).get("envVars", []) if catalog and same_endpoint else []
        if catalog_envs and api == "anthropic-messages" and auth == "bearer" and official_endpoint(endpoint):
            catalog_envs = [default_env]
        key_envs = fields.get("api_key_env", catalog_envs or ([default_env] if endpoint is None else []))
        promote_to = fields.get("promote_to")
        fallbacks = fields.get("fallbacks", [])
        effort = fields.get("reasoning_effort")
        if isinstance(key_envs, str):
            key_envs = [key_envs]
        if not name.strip() or not isinstance(model, str) or not model.strip():
            raise ValueError(f"Route {name!r} requires a model ID.")
        if context_window is not None and (type(context_window) is not int or context_window < 128):
            raise ValueError(f"Route {name!r} requires a valid context window.")
        if not isinstance(key_envs, list) or not (key_envs or provider or connection and (connection.key or mode == "none")) or not all(
            isinstance(key, str) and key.isascii() and key.isidentifier() for key in key_envs
        ):
            raise ValueError(f"Route {name!r} requires an API key environment variable list.")
        if endpoint is not None:
            parsed = urlsplit(endpoint) if isinstance(endpoint, str) else None
            if not parsed or parsed.scheme not in ("https", "http") or not parsed.hostname or (
                parsed.username is not None or parsed.query or parsed.fragment
            ):
                raise ValueError(f"Route {name!r} requires an HTTP(S) base URL without credentials.")
        if native_compaction and responses_remote:
            for key in ("endpoint", "v2Endpoint", "streamingEndpoint"):
                if key in remote:
                    url = httpx.URL(remote[key])
                    if not url.is_absolute_url or url.userinfo or url.fragment or origin(url) != origin(httpx.URL(endpoint)):
                        raise ValueError(f"Route {name!r} requires same-origin remote compaction URLs.")
        if provider is not None and (not isinstance(provider, str) or not provider or provider != provider.strip()
            or provider.startswith(("mcp_oauth:", "mcp_oauth_"))):
            raise ValueError(f"Route {name!r} requires a model broker provider ID.")
        if ("broker_accounts" in fields and (not provider or not isinstance(accounts, list) or not all(
            isinstance(value, str) and value and value == value.strip() for value in accounts
        ))) or not isinstance(scope, str) or scope != scope.strip() or "broker_scope" in fields and not provider:
            raise ValueError(f"Route {name!r} has invalid broker account settings.")
        key_override = "api_key_env" in fields and bool(key_envs)
        replay_unsigned = fields.get("replay_unsigned_thinking")
        prefix_mismatch = fields.get("thinking_prefix_mismatch")
        if api == "anthropic-messages" and isinstance(endpoint, str) and official_endpoint(endpoint):
            if "replay_unsigned_thinking" not in fields:
                replay_unsigned = compat.get("replayUnsignedThinking")
            if "thinking_prefix_mismatch" not in fields and compat.get("supportsThinkingBindingControls") and (
                (catalog.get("thinking") or {}).get("prefixBinding")
            ):
                prefix_mismatch = "drop_block"
        fallback_credit = fields.get("fallback_credit")
        thinking_fields = {"replay_unsigned_thinking", "thinking_prefix_mismatch", "fallback_credit",
                           "server_side_fallbacks"}
        if (
            ("replay_unsigned_thinking" in fields and type(replay_unsigned) is not bool)
            or ("thinking_prefix_mismatch" in fields and prefix_mismatch not in ("drop_block", "error"))
            or ("fallback_credit" in fields and type(fallback_credit) is not bool)
            or (api != "anthropic-messages" and thinking_fields & fields.keys())
        ):
            raise ValueError(f"Route {name!r} has invalid Anthropic thinking compatibility settings.")
        if key_override and accounts is not None:
            raise ValueError(f"Route {name!r} cannot combine an explicit API key with an OAuth account pool.")
        if promote_to is not None and (not isinstance(promote_to, str) or promote_to not in raw):
            raise ValueError(f"Route {name!r} references an unknown promotion target.")
        if not isinstance(fallbacks, list) or not all(
            isinstance(target, str) and target in raw for target in fallbacks
        ):
            raise ValueError(f"Route {name!r} references an unknown fallback route.")
        if effort is not None and (not isinstance(effort, str) or not effort.strip()):
            raise ValueError(f"Route {name!r} requires a nonempty reasoning effort.")
        if (
            type(vision) is not bool
            or image_max_edge is not None and (type(image_max_edge) is not int or not 128 <= image_max_edge <= 16_384)
            or type(image_count) is not int or not 1 <= image_count <= 10_000
            or type(image_bytes) is not int or image_bytes < 1
            or image_tokens is not None and (type(image_tokens) is not int or image_tokens < 1)
            or image_detail not in (None, "auto", "low", "high", "original")
            or "image_read_results" in fields and type(image_reads) is not bool
        ):
            raise ValueError(f"Route {name!r} has invalid vision or image budget settings.")
        if (
            api not in model_catalog.APIS.values() and not (allow_judge and api in model_catalog.JUDGE_APIS)
            or auth not in ("api-key", "bearer")
            or (not (max_tokens is None and catalog and api == "chat-completions")
                and (type(max_tokens) is not int or max_tokens < 1))
            or cache_retention not in ("none", "5m", "1h")
            or type(native_compaction) is not bool
            or type(responses_compaction_v2) is not bool
            or type(prefer_websockets) is not bool
            or "prefer_websockets" in fields and api not in model_catalog.RESPONSES_APIS
            or "responses_compaction_v2" in fields and api not in model_catalog.RESPONSES_APIS
            or responses_compaction_v2 and not native_compaction
            or api != "anthropic-messages" and {"auth", "cache_retention"} & fields.keys()
            or api == "chat-completions" and (
                "native_compaction" in fields or "max_output_tokens" in fields and not catalog
            )
        ):
            raise ValueError(f"Route {name!r} has invalid API settings.")
        chain = validate_chain(fields["server_side_fallbacks"], model, name) if "server_side_fallbacks" in fields else ()
        route = Route(
            name, model, context_window, endpoint, tuple(key_envs), promote_to, tuple(fallbacks), effort,
            api, auth, max_tokens, cache_retention, native_compaction, cache_prices, responses_compaction_v2,
            vision, image_max_edge, image_count, image_bytes, image_tokens, image_detail, image_reads,
            provider, frozenset(accounts) if accounts is not None else None, scope, key_override,
            replay_unsigned, prefix_mismatch, fallback_credit, chain, catalog or None, connection=connection,
            prefer_websockets=prefer_websockets,
            metadata_pins=frozenset(key for field_name, key in (
                ("context_window", "contextWindow"), ("max_output_tokens", "maxTokens"), ("vision", "input"),
            ) if field_name in fields) | (
                {"contextWindow"} if (name == primary and window is not None)
                or (extended_context and maximum is not None and context_window == maximum) else set()
            ),
        )
        if (
            anthropic_server_side_fallback and "server_side_fallbacks" not in fields and api == "anthropic-messages"
            and official_endpoint(route.endpoint())
        ):
            fallback_policy = model_catalog.catalog_policy("anthropic", model)
            targets = fallback_policy.get("serverSideFallbackModels", ())
            route = replace(route, server_side_fallbacks=tuple({"model": target} for target in targets))
        routes[name] = route
        if provider:
            try:
                oauth_url(routes[name].endpoint())
            except (OAuthFlowError, ValueError) as error:
                raise ValueError(f"Broker route {name!r} requires HTTPS or HTTP loopback.") from error
    for route in routes.values():
        if route.promote_to:
            target = routes[route.promote_to]
            if not route.window or not target.window or target.window <= route.window:
                raise ValueError(f"Promotion target for route {route.name!r} must have a larger known context window.")
            if target.model == route.model and target.endpoint() == route.endpoint() and target.upstream == route.upstream:
                raise ValueError(f"Promotion target for route {route.name!r} must select a different deployment.")
    if context_promotion:
        for name, route in tuple(routes.items()):
            if "promote_to" in raw[name] or not route.catalog_model or not route.window:
                continue
            target = model_target(registry, route.catalog_model["provider"], route.catalog_model.get("contextPromotionTarget"))
            if not target:
                continue
            matched = next((candidate for candidate in routes.values() if candidate.catalog_model
                            and f"{candidate.catalog_model['provider']}/{candidate.catalog_model['id']}" == target), None)
            if matched is None and "base_url" not in raw[name]:
                entry = registry.select(target)
                if entry and entry.get("api") in model_catalog.APIS and not entry.get("_transport") and target not in routes:
                    matched = load_routes(
                        None, target, None, anthropic_server_side_fallback=anthropic_server_side_fallback,
                        configured_models=configured_models, extended_context=extended_context,
                        responses_websockets=responses_websockets, codex_websockets=codex_websockets,
                    )[target]
            if matched and matched.window and matched.window > route.window and matched.name != name:
                routes[matched.name] = matched
                routes[name] = replace(route, promote_to=matched.name)
    return routes


class BrokerModelAuth(httpx.Auth):
    requires_request_body = True

    def __init__(self, broker, loop, route, identifier, secrets=None):
        self.broker, self.loop, self.route, self.identifier = broker, loop, route, identifier
        self.endpoint = httpx.URL(route.endpoint())
        self.secrets = secrets

    async def row(self, observed=None):
        route = self.route
        rows, _ = await self.broker.model_rows(route.broker_provider, route.quota_scope(), route.broker_accounts, route.model)
        row = next((row for row in rows if row["broker_id"] == self.identifier), None)
        if row is None:
            raise ModelCredentialError(f"Stored credential is unavailable for route {route.name!r}.")
        if row["kind"] == "oauth" and (observed is not None or row["expires"] <= time.time() + self.broker.refresh_skew):
            if observed is None or (
                row["tokens"] == observed["tokens"] and row.get("credential_revision") == observed.get("credential_revision")
            ):
                try:
                    row = await self.broker.refresh(row, observed is not None)
                except AuthBrokerError:
                    if row["expires"] <= time.time():
                        raise ModelCredentialError(f"OAuth refresh failed for route {route.name!r}.") from None
            if not row or row["expires"] <= time.time() or (
                route.broker_accounts is not None and row["identity"] not in route.broker_accounts
            ):
                raise ModelCredentialError(f"OAuth credential has expired or been removed for route {route.name!r}.")
        return row

    def authorize(self, request, row):
        url = request.url
        if (url.scheme, url.host, url.port) != (self.endpoint.scheme, self.endpoint.host, self.endpoint.port):
            raise ModelCredentialError("Broker model request changed origin.")
        key = row["tokens"]["access_token"]
        request.headers.pop("Authorization", None)
        request.headers.pop("x-api-key", None)
        if self.route.api == "anthropic-messages" and row["kind"] == "api_key" and self.route.auth == "api-key":
            request.headers["x-api-key"] = key
            if self.route.fallback_credit is not False and official_endpoint(self.route.endpoint()):
                request.headers["anthropic-beta"] = credit_header(request.headers.get("anthropic-beta"))
        else:
            request.headers["Authorization"] = "Bearer " + key
        if self.secrets and self.route.connection:
            request.headers.update(self.secrets.headers(self.route, None))
        if self.route.api == codex_responses.API:
            codex_responses.account_headers(request.headers, key)

    def observation(self, response, row):
        if row["kind"] == "oauth":
            limits = response_limits(row["provider"], response.headers, response.status_code)
            if limits:
                return self.broker.note_usage(row, limits)
        return None

    def sync_auth_flow(self, request):
        request.read()
        row = self.loop.call(self.row(), active_turn())
        self.authorize(request, row)
        response = yield request
        if observation := self.observation(response, row):
            self.loop.call(observation, active_turn())
        if response.status_code == 401 and row["kind"] == "oauth":
            try:
                renewed = self.loop.call(self.row(row), active_turn())
            except (AuthBrokerError, ModelCredentialError):
                return
            if renewed["tokens"] != row["tokens"]:
                self.authorize(request, renewed)
                response = yield request
                if observation := self.observation(response, renewed):
                    self.loop.call(observation, active_turn())

    async def async_auth_flow(self, request):
        async def run(coroutine):
            future = asyncio.run_coroutine_threadsafe(coroutine, self.loop.loop)
            return await asyncio.wrap_future(future)

        await request.aread()
        row = await run(self.row())
        await asyncio.to_thread(self.authorize, request, row)
        response = yield request
        if observation := self.observation(response, row):
            await run(observation)
        if response.status_code == 401 and row["kind"] == "oauth":
            try:
                renewed = await run(self.row(row))
            except (AuthBrokerError, ModelCredentialError):
                return
            if renewed["tokens"] != row["tokens"]:
                await asyncio.to_thread(self.authorize, request, renewed)
                response = yield request
                if observation := self.observation(response, renewed):
                    await run(observation)


class Router:
    def __init__(self, routes, primary, session, fallback_on, clients_stack, transport_loop=None, broker=None,
                 *, profile=None, primary_candidates=(), role_hint=None, secrets=None, fixed_primary=False):
        self.routes = routes
        self.primary = routes[primary]
        self.session = session
        self.fallback_on = fallback_on
        self.fixed_primary = fixed_primary
        self.clients = {}
        self.responses_sockets = {}
        self.codex_identity = None
        self.blocked = {}
        self.local_quota = {}
        self.quota_blocked = set()
        self.client_stack = clients_stack.enter_context(ExitStack())
        clients_stack.callback(self.close_responses_sockets)
        self.transport_loop = transport_loop
        self.broker = broker
        self.secrets = secrets or ConfigSecrets(None)
        self.pins = {}
        self.pin_used = {}
        self.profile = profile
        self.primary_candidates = primary_candidates or (primary,)
        self.role_hint = role_hint
        self.pinned_chain = None
        self.fallback_origin = None
        self.refusal_pinned = False
        self.cooldowns = {}
        self.inherited_effort = NO_INHERITED_EFFORT
        self.thinking = lambda: profile.default_effort if profile and self.route.catalog_model else None
        self.restore()

    def restore(self):
        self.pinned_chain = self.fallback_origin = None
        self.clear_inherited_effort()
        self.refusal_pinned = False
        primaries = [self.routes[name] for name in self.primary_candidates]
        if not self.fixed_primary:
            for primary in primaries:
                restored = self.session.last_route(primary.identity())
                selected = next((route for route in self.routes.values() if route.identity() == restored), None)
                if selected and (credential := self.available(selected)) is not None:
                    self.primary, self.route, self.credential = primary, selected, credential
                    return
        for primary in primaries:
            if (credential := self.available(primary)) is not None:
                self.primary, self.route, self.credential = primary, primary, credential
                return
        self.primary = self.route = primaries[0]
        self.credential = None
        if not self.fallback(set(), usage_limit=self.route.name in self.quota_blocked, cooldown=0):
            raise ValueError(f"No credential available for route {self.route.name!r} or its eligible fallbacks.")

    def requested_effort(self):
        return self.thinking() if self.inherited_effort is NO_INHERITED_EFFORT else self.inherited_effort

    def clear_inherited_effort(self):
        self.inherited_effort = NO_INHERITED_EFFORT

    def maybe_revert(self):
        origin = self.fallback_origin
        if self.profile and self.profile.revert and origin and not self.refusal_pinned:
            if time.monotonic() >= self.cooldowns.get(origin.name, 0) and self.switch(origin, "model fallback restored"):
                self.pinned_chain = self.fallback_origin = None
                self.clear_inherited_effort()

    def available(self, route, excluded=(), *, track=True):
        if track:
            self.quota_blocked.discard(route.name)
        now = time.monotonic()
        wall = time.time() * 1000
        keyless = route.connection is not None and route.connection.mode == "none" and not route.key_override
        placeholders = model_catalog.local_key_placeholders() if keyless else ()
        env = next((
            name for name in route.key_envs
            if name not in excluded and (value := os.getenv(name)) and value not in placeholders
            and not self.locally_blocked(route, name, now, wall)
        ), None) if route.broker_accounts is None else None
        if route.connection and route.broker_accounts is None and not route.key_override:
            if (CONFIG_KEY not in excluded and not self.locally_blocked(route, CONFIG_KEY, now, wall)
                and (key := self.secrets.resolve(route.connection.key)) and key not in placeholders):
                return CONFIG_KEY
            if keyless and not env and KEYLESS not in excluded and not self.locally_blocked(
                route, KEYLESS, now, wall,
            ):
                env = KEYLESS
        if not route.broker_provider or not self.broker or route.key_override:
            return env
        try:
            rows, quota_blocked = self.transport_loop.call(self.broker.model_rows(
                route.broker_provider, route.quota_scope(), route.broker_accounts, route.model, heal=True,
            ), active_turn())
        except AuthBrokerError:
            if env:
                return env
            raise
        rows = [row for row in rows if row["broker_id"] not in excluded
                and not self.locally_blocked(route, row["broker_id"], now, wall)]
        if quota_blocked and track:
            self.quota_blocked.add(route.name)
        pinned = self.pins.get(route.broker_provider)
        oauth = [row for row in rows if row["kind"] == "oauth"]
        if oauth:
            index = xxh32_intdigest(self.session.transcript_id.encode()) % len(oauth)
            warm = route.broker_provider != "anthropic" or now - self.pin_used.get(route.broker_provider, now) < 3600
            chosen = self.transport_loop.call(self.broker.choose_oauth(
                route.broker_provider, route.model, oauth[index:] + oauth[:index], pinned, warm,
                route.broker_accounts is not None, route.quota_scope() if track else None,
            ), active_turn())
            if chosen is not None:
                return chosen
            if track:
                self.quota_blocked.add(route.name)
        login = [row for row in rows if row.get("source") == "login"]
        for choices in (login, [] if env else [row for row in rows if row["kind"] == "api_key"]):
            if choices:
                index = xxh32_intdigest(self.session.transcript_id.encode()) % len(choices)
                return next((row["broker_id"] for row in choices if row["broker_id"] == pinned), choices[index]["broker_id"])
        return env

    def locally_blocked(self, route, credential, monotonic, wall):
        slot = route.name, credential
        return self.blocked.get(slot, 0) > monotonic or self.local_quota.get(slot, (0, False))[0] > wall

    def prepare(self):
        slot = self.route.name, self.credential
        now = time.time() * 1000
        if not self.route.broker_provider and not self.locally_blocked(self.route, self.credential, time.monotonic(), now):
            return
        try:
            selected = self.available(self.route)
        except AuthBrokerError as error:
            raise ModelCredentialError(str(error)) from None
        if selected is None:
            if self.route.name in self.quota_blocked or self.local_quota.get(slot, (0, False))[0] > now:
                raise ModelQuotaError(f"Broker accounts have reached their quota for route {self.route.name!r}.")
            raise ModelCredentialError(f"No credential is available for route {self.route.name!r}.")
        self.credential = selected
        if type(selected) is int:
            self.pins[self.route.broker_provider] = selected
            self.pin_used[self.route.broker_provider] = time.monotonic()
        else:
            self.pins.pop(self.route.broker_provider, None)

    def request_settings(self, route=None, credential=None):
        route, credential = (self.route, self.credential) if route is None else (route, credential)
        auth = None
        if type(credential) is int:
            auth = BrokerModelAuth(self.broker, self.transport_loop, route, credential, self.secrets)
        elif route.connection:
            auth = ConfigRequestAuth(route, credential, self.secrets)
        if auth:
            return {"auth": auth, "follow_redirects": False}
        return {"follow_redirects": False} if route.api == codex_responses.API else {}

    def usage_observer(self):
        route = self.route
        if not self.broker or self.broker.observed_usage is None or not route.broker_provider:
            return None

        def observe(at, raw, cached_tokens_header=None, *, iteration_cost=None):
            counters = normalize_usage(
                route.api, route.broker_provider, raw, partial(route.prices, timestamp=at),
                route.cache_retention, cached_tokens_header,
            )
            if counters is not None:
                if iteration_cost is not None:
                    counters["costUsd"] = iteration_cost["total"]
                self.transport_loop.loop.call_soon_threadsafe(
                    self.broker.observed_usage.record, route.broker_provider, route.model, at, counters,
                )

        return observe

    def client(self):
        route, credential = self.route, self.credential
        cache_key = (route.name, credential)
        if cache_key not in self.clients:
            self.clients[cache_key] = self.client_stack.enter_context(self.new_client(route, credential))
        return self.clients[cache_key]

    def reset_provider_state(self):
        count = len(self.clients) + self.close_responses_sockets()
        self.clients.clear()
        self.codex_identity = None
        self.pins.clear()
        self.pin_used.clear()
        self.client_stack.close()
        return count

    def close_responses_sockets(self):
        sockets, self.responses_sockets = self.responses_sockets, {}
        count = sum(socket.connection is not None for socket in sockets.values())
        for socket in sockets.values():
            socket.close()
        return count

    def responses_socket(self, route=None, credential=None, *, lite=None, model=None):
        route, credential = (self.route, self.credential) if route is None else (route, credential)
        if not route.prefer_websockets or not self.transport_loop:
            return None
        key = route.identity(), credential, self.session.transcript_id
        if route.api == codex_responses.API:
            lite = codex_responses.lite_enabled(route) if lite is None else lite
            model = model or (route.catalog_model or {}).get("requestModelId") or route.model
            key += (lite, model)
            if key not in self.responses_sockets:
                self.responses_sockets[key] = codex_responses.CodexSocket(
                    self.transport_loop, route.endpoint(), partial(self.responses_headers, route, credential),
                )
            return self.responses_sockets[key]
        if key not in self.responses_sockets:
            compat = (route.catalog_model or {}).get("compat") or {}
            setting = os.getenv("PI_OPENAI_STATEFUL", "").strip().lower()
            chain = setting not in ("0", "false", "off", "no") if setting else compat.get("statefulResponses") is not False
            self.responses_sockets[key] = ResponsesSocket(self.transport_loop, route.endpoint(),
                                                          partial(self.responses_headers, route, credential), chain=chain)
        return self.responses_sockets[key]

    def codex_turn(self):
        if self.route.api != codex_responses.API:
            return None
        return self.codex_session(self.session.transcript_id).turn((self.route.identity(), self.credential))

    def codex_session(self, session_id):
        if session_id != self.session.transcript_id:
            return codex_responses.SessionIdentity(session_id)
        if self.codex_identity is None or self.codex_identity.session_id != session_id:
            self.codex_identity = codex_responses.SessionIdentity(session_id)
        return self.codex_identity

    def responses_headers(self, route, credential):
        auth = self.request_settings(route, credential).get("auth")
        if isinstance(auth, BrokerModelAuth):
            row = self.transport_loop.call(auth.row(), active_turn())
            request = httpx.Request("GET", route.endpoint())
            auth.authorize(request, row)
            return [(key, value) for key, value in request.headers.items() if key != "host"]
        if isinstance(auth, ConfigRequestAuth):
            return list(auth.resolved_headers().items())
        key = os.environ[credential]
        headers = httpx.Headers({"Authorization": "Bearer " + key})
        if route.api == codex_responses.API:
            codex_responses.account_headers(headers, key)
        return list(headers.items())

    def new_client(self, route, credential, timeout=120):
        transport = {"transport": CancellableTransport(self.transport_loop)} if self.transport_loop else {}
        http_settings = self.request_settings(route, credential)
        if route.api == "anthropic-messages":
            return httpx.Client(headers=self.native_headers(route, credential), timeout=timeout, **transport, **http_settings)
        http_client = {"http_client": httpx.Client(timeout=timeout, **transport, **({"follow_redirects": True} | http_settings))
                       } if transport or http_settings else {}
        return OpenAI(**self.client_settings(route, credential), timeout=timeout, **http_client)

    def native_headers(self, route=None, credential=None):
        route, credential = (self.route, self.credential) if route is None else (route, credential)
        if type(credential) is int or route.connection:
            return {"anthropic-version": "2023-06-01"}
        return anthropic_headers(os.environ[credential], route.auth)

    def client_settings(self, route=None, credential=None):
        route, credential = (self.route, self.credential) if route is None else (route, credential)
        managed = type(credential) is int or route.connection is not None
        key = "managed-placeholder" if managed else os.environ[credential]
        settings = {"base_url": route.endpoint() if managed or route.api == codex_responses.API else route.base_url,
                    "api_key": key, "max_retries": 0}
        if route.api == codex_responses.API and not managed:
            headers = httpx.Headers()
            codex_responses.account_headers(headers, key)
            settings["default_headers"] = dict(headers)
        return settings

    def switch(self, target, reason):
        credential = self.available(target)
        if credential is None or target == self.route:
            return False
        previous = self.route
        self.session.record_route(self.primary.identity(), target.identity(), reason)
        self.route, self.credential = target, credential
        print(f"[{reason}: {previous.name} → {target.name}]", file=sys.stderr)
        return True

    def update_runtime_metadata(self, metadata):
        current = self.route
        if not metadata or not current.catalog_model:
            return
        provider = current.catalog_model["provider"]
        for name, route in tuple(self.routes.items()):
            model = route.catalog_model or {}
            if model.get("provider") != provider or route.model != current.model or route.endpoint() != current.endpoint():
                continue
            pins = route.metadata_pins | set(model.get("_metadata_pins", ()))
            updates = {key: value for key, value in metadata.items() if key not in pins}
            window = updates.get("contextWindow", route.window)
            if window is not None and window < 128:
                window = route.window
                updates.pop("contextWindow", None)
            output = updates.get("maxTokens", route.max_output_tokens)
            if output is not None and window is not None and "maxTokens" in updates:
                output = updates["maxTokens"] = min(output, window)
            vision = route.vision
            if "input" in updates:
                vision = "image" in updates["input"] and not (model.get("compat") or {}).get("stripImageInput")
            updated = replace(route, window=window, max_output_tokens=output, vision=vision,
                              catalog_model={**model, **updates})
            self.routes[name] = updated
            if name == self.route.name:
                self.route = updated
            if name == self.primary.name:
                self.primary = updated

    def promote(self, visited):
        target = self.route.promote_to
        if self.fixed_primary or not target or target in visited:
            return False
        candidate = self.routes[target]
        if not self.route.window or not candidate.window or candidate.window <= self.route.window:
            return False
        previous = self.route.name
        if not self.switch(candidate, "context promotion"):
            return False
        visited.update((previous, target))
        return True

    def rotate(self, delay, attempted):
        broker = type(self.credential) is int
        if broker and math.isfinite(delay):
            self.transport_loop.call(self.broker.block_model(
                self.credential, self.route.broker_provider, self.route.quota_scope(), delay,
            ), active_turn())
        else:
            self.blocked[(self.route.name, self.credential)] = time.monotonic() + (60 if broker else delay)
        return self.select_alternate(attempted)

    def select_alternate(self, attempted):
        attempted.add(self.credential)
        alternate = self.available(self.route, excluded=attempted)
        if alternate is None:
            return False
        self.credential = alternate
        if type(alternate) is int:
            self.pins[self.route.broker_provider] = alternate
        print(f"[credential rotation: {self.route.name}]", file=sys.stderr)
        return True

    def mark_quota(self, deadline, timed):
        route, identifier = self.route, self.credential
        if type(identifier) is int and self.broker:
            try:
                return self.transport_loop.call(self.broker.mark_usage_limit(
                    identifier, route.broker_provider, route.quota_scope(), route.broker_accounts, route.model,
                    deadline, timed,
                ), active_turn())
            except AuthBrokerError:
                pass
        key = route.name, identifier
        now = time.time() * 1000
        prior, prior_timed = self.local_quota.get(key, (0, False))
        actual = max(deadline, prior)
        self.local_quota[key] = actual, (timed if actual > prior else prior_timed or timed and deadline == prior)
        return UsageBlock(deadline, actual, prior_timed=prior if prior_timed and prior > now else None)

    def quota_availability(self):
        route, active = self.route, self.credential
        candidates = {}
        if route.broker_provider and self.broker and not route.key_override:
            try:
                candidates = self.transport_loop.call(self.broker.usage_candidates(
                    route.broker_provider, route.quota_scope(), route.broker_accounts, route.model, active,
                ), active_turn())
            except AuthBrokerError:
                pass
        if type(active) is not int and route.broker_accounts is None:
            candidates.update((key, (None, None)) for key in route.key_envs if os.getenv(key) and key not in candidates)
            if active is CONFIG_KEY or active is KEYLESS:
                candidates[active] = None, None
        now, monotonic = time.time() * 1000, time.monotonic()
        for key, (until, timed) in candidates.items():
            slot = route.name, key
            local = self.blocked.get(slot, 0)
            if local > monotonic:
                until = max(until or 0, now + (local - monotonic) * 1000)
            quota, provider_timed = self.local_quota.get(slot, (0, False))
            if quota > now:
                until = max(until or 0, quota)
                if provider_timed:
                    timed = max(timed or 0, quota)
            candidates[key] = until, timed
        current, timed = candidates.pop(active, (None, None))
        sibling = min((until if until is not None else now for until, _ in candidates.values()), default=None)
        return current, timed, sibling

    def chain(self, start):
        found, queue = [], list(start.fallbacks)
        seen = {start.name}
        for name in queue:
            if name not in seen:
                seen.add(name)
                found.append(name)
                queue.extend(self.routes[name].fallbacks)
        return found

    def fallback(self, visited, usage_limit, defer_anthropic=None, *, credit_targets=(), cooldown=None, pinned=False,
                 context_tokens=None):
        if self.fallback_on == "off" or (
            self.fallback_on == "usage-limit" and not usage_limit
        ) or (self.fallback_on == "except-usage-limit" and usage_limit):
            return False
        direct = self.chain(self.route)
        inherited = self.chain(self.primary)
        if self.route.name in inherited:
            direct.extend(inherited[inherited.index(self.route.name) + 1:])
        key = None
        effort = None
        profile_candidates = []
        if self.profile and self.profile.chains:
            effort = self.route.effort(self.requested_effort())
            key = self.profile.chain_key(self.route, effort, self.role_hint)
            for chain in dict.fromkeys((self.pinned_chain, key)):
                if chain:
                    profile_candidates.extend(self.profile.candidates(chain, self.route, effort))
        direct.extend(profile_candidates)
        source = self.route
        visited.add(self.route.name)
        for name in direct:
            candidate = self.routes[name]
            if defer_anthropic and candidate.api == "anthropic-messages" and (
                candidate.provider() == defer_anthropic.provider() and candidate.model != defer_anthropic.model
                and not defer_anthropic.can_redeem_credit(candidate, credit_targets)
            ):
                continue
            if name in visited or time.monotonic() < self.cooldowns.get(name, 0):
                continue
            if name in profile_candidates and not self.profile.fits(candidate, context_tokens):
                continue
            if self.profile and key and (
                candidate.provider(), candidate.model, candidate.api, candidate.endpoint(), candidate.upstream, candidate.effort(effort)
            ) == (
                source.provider(), source.model, source.api, source.endpoint(), source.upstream, effort
            ):
                continue
            if self.switch(candidate, "model fallback"):
                visited.add(name)
                if name in profile_candidates or self.pinned_chain:
                    self.pinned_chain = self.pinned_chain or key
                    self.fallback_origin = self.fallback_origin or source
                    self.refusal_pinned = self.refusal_pinned or pinned
                    if not pinned:
                        self.cooldowns[source.name] = time.monotonic() + (300 if cooldown is None else cooldown)
                    self.inherited_effort = effort
                return True
        return False
