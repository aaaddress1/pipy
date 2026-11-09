"""Live subscription quota probes for Claude and Codex."""

import asyncio
import math
import os
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from time import time

import httpx2

from auth_broker_cache import number
from broker_usage import HOUR, WEEK, header_limit
from local_oauth import jwt_payload, object_or_empty, parse_object, text


CLAUDE_URL = "https://api.anthropic.com/api/oauth"
CODEX_URL = "https://chatgpt.com/backend-api"
CLAUDE_BETAS = (
    "claude-code-20250219,oauth-2025-04-20,interleaved-thinking-2025-05-14,redact-thinking-2026-02-12,"
    "context-management-2025-06-27,prompt-caching-scope-2026-01-05,mid-conversation-system-2026-04-07,"
    "advanced-tool-use-2025-11-20,effort-2025-11-24,extended-cache-ttl-2025-04-11"
)
AUTH_CLAIM = "https://api.openai.com/auth"
PROFILE_CLAIM = "https://api.openai.com/profile"
VERSIONS = {"anthropic": 3, "openai-codex": 1}
RESET_WINDOWS = {"five_hour": "anthropic:5h", "seven_day": "anthropic:7d",
                 "seven_day_opus": "anthropic:7d:opus", "seven_day_sonnet": "anthropic:7d:sonnet"}


class ProbeRejected(OSError):
    """The subscription service rejected the current credential."""


def numeric(value):
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return None
    return value if number(value) else None


def iso_ms(value):
    if not isinstance(value, str):
        return None
    try:
        date = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return int((date if date.tzinfo else date.replace(tzinfo=timezone.utc)).timestamp() * 1000)
    except (ValueError, OverflowError, OSError):
        return None


def slug(value):
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def identity(credential):
    return {field: value for field in ("email", "accountId", "orgId", "orgName", "projectId")
            if (value := text(credential.get(field)))}


def percent_limit(provider, key, label, window_id, window_label, duration, reset, percent, *, tier=None, allowed=False):
    if percent is not None:
        result = header_limit(provider, f"{provider}:{key}", window_id, duration, reset, percent / 100,
                              tier=tier, served=allowed)
    else:
        window = {"id": window_id}
        if duration is not None:
            window["durationMs"] = duration
        if reset is not None:
            window["resetsAt"] = reset
        result = {"id": f"{provider}:{key}", "scope": {"provider": provider, "windowId": window_id, "shared": True},
                  "window": window, "amount": {"unit": "percent"}, "status": "unknown"}
    result["label"] = label
    result["window"]["label"] = window_label
    return result


def usd(value, exponent, currency, *, require_currency=False):
    if not number(value) or int(value) != value or not 0 <= value <= 2**53 - 1:
        return None
    if not number(exponent) or int(exponent) != exponent or not 0 <= exponent <= 308:
        return None
    if not ((currency is None and not require_currency) or isinstance(currency, str) and currency.upper() == "USD"):
        return None
    return value / 10.0**exponent


def claude_extra(payload):
    if payload.get("spend") is not None:
        spend = object_or_empty(payload["spend"])
        if spend.get("enabled") is not True or "limit" not in spend:
            return None
        def dollars(raw):
            raw = object_or_empty(raw)
            return usd(raw.get("amount_minor"), raw.get("exponent"), raw.get("currency"), require_currency=True)
        used, cap = dollars(spend.get("used")), dollars(spend.get("limit"))
        unlimited = spend["limit"] is None
    else:
        extra = object_or_empty(payload.get("extra_usage"))
        if extra.get("is_enabled") is not True or "monthly_limit" not in extra:
            return None
        exponent, currency = extra.get("decimal_places", 2), extra.get("currency")
        used = usd(extra.get("used_credits"), exponent, currency)
        cap = usd(extra.get("monthly_limit"), exponent, currency)
        unlimited = extra["monthly_limit"] is None
    if used is None or not unlimited and (cap is None or cap <= 0):
        return None
    result = {"id": "anthropic:extra", "label": "Claude Extra Usage",
              "scope": {"provider": "anthropic", "windowId": "extra"}, "amount": {"unit": "usd", "used": used}}
    if cap:
        fraction = used / cap
        remaining = max(0, cap - used)
        if not number(fraction):
            return None
        result["amount"].update(limit=cap, remaining=remaining, usedFraction=fraction, remainingFraction=remaining / cap)
        result["status"] = "exhausted" if fraction >= 1 else "warning" if fraction >= .9 else "ok"
    return result


def claude_identity(payload):
    account, user = (object_or_empty(payload.get(field)) for field in ("account", "user"))
    candidates = {
        "accountId": [*(payload.get(key) for key in ("account_id", "accountId", "user_id", "userId")),
                      account.get("uuid"), account.get("id"), user.get("uuid"), user.get("id")],
        "email": [*(payload.get(key) for key in ("email", "user_email", "userEmail")), account.get("email"), user.get("email")],
    }
    return {field: value for field, values in candidates.items()
            if (value := next((parsed for raw in values if (parsed := text(raw))), None))}


def reset_window(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-z0-9_-]{1,80}", value):
        raise ValueError("Invalid reset window.")
    return RESET_WINDOWS.get(value, "anthropic:reset:" + value)


def reset_date(value):
    if value is None:
        return None
    stamp = iso_ms(value)
    if stamp is None:
        raise ValueError("Invalid reset timestamp.")
    return datetime.fromtimestamp(stamp / 1000, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def claude_reset_inventory(program, raw, now):
    if not isinstance(raw, dict) or type(eligible := raw.get("eligible")) is not bool:
        return None
    reason = text(raw.get("ineligible_reason"))
    try:
        if program == "juniper_tide":
            arm, available = raw.get("arm"), raw.get("available", False)
            if arm not in (None, "reset", "control") or type(available) is not bool:
                return None
            eligible = eligible and arm == "reset"
            usable = eligible and available
            info = {"availableCount": int(usable), "redeemableCount": int(usable), "eligible": eligible, "credits": []}
            if eligible:
                info["nextCreditId"] = program
            if not reason and not usable:
                reason = "ineligible" if not raw["eligible"] else "control" if arm != "reset" else "unavailable"
            if not usable and (cooldown := reset_date(raw.get("next_available_at"))):
                info["cooldownUntil"] = cooldown
            if arm == "reset":
                credit = {"id": program, "title": "Claude session limit reset", "program": program, "usable": usable,
                          "remainingCount": int(usable), "requiresLimit": True, "clears": ["anthropic:5h"],
                          "blocking": [], "usedFractions": {}, "status": "available" if usable else "unavailable"}
                if expiry := reset_date(raw.get("weekly_resets_at")):
                    credit["expiresAt"] = expiry
                info["credits"].append(credit)
        else:
            cooldown = reset_date(raw.get("cooldown_until"))
            cooling = cooldown is not None and iso_ms(cooldown) > now
            grants, next_id = raw.get("grants", []), raw.get("next_grant_id")
            if not isinstance(grants, list) or (next_id is not None and (
                not isinstance(next_id, str) or not re.fullmatch(r"[a-z0-9_-]{1,40}", next_id)
            )):
                return None
            info = {"availableCount": 0, "redeemableCount": 0, "eligible": eligible, "credits": []}
            for item in grants:
                grant = object_or_empty(item)
                identifier, remaining = grant.get("id"), grant.get("resets_left")
                clears, blocking, percents = grant.get("clears"), grant.get("blocking"), grant.get("percent_used", {})
                if (not isinstance(identifier, str) or not re.fullmatch(r"[a-z0-9_-]{1,40}", identifier)
                    or not number(remaining) or remaining < 0 or int(remaining) != remaining
                    or not isinstance(clears, list) or not isinstance(blocking, list) or not isinstance(percents, dict)
                    or any(not number(value) or int(value) != value or not 0 <= value <= 100 for value in percents.values())
                    or any(key in grant and type(grant[key]) is not bool for key in ("paused", "usable_now", "use_requires_limit"))):
                    return None
                start, expiry = reset_date(grant.get("starts_at")), reset_date(grant.get("ends_at"))
                expired = expiry is not None and iso_ms(expiry) <= now
                selected = identifier == next_id
                paused = grant.get("paused", False)
                usable = bool(eligible and selected and grant.get("usable_now") and not (paused or expired or cooling) and remaining)
                status = "expired" if expired else "redeemed" if not remaining else (
                    "paused" if paused else "available" if usable else "unavailable"
                )
                credit = {"id": identifier, "title": text(grant.get("label")) or "Claude limit reset", "program": program,
                          "remainingCount": remaining, "usable": usable, "requiresLimit": grant.get("use_requires_limit", True),
                          "clears": list(dict.fromkeys(map(reset_window, clears))),
                          "blocking": list(dict.fromkeys(map(reset_window, blocking))),
                          "usedFractions": {reset_window(key): value / 100 for key, value in percents.items()}, "status": status}
                if start:
                    credit["grantedAt"] = start
                if expiry:
                    credit["expiresAt"] = expiry
                info["credits"].append(credit)
                if not expired:
                    info["availableCount"] += remaining
                if selected and eligible:
                    info["nextCreditId"] = identifier
                    if usable:
                        info["redeemableCount"] = remaining
            if cooldown:
                info["cooldownUntil"] = cooldown
            if not reason and cooling:
                reason = "cooldown"
    except (ValueError, OverflowError, OSError):
        return None
    if reason:
        info["reason"] = reason
    return info


async def claude_resets(client, endpoint, headers, payload, now, *, evidence=None, timeout=2, on_throttle=None):
    async def lookup(program, query):
        if isinstance(payload.get(program), dict):
            result = claude_reset_inventory(program, payload[program], now)
            return result is not None, result
        try:
            response = await client.get(endpoint, params=query, headers={**headers, "anthropic-beta": "oauth-2025-04-20"},
                                        timeout=timeout)
        except httpx2.HTTPError:
            return False, None
        if not response.is_success:
            if response.status_code == 429 and on_throttle:
                on_throttle(response.headers)
            return False, None
        body = parse_object(response.content)
        if evidence is not None:
            evidence[program] = body
        if body.get(program) is None:
            return program in body or any(key in body for key in ("five_hour", "seven_day", "limits", "extra_usage", "spend")), None
        result = claude_reset_inventory(program, body[program], now)
        return result is not None, result

    ok, cedar = await lookup("cedar_ember", {"cedar_ember": "1", "skip_spend": "1"})
    if not ok:
        # An embedded Juniper verdict already provides an authoritative answer.
        return claude_reset_inventory("juniper_tide", payload.get("juniper_tide"), now)
    if cedar and (cedar["availableCount"] or cedar.get("nextCreditId")):
        return cedar
    ok, juniper = await lookup("juniper_tide", {"at_wall": "1", "skip_spend": "1"})
    if not ok:
        return cedar if isinstance(payload.get("cedar_ember"), dict) else None
    return juniper or cedar or {"availableCount": 0, "redeemableCount": 0, "eligible": False, "credits": []}


def parse_claude(payload, credential, endpoint, now):
    payload = object_or_empty(payload)
    entries = payload.get("limits")
    entries = [entry for entry in entries if isinstance(entry, dict)] if isinstance(entries, list) else []
    limits = []

    def add(window, label, bucket, *, tier=None, generic=False):
        bucket = object_or_empty(bucket)
        percent = numeric(bucket.get("percent" if generic else "utilization"))
        if percent is None:
            return
        duration, window_label = (5 * HOUR, "5 Hour") if window == "5h" else (WEEK, "7 Day")
        limits.append(percent_limit("anthropic", f"{window}:{tier}" if tier else window, label, window, window_label,
                                    duration, iso_ms(bucket.get("resets_at")), percent, tier=tier))

    for field, kind, window, label in (("five_hour", "session", "5h", "Claude 5 Hour"),
                                        ("seven_day", "weekly_all", "7d", "Claude 7 Day")):
        legacy = object_or_empty(payload.get(field))
        if numeric(legacy.get("utilization")) is not None or iso_ms(legacy.get("resets_at")) is not None:
            add(window, label, legacy)
        else:
            fallback = next((entry for entry in entries if entry.get("kind") == kind and (
                numeric(entry.get("percent")) is not None or iso_ms(entry.get("resets_at")) is not None
            )), None)
            add(window, label, fallback, generic=True)
    for tier in ("opus", "sonnet"):
        add("7d", f"Claude 7 Day ({tier.title()})", payload.get(f"seven_day_{tier}"), tier=tier)
    seen = set()
    for entry in entries:
        scope = object_or_empty(object_or_empty(entry.get("scope")).get("model"))
        display = text(scope.get("display_name"))
        if entry.get("kind") == "weekly_scoped" and display and (tier := slug(display)) and tier not in seen:
            seen.add(tier)
            add("7d", f"Claude 7 Day ({display})", entry, tier=tier, generic=True)
    if extra := claude_extra(payload):
        limits.append(extra)
    if not limits:
        return None
    return {"provider": "anthropic", "fetchedAt": now, "limits": limits,
            "metadata": {**identity(credential), **claude_identity(payload), "endpoint": endpoint}}


def reset_time(payload, now):
    if (reset := numeric(payload.get("reset_at"))) is not None:
        result = reset if reset > 10**12 else reset * 1000
        if number(result):
            return result
    if (seconds := numeric(payload.get("reset_after_seconds"))) is not None and number(result := now + seconds * 1000):
        return result
    return None


def verdict(payload):
    return {key: value for key, source in (("allowed", "allowed"), ("limitReached", "limit_reached"))
            if type(value := payload.get(source)) is bool}


def codex_window(key, raw, state, now, *, name=None, tier=None, account=None, model=None):
    payload = object_or_empty(raw)
    seconds, percent = numeric(payload.get("limit_window_seconds")), numeric(payload.get("used_percent"))
    reset = reset_time(payload, now)
    if seconds is not None and not number(seconds * 1000):
        seconds = None
    if seconds is None and reset is None and percent is None:
        return None
    if seconds is not None:
        days = seconds >= 86400
        count = max(1, math.floor(seconds / (86400 if days else 3600) + .5))
        window_id = f"{count}{'d' if days else 'h'}"
        unit = "day" if days else "hour"
        window_label = f"{count} {unit}{'s' if count != 1 else ''}"
    else:
        window_id, window_label = key, key.title() + " window"
    limit = percent_limit(
        "openai-codex", f"{tier}:{key}" if tier else key, f"{window_label} ({name})" if name else window_label,
        window_id, window_label, seconds * 1000 if seconds is not None else None, reset, percent, tier=tier,
        allowed=state.get("allowed") is True and state.get("limitReached") is False,
    )
    if tier:
        limit["scope"].update(tier=tier, shared=True)
        if account:
            limit["scope"]["accountId"] = account
        if model:
            limit["scope"]["modelId"] = model
    return limit


def codex_spend(payload, account, now):
    control = object_or_empty(payload.get("spend_control"))
    cap = object_or_empty(control.get("individual_limit"))
    used, maximum, remaining, percent = (numeric(cap.get(key)) for key in ("used", "limit", "remaining", "used_percent"))
    if used is not None and maximum is not None and maximum > 0:
        raw_fraction = used / maximum
    elif percent is not None:
        raw_fraction = percent / 100
    else:
        raw_fraction = 1 if remaining is not None and remaining <= 0 else None
    if all(value is None for value in (used, maximum, remaining, raw_fraction)):
        return None
    fraction = min(max(raw_fraction, 0), 1) if raw_fraction is not None else None
    amount = {"unit": "credits"}
    for key, value in (("used", used), ("limit", maximum), ("remaining", remaining), ("usedFraction", fraction),
                       ("remainingFraction", 1 - fraction if fraction is not None else None)):
        if value is not None:
            amount[key] = value
    reached = control.get("reached")
    status = "exhausted" if reached is True or reached is not False and fraction == 1 else (
        "warning" if fraction is not None and fraction >= .9 else "ok" if fraction is not None else "unknown"
    )
    window = {"id": "spend-cap", "label": "Spend cap"}
    if (reset := reset_time(cap, now)) is not None:
        window["resetsAt"] = reset
    scope = {"provider": "openai-codex", "tier": "credits", "windowId": "spend-cap", "shared": True}
    if account:
        scope["accountId"] = account
    return {"id": "openai-codex:spend-cap", "label": "Spend cap", "scope": scope, "window": window,
            "amount": amount, "status": status}


def parse_codex(payload, credential, now):
    payload = object_or_empty(payload)
    account = credential.get("accountId")
    spend = codex_spend(payload, account, now)
    plan = object_or_empty(payload.get("rate_limit"))
    plan_state = verdict(plan)
    reached = object_or_empty(payload.get("spend_control")).get("reached")
    credits = object_or_empty(payload.get("credits"))
    if not isinstance(payload.get("rate_limit"), dict) and spend and type(reached) is bool:
        plan_state = {"allowed": not reached, "limitReached": reached}
    elif (plan_state.get("limitReached") is True and (credits.get("unlimited") is True or credits.get("has_credits") is True)
          and credits.get("overage_limit_reached") is not True and reached is not True
          and not (spend and reached is not False and spend["amount"].get("usedFraction", 0) >= 1)):
        plan_state = {"allowed": True, "limitReached": False}
    states = {"chat": plan_state}
    limits = [limit for key in ("primary", "secondary")
              if (limit := codex_window(key, plan.get(f"{key}_window"), plan_state, now))]
    extra = payload.get("additional_rate_limits")
    for raw in extra if isinstance(extra, list) else ():
        item = object_or_empty(raw)
        meter = object_or_empty(item.get("rate_limit"))
        name, feature = text(item.get("limit_name")), text(item.get("metered_feature"))
        probe = " ".join(value for value in (name, feature) if value).lower()
        tier = "spark" if "spark" in probe or "bengalfox" in probe else (
            slug(re.sub(r"^codex[-_]", "", name or feature or "extra", flags=re.I)) or "extra"
        )
        state = verdict(meter)
        scoped = [limit for key in ("primary", "secondary") if (limit := codex_window(
            key, meter.get(f"{key}_window"), state, now,
            name="Spark" if tier == "spark" else name or tier.replace("-", " ").title(),
            tier=tier, account=account, model=name,
        ))]
        if state or scoped:
            states[tier] = state
            limits.extend(scoped)
    if spend:
        limits.append(spend)
    metadata = {**identity(credential), **plan_state, "meterStates": states}
    if plan_type := text(payload.get("plan_type")):
        metadata["planType"] = plan_type
    report = {"provider": "openai-codex", "fetchedAt": now, "limits": limits, "metadata": metadata}
    reset = object_or_empty(payload.get("rate_limit_reset_credits"))
    if (count := numeric(reset.get("available_count"))) is not None:
        report["resetCredits"] = {"availableCount": max(0, int(count))}
    return report


def retry_delay(value, attempt):
    minimum = .5 * 2**attempt
    if (seconds := numeric(value)) is not None:
        return max(minimum, seconds)
    if value:
        try:
            return max(minimum, parsedate_to_datetime(value).timestamp() - time())
        except (TypeError, ValueError, OverflowError, OSError):
            pass
    return minimum


async def fetch_claude(client, credential):
    headers = {"Authorization": "Bearer " + credential["access"], "anthropic-beta": CLAUDE_BETAS,
               "Accept": "application/json, text/plain, */*", "Content-Type": "application/json",
               "User-Agent": f"claude-cli/{os.getenv('PI_AI_CLAUDE_CODE_VERSION') or '2.1.280'} (external, cli)"}
    endpoint = CLAUDE_URL + "/usage"
    for attempt in range(3):
        after = None
        try:
            response = await client.get(endpoint, headers=headers)
            if response.is_success:
                body = parse_object(response.content)
                now = int(time() * 1000)
                report = parse_claude(body, credential, endpoint, now)
                if report or isinstance(body.get("cedar_ember"), dict) or isinstance(body.get("juniper_tide"), dict):
                    reset = await claude_resets(client, endpoint, headers, body, now)
                    if not report and not (reset and (reset["availableCount"] or reset["credits"] or reset.get("nextCreditId"))):
                        return None
                    if not report:
                        report = {"provider": "anthropic", "fetchedAt": now, "limits": [],
                                  "metadata": {**identity(credential), **claude_identity(body), "endpoint": endpoint}}
                    if reset is not None:
                        report["resetCredits"] = reset
                    metadata = report["metadata"]
                    if not metadata.get("accountId") or not metadata.get("email"):
                        try:
                            profile = await client.get(CLAUDE_URL + "/profile", headers=headers)
                            if profile.is_success:
                                payload = parse_object(profile.content)
                                recovered = claude_identity(payload)
                                if value := text(payload.get("uuid")):
                                    recovered["accountId"] = value
                                report["metadata"] = {**recovered, **metadata}
                        except httpx2.HTTPError:
                            pass
                    return report
            elif response.status_code not in (408, 409, 425) and (
                response.status_code < 500 or response.status_code == 501
            ):
                return None
            after = response.headers.get("Retry-After")
        except httpx2.HTTPError:
            pass
        if attempt < 2:
            await asyncio.sleep(retry_delay(after, attempt))
    return None


async def codex_optional(client, path, headers):
    try:
        response = await client.get(f"{CODEX_URL}/{path}", headers=headers, timeout=2)
        return parse_object(response.content) if response.is_success else None
    except httpx2.HTTPError:
        return None


async def fetch_codex(client, credential):
    now = int(time() * 1000)
    if credential["expires"] <= now:
        return None
    claims = jwt_payload(credential["access"])
    metadata = identity(credential)
    if account := (metadata.get("accountId") or text(object_or_empty(claims.get(AUTH_CLAIM)).get("chatgpt_account_id"))):
        metadata["accountId"] = account
    if email := (metadata.get("email") or text(object_or_empty(claims.get(PROFILE_CLAIM)).get("email"))):
        metadata["email"] = email.lower()
    headers = {"Authorization": "Bearer " + credential["access"], "User-Agent": "omp/18.18.1"}
    if account:
        headers["ChatGPT-Account-Id"] = account
    async with asyncio.TaskGroup() as tasks:
        access = tasks.create_task(codex_optional(client, "accounts/verified_access", headers))
        try:
            response = await client.get(f"{CODEX_URL}/wham/usage", headers=headers)
        except httpx2.HTTPError:
            access.cancel()
            return None
        if response.status_code in (401, 403):
            access.cancel()
            rejected = True
        elif not response.is_success or not (payload := parse_object(response.content)):
            access.cancel()
            return None
        else:
            rejected = False
            report = parse_codex(payload, metadata, now)
            if report.get("resetCredits", {}).get("availableCount", 0) > 0:
                details = tasks.create_task(codex_optional(client, "wham/rate-limit-reset-credits", headers))
            else:
                details = None
    if rejected:
        raise ProbeRejected("Codex rejected the quota probe.")
    programs = (access.result() or {}).get("programs")
    if isinstance(programs, list) and any(isinstance(program, dict) and program.get("program") == "cyber"
                                         and (program.get("state") != "inactive" or program.get("grants")) for program in programs):
        report["metadata"]["daybreak"] = True
    if details is not None and (data := details.result()) is not None:
        rows = data.get("credits")
        available = [item for item in rows if isinstance(item, dict) and text(item.get("id"))
                     and item.get("status", "available") == "available"] if isinstance(rows, list) else []
        count = numeric(data.get("available_count"))
        report["resetCredits"]["availableCount"] = max(0, int(count)) if count is not None else len(available)
        if available:
            report["resetCredits"]["credits"] = [
                {field: value for field, key in (("grantedAt", "granted_at"), ("expiresAt", "expires_at"), ("status", "status"))
                 if (value := text(item.get(key)))} for item in available
            ]
    return report


async def fetch_quota(provider, credential):
    async with httpx2.AsyncClient(timeout=5, follow_redirects=False) as client:
        if provider == "anthropic":
            return await fetch_claude(client, credential)
        return await fetch_codex(client, credential)
