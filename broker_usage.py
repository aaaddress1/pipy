"""Quota windows and OAuth account ranking for broker-backed model routes."""

import math
import re
from dataclasses import dataclass
from functools import cmp_to_key

from auth_broker_cache import number


HOUR = 3_600_000
WEEK = 7 * 24 * HOUR
PROVIDERS = frozenset(("anthropic", "openai-codex"))
IDENTITY = ("email", "accountId", "projectId", "orgId")


def parse_policies(entries, reserve):
    if entries is None:
        entries = []
    if reserve is not None and not number(reserve) or not isinstance(entries, list):
        raise ValueError("Invalid Auth Broker quota policies.")
    reserve = max(0, min(100, reserve)) if reserve is not None else 10
    policies = {}
    for rule in entries:
        if not isinstance(rule, dict) or rule.keys() - {"provider", "account", "priority", "reservePct", "reclaimAbovePct"}:
            raise ValueError("Invalid auth.accountPolicies entry.")
        provider, account = rule.get("provider"), rule.get("account")
        priority, threshold, reclaim = rule.get("priority", 0), rule.get("reservePct", reserve), rule.get("reclaimAbovePct")
        if (not isinstance(provider, str) or not provider or provider.strip() != provider or not isinstance(account, dict)
            or account.keys() - set(IDENTITY) or not account.keys() & set(IDENTITY[:3])
            or not all(isinstance(value, str) and value for value in account.values()) or not number(priority)
            or not number(threshold) or not 0 <= threshold <= 100
            or "reclaimAbovePct" in rule and (not number(reclaim) or not threshold < reclaim <= 100)):
            raise ValueError("Invalid auth.accountPolicies entry.")
        policies.setdefault(provider, []).append((account, priority, threshold, reclaim))
    return policies, reserve


def account_policy(row, policies, reserve):
    matching = [(priority, threshold, reclaim) for account, priority, threshold, reclaim in policies
                if all(row["account"].get(field) == value for field, value in account.items())]
    if len(matching) > 1:
        raise ValueError("Overlapping auth.accountPolicies match one broker account.")
    return matching[0] if matching else (0, reserve if policies else None, None)


def text_field(source, key):
    value = source.get(key)
    return value.strip().lower() if isinstance(value, str) and value.strip() else None


def match_report(reports, row, allow_unattributed):
    account = row["account"]
    org = text_field(account, "orgId")
    same_org = [report for report in reports if text_field(report["metadata"], "orgId") == org]
    if not any(account.get(field) for field in IDENTITY[:3]) and org:
        return same_org[0] if len(same_org) == 1 else None
    comparable = False
    for report in same_org:
        metadata = report["metadata"]
        email, report_email = text_field(account, "email"), text_field(metadata, "email")
        if email and report_email:
            comparable = True
            if email == report_email:
                return report
            continue
        for field, legacy in (("accountId", "account_id"), ("projectId", "project_id")):
            value = text_field(account, field)
            if not value:
                continue
            found = {text_field(metadata, field) or text_field(metadata, legacy),
                     *(text_field(limit["scope"], field) for limit in report["limits"])} - {None}
            comparable |= bool(found)
            if value in found:
                return report
    if not org and not comparable and allow_unattributed and len(reports) == len(same_org) == 1:
        return same_org[0]
    return None


def usage_reports(payload, provider):
    reports = []
    for report in payload.get("reports", ()) if isinstance(payload.get("reports"), list) else ():
        if not isinstance(report, dict) or report.get("provider") != provider or not number(report.get("fetchedAt")):
            continue
        limits = report.get("limits")
        if not isinstance(limits, list):
            continue
        metadata = report.get("metadata")
        reports.append({"provider": provider, "fetchedAt": report["fetchedAt"], "metadata": metadata if isinstance(metadata, dict) else {},
                        **({"resetCredits": report["resetCredits"]} if isinstance(report.get("resetCredits"), dict) else {}),
                        "limits": [limit for limit in limits if isinstance(limit, dict) and isinstance(limit.get("id"), str)
                                   and isinstance(limit.get("scope"), dict) and isinstance(limit.get("amount"), dict)]})
    return reports


def header_number(headers, key):
    try:
        value = float(headers.get(key, ""))
        return value if number(value) else None
    except (TypeError, ValueError):
        return None


def header_limit(provider, identifier, window_id, duration, reset, used, *, tier=None, served=False):
    used = max(0, min(1, used))
    scope = {"provider": provider, "windowId": window_id}
    if tier:
        scope["tier"] = tier
    else:
        scope["shared"] = True
    detail = {"id": window_id}
    if duration is not None:
        detail["durationMs"] = duration
    if reset is not None:
        detail["resetsAt"] = reset
    return {"id": identifier, "scope": scope, "window": detail,
            "amount": {"unit": "percent", "used": used * 100, "limit": 100, "remaining": (1 - used) * 100,
                       "usedFraction": used, "remainingFraction": 1 - used},
            "status": "exhausted" if used == 1 and not served else "warning" if used >= 0.9 else "ok"}


def response_limits(provider, headers, status):
    limits = []
    if provider == "anthropic":
        for key, window_id, duration, tier in (("5h", "5h", 5 * HOUR, None), ("7d", "7d", WEEK, None),
                                               ("7d_oi", "7d", WEEK, "fable")):
            prefix = f"anthropic-ratelimit-unified-{key}-"
            used = header_number(headers, prefix + "utilization")
            if used is None:
                continue
            seconds = header_number(headers, prefix + "reset")
            reset = seconds * 1000 if seconds is not None and seconds > 0 and number(seconds * 1000) else None
            identifier = f"anthropic:{window_id}" + (f":{tier}" if tier else "")
            limits.append(header_limit(provider, identifier, window_id, duration, reset, used, tier=tier))
    elif provider == "openai-codex":
        for key in ("primary", "secondary"):
            prefix = f"x-codex-{key}-"
            percent = header_number(headers, prefix + "used-percent")
            if percent is None:
                continue
            minutes = header_number(headers, prefix + "window-minutes")
            duration = minutes * 60_000 if minutes is not None and number(minutes * 60_000) else None
            window_id = key
            if duration is not None:
                window_id = (f"{math.floor(duration / (24 * HOUR) + .5)}d" if duration >= 24 * HOUR
                             else f"{max(1, math.floor(duration / HOUR + .5))}h")
            reset = header_number(headers, prefix + "reset-at")
            if reset is not None and reset <= 1_000_000_000_000:
                reset *= 1000
            limits.append(header_limit(provider, f"openai-codex:{key}", window_id, duration, reset, percent / 100,
                                       served=200 <= status < 300))
    return limits


def overlay_key(row):
    account = row["account"]
    identity = next(((field, value) for field in IDENTITY[:3] if (value := text_field(account, field))), None)
    organization = text_field(account, "orgId")
    return (row["provider"], organization, identity) if identity or organization else None


def merge_report(base, overlay):
    if not base or not overlay:
        return overlay or base
    limits = {limit["id"]: {"_fetchedAt": base["fetchedAt"], **limit} for limit in base["limits"]}
    limits.update((limit["id"], {"_fetchedAt": overlay["fetchedAt"], **limit}) for limit in overlay["limits"])
    return {"fetchedAt": max(base["fetchedAt"], overlay["fetchedAt"]), "limits": list(limits.values()),
            "metadata": {**overlay["metadata"], **base["metadata"]}}


def quota_scope(provider, model, configured):
    if configured:
        return configured
    if provider == "openai-codex":
        return "spark" if "-spark" in model.lower() else "chat"
    if provider == "anthropic" and (family := claude_family(model)) in ("fable", "mythos"):
        return f"tier:{family}"
    return ""


def claude_family(model):
    found = re.search(r"(?:^|[-/])(opus|sonnet|fable|mythos)(?:-|$)", model.lower())
    return found[1] if found else None


def plan_requirement(provider, model):
    if provider != "openai-codex":
        return None
    if "-spark" in model.lower():
        return "pro"
    name = model.rsplit("/", 1)[-1].lower()
    if name.startswith("gpt-6.1-terra") or name in (
        "gpt-5.6", "gpt-5.6-sol", "gpt-5.6-sol-pro", "gpt-5.6-luna", "gpt-5.6-luna-pro",
    ):
        return "paid"
    return None


def plan_rank(requirement, metadata):
    if not requirement:
        return 0
    plan = re.sub(r"[\s-]+", "_", text_field(metadata, "planType") or "").removeprefix("chatgpt_")
    tokens = set(plan.split("_"))
    if plan not in ("prolite", "pro_lite") and tokens & {"pro", "promax"}:
        return 0
    if plan in ("prolite", "pro_lite") or tokens & {
        "plus", "business", "team", "enterprise", "edu", "education", "teacher", "teachers", "health", "gov", "government",
    }:
        return 0 if requirement == "paid" else 2
    return 2 if tokens & {"free", "go"} else 1


def window(limit):
    value = limit.get("window")
    return value if isinstance(value, dict) else {}


def fraction(limit):
    amount = limit["amount"]
    if number(value := amount.get("usedFraction")):
        return value
    used, maximum = amount.get("used"), amount.get("limit")
    if number(used) and number(maximum) and maximum > 0:
        return used / maximum
    if number(used) and amount.get("unit") == "percent":
        return used / 100
    if number(value := amount.get("remainingFraction")):
        return max(0, 1 - value)
    return None


def exhausted(limit):
    status = limit.get("status")
    if status in ("ok", "warning", "exhausted"):
        return status == "exhausted"
    used, remaining = fraction(limit), limit["amount"].get("remaining")
    return used is not None and used >= 1 or number(remaining) and remaining <= 0


def used_fraction(limit):
    used = limit["amount"].get("usedFraction") if limit else None
    return max(0, min(1, used)) if number(used) else 0.5


def drain(limit, now, default):
    detail = window(limit) if limit else {}
    duration, reset = detail.get("durationMs"), detail.get("resetsAt")
    duration = duration if number(duration) else default
    remaining = reset - now if number(reset) else duration
    if duration > 0:
        remaining = min(remaining, duration)
    return (1 - used_fraction(limit)) * HOUR / max(60_000, remaining)


def claude_pressure(limit, now):
    used = fraction(limit)
    used = max(0, min(1, used)) if used is not None else 0.5
    duration, reset = window(limit).get("durationMs", WEEK), window(limit).get("resetsAt")
    elapsed = duration - min(max(reset - now, 0), duration) if number(reset) and number(duration) and duration > 0 else 0
    return (used * HOUR / elapsed if elapsed > 0 else used, used)


def limits_for(provider, model, limits, now):
    if provider == "anthropic":
        family = claude_family(model)
        scoped = [limit for limit in limits if limit["scope"].get("shared") is True
                  or family is not None and limit["scope"].get("tier") == family]
        gates = [limit for limit in scoped if limit["scope"].get("shared") is True or family not in ("fable", "mythos")
                 or (number(reset := window(limit).get("resetsAt")) and reset > now and (
                     limit.get("status") == "exhausted" or (fraction(limit) or 0) >= 1))]
        primary = next((limit for limit in limits if limit["id"] == "anthropic:5h"), None)
        secondary = max((limit for limit in scoped if "7d" in (limit["scope"].get("windowId"), window(limit).get("id"))),
                        key=lambda limit: claude_pressure(limit, now), default=None)
        return scoped, gates, primary, secondary, False
    if provider == "openai-codex":
        spark = "-spark" in model.lower()
        plan = any(limit["id"] in ("openai-codex:primary", "openai-codex:secondary") for limit in limits)
        scoped = [limit for limit in limits if (limit["id"].startswith("openai-codex:spark:") if spark else (
            limit["id"] in ("openai-codex:primary", "openai-codex:secondary")
            or not plan and limit["id"] == "openai-codex:spend-cap"))]
        def find(key, window_id):
            return next((limit for limit in scoped if limit["id"].endswith(":" + key)), None) or next(
                (limit for limit in scoped if limit["scope"].get("windowId") == window_id), None)
        primary, secondary = find("primary", "1h"), find("secondary", "7d")
        duration = window(primary).get("durationMs") if primary else None
        boost = (not spark and secondary is not None) if not primary else (
            (primary["scope"].get("windowId") == "5h" or number(duration) and abs(duration - 5 * HOUR) <= 60_000)
            and primary["amount"].get("usedFraction") == 0)
        return scoped, scoped, primary, secondary, boost
    scoped = [limit for limit in limits if limit["scope"].get("modelId") in (None, model)]
    return scoped, (), None, None, False


def healable_scopes(provider, report):
    limits, metadata = report["limits"], report["metadata"]
    if provider == "anthropic":
        shared = [limit for limit in limits if limit["scope"].get("shared") is True]
        if not all(any(identifier in (limit["scope"].get("windowId"), window(limit).get("id"))
                       for limit in shared) for identifier in ("5h", "7d")):
            return set()
        scopes = {f"tier:{tier}": shared + [limit for limit in limits if limit["scope"].get("tier") == tier]
                  for tier in ("fable", "mythos") if any(limit["scope"].get("tier") == tier for limit in limits)}
        if metadata.get("source") != "ratelimit-headers":
            scopes[""] = [limit for limit in limits if limit["scope"].get("shared") is True
                          or limit["scope"].get("tier") is not None]
        return {scope for scope, gates in scopes.items() if not any(map(exhausted, gates))}
    if provider == "openai-codex":
        spark = metadata.get("meterStates")
        spark = spark.get("spark") if isinstance(spark, dict) else None
        chat_limits = limits_for(provider, "", limits, 0)[0]
        spark_limits = limits_for(provider, "codex-spark", limits, 0)[0]
        plan = any(limit["id"] in ("openai-codex:primary", "openai-codex:secondary") for limit in limits)
        shared = [limit for limit in limits if not plan or limit["id"] != "openai-codex:spend-cap"]
        return {scope for scope, verdict, gates in (("chat", metadata, chat_limits), ("spark", spark, spark_limits),
                                                    ("shared", metadata, shared))
                if isinstance(verdict, dict) and verdict.get("allowed") is True and verdict.get("limitReached") is False
                and not any(map(exhausted, gates))}
    return set()


def usage_reset_times(provider, model, report, now):
    """A complete report authorizes waiting past the cap. Partial reports can only extend a block."""
    if not report:
        return None, None
    _, gates, _, _, _ = limits_for(provider, model, report["limits"], now)
    exhausted_windows = []
    for limit in gates:
        fetched = limit.get("_fetchedAt", report["fetchedAt"])
        if not exhausted(limit):
            continue
        reset = window(limit).get("resetsAt")
        if number(reset) and reset <= now and fetched < reset:
            continue
        if now - fetched > 300_000:
            exhausted_windows.append(None)
            continue
        exhausted_windows.append(reset if number(reset) and reset > now else None)
    latest = max((reset for reset in exhausted_windows if reset is not None), default=None)
    return latest, latest if None not in exhausted_windows else None


@dataclass(frozen=True)
class Candidate:
    identifier: int
    order: tuple
    metrics: tuple
    remaining: float | None
    priority: float
    reserve: bool
    reclaim: bool


def rank_oauth(provider, model, rows, reports, now, pinned, warm, policies, reserve, allow_unattributed, overlays=None,
               blocked=None):
    requirement = plan_requirement(provider, model)
    ranked = []
    for row in rows:
        report = match_report(reports, row, allow_unattributed)
        if overlays:
            report = merge_report(report, overlays.get(overlay_key(row)))
        metadata, limits = (report["metadata"], report["limits"]) if report else ({}, [])
        scoped, gates, primary, secondary, boost = limits_for(provider, model, limits, now)
        def live(limit):
            reset = window(limit).get("resetsAt")
            return not number(reset) or reset > now or limit.get("_fetchedAt", report["fetchedAt"]) >= reset
        reached = [limit for limit in gates if live(limit) and exhausted(limit)]
        if reached:
            if blocked is not None:
                resets = [reset for limit in reached if number(reset := window(limit).get("resetsAt")) and reset > now]
                blocked[row["broker_id"]] = int(max(resets, default=now + 60_000))
            continue
        measured = [value for limit in scoped if live(limit) and (value := fraction(limit)) is not None]
        remaining = max(0, 1 - max(measured)) if measured else None
        priority, threshold, reclaim = account_policy(row, policies, reserve)
        in_reserve = remaining is not None and threshold is not None and remaining <= threshold / 100 + 1e-9
        reclaim_ready = reclaim is not None and remaining is not None and remaining * 100 >= reclaim
        primary_used = used_fraction(primary)
        usage_measured = bool(primary or secondary) if provider in PROVIDERS else remaining is not None
        order = (plan_rank(requirement, metadata), remaining == 0, in_reserve, not boost, primary_used >= .85,
                 -priority, not usage_measured)
        metrics = (-drain(secondary, now, WEEK), used_fraction(secondary),
                   -drain(primary, now, 5 * HOUR if provider == "anthropic" else HOUR), primary_used)
        ranked.append(Candidate(row["broker_id"], order, metrics, remaining, priority, in_reserve, reclaim_ready))

    def compare(left, right):
        if left.order != right.order:
            return -1 if left.order < right.order else 1
        for a, b in zip(left.metrics, right.metrics):
            if not math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9):
                return -1 if a < b else 1
        return (right.identifier == pinned) - (left.identifier == pinned)

    ranked.sort(key=cmp_to_key(compare))
    if not ranked:
        return None
    best = ranked[0]
    current = next((candidate for candidate in ranked if candidate.identifier == pinned), None)
    if current and warm and not requirement:
        alternatives = [candidate for candidate in ranked if candidate is not current and candidate.remaining is not None]
        if not (best.reclaim and best.priority > current.priority or any(
            current.reserve and not candidate.reserve or current.remaining == 0 < candidate.remaining
            for candidate in alternatives
        )):
            return current.identifier
    return best.identifier
