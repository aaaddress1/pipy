"""Pure saved-reset decisions ported from Oh My Pi's Claude and Codex planners."""

from dataclasses import dataclass
from math import floor, inf

from auth_broker_cache import number
from broker_usage import HOUR, WEEK, claude_family, fraction, text_field, window
from provider_quota import iso_ms


CLAUDE, CODEX = "anthropic", "openai-codex"
CEDAR, JUNIPER = "cedar_ember", "juniper_tide"
FIVE_HOUR = "anthropic:5h"
COOLDOWN = 60_000
FRESHNESS = 10 * COOLDOWN
IMMINENT = 5 * COOLDOWN
DEFER = 30 * COOLDOWN
EXHAUSTED = .999
SALVAGE_USED = .25


@dataclass(frozen=True)
class ResetPolicy:
    mode: str = "unset"
    min_blocked_minutes: float = 60
    keep_credits: int = 0
    salvage_hours: float = 12

    @classmethod
    def from_settings(cls, config, prefix):
        from auth_broker import dotted

        mode = dotted(config, prefix + ".autoRedeem")
        mode = "yes" if mode is True else "no" if mode is False else mode

        def nonnegative(key, default):
            value = dotted(config, prefix + "." + key)
            return max(0, value) if number(value) else default

        return cls(mode if mode in ("yes", "no") else "unset", nonnegative("minBlockedMinutes", 60),
                   int(nonnegative("keepCredits", 0)), nonnegative("salvageHorizonHours", 12))


@dataclass(frozen=True)
class ResetAction:
    target: dict
    account_key: str
    attempt_key: str
    label: str
    reason: str
    available_count: int
    active: bool
    credit_id: str | None
    expires_in: float | None
    remaining: float | None = None
    blocked_windows: tuple[str, ...] = ()
    salvage_window: str | None = None
    salvage_used: float | None = None
    weekly_used: float | None = None
    program: str | None = None
    title: str | None = None
    requires_limit: bool = False


@dataclass(frozen=True)
class PlanState:
    now: int
    policy: ResetPolicy
    attempted: set
    deferred: dict
    last_attempt: dict
    permits: object = None

    def gated(self, account, episode, *, imminent=False, account_defer=False):
        if episode in self.attempted:
            return "already-attempted"
        if not imminent:
            deferred = self.deferred.get(episode, 0) > self.now
            if account_defer:
                deferred = any(until > self.now and key.startswith((f"block|{account}|", f"salvage|{account}|"))
                               for key, until in self.deferred.items())
            if deferred:
                return "deferred"
        if self.now - self.last_attempt.get(account, -inf) < COOLDOWN:
            return "cooldown"
        return None


def minute(value):
    return floor(value / COOLDOWN + .5)


def report_matches(report, status):
    data = report.get("metadata", {})
    if report.get("provider") != status["provider"] or text_field(data, "orgId") != text_field(status, "orgId"):
        return False
    account = text_field(status, "accountId")
    observed = text_field(data, "accountId") or text_field(data, "account_id")
    if account and observed:
        return account == observed
    email = text_field(status, "email")
    return bool(email and email == text_field(data, "email"))


def active_report(report, identity):
    if not identity or not report["limits"]:
        return False
    data = report.get("metadata", {})
    org = text_field(identity, "orgId")
    if org != text_field(data, "orgId"):
        return False
    email, account, project = (text_field(identity, key) for key in ("email", "accountId", "projectId"))
    if org and not (email or account or project):
        return True
    if email and (observed := text_field(data, "email")):
        return email == observed
    for field, value in (("accountId", account), ("projectId", project)):
        if not value:
            continue
        legacy = "account_id" if field == "accountId" else field
        if value in {text_field(data, field), text_field(data, legacy),
                     *(text_field(limit["scope"], field) for limit in report["limits"])}:
            return True
    return False


def claude_cached_statuses(accounts, reports):
    stored = [{**account, "provider": CLAUDE, "availableCount": 0, "credits": []} for account in accounts]
    statuses = []
    for status in stored:
        report = next((row for row in reports if report_matches(row, status)), None)
        if report is None and not status.get("orgId"):
            adopted = []
            for row in reports:
                org = row.get("metadata", {}).get("orgId")
                if not isinstance(org, str) or any(report_matches(row, other) for other in stored):
                    continue
                qualified = {**status, "orgId": org}
                if report_matches(row, qualified):
                    adopted.append((qualified, row))
            if len(adopted) == 1:
                status, report = adopted[0]
        inventory = report.get("resetCredits") if report else None
        if isinstance(inventory, dict):
            raw = inventory.get("credits")
            credits = [credit for credit in raw if isinstance(credit, dict) and isinstance(credit.get("id"), str)
                       ] if isinstance(raw, list) else []
            statuses.append({**status, **inventory, "credits": credits, "report": report})
        else:
            statuses.append({**status, "error": "credits-unknown"})
    return statuses


def codex_live_reports(reports, statuses, now, *, synthesize_active=False):
    def matches(report, status):
        data = report.get("metadata", {})
        if status.get("provider") != CODEX or text_field(status, "orgId") != text_field(data, "orgId"):
            return False
        email, report_email = text_field(status, "email"), text_field(data, "email")
        if status.get("accountId") and isinstance(data.get("accountId"), str):
            # Codex Team seats may share the workspace account id.
            return status["accountId"] == data["accountId"] and not (email and report_email and email != report_email)
        return bool(email and email == report_email)

    def credits(status):
        return {"availableCount": status["availableCount"],
                "credits": [credit for credit in status["credits"] if credit.get("status", "available") == "available"]}

    merged = []
    for report in reports:
        if report.get("provider") != CODEX:
            merged.append(report)
            continue
        matches_report = [row for row in statuses if matches(report, row)]
        status = matches_report[0] if len(matches_report) == 1 else None
        if not status or status.get("error"):
            merged.append({key: value for key, value in report.items() if key != "resetCredits"})
        else:
            merged.append({**report, "resetCredits": credits(status),
                           "metadata": {**report.get("metadata", {}), "resetCreditCredentialId": status["credentialId"]}})
    if synthesize_active:
        active = next((row for row in statuses if row.get("provider") == CODEX and row.get("active")
                       and not row.get("error") and row["availableCount"] > 0), None)
        if active and not any(row.get("metadata", {}).get("resetCreditCredentialId") == active["credentialId"]
                              and now - row["fetchedAt"] <= FRESHNESS for row in merged if row.get("provider") == CODEX):
            identity = {field: active[field] for field in ("accountId", "email", "orgId") if field in active}
            merged.append({"provider": CODEX, "fetchedAt": now, "limits": [], "resetCredits": credits(active),
                           "metadata": {**identity, "limitReached": True, "resetCreditActive": True,
                                        "resetCreditCredentialId": active["credentialId"]}})
    return merged


def codex_windows(report):
    rows = []
    for slot, default in (("primary", "5h"), ("secondary", "weekly")):
        limit = next((row for row in report["limits"] if row["id"] == f"{CODEX}:{slot}"), None)
        if not limit:
            continue
        detail = window(limit)
        duration = detail.get("durationMs")
        ids = {text_field(detail, "id"), text_field(limit["scope"], "windowId")}
        if number(duration) and abs(duration - WEEK) <= COOLDOWN:
            name = "weekly"
        elif number(duration) and abs(duration - 5 * HOUR) <= COOLDOWN:
            name = "5h"
        else:
            name = "weekly" if "7d" in ids else "5h" if "5h" in ids else default
        rows.append((name, limit["amount"].get("usedFraction"), detail.get("resetsAt"),
                     (WEEK if name == "weekly" else 5 * HOUR) + HOUR, limit["id"]))
    return rows


def fullest_codex_window(windows):
    return max((row for row in windows if number(row[1]) and row[1] >= 0), key=lambda row: row[1], default=None)


def codex_unblock(windows, reached, hint, now):
    if not reached and hint is None:
        return None, (), "not-limit-reached"
    spent = [row for row in windows if number(row[1]) and row[1] >= EXHAUSTED]
    if not spent:
        if hint is None:
            return None, (), "no-exhausted-window"
        if hint - now > WEEK + HOUR:
            return None, (), "reset-implausible"
        return hint, ("weekly" if hint - now > 6 * HOUR else "5h",), None
    for _, _, reset, plausible, _ in spent:
        if not number(reset):
            return None, (), "no-reset-time"
        if reset - now > plausible:
            return None, (), "reset-implausible"
    return max(row[2] for row in spent), tuple(row[0] for row in spent), None


def plan_codex(state, reports, *, blocked=False, provider=CODEX, model="", identity=None, hint=None):
    if state.policy.mode == "no":
        return [], [("*", "account", "disabled")]
    skipped, restores, salvages = [], [], []
    blocked_rule = blocked and provider == CODEX and "-spark" not in model
    if blocked and not blocked_rule:
        skipped.append(("*", "blocked-account", "wrong-provider" if provider != CODEX else "spark-model"))
    for report in reports:
        if report.get("provider") != CODEX:
            continue
        data = report.get("metadata", {})
        identifier = data.get("resetCreditCredentialId")
        if type(identifier) is not int:
            skipped.append(("*", "account", "credits-unknown"))
            continue
        account = f"{CODEX}|{text_field(data, 'orgId') or '-'}|{identifier}"
        inventory = report.get("resetCredits") or {}
        count = inventory.get("availableCount")
        error = ("stale-report" if state.now - report["fetchedAt"] > FRESHNESS else
                 "credits-unknown" if not number(count) else "no-credits" if count < 1 else None)
        if error:
            skipped.append((account, "account", error))
            continue
        dated = [(expiry, item.get("id")) for item in inventory.get("credits", ())
                 if item.get("status", "available") == "available" and
                 (expiry := iso_ms(item.get("expiresAt"))) is not None and expiry > state.now]
        expiry, credit = min(dated, key=lambda row: row[0], default=(inf, None))
        expires_in = expiry - state.now if expiry != inf else None
        windows = codex_windows(report)
        weekly = max((row[1] for row in windows if row[0] == "weekly" and number(row[1])), default=None)
        active = data.get("resetCreditActive") is True or active_report(report, identity)
        target = {"provider": CODEX, "credentialId": identifier,
                  **{field: data[field] for field in ("accountId", "email", "orgId") if text_field(data, field)}}
        label = target.get("email") or target.get("accountId") or account
        if (org := target.get("orgId")) and org != label:
            label = f"{label} ({org})"
        shared = (target, account)
        if blocked_rule:
            unblock, blockers, error = codex_unblock(windows, data.get("limitReached") is True,
                                                     hint if active else None, state.now)
            if state.permits and not state.permits(identifier):
                error = "outside-account-pool"
            if not error:
                remaining = unblock - state.now
                episode = f"block|{account}|{minute(unblock)}"
                error = ("reset-too-soon" if remaining < state.policy.min_blocked_minutes * COOLDOWN else
                         "reserve" if count - state.policy.keep_credits < 1 else state.gated(account, episode))
                if not error:
                    action = ResetAction(*shared, episode, label, "blocked-account", count, active, credit, expires_in,
                                         remaining, blockers, weekly_used=weekly)
                    restores.append(((not active, expiry, -count, -remaining), action))
            if error:
                skipped.append((account, "blocked-account", error))
        fullest = fullest_codex_window(windows)
        imminent = expires_in is not None and expires_in <= IMMINENT
        if expires_in is None or not imminent and expires_in > state.policy.salvage_hours * HOUR:
            error = "no-expiring-credit"
        elif not imminent and (not fullest or fullest[1] < SALVAGE_USED):
            error = "window-mostly-free"
        else:
            episode = f"salvage|{account}|{minute(expiry)}"
            error = state.gated(account, episode, imminent=imminent)
            if not error:
                salvages.append(ResetAction(*shared, episode, label, "expiring-credit", count, active, credit, expires_in,
                                            salvage_window=fullest[0] if fullest else None,
                                            salvage_used=fullest[1] if fullest else None, weekly_used=weekly))
        if error:
            skipped.append((account, "expiring-credit", error))
    return combine(restores, salvages, skipped)


def claude_exhausted(limit):
    used = fraction(limit)
    return limit.get("status") == "exhausted" or used is not None and used >= EXHAUSTED


def claude_covered(limits, credit):
    clears = set(credit.get("clears") or ())
    return [key for key in limits if key in clears and (credit.get("program") != JUNIPER or key == FIVE_HOUR)]


def fullest_claude_window(limits, covered, credit):
    used = credit.get("usedFractions") or {}
    return max(((key, max(0, fraction(limits[key]) or 0, used.get(key, 0))) for key in covered),
               key=lambda item: item[1], default=None)


def claude_unblock(limits, blockers, hint, now):
    resets = []
    for identifier in blockers:
        detail = window(limits[identifier])
        reset, duration = detail.get("resetsAt"), detail.get("durationMs")
        if not number(reset):
            if hint is None:
                return None, "no-reset-time"
            maximum = 6 * HOUR if all(item == FIVE_HOUR for item in blockers) else 8 * 24 * HOUR
            return (None, "reset-implausible") if hint - now > maximum else (hint, None)
        maximum = duration + HOUR if number(duration) and duration > 0 else (
            6 * HOUR if identifier == FIVE_HOUR else 8 * 24 * HOUR
        )
        if reset - now > maximum:
            return None, "reset-implausible"
        resets.append(reset)
    return max(resets), None


def claude_offer(status, reports, state):
    if status.get("error"):
        return None, None, "credits-unknown"
    if status.get("eligible") is not True:
        return None, None, "ineligible"
    if (iso_ms(status.get("cooldownUntil")) or 0) > state.now:
        return None, None, "provider-cooldown"
    available, redeemable = status.get("availableCount"), status.get("redeemableCount", 1)
    if not number(available) or available < 1 or not number(redeemable) or redeemable < 1:
        return None, None, "no-credits"
    credit = next((item for item in status["credits"] if item["id"] == status.get("nextCreditId")), None)
    if not credit:
        return None, None, "no-selected-credit"
    remaining = credit.get("remainingCount")
    if (credit.get("usable") is not True or credit.get("status", "available") != "available"
        or not number(remaining) or remaining < 1):
        return None, None, "credit-unusable"
    if credit.get("program") not in (CEDAR, JUNIPER):
        return None, None, "unsupported-program"
    expiry = iso_ms(credit.get("expiresAt"))
    if credit.get("expiresAt") and (expiry is None or expiry <= state.now):
        return None, None, "credit-expired"
    if (expiry is None or expiry - state.now > IMMINENT) and status["availableCount"] - state.policy.keep_credits < 1:
        return None, None, "reserve"
    report = status.get("report")
    if not report or not report_matches(report, status):
        report = next((item for item in reports if report_matches(item, status)), None)
    if not report:
        return None, None, "no-report"
    if state.now - report["fetchedAt"] > FRESHNESS:
        return None, None, "stale-report"
    return credit, report, None


def plan_claude(state, statuses, reports=(), *, blocked=False, provider=CLAUDE, model="", hint=None):
    if state.policy.mode == "no":
        return [], [("*", "account", "disabled")]
    if blocked and provider != CLAUDE:
        return [], [("*", "blocked-account", "wrong-provider")]
    skipped, restores, salvages = [], [], []
    family = claude_family(model)
    for status in statuses:
        if status.get("provider") != CLAUDE:
            continue
        identifier = status.get("credentialId")
        if type(identifier) is not int:
            skipped.append(("*", "account", "no-identity"))
            continue
        account = f"{CLAUDE}|{text_field(status, 'orgId') or '-'}|{identifier}"
        credit, report, error = claude_offer(status, reports, state)
        if error:
            skipped.append((account, "account", error))
            continue
        limits = {row["id"]: row for row in report["limits"] if not blocked or row["scope"].get("shared") is True
                  or family is not None and row["scope"].get("tier") == family}
        program, credit_id = credit["program"], credit["id"]
        expiry = iso_ms(credit.get("expiresAt"))
        expires_in = expiry - state.now if expiry is not None else None
        imminent = expires_in is not None and expires_in <= IMMINENT
        count, active = status["availableCount"], bool(status.get("active"))
        target = {"provider": CLAUDE, "credentialId": identifier, "creditId": credit_id,
                  **{field: status[field] for field in ("accountId", "email", "orgId") if field in status}}
        label = status.get("email") or status.get("accountId") or status.get("orgId") or account
        if (org := status.get("orgName") or status.get("orgId")) and org != label:
            label = f"{label} ({org})"
        grant = f"{account}|{credit_id}|{credit['remainingCount']}"
        clears, server = set(credit.get("clears") or ()), tuple(credit.get("blocking") or ())
        used = credit.get("usedFractions") or {}
        exhausted = tuple(key for key, limit in limits.items() if claude_exhausted(limit))
        extra = {"program": program, "title": credit.get("title"), "requires_limit": credit.get("requiresLimit") is True}
        if blocked:
            blockers = tuple(dict.fromkeys([*(key for key in limits if key in exhausted or used.get(key, 0) >= EXHAUSTED), *server]))
            error = ("outside-account-pool" if state.permits and not state.permits(identifier) else
                     "reserve" if count - state.policy.keep_credits < 1 else
                     "no-blocked-window" if not blockers else
                     "unsupported-window" if not limits.keys() >= set(blockers) else
                     "incomplete-coverage" if not clears >= set(blockers) or program == JUNIPER and blockers != (FIVE_HOUR,) else None)
            if not error:
                unblock, error = claude_unblock(limits, blockers, hint if active else None, state.now)
                if not error:
                    remaining = unblock - state.now
                    episode = f"block|{grant}|{','.join(sorted(blockers))}|{minute(unblock)}"
                    error = ("reset-too-soon" if remaining < state.policy.min_blocked_minutes * COOLDOWN else
                             state.gated(account, episode, account_defer=True))
                    if not error:
                        action = ResetAction(target, account, episode, label, "blocked-account", count, active,
                                             credit_id, expires_in, remaining, blockers, **extra)
                        restores.append(((not active, expiry if expiry is not None else inf, -remaining), action))
            if error:
                skipped.append((account, "blocked-account", error))

        if not imminent and state.policy.salvage_hours <= 0:
            continue
        covered = claude_covered(limits, credit)
        blockers = set(exhausted) | set(server)
        fullest = fullest_claude_window(limits, covered, credit)
        error = ("unsupported-program" if not imminent and program != CEDAR else
                 "no-expiring-credit" if expires_in is None or not imminent and expires_in > state.policy.salvage_hours * HOUR else
                 "unsupported-window" if not limits.keys() >= set(server) else
                 "incomplete-coverage" if not clears >= blockers or program == JUNIPER and blockers - {FIVE_HOUR} else
                 "unsupported-window" if not covered else
                 "window-mostly-free" if not fullest or not imminent and fullest[1] < SALVAGE_USED else
                 "no-blocked-window" if extra["requires_limit"] and not server and not set(covered).intersection(exhausted)
                 else None)
        if not error:
            episode = f"salvage|{grant}|{minute(expiry)}"
            error = state.gated(account, episode, imminent=imminent, account_defer=True)
            if not error:
                salvages.append(ResetAction(target, account, episode, label, "expiring-credit", count, active,
                                            credit_id, expires_in, salvage_window=fullest[0], salvage_used=fullest[1], **extra))
        if error:
            skipped.append((account, "expiring-credit", error))
    return combine(restores, salvages, skipped)


def combine(restores, salvages, skipped):
    restore = min(restores, key=lambda item: item[0])[1] if restores else None
    if restore:
        salvages = [action for action in salvages if action.account_key != restore.account_key]
        skipped = [item for item in skipped if item[:2] != (restore.account_key, "expiring-credit")]
    return ([restore] if restore else []) + sorted(salvages, key=lambda action: action.expires_in), skipped


def headless_actions(actions):
    return [action for action in actions if action.expires_in is not None and action.expires_in <= IMMINENT]
