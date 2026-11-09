"""Identify perishable reset credits and claim their one-time terminal notices."""

import re
import sqlite3
import unicodedata
from contextlib import closing
from dataclasses import dataclass
from math import floor

from auto_reset_plan import (
    CLAUDE, CODEX, HOUR, WEEK, SALVAGE_USED, claude_covered, codex_windows, fullest_claude_window,
    fullest_codex_window, minute,
)
from auth_broker_cache import number
from local_oauth import text
from provider_quota import iso_ms


NOTICE_LEAD = 6 * HOUR
ANSI = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[@-_])")


@dataclass(frozen=True)
class ResetExpiry:
    provider: str
    tier: str
    count: int
    expires_at: int
    usable: bool
    window: str
    used: float


def expiry_warning(report, now):
    provider = report.get("provider")
    if provider not in (CLAUDE, CODEX):
        return None
    inventory = report.get("resetCredits") or {}
    raw = inventory.get("credits")
    if not isinstance(raw, list):
        return None
    limits = {limit["id"]: limit for limit in report["limits"]}
    codex = fullest_codex_window(codex_windows(report)) if provider == CODEX else None
    worth = []
    for credit in raw:
        if not isinstance(credit, dict) or credit.get("status") == "redeemed" or credit.get("remainingCount") == 0:
            continue
        expiry = iso_ms(credit.get("expiresAt"))
        if expiry is None or not 0 < expiry - now <= WEEK:
            continue
        if provider == CLAUDE:
            window = fullest_claude_window(limits, claude_covered(limits, credit), credit)
        else:
            window = (codex[4], codex[1]) if codex else None
        if window and window[1] >= SALVAGE_USED:
            worth.append((expiry, credit, *window))
    if not worth:
        return None
    expiry, credit, window, used = min(worth, key=lambda row: row[0])
    imminent = expiry - now <= 24 * HOUR
    horizon = 24 * HOUR if imminent else WEEK
    count = sum(int(remaining) if number(remaining := grant.get("remainingCount")) else 1
                for expires, grant, _, _ in worth if expires - now <= horizon)
    available = inventory.get("redeemableCount", inventory.get("availableCount"))
    usable = (inventory.get("eligible") is not False and number(available) and available > 0
              and credit.get("status", "available") == "available" and credit.get("usable") is not False
              and (provider == CODEX or "nextCreditId" in inventory and inventory["nextCreditId"] == credit.get("id")))
    return ResetExpiry(provider, "imminent" if imminent else "soon", count, expiry, usable, window, used)


def report_identity(report):
    metadata = report["metadata"]

    def get(field):
        return text(metadata.get(field)) or next((value for limit in report["limits"]
                                                  if (value := text(limit["scope"].get(field)))), None)

    email = text(metadata.get("email"))
    account, org = get("accountId"), get("orgId")
    base = email or account or get("projectId")
    organization = text(metadata.get("orgName")) or org
    label = f"{base} ({organization})" if base and organization and organization != base else base
    identity = account or email
    key = f"{report['provider']}|{org.lower() if org else '-'}|{identity.lower()}" if identity else report["provider"]
    return key, label


def duration(ms):
    if ms < 1000:
        return f"{max(0, int(ms))}ms"
    if ms < 60_000:
        tenths = floor(ms / 100 + .5)
        return f"{tenths / 10:.1f}s" if tenths < 600 else "1m"
    minutes, seconds = divmod(int(ms) // 1000, 60)
    if minutes < 60:
        return f"{minutes}m" + (f"{seconds}s" if seconds else "")
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h" + (f"{minutes}m" if minutes else "")


def terminal_text(value):
    return " ".join("".join(char if char.isprintable() else " " for char in ANSI.sub("", value)).split())


def short_label(value):
    label = terminal_text(value)

    def cells(char):
        return 0 if unicodedata.combining(char) else 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1

    if sum(map(cells, label)) <= 80:
        return label
    width = 0
    for offset, char in enumerate(label):
        width += cells(char)
        if width > 79:
            return label[:offset] + "…"
    return label


def claim_expiry_notice(path, reports, now):
    due = []
    for report in reports:
        warning = expiry_warning(report, now)
        if warning and warning.expires_at - now <= NOTICE_LEAD:
            identity, label = report_identity(report)
            due.append((f"reset-expiry|{identity}|{minute(warning.expires_at)}", warning, label))
    if not due:
        return None
    claimed = []
    with closing(sqlite3.connect(path.as_uri() + "?mode=rw", uri=True, timeout=2)) as db, db:
        db.execute("CREATE TABLE IF NOT EXISTS notice_claims(key TEXT PRIMARY KEY, expires_at_ms INTEGER NOT NULL)")
        db.execute("DELETE FROM notice_claims WHERE expires_at_ms <= ?", (now,))
        for key, warning, label in due:
            if db.execute("INSERT OR IGNORE INTO notice_claims(key,expires_at_ms) VALUES(?,?)",
                          (key, warning.expires_at)).rowcount:
                claimed.append((warning, label))
    if not claimed:
        return None
    warning, label = min(claimed, key=lambda item: item[0].expires_at)
    label = short_label(label or "")
    account = f" on {label}" if label else ""
    provider = "Claude" if warning.provider == CLAUDE else "Codex"
    remaining = duration(warning.expires_at - now)
    if warning.count == 1:
        message = f"Saved {provider} reset{account} expires in {remaining}"
    else:
        message = f"{warning.count} saved {provider} resets{account} expire, soonest in {remaining}"
    if len(claimed) > 1:
        message += f" (and {len(claimed) - 1} more {'account' if len(claimed) == 2 else 'accounts'})"
    return message + " · /usage reset"
