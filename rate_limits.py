"""Retry evidence and account availability for provider rate limits."""

import json
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime


QUOTA_GUESS = 1800
MAX_DELAY_SECONDS = (2**53 - 1) / 1000
TRANSIENT_DELAYS = {"concurrent": 5, "rate": 30, "capacity": 45}
CN_QUOTA = re.compile(r"(?:使用|用量).{0,30}?上限|(?:额度|配额)已?(?:用|耗)(?:完|尽)|限额.{0,30}重置|余额不足")
CN_THROTTLE = re.compile(r"(?:速率|频率|每分钟|并发).{0,30}上限|速率(?:限制|过快)|频率(?:过高|过快)|过于频繁|稍后[重再]试")
CONCURRENT = re.compile(
    r"too many\s+concurren\w*\s+(?:requests?|invocations?)|"
    r"\bconcurren\w*\b[^\n]{0,60}\b(?:limit|quota|exceed\w*|reach\w*)|"
    r"\b(?:limit|quota|exceed\w*|reach\w*)\b[^\n]{0,60}\bconcurren\w*|concurren\w*[-_](?:limit|quota)"
)
TPM_RPM = re.compile(
    r"\b(?:tpm|rpm)\b[^\n]{0,40}\b(?:exhaust|exceed|limit|throttl|reach)|"
    r"\b(?:exhaust\w*|exceed\w*|limit\w*|throttl\w*|reach\w*)\b[^\n]{0,40}\b(?:tpm|rpm)\b|"
    r"ratelimitexceeded\.(?:endpoint)?(?:tpm|rpm)"
)
BILLING = re.compile(
    r"\b(?:out of (?:extra )?(?:credits?|usage)|add (?:more )?credit|top ?up|insufficient[_ -]quota|"
    r"insufficient.?balance|usage credits are required|credits_required|usage_pricing_required)\b|"
    r"\bprepaid balance\b[^\n]{0,40}\b(?:used up|exhausted|depleted)|spend(?:ing)?[\s_-]?(?:limit|cap)\b|"
    r"\b(?:exceed\w*|insufficient|not enough)\b[^\n]{0,40}\bcredits?\b|"
    r"\bcredits?\b[^\n]{0,40}\b(?:exhausted|depleted)\b"
)
ACCOUNT = re.compile(
    r"\baccount(?:'s)?\b[^\n]{0,80}\brate.?limit\b|\brate.?limit\b[^\n]{0,80}\baccount\b|"
    r"free[-_ ]models[-_ ]per[-_ ]day|clinepass limit|free limit reached on model|usage_not_included|"
    r"access_terminated_error|personal-team-blocked"
)
SUBSCRIPTION = re.compile(
    r"\b(?:subscription|plan|membership)\b[^\n]{0,80}\b(?:rate.?limits?|quota|cap)\b|"
    r"\b(?:rate.?limits?|quota|cap)\b[^\n]{0,80}\b(?:subscription|plan|membership)\b"
)
RELATIVE = re.compile(
    r'(?:try again in|please retry in|resets?\s+in|"retryDelay":\s*")\s*~?\s*'
    r"(\d+(?:\.\d+)?)\s*(ms|sec|s|minutes?|mins?|m|hours?|hrs?|h|days?|d)\b", re.I
)
ABSOLUTE = re.compile(
    r"(?:reset at\s+|将在\s*)(\d{4}-\d\d-\d\d[ T]\d\d:\d\d:\d\d"
    r"(?:\.\d+)?(?:Z|[+-]\d\d:?\d\d)?)", re.I
)
TIMING_FIELDS = ("retry-after-ms", "retry-after", "x-ratelimit-reset-ms", "x-ratelimit-reset", "x-ratelimit-reset-after")
TIMING_TEXT = re.compile(r"\b(" + "|".join(TIMING_FIELDS) + r")\s*[:=]\s*([^\s,;]+)", re.I)


def numeric(value):
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def timestamp(value, zone=timezone.utc):
    if not isinstance(value, str):
        return None
    try:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            parsed = parsedate_to_datetime(value)
        return (parsed if parsed.tzinfo else parsed.replace(tzinfo=zone)).timestamp()
    except (TypeError, ValueError, OSError, OverflowError):
        return None


def header_delay(key, value, now):
    amount = numeric(value)
    if amount is None and key == "retry-after":
        date = timestamp(value)
        return max(0, date - now) if date is not None else None
    if amount is None or amount < 0:
        return None
    if key == "retry-after-ms":
        return amount / 1000
    if key.startswith("x-ratelimit-reset") and key != "x-ratelimit-reset-after":
        if amount > 1e12:
            return max(0, amount / 1000 - now)
        if amount > 1e9:
            return max(0, amount - now)
        return amount / 1000 if key.endswith("-ms") else amount
    return amount


def error_body(error):
    body = error.body
    if isinstance(body, dict):
        return body["error"] if isinstance(body.get("error"), dict) else body
    return {}


def error_text(error):
    if isinstance(error.body, (dict, list)):
        return json.dumps(error.body, ensure_ascii=False, default=str) + " " + error.message
    return " ".join(part for part in (error.body, error.message) if isinstance(part, str))


def delay_unit(unit):
    return .001 if unit == "ms" else {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit[0]]


def retry_after(response, body=None, *, provider=None, now=None):
    now = time.time() if now is None else now
    delays = [delay for key in TIMING_FIELDS if (delay := header_delay(key, response.headers.get(key), now)) is not None]
    if body is not None:
        text, details = error_text(body), error_body(body)
        delays += [delay for key, value in TIMING_TEXT.findall(text)
                   if (delay := header_delay(key.lower(), value, now)) is not None]
        delays += [float(value) * delay_unit(unit.lower()) for value, unit in RELATIVE.findall(text)]
        for h, m, s in re.findall(r"reset after\s+(?:(\d+)h)?(?:(\d+)m)?(\d+(?:\.\d+)?)s", text, re.I):
            delays.append(float(h or 0) * 3600 + float(m or 0) * 60 + float(s))
        for h, m in re.findall(r"resets? in\s+~?\s*(\d+(?:\.\d+)?)\s*hr\s*(\d+(?:\.\d+)?)\s*min\b", text, re.I):
            delays.append(float(h) * 3600 + float(m) * 60)
        relative = numeric(details.get("resets_in_seconds"))
        if relative is not None:
            delays.append(max(0, relative))
        raw = details.get("resets_at")
        absolute = numeric(raw)
        if absolute is not None and absolute > 1e12:
            absolute /= 1000
        if absolute is None:
            absolute = timestamp(raw)
        if absolute is not None:
            delays.append(max(0, absolute - now))
        naive = []
        china = provider in ("zai", "zhipu", "zai-coding-plan", "zhipuai")
        zone = timezone(timedelta(hours=8)) if china else timezone.utc
        for value in ABSOLUTE.findall(text):
            date = timestamp(value, zone)
            if date is not None and date > now:
                target = delays if china or re.search(r"(?:Z|[+-]\d\d:?\d\d)$", value, re.I) else naive
                target.append(date - now)
        delays = delays or naive
    return max((delay for delay in delays if 0 <= delay <= MAX_DELAY_SECONDS), default=None)


def google_limit(body, text, hint):
    if str(body.get("status", "")).strip().upper() != "RESOURCE_EXHAUSTED" or not isinstance(body.get("details"), list):
        return None
    rows = [row for row in body["details"] if isinstance(row, dict)]
    reasons = [str(row.get("reason", "")).strip().upper() for row in rows
               if row.get("@type") == "type.googleapis.com/google.rpc.ErrorInfo"]
    for reason in reasons:
        if reason in ("MODEL_CAPACITY_EXHAUSTED", "MODEL_CAPACITY_EXCEEDED"):
            return "capacity"
        if reason in ("QUOTA_EXHAUSTED", "INSUFFICIENT_G1_CREDITS_BALANCE"):
            return "quota"
        if reason == "RATE_LIMIT_EXCEEDED" and "exhausted your capacity on this model" in text:
            return "quota"
    short = "RATE_LIMIT_EXCEEDED" in reasons
    for row in rows:
        if row.get("@type") != "type.googleapis.com/google.rpc.QuotaFailure":
            continue
        violations = row.get("violations")
        for violation in violations if isinstance(violations, list) else ():
            if not isinstance(violation, dict):
                continue
            metric = f"{violation.get('quotaId', '')} {violation.get('quotaMetric', '')}".lower()
            if numeric(violation.get("quotaValue")) == 0 or re.search(r"per_?day|daily", metric):
                return "quota"
            short |= bool(re.search(r"per_?(minute|second|100_?seconds)", metric))
    if short or hint is not None and any(row.get("@type") == "type.googleapis.com/google.rpc.RetryInfo" for row in rows):
        return "quota" if hint is not None and hint >= 300 else "rate"
    return None


def classify_rate_limit(error, hint):
    """Return quota, billing, a transient rate reason, or None for unrelated HTTP errors."""
    status = error.status_code
    text = error_text(error).lower()
    body = error_body(error)
    if reason := google_limit(body, text, hint):
        return "billing" if reason == "quota" and "insufficient_g1_credits_balance" in text else reason
    if status not in (402, 403, 429):
        return None
    if status != 402 and CONCURRENT.search(text):
        return "concurrent" if status == 429 else None
    if "#token-limit" in text and "you exceeded your current quota, please check your plan and billing details" in text:
        return "rate"
    if BILLING.search(text):
        return "billing"
    stripped = re.sub(r"\b(?:429|402|http|https|status|error|code|response|message|no body)\b", "", text)
    opaque = not re.search(r"[a-z\d]{3,}", stripped) and not CN_THROTTLE.search(text) and not CN_QUOTA.search(text)
    quota = (ACCOUNT.search(text) or SUBSCRIPTION.search(text) and not re.search(r"\bper (?:second|minute)", text)
             or CN_QUOTA.search(text) and not CN_THROTTLE.search(text)
             or "quota will reset" in text or "exhausted your capacity" in text)
    if not quota and TPM_RPM.search(text):
        return "rate"
    quota = quota or re.search(r"usage.?limit|limit_reached|quota.?(?:exceed|reach|insufficien)|额度不足|额度耗尽", text)
    if status == 402:
        return "billing" if opaque or quota or CONCURRENT.search(text) or re.search(
            r"payment[\s_.-]+required|deactivated_workspace|insufficient.?funds", text,
        ) else None
    if status == 403:
        return "quota" if quota or re.search(
            r"\b(?:overall|account|organization|team|workspace)\b[^\n]{0,40}\brate.?limit|"
            r"\byour\b[^\n]{0,30}\b(?:limit )?will reset", text,
        ) else None
    if quota or opaque:
        return "quota"
    if re.search(r"capacity|overloaded|resource.?exhausted", text):
        return "capacity"
    if re.search(r"rate.?limit|too many requests|per (?:minute|second)|presque", text) or CN_THROTTLE.search(text):
        return "rate"
    return "quota" if "quota" in text or "exhausted" in text else None


@dataclass(frozen=True)
class UsageBlock:
    requested: float | None
    actual: float | None
    report: float | None = None
    prior_timed: float | None = None


def quota_wait(block, sibling, now, *, has_hint=False):
    """Resolve one account's longest gate, then choose an earlier sibling."""
    deadline = block.requested
    if block.report is not None and not has_hint:
        deadline = block.report
    for candidate in (block.prior_timed, block.actual if block.requested is None
                      or block.actual is not None and block.actual > block.requested else None):
        if candidate is not None:
            deadline = max(deadline, candidate) if deadline is not None else candidate
    earlier = sibling is not None and (deadline is None or sibling + 1000 < deadline)
    if earlier:
        deadline = sibling + 1000
    return (max(now, deadline) if deadline is not None else None), earlier, (has_hint or block.report is not None)
