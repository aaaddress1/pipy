"""Bounded turn recovery for the Python agent loop."""

import math
import random
import sys
import time
from dataclasses import replace

from httpx import NetworkError, RemoteProtocolError, TimeoutException
from openai import APIConnectionError, APIStatusError

from auth_broker import ModelCredentialError, ModelQuotaError
from rate_limits import QUOTA_GUESS, TRANSIENT_DELAYS, UsageBlock, classify_rate_limit, quota_wait, retry_after

TRANSPORT_ERRORS = (APIConnectionError, NetworkError, RemoteProtocolError, TimeoutException)
MAX_RECOVERIES = 3


class StepLimitReached(RuntimeError):
    pass


class IncompleteTurn(RuntimeError):
    def __init__(self, message, *, text="", tool_draft=False, usage=None):
        super().__init__(message)
        self.text = text
        self.tool_draft = tool_draft
        self.usage = usage


class StreamInterrupted(IncompleteTurn):
    unexecuted_turn = None


class ClassifierRefusal(IncompleteTurn):
    def __init__(self, message, *, safe, credit, unexecuted_turn=None, refused_model=None):
        super().__init__(message)
        self.safe = safe
        self.credit = credit
        self.unexecuted_turn = unexecuted_turn
        self.refused_model = refused_model


class EmptyTurn(IncompleteTurn):
    pass


class ContextWindowExceeded(IncompleteTurn):
    pass


RECOVERABLE_ERRORS = (IncompleteTurn, APIStatusError, ModelCredentialError)


def context_overflow(error):
    if error.status_code not in (400, 422):
        return False
    if error.code in ("context_length_exceeded", "context_window_exceeded", "context_too_large"):
        return True
    message = error.message.lower()
    return any(phrase in message for phrase in (
        "maximum context length", "context window exceeded", "exceeds the context window",
        "exceeds the available context", "context_length_exceeded", "prompt is too long",
    ))


def media_rejection(error):
    if error.status_code == 413:
        return True
    if error.status_code not in (400, 422):
        return False
    if error.code in ("image_too_large", "too_many_images", "invalid_image", "unsupported_image"):
        return True
    return any(phrase in error.message.lower() for phrase in (
        "too many images", "image exceeds", "image is too large", "images are not supported", "does not support images",
        "image byte limit", "request body too large", "unsupported image",
    ))


class TurnRecovery:
    def __init__(self, session, compaction, router, rate_limit_walks_chain, bind_context, max_retries, max_delay,
                 wait=time.sleep, auto_reset=None, wait_for_usage_reset=False, base_delay=.5, retry_enabled=True):
        self.session = session
        self.compaction = compaction
        self.router = router
        self.rate_limit_walks_chain = rate_limit_walks_chain
        self.bind_context = bind_context
        self.max_retries = max_retries
        self.max_delay = max_delay
        self.wait = wait
        self.auto_reset = auto_reset
        self.wait_for_usage_reset = wait_for_usage_reset
        self.base_delay = base_delay
        self.retry_enabled = retry_enabled
        self.fallback_cooldown = None
        self.request_system = None
        self.attempts = {}
        self.visited_routes = set()
        self.attempted_credentials = set()
        self.pending_credit = None

    def maintain(self, system):
        self.bind_context()
        while (
            self.router.route.promote_to
            and self.compaction.threshold is not None
            and self.compaction.estimate(self.compaction.messages(system)) > self.compaction.threshold
            and self.promote()
        ):
            self.bind_context()
        self.compaction.maintain(system)

    def fallback(self, usage_limit=False, *, signer_can_retry=False, credit_targets=(), pinned=False):
        if not self.retry_enabled:
            return None
        deferred = None
        route = self.router.route
        if (signer_can_retry or credit_targets) and route.api == "anthropic-messages":
            latest = next((message for _, batch in reversed(self.session.batches)
                           for message in reversed(batch) if message["role"] == "assistant"), {})
            native = latest.get("_anthropic") or {}
            signer = native.get("served_model", route.model)
            if native.get("route") == route.identity() and (credit_targets or signer == route.model) and any(
                block["type"] == "redacted_thinking"
                or block["type"] == "thinking"
                and isinstance(signature := block.get("signature"), str) and signature.strip()
                for block in native.get("blocks", ())
            ):
                deferred = route if signer == route.model else replace(route, model=signer)
        options = {"credit_targets": credit_targets} if credit_targets else {}
        if (profile := getattr(self.router, "profile", None)) and profile.chains:
            options.update(cooldown=self.fallback_cooldown, pinned=pinned)
            if self.request_system is not None:
                options["context_tokens"] = self.compaction.estimate(self.compaction.messages(self.request_system))
        if not self.router.fallback(self.visited_routes, usage_limit, deferred, **options):
            return None
        self.attempts.clear()
        self.attempted_credentials.clear()
        return "retry"

    def consume_credit(self):
        pending, self.pending_credit = self.pending_credit, None
        if pending:
            target, leaf, credit = pending
            if self.router.route == target and self.session.leaf == leaf and credit.live():
                return credit
        return None

    def remember_unexecuted(self, error):
        assistant = error.unexecuted_turn
        self.session.append([assistant, *[
            {"role": "tool", "tool_call_id": call["id"], "_tool_error": True,
             "content": "Tool call was not executed because the provider stream ended with an error "
                        f"before the tool could run: {error}",
             "_tool_details": {"__synthetic": True, "source": "assistant_stop_error",
                               "executed": False, "upstreamError": str(error)}}
            for call in assistant["tool_calls"]
        ]])

    def take(self, reason, limit=MAX_RECOVERIES):
        count = self.attempts.get(reason, 0) + 1
        self.attempts[reason] = count
        return count if count <= limit else None

    def restore_quota(self, hint=None):
        if not self.auto_reset or type(self.router.credential) is not int:
            return None
        started = time.monotonic()
        for attempt in range(self.max_retries + 1):
            restored, retry_after_read = self.auto_reset(hint)
            if restored:
                self.attempted_credentials.clear()
                return "retry"
            if retry_after_read is None or attempt >= self.max_retries:
                break
            delay = max(1, retry_after_read, min(2 ** min(attempt, 8), 120))
            if self.max_delay and time.monotonic() - started + delay > self.max_delay:
                break
            self.wait(delay)
        return None

    def wait_quota(self, block, has_hint, attempt, retry_allowed=True, *, rate_limited=True):
        if attempt is None or not retry_allowed:
            return self.fallback(usage_limit=True)
        active, prior, sibling = self.router.quota_availability()
        block = UsageBlock(block.requested, max((value for value in (block.actual, active) if value is not None),
                                                default=None),
                           block.report, max((value for value in (block.prior_timed, prior) if value is not None),
                                             default=None))
        now = time.time() * 1000
        deadline, earlier, authoritative = quota_wait(block, sibling, now, has_hint=has_hint)
        if deadline is None:
            return self.fallback(usage_limit=True)
        wait = max(0, (deadline - now) / 1000)
        delay = max(wait, retry_backoff(attempt, self.base_delay))
        self.fallback_cooldown = wait or None
        over_cap = bool(self.max_delay and delay > self.max_delay)
        # A sibling whose block itself fits the cap can have one extra second to re-rank.
        sibling_within_cap = earlier and (not self.max_delay or sibling - now <= self.max_delay * 1000)
        can_wait = not over_cap or self.wait_for_usage_reset and authoritative and delay <= wait
        if not sibling_within_cap:
            walks_chain = rate_limited and self.rate_limit_walks_chain
            if switched := self.fallback(usage_limit=True, signer_can_retry=can_wait and not walks_chain):
                return switched
        if not (can_wait or sibling_within_cap):
            return None
        print(f"[provider quota; retrying {attempt}/{self.max_retries} in {delay:g}s]", file=sys.stderr)
        target = now + delay * 1000
        while (remaining := (target - time.time() * 1000) / 1000) > 0:
            self.wait(min(remaining, 1))
        self.attempted_credentials.clear()
        return "retry"

    def recover_quota(self, hint, kind, retry_allowed, *, rate_limited):
        now = time.time() * 1000
        deadline = now + (hint if hint is not None else QUOTA_GUESS) * 1000
        block = self.router.mark_quota(deadline, hint is not None)
        self.fallback_cooldown = hint if hint is not None else QUOTA_GUESS
        attempt = self.take("provider quota", self.max_retries)
        if attempt is not None and self.router.select_alternate(self.attempted_credentials):
            return "retry"
        if kind == "billing":
            return self.fallback(usage_limit=True)
        restored = self.restore_quota(int(deadline) if hint is not None else None)
        return restored or self.wait_quota(block, hint is not None, attempt, retry_allowed, rate_limited=rate_limited)

    def recover(self, error, system, messages):
        """Return 'retry', 'finish', or None to surface the original failure."""
        self.fallback_cooldown = None
        self.request_system = system
        if getattr(error, "replay_unsafe", False):
            return None
        if isinstance(error, ClassifierRefusal):
            if error.unexecuted_turn:
                self.remember_unexecuted(error)
            if not error.safe:
                return None
            source, credit = self.router.route, error.credit
            targets = source.fallback_credit_targets(error.refused_model) if credit and credit.live() else ()
            if not self.fallback(credit_targets=targets, pinned=True):
                return None
            target = self.router.route
            if credit and credit.live() and source.can_redeem_credit(target, targets):
                self.pending_credit = (target, self.session.leaf, credit)
            return "retry"
        if isinstance(error, ModelQuotaError):
            if not self.retry_enabled:
                return None
            if restored := self.restore_quota():
                return restored
            attempt = self.take("provider quota", self.max_retries)
            active, timed, _ = self.router.quota_availability()
            return self.wait_quota(UsageBlock(None, active, prior_timed=timed), timed is not None, attempt)
        if isinstance(error, ModelCredentialError):
            if not self.retry_enabled:
                return None
            return "retry" if self.router.rotate(math.inf, self.attempted_credentials) else self.fallback()
        if isinstance(error, ContextWindowExceeded) or (
            isinstance(error, APIStatusError) and context_overflow(error) and not media_rejection(error)
        ):
            if isinstance(error, ContextWindowExceeded):
                self.compaction.observe(messages, error.usage)
            if self.promote() or self.take("overflow") and self.compaction.maintain(system, reason="overflow"):
                return "retry"
            return None
        if isinstance(error, APIStatusError):
            if media_rejection(error):
                if self.take("media rejection", 1) and self.compaction.recover_media(system):
                    return "retry"
                return self.fallback()
            if not self.retry_enabled:
                return None
            status = error.status_code
            should_retry = error.response.headers.get("x-should-retry")
            provider = self.router.route.broker_provider
            hint = retry_after(error.response, error, provider=provider)
            self.fallback_cooldown = hint or None
            if status in (402, 403, 429):
                delay = hint
                reason = classify_rate_limit(error, delay)
                if reason in ("quota", "billing"):
                    return self.recover_quota(delay, reason, should_retry != "false", rate_limited=status == 429)
            if status == 401 or (
                status == 403 and error.code in ("insufficient_permissions", "account_disabled", "invalid_api_key")
            ):
                return "retry" if self.router.rotate(math.inf, self.attempted_credentials) else self.fallback()
            if status == 429:
                if delay is None:
                    delay = TRANSIENT_DELAYS.get(reason)
                    if reason == "capacity":
                        delay += random.uniform(0, 30)
                return self.retry_or_fallback(
                    "provider rate limit", delay, retry_allowed=should_retry != "false", rate_limited=True,
                )
            if should_retry == "true" or status in (408, 409) or 500 <= status <= 599:
                return self.retry_or_fallback(
                    "transient provider error", hint,
                    retry_allowed=should_retry != "false",
                )
            if status == 404:
                return self.fallback()
            return None

        if isinstance(error, StreamInterrupted):
            if error.unexecuted_turn:
                self.remember_unexecuted(error)
                return self.retry_or_fallback("stream interrupted", retry_first=True) if self.retry_enabled else None
            if not self.retry_enabled:
                return None
            if not error.text.strip() or error.tool_draft:
                return self.retry_or_fallback("stream interrupted")
            attempt = self.take("visible stream interruption")
            if attempt:
                self.remind(
                    "The provider interrupted the previous response. The user already saw its text. "
                    "Continue exactly where it stopped without repeating it.",
                    text=error.text,
                )
                print(f"[stream interrupted after visible text; continuing {attempt}/{MAX_RECOVERIES}]", file=sys.stderr)
                return "retry"
            self.remind(
                "The previous response was interrupted. Continue where its visible text stopped.",
                text=error.text,
            )
            return None

        if isinstance(error, EmptyTurn):
            if self.take("empty stop"):
                self.remind("The previous turn stopped without actionable output. Provide an answer or use a tool.")
                return "retry"
            return None

        self.compaction.observe(messages, error.usage)
        threshold = self.compaction.threshold
        below_threshold = threshold is not None and self.compaction.estimate(messages) <= threshold
        if below_threshold and error.text.strip() and not error.tool_draft:
            self.session.append([{"role": "assistant", "content": error.text}])
            print("[generation limit reached; kept the visible response]", file=sys.stderr)
            return "finish"
        if not below_threshold and self.promote():
            return "retry"
        if self.take("incomplete output"):
            if not below_threshold and self.compaction.maintain(system, reason="incomplete"):
                return "retry"
            if below_threshold:
                if self.attempts["incomplete output"] == 1:
                    self.remind("The model hit its output limit. Continue in smaller steps.")
                return "retry"
        if error.text.strip() and not error.tool_draft:
            self.remind("The previous response hit its output limit. Continue where its visible text stopped.", text=error.text)
        return None

    def promote(self):
        if not self.router.promote(self.visited_routes):
            return False
        self.attempts.clear()
        self.attempted_credentials.clear()
        return True

    def retry_or_fallback(self, reason, provider_delay=None, *, retry_allowed=True, rate_limited=False, retry_first=False):
        attempt = self.take(reason, self.max_retries) if retry_allowed else None
        delay = max(provider_delay or 0, retry_backoff(attempt, self.base_delay)) if attempt is not None else None
        if self.fallback_cooldown is None and provider_delay:
            self.fallback_cooldown = provider_delay
        can_wait = delay is not None and (not self.max_delay or delay <= self.max_delay)
        if not (retry_first and attempt == 1 and can_wait):
            switched = self.fallback(signer_can_retry=can_wait and not (rate_limited and self.rate_limit_walks_chain))
            if switched or not can_wait:
                return switched
        print(f"[{reason}; retrying {attempt}/{self.max_retries} in {delay:g}s]", file=sys.stderr)
        self.wait(delay)
        return "retry"

    def remind(self, instruction, *, text=""):
        messages = [{"role": "assistant", "content": text}] if text else []
        messages.append({"role": "developer", "content": instruction})
        self.session.append(messages)


def retry_backoff(attempt, base=.5):
    return random.uniform(0.75, 1) * min(8, base * 2 ** min(attempt - 1, 1023))
