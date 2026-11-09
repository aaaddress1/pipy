"""Retry a compaction request without switching the foreground route."""

import math
import sys
from concurrent.futures import CancelledError
from dataclasses import dataclass

from httpx import TimeoutException
from openai import APIStatusError, APITimeoutError

from auth_broker import AuthBrokerError, ModelCredentialError
from cost_gate import CostCapError
from rate_limits import classify_rate_limit, retry_after
from recovery import StreamInterrupted, TRANSPORT_ERRORS, context_overflow


@dataclass(frozen=True)
class MaintenanceRetry:
    max_retries: int
    max_delay: float
    wait_for_usage_reset: bool
    base_delay: float = .5
    enabled: bool = True

    def delay(self, error, provider, attempt, has_more):
        if not self.enabled or attempt >= self.max_retries:
            return None
        cause = error
        while cause:
            if isinstance(cause, (TimeoutError, APITimeoutError, TimeoutException)):
                return None
            cause = cause.__cause__
        if isinstance(error, APIStatusError):
            status = error.status_code
            verdict = error.response.headers.get("x-should-retry")
            if status in (401, 408, 413) or context_overflow(error) or verdict == "false":
                return None
            hint = retry_after(error.response, error, provider=provider)
            reason = classify_rate_limit(error, hint)
            if reason == "billing" or status == 403 and reason != "quota":
                return None
            if hint is None and reason != "quota" and verdict != "true" and status not in (409, 425, 429) and status < 500:
                return None
        elif isinstance(error, (StreamInterrupted, *TRANSPORT_ERRORS)):
            hint = reason = None
        else:
            return None
        delay = max(self.base_delay * 2 ** min(attempt, 40), hint or 0)
        if not math.isfinite(delay):
            return None
        cap = 30 if has_more else self.max_delay
        opt_in = not has_more and self.wait_for_usage_reset and reason == "quota" and hint is not None and delay <= hint
        return delay if not cap or delay <= cap or opt_in else None

    def run(self, routes, credential, send, cancellation, *, retry=True):
        candidates = []
        for route in routes:
            cancellation.check()
            try:
                selected = credential(route)
            except AuthBrokerError:
                continue
            if selected is not None:
                candidates.append((route, selected))
        if not candidates:
            raise ModelCredentialError("No usable credential for compaction.")

        last_error = None
        for index, (route, selected) in enumerate(candidates):
            for attempt in range(self.max_retries + 1):
                cancellation.check()
                try:
                    result = send(route, selected)
                except (CancelledError, CostCapError):
                    raise
                except Exception as error:
                    cancellation.check()
                    last_error = error
                    delay = self.delay(error, route.broker_provider, attempt, index < len(candidates) - 1) if retry else None
                    if delay is None:
                        break
                    print(f"[compaction: retrying {route.name} in {delay:g}s]", file=sys.stderr)
                    cancellation.wait(delay)
                else:
                    cancellation.check()
                    return result
        raise last_error
