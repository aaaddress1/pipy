"""Account-scoped stored continuation for the HTTP Responses transport."""

import os
import re
from copy import deepcopy
from urllib.parse import urlsplit

from responses_websocket import delta_input, stale_anchor


def env_flag(name, default):
    value = os.getenv(name, "").strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    return default


def stored_host(route):
    compat = (route.catalog_model or {}).get("compat") or {}
    return route.api == "openai-responses" and compat.get("storeResponses") is True and route.store_responses


def stateful_enabled(route, *, websocket=False):
    if route.api != "openai-responses":
        return False
    compat = (route.catalog_model or {}).get("compat") or {}
    if not websocket and compat.get("storeResponses") is True and not stored_host(route):
        return False
    if route.stateful_responses is not None:
        return route.stateful_responses
    url = urlsplit(route.endpoint())
    official = url.scheme == "https" and url.hostname == "api.openai.com" and url.port in (None, 443)
    default = compat.get("statefulResponses")
    if default is None:
        default = websocket or (official and compat.get("officialEndpoint", True))
    return env_flag("PI_OPENAI_STATEFUL", default)


class ResponsesChain:
    def __init__(self):
        self.baseline = None
        self.stale_failures = 0
        self.disabled = False
        self.zero_retention = False

    def reset(self):
        self.baseline = None

    def request(self, body):
        full = {**body, "store": not self.zero_retention and (body.get("store") is True or not self.disabled)}
        baseline, self.baseline = self.baseline, None
        delta = delta_input(baseline, full) if not self.disabled else None
        if delta is None:
            return full, full
        return full, {**full, "input": delta, "previous_response_id": baseline[0]}

    def retry(self, wire, error, status):
        if status not in (None, 400, 403, 404, 422):
            return False
        message = str(error.get("message") or "")
        code = error.get("code")
        chained = bool(wire.get("previous_response_id"))
        zero_retention = wire.get("store") is True and (
            code == "zero_data_retention"
            or re.search(r"zero[ _-]?data[ _-]?retention", message, re.I)
            and (chained or re.search(r"stor(?:e|ing|age)", message, re.I))
        )
        if zero_retention:
            self.disabled = self.zero_retention = True
            self.stale_failures = 3
            return True
        if chained and (stale_anchor(error) or code == "invalid_prompt"
                        or re.search(r"invalid_prompt|request blocked", message, re.I)):
            self.stale_failures += 1
            self.disabled = self.stale_failures >= 3
            return True
        return False

    def complete(self, full, chained, response, message):
        items = (message.get("_responses") or {}).get("items")
        identifier = response.get("id")
        if (not self.disabled and full["store"] and isinstance(identifier, str) and identifier and items
            and len(items) == len(response["output"])):
            self.baseline = deepcopy((identifier, full, items))
            if chained:
                self.stale_failures = 0
