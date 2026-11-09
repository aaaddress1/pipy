"""Pin the Responses reasoning baseline and replay effort changes in order."""

import json
from hashlib import sha256
from threading import Lock

import model_catalog


def supported(route):
    model = route.catalog_model or {}
    return (route.api in model_catalog.RESPONSES_APIS and model.get("reasoningMode") != "pro"
            and (model.get("compat") or {}).get("supportsConfigurationUpdate") is True)


def anchor(items, index):
    if index == 0:
        return b""
    item = {key: value for key, value in items[index - 1].items() if key not in ("id", "status")}
    return sha256(json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).digest()


class EffortControl:
    def __init__(self):
        self.lock = Lock()
        self.model = self.base = self.current = None
        self.transitions = {}

    def copy(self):
        result = EffortControl()
        with self.lock:
            result.model, result.base, result.current = self.model, self.base, self.current
            result.transitions = self.transitions.copy()
        return result

    def apply(self, body):
        reasoning = body.get("reasoning") or {}
        requested = reasoning.get("effort")
        if requested not in model_catalog.EFFORTS or reasoning.get("mode") == "pro":
            return
        with self.lock:
            items = body["input"]
            if self.model != body["model"] or self.base is None or any(
                index > len(items) or saved != anchor(items, index)
                for index, (saved, _) in self.transitions.items()
            ):
                self.model = body["model"]
                self.base = self.current = requested
                self.transitions.clear()
                return

            if requested != self.current:
                index = len(items) - (bool(items) and items[-1].get("role") == "user")
                previous = max((position for position in self.transitions if position < index), default=None)
                preceding = self.base if previous is None else self.transitions[previous][1]
                if requested == preceding:
                    self.transitions.pop(index, None)
                else:
                    self.transitions[index] = (anchor(items, index), requested)
                self.current = requested
            for offset, (index, (_, level)) in enumerate(sorted(self.transitions.items())):
                items.insert(index + offset, {"type": "configuration_update", "reasoning": {"effort": level}})
            body["reasoning"] = {**reasoning, "effort": self.base}
