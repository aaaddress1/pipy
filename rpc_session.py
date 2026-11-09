"""Session inspection and maintenance commands for JSONL RPC."""

import base64
import json
import re
from collections import Counter

from cost_gate import entry_usage, total_cost
from rpc_mode import API_NAMES, RpcFailure, encoded, wire_message
from runtime import database
from session_jsonl import journal_entries


def model_info(route):
    prices = route.cache_prices
    catalog = route.catalog_model or {}
    discovered = catalog.get("_discovered")
    qualified = discovered and route.name == f"{catalog['provider']}/{catalog['id']}"
    return {
        "id": route.model, "name": catalog.get("name", route.name) if qualified else route.name,
        "api": API_NAMES[route.api], "provider": catalog["provider"] if qualified else route.name,
        "baseUrl": route.endpoint(), "reasoning": bool(catalog.get("reasoning") if discovered else route.reasoning_effort),
        "input": ["text", "image"] if route.vision else ["text"], "contextWindow": route.window,
        "maxTokens": route.max_output_tokens if discovered or route.api != "chat-completions" else None,
        "cost": {"input": prices.input if prices else 0, "output": prices.output if prices else 0,
                 "cacheRead": prices.read if prices else 0, "cacheWrite": prices.write_5m if prices else 0},
    }


def message_page(messages, session_id, leaf, command):
    limit = command.get("limit", 100)
    if type(limit) is not int or not 1 <= limit <= 256:
        raise RpcFailure("RPC message page limit must be between 1 and 256.")
    snapshot = {"version": 1, "sessionId": session_id, "leafId": leaf, "messageCount": len(messages)}
    offset = 0
    if "cursor" in command:
        value = command["cursor"]
        try:
            if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,2048}", value):
                raise ValueError
            raw = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
            cursor = json.loads(raw)
            if not isinstance(cursor, dict) or base64.urlsafe_b64encode(raw).decode().rstrip("=") != value:
                raise ValueError
            offset = cursor.pop("offset", None)
            count = cursor.get("messageCount")
            if cursor.get("version") != 1 or type(offset) is not int or type(count) is not int or not 0 <= offset <= count:
                raise ValueError
        except ValueError as error:
            raise RpcFailure("Invalid RPC message cursor.") from error
        if cursor != snapshot:
            raise RpcFailure("RPC message cursor is stale.", "stale_cursor")
    page = []
    size = 2
    for message in messages[offset:offset + limit]:
        size += len(encoded(message)) + bool(page)
        if page and size > 768 * 1024:
            break
        page.append(message)
    result = {"messages": page, "totalMessages": len(messages)}
    if offset + len(page) < len(messages):
        cursor = encoded({**snapshot, "offset": offset + len(page)})
        result["nextCursor"] = base64.urlsafe_b64encode(cursor).decode().rstrip("=")
    return result


class SessionRpc:
    def __init__(self, rpc, session, scope, router, compaction, system_prompt, schemas, bind_context, thinking):
        self.rpc, self.session, self.scope, self.router = rpc, session, scope, router
        self.compaction, self.system_prompt, self.schemas, self.bind_context = compaction, system_prompt, schemas, bind_context
        self.thinking = thinking

    def handlers(self):
        return {
            "get_state": (self.state, False),
            "get_messages": (lambda _: {"messages": self.messages()}, False),
            "get_messages_page": (self.page, True),
            "get_entries": (self.entries, True),
            "get_tree": (self.tree, True),
            "get_branch_messages": (self.branch_messages, True),
            "get_last_assistant_text": (self.last_text, False),
            "get_session_stats": (self.stats, True),
            "get_available_models": (self.models, False),
            "set_model": (self.set_model, True),
            "set_auto_compaction": (self.set_auto, True),
            "set_label": (self.set_label, True),
            "set_session_name": (self.set_name, True),
        }

    def usage(self):
        window = self.compaction.window
        if window:
            tokens = self.compaction.estimate(self.compaction.messages(self.system_prompt()))
            return {"tokens": tokens, "contextWindow": window, "percent": 100 * tokens / window}
        return None

    def state(self, _):
        with self.rpc.lock:
            streaming = self.rpc.turn is not None
            queued = self.rpc.queues()
            pending = self.compaction.pending
            result = {
                "sessionId": self.session.id, "model": model_info(self.router.route), "isStreaming": streaming,
                "isCompacting": bool(pending and not pending.future.done()), "isSettled": not streaming,
                "steeringMode": self.rpc.steering_mode, "followUpMode": self.rpc.follow_up_mode,
                "autoCompactionEnabled": self.compaction.auto_enabled, "queuedMessages": queued,
                "queuedMessageCount": len(queued["steering"]) + len(queued["followUp"]),
                "messageCount": sum(len(batch) for _, batch in self.session.batches) + bool(self.session.summary),
                "dumpTools": [tool["function"] for tool in self.schemas()],
            }
            configured, resolved = self.thinking()
            result.update(thinkingLevel=resolved or "off", configuredThinkingLevel=configured or "off")
        if title := self.session.title():
            result["sessionName"] = title
        if not streaming and (usage := self.usage()):
            result["contextUsage"] = usage
        return result

    def messages(self):
        call_names = {}
        # Capture the batch order before serializing.
        batches = tuple(self.session.batches)
        summary = self.session.summary
        messages = [wire_message(message, call_names) for _, batch in batches for message in batch]
        if summary:
            messages.insert(0, {"role": "compactionSummary", "summary": summary, "tokensBefore": 0})
        return messages

    def page(self, command):
        leaf = self.session.leaf
        return message_page(self.messages(), self.session.id, str(leaf) if leaf is not None else None, command)

    def journal(self):
        ids = {}
        with database(self.session.path) as db:
            records = iter(journal_entries(db, self.session.id, self.scope, self.session.leaf, ids))
            next(records)  # reserved title slot
            next(records)  # session header
            entries = [{key: value for key, value in entry.items() if key != "miniPi"} for entry in records]
        return entries[:-1], entries[-1]["activeLeafId"], ids

    def entries(self, command):
        entries, leaf, _ = self.journal()
        if "since" in command:
            since = command["since"]
            for index, entry in enumerate(entries):
                if entry["id"] == since:
                    entries = entries[index + 1:]
                    break
            else:
                raise RpcFailure(f"Unknown entries cursor: {since}", "unknown_since")
        return {"entries": entries, "leafId": leaf}

    def tree(self, _):
        entries, leaf, _ = self.journal()
        roots, nodes = [], {}
        for entry in entries:
            if entry["type"] == "label":
                if node := nodes.get(entry["targetId"]):
                    node["label"] = entry["label"]
            else:
                node = {"entry": entry, "children": []}
                nodes[entry["id"]] = node
                parent = nodes.get(entry["parentId"])
                (parent["children"] if parent else roots).append(node)
        return {"tree": roots, "leafId": leaf}

    def target(self, command):
        identifier = command.get("entryId")
        _, _, ids = self.journal()
        if not isinstance(identifier, str) or identifier not in ids:
            raise RpcFailure(f"Entry {identifier} not found.", "entry_not_found")
        return ids[identifier]

    def branch_messages(self, _):
        entries, _, _ = self.journal()
        messages = []
        for entry in entries:
            message = entry.get("message")
            if message and message.get("role") == "user":
                if text := self.text(message):
                    messages.append({"entryId": entry["id"], "text": text})
        return {"messages": messages}

    @staticmethod
    def text(message):
        content = message["content"]
        return content if isinstance(content, str) else "".join(block["text"] for block in content if block["type"] == "text")

    def last_text(self, _):
        text = next((self.text(message) for message in reversed(self.messages()) if message["role"] == "assistant"), None)
        return {"text": text}

    def stats(self, _):
        entries, _, _ = self.journal()
        messages = [entry["message"] for entry in entries if entry["type"] == "message"]
        counts = Counter(message["role"] for message in messages)
        assistants = [message for message in messages if message["role"] == "assistant"]
        usages = [entry_usage(entry) for entry in entries]
        tokens = {key: sum(usage.get(key, 0) for usage in usages)
                  for key in ("input", "output", "cacheRead", "cacheWrite")}
        result = {
            "sessionId": self.session.id, "userMessages": counts["user"], "assistantMessages": counts["assistant"],
            "toolCalls": sum(block["type"] == "toolCall" for message in assistants for block in message["content"]),
            "toolResults": counts["toolResult"], "totalMessages": len(messages),
            "tokens": {**tokens, "total": sum(tokens.values())}, "premiumRequests": 0,
            "cost": sum(total_cost(usage) for usage in usages),
        }
        if usage := self.usage():
            result["contextUsage"] = usage
        return result

    def models(self, _):
        return {"models": [model_info(route) for route in self.router.routes.values()]}

    def set_model(self, command):
        provider, identifier = command.get("provider"), command.get("modelId")
        if not isinstance(provider, str) or not isinstance(identifier, str):
            raise RpcFailure("A provider and modelId are required.")
        route = self.router.routes.get(provider)
        if not route or route.model != identifier:
            route = self.router.routes.get(f"{provider}/{identifier}")
        if not route or route.model != identifier:
            raise RpcFailure("The configured model route was not found.")
        if route != self.router.route:
            if not self.router.switch(route, "RPC model switch"):
                raise RpcFailure("No credential is available for the selected model route.")
        self.bind_context(refresh=True)
        return model_info(self.router.route)

    def set_auto(self, command):
        enabled = command.get("enabled")
        if type(enabled) is not bool:
            raise RpcFailure("enabled must be a boolean.")
        self.compaction.auto_enabled = enabled
        if not enabled:
            self.compaction.discard_pending()

    def set_name(self, command):
        name = command.get("name")
        if not isinstance(name, str) or not self.session.set_title(name):
            raise RpcFailure("Session name cannot be empty.")
        self.rpc.event("session_info_update", title=self.session.title(), sessionId=self.session.id)

    def set_label(self, command):
        label = command.get("label", "")
        if not isinstance(label, str):
            raise RpcFailure("label must be a string.")
        self.session.label(self.target(command), label)
