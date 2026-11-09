"""Expose retained eval functions as child-session tools."""

import json
import re
from concurrent.futures import CancelledError, ThreadPoolExecutor
from threading import Lock
from urllib.parse import unquote
from uuid import uuid4
from weakref import WeakValueDictionary, ref

from jsonschema import Draft202012Validator
from referencing import Registry

from cancellable_http import active_turn
from eval_completion import CellCancellation
from handoff import Cancellation
from mcp_tools import ExternalToolResult
from subagents import Tool


OWNERS = WeakValueDictionary()
OWNER_LOCK = Lock()
LANGUAGES = ("python", "js")
DISABLED = "Eval-defined tools are disabled; set eval.tools.enabled=true to expose them to subagents."


def declares_intent(schema):
    visited = set()

    def resolve(reference):
        if not isinstance(reference, str) or not reference.startswith("#/"):
            return None
        target = schema
        for key in unquote(reference[2:]).split("/"):
            key = key.replace("~1", "/").replace("~0", "~")
            if isinstance(target, dict):
                target = target.get(key)
            elif isinstance(target, list) and key.isascii() and key.isdecimal() and len(key) <= 9 and int(key) < len(target):
                target = target[int(key)]
            else:
                return None
        return target

    def only_presence(node, seen):
        if (not isinstance(node, dict) or id(node) in seen or len(seen) >= 64
                or node.keys() - {"required", "type", "$ref", "description", "title", "$comment"}):
            return False
        if "$ref" in node:
            return not node.keys() & {"required", "type"} and only_presence(resolve(node["$ref"]), seen | {id(node)})
        required = node.get("required")
        return (node.get("type", "object") == "object" and isinstance(required, list)
                and "i" in required and all(isinstance(name, str) for name in required))

    def visit(node, context="positive"):
        visit_id = id(node), context
        if not isinstance(node, dict) or visit_id in visited:
            return False
        visited.add(visit_id)
        if (isinstance(node.get("const"), dict) and "i" in node["const"]
                or isinstance(node.get("enum"), list) and any(isinstance(item, dict) and "i" in item for item in node["enum"])):
            return True
        if isinstance(node.get("properties"), dict) and "i" in node["properties"]:
            return node["properties"]["i"] is not False or context != "positive"
        if isinstance(node.get("required"), list) and "i" in node["required"]:
            return True
        if "propertyNames" in node and node.get("additionalProperties") is not False:
            try:
                if Draft202012Validator(schema, registry=Registry()).evolve(schema=node["propertyNames"]).is_valid("i"):
                    return True
            except Exception:
                pass
        patterns = node.get("patternProperties")
        if isinstance(patterns, dict):
            for pattern, sub_schema in patterns.items():
                try:
                    if sub_schema is not False and re.search(pattern, "i"):
                        return True
                except re.error:
                    pass
        if any(isinstance(node.get(key), dict) for key in ("additionalProperties", "unevaluatedProperties")):
            return True
        if any(visit(branch, context) for key in ("oneOf", "anyOf", "allOf")
               if isinstance(node.get(key), list) for branch in node[key]):
            return True
        if visit(node.get("if"), "predicate") or any(visit(node.get(key), context) for key in ("then", "else")):
            return True
        negation = node.get("not")
        if context != "positive" or not only_presence(negation, set()):
            if visit(negation, "negative" if context == "positive" else "positive" if context == "negative" else context):
                return True
        for key in ("dependentSchemas", "dependentRequired", "dependencies"):
            group = node.get(key)
            if not isinstance(group, dict):
                continue
            if "i" in group:
                return True
            for dependency in group.values():
                if key != "dependentSchemas" and isinstance(dependency, list) and "i" in dependency:
                    return True
                if key != "dependentRequired" and visit(dependency, context):
                    return True
        return visit(resolve(node.get("$ref")), context)

    return visit(schema)


class KernelTools:
    def __init__(self, owner, enabled, request):
        self.owner = ref(owner)
        self.enabled, self.invoke = enabled, request
        self.id = uuid4().hex
        self.key = owner.session.path, self.id
        self.active = set()
        self.lock = Lock()
        self.closed = False
        with OWNER_LOCK:
            OWNERS[self.key] = self

    def request(self, language, message, parent):
        owner = self.owner()
        if owner is None:
            raise ValueError("The session whose eval kernel defined the tool has ended.")
        cancellation = CellCancellation(parent)
        with self.lock:
            if self.closed:
                raise ValueError("The session whose eval kernel defined the tool has ended.")
            invoke = self.invoke
            self.active.add(cancellation)
        try:
            cancellation.check()
            return owner.invoke_kernel_tool(language, "tool-" + uuid4().hex, message, invoke, cancellation)
        except CancelledError:
            cancellation.cancel()
            raise
        finally:
            cancellation.finish()
            with self.lock:
                self.active.discard(cancellation)

    def query(self, names, cancellation):
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="mini-pi-eval-describe") as workers:
            futures = [workers.submit(self.request, language, {"op": "describe", "names": names}, cancellation)
                       for language in LANGUAGES]
            by_name = {}
            for language, future in zip(LANGUAGES, futures):
                result = future.result()
                if not isinstance(result, dict) or result.get("ok") is not True or not isinstance(result.get("tools"), list):
                    error = result.get("error") if isinstance(result, dict) else None
                    raise ValueError(error or f"{language} tool describe returned an invalid response.")
                for descriptor in result["tools"]:
                    if not isinstance(descriptor, dict):
                        continue
                    name, description, parameters = (descriptor.get(key) for key in ("name", "description", "parameters"))
                    if not (isinstance(name, str) and isinstance(description, str) and isinstance(parameters, dict)):
                        continue
                    if name in by_name and by_name[name]["language"] != language:
                        raise ValueError(f'eval tool "{name}" is defined in both the Python and JS kernels; undefine one')
                    by_name[name] = {"name": name, "description": description, "parameters": parameters, "language": language}
        return by_name

    def describe(self, names, cancellation=None):
        names = list(dict.fromkeys(name for name in names if name))
        if not names:
            return ()
        if not self.enabled():
            raise ValueError(DISABLED)
        cancellation = cancellation or active_turn() or Cancellation()
        found = self.query(names, cancellation)
        missing = [name for name in names if name not in found]
        if missing:
            available = ", ".join(sorted(self.query([], cancellation), key=str.casefold)) or "none"
            raise ValueError(f"Unknown eval tool(s): {', '.join(missing)}. Define them with @tool (Python) or "
                             f"tool(fn, options) (JS) in an eval cell first. Available: {available}")
        return tuple(found[name] for name in names)

    def close(self):
        with OWNER_LOCK:
            if OWNERS.get(self.key) is self:
                OWNERS.pop(self.key, None)
        with self.lock:
            self.closed = True
            self.invoke = None
            active = tuple(self.active)
        for cancellation in active:
            cancellation.cancel()


def child_tools(path, task):
    owner = path, task.eval_tool_owner
    return {descriptor["name"]: child_tool(owner, descriptor) for descriptor in task.eval_tools}


def child_tool(owner, descriptor):
    name, language, parameters = (descriptor[key] for key in ("name", "language", "parameters"))
    has_intent = declares_intent(parameters)

    def execute(args):
        details = {"evalTool": name, "language": language}

        def result(text, *, error=False):
            return ExternalToolResult([{"type": "text", "text": text}], {**details, **({"isError": True} if error else {})}, error)

        with OWNER_LOCK:
            registry = OWNERS.get(owner)
        if registry is None:
            return result(f'The session whose {language} kernel defined "{name}" has ended.', error=True)
        args = args if isinstance(args, dict) else {}
        if "i" in args and not has_intent:
            args = {key: value for key, value in args.items() if key != "i"}
        try:
            reply = registry.request(language, {"op": "call", "name": name, "args": args}, active_turn())
            if not isinstance(reply, dict) or reply.get("ok") is not True or "value" not in reply:
                error = reply.get("error") if isinstance(reply, dict) else None
                return result(error or f"{language} tool call returned an invalid response.", error=True)
            value = reply["value"]
            if value is None:
                text = "(no result)"
            elif isinstance(value, str):
                text = value or "(empty result)"
            else:
                text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
            return result(text)
        except (OSError, ValueError, CancelledError) as error:
            return result(str(error) or f'Eval tool "{name}" was cancelled.', error=True)

    return Tool(descriptor["description"], parameters, execute)
