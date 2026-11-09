"""Collect and validate the result of a structured child turn."""

import json
import re
from collections import Counter
from functools import cached_property

from jsonschema import Draft202012Validator
from referencing import Registry
from referencing.exceptions import Unresolvable
from referencing.jsonschema import DRAFT202012

from async_jobs import tool_result
from session_jsonl import plain_text


RETRIES = 3
OVERRIDDEN = "Subagent exhausted schema-retry budget; result was accepted despite failing the output schema."
PLACEHOLDERS = {"ignore", "ignored", "n/a", "none", "not applicable", "null", "placeholder", "undefined", "unused"}
UNIONS = ("oneOf", "anyOf")
BRANCHES = ("allOf", *UNIONS)
SCHEMA_MAPS = ("properties", "patternProperties", "dependentSchemas", "dependencies")
SCHEMA_CHILDREN = (*BRANCHES, "prefixItems", "items", "additionalItems", "additionalProperties", "contains",
                   "not", "if", "then", "else", "propertyNames", "unevaluatedItems", "unevaluatedProperties")


def yield_error(data, value):
    if value is not None and not isinstance(value, str):
        raise ValueError("error must be a string")
    value = value.strip() if value else ""
    return value if value and (data is None or value.lower() not in PLACEHOLDERS) else None


def variants(schema, keys):
    for key in keys:
        for branch in schema.get(key, ()):
            if isinstance(branch, dict):
                yield branch


def inline_refs(schema, resolver, seen=frozenset()):
    if isinstance(schema, list):
        return [inline_refs(value, resolver, seen) for value in schema]
    if not isinstance(schema, dict):
        return schema
    node = schema
    if isinstance(ref := node.get("$ref"), str) and ref not in seen:
        try:
            target = resolver.lookup(ref)
        except Unresolvable:
            pass
        else:
            resolved = inline_refs(target.contents, target.resolver, seen | {ref})
            siblings = {key: value for key, value in node.items() if key not in ("$ref", "$defs", "definitions")}
            if isinstance(resolved, dict):
                return {**resolved, **inline_refs(siblings, resolver, seen)}
            elif not siblings:
                return resolved
    result = {}
    for key, value in node.items():
        if key in ("$defs", "definitions"):
            continue
        if key in SCHEMA_MAPS and isinstance(value, dict):
            value = {name: inline_refs(child, resolver, seen) for name, child in value.items()}
        elif key in SCHEMA_CHILDREN:
            value = inline_refs(value, resolver, seen)
        result[key] = value
    return result


def has_ref(schema):
    if isinstance(schema, list):
        return any(map(has_ref, schema))
    if not isinstance(schema, dict):
        return False
    if "$ref" in schema or "$dynamicRef" in schema:
        return True
    return (any(has_ref(schema.get(key)) for key in SCHEMA_CHILDREN)
            or any(has_ref(child) for key in SCHEMA_MAPS for child in schema.get(key, {}).values()))


def array_schema(schema):
    if not isinstance(schema, dict):
        return False
    kind = schema.get("type")
    return kind == "array" or isinstance(kind, list) and "array" in kind or any(
        array_schema(branch) for branch in variants(schema, BRANCHES)
    )


def shapes(schema, found):
    for name, section in schema.get("properties", {}).items():
        kind = "array" if array_schema(section) else "scalar"
        found[name] = kind if name not in found or found[name] == kind else "mixed"
    for branch in variants(schema, BRANCHES):
        shapes(branch, found)


def closed_conjuncts(schema):
    if schema.get("additionalProperties") is False:
        yield schema
    for branch in variants(schema, ("allOf",)):
        yield from closed_conjuncts(branch)


def closed_unions(schema):
    for key in UNIONS:
        if branches := schema.get(key):
            groups = [tuple(closed_conjuncts(branch)) if isinstance(branch, dict) else () for branch in branches]
            if all(groups):
                yield groups
    for branch in variants(schema, ("allOf",)):
        yield from closed_unions(branch)


def accepts_label(schema, name):
    return name in schema.get("properties", {}) or any(re.search(pattern, name) for pattern in schema.get("patternProperties", {}))


def strict_output(value, schema, validator):
    if not isinstance(schema, dict):
        return value
    for branch in schema.get("allOf", ()):
        value = strict_output(value, branch, validator)
    for key in UNIONS:
        if key in schema and not validator.evolve(schema=schema).is_valid(value):
            for branch in schema[key]:
                candidate = strict_output(value, branch, validator)
                if validator.evolve(schema=schema).is_valid(candidate):
                    return candidate
    if isinstance(value, list):
        prefix = schema.get("prefixItems", ())
        return [strict_output(item, prefix[index] if index < len(prefix) else schema.get("items"), validator)
                for index, item in enumerate(value)]
    if isinstance(value, dict) and (properties := schema.get("properties")):
        result = value.copy()
        for name, child in properties.items():
            if name not in result:
                continue
            if result[name] is None and name not in schema.get("required", ()) and not validator.evolve(schema=child).is_valid(None):
                del result[name]
            else:
                result[name] = strict_output(result[name], child, validator)
        return result
    return value


class OutputSchema:
    def __init__(self, schema, mode):
        self.validator = Draft202012Validator(schema) if schema not in (None, True) else None
        self.strict = mode == "strict"
        resolver = Registry().resolver_with_root(DRAFT202012.create_resource(schema)) if isinstance(schema, dict) else None
        self.resolved = inline_refs(schema, resolver) if resolver else schema
        self.shapes, self.sections = {}, {}
        self.conjuncts, self.unions, self.labels = (), (), ()
        if isinstance(self.resolved, dict):
            shapes(self.resolved, self.shapes)
            self.sections = {name: prop.get("items", prop) if isinstance(prop, dict) and prop.get("type") == "array" else prop
                             for name, prop in self.resolved.get("properties", {}).items()}
            self.conjuncts = tuple(closed_conjuncts(self.resolved))
            self.unions = tuple(closed_unions(self.resolved))
            closed = [*self.conjuncts, *(schema for groups in self.unions for group in groups for schema in group)]
            self.labels = tuple(dict.fromkeys(name for schema in closed for name in schema.get("properties", {})))

    def known(self, name):
        return all(accepts_label(schema, name) for schema in self.conjuncts) and all(
            any(all(accepts_label(schema, name) for schema in group) for group in groups) for groups in self.unions
        )

    def normalize(self, value, labels=None):
        if not self.strict or not self.validator:
            return value
        node = self.resolved if labels is None else self.sections.get(labels[0]) if len(labels) == 1 else None
        return strict_output(value, node, self.validator)

    def errors(self, value, labels=None):
        if not self.validator:
            return []
        if labels is None:
            return list(self.validator.iter_errors(value))
        for name in labels:
            if name in self.sections:
                errors = list(self.validator.evolve(schema=self.sections[name]).iter_errors(value))
                if errors:
                    return errors
        return []

    @cached_property
    def parameters(self):
        data = self.resolved if isinstance(self.resolved, dict) and not has_ref(self.resolved) else {}
        if data.get("type") == "object":
            options = [data]
            for prop in data.get("properties", {}).values():
                candidates = (prop, prop.get("items")) if array_schema(prop) and isinstance(prop, dict) else (prop,)
                options.extend(node for node in candidates if isinstance(node, dict) and node not in options)
            if len(options) > 1:
                data = {"anyOf": options}
        return {"type": "object", "additionalProperties": False, "properties": {
            "data": data, "error": {"type": "string"},
            "type": {"anyOf": [{"type": "string"}, {"type": "array", "minItems": 1, "items": {"type": "string"}}],
                     "description": "A string ends the turn. An array of labels submits a section."},
        }, "required": []}


def quoted(labels):
    return ", ".join(json.dumps(label, ensure_ascii=False) for label in labels)


def validation_text(errors):
    return "; ".join(f"{error.json_path}: {error.message}" for error in errors)


def decode_json(value):
    try:
        return json.loads(value)
    except ValueError:
        return value


class YieldState:
    def __init__(self, schema, mode):
        self.schema = OutputSchema(schema, mode)
        self.sections, self.counts, self.overrides = {}, Counter(), set()
        self.invalid = self.empty = 0
        self.has_sections = False
        self.terminal = self.stale = self.assistant = self.previous = None

    def note_assistant(self, assistant, previous):
        self.assistant, self.previous = assistant, previous

    def report_text(self, incremental):
        current = self.assistant or {}
        text = plain_text(current.get("content"))
        if incremental:
            return text if text.strip() else None
        calls = current.get("tool_calls", ())
        if len(calls) != 1 or calls[0]["function"]["name"] != "yield":
            return None
        if text.strip():
            return text
        previous = self.previous or {}
        if previous.get("role") == "assistant" and not previous.get("tool_calls"):
            text = plain_text(previous.get("content"))
        return text if text.strip() else None

    def empty_submission(self, reason, kind):
        self.empty += 1
        if self.empty <= RETRIES:
            raise ValueError(f"{reason} Put your result in data. {RETRIES - self.empty} empty-result retries remain.")
        error = f"{reason} Stopping after {self.empty} consecutive empty attempts."
        self.empty = 0
        self.terminal = {"status": "aborted", "error": error, "type": kind}
        return tool_result(f"Task aborted: {error}", self.terminal)

    def submit(self, args, *, force=False):
        if not isinstance(args, dict):
            raise ValueError("yield arguments must be an object.")
        unknown = [name for name, value in args.items() if value is not None and name not in ("data", "error", "type")]
        if unknown:
            raise ValueError(f"Unsupported yield arguments {quoted(unknown)}. Put the entire result inside data.")
        kind = args.get("type")
        incremental = isinstance(kind, list) and kind and all(isinstance(name, str) for name in kind)
        if kind is not None and not isinstance(kind, str) and not incremental:
            raise ValueError("type must be a string or non-empty array of strings")
        data = args.get("data")
        error = yield_error(data, args.get("error"))
        if data is not None and error:
            raise ValueError("yield received both data and error. Resubmit exactly one of them.")
        if error:
            self.empty = 0
            self.terminal = {"status": "aborted", "error": error, "type": kind}
            return tool_result(f"Task aborted: {error}", self.terminal)
        labels = kind if incremental else None
        if labels:
            rejected = [name for name in labels if not self.schema.known(name)]
            if rejected:
                allowed = quoted(self.schema.labels) or "none"
                raise ValueError(f"Section {quoted(labels)} uses unknown incremental yield label(s): {quoted(rejected)}. "
                                 f"Resubmit with one of the schema's labels: {allowed}.")
        last_turn = data is None
        if last_turn:
            if kind is None:
                return self.empty_submission("yield must contain either data or error.", kind)
            if not incremental and self.schema.validator and not self.has_sections:
                raise ValueError("This task requires structured output matching the declared schema. Submit the full object in data.")
            if incremental or not self.has_sections:
                data = self.report_text(incremental)
                if data is None:
                    return self.empty_submission("yield resolved to an empty last-turn result.", kind)
        overridden = False
        response = "Result submitted."
        if not last_turn:
            data = self.schema.normalize(data, labels)
            errors = self.schema.errors(data, labels)
            if errors and isinstance(data, str):
                decoded = self.schema.normalize(decode_json(data), labels)
                if decoded is not None and not self.schema.errors(decoded, labels):
                    data, errors = decoded, []
            if errors:
                self.invalid += 1
                if self.invalid <= RETRIES:
                    scope = f"Section {quoted(labels)}" if labels else "Output"
                    raise ValueError(f"{scope} does not match schema: {validation_text(errors)}. "
                                     f"Call yield again with corrected data ({RETRIES - self.invalid} retries before override).")
                response = f"Result submitted (schema validation overridden after {self.invalid} failed attempts)."
                overridden = True
            self.invalid = 0
        self.empty = 0
        details = {"status": "success", "data": data, **({"type": kind} if kind is not None else {})}
        if last_turn:
            details["useLastTurn"] = True
        if overridden:
            details["schemaOverridden"] = True
        if incremental:
            self.has_sections = True
            self.append_sections(labels, data, overridden)
            if force:
                self.terminal = {"status": "success"}
        else:
            self.terminal = details
        return tool_result(response, details)

    def append_sections(self, labels, data, overridden):
        for name in map(str.strip, labels):
            if not name:
                continue
            shape = self.schema.shapes.get(name)
            previous = self.sections.get(name)
            if shape == "scalar":
                self.sections[name] = data
                self.overrides.discard(name)
            elif self.counts[name]:
                self.sections[name] = [*previous, data] if isinstance(previous, list) else [previous, data]
            else:
                self.sections[name] = [data] if shape == "array" else data
            self.counts[name] += 1
            if overridden:
                self.overrides.add(name)

    def finalize(self):
        terminal = self.terminal or {}
        if terminal.get("status") == "aborted":
            return None, terminal["error"], None
        data = terminal.get("data")
        raw_text = terminal.get("useLastTurn")
        overridden = terminal.get("schemaOverridden")
        if data is None:
            data = self.sections.copy() if self.sections else None
            overridden = bool(self.overrides)
        if data is None:
            return None, "Subagent called yield with no result data.", None
        if isinstance(data, str) and not raw_text and data.lstrip().startswith(("{", "[")):
            data = decode_json(data)
        data = self.schema.normalize(data)
        errors = self.schema.errors(data)
        if errors and (self.schema.strict or not overridden):
            return data, f"schema_violation: {validation_text(errors)}", None
        if self.schema.strict and overridden:
            return data, f"schema_violation: {OVERRIDDEN}", None
        return data, None, OVERRIDDEN if overridden else None

    def invalidate(self, submission):
        self.stale, self.terminal = submission, None
