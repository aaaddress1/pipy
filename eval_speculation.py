"""Project streamed eval cells and claim verified local reads at dispatch."""

import json
import math
import re
from concurrent.futures import CancelledError, ThreadPoolExecutor, wait
from threading import Lock

from handoff import Cancellation
from read_speculation import local_read_path


MAX_ARGUMENT_BYTES = 1024 * 1024
FIELDS = {"language", "reset", "timeout", "title", "code"}
SPACE = re.compile(r"[ \r\t\n]*")
PLAIN_STRING = re.compile(r'[^"\\\x00-\x1f\ud800-\udfff]+')
JSON = json.JSONDecoder()


def string_prefix(raw, start, previous=None):
    if raw[start] != '"':
        raise ValueError("Expected a JSON string.")
    value, offset, closed = previous or ("", start + 1, False)
    if closed:
        return previous
    parts = [value]
    while offset < len(raw):
        if run := PLAIN_STRING.match(raw, offset):
            parts.append(run[0])
            offset = run.end()
            continue
        char = raw[offset]
        if char == '"':
            return "".join(parts), offset + 1, True
        if char != "\\":
            raise ValueError("Invalid character in a JSON string.")
        if offset + 1 == len(raw):
            break
        size = 6 if raw[offset + 1] == "u" else 2
        if len(raw) - offset < size:
            break
        if size == 6 and 0xD800 <= int(raw[offset + 2:offset + 6], 16) <= 0xDBFF:
            if len(raw) - offset < 12:
                break
            if raw[offset + 6:offset + 8] != "\\u":
                raise ValueError("Invalid surrogate escape.")
            size = 12
        decoded = json.loads('"' + raw[offset:offset + size] + '"')
        decoded.encode("utf-8")
        parts.append(decoded)
        offset += size
    return "".join(parts), offset, False


class EvalArguments:
    """Decode unambiguous JSON string prefixes without inventing missing bytes."""

    def __init__(self):
        self.raw = ""
        self.code_scan = None
        self.args = {}
        self.complete = False

    def feed(self, raw):
        if not isinstance(raw, str) or not raw.startswith(self.raw):
            raise ValueError("Provider replaced streamed eval arguments.")
        if len(raw) > MAX_ARGUMENT_BYTES or (len(raw) * 4 > MAX_ARGUMENT_BYTES and len(raw.encode("utf-8")) > MAX_ARGUMENT_BYTES):
            raise ValueError("Eval argument stream exceeds the planning limit.")
        if raw == self.raw:
            return self.args, self.complete
        self.raw = raw
        self.args, self.complete = self.parse(raw)
        return self.args, self.complete

    def parse(self, raw):
        offset = SPACE.match(raw).end()
        args, seen = {}, set()
        if offset == len(raw):
            return args, False
        if raw[offset] != "{":
            raise ValueError("Expected an eval argument object.")
        offset += 1
        after_comma = False
        while True:
            offset = SPACE.match(raw, offset).end()
            if offset == len(raw):
                return args, False
            if raw[offset] == "}":
                if after_comma or "language" not in args or SPACE.match(raw, offset + 1).end() != len(raw):
                    raise ValueError("Malformed final eval arguments.")
                return args, True
            key, offset, complete = string_prefix(raw, offset)
            if not complete:
                return args, False
            if key in seen and key in FIELDS - {"title"}:
                raise ValueError("Duplicate eval argument.")
            seen.add(key)
            offset = SPACE.match(raw, offset).end()
            if offset == len(raw):
                return args, False
            if raw[offset] != ":":
                raise ValueError("Expected a colon after an eval argument.")
            offset = SPACE.match(raw, offset + 1).end()
            if offset == len(raw):
                return args, False
            if key == "code":
                previous = self.code_scan[1] if self.code_scan and self.code_scan[0] == offset else None
                scanned = string_prefix(raw, offset, previous)
                self.code_scan = offset, scanned
                args[key], offset, complete = scanned
                if not complete:
                    return args, False
            else:
                if raw[offset] == '"':
                    value, end, complete = string_prefix(raw, offset)
                    if not complete:
                        return args, False
                else:
                    try:
                        value, end = JSON.raw_decode(raw, offset)
                    except json.JSONDecodeError:
                        return args, False
                if key == "language" and value not in ("py", "js", "rb", "jl"):
                    raise ValueError("Unknown eval language.")
                if key == "reset" and type(value) is not bool:
                    raise ValueError("Invalid eval reset.")
                if key == "title" and not isinstance(value, str):
                    raise ValueError("Invalid eval title.")
                if key == "timeout" and (
                    type(value) not in (int, float) or value < 0 or (type(value) is float and not math.isfinite(value))
                ):
                    raise ValueError("Invalid eval timeout.")
                if end < len(raw) and raw[end] in ".eE+-" and type(value) in (int, float):
                    if all(char in "0123456789.eE+-" for char in raw[offset:]):
                        return args, False
                    raise ValueError("Invalid eval number.")
                args[key], offset = value, end
            offset = SPACE.match(raw, offset).end()
            if offset == len(raw):
                return args, False
            if raw[offset] == ",":
                after_comma = True
                offset += 1
            elif raw[offset] == "}":
                after_comma = False
            else:
                raise ValueError("Expected an eval argument delimiter.")


def evaluate(expression, snapshot, results):
    kind = expression["kind"]
    if kind == "literal":
        return expression["value"]
    if kind == "snapshot":
        return snapshot[expression["name"]]
    if kind == "operation_result":
        return results[expression["operationId"]]
    if kind == "property":
        value = evaluate(expression["target"], snapshot, results)
        key = expression["property"]
        if type(value) is list:
            if type(key) is int and 0 <= key < len(value):
                return value[key]
            if key == "length":
                return len(value)
        if type(value) is dict:
            return value[str(key)]
    elif kind == "array":
        return [evaluate(item, snapshot, results) for item in expression["items"]]
    elif kind == "object":
        return {item["key"]: evaluate(item["value"], snapshot, results) for item in expression["entries"]}
    elif kind == "concat":
        return "".join(evaluate(item, snapshot, results) for item in expression["items"])
    elif kind == "transform" and expression["name"] == "Python.str":
        value = evaluate(expression["input"], snapshot, results)
        if type(value) is str:
            return value
    raise ValueError("Unsupported Python shadow expression.")


class ShadowCell:
    def __init__(self, identifier, tool, reads, executor, parent):
        self.identifier, self.tool, self.reads, self.executor = identifier, tool, reads, executor
        self.signal = Cancellation(parent, reason="Eval shadow discarded.")
        self.decoder = EvalArguments()
        self.lock = Lock()
        self.pending = self.future = self.kernel = self.token = self.final_args = None
        self.last_code = self.language = None
        self.planning = self.closed = False
        self.operations = []
        self.candidates = {}
        self.attempted = set()
        self.claims = {}
        self.results = {}

    def update(self, raw):
        if self.closed or raw is None:
            return
        try:
            args, complete = self.decoder.feed(raw)
        except (ValueError, TypeError, RecursionError):
            self.invalidate()
            return
        language = args.get("language")
        if args.get("reset") is True or language not in (None, "py", "js"):
            self.invalidate()
            return
        if language and (complete or args.get("reset") is False) and (code := args.get("code")):
            with self.lock:
                if self.closed or code == self.last_code:
                    return
                self.language = language
                self.pending = code
                if not self.planning:
                    self.planning = True
                    self.future = self.executor.submit(self.drain)

    def drain(self):
        while True:
            with self.lock:
                if self.closed or self.pending is None:
                    self.planning = False
                    return
                code, self.pending = self.pending, None
            self.project(code)

    def project(self, code):
        try:
            plan = self.tool.shadow_plan(code, self.signal, self.language)
            with self.lock:
                if self.closed:
                    return
                self.last_code = code
                if plan is None:
                    if self.candidates:
                        self.invalidate()
                    return
                kernel, frame = plan
                token = frame["revision"], frame["digest"]
                if self.token is not None and (self.kernel is not kernel or self.token != token):
                    self.invalidate()
                    return
                self.kernel, self.token = kernel, token
                self.snapshot = frame["values"]
                barrier = min((control["span"]["start"] for control in frame.get("controls", ())
                               if control["kind"] == "conditional"), default=math.inf)
                operations = [entry["call"] for entry in frame["operations"] if not entry["call"]["controlDependencies"]
                              and entry["call"]["span"]["start"] < barrier]
                if not self.candidates.keys() <= {call["id"] for call in operations}:
                    self.invalidate()
                    return
                self.operations = operations
                self.admit(dict(frame["args"]) if self.language == "js" else None)
        except (OSError, ValueError, RuntimeError, KeyError, TypeError, RecursionError, CancelledError):
            self.invalidate()

    def admit(self, resolved=None):
        occurrences = {}
        ready = []
        for operation in self.operations:
            site, name, identifier = (operation[key] for key in ("siteId", "name", "id"))
            key = site, name
            occurrence = occurrences.get(key, 0)
            occurrences[key] = occurrence + 1
            if identifier in self.attempted or name != "read" or any(dep not in self.results for dep in operation["dependencies"]):
                continue
            self.attempted.add(identifier)
            ready.append((operation, (site, name, occurrence)))
        if self.language == "js" and resolved is None and ready:
            try:
                resolved = self.tool.js.shadow_arguments(
                    self.kernel, self.snapshot, self.results, [operation for operation, _ in ready], self.signal,
                )
            except (KeyError, ValueError, TypeError, OSError, CancelledError):
                return
        for operation, key in ready:
            identifier = operation["id"]
            try:
                args = resolved.get(identifier) if resolved is not None else evaluate(operation["args"], self.snapshot, self.results)
                if path := local_read_path(args):
                    if candidate := self.reads.submit(f"{self.identifier}:{identifier}", path, self.signal):
                        self.candidates[identifier] = candidate
                        self.claims[key] = identifier
            except (KeyError, ValueError, TypeError, RecursionError, OSError, CancelledError):
                continue

    def finalize(self, raw):
        if self.closed:
            return False
        try:
            args, complete = EvalArguments().feed(raw)
            self.tool.validate(args)
            streamed = self.decoder.args
            if not complete or args.get("reset") or args["language"] not in ("py", "js"):
                return False
            if not args["code"].startswith(streamed.get("code", "")):
                return False
            if any(key in streamed and streamed[key] != args.get(key) for key in ("language", "reset")):
                return False
        except (ValueError, TypeError, RecursionError):
            return False
        if self.future:
            self.future.result()
        self.language = args["language"]
        if not self.closed and self.last_code != args["code"]:
            self.project(args["code"])
        if not self.closed and self.candidates:
            self.final_args = args
            return True
        return False

    def execution_token(self, kernel, args):
        if not self.closed and self.kernel is kernel and self.final_args == args:
            return self.token
        return None

    def claim(self, name, args, identity, cancel, clock, render):
        path = local_read_path(args)
        site, occurrence = identity.get("siteId"), identity.get("occurrence")
        if path is None or name != "read" or not isinstance(site, str) or type(occurrence) is not int or occurrence < 0:
            return None
        key = site, name, occurrence
        with self.lock:
            identifier = self.claims.get(key) if not self.closed else None
            candidate = self.candidates.get(identifier)
            if candidate is None or candidate.path != path:
                return None
            del self.claims[key]
        try:
            text = self.reads.verify(candidate, self.signal, cancel=cancel, expired=clock.expired)
        except Exception:
            cancel.check()
            return None
        if text is None:
            return None
        value = render(text)
        with self.lock:
            if self.closed:
                return None
            self.results[identifier] = value
            try:
                self.admit()
            except Exception:
                pass
        return value

    def invalidate(self):
        self.closed = True
        self.signal.cancel()

    def close(self):
        self.invalidate()
        if self.future:
            self.future.result()
        with self.lock:
            futures = [candidate.future for candidate in self.candidates.values()]
        for future in futures:
            future.cancel()
        wait(futures)


class EvalSpeculation:
    def __init__(self, tool, reads):
        self.tool, self.reads = tool, reads
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mini-pi-eval-plan")
        self.parent = None
        self.cells, self.claims = {}, {}
        self.lock = Lock()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.discard()
        self.executor.shutdown(wait=True, cancel_futures=True)

    def begin(self, parent):
        self.discard()
        self.parent = parent

    def restart(self):
        self.discard()

    def observe(self, index, name, identifier, raw, *, final=False):
        if name != "eval" or not identifier:
            return
        cell = self.cells.get(index)
        if cell is None:
            cell = self.cells[index] = ShadowCell(identifier, self.tool, self.reads, self.executor, self.parent)
        if cell.identifier == identifier:
            cell.update(raw)
        else:
            cell.invalidate()

    def seal(self, calls):
        authoritative = {call["id"]: call["function"]["arguments"] for call in calls if call["function"]["name"] == "eval"}
        for cell in self.cells.values():
            if cell.identifier in authoritative and cell.finalize(authoritative[cell.identifier]):
                self.claims[cell.identifier] = cell
            else:
                cell.close()

    def take(self, call):
        with self.lock:
            return self.claims.pop(call["id"], None)

    def cancel(self):
        for cell in self.cells.values():
            cell.invalidate()

    def discard(self):
        self.cancel()
        for cell in self.cells.values():
            cell.close()
        self.cells.clear()
        self.claims.clear()
