"""Project memory artifacts and the read-only memory://root namespace."""

import json
import os
import re
import unicodedata
from hashlib import sha256
from pathlib import Path, PureWindowsPath
from tempfile import NamedTemporaryFile
from urllib.parse import quote, unquote

from filelock import FileLock
from wcmatch import glob


SUMMARY = "memory_summary.md"
LEARNED = "learned.md"
MANIFEST = ".mini-pi-artifacts.json"
LINE_RANGE = re.compile(r"L?(\d+)(?:(?:-|\.\.)L?(\d*)|\+L?(\d+))?\Z")
GLOB_CHAR = re.compile(r"[*?\[]")


def neutral_line(text):
    text = "".join(" " if unicodedata.category(char) in ("Cc", "Cf") else char for char in text)
    return " ".join(re.sub(r"~{2,}", "~", text.translate(str.maketrans("", "", "<>`"))).split())


def path_parts(relative):
    parts = relative.replace("\\", "/").split("/")
    if any(not part or part in (".", "..") or ":" in part or "\0" in part
           or part.endswith((" ", ".")) or PureWindowsPath(part).is_reserved() for part in parts):
        raise ValueError("Memory path must stay within its project directory.")
    return parts


def atomic_text(target, text, *, durable=False):
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", dir=target.parent, encoding="utf-8", newline="\n", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            if durable:
                stream.flush()
                os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def read_utf8(path):
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def select_memory_lines(stream, suffix):
    parts = suffix.split(":")
    raw = "raw" in parts
    ranges = next((part for part in parts if part != "raw"), "")
    if len(parts) > 2 or len(parts) == 2 and not raw:
        raise ValueError("Invalid memory line selector.")
    intervals = []
    for interval in ranges.split(",") if ranges else ():
        match = LINE_RANGE.fullmatch(interval)
        if not match:
            raise ValueError("Use a line selector such as :2-5 or :raw:2+4.")
        start, end, count = match.groups()
        first = int(start)
        last = first + int(count) - 1 if count else int(end) if end else first if raw and end is None else None
        if first < 1 or last is not None and last < first:
            raise ValueError("Memory line numbers are 1-based and ranges must be increasing.")
        intervals.append((first, last))
    lines, size = [], 0
    for number, line in enumerate(stream, 1):
        if not intervals or any(first <= number and (last is None or number <= last) for first, last in intervals):
            lines.append(line if raw else f"{number}\t{line}")
            size += len(lines[-1])
        if size >= 128_000 or intervals and all(last is not None and number >= last for _, last in intervals):
            break
    return "".join(lines)[:128_000].rstrip("\n")


class MemoryFiles:
    def __init__(self, agent_dir, workspace, redact):
        self.workspace = workspace
        self.redact = redact
        base = agent_dir / "memories"
        key = sha256(workspace.encode("utf-8")).hexdigest()
        relative = workspace[1:] if workspace.startswith(("/", "\\")) else workspace
        name = "--" + re.sub(r"[/\\:]", "-", relative) + "--"
        if len(name.encode("utf-8")) > 180:
            name = "--" + key + "--"
        self.base, self.name, self.key = base, name, key

    def root(self, *, create=False):
        root = self.base / self.name
        marker = root / ".mini-pi-scope"
        if create:
            self.base.mkdir(parents=True, exist_ok=True)
            with FileLock(self.base / (self.name + ".scope.lock")):
                root.mkdir(exist_ok=True)
                if not marker.is_symlink() and not marker.exists():
                    atomic_text(marker, self.workspace)
        if marker.is_symlink() or marker.is_file() and read_utf8(marker) != self.workspace:
            root = self.base / (self.name.rstrip("-") + "-" + self.key[:16] + "--")
            if create:
                root.mkdir(exist_ok=True)
        return root.resolve()

    def lock(self, name):
        return FileLock(self.target(".mini-pi-" + name + ".lock", create=True, writing=True))

    def target(self, relative, *, create=False, writing=False):
        root = self.root(create=create)
        original = root.joinpath(*path_parts(relative))
        resolved = original.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError("Memory path resolves outside its project directory.")
        if writing and resolved != original:
            raise ValueError("Cannot write a memory artifact through a symlink.")
        return resolved

    def read_artifact(self, relative):
        return self.redact(read_utf8(self.target(relative))).strip()

    def lessons(self):
        return "\n".join(self.redact(neutral_line(line)) for line in self.read_artifact(LEARNED).splitlines()).strip()

    def save_lesson(self, memory, context=""):
        memory = self.redact(neutral_line(memory))[:2000].strip()
        context = self.redact(neutral_line(context))[:400].strip()
        if not memory:
            raise ValueError("Lesson was empty after sanitization; nothing stored.")
        lesson = f"- {memory}" + (f" _(context: {context})_" if context else "")
        with self.lock("lessons"):
            target = self.target(LEARNED, writing=True)
            existing = read_utf8(target).split("\n")
            if existing[-1] == "":
                existing.pop()
            lines = [line for line in existing if not (line.lstrip().startswith("- ") and line.strip() == lesson)]
            first = next((i for i, line in enumerate(lines) if line.lstrip().startswith("- ")), len(lines))
            lines.insert(first, lesson)
            count = 0
            kept = []
            for line in lines:
                if line.lstrip().startswith("- "):
                    count += 1
                    if count > 100:
                        continue
                kept.append(line)
            atomic_text(target, "\n".join(kept) + "\n")

    def publish(self, artifacts):
        manifest = self.target(MANIFEST, create=True, writing=True)
        if any(not self.generated(relative) for relative in artifacts):
            raise ValueError("Memory consolidation can publish only generated artifacts.")
        targets = {relative: self.target(relative, writing=True) for relative in artifacts}
        previous = self.published()
        for relative, content in artifacts.items():
            target = targets[relative]
            target.parent.mkdir(parents=True, exist_ok=True)
            if read_utf8(target) != content:
                atomic_text(target, content)
        for relative in previous.keys() - artifacts.keys():
            try:
                target = self.target(relative, writing=True)
            except ValueError:
                continue
            target.unlink(missing_ok=True)
            root = self.root()
            for parent in target.parents:
                if parent == root:
                    break
                try:
                    parent.rmdir()
                except OSError:
                    break
        atomic_text(manifest, json.dumps(dict.fromkeys(artifacts), ensure_ascii=False))

    def published(self):
        try:
            record = json.loads(read_utf8(self.target(MANIFEST)))
        except (ValueError, OSError):
            return {}
        if not isinstance(record, dict):
            return {}
        return {name: None for name in record if self.generated(name)}

    @staticmethod
    def generated(relative):
        return relative in (SUMMARY, "MEMORY.md", "raw_memories.md") or relative.startswith(("rollout_summaries/", "skills/"))

    def clear(self):
        self.publish({})
        for relative in (SUMMARY, "MEMORY.md", "raw_memories.md", LEARNED):
            self.target(relative, writing=True).unlink(missing_ok=True)

    def read(self, url):
        location = url.removeprefix("memory://")
        namespace, _, raw = location.partition("/")
        if not raw and ":" in namespace:
            namespace, _, raw = namespace.partition(":")
            raw = ":" + raw
        if namespace != "root":
            raise ValueError("Local memory URLs use memory://root.")
        raw, separator, suffix = raw.partition(":")
        relative = unquote(raw, errors="strict").rstrip("/") or SUMMARY
        path_parts(relative)
        if GLOB_CHAR.search(raw):
            return self.listing(raw)
        target = self.target(relative)
        if target.is_dir():
            if separator:
                raise ValueError("Memory line selectors require a file.")
            return self.listing(raw.rstrip("/") + "/*")
        if not target.is_file():
            raise ValueError(f"Memory file not found: {url}")
        if not separator:
            with target.open(encoding="utf-8") as stream:
                return self.redact(stream.read(128_001))
        return self.read_lines(target, suffix)

    def listing(self, raw):
        root = self.root()
        pattern = re.sub(r"(?:%[0-9A-Fa-f]{2})+", lambda match: glob.escape(unquote(match[0])), raw)
        entries = []
        for relative in glob.iglob(pattern, root_dir=root, flags=glob.GLOBSTAR):
            target = self.target(relative)
            name = relative.replace("\\", "/")
            if not any(part.startswith(".") for part in Path(name).parts):
                entries.append("memory://root/" + quote(name) + ("/" if target.is_dir() else ""))
            if len(entries) >= 1000:
                break
        return "\n".join(sorted(entries)) if entries else "No matching memory files."

    def read_lines(self, target, suffix):
        with target.open(encoding="utf-8") as stream:
            return self.redact(select_memory_lines(stream, suffix))
