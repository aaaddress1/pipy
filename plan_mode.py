"""ACP read-only planning, local drafts, and approved plan references."""

import json
import os
import re
from fnmatch import fnmatchcase
from io import StringIO
from pathlib import Path, PureWindowsPath
from tempfile import NamedTemporaryFile

import regex

from async_jobs import tool_result
from cancellable_http import active_turn
from runtime import database
from session_graph import ANCESTORS
from tool_pruning import ReadText, selected_lines, split_read


ENTRY = "acp-plan-mode"
DEFAULT_DRAFT = "local://PLAN.md"
APPROVE, REFINE = "Approve and execute", "Refine plan"
LOCAL = "local://"
MAX_DRAFT = 128 * 1024
OUTPUT_LIMIT = 16_000
MAX_SEARCH_FILES = 12_000
MAX_SEARCH_HITS = 200
SKIP_DIRS = {".git", ".venv", "__pycache__", "node_modules"}
DRAFT_NAME = re.compile(r"[A-Za-z0-9_.-]{1,180}", re.ASCII)


def normalize_title(title):
    title = title.strip()
    if not title or "/" in title or "\\" in title or ".." in title:
        return None
    title = re.sub(r"\.md$", "", title, flags=re.I)
    title = re.sub(r"[^A-Za-z0-9_-]", "", re.sub(r"\s+", "-", title))
    return re.sub(r"-{2,}", "-", title).strip("-")[:120] or None


class PlanMode:
    def __init__(self, record, *, autosave=False, autosave_dir=None):
        self.record = record
        self.session = record.handle.session
        self.workspace = record.cwd
        self.root = self.session.path.parent.resolve() / "local" / self.session.id
        self.autosave, self.autosave_dir = autosave, autosave_dir
        with database(self.session.path) as db:
            row = db.execute(ANCESTORS + """
                SELECT json_extract(n.payload,'$.data') FROM ancestry a JOIN session_nodes n ON n.id=a.id
                WHERE n.kind='foreign' AND json_extract(n.payload,'$.customType')=? ORDER BY n.id DESC LIMIT 1
            """, (self.session.leaf, self.session.id, self.session.id, ENTRY)).fetchone()
        self.state = json.loads(row[0]) if row else {"enabled": False, "path": DEFAULT_DRAFT}

    @property
    def enabled(self):
        return self.state["enabled"]

    def save_state(self, **changes):
        state = {**self.state, **changes}
        if state != self.state:
            self.session.record_metadata({"type": "custom", "customType": ENTRY, "data": state})
            self.state = state

    def set_mode(self, enabled):
        if enabled != self.enabled:
            self.save_state(enabled=enabled, path=DEFAULT_DRAFT)

    def instructions(self, window):
        if self.enabled:
            return (
                "<critical>Plan mode is active. The working tree and system are read-only. "
                "Investigate with read, glob and grep. Draft a decision-complete implementation plan with "
                f"write or edit in local://<slug>-plan.md. Current draft reference: {self.state['path']}. "
                "Include relevant files, implementation steps and verification. Submit the slug as plain text "
                "by writing to xd://propose. Use that tool for approval. Revise the draft if refinement is requested.</critical>"
            )
        if approved := self.state.get("approved"):
            reference = f"## Existing Plan\nApproved plan at {approved['path']}. Use it when relevant to the current work."
            if len(approved["content"]) > max(4096, (window or 32768) // 2):
                return reference + " Read the approved draft from that URL."
            return f"{reference}\n<plan>\n{approved['content']}\n</plan>"
        return ""

    def draft_path(self, url):
        name = url.removeprefix(LOCAL)
        if (not url.startswith(LOCAL) or not DRAFT_NAME.fullmatch(name) or name in (".", "..")
            or name.endswith(".") or PureWindowsPath(name).is_reserved()):
            raise ValueError("Use a session-local file name such as local://feature-plan.md.")
        target = self.root / name
        if self.root.resolve() != self.root or target.resolve() != target:
            raise ValueError("Planning drafts cannot follow filesystem links.")
        return target

    def read_draft(self, url):
        with self.draft_path(url).open("rb") as file:
            payload = file.read(MAX_DRAFT + 1)
        if len(payload) > MAX_DRAFT:
            raise ValueError("Planning draft exceeds the session limit.")
        return payload.decode("utf-8")

    def write_draft(self, url, content):
        target = self.draft_path(url)
        payload = content.encode("utf-8")
        if len(payload) > MAX_DRAFT:
            raise ValueError("Planning draft exceeds the session limit.")
        target.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=target.parent, delete=False) as file:
            staged = Path(file.name)
            try:
                file.write(payload)
                file.close()
                staged.replace(self.draft_path(url))
            finally:
                staged.unlink(missing_ok=True)

    def find_draft(self, title):
        stem = normalize_title(title)
        preferred = f"{LOCAL}{re.sub(r'-plan$', '', stem, flags=re.I) or stem}-plan.md" if stem else None
        try:
            files = sorted(
                (file for file in self.root.iterdir()
                 if file.is_file() and not file.is_symlink() and file.name.lower().endswith("plan.md")
                 and DRAFT_NAME.fullmatch(file.name)),
                key=lambda file: file.stat().st_mtime_ns, reverse=True,
            ) if self.root.resolve() == self.root else []
        except FileNotFoundError:
            files = []
        listed = [LOCAL + file.name for file in files]
        current = self.state["path"]
        for url in dict.fromkeys([preferred, current if current not in listed else None, *listed, current]):
            if url is None:
                continue
            try:
                content = self.read_draft(url)
            except FileNotFoundError:
                continue
            if content.strip():
                heading = re.search(r"^[ \t]*#[ \t]+(.+)$", content, re.M)
                url = LOCAL + self.draft_path(url).resolve().name
                return url, content, (stem or normalize_title(heading[1] if heading else "")
                                      or normalize_title(self.draft_path(url).stem) or "plan")
        raise ValueError(f"Plan file not found at {preferred or current}. Write the finalized plan there first.")

    def propose(self, title):
        if not self.enabled:
            raise ValueError("Plan mode is not active.")
        turn = self.record.turn
        if turn is None:
            raise ValueError("A plan proposal requires an active ACP turn.")
        url, content, title = self.find_draft(title)
        lines = content.splitlines()
        preview = "\n".join(lines[:12])[:4000]
        if len(lines) > 12 or len(preview) == 4000:
            preview += "\n…"
        approved = not turn.supports_forms or turn.elicit_value(
            f'Approve plan "{title}" and start implementation?\n\n{preview}',
            {"type": "string", "enum": [APPROVE, REFINE]},
        ) == APPROVE
        with self.record.lock:
            signal = active_turn() or turn
            signal.check()
            if self.read_draft(url) != content:
                raise ValueError("The draft changed during review. Submit the updated plan again.")
            details = {"planFilePath": url, "title": title, "planExists": True}
            if not approved:
                self.save_state(path=url)
                return tool_result(f"Plan refinement requested. Update {url}, then write {title} to xd://propose again.", details)
            self.save_state(enabled=False, path=url, approved={"path": url, "content": content})
            self.record.mode_changed()
        result = f"Plan approved at {url}. Plan mode exited; proceed with the implementation."
        if self.autosave:
            try:
                self.autosave_plan(title, content)
            except OSError:
                result += " Plan autosave failed."
        return tool_result(result, details)

    def autosave_plan(self, title, content):
        directory = (self.workspace / Path(self.autosave_dir.strip()).expanduser()
                     if isinstance(self.autosave_dir, str) and self.autosave_dir.strip()
                     else self.workspace / ".omp" / "plans")
        directory.mkdir(parents=True, exist_ok=True)
        stem = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_").upper()
        if len(stem) > 32:
            stem = stem[:32].rsplit("_", 1)[0] if "_" in stem[:32] else stem[:32]
        stem = stem if stem == "PLAN" or stem.endswith("_PLAN") else (stem + "_PLAN" if stem else "PLAN")
        for index in range(1000):
            target = directory / (f"{stem}-{index}.md" if index else f"{stem}.md")
            try:
                with target.open("x", encoding="utf-8", newline="") as file:
                    file.write(content)
                return
            except FileExistsError:
                continue
        raise OSError("Too many plan autosave name collisions.")

    def tools(self, available, builtin, *, force_planning=False):
        planning = force_planning or self.enabled
        if not planning and not self.root.exists() and not self.state.get("approved"):
            return available
        description, fields, read = builtin["read"]

        def read_plan(path):
            if not path.startswith(LOCAL):
                return read(path)
            url, selector = split_read(path)
            approved = self.state.get("approved")
            content = approved["content"] if not planning and approved and url == approved["path"] else self.read_draft(url)
            if selector:
                signal = active_turn()
                content = selected_lines(StringIO(content), selector, OUTPUT_LIMIT, signal.check if signal else None)
            if len(content) > OUTPUT_LIMIT:
                return ReadText(content[:OUTPUT_LIMIT] + "\n[More text available through a line selector]", complete=False)
            return ReadText(content, complete=True if not selector else None)

        read_plan.speculative_local = getattr(read, "speculative_local", False)
        reader = description + " Use local:// for planning drafts.", fields, read_plan
        if not planning:
            return {**available, "read": reader}

        def write(path, content=None, text=None):
            if (content is None) == (text is None):
                raise ValueError("Provide only one of content or text.")
            payload = content if content is not None else text
            if path == "xd://propose":
                return self.propose(payload)
            self.write_draft(path, payload)
            return f"Wrote {path}"

        def edit(path, old, new):
            text = self.read_draft(path)
            if not old or text.count(old) != 1:
                raise ValueError("The old text must appear exactly once.")
            self.write_draft(path, text.replace(old, new, 1))
            return f"Edited {path}"

        return {
            "read": reader,
            "glob": ("List workspace files matching a glob.", {"pattern": "Glob pattern"}, self.glob),
            "grep": ("Search UTF-8 workspace files without changing them.",
                     {"pattern": "Regular expression", "path": ("Workspace file or directory", "string")}, self.grep),
            "write": ("Write a local:// draft or submit its slug to xd://propose.", builtin["write"][1], write),
            "edit": ("Edit a local:// draft.", builtin["edit"][1], edit),
        }

    def files(self, path):
        root = (self.workspace / path).resolve()
        if not root.is_relative_to(self.workspace):
            raise ValueError("Search path must be inside the workspace.")
        if root.is_file():
            yield root
            return
        if not root.is_dir():
            raise ValueError("Search path does not exist.")
        signal = active_turn()
        inspected = 0
        for directory, dirs, names in os.walk(root, followlinks=False):
            current = Path(directory)
            if not current.resolve().is_relative_to(self.workspace):
                dirs.clear()
                continue
            dirs[:] = sorted(name for name in dirs if name not in SKIP_DIRS and not (current / name).is_symlink())
            for name in sorted(names):
                if signal:
                    signal.check()
                inspected += 1
                if inspected > MAX_SEARCH_FILES:
                    raise ValueError("Search scan limit reached. Narrow the path.")
                file = current / name
                if file.resolve().is_relative_to(self.workspace) and file.is_file():
                    yield file

    def glob(self, pattern):
        matches = []
        for file in self.files("."):
            path = file.relative_to(self.workspace).as_posix()
            if fnmatchcase(path, pattern) or pattern.startswith("**/") and fnmatchcase(path, pattern[3:]):
                matches.append(path)
                if len(matches) >= MAX_SEARCH_HITS:
                    matches.append("[More results omitted]")
                    break
        return ("\n".join(matches) or "No matching files.")[:OUTPUT_LIMIT]

    def grep(self, pattern, path="."):
        pattern = regex.compile(pattern)
        matches = []
        for file in self.files(path):
            try:
                with file.open("rb") as stream:
                    data = stream.read(MAX_DRAFT)
                if b"\0" in data:
                    continue
                for number, line in enumerate(data.decode("utf-8").splitlines(), 1):
                    if pattern.search(line, timeout=.05):
                        matches.append(f"{file.relative_to(self.workspace).as_posix()}:{number}: {line[:600]}")
                        if len(matches) >= MAX_SEARCH_HITS:
                            return "\n".join(matches)[:OUTPUT_LIMIT] + "\n[More results omitted]"
            except (OSError, UnicodeError):
                continue
        return ("\n".join(matches) or "No matches.")[:OUTPUT_LIMIT]
