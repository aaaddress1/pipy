"""Terminal view for live side answers and the session's BTW topics."""

import os
import shutil
import subprocess
import sys
from textwrap import wrap

from prompt_toolkit.application.current import get_app
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import to_formatted_text
from prompt_toolkit.key_binding import KeyBindings

from btw import ANSI, copy_text, latest, split_inject, turns


def clipboard(text):
    if os.name == "nt":
        command = ["clip.exe"]
        data = ("\ufeff" + text).encode("utf-16-le")
        options = {"creationflags": subprocess.CREATE_NO_WINDOW}
    else:
        command = (["pbcopy"] if sys.platform == "darwin" else ["wl-copy"] if os.getenv("WAYLAND_DISPLAY")
                   and shutil.which("wl-copy") else ["xclip", "-selection", "clipboard"])
        data, options = text.encode("utf-8"), {}
    subprocess.run(command, input=data, check=True, timeout=5, **options)


class BtwPanel:
    def __init__(self, editor, btw, leaf):
        self.editor, self.btw, self.leaf = editor, btw, leaf
        self.view, self.identifier, self.index, self.first = None, None, 0, None
        self.notice = ""
        self.records = {}
        self.keys = KeyBindings()
        visible = Condition(lambda: self.view is not None)
        empty = Condition(lambda: not self.editor.default_buffer.text.strip())
        browsing = Condition(lambda: self.view == "list" and bool(self.records)) & empty
        detail = Condition(lambda: self.view == "detail" and self.record() is not None)
        complete = Condition(lambda: bool((record := self.record()) and latest(record)["status"] != "running"))

        @self.keys.add("escape", filter=visible)
        def escape(event):
            if self.view == "detail" and self.btw.cancel(self.identifier):
                return
            self.hide()

        @self.keys.add("up", filter=browsing)
        def up(event):
            self.index = max(0, self.index - 1)
            event.app.invalidate()

        @self.keys.add("down", filter=browsing)
        def down(event):
            self.index = min(len(self.records) - 1, self.index + 1)
            event.app.invalidate()

        @self.keys.add("enter", filter=browsing)
        def select(event):
            self.show(self.ordered()[self.index])

        @self.keys.add("pageup", filter=detail)
        def page_up(event):
            self.scroll(-1)

        @self.keys.add("pagedown", filter=detail)
        def page_down(event):
            self.scroll(1)

        @self.keys.add("f", filter=visible & complete & empty)
        def follow(event):
            identifier = self.record()["id"]
            event.current_buffer.text = f"/btw --follow-up {identifier} "
            event.current_buffer.cursor_position = len(event.current_buffer.text)

        @self.keys.add("c", filter=visible & empty & Condition(lambda: bool(
            (record := self.record()) and copy_text(record)
        )))
        def copy(event):
            try:
                clipboard(ANSI.sub("", copy_text(self.record())).expandtabs(4).strip())
                self.notice = "Copied /btw answer to clipboard."
            except (OSError, subprocess.SubprocessError) as error:
                self.notice = f"Clipboard copy failed · {error}"
            event.app.invalidate()

        @self.keys.add("b", filter=detail & empty & Condition(lambda: bool(
            self.btw.completed and self.btw.completed.record["id"] == self.identifier
        )))
        def branch(event):
            event.app.exit(result=(f"/btw --branch {self.identifier}", event.current_buffer.text))

        @self.keys.add("escape", "i", filter=detail & Condition(lambda: bool(
            (record := self.record()) and self.btw.injection(record)
        )), eager=True)
        def inject(event):
            event.app.exit(result=(f"/btw --inject {self.identifier}", event.current_buffer.text))

    def ordered(self):
        return sorted(self.records.copy().values(), key=lambda record: (-record["createdAt"], record["id"]))

    def record(self):
        if self.view == "list":
            records = self.ordered()
            return records[min(self.index, len(records) - 1)] if records else None
        return self.records.get(self.identifier)

    def open(self):
        self.records = {record["id"]: record for record in self.btw.records()}
        self.view, self.index, self.notice = "list", 0, ""
        if request := self.btw.running:
            self.show(request.record)
        self.editor.app.invalidate()

    def show(self, record):
        current = self.records.get(record["id"])
        if current is None or latest(current)["updatedAt"] < latest(record)["updatedAt"]:
            self.records[record["id"]] = record
        elif "title" in record and "title" not in current:
            self.records[record["id"]] = {**current, "title": record["title"]}
        self.view, self.identifier, self.first, self.notice = "detail", record["id"], None, ""
        self.editor.app.invalidate()

    def hide(self):
        self.view, self.notice = None, ""
        self.editor.app.invalidate()

    def event(self, kind, **fields):
        if kind == "btw_record":
            record = fields["record"]
            previous = self.records.get(record["id"], {})
            self.records[record["id"]] = {**record, "title": previous["title"]} if "title" in previous else record
        elif kind == "btw_delta" and (request := self.btw.running):
            record = request.record
            self.records[record["id"]] = record
        elif kind == "notice":
            self.notice = fields["message"]
            if self.view is None:
                print(self.notice, file=sys.stderr)
        self.editor.app.invalidate()

    def dimensions(self):
        size = self.editor.app.output.get_size()
        return max(12, size.columns - 3), max(2, min(16, size.rows // 2))

    def body(self, record, width):
        parts = []
        for index, turn in enumerate(turns(record)):
            parts.append("› " + turn["question"])
            if turn["answer"]:
                mark = self.btw.injection_state(record, index)
                if mark and (block := split_inject(turn["answer"])):
                    before, text, after = block
                    label = "Injected into main session" if mark == "sent" else "Inject into main session"
                    parts.extend([before.strip(), *("│ " + line for line in text.splitlines()), "│ " + label, after.strip()])
                else:
                    parts.append(turn["answer"])
            if turn.get("error"):
                parts.append(turn["error"])
            if turn["status"] != "complete":
                parts.append(f"[{turn['status']}]")
        text = ANSI.sub("", "\n".join(parts)).expandtabs(4)
        return [line for source in text.splitlines() for line in (wrap(source, width, drop_whitespace=False) or [""])]

    def scroll(self, direction):
        width, height = self.dimensions()
        last = max(0, len(self.body(self.record(), width)) - height)
        first = last if self.first is None else min(self.first, last)
        next_line = min(last, max(0, first + direction * height))
        self.first = None if next_line == last else next_line
        self.editor.app.invalidate()

    def message(self, prompt):
        if self.view is None or get_app().is_done:
            return prompt
        width, height = self.dimensions()
        record = self.record()
        if self.view == "list":
            records = self.ordered()
            first = max(0, self.index - height + 1)
            lines = [("› " if index == self.index else "  ") + ANSI.sub("", item.get("title") or item["question"])
                     .replace("\n", " ") + f" [{latest(item)['status']}]"
                     for index, item in enumerate(records[first:first + height], first)]
            title = "/btw history"
            footer = "↑/↓ select · Enter open · f follow up · c copy · Esc close"
        elif record:
            lines = self.body(record, width)
            last = max(0, len(lines) - height)
            first = last if self.first is None else min(last, self.first)
            lines = lines[first:first + height]
            title = "/btw · " + ANSI.sub("", record.get("title") or turns(record)[-1]["question"]).replace("\n", " ")
            if latest(record)["status"] == "running":
                footer = "Esc cancel"
            else:
                footer = "f follow up · c copy · Esc close"
                if (request := self.btw.completed) and request.record["id"] == record["id"] and request.leaf is not None:
                    if request.leaf == self.leaf() and not record.get("followUps"):
                        footer = "b branch · " + footer
                if self.btw.injection(record):
                    footer = "Alt+I inject · " + footer
            if last:
                footer = "PgUp/PgDn scroll · " + footer
        else:
            return prompt
        body = "\n".join(line[:width] for line in (lines or ["No side conversations."]))
        notice = ANSI.sub("", self.notice).replace("\n", " ")
        return [("bold", title[:width] + "\n"), ("", body + "\n"), ("dim", footer[:width] + "\n"),
                ("", notice[:width] + "\n" if notice else ""), *to_formatted_text(prompt)]
