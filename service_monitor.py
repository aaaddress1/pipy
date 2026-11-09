"""Keyboard and mouse monitor for service scopes in a terminal."""

import asyncio
from functools import partial
from sqlite3 import Error as SQLiteError
from time import monotonic

from prompt_toolkit.application import Application
from prompt_toolkit.data_structures import Point
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.mouse_events import MouseButton, MouseEventType
from prompt_toolkit.styles import Style

from service_logs import log_lines
from service_ps import action, describe, detail_text, named, reports, scope_title, table
from service_runtime import log_path


class Monitor:
    def __init__(self, path, scope, workspace, all_scopes, explicit):
        self.path, self.scope, self.workspace = path, scope, workspace
        self.all_scopes, self.explicit = all_scopes, explicit
        self.data, self.entries = {}, []
        self.selected, self.index = None, 0
        self.view, self.detail = "table", ""
        self.notice, self.busy = "", False
        self.confirm, self.last_click = None, (None, 0)
        self.cursor_line, self.scroll = 0, 0
        content = FormattedTextControl(self.render, get_cursor_position=self.cursor_position)
        footer = FormattedTextControl(self.footer)
        self.app = Application(
            layout=Layout(HSplit([Window(content, always_hide_cursor=True, wrap_lines=False), Window(footer, height=2)])),
            key_bindings=self.keys(), full_screen=True, mouse_support=True,
            style=Style.from_dict({"heading": "bold", "selected": "reverse", "dim": "ansibrightblack"}),
        )

    def footer(self):
        if self.confirm:
            return [("class:heading", f"Kill {self.confirm[1]}? "),
                    ("", "[y] Yes", partial(self.mouse, self.confirm_kill)),
                    ("", "  [n] Cancel\n", partial(self.mouse, self.cancel_kill)), ("", self.notice)]
        if self.view != "table":
            return [("class:dim", "Esc / q back", partial(self.mouse, self.leave)),
                    ("class:dim", " · ↑ ↓ scroll\n"), ("", self.notice)]
        buttons = [
            ("Enter info", partial(self.open, "info")), ("l logs", partial(self.open, "logs")),
            ("s stop", partial(self.queue_action, "stop", None)), ("x kill", self.ask_kill),
            ("r restart", partial(self.queue_action, "restart", None)), ("a all scopes", self.toggle_all),
            ("q quit", self.leave),
        ]
        fragments = [("class:dim", "↑ ↓ select")]
        for title, operation in buttons:
            fragments.extend([("class:dim", " · "), ("class:dim", title, partial(self.mouse, operation))])
        return [*fragments, ("", "\n"), ("", self.notice)]

    def mouse(self, operation, event):
        if event.event_type == MouseEventType.MOUSE_UP and event.button == MouseButton.LEFT:
            operation()
            return None
        return NotImplemented

    def cursor_position(self):
        return Point(x=0, y=self.cursor_line if self.view == "table" else self.scroll)

    def render(self):
        if self.view != "table":
            return [("", self.detail or "Loading…")]
        fragments = [("class:heading", "Mini Pi service monitor\n\n")]
        line = 2
        for scope, daemons in self.data.items():
            fragments.append(("class:heading", scope_title(scope) + "\n"))
            line += 1
            if not daemons:
                fragments.append(("class:dim", "  no processes\n\n"))
                line += 2
                continue
            header, *lines = table(daemons)
            fragments.append(("class:dim", "  " + header + "\n"))
            line += 1
            for daemon, content in zip(daemons, lines):
                key = (scope, daemon["name"])
                selected = key == self.selected
                if selected:
                    self.cursor_line = line
                fragments.append(("class:selected" if selected else "", "  " + content + "\n",
                                  partial(self.click, key)))
                line += 1
            fragments.append(("", "\n"))
            line += 1
        if not self.data:
            fragments.append(("class:dim", "No service scopes found.\n"))
        return fragments

    def click(self, key, event):
        if (event.event_type == MouseEventType.MOUSE_UP and event.button == MouseButton.LEFT
                and key in self.entries and not self.confirm):
            self.select(key)
            previous, then = self.last_click
            now = monotonic()
            self.last_click = (key, now)
            if previous == key and now - then < .4:
                self.open("info")
            self.app.invalidate()
            return None
        return NotImplemented

    def select(self, key):
        self.selected = key
        self.index = self.entries.index(key)

    def move(self, amount):
        if self.view == "table" and self.entries:
            self.index = max(0, min(len(self.entries) - 1, self.index + amount))
            self.selected = self.entries[self.index]
        elif self.view != "table":
            self.scroll = max(0, min(max(0, len(self.detail.splitlines()) - 1), self.scroll + amount))
        self.app.invalidate()

    def leave(self):
        if self.confirm:
            self.cancel_kill()
        elif self.view == "table":
            self.app.exit()
        else:
            self.view = "table"
            self.detail = ""
            self.app.invalidate()

    def ask_kill(self):
        if self.selected and not self.busy:
            self.confirm = self.selected
            self.app.invalidate()

    def cancel_kill(self):
        self.confirm = None
        self.app.invalidate()

    def confirm_kill(self):
        selected = self.confirm
        self.cancel_kill()
        if selected and not self.busy:
            self.busy = True
            self.app.create_background_task(self.perform(selected, "kill"))

    def toggle_all(self):
        if self.view == "table" and not self.confirm:
            self.all_scopes = not self.all_scopes
            self.notice = "Showing all scopes." if self.all_scopes else "Showing the selected scope."
            self.app.create_background_task(self.refresh())

    def keys(self):
        keys = KeyBindings()

        @keys.add("c-c")
        def quit_monitor(event):
            event.app.exit()

        @keys.add("q")
        @keys.add("escape")
        def back(_):
            self.leave()

        @keys.add("up")
        @keys.add("k")
        def up(_):
            self.move(-1)

        @keys.add("down")
        @keys.add("j")
        def down(_):
            self.move(1)

        @keys.add("a")
        def toggle(_):
            self.toggle_all()

        @keys.add("y")
        def yes(_):
            self.confirm_kill()

        @keys.add("n")
        def no(_):
            if self.confirm:
                self.cancel_kill()

        @keys.add("enter")
        @keys.add("i")
        def details(_):
            self.open("info")

        @keys.add("l")
        def logs(_):
            self.open("logs")

        for key, verb in (("s", "stop"), ("x", "kill"), ("r", "restart")):
            keys.add(key)(partial(self.queue_action, verb))
        return keys

    def open(self, view):
        if self.view == "table" and self.selected and not self.confirm:
            self.view, self.detail, self.scroll = view, "", 0
            self.app.create_background_task(self.refresh())

    def queue_action(self, verb, _):
        if self.view == "table" and self.selected and not self.busy and not self.confirm:
            self.busy = True
            self.app.create_background_task(self.perform(self.selected, verb))

    async def perform(self, selected, verb):
        scope, name = selected
        self.notice = f"{verb.capitalize()} {name}…"
        self.app.invalidate()
        try:
            daemon = await asyncio.to_thread(action, self.path, scope, self.workspace, name, verb)
            self.notice = f"{name} [{daemon['state']}]"
        except (OSError, SQLiteError, ValueError) as error:
            self.notice = str(error)
        finally:
            self.busy = False
        await self.refresh()

    async def refresh(self):
        view, selected = self.view, self.selected
        try:
            if view == "table":
                all_scopes = self.all_scopes
                data = await asyncio.to_thread(reports, self.path, self.scope, all_scopes=all_scopes, explicit=self.explicit)
                if self.view != view or self.all_scopes != all_scopes:
                    return
                self.data = data
                self.entries = [(scope, daemon["name"]) for scope, daemons in data.items() for daemon in daemons]
                if self.entries:
                    self.select(self.selected if self.selected in self.entries else self.entries[min(self.index, len(self.entries) - 1)])
                else:
                    self.selected, self.index, self.cursor_line = None, 0, 0
            elif selected:
                detail = await asyncio.to_thread(self.read_view, view, *selected)
                if self.view == view and self.selected == selected:
                    self.detail = detail
        except (OSError, SQLiteError, ValueError) as error:
            self.notice = str(error)
        self.app.invalidate()

    def read_view(self, view, scope, name):
        row = named(self.path, scope, name)
        daemon = describe(row, spec=True)
        if view == "info":
            return detail_text(daemon)
        lines = log_lines(log_path(self.path, row["id"]), 1000)
        return "\n".join([f"{name} [{daemon['state']}]", "", *lines])

    async def poll(self):
        while True:
            await self.refresh()
            await asyncio.sleep(.5)

    def run(self):
        self.app.run(pre_run=lambda: self.app.create_background_task(self.poll()))


def monitor(path, scope, workspace, all_scopes, explicit):
    Monitor(path, scope, workspace, all_scopes, explicit).run()
