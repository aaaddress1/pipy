"""Live /omfg panel and the boundary between terminal input and side requests."""

import re
from concurrent.futures import CancelledError
from contextvars import copy_context
from dataclasses import dataclass
from textwrap import wrap

from prompt_toolkit import PromptSession
from prompt_toolkit.application import create_app_session, in_terminal
from prompt_toolkit.application.current import get_app, set_app
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import to_formatted_text
from prompt_toolkit.key_binding import KeyBindings

import omfg
from cancellable_http import turn_scope
from handoff import Cancellation


CONTROLS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


@dataclass
class Request:
    complaint: str
    flow: object
    signal: Cancellation
    status: str = ""
    preview: str = ""
    action: object = None
    future: object = None
    first_line: int | None = None


class OmfgPanel:
    def __init__(self, editor, workers, workspace, user, budget, prepare_request, prepare_validation, saved):
        self.editor, self.workers = editor, workers
        self.workspace, self.user, self.budget = workspace, user, budget
        self.prepare_request, self.prepare_validation, self.saved = prepare_request, prepare_validation, saved
        self.request = None
        self.wake = None
        self.keys = KeyBindings()
        self.dialog_keys = KeyBindings()

        @self.keys.add("escape", filter=Condition(lambda: self.request is not None))
        @self.keys.add("c-c", filter=Condition(lambda: self.request is not None), eager=True)
        def dismiss(event):
            self.cancel()

        @self.dialog_keys.add("escape", eager=True)
        @self.dialog_keys.add("c-c", eager=True)
        def dismiss_dialog(event):
            self.cancel()
            event.app.exit(exception=KeyboardInterrupt())

        has_preview = Condition(lambda: bool(self.request and self.request.preview))
        for bindings in (self.keys, self.dialog_keys):
            bindings.add("pageup", filter=has_preview)(self.page_up)
            bindings.add("pagedown", filter=has_preview)(self.page_down)

    def cancel(self):
        request, self.request = self.request, None
        if request:
            request.signal.cancel()
            if request.future:
                request.future.cancel()
            request.flow.close()
            self.editor.app.invalidate()
            self.notify()

    def start(self, complaint):
        if not complaint:
            print("Usage: /omfg <complaint>")
            return
        self.cancel()
        self.request = Request(complaint, omfg.workflow(complaint, self.workspace, self.user),
                               Cancellation(self.budget, reason="Rule generation cancelled."))
        self.advance(self.request)

    def delta(self, request, piece):
        if request.signal.is_cancelled():
            return
        request.preview = (request.preview + piece)[-24_000:] if piece is not None else ""
        self.editor.app.invalidate()

    def advance(self, request, answer=None):
        try:
            request.signal.check()
            if request.future is not None:
                pending, request.future = request.future, None
                answer = pending.result()
            try:
                action = request.action = request.flow.send(answer)
            except StopIteration as done:
                request.action = None
                request.status = f"Saved {done.value}" if done.value else "Rule was not saved."
                if done.value:
                    try:
                        self.saved()
                    except (OSError, ValueError) as error:
                        request.status = f"Saved rule · refresh failed: {error}"
                return
            if isinstance(action, omfg.Generate):
                request.preview = ""
                request.first_line = None
                request.status = f"Attempt {action.attempt}/{omfg.ATTEMPTS} · generating…"
                send = self.prepare_request()
                request.future = self.submit(request.signal, send, action.prompt, request.signal,
                                             lambda piece: self.delta(request, piece))
            elif isinstance(action, omfg.Validate):
                request.preview = action.content
                request.first_line = 0
                request.status = f"Attempt {action.attempt}/{omfg.ATTEMPTS} · validating…"
                outputs, judge = self.prepare_validation()
                request.future = self.submit(request.signal, omfg.validate, action.item, action.content, outputs, judge)
            else:
                request.preview, request.status = action.content, action.status
        except CancelledError:
            self.cancel()
        except Exception as error:
            request.action = None
            request.preview = ""
            request.status = f"Could not create rule · {error}"
            request.flow.close()
        finally:
            self.editor.app.invalidate()

    def submit(self, signal, action, *arguments):
        def work():
            with turn_scope(signal):
                signal.check()
                return action(*arguments)

        future = self.workers.submit(copy_context().run, work)
        future.add_done_callback(self.notify)
        return future

    def notify(self, _=None):
        if wake := self.wake:
            wake()

    def busy(self):
        return bool(self.request and self.request.action)

    def ready(self):
        if not self.busy():
            return False
        return self.request.future is None or self.request.future.done()

    async def update(self, reader):
        request = self.request
        if request.future is not None:
            self.advance(request)
        if self.request is not request or not isinstance(action := request.action, omfg.Ask):
            return
        app = self.editor.app
        with create_app_session(input=app.input, output=app.output), set_app(app):
            async with in_terminal():
                if reader.done() or self.request is not request:
                    return
                dialog = PromptSession(input=app.input, output=app.output, key_bindings=self.dialog_keys)
                try:
                    answer = await dialog.prompt_async(lambda: self.message(action.prompt), handle_sigint=False)
                except (EOFError, KeyboardInterrupt):
                    self.cancel()
                    return
        if self.request is request:
            self.advance(request, answer)

    def viewport(self, request):
        size = self.editor.app.output.get_size()
        width, height = max(12, size.columns - 3), max(2, min(12, size.rows // 3))
        lines = [line for source in CONTROLS.sub("", request.preview).expandtabs(4).strip().splitlines()
                 for line in (wrap(source, width, drop_whitespace=False) or [""])]
        last = max(0, len(lines) - height)
        first = min(request.first_line, last) if request.first_line is not None else last
        return width, height, lines, first, last

    def page_up(self, event):
        _, height, _, first, _ = self.viewport(self.request)
        self.request.first_line = max(0, first - height)
        event.app.invalidate()

    def page_down(self, event):
        _, height, _, first, last = self.viewport(self.request)
        self.request.first_line = None if first + height >= last else first + height
        event.app.invalidate()

    def message(self, prompt="pi> "):
        request = self.request
        if request is None or get_app().is_done:
            return prompt
        width, height, lines, first, last = self.viewport(request)
        title = CONTROLS.sub("", request.complaint).replace("\n", " ")
        status = CONTROLS.sub("", request.status).replace("\n", " ")
        visible = lines[first:first + height]
        if not visible and self.busy():
            visible = ["Waiting for candidate rule…"]
        footer = "Esc cancel /omfg" if self.busy() else "Esc dismiss"
        if last:
            footer += f" · PgUp/PgDn · {first + 1}–{first + len(visible)}/{len(lines)}"
        return [("bold", f"/omfg {title}"[:width] + "\n"), ("", status[:width] + "\n"),
                ("", "\n".join(visible) + "\n" if visible else ""), ("dim", footer + "\n"), *to_formatted_text(prompt)]
