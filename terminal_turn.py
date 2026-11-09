"""A single writer for interactive turns, with a live terminal composer."""

import sys
from collections import deque
from concurrent.futures import CancelledError
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock, RLock, Timer

from prompt_toolkit.application.current import get_app
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import to_formatted_text
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.output.vt100 import Vt100_Output

from btw import ANSI
from cost_gate import usd
from handoff import Cancellation
from live_steering import LiveQueue
from thinking_loop import ThinkingLoop


def clear_terminal(editor):
    app = editor.app
    app.renderer.reset()
    output = app.output
    ansi_outputs = (Vt100_Output,)
    if sys.platform == "win32":
        from prompt_toolkit.output.conemu import ConEmuOutput
        from prompt_toolkit.output.windows10 import Windows10_Output

        ansi_outputs += (ConEmuOutput, Windows10_Output)
    output.cursor_goto(row=0, column=0)
    output.erase_screen()
    if isinstance(output, ansi_outputs):
        output.write_raw("\x1b[3J")
    output.flush()


@dataclass(eq=False)
class TerminalWork:
    content: object
    background: bool = False
    expand_prompt: bool = True
    on_admitted: object = None
    on_drop: object = None
    operation: object = None
    admitted: bool = False
    returned: bool = False
    live_steered: bool = False

    def accept(self):
        self.admitted = True
        if self.on_admitted:
            self.on_admitted()

    def restore(self):
        if self.admitted or self.returned:
            return ""
        self.returned = True
        if self.on_drop:
            self.on_drop()
            return ""
        return self.content if isinstance(self.content, str) else ""


class TerminalTurn(Cancellation):
    steering_at_boundary = True

    def __init__(self, lane, works):
        super().__init__(lane.budget, reason="Turn interrupted.")
        self.server = lane
        self.works = works
        self.claimed = []
        self.live = LiveQueue(self, lane.steers, lambda: lane.steering_mode, lane.repaint)
        self.background = works[0].background
        self.soft_steering = lane.interrupt_mode != "wait"
        self.compacting = bool(works[0].operation)
        self.request_leaf = None
        self.system = None
        self.draft_lock = Lock()
        self.text, self.thinking = [], []
        self.preview = ""

    def request(self):
        self.check()
        self.compacting = False
        with self.draft_lock:
            self.text.clear()
            self.thinking.clear()
            self.preview = ""
        self.server.repaint()

    def delta(self, kind, value):
        if value:
            with self.draft_lock:
                (self.text if kind == "text" else self.thinking).append(value)
                if kind == "text":
                    self.preview = (self.preview + value)[-6000:]
            if kind == "text":
                self.server.repaint()

    def snapshot(self):
        with self.draft_lock:
            text, thinking = "".join(self.text), "".join(self.thinking)
        if not (text or thinking):
            return None
        draft = {"role": "assistant", "content": text or None}
        if thinking:
            draft["_thinking"] = thinking
        return draft

    def flush_text(self, text):
        with self.draft_lock:
            self.text.clear()
            self.thinking.clear()
            self.preview = ""
        if text:
            print(ANSI.sub("", text), flush=True)
        self.server.repaint()

    def response(self, message):
        self.flush_text(message.get("content"))
        return message

    def publish(self, message):
        if message["role"] == "assistant":
            self.flush_text(message.get("content"))

    def interrupted(self, error, aborted=False):
        self.flush_text(None if isinstance(error, ThinkingLoop) else "".join(self.text))

    def steering(self):
        with self.server.lock:
            self.check()
            accepted, deferred = self.live.drain()
            for work in accepted:
                work.live_steered = True
            incoming = accepted + deferred if accepted else self.server.take(self.server.steers, self.server.steering_mode)
            self.claimed.extend(incoming)
        if incoming:
            self.server.repaint()
        return incoming

    def has_steering(self):
        with self.server.lock:
            return bool(self.live.accepted or self.server.steers)

    @contextmanager
    def admission(self, work):
        with self.server.lock:
            self.check()
            yield
            work.admitted = True
            if work in self.claimed:
                self.claimed.remove(work)

    def tool_start(self, call):
        pass

    def allow_tool(self, call):
        return True

    def tool_finished(self, call, result):
        pass


class TerminalLane:
    def __init__(self, editor, executor, run_task, budget, settings, notify_jobs):
        self.editor, self.executor, self.run_task, self.budget = editor, executor, run_task, budget
        self.notify_jobs = notify_jobs
        self.steering_mode = settings.get("steeringMode", "one-at-a-time")
        self.followup_mode = settings.get("followUpMode", "one-at-a-time")
        self.interrupt_mode = settings.get("interruptMode", "immediate")
        self.lock = RLock()
        self.steers, self.followups = deque(), deque()
        self.judgment_batches = {}
        self.priority = None
        self.returned = []
        self.turn, self.settled = None, None
        self.background_paused = False
        self.wake = None
        self.keys = KeyBindings()

        @self.keys.add("c-q", eager=True)
        def follow(event):
            if text := event.current_buffer.text.strip():
                event.app.exit(result=(text, "", "followUp"))

        @self.keys.add("escape", "up")
        @self.keys.add("s-up")
        def dequeue(event):
            if text := self.dequeue():
                self.restore(event.current_buffer, [text])

        @self.keys.add("escape", filter=Condition(self.busy))
        @self.keys.add("c-c", filter=Condition(self.busy), eager=True)
        def interrupt(event):
            self.restore(event.current_buffer, self.abort())

    @staticmethod
    def restore(buffer, texts):
        restored = "\n\n".join(part for part in [*texts, buffer.text] if part.strip())
        buffer.document = Document(restored, len(restored))

    @staticmethod
    def take(queue, mode):
        if not queue:
            return []
        result = [queue.popleft()]
        if mode == "all" and not result[0].operation:
            while queue and not queue[0].operation:
                result.append(queue.popleft())
        return result

    def busy(self):
        return self.turn is not None

    def repaint(self):
        self.editor.app.invalidate()

    def changed(self):
        self.repaint()
        if wake := self.wake:
            wake()

    def submit(self, work, mode="steer", *, urgent=False):
        with self.lock:
            if self.budget:
                self.budget.check()
            if not work.background:
                self.background_paused = False
            self.settled = None
            if self.turn and urgent:
                if self.priority or self.turn.works[0].operation:
                    raise ValueError("Compaction already in progress.")
                self.priority = work
                self.turn.cancel()
            elif self.turn:
                ordered = (mode == "followUp" or self.priority or self.turn.works[0].operation
                           or any(item.operation for item in self.followups))
                (self.followups if ordered else self.steers).append(work)
            else:
                self.turn = TerminalTurn(self, [work])
                self.executor.submit(self.run, self.turn)
        if self.notify_jobs:
            self.notify_jobs()
        self.changed()

    def run(self, turn):
        while turn:
            work = turn.works[0]
            answer = None
            failed = False
            try:
                if work.operation:
                    turn.check()
                    work.accept()
                    answer = work.operation(turn)
                else:
                    answer = self.run_task(work.content, turn, background=work.background,
                                           expand_prompt=work.expand_prompt, on_admitted=work.accept)
            except CancelledError as error:
                failed = True
                turn.interrupted(error, aborted=True)
                print("[Turn interrupted.]", file=sys.stderr)
            except Exception as error:
                failed = True
                turn.interrupted(error)
                print(f"mini-pi: {error}", file=sys.stderr)
            finally:
                with self.lock:
                    for item in [*turn.claimed, *turn.live.take_all(), *turn.works[1:]]:
                        if text := item.restore():
                            self.returned.append(text)
                    if (work.on_drop or failed) and (text := work.restore()):
                        self.returned.append(text)
                    queued = []
                    if not (self.budget and self.budget.is_cancelled()):
                        if self.priority:
                            queued, self.priority = [self.priority], None
                        else:
                            queued = self.take(self.steers, self.steering_mode) or self.take(self.followups, self.followup_mode)
                    turn = self.turn = TerminalTurn(self, queued) if queued else None
                    if not turn:
                        self.settled = bool(answer and answer.strip())
                self.changed()

    def take_settled(self):
        with self.lock:
            result, self.settled = self.settled, None
            return result

    def take_returned(self):
        with self.lock:
            texts, self.returned = self.returned, []
            return texts

    def has_queued(self):
        with self.lock:
            return bool(self.priority or self.steers or self.followups)

    def abort(self):
        with self.lock:
            turn = self.turn
            waiting = [*self.steers, *([self.priority] if self.priority else []), *self.followups]
            self.priority = None
            self.steers.clear()
            self.followups.clear()
            if turn:
                waiting = [*turn.claimed, *turn.live.take_all(), *turn.works[1:], *waiting]
                turn.claimed.clear()
                turn.cancel()
            self.background_paused = True
            restored = [text for work in waiting if (text := work.restore())]
        if self.notify_jobs:
            self.notify_jobs()
        self.changed()
        return restored

    def dequeue(self):
        with self.lock:
            for queue in (self.steers, self.followups):
                for index in range(len(queue) - 1, -1, -1):
                    if isinstance(queue[index].content, str) and not queue[index].on_drop:
                        work = queue[index]
                        del queue[index]
                        self.changed()
                        return work.content
        return ""

    def event(self, kind, **fields):
        if kind == "ttsr_triggered":
            print("[TTSR: " + ", ".join(rule["name"] for rule in fields["rules"]) + "]", file=sys.stderr)
        if kind == "irc_message" and fields["message"].get("customType") == "irc:workpool":
            print(fields["message"]["content"], file=sys.stderr)

    def judgment_progress(self, progress):
        identifier = progress["id"]
        with self.lock:
            self.judgment_batches[identifier] = progress
        self.repaint()
        if not progress["running"]:
            def expire():
                with self.lock:
                    self.judgment_batches.pop(identifier, None)
                self.repaint()

            timer = Timer(1, expire)
            timer.daemon = True
            timer.start()

    def message(self, prompt):
        turn = self.turn
        if get_app().is_done:
            return prompt
        with self.lock:
            steers, followups = tuple(self.steers), tuple(self.followups)
            batches = tuple(self.judgment_batches.values())
        size = self.editor.app.output.get_size()
        width = max(12, size.columns - 3)
        progress = []
        for batch in batches:
            intent = " ".join(ANSI.sub("", batch["intent"]).split())
            row = f"{intent} · {batch['done']}/{batch['total']}"
            if batch["failed"]:
                row += f" · {batch['failed']} failed"
            if batch["cost"] > 0:
                row += " · " + usd(batch["cost"])
            progress.append(("dim", row[:width] + "\n"))
        if not turn:
            return [*progress, *to_formatted_text(prompt)] if progress else prompt
        draft = ANSI.sub("", turn.preview)
        tail = "\n".join(draft.splitlines()[-max(1, min(8, size.rows // 4)):])
        status = "Interrupting…" if turn.is_cancelled() else "Compacting context…" if turn.compacting else "Working…"
        labels = [f"{mode} · {item.content}" for mode, queue in (("Steer", steers), ("Follow-up", followups))
                  for item in queue if isinstance(item.content, str)]
        rows = [ANSI.sub("", label).replace("\n", " ")[:width] for label in labels[-4:]]
        return [("", tail + "\n" if tail else ""), ("dim", status + "  Enter steer · Ctrl+Q follow-up · Esc stop\n"),
                *progress, *[("dim", row + "\n") for row in rows], *to_formatted_text(prompt)]
