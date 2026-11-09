"""A single writer for interactive turns, with a live terminal composer."""

import sys
from collections import deque
from concurrent.futures import CancelledError
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock, RLock, Timer, current_thread

from prompt_toolkit.application.current import get_app
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import to_formatted_text
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.output.vt100 import Vt100_Output
from prompt_toolkit.utils import get_cwidth

from btw import ANSI
from cost_gate import usd
from handoff import Cancellation
from live_steering import LiveQueue
from loop_mode import LIMIT_REACHED, TIME_REACHED, Loop, LoopReset, condition_stop, parse
from thinking_loop import ThinkingLoop
from todo_hud import COMPACT_ROWS, truncate as truncate_cells


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
        self.goal_next = None
        self.goal_internal_abort = False
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
    def __init__(self, editor, executor, run_task, budget, settings, notify_jobs, *, goal_status=None, todo_hud=None,
                 loop_action="prompt", loop_timeout=30, loop_shell=None, loop_compact=None):
        self.editor, self.executor, self.run_task, self.budget = editor, executor, run_task, budget
        self.notify_jobs = notify_jobs
        self.goal_status = goal_status
        self.todo_hud = todo_hud
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
        self.goal_timer = None
        self.loop = self.loop_timer = self.loop_gate = self.loop_reset = None
        self.loop_workers = set()
        self.loop_action, self.loop_timeout = loop_action, loop_timeout
        self.loop_shell, self.loop_compact = loop_shell, loop_compact
        self.idle_compacting = False
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

        @self.keys.add("escape", filter=Condition(lambda: self.busy() or self.loop is not None))
        @self.keys.add("c-c", filter=Condition(lambda: self.busy() or self.loop_pending()), eager=True)
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

    def submit(self, work, mode="steer", *, urgent=False, loop_prompt=False):
        with self.lock:
            self.cancel_goal()
            if self.budget:
                self.budget.check()
            if self.loop and loop_prompt and isinstance(work.content, str):
                self.cancel_loop_pending()
                self.loop.capture(work.content)
            if not work.background:
                self.background_paused = False
            self.settled = None
            if self.turn and urgent:
                if self.priority or self.turn.works[0].operation:
                    raise ValueError("Compaction already in progress.")
                self.priority = work
                self.turn.goal_internal_abort = True
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
                    next_goal = turn.goal_next if not queued and not failed and not turn.is_cancelled() else None
                    turn = self.turn = TerminalTurn(self, queued) if queued else None
                    if not turn:
                        self.settled = bool(answer and answer.strip())
                    if not turn and self.loop:
                        if failed and not work.background:
                            self.cancel_loop_pending()
                            self.loop.capture(None)
                        else:
                            self.schedule_loop()
                    elif next_goal:
                        self.schedule_goal(next_goal)
                self.changed()

    def cancel_goal(self):
        if self.goal_timer:
            self.goal_timer.cancel()
            self.goal_timer = None

    def schedule_goal(self, next_goal):
        with self.lock:
            self.cancel_goal()
            if self.turn or self.loop or not next_goal():
                return

            def admit():
                with self.lock:
                    if self.goal_timer is not timer:
                        return
                    self.goal_timer = None
                    if (not self.turn and not self.editor.default_buffer.text.strip()
                            and not (self.budget and self.budget.is_cancelled()) and (content := next_goal())):
                        self.submit(TerminalWork(content, expand_prompt=False))

            timer = self.goal_timer = Timer(.8, admit)
            timer.daemon = True
            timer.start()

    def cancel_loop_pending(self):
        if self.loop_timer:
            self.loop_timer.cancel()
        if self.loop_gate:
            self.loop_gate.cancel()
        self.loop_timer = self.loop_gate = self.loop_reset = None

    def disable_loop(self, message="Loop mode disabled."):
        self.cancel_loop_pending()
        self.loop = None
        print(message, file=sys.stderr)
        self.changed()

    def toggle_loop(self, arguments):
        with self.lock:
            if self.loop:
                self.disable_loop()
                return ""
            count, seconds, condition, prompt = parse(arguments)
            self.cancel_goal()
            self.loop = Loop(count, seconds, condition)
            print("Loop mode enabled. Esc pauses. /loop disables.", file=sys.stderr)
        self.changed()
        return prompt

    def loop_pending(self):
        return bool(self.loop_timer or self.loop_gate or self.loop_reset)

    def schedule_loop(self):
        loop = self.loop
        if not (loop and loop.prompt) or self.loop_pending():
            return

        def admit():
            with self.lock:
                if self.loop_timer is not timer:
                    return
                self.loop_timer = None
                if self.budget and self.budget.is_cancelled():
                    self.disable_loop("Loop stopped because the run deadline was reached.")
                    return
                if loop.exhausted():
                    self.disable_loop(LIMIT_REACHED)
                    return
                if self.loop_blocked():
                    self.schedule_loop()
                    return
                gate = self.loop_gate = Cancellation(self.budget)
                self.loop_workers.add(current_thread())
            try:
                verdict = condition_stop(loop.condition, self.loop_shell, self.loop_timeout, gate) if loop.condition else None
                with self.lock:
                    if self.loop_gate is gate and not gate.is_cancelled():
                        self.loop_gate = None
                        self.admit_loop(loop, verdict)
            except CancelledError:
                pass
            finally:
                with self.lock:
                    self.loop_workers.discard(current_thread())
                    if self.loop_gate is gate:
                        self.loop_gate = None
                        if self.budget and self.budget.is_cancelled():
                            self.disable_loop("Loop stopped because the run deadline was reached.")
                self.changed()

        timer = self.loop_timer = Timer(.8, admit)
        timer.daemon = True
        timer.start()

    def loop_blocked(self):
        return self.turn or self.idle_compacting or self.editor.default_buffer.text or not self.editor.app.is_running

    def admit_loop(self, loop, verdict):
        if verdict:
            self.disable_loop(verdict)
            return
        if self.loop_blocked():
            self.schedule_loop()
            return
        if not loop.consume():
            self.disable_loop(LIMIT_REACHED)
            return
        if self.loop_action == "reset":
            self.loop_reset = LoopReset(loop)
        elif self.loop_action == "compact":
            revision = loop.revision

            def compact_then_run(turn):
                self.loop_compact(turn)
                with self.lock:
                    turn.check()
                    if self.loop is not loop or loop.revision != revision:
                        return None
                    if loop.expired():
                        self.disable_loop(TIME_REACHED)
                        return None
                    prompt = loop.prompt
                    turn.works[0].operation = None
                return self.run_task(prompt, turn)

            self.submit(TerminalWork(None, operation=compact_then_run))
        else:
            self.submit(TerminalWork(loop.prompt))

    def take_loop_reset(self, reset):
        with self.lock:
            if self.loop_reset is not reset:
                return None
            self.loop_reset = self.loop = None
            return reset.loop

    def close(self):
        self.abort()
        with self.lock:
            workers = tuple(self.loop_workers)
        for worker in workers:
            worker.join()

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
            self.cancel_goal()
            self.cancel_loop_pending()
            if self.loop:
                self.loop.capture(None)
            turn = self.turn
            waiting = [*self.steers, *([self.priority] if self.priority else []), *self.followups]
            self.priority = None
            self.steers.clear()
            self.followups.clear()
            if turn:
                waiting = [*turn.claimed, *turn.live.take_all(), *turn.works[1:], *waiting]
                turn.claimed.clear()
                turn.goal_internal_abort = False
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
            loop_status = self.loop.status() if self.loop else ""
        size = self.editor.app.output.get_size()
        width = max(12, size.columns - 3)
        progress = []
        compact_todo = ""
        if self.todo_hud:
            if size.rows < COMPACT_ROWS:
                compact_todo = self.todo_hud.short(width if not turn else max(10, int(width * .45)))
            else:
                progress.extend(self.todo_hud.full(width))
        if self.goal_status and (status := self.goal_status()):
            progress.append(("dim", status + "\n"))
        if loop_status:
            progress.append(("dim", truncate_cells(loop_status, width) + "\n"))
        for batch in batches:
            intent = " ".join(ANSI.sub("", batch["intent"]).split())
            row = f"{intent} · {batch['done']}/{batch['total']}"
            if batch["failed"]:
                row += f" · {batch['failed']} failed"
            if batch["cost"] > 0:
                row += " · " + usd(batch["cost"])
            progress.append(("dim", row[:width] + "\n"))
        if not turn:
            if compact_todo:
                progress.append(("fg:ansicyan", " " * max(0, width - get_cwidth(compact_todo)) + compact_todo + "\n"))
            return [*progress, *to_formatted_text(prompt)] if progress else prompt
        draft = ANSI.sub("", turn.preview)
        tail = "\n".join(draft.splitlines()[-max(1, min(8, size.rows // 4)):])
        status = "Interrupting…" if turn.is_cancelled() else "Compacting context…" if turn.compacting else "Working…"
        status += "  Enter steer · Ctrl+Q follow-up · Esc stop"
        status_row = [("dim", status + "\n")]
        if compact_todo:
            room = max(0, width - get_cwidth(compact_todo) - 2)
            left = truncate_cells(status, room)
            gap = " " * max(1, width - get_cwidth(left) - get_cwidth(compact_todo))
            status_row = [("dim", left + gap), ("fg:ansicyan", compact_todo + "\n")]
        labels = [f"{mode} · {item.content}" for mode, queue in (("Steer", steers), ("Follow-up", followups))
                  for item in queue if isinstance(item.content, str)]
        rows = [ANSI.sub("", label).replace("\n", " ")[:width] for label in labels[-4:]]
        return [("", tail + "\n" if tail else ""), *status_row,
                *progress, *[("dim", row + "\n") for row in rows], *to_formatted_text(prompt)]
