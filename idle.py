"""Idle recap and context maintenance for an open terminal session."""

import asyncio
import sys
from concurrent.futures import InvalidStateError
from contextlib import suppress

from prompt_toolkit.application import create_app_session
from prompt_toolkit.key_binding import merge_key_bindings
from prompt_toolkit.patch_stdout import patch_stdout


class IdleMaintenance:
    def __init__(
        self, editor, *, recap_seconds, compact_seconds, jobs_busy,
        needs_compaction, compact, recap, record_recap, compact_deadline=None,
        background=None, background_ready=None, background_enabled=None, take_confirmation=None, side_panel=None,
        btw_panel=None, lane=None,
    ):
        self.editor = editor
        self.recap_seconds = recap_seconds
        self.compact_seconds = compact_seconds
        self.jobs_busy = jobs_busy
        self.needs_compaction = needs_compaction
        self.compact = compact
        self.recap = recap
        self.record_recap = record_recap
        self.compact_deadline = compact_deadline or (lambda start, delay: start + delay)
        self.background = background
        self.background_ready = background_ready
        self.background_enabled = background_enabled or (lambda: bool(background))
        self.take_confirmation = take_confirmation
        self.pending_confirmation = None
        self.pending_foreground = None
        self.side_panel = side_panel
        self.btw_panel = btw_panel
        self.lane = lane

    async def prompt(self, settled, can_recap, draft=""):
        run_background = self.background_enabled()
        panel = self.side_panel

        def message():
            base = self.lane.message("pi> ") if self.lane else "pi> "
            if panel:
                base = panel.message(base)
            if self.btw_panel:
                base = self.btw_panel.message(base)
            return base

        bindings = [item.keys for item in (self.lane, panel, self.btw_panel) if item]
        options = {"key_bindings": merge_key_bindings(bindings)} if bindings else {}
        if not (self.lane or settled or run_background or self.take_confirmation or panel and panel.busy()):
            return await self.editor.prompt_async(message, default=draft, **options)
        wake = asyncio.Event()
        revision = 0
        loop = asyncio.get_running_loop()

        def side_ready():
            with suppress(RuntimeError):
                loop.call_soon_threadsafe(wake.set)

        def edited(_):
            nonlocal revision
            revision += 1
            wake.set()

        changes = self.editor.default_buffer.on_text_changed
        changes += edited
        for source in (panel, self.lane):
            if source:
                source.wake = side_ready
        try:
            with create_app_session(input=self.editor.app.input, output=self.editor.app.output), patch_stdout():
                reader = asyncio.create_task(self.editor.prompt_async(message, default=draft, **options))
                reader.add_done_callback(lambda _: wake.set())
                maintenance = asyncio.create_task(self.watch(reader, wake, lambda: revision, settled, can_recap, run_background))
                try:
                    return await reader
                finally:
                    wake.set()
                    await maintenance
        finally:
            changes -= edited
            for source in (panel, self.lane):
                if source:
                    source.wake = None

    async def watch(self, reader, wake, revision, settled, can_recap, run_background):
        now = asyncio.get_running_loop().time
        armed_at = None
        seen_revision = revision()
        recap_job = None
        recap_owed = bool(settled and can_recap and self.recap_seconds)
        compaction_owed = bool(settled and self.compact_seconds)
        panel = self.side_panel
        lane = self.lane
        try:
            while not reader.done() and (recap_owed or compaction_owed or recap_job or run_background
                                         or self.take_confirmation or lane or panel and panel.busy()):
                wake.clear()
                if lane:
                    if (reset := lane.loop_reset) is not None and not lane.busy() and not self.editor.default_buffer.text:
                        app = self.editor.app
                        if app.is_running and not app.is_done:
                            app.exit(result=reset)
                            return
                    if returned := lane.take_returned():
                        lane.restore(self.editor.default_buffer, returned)
                    if (completed := lane.take_settled()) is not None:
                        recap_owed = bool(completed and self.recap_seconds)
                        compaction_owed = bool(self.compact_seconds)
                        armed_at = None
                if panel and panel.ready():
                    await self.cancel_recap(recap_job)
                    recap_job = None
                    await panel.update(reader)
                    armed_at = None
                    continue
                draft = bool(self.editor.default_buffer.text)
                busy = self.jobs_busy()
                foreground = bool(lane and lane.busy())
                if revision() != seen_revision or draft or busy or foreground:
                    seen_revision = revision()
                    armed_at = None
                    if recap_job:
                        await self.cancel_recap(recap_job)
                        recap_job = None
                        recap_owed = True
                if foreground and self.take_confirmation:
                    offer = self.pending_foreground
                    if offer is None or offer.answer.done():
                        offer = self.pending_foreground = self.take_confirmation(foreground_only=True)
                    if offer:
                        if await self.confirm(reader, offer):
                            self.pending_foreground = None
                        continue
                if draft or busy or foreground:
                    await self.wait(wake, 0.25 if busy or foreground and self.take_confirmation else None)
                    continue

                if self.take_confirmation:
                    offer = self.pending_confirmation
                    if offer is None or offer.answer.done():
                        offer = self.pending_confirmation = self.take_confirmation()
                    if offer is not None:
                        if recap_job:
                            await self.cancel_recap(recap_job)
                            recap_job = None
                            recap_owed = True
                        if await self.confirm(reader, offer):
                            self.pending_confirmation = None
                        armed_at = None
                        continue

                if run_background and not reader.done() and self.background_ready():
                    await self.cancel_recap(recap_job)
                    recap_job = None
                    try:
                        await asyncio.to_thread(self.background)
                    except Exception as error:
                        print(f"[MCP background turn failed: {type(error).__name__}]", file=sys.stderr)
                    armed_at = None
                    continue

                if lane and lane.loop_pending():
                    await self.wait(wake, None)
                    continue

                if armed_at is None:
                    armed_at = now()
                compact_due = (
                    self.compact_deadline(armed_at, self.compact_seconds)
                    if compaction_owed and self.needs_compaction() else None
                )
                recap_due = armed_at + self.recap_seconds if recap_owed else None
                current = now()
                if compact_due is not None and compact_due <= current:
                    await self.cancel_recap(recap_job)
                    recap_job = None
                    if not reader.done() and not self.editor.default_buffer.text and self.needs_compaction():
                        if lane:
                            with lane.lock:
                                if lane.busy() or lane.loop_pending():
                                    continue
                                lane.idle_compacting = True
                        try:
                            await asyncio.to_thread(self.compact)
                        except Exception as error:
                            print(f"[idle compaction failed: {type(error).__name__}]", file=sys.stderr)
                        finally:
                            if lane:
                                with lane.lock:
                                    lane.idle_compacting = False
                    if not (lane or run_background or self.take_confirmation or panel and panel.busy()):
                        return
                    recap_owed = compaction_owed = False
                    armed_at = None
                    continue

                if recap_job and recap_job.done():
                    try:
                        text = recap_job.result()
                        if text and text.strip():
                            self.record_recap(text.strip())
                            print(f"\n[recap] {text.strip()}", file=sys.stderr)
                    except Exception as error:
                        print(f"[idle recap failed: {type(error).__name__}]", file=sys.stderr)
                    recap_job = None
                if recap_due is not None and recap_due <= current:
                    recap_owed = False
                    recap_job = asyncio.create_task(self.recap())
                    recap_job.add_done_callback(lambda _: wake.set())
                    recap_due = None

                deadlines = [deadline for deadline in (compact_due, recap_due) if deadline is not None]
                if run_background or self.take_confirmation:
                    deadlines.append(now() + 0.25)
                if not deadlines and recap_job is None and not (lane or panel and panel.busy()):
                    return
                await self.wait(wake, max(0, min(deadlines) - now()) if deadlines else None)
        finally:
            await self.cancel_recap(recap_job)

    async def confirm(self, reader, offer):
        from prompt_toolkit import PromptSession
        from prompt_toolkit.application import in_terminal
        from prompt_toolkit.application.current import set_app

        from auto_reset import CONSENT_PROMPT, consent_value

        app = self.editor.app
        with create_app_session(input=app.input, output=app.output), set_app(app):
            async with in_terminal():
                if offer.answer.done():
                    return True
                if reader.done() or not offer.foreground and (self.editor.default_buffer.text or self.jobs_busy()):
                    return False
                app.output.write(offer.question + "\n")
                app.output.flush()
                prompt = PromptSession(input=app.input, output=app.output)

                async def read_choice():
                    try:
                        return consent_value(await prompt.prompt_async(CONSENT_PROMPT, handle_sigint=False))
                    except (EOFError, KeyboardInterrupt):
                        return None

                choice = asyncio.create_task(read_choice())
                expiry = asyncio.wrap_future(offer.answer)
                try:
                    await asyncio.wait((choice, expiry, reader), return_when=asyncio.FIRST_COMPLETED)
                    if choice.done() and not reader.done() and not offer.answer.done():
                        with suppress(InvalidStateError):
                            offer.answer.set_result(choice.result())
                    return not reader.done() or offer.answer.done()
                finally:
                    choice.cancel()
                    await asyncio.gather(choice, return_exceptions=True)
                    if not expiry.done() and not reader.done():
                        expiry.cancel()

    @staticmethod
    async def cancel_recap(job):
        if job:
            job.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await job

    @staticmethod
    async def wait(wake, timeout):
        if timeout is None:
            await wake.wait()
        else:
            with suppress(TimeoutError):
                await asyncio.wait_for(wake.wait(), timeout)
