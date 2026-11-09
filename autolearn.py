"""Coalesce private capture turns without changing the primary transcript."""

import logging
from threading import Condition

from handoff import Cancellation, deadline


PROMPT = (
    "This is an automatic memory-capture turn. Look only at the work just completed. "
    "Record a reusable, verified lesson with learn, or a repeatable procedure with manage_skill. "
    "If there is nothing worth retaining, stop. Do not resume the user's task or use other tools."
)
logger = logging.getLogger(__name__)


class CaptureQueue:
    def __init__(self, workers, parent):
        self.workers = workers
        self.signal = Cancellation(parent, reason="Auto-learn capture cancelled.")
        self.condition = Condition()
        self.current = self.pending = None
        self.closed = False

    def submit(self, capture):
        with self.condition:
            if self.closed or self.signal.is_cancelled():
                return
            if self.current:
                self.pending = capture
            else:
                self.start(capture)

    def start(self, capture):
        def run():
            with deadline(60, self.signal) as signal:
                capture(signal)

        self.current = self.workers.submit(run)
        self.current.add_done_callback(self.settle)

    def settle(self, future):
        try:
            future.result()
        except Exception:
            if not self.signal.is_cancelled():
                logger.debug("Auto-learn capture failed", exc_info=True)
        with self.condition:
            self.current = None
            pending, self.pending = self.pending, None
            if pending and not self.signal.is_cancelled():
                self.start(pending)
            self.condition.notify_all()

    def close(self, *, wait):
        with self.condition:
            self.closed = True
            if wait:
                self.condition.wait_for(lambda: self.current is None)
            else:
                self.pending = None
                self.signal.cancel()
