"""Bounded log snapshots and byte-cursor following for service output."""

import codecs
import os
import re
from contextlib import ExitStack, contextmanager

from service_runtime import TERMINAL_ESCAPE, previous_log


READ_LIMIT = 2 * 1024 * 1024
CHUNK_LIMIT = 256 * 1024
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def clean(text):
    text = TERMINAL_ESCAPE.sub("", text).replace("\r\n", "\n")
    return CONTROL.sub("", text)


@contextmanager
def log_files(path):
    with ExitStack() as stack:
        files = {}
        # Capture the active inode before the previous path. A rotation between
        # opens can expose the active inode twice; only read it once.
        for name in (path, previous_log(path)):
            try:
                stream = stack.enter_context(name.open("rb"))
            except FileNotFoundError:
                continue
            info = os.fstat(stream.fileno())
            identity = (info.st_dev, info.st_ino)
            files.setdefault(identity, (stream, info.st_size))
        yield list(reversed(list(files.items())))


def read_window(files, head):
    remaining = READ_LIMIT
    chunks = []
    for _, (stream, size) in files if head else reversed(files):
        amount = min(remaining, size)
        if amount:
            stream.seek(0 if head else size - amount)
            chunks.append(stream.read(amount))
            remaining -= amount
        if not remaining:
            break
    return b"".join(chunks if head else reversed(chunks))


def select_lines(text, count, head, pattern):
    lines = clean(text).splitlines()
    if pattern:
        lines = [line for line in lines if pattern.search(line)]
    return lines[:count] if head else lines[-count:]


def log_lines(path, count, *, head=False, pattern=None):
    with log_files(path) as files:
        return select_lines(read_window(files, head).decode("utf-8", errors="replace"), count, head, pattern)


class LogFollower:
    def __init__(self, pattern):
        self.pattern = pattern
        self.offsets = {}
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.pending = ""

    def start(self, path, count, head):
        with log_files(path) as files:
            self.offsets = {key: size for key, (_, size) in files}
            data = read_window(files, head)
        text = data.decode("utf-8", errors="replace") if head else self.decoder.decode(data)
        if not head and text and not text.endswith("\n"):
            text, _, self.pending = text.rpartition("\n")
            self.pending = self.pending[-CHUNK_LIMIT:]
        return select_lines(text, count, head, self.pattern)

    def poll(self, path):
        parts = []
        offsets = {}
        remaining = CHUNK_LIMIT
        with log_files(path) as files:
            for key, (stream, size) in files:
                offset = self.offsets.get(key, 0)
                if offset > size:
                    offset = 0
                stream.seek(offset)
                data = stream.read(min(remaining, size - offset))
                offsets[key] = offset + len(data)
                remaining -= len(data)
                if data:
                    parts.append(self.decoder.decode(data))
        self.offsets = offsets
        text = self.pending + "".join(parts)
        lines = text.split("\n")
        self.pending = lines.pop()[-CHUNK_LIMIT:]
        return select_lines("\n".join(lines) + ("\n" if lines else ""), len(lines), True, self.pattern), not remaining

    def finish(self):
        text = self.pending + self.decoder.decode(b"", final=True)
        self.pending = ""
        return select_lines(text, 1, True, self.pattern)
