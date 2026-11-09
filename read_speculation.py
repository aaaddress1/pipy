"""Discardable, digest-checked prefetch for finalized local read calls."""

import json
import math
import os
import stat
from concurrent.futures import CancelledError, ThreadPoolExecutor, wait
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from threading import Lock

from auth_broker import dotted
from handoff import Cancellation


if os.name == "nt":
    import ctypes
    import msvcrt
    from ctypes import wintypes

    final_path = ctypes.WinDLL("kernel32", use_last_error=True).GetFinalPathNameByHandleW
    final_path.argtypes = (wintypes.HANDLE, wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD)
    final_path.restype = wintypes.DWORD


MAX_BYTES = 4 * 1024 * 1024
SPECIAL = frozenset(
    ".ipynb .cpuprofile .pdf .docx .pptx .xlsx .epub .svg .svgz .mp4 .mov .mkv .webm .avi .m4v .wmv".split()
)


def configured_limit(layers):
    settings = {}
    keys = ("tools.speculativeExecution.enabled", "tools.speculativeExecution.maxInFlight",
            "tools.approvalMode", "tools.approval.read", "tools.restrictedFolders")
    for layer in layers:
        for key in keys:
            if (value := dotted(layer, key)) is not None:
                settings[key] = value
    if settings.get(keys[0]) is not True or settings.get(keys[4]):
        return 0
    approval = settings.get(keys[3])
    if approval is not None and approval != "allow":
        return 0
    if approval is None and settings.get(keys[2], "yolo") not in ("yolo", "write", "always-ask"):
        return 0
    limit = settings.get(keys[1], 2)
    if type(limit) is int:
        return max(1, limit)
    return max(1, math.floor(limit)) if type(limit) is float and math.isfinite(limit) else 2


def read_path(name, raw):
    if name != "read":
        return None
    try:
        args = json.loads(raw)
    except (TypeError, ValueError, RecursionError):
        return None
    return local_read_path(args)


def local_read_path(args):
    if not isinstance(args, dict) or set(args) != {"path"}:
        return None
    path = args["path"]
    if not isinstance(path, str) or not path or path.startswith("www.") or any(c in path for c in ":?#\0"):
        return None
    return path


def special(path):
    return path.suffix.lower() in SPECIAL or path.name.lower().endswith(".sample.txt")


def target_for(workspace, path):
    try:
        original = workspace / path
        lexical = Path(os.path.abspath(original))
        if lexical == workspace or not lexical.is_relative_to(workspace) or lexical.is_reserved() or special(lexical):
            return None
        target = original.resolve(strict=True)
        if target == workspace or not target.is_relative_to(workspace) or special(target):
            return None
        info = target.stat(follow_symlinks=False)
        if not regular(info):
            return None
        return target
    except (OSError, ValueError, RuntimeError):
        return None


def regular(info):
    return (stat.S_ISREG(info.st_mode) and info.st_size <= MAX_BYTES and info.st_ino
            and not getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def open_target(workspace, root_identity, target, flags):
    if os.name == "nt":
        fd = os.open(target, flags)
        try:
            buffer = ctypes.create_unicode_buffer(32_768)
            size = final_path(msvcrt.get_osfhandle(fd), buffer, len(buffer), 0)
            if not 0 < size < len(buffer):
                raise OSError("Could not verify the speculative read handle.")
            actual = buffer.value
            actual = "\\\\" + actual[8:] if actual.startswith("\\\\?\\UNC\\") else actual.removeprefix("\\\\?\\")
            if os.path.normcase(actual) != os.path.normcase(str(target)):
                raise OSError("Speculative read target changed before opening.")
            return fd
        except BaseException:
            os.close(fd)
            raise

    if os.open not in os.supports_dir_fd or not hasattr(os, "O_NOFOLLOW"):
        raise OSError("Safe directory-relative opens are unavailable.")
    parts = target.relative_to(workspace).parts
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory = os.open(workspace, directory_flags)
    try:
        if identity(os.fstat(directory))[:2] != root_identity:
            raise OSError("Workspace changed before opening the speculative read.")
        for part in parts[:-1]:
            next_directory = os.open(part, directory_flags, dir_fd=directory)
            os.close(directory)
            directory = next_directory
        return os.open(parts[-1], flags, dir_fd=directory)
    finally:
        os.close(directory)


def stable_bytes(workspace, root_identity, target, signal):
    signal.check()
    initial = target.stat(follow_symlinks=False)
    if not regular(initial):
        return None
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
    with os.fdopen(open_target(workspace, root_identity, target, flags), "rb") as file:
        before = os.fstat(file.fileno())
        if not regular(before) or identity(initial) != identity(before):
            return None
        data = bytearray()
        while len(data) <= MAX_BYTES:
            signal.check()
            if not (chunk := file.read(min(65_536, MAX_BYTES + 1 - len(data)))):
                break
            data.extend(chunk)
        if len(data) > MAX_BYTES or identity(before) != identity(os.fstat(file.fileno())):
            return None
        if identity(before) != identity(target.stat(follow_symlinks=False)):
            return None
    signal.check()
    return data


def prefetched(workspace, root_identity, path, target, limit, signal):
    try:
        if target_for(workspace, path) != target:
            return None
        data = stable_bytes(workspace, root_identity, target, signal)
        if data is None or b"\0" in data[:8192] or data.startswith((b"%PDF-", b"GIF87a", b"GIF89a")):
            return None
        if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
            return None
        text = data.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        return sha256(data).digest(), text[:limit + 1]
    except (CancelledError, OSError, UnicodeError, ValueError):
        return None


@dataclass(frozen=True)
class Candidate:
    identifier: str
    path: str
    target: Path
    future: object


class ReadSpeculation:
    def __init__(self, workspace, workers, limit):
        self.workspace, self.limit = workspace, limit
        self.root_identity = identity(workspace.stat())[:2]
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mini-pi-read")
        self.lock = Lock()
        self.signal = None
        self.slots = {}
        self.processed = set()
        self.admitted = []
        self.sealed = False
        self.barrier = False
        self.claims = {}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.discard()
        self.executor.shutdown(wait=True, cancel_futures=True)

    def begin(self, parent):
        self.discard()
        self.signal = Cancellation(parent, reason="Speculative read discarded.")
        self.slots.clear()
        self.sealed = self.barrier = False

    def restart(self):
        self.begin(self.signal.parent)

    def observe(self, index, name, identifier, raw, *, final=False):
        if self.barrier or self.sealed:
            return
        if type(index) is not int or index < 0:
            self.barrier = True
            return
        if index in self.slots and self.slots[index][0]:
            return
        self.slots[index] = (final, name, identifier, raw)
        for slot, (closed, candidate_name, candidate_id, arguments) in sorted(self.slots.items()):
            if not closed:
                break
            if slot in self.processed:
                continue
            self.processed.add(slot)
            path = read_path(candidate_name, arguments) if candidate_id else None
            if path is None or any(item.identifier == candidate_id for item in self.admitted):
                self.barrier = True
                break
            candidate = self.submit(candidate_id, path, self.signal)
            if candidate is None:
                self.barrier = True
                break
            self.admitted.append(candidate)

    def submit(self, identifier, path, signal):
        signal.check()
        target = target_for(self.workspace, path)
        if target is None:
            return None
        future = self.executor.submit(prefetched, self.workspace, self.root_identity, path, target, self.limit, signal)
        return Candidate(identifier, path, target, future)

    def seal(self, calls):
        self.sealed = True
        for candidate, call in zip(self.admitted, calls):
            function = call["function"]
            if candidate.identifier != call["id"] or candidate.path != read_path(function["name"], function["arguments"]):
                break
            self.claims[candidate.identifier] = candidate

    def take(self, call):
        with self.lock:
            candidate = self.claims.pop(call["id"], None)
        if candidate is None:
            return None
        function = call["function"]
        if read_path(function["name"], function["arguments"]) != candidate.path:
            return None
        return self.verify(candidate, self.signal)

    def verify(self, candidate, signal, *, cancel=None, expired=None):
        try:
            signal.check()
            if cancel is None:
                snapshot = candidate.future.result()
            else:
                while True:
                    cancel.check()
                    if expired():
                        return None
                    try:
                        snapshot = candidate.future.result(timeout=.05)
                        break
                    except TimeoutError:
                        if candidate.future.done():
                            return None
            if snapshot is None or target_for(self.workspace, candidate.path) != candidate.target:
                return None
            data = stable_bytes(self.workspace, self.root_identity, candidate.target, cancel or signal)
            if data is None or sha256(data).digest() != snapshot[0]:
                return None
            if target_for(self.workspace, candidate.path) == candidate.target:
                return snapshot[1]
        except (OSError, ValueError):
            pass
        return None

    def cancel(self):
        if self.signal:
            self.signal.cancel()

    def discard(self):
        self.cancel()
        pending = [candidate.future for candidate in self.admitted]
        for future in pending:
            future.cancel()
        wait(pending)
        self.admitted.clear()
        self.claims.clear()
        self.processed.clear()
