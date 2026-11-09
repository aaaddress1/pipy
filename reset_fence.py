"""The account reset fence shared with Oh My Pi's native lock."""

import asyncio
import errno
import os
import socket
import sys
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from time import monotonic, sleep

from xxhash import xxh64_intdigest


MASK64 = (1 << 64) - 1
WYHASH_SECRET = (0xA0761D6478BD642F, 0xE7037ED1A0B428DB, 0x8EBC6AF09C88C6E3, 0x589965CC75374CC3)


def mix(left, right):
    product = left * right
    return (product & MASK64) ^ (product >> 64)


def bun_hash(value):
    """Wyhash final4 with the default seed used by Bun.hash."""
    data = value.encode("utf-8")
    size = len(data)
    k0, k1, k2, k3 = WYHASH_SECRET
    state = mix(k0, k1)

    def read(offset, count=8):
        return int.from_bytes(data[offset:offset + count], "little")

    if size >= 4 and size <= 16:
        quarter = (size >> 3) << 2
        left = read(0, 4) << 32 | read(quarter, 4)
        right = read(size - 4, 4) << 32 | read(size - 4 - quarter, 4)
    elif size < 4:
        left = data[0] << 16 | data[size // 2] << 8 | data[-1] if size else 0
        right = 0
    else:
        second = third = state
        offset = 0
        while offset + 48 < size:
            state = mix(read(offset) ^ k1, read(offset + 8) ^ state)
            second = mix(read(offset + 16) ^ k2, read(offset + 24) ^ second)
            third = mix(read(offset + 32) ^ k3, read(offset + 40) ^ third)
            offset += 48
        state ^= second ^ third
        while offset + 16 < size:
            state = mix(read(offset) ^ k1, read(offset + 8) ^ state)
            offset += 16
        left, right = read(size - 16), read(size - 8)
    product = (left ^ k1) * (right ^ state)
    return mix((product & MASK64) ^ k0 ^ size, (product >> 64) ^ k1)


def marker_path(database, account):
    return Path(f"{database}.reset-{bun_hash(account):x}")


def lock_name(path):
    encoded = str(path).encode("utf-8")
    high = xxh64_intdigest(encoded, seed=0x4F4D502D4C4F434B)
    low = xxh64_intdigest(encoded, seed=0x50492D46494C454C)
    return f"omp-file-lock-{high:016x}{low:016x}"


def try_lock(path):
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        create = kernel.CreateMutexExW
        create.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD)
        create.restype = wintypes.HANDLE
        close = kernel.CloseHandle
        close.argtypes, close.restype = (wintypes.HANDLE,), wintypes.BOOL
        ctypes.set_last_error(0)
        handle = create(None, "Global\\" + lock_name(path), 0, 0x00100000)
        error = ctypes.get_last_error()
        if not handle:
            raise ctypes.WinError(error)
        if error == 183:
            close(handle)
            return None
        return lambda: close(handle)
    if sys.platform == "linux":
        handle = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        try:
            handle.bind(b"\0" + lock_name(path).encode("ascii"))
        except OSError as error:
            handle.close()
            if error.errno == errno.EADDRINUSE:
                return None
            raise
        return handle.close

    import fcntl

    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        os.close(descriptor)
        if error.errno in (errno.EAGAIN, errno.EACCES):
            return None
        raise
    return lambda: os.close(descriptor)


@contextmanager
def shared_file_lock(path, timeout=0):
    deadline = monotonic() + timeout
    while True:
        if release := try_lock(path):
            try:
                yield
            finally:
                release()
            return
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Lock unavailable: {path}")
        sleep(min(.05, remaining))


@asynccontextmanager
async def reset_lock(marker, stop):
    path = Path(os.path.abspath(f"{marker}.lock"))
    deadline = monotonic() + 30
    while not stop.is_set() and monotonic() < deadline:
        if release := try_lock(path):
            try:
                yield True
            finally:
                release()
            return
        await asyncio.sleep(.1)
    yield False


def read_marker(path):
    try:
        state, _, stamp = path.read_text(encoding="utf-8").partition(":")
        return state, int(stamp)
    except (FileNotFoundError, ValueError, UnicodeError):
        return "", 0


def write_marker(path, state="", started=0):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as file:
        if state:
            file.write(f"{state}:{started}")
        file.flush()
        os.fsync(file.fileno())
