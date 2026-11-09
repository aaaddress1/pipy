"""The OS ownership leases shared with Oh My Pi session journals."""

import errno
import os
import re
import socket
import stat
import sys
from contextlib import contextmanager
from pathlib import Path

import xxhash
from filelock import Timeout


SESSION_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")


def lock_path(identifier):
    if not SESSION_ID.fullmatch(identifier):
        raise ValueError("Invalid session lease id.")
    owners = os.getenv("PI_TEST_SESSION_OWNERS_DIR")
    if not owners:
        # Node's path.join appends PI_CONFIG_DIR to the home directory even when it starts with a separator.
        base = Path(os.path.normpath(str(Path.home()) + os.sep + (os.getenv("PI_CONFIG_DIR") or ".omp")))
        if (sys.platform.startswith("linux") or sys.platform == "darwin") and (xdg := os.getenv("XDG_STATE_HOME")):
            migrated = Path(xdg) / "omp"
            if migrated.exists():
                base = migrated
        owners = base / "run" / "session-owners"
    return Path(os.path.abspath(owners)) / (identifier + ".lock")


def lock_name(path):
    data = str(path).encode("utf-8")
    high = xxhash.xxh64_hexdigest(data, seed=0x4F4D502D4C4F434B)
    low = xxhash.xxh64_hexdigest(data, seed=0x50492D46494C454C)
    return f"omp-file-lock-{high}{low}"


@contextmanager
def windows_lease(path):
    import ctypes as c
    from ctypes import wintypes as w

    kernel = c.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateMutexExW
    create.argtypes = (c.c_void_p, w.LPCWSTR, w.DWORD, w.DWORD)
    create.restype = w.HANDLE
    close = kernel.CloseHandle
    close.argtypes, close.restype = (w.HANDLE,), w.BOOL
    c.set_last_error(0)
    handle = create(None, "Global\\" + lock_name(path), 0, 0x00100000)
    error = c.get_last_error()
    if not handle:
        raise c.WinError(error)
    try:
        if error == 183:  # ERROR_ALREADY_EXISTS. The handle's lifetime owns the name.
            raise Timeout(str(path))
        yield
    finally:
        close(handle)


def windows_held(path):
    import ctypes as c
    from ctypes import wintypes as w

    class UnicodeString(c.Structure):
        _fields_ = [("length", w.USHORT), ("capacity", w.USHORT), ("buffer", c.c_void_p)]

    class ObjectAttributes(c.Structure):
        _fields_ = [("length", w.ULONG), ("root", w.HANDLE), ("name", c.POINTER(UnicodeString)),
                    ("attributes", w.ULONG), ("security", c.c_void_p), ("quality", c.c_void_p)]

    class DirectoryEntry(c.Structure):
        _fields_ = [("name", UnicodeString), ("kind", UnicodeString)]

    ntdll = c.WinDLL("ntdll")
    open_directory = ntdll.NtOpenDirectoryObject
    open_directory.argtypes = (c.POINTER(w.HANDLE), w.DWORD, c.POINTER(ObjectAttributes))
    open_directory.restype = w.LONG
    query = ntdll.NtQueryDirectoryObject
    query.argtypes = (w.HANDLE, c.c_void_p, w.ULONG, c.c_ubyte, c.c_ubyte, c.POINTER(w.ULONG), c.POINTER(w.ULONG))
    query.restype = w.LONG
    kernel = c.WinDLL("kernel32")
    close = kernel.CloseHandle
    close.argtypes, close.restype = (w.HANDLE,), w.BOOL

    directory = c.create_unicode_buffer(r"\BaseNamedObjects")
    size = c.sizeof(directory)
    name = UnicodeString(size - c.sizeof(c.c_wchar), size, c.addressof(directory))
    attrs = ObjectAttributes(c.sizeof(ObjectAttributes), None, c.pointer(name), 0x40, None, None)
    handle = w.HANDLE()
    if (status := open_directory(c.byref(handle), 1, c.byref(attrs))) < 0:
        raise OSError(f"Cannot list Windows session leases (NTSTATUS {status & 0xffffffff:#x}).")
    wanted = lock_name(path)
    try:
        size, restart, context = 65536, True, w.ULONG()
        buffer = (c.c_uint64 * (size // 8))()
        while True:
            written = w.ULONG()
            status = query(handle, buffer, size, False, restart, c.byref(context), c.byref(written)) & 0xffffffff
            if status == 0x8000001a:  # STATUS_NO_MORE_ENTRIES
                return False
            if restart and status in (0x105, 0xc0000023) and size < 4 * 1024**2:
                size *= 2
                buffer = (c.c_uint64 * (size // 8))()
                continue
            if status not in (0, 0x105):
                raise OSError(f"Cannot query Windows session leases (NTSTATUS {status:#x}).")
            restart = False
            count = min(size, written.value) // c.sizeof(DirectoryEntry)
            for entry in (DirectoryEntry * count).from_buffer(buffer):
                if not entry.name.buffer:
                    break
                if c.wstring_at(entry.name.buffer, entry.name.length // c.sizeof(c.c_wchar)) == wanted:
                    return True
            if status == 0:
                return False
    finally:
        close(handle)


def private_directory(path, *, create=False):
    if create:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise OSError(f"The session ownership directory must be private: {path}")


@contextmanager
def upstream_lease(identifier):
    path = lock_path(identifier)
    if sys.platform == "win32":
        with windows_lease(path):
            yield
        return
    if sys.platform.startswith("linux"):
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as owner:
            try:
                owner.bind("\0" + lock_name(path))
            except OSError as error:
                if error.errno == errno.EADDRINUSE:
                    raise Timeout(str(path)) from None
                raise
            yield
        return
    import fcntl

    private_directory(path.parent, create=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "rb+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise Timeout(str(path)) from None
        yield


def upstream_held(identifier):
    path = lock_path(identifier)
    if sys.platform == "win32":
        return windows_held(path)
    if sys.platform.startswith("linux"):
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as probe:
            try:
                probe.connect("\0" + lock_name(path))
            except ConnectionRefusedError:
                return False
            return True
    if sys.platform == "darwin":
        import fcntl
        import struct

        try:
            private_directory(path.parent)
            with path.open("rb") as lock:
                layout = struct.Struct("@qqihh")
                query = layout.pack(0, 0, 0, fcntl.F_RDLCK, os.SEEK_SET)
                return layout.unpack(fcntl.fcntl(lock, fcntl.F_GETLK, query))[3] != fcntl.F_UNLCK
        except FileNotFoundError:
            return False
    return None
