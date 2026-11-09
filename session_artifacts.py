"""Transfer the artifact directory that accompanies a session journal."""

import os
import re
import shutil
import stat
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from shake import artifact_id


ARTIFACT_URL = re.compile(r"artifact://([a-f0-9]{64}|[0-9]+)(?:\?offset=([0-9]{1,12}))?")
PIPY_ARTIFACT = re.compile(r"([a-f0-9]{64})\.pipy\.log")


def sidecar(journal):
    name = journal.name
    for suffix in (".jsonl.gz", ".jsonl"):
        if name.lower().endswith(suffix) and len(name) > len(suffix):
            return journal.with_name(name[:-len(suffix)])
    return journal.with_name(name + ".artifacts")


def file_mode(path):
    try:
        info = path.lstat()
    except FileNotFoundError:
        return 0
    # Windows directory junctions can look like ordinary directories to is_symlink().
    if getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
        return 0
    return info.st_mode


def copy_files(source, target, *, skip=None):
    for entry in source.iterdir():
        if skip and skip(entry.name):
            continue
        destination = target / entry.name
        mode = file_mode(entry)
        if stat.S_ISDIR(mode):
            destination.mkdir(mode=0o700)
            copy_files(entry, destination)
        elif stat.S_ISREG(mode):
            with entry.open("rb") as reader, destination.open("xb") as writer:
                shutil.copyfileobj(reader, writer)


def artifact_files(directory):
    if stat.S_ISDIR(file_mode(directory)):
        for path in sorted(directory.iterdir()):
            mode = file_mode(path)
            if stat.S_ISDIR(mode):
                yield from artifact_files(path)
            elif stat.S_ISREG(mode):
                yield path


@contextmanager
def copy_sidecar(source, target, *, create=False, skip=None):
    source_exists = stat.S_ISDIR(file_mode(source))
    if not source_exists and not create:
        yield
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(mode=0o700)
    try:
        if source_exists:
            copy_files(source, target, skip=skip)
        yield
    except BaseException:
        shutil.rmtree(target, ignore_errors=True)
        raise


def stored_artifacts(directory):
    if not stat.S_ISDIR(file_mode(directory)):
        return
    for path in directory.iterdir():
        if (match := PIPY_ARTIFACT.fullmatch(path.name)) and stat.S_ISREG(file_mode(path)):
            content = path.read_bytes().decode("utf-8")
            if artifact_id(content) != match[1]:
                raise ValueError(f"Artifact content does not match its id: {path.name}")
            yield match[1], content


@contextmanager
def install_sidecar(source, target, *, replace=False):
    if not replace:
        with copy_sidecar(source, target, skip=PIPY_ARTIFACT.fullmatch):
            yield
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=target.parent, prefix=".session-unpack-") as temporary:
        incoming = Path(temporary) / "incoming"
        backup = incoming.with_name("backup")
        with copy_sidecar(source, incoming, skip=PIPY_ARTIFACT.fullmatch):
            if os.path.lexists(target):
                target.rename(backup)
            try:
                if incoming.exists():
                    incoming.rename(target)
                yield
            except BaseException:
                if target.exists():
                    target.rename(incoming)
                if os.path.lexists(backup):
                    backup.rename(target)
                raise


def read_numbered(directory, identifier, offset, limit):
    if not identifier.isascii() or not identifier.isdecimal() or not stat.S_ISDIR(file_mode(directory)):
        return None
    prefix = identifier + "."
    path = next((path for path in sorted(directory.iterdir())
                 if path.name.startswith(prefix) and stat.S_ISREG(file_mode(path))), None)
    if path is None:
        return None
    page, length = [], 0
    with path.open(encoding="utf-8", errors="replace", newline="") as reader:
        while chunk := reader.read(65536):
            end = length + len(chunk)
            if offset < end and offset + limit > length:
                page.append(chunk[max(0, offset - length):offset + limit - length])
            length = end
    return "".join(page), length
