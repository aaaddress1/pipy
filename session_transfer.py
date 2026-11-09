"""Move a session journal, its artifacts, and its referenced blobs as a version 1 tar."""

import argparse
import gzip
import io
import json
import os
import re
import shutil
import sys
import tarfile
from pathlib import Path, PureWindowsPath
from sqlite3 import Error as SQLiteError
from tempfile import NamedTemporaryFile, TemporaryDirectory
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from filelock import FileLock, Timeout

from runtime import Session, database, initialize, pack
from session_artifacts import artifact_files, sidecar
from session_blobs import BLOB_HASH, blob_path, hydrate_images, references, sha256_file
from session_jsonl import export_journal, import_journal, import_message, journal_entries, ordered_rows, read_journal


ARCHIVE_VERSION = 1
ARCHIVE_LIMIT = 2 * 1024**3
SESSION_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")
WINDOWS_SPECIAL = re.compile(r'[<>:"\\|?*\x00-\x1f]')


def local_id(identifier):
    try:
        return str(UUID(identifier))
    except ValueError:
        return str(uuid5(NAMESPACE_URL, "oh-my-pi:session:" + identifier))


def lease(state, identifier):
    directory = state / "locks"
    directory.mkdir(parents=True, exist_ok=True)
    return FileLock(directory / (identifier + ".lock"), timeout=0)


def signature(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def add_file(archive, name, path):
    with path.open("rb") as source:
        stat = os.fstat(source.fileno())
        member = tarfile.TarInfo(name)
        member.size, member.mtime, member.mode = stat.st_size, int(stat.st_mtime), 0o600
        archive.addfile(member, source)
        if signature(os.fstat(source.fileno())) != signature(stat):
            raise ValueError(f"Session file changed while packing: {path}")


def pack_journal(source, output, blob_dirs):
    source = source.resolve(strict=True)
    if output == source:
        raise ValueError("The archive path must differ from the source journal.")
    before = signature(source.stat())
    header, _, _ = read_journal(source)
    identifier, cwd = header.get("id"), header.get("cwd")
    if not isinstance(identifier, str) or not SESSION_ID.fullmatch(identifier) or not isinstance(cwd, str):
        raise ValueError("The session journal must have a valid id and working directory.")
    manifest = (pack({"version": ARCHIVE_VERSION, "id": identifier, "cwd": cwd, "omp": "pipy"}) + "\n").encode()
    directory = sidecar(source)
    artifacts = list(artifact_files(directory))
    if output in artifacts:
        raise ValueError("The archive path must differ from the source artifacts.")
    missing = []
    with TemporaryDirectory(prefix="pipy-pack-") as temporary:
        journal = source
        with source.open("rb") as probe:
            compressed = probe.read(2) == b"\x1f\x8b"
        if compressed:
            journal = Path(temporary) / "session.jsonl"
            with gzip.open(source, "rb") as reader, journal.open("wb") as writer:
                shutil.copyfileobj(reader, writer)
        hashes = references(journal)
        for artifact in artifacts:
            if artifact.suffix == ".jsonl":
                hashes.update(references(artifact))
        with NamedTemporaryFile(dir=output.parent, delete=False) as temporary_archive:
            staged = Path(temporary_archive.name)
        try:
            with tarfile.open(staged, "w", format=tarfile.PAX_FORMAT) as archive:
                member = tarfile.TarInfo("manifest.json")
                member.size, member.mode = len(manifest), 0o600
                archive.addfile(member, io.BytesIO(manifest))
                add_file(archive, "session.jsonl", journal)
                for artifact in artifacts:
                    name = "artifacts/" + artifact.relative_to(directory).as_posix()
                    validate_name(name)
                    add_file(archive, name, artifact)
                for digest in sorted(hashes):
                    path = blob_path(blob_dirs, digest)
                    if path is None:
                        missing.append(digest)
                        continue
                    if path.resolve() == output:
                        raise ValueError("The archive path must differ from the source blobs.")
                    add_file(archive, "blobs/" + digest, path)
            if signature(source.stat()) != before:
                raise ValueError("The source journal changed while packing. Close its writer and retry.")
            if staged.stat().st_size > ARCHIVE_LIMIT:
                raise ValueError("The session archive exceeds 2 GiB.")
            os.replace(staged, output)
        finally:
            staged.unlink(missing_ok=True)
    return missing


def pack_session(state, source, output, blob_dirs):
    try:
        identifier = str(UUID(source))
    except ValueError:
        return pack_journal(Path(source).expanduser(), output, blob_dirs)
    path = state / "sessions.sqlite3"
    if not path.is_file():
        raise ValueError(f"Session {identifier} was not found.")
    if output == path or state / "artifacts" in output.parents or state / "blobs" in output.parents:
        raise ValueError("Choose an archive path outside the session database and artifact store.")
    with lease(state, identifier), TemporaryDirectory(prefix="pipy-session-pack-") as temporary:
        initialize(path)
        with database(path) as db:
            row = db.execute("""SELECT s.workspace, json_extract(i.header, '$.id') FROM sessions s
                                LEFT JOIN session_imports i ON i.session = s.id WHERE s.id = ?""",
                             (identifier,)).fetchone()
        if row is None:
            raise ValueError(f"Session {identifier} was not found.")
        workspace, origin = row
        header_id = origin if isinstance(origin, str) and SESSION_ID.fullmatch(origin) and local_id(origin) == identifier else None
        session = Session(path, identifier, workspace)
        journal = Path(temporary) / "session.jsonl"
        export_journal(path, identifier, workspace, session.leaf, journal, header_id=header_id)
        return pack_journal(journal, output, blob_dirs)


def validate_name(name):
    parts = name.split("/")
    if any(not part or part in (".", "..") or part.endswith((" ", ".")) or WINDOWS_SPECIAL.search(part)
           or PureWindowsPath(part).is_reserved() for part in parts):
        raise ValueError(f"Invalid session archive member {name!r}.")
    return parts


def stage_archive(archive, staging):
    if not archive.is_file() or archive.stat().st_size > ARCHIVE_LIMIT:
        raise ValueError("A session archive must be a tar file of at most 2 GiB.")
    names, paths, total = set(), {}, 0
    with tarfile.open(archive, "r:") as tar:
        for member in tar:
            name = member.name
            parts = validate_name(name)
            blob = len(parts) == 2 and parts[0] == "blobs" and BLOB_HASH.fullmatch(parts[1])
            artifact = len(parts) > 1 and parts[0] == "artifacts"
            total += member.size
            if (member.type not in (tarfile.REGTYPE, tarfile.AREGTYPE) or member.sparse is not None or member.size < 0
                or total > ARCHIVE_LIMIT or not (name in ("session.jsonl", "manifest.json") or artifact or blob)
                or name in names or name == "manifest.json" and member.size > 65536):
                raise ValueError(f"Unexpected session archive member {name!r}.")
            for end in range(1, len(parts) + 1):
                prefix = "/".join(parts[:end])
                if paths.setdefault(prefix.casefold(), prefix) != prefix:
                    raise ValueError(f"Conflicting session archive member {name!r}.")
            names.add(name)
            target = staging.joinpath(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, target.open("xb") as destination:
                shutil.copyfileobj(source, destination)
            if blob:
                with target.open("rb") as file:
                    if sha256_file(file) != parts[1]:
                        raise ValueError(f"Blob {parts[1]} fails its SHA-256 check.")
    if not {"manifest.json", "session.jsonl"} <= names:
        raise ValueError("A session archive requires manifest.json and session.jsonl.")
    manifest = json.loads((staging / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or type(manifest.get("version")) is not int or manifest["version"] != ARCHIVE_VERSION:
        raise ValueError("Unsupported session archive manifest version.")
    if not all(isinstance(manifest.get(key), str) for key in ("id", "cwd", "omp")) or not SESSION_ID.fullmatch(manifest["id"]):
        raise ValueError("Invalid session archive manifest.")
    journal = staging / "session.jsonl"
    header, rows, _ = read_journal(journal)
    if header.get("id") != manifest["id"]:
        raise ValueError("The archive manifest id does not match the session journal.")
    ordered_rows(rows)
    return journal, header, rows, manifest


def store_blobs(source, target):
    for path in artifact_files(source):
        if blob_path((target,), path.name):
            continue
        target.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=target, delete=False) as temporary:
            staged = Path(temporary.name)
        try:
            shutil.copyfile(path, staged)
            os.replace(staged, target / path.name)
        finally:
            staged.unlink(missing_ok=True)


def active_entries(rows):
    return [row for row in rows if row.get("type") not in ("title", "session")
            and row.get("customType") != "mini-pi.active-leaf"]


def comparable(entry):
    if entry.get("type") == "message":
        return entry.get("id"), entry.get("parentId"), import_message(entry)
    return {key: value for key, value in entry.items() if key not in ("timestamp", "activeLeafId")}


def prefix_matches(local, incoming):
    pending = iter(incoming)
    next_entry = next(pending, None)
    for entry in local:
        if next_entry is not None and comparable(entry) == comparable(next_entry):
            next_entry = next(pending, None)
        elif not (entry.get("customType") == "session-transfer" and isinstance(entry.get("details"), dict)
                  and entry["details"].get("kind") == "departed"):
            return False
    return True


def unpack_mode(path, identifier, incoming):
    with database(path) as db:
        db.execute("BEGIN")
        row = db.execute("""SELECT s.workspace,h.leaf FROM sessions s LEFT JOIN session_heads h ON h.session=s.id
                            WHERE s.id=?""", (identifier,)).fetchone()
        if row is None:
            return "created"
        local = active_entries(journal_entries(db, identifier, *row))
        return "replaced" if prefix_matches(local, incoming) else "forked"


def unpack_session(state, archive, workspace):
    if not workspace.is_dir():
        raise ValueError(f"Working directory does not exist: {workspace}")
    with TemporaryDirectory(prefix="pipy-unpack-") as temporary:
        staging = Path(temporary)
        journal, header, rows, manifest = stage_archive(archive, staging)
        old_cwd = header.get("cwd") or manifest["cwd"]
        scope = os.path.normcase(str(workspace))
        notice = {
            "role": "developer",
            "content": f"This session was moved here from another machine. It ran in `{old_cwd}` there; "
                       f"its working directory is now `{workspace}`. Paths from earlier turns under `{old_cwd}` now live "
                       f"under `{workspace}`. Running processes, shell state, and files outside the working directory "
                       "did not move with it.",
            "_session_transfer": {"kind": "arrived", "fromCwd": old_cwd, "cwd": str(workspace)},
        }
        if (staging / "artifacts").is_dir():
            (staging / "artifacts").rename(sidecar(journal))
        state.mkdir(parents=True, exist_ok=True)
        path = state / "sessions.sqlite3"
        initialize(path)
        blobs = state / "blobs"
        store_blobs(staging / "blobs", blobs)
        cache = {}
        for entry in rows:
            hydrate_images(entry, (blobs,), cache)
        incoming = active_entries(rows)
        identifier = local_id(manifest["id"])
        mode = unpack_mode(path, identifier, incoming)
        if mode != "forked":
            with lease(state, identifier):
                mode = unpack_mode(path, identifier, incoming)
                if mode != "forked":
                    import_journal(path, scope, journal, session_id=identifier, replace=mode == "replaced", notice=notice,
                                   preserve_identity=True)
                    return identifier, mode
        identifier = str(uuid4())
        import_journal(path, scope, journal, session_id=identifier, notice=notice)
        return identifier, "forked"


def run(argv, agent_dir):
    parser = argparse.ArgumentParser(prog="mini_pi.py session", description=__doc__)
    parser.add_argument("action", choices=("pack", "unpack"))
    parser.add_argument("source")
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--cwd", type=Path)
    parser.add_argument("--state-dir", type=Path, default=Path.home() / ".mini-pi")
    parser.add_argument("--blobs-dir", type=Path, help="additional upstream blob store for pack")
    args = parser.parse_args(argv)
    state = args.state_dir.expanduser().resolve()
    try:
        if args.action == "pack":
            if args.output is None or args.cwd is not None:
                parser.error("pack requires a source and output archive")
            output = args.output.expanduser().resolve()
            blob_dirs = (state / "blobs", (args.blobs_dir or agent_dir / "blobs").expanduser().resolve())
            if missing := pack_session(state, args.source, output, blob_dirs):
                print(f"warning: {len(missing)} referenced blob(s) are missing from the archive.", file=sys.stderr)
            print(output)
        else:
            if args.cwd is None or args.output is not None or args.blobs_dir is not None:
                parser.error("unpack requires an archive and --cwd")
            identifier, mode = unpack_session(state, Path(args.source).expanduser().resolve(), args.cwd.expanduser().resolve())
            print(f"Unpacked session ({mode}): {identifier}", file=sys.stderr)
            print(identifier)
    except Timeout:
        print("mini-pi: the session is already open", file=sys.stderr)
        return 1
    except (OSError, ValueError, SQLiteError, tarfile.TarError) as error:
        print(f"mini-pi: {error}", file=sys.stderr)
        return 1
    return 0
