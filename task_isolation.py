"""Private Git checkouts and recovery artifacts for isolated tasks."""

import os
import shutil
import stat
import subprocess
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from sqlite3 import Error as SQLiteError
from tempfile import gettempdir, mkdtemp
from uuid import uuid4

from filelock import FileLock


class GitError(RuntimeError):
    pass


ROUTING_ENV = {"GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
               "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_NAMESPACE", "GIT_PREFIX", "GIT_CONFIG_COUNT",
               "GIT_CONFIG_PARAMETERS", "GIT_SHALLOW_FILE", "GIT_GRAFT_FILE"}
IDENTITY = {"GIT_AUTHOR_NAME": "Mini Pi", "GIT_AUTHOR_EMAIL": "mini-pi@localhost",
            "GIT_COMMITTER_NAME": "Mini Pi", "GIT_COMMITTER_EMAIL": "mini-pi@localhost"}
DIFF = ("--binary", "--full-index", "--no-color", "--no-ext-diff", "--no-textconv", "--no-renames")


def checkout_environment():
    return {key: value for key, value in os.environ.items() if key not in ROUTING_ENV}


def environment(extra=None):
    return {**checkout_environment(),
            "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0", **(extra or {})}


def command(cwd, *args):
    return ["git", "-C", str(cwd), "-c", "core.hooksPath=" + os.devnull, *map(str, args)]


def git(cwd, *args, data=None, index=None, extra=None, check=True):
    extra = {**(extra or {}), **({"GIT_INDEX_FILE": str(index)} if index else {})}
    source = {"stdin": subprocess.DEVNULL} if data is None else {"input": data}
    result = subprocess.run(command(cwd, *args), capture_output=True, env=environment(extra), timeout=180, **source)
    if result.returncode and check:
        error = result.stderr.decode("utf-8", errors="replace").strip()
        raise GitError(f"git {args[0]} failed: {error[-1800:] or result.returncode}")
    return result


def out(cwd, *args, **kwargs):
    return git(cwd, *args, **kwargs).stdout


def ref(cwd, name="HEAD"):
    return out(cwd, "rev-parse", "--verify", name, check=False).decode("ascii", errors="replace").strip()


def repository(cwd):
    return Path(os.fsdecode(out(cwd, "rev-parse", "--show-toplevel").rstrip(b"\r\n"))).resolve()


def git_paths(raw):
    return [os.fsdecode(part) for part in raw.split(b"\0") if part]


def common_dir(repo):
    return Path(os.fsdecode(out(repo, "rev-parse", "--path-format=absolute", "--git-common-dir").rstrip(b"\r\n"))).resolve()


def linked(path):
    info = path.lstat()
    return path.is_symlink() or bool(getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def remove_tree(path, owner):
    if path.resolve() != path or not path.is_relative_to(owner) or linked(path):
        raise OSError(f"Cleanup path no longer belongs to the task checkout: {path}")

    def retry(function, filename, failure):
        target = Path(filename)
        if (os.name != "nt" or not isinstance(failure[1], PermissionError) or target.resolve() != target
                or not target.is_relative_to(owner) or linked(target) or target.stat().st_nlink > 1):
            raise failure[1]
        target.chmod(stat.S_IREAD | stat.S_IWRITE)
        function(filename)

    shutil.rmtree(path, onerror=retry)


def nested_repos(root, ignored):
    parent_common = common_dir(root)
    result = []
    for directory, dirs, _ in os.walk(root, followlinks=False):
        parent = Path(directory)
        for name in list(dirs):
            path = parent / name
            if name in (".git", "node_modules") or path in ignored or linked(path):
                dirs.remove(name)
                continue
            if (path / ".git").exists():
                dirs.remove(name)
                if repository(path) == path and common_dir(path) != parent_common:
                    result.append(path.relative_to(root))
    return sorted(result)


def clone(source, target, head):
    git(source, "clone", "--shared", "--no-checkout", "--quiet", "--", source, target)
    git(target, "remote", "remove", "origin")
    for key in ("core.autocrlf", "core.eol", "core.filemode", "core.symlinks", "user.name", "user.email"):
        value = out(source, "config", "--get", key, check=False).rstrip(b"\r\n")
        if value:
            git(target, "config", key, os.fsdecode(value))
    exclude_file = common_dir(source) / "info" / "exclude"
    if exclude_file.is_file():
        shutil.copyfile(exclude_file, target / ".git" / "info" / "exclude")
    if head:
        git(target, "checkout", "--detach", "--quiet", head)


def excluded(path, directories):
    return any(path == directory or path.is_relative_to(directory) for directory in directories)


def seed(source, target, skip):
    pathspec = [f":(exclude,top,literal){path.as_posix()}" for path in skip]
    for flags, cached in ((["--cached"], ["--index"]), ([], [])):
        patch = out(source, "diff", *DIFF, *flags, "--ignore-submodules=all", "--", ".", *pathspec)
        if patch:
            git(target, "apply", "--whitespace=nowarn", *cached, data=patch)
    for value in git_paths(out(source, "ls-files", "--others", "--exclude-standard", "-z")):
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts or excluded(relative, skip):
            continue
        src, dst = source / relative, target / relative
        if not src.parent.resolve().is_relative_to(source) or not dst.parent.resolve().is_relative_to(target):
            raise GitError(f"Untracked path leaves the checkout: {relative}")
        if src.is_symlink() or src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst, follow_symlinks=False)


def tree(cwd, index, skip):
    staged = out(cwd, "write-tree").decode().strip()
    git(cwd, "read-tree", staged, index=index)
    pathspec = [f":(exclude,top,literal){path.as_posix()}" for path in skip]
    git(cwd, "add", "-A", "--", ".", *pathspec, index=index)
    return out(cwd, "write-tree", index=index).decode().strip()


def delta(cwd, before, after):
    return out(cwd, "diff", *DIFF, before, after, "--") if before != after else b""


@dataclass
class Snapshot:
    source: Path
    checkout: Path
    relative: Path
    head: str
    baseline: str
    index: Path
    skip: tuple

    def current(self):
        return tree(self.checkout, self.index, self.skip)

    def clean_tree(self):
        return ref(self.checkout, self.head + "^{tree}") if self.head else out(self.checkout, "mktree", data=b"").decode().strip()


def commit_tree(cwd, result, parent, message, author=None):
    args = ["commit-tree", result, *(["-p", parent] if parent else []), "-F", "-"]
    return out(cwd, *args, data=message, extra={**IDENTITY, **(author or {})}).decode().strip()


def changed(cwd, before, after):
    return set(git_paths(out(cwd, "diff", "--name-only", "--no-renames", "-z", before, after, "--")))


def filtered_tree(snapshot, clean, current):
    if snapshot.baseline == clean:
        return current
    repo, index = snapshot.checkout, snapshot.index
    patch = delta(repo, snapshot.baseline, current)
    if not patch:
        return clean
    for flags in ([], ["--3way"]):
        git(repo, "read-tree", clean, index=index)
        applied = git(repo, "apply", "--cached", "--whitespace=nowarn", *flags, data=patch, index=index, check=False)
        if applied.returncode == 0:
            return out(repo, "write-tree", index=index).decode().strip()
    # Keep the task's postimage on paths it changed and drop unrelated baseline edits.
    unrelated = changed(repo, clean, snapshot.baseline) - changed(repo, snapshot.baseline, current)
    git(repo, "read-tree", current, index=index)
    if unrelated:
        paths = b"\0".join(os.fsencode(path) for path in sorted(unrelated)) + b"\0"
        git(repo, "--literal-pathspecs", "reset", "-q", clean, "--pathspec-from-file=-", "--pathspec-file-nul",
            index=index, data=paths)
    return out(repo, "write-tree", index=index).decode().strip()


def task_commits(snapshot, current, name):
    repo = snapshot.checkout
    clean, head = snapshot.clean_tree(), ref(repo)
    base = snapshot.head or commit_tree(repo, clean, None, b"Empty task baseline\n")
    tip, previous = base, clean
    if head and snapshot.head and git(repo, "merge-base", "--is-ancestor", base, head, check=False).returncode == 0:
        if snapshot.baseline == clean:
            tip, previous = head, ref(repo, head + "^{tree}")
        else:
            commits = out(repo, "rev-list", "--reverse", "--first-parent", f"{base}..{head}").decode().splitlines()
            for commit in commits:
                result = filtered_tree(snapshot, clean, ref(repo, commit + "^{tree}"))
                if result == previous:
                    continue
                raw = out(repo, "show", "-s", "--format=%an%x00%ae%x00%aI%x00%B", commit)
                author, email, date, message = raw.split(b"\0", 3)
                tip = commit_tree(repo, result, tip, message, dict(zip(
                    ("GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_AUTHOR_DATE"), map(os.fsdecode, (author, email, date)),
                )))
                previous = result
    final = filtered_tree(snapshot, clean, current)
    if final != previous:
        tip = commit_tree(repo, final, tip, f"Changes from isolated task {name}\n".encode())
    return base, tip


def apply_patch(repo, patch):
    if git(repo, "apply", "--check", "--whitespace=nowarn", data=patch, check=False).returncode == 0:
        git(repo, "apply", "--whitespace=nowarn", data=patch)
        return "applied"
    if git(repo, "apply", "--reverse", "--check", "--whitespace=nowarn", data=patch, check=False).returncode == 0:
        return "already-applied"
    raise GitError("The patch conflicts with the current working tree.")


def install_commit(repo, before, after, patch):
    git(repo, "apply", "--index", "--check", "--whitespace=nowarn", data=patch)
    with subprocess.Popen(command(repo, "update-ref", "--stdin"), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, env=environment()) as update:
        update.stdin.write(f"start\nupdate HEAD {after} {before}\nprepare\n".encode())
        update.stdin.flush()
        started, prepared = update.stdout.readline(), update.stdout.readline()
        if started.strip() != b"start: ok" or prepared.strip() != b"prepare: ok":
            update.stdin.close()
            raise GitError("Parent HEAD changed or is locked; the branch was retained.")
        installed = False
        try:
            git(repo, "apply", "--index", "--whitespace=nowarn", data=patch)
            installed = True
            update.stdin.write(b"commit\n")
            update.stdin.flush()
            if update.stdout.readline().strip() != b"commit: ok":
                raise GitError("Parent HEAD could not be updated; the branch was retained.")
        except BaseException:
            update.stdin.close()
            if installed and ref(repo) == before:
                git(repo, "apply", "--reverse", "--index", "--whitespace=nowarn", data=patch)
            raise
        finally:
            update.stdin.close()


class Workspace:
    def __init__(self, root, cwd, storage, session, task, backend, mode, apply):
        self.root, self.cwd, self.storage, self.session, self.task = root, cwd, storage, session, task
        self.backend, self.mode, self.apply = backend, mode, apply
        self.base = Path(mkdtemp(prefix="mini-pi-isolation-")).resolve()
        self.checkout = self.base / "checkout"
        self.repos = []
        self.retain = False
        self.handed_off = None

    @property
    def directory(self):
        return self.checkout / self.cwd.relative_to(self.root)

    def open(self, cancel):
        skip_state = {self.storage} if self.storage.is_relative_to(self.root) and self.storage != self.root else set()
        nested = nested_repos(self.root, skip_state)
        for relative in [Path(), *nested]:
            cancel.check()
            source, target = self.root / relative, self.checkout / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                remove_tree(target, self.base)
            head = ref(source)
            clone(source, target, head)
            if repository(target) != target:
                raise GitError("The isolated Git checkout has an unexpected worktree location.")
            skip = tuple(nested) if relative == Path() else ()
            if self.storage.is_relative_to(source):
                if self.storage != source:
                    skip += (self.storage.relative_to(source),)
                else:
                    skip += tuple(Path(name) for name in ("sessions.sqlite3", "sessions.sqlite3-wal", "sessions.sqlite3-shm",
                                                          "artifacts", "locks", "isolation-locks"))
            seed(source, target, skip)
            index = self.base / ("index-" + str(len(self.repos)))
            self.repos.append(Snapshot(source, target, relative, head, tree(target, index, skip), index, skip))
        child_cwd = self.directory
        child_cwd.mkdir(parents=True, exist_ok=True)
        return child_cwd

    def lock(self, source):
        directory = self.storage / "isolation-locks"
        directory.mkdir(parents=True, exist_ok=True)
        name = sha256(os.fsencode(os.path.normcase(str(common_dir(source))))).hexdigest() + ".lock"
        return FileLock(directory / name, timeout=60)

    def save(self, directory, snapshot, patch):
        if snapshot.relative == Path():
            filename = f"{self.task.name}.patch"
        else:
            identifier = sha256(os.fsencode(snapshot.relative)).hexdigest()[:10]
            filename = f"{self.task.name}.nested-{identifier}.patch"
        path = directory / filename
        path.write_bytes(patch)
        result = {"repo": snapshot.relative.as_posix(), "patchPath": str(path)}
        try:
            result["artifact"] = "artifact://" + self.session.save_artifact(patch.decode("utf-8"))
        except (UnicodeDecodeError, SQLiteError):
            pass
        return result

    def branch(self, snapshot, current, entry, preserve):
        repo = snapshot.source
        base, tip = task_commits(snapshot, current, self.task.name)
        if base == tip:
            return "already-applied"
        with self.lock(repo):
            if repository(repo) != repo:
                raise GitError("The parent repository moved during the task.")
            name = "omp/task/" + self.task.name
            if ref(repo, "refs/heads/" + name):
                name += "-" + uuid4().hex[:8]
            branch = "refs/heads/" + name
            temporary_ref = "refs/mini-pi/isolation-result"
            git(snapshot.checkout, "update-ref", temporary_ref, tip)
            git(repo, "fetch", "--quiet", "--no-tags", "--no-recurse-submodules", "--no-write-fetch-head",
                "--", snapshot.checkout, f"{temporary_ref}:{branch}")
            entry.update(branchName=name, branchBaseSha=base)
            if preserve:
                return "preserved"
            head = ref(repo)
            if not head:
                raise GitError("The parent branch has no commit yet; the task branch was retained.")
            replay = self.base / ("replay-" + uuid4().hex[:8])
            clone(repo, replay, head)
            commits = out(repo, "rev-list", "--reverse", "--first-parent", f"{base}..{branch}").decode().splitlines()
            git(replay, "cherry-pick", "--empty=drop", *commits, extra=IDENTITY)
            result = ref(replay)
            if result != head:
                patch = delta(replay, head, result)
                git(repo, "fetch", "--quiet", "--no-tags", "--no-recurse-submodules", "--no-write-fetch-head", "--", replay, "HEAD")
                if patch:
                    install_commit(repo, head, result, patch)
                else:
                    git(repo, "update-ref", "HEAD", result, head)
            git(repo, "update-ref", "-d", branch, tip, check=False)
            return "applied" if result != head else "already-applied"

    def finish(self, succeeded, *, release=False):
        result = {"backend": "git-clone", "merge": self.mode, "patches": []}
        if self.backend not in ("auto", "rcopy"):
            result["requestedBackend"] = self.backend
        trees, captures = [], []
        directory = self.storage / "artifacts" / self.session.id / "isolation" / (self.task.name + "-" + self.task.sid[:8])
        try:
            existing = {snapshot.relative for snapshot in self.repos[1:]}
            new_repos = set(nested_repos(self.checkout, set())) - existing
            if new_repos:
                names = ", ".join(path.as_posix() for path in sorted(new_repos))
                raise GitError(f"New nested Git repositories require manual recovery: {names}")
            for snapshot in self.repos:
                if (snapshot.checkout.resolve() != snapshot.checkout or linked(snapshot.checkout / ".git")
                        or repository(snapshot.checkout) != snapshot.checkout):
                    raise GitError("The task's private checkout changed identity.")
                current = snapshot.current()
                trees.append((snapshot, current))
            fingerprint = tuple(current for _, current in trees)
            if release and fingerprint == self.handed_off:
                return None
            if succeeded and not release:
                self.handed_off = fingerprint
            for snapshot, current in trees:
                if patch := delta(snapshot.checkout, snapshot.baseline, current):
                    directory.mkdir(parents=True, exist_ok=True)
                    entry = self.save(directory, snapshot, patch)
                    result["patches"].append(entry)
                    captures.append((snapshot, current, patch, entry))
        except (GitError, OSError, subprocess.SubprocessError) as error:
            self.retain = True
            result.update(error=str(error), retainedWorkspace=str(self.checkout))
            return result
        root_conflict = False
        for snapshot, current, patch, entry in captures:
            try:
                if release:
                    entry["status"] = self.branch(snapshot, current, entry, preserve=True)
                elif not succeeded or root_conflict:
                    entry["status"] = "preserved"
                elif snapshot.relative == Path() and self.mode == "branch":
                    entry["status"] = self.branch(snapshot, current, entry, preserve=not self.apply)
                elif not self.apply:
                    entry["status"] = "preserved"
                elif snapshot.relative != Path():
                    entry["status"] = self.branch(snapshot, current, entry, preserve=False)
                else:
                    with self.lock(snapshot.source):
                        if repository(snapshot.source) != snapshot.source:
                            raise GitError("The parent repository moved during the task.")
                        entry["status"] = apply_patch(snapshot.source, patch)
            except (GitError, OSError, subprocess.SubprocessError, TimeoutError) as error:
                entry.update(status="conflict", error=str(error))
                root_conflict = snapshot.relative == Path()
        return result

    def close(self):
        if self.retain:
            return
        if self.base.resolve() == self.base and self.base.parent == Path(gettempdir()).resolve() and self.base.name.startswith("mini-pi-isolation-"):
            remove_tree(self.base, self.base)
        else:
            raise OSError("The task's temporary directory changed identity.")


def report(result):
    text = []
    for patch in result["patches"]:
        location = patch.get("artifact") or patch["patchPath"]
        text.append(f"Isolation {patch['repo']}: {patch.get('status', 'captured')}. Patch: {location}")
        if patch.get("branchName") and patch.get("status") not in ("applied", "already-applied"):
            text.append("Recovery branch: " + patch["branchName"])
        if patch.get("error"):
            text.append(patch["error"])
    if result.get("error"):
        text.append("Isolation capture failed. Workspace retained at " + result["retainedWorkspace"] + "\n" + result["error"])
    elif result.get("cleanupError"):
        text.append("Isolation cleanup failed at " + result["retainedWorkspace"] + "\n" + result["cleanupError"])
    return "\n".join(text) or "Isolation: no changes."
