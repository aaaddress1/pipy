"""Oh My Pi external shell selection for supervised services."""

import os
import shutil
import subprocess
from pathlib import Path


GIT_LOCATION = {"GIT_DIR", "GIT_COMMON_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE",
                "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES"}
UNSAFE = GIT_LOCATION | {"MallocStackLogging", "MallocStackLoggingNoCompact"}


def default_shell(env):
    path = env.get("PATH")
    if os.name != "nt":
        user = env.get("SHELL")
        if user and ("bash" in user or "zsh" in user) and os.path.isfile(user) and os.access(user, os.X_OK):
            return user
        return shutil.which("bash", path=path) or shutil.which("sh", path=path) or "/bin/sh"
    roots = [(env.get("PROGRAMFILES"), "Git"), (env.get("PROGRAMFILES(X86)"), "Git"),
             (env.get("LOCALAPPDATA"), "Programs", "Git"), (env.get("GIT_INSTALL_ROOT"),),
             (env.get("SCOOP"), "apps", "git", "current"), (env.get("USERPROFILE"), "scoop", "apps", "git", "current")]
    for base, *parts in roots:
        if base and (candidate := Path(base, *parts, "bin", "bash.exe")).is_file():
            return str(candidate)
    if bash := shutil.which("bash.exe", path=path):
        return bash
    if sh := shutil.which("sh.exe", path=path):
        sibling = Path(sh).with_name("bash.exe")
        return str(sibling) if sibling.is_file() else sh
    return env.get("COMSPEC") or str(Path(env.get("SYSTEMROOT", r"C:\Windows")) / "System32" / "cmd.exe")


def shell_plan(command, configured, env):
    shell = configured or default_shell(env)
    if configured and not Path(shell).is_file():
        raise ValueError(f"Custom shell path does not exist: {shell}. Update shellPath in the active settings.")
    no_login = env.get("PI_BASH_NO_LOGIN") or env.get("CLAUDE_BASH_NO_LOGIN")
    name = Path(shell).stem.lower()
    if name == "cmd":
        flags = ["/d", "/s", "/c"]
    elif name in ("powershell", "pwsh"):
        flags = ["-NoLogo", "-NoProfile", "-Command"] if no_login else ["-NoLogo", "-Command"]
    else:
        flags = ["-c"] if no_login else ["-l", "-c"]
    prefix = env.get("PI_SHELL_PREFIX") or env.get("CLAUDE_CODE_SHELL_PREFIX")
    return [shell, *flags, f"{prefix} {command}" if prefix else command]


def shell_environment(argv, source):
    excluded = {key.upper() for key in UNSAFE} if os.name == "nt" else UNSAFE
    env = {key: value for key, value in source.items()
           if (key.upper() if os.name == "nt" else key) not in excluded}
    env.update(SHELL=argv[0], GIT_EDITOR="true", GPG_TTY="not a tty", OMPCODE="1", CLAUDECODE="1")
    if not (env.get("PI_BASH_NO_CI") or env.get("CLAUDE_BASH_NO_CI")):
        env["CI"] = "true"
    return env


def windows_arguments(argv):
    if Path(argv[0]).stem.lower() == "cmd":
        return f'{subprocess.list2cmdline(argv[1:-1])} "{argv[-1]}"'
    return subprocess.list2cmdline(argv[1:])
