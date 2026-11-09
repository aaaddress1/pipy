"""Discover and normalize Oh My Pi rule files."""

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import unquote

import yaml
from wcmatch import glob

from project_context import project_directories, settings_layers


BUILTINS = Path(__file__).with_name("prompts") / "ttsr"
GLOB_FLAGS = glob.GLOBSTAR | glob.BRACE | glob.DOTMATCH | glob.FORCEUNIX
MODES = ("always", "never", "prose-only", "tool-only")
SETTING_KEYS = ("enabled", "judge", "contextMode", "interruptMode", "repeatMode",
                "repeatGap", "builtinRules", "disabledRules")


def values(raw):
    raw = [raw] if isinstance(raw, str) else raw if isinstance(raw, list) else []
    return tuple(dict.fromkeys(item.strip() for item in raw if isinstance(item, str) and item.strip()))


def split_scopes(raw):
    tokens = []
    for value in values(raw):
        start, level, quote, previous = 0, 0, None, ""
        for index, char in enumerate(value):
            if quote:
                if char == quote and previous != "\\":
                    quote = None
            elif char in "\"'":
                quote = char
            elif char in "([{":
                level += 1
            elif char in ")]}":
                level = max(0, level - 1)
            elif char == "," and not level:
                tokens.append(value[start:index])
                start = index + 1
            previous = char
        tokens.append(value[start:])
    return values([token.strip().strip("\"'") for token in tokens])


def parse_yaml(source):
    try:
        value = yaml.safe_load(source)
        return value if isinstance(value, dict) else {}
    except yaml.YAMLError:
        result = {}
        for key, value in re.findall(r"^([\w-]+):[ \t]*(.*)$", source, re.M):
            try:
                result[key] = yaml.safe_load(value)
            except yaml.YAMLError:
                result[key] = value.strip()
        return result


def fields(raw):
    if not raw.startswith("---\n"):
        return {}, raw.strip()
    front, separator, body = raw[4:].partition("\n---")
    if not separator or body and not body.startswith("\n"):
        return {}, raw.strip()
    parsed = parse_yaml(front)
    normalized = {}
    for key, value in parsed.items():
        if isinstance(key, str):
            normalized[re.sub(r"-([a-z])", lambda part: part[1].upper(), key)] = value
    return normalized, body.strip()


def looks_like_glob(value):
    return (not re.search(r"[\\^$+|()]", value) and any(char in value for char in "?*[]{}")
            and ("/" in value or re.fullmatch(r"\*\.[^\s/]+", value) is not None))


@lru_cache(maxsize=1024)
def compile_glob(pattern):
    try:
        return glob.compile(pattern, flags=GLOB_FLAGS)
    except (TypeError, ValueError):
        return None


def glob_matches(pattern, path):
    path = path.replace("\\", "/")
    compiled = compile_glob(pattern)
    return compiled is not None and any(compiled.match(candidate) for candidate in (path, path.rsplit("/", 1)[-1]))


@dataclass(frozen=True)
class Rule:
    name: str
    path: Path
    content: str
    source: str
    description: str
    always: bool
    conditions: tuple
    ast: tuple
    question: str
    scopes: tuple
    paths: tuple
    agents: tuple
    mode: str | None

    def display_path(self, workspace):
        return self.path.relative_to(workspace).as_posix() if self.path.is_relative_to(workspace) else self.path.as_posix()


def load_rule(path, source, name=None, sticky=False, *, explicit=False):
    try:
        if path.stat().st_size > 4 * 1024 * 1024:
            return None
        raw = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError):
        return None
    return parse_rule(raw, path, source, name, sticky, explicit=explicit)


def parse_rule(raw, path, source, name=None, sticky=False, *, explicit=False):
    meta, content = fields(raw)
    if not explicit and meta.get("enabled") is False:
        return None
    conditions, inferred = [], []
    for token in values(meta.get("condition", meta.get("ttsr_trigger", meta.get("ttsrTrigger")))):
        if looks_like_glob(token):
            inferred.extend(f"tool:{tool}({token})" for tool in ("edit", "write"))
        else:
            conditions.append(token)
    if inferred and not conditions:
        conditions.append(".*")
    paths = values(meta.get("globs"))
    always = sticky or meta.get("alwaysApply") is True
    description = meta.get("description") if isinstance(meta.get("description"), str) else ""
    if source == "github" and not sticky:
        paths = split_scopes(meta.get("applyTo"))
        always = any(pattern in ("*", "**", "**/*") for pattern in paths)
        if always:
            paths = ()
        elif not description:
            description = "Instructions for " + ", ".join(paths) if paths else "GitHub Copilot instructions"
    mode = meta.get("interruptMode")
    question = meta.get("question")
    return Rule(name or path.stem, path, content, source, description, always, tuple(conditions),
                values(meta.get("astCondition")), question.strip() if isinstance(question, str) else "",
                (*split_scopes(meta.get("scope")), *inferred), paths,
                tuple(re.sub(r"\s*,\s*", ",", pattern.lower()) for pattern in split_scopes(meta.get("agents"))),
                mode if mode in MODES else None)


def rule_files(directory, patterns=("*.md", "*.mdc"), *, recursive=False):
    scan = directory.rglob if recursive else directory.glob
    try:
        return sorted({path for pattern in patterns for path in scan(pattern) if path.is_file()})
    except OSError:
        return []


def extension_roots(workspace, user, paths):
    for entry in values(paths):
        if "://" in entry or entry.startswith(("npm:", "git:")):
            continue
        path = Path(os.path.expandvars(entry.removeprefix("file:"))).expanduser()
        candidates = (path,) if path.is_absolute() else (workspace / path, user / path)
        for candidate in candidates:
            if candidate.is_dir() or candidate.is_file():
                yield candidate if candidate.is_dir() else candidate.parent
                break


def settings_for(workspace, defaults):
    ttsr, plugins = {}, []
    for layer in (defaults, *settings_layers(workspace)):
        group = layer.get("ttsr")
        for key in SETTING_KEYS:
            if isinstance(group, dict) and key in group:
                ttsr[key] = group[key]
            elif "ttsr." + key in layer:
                ttsr[key] = layer["ttsr." + key]
        plugins.extend(values(layer.get("extensions")))
    return ttsr, list(dict.fromkeys(plugins))


def discover(workspace, user, plugin_roots=()):
    project = project_directories(workspace)
    home = Path.home()
    candidates = []

    def add(source, directory, patterns=("*.md", "*.mdc"), recursive=False):
        candidates.extend((path, source, None, False) for path in rule_files(directory, patterns, recursive=recursive))

    add("native", workspace / ".omp" / "rules")
    add("native", user / "rules")
    candidates.append((user / "RULES.md", "native", "RULES", True))
    for parent in project:
        native = parent / ".omp"
        try:
            if native.is_dir() and next(native.iterdir(), None) is not None:
                candidates.append((native / "RULES.md", "native", "RULES@project", True))
                break
        except OSError:
            continue
    for directory in plugin_roots:
        add("omp-plugins", directory / "rules")
    for parent in (*project, home):
        for directory in (".agent", ".agents"):
            add("agents", parent / directory / "rules")
    add("cursor", home / ".cursor" / "rules")
    add("cursor", workspace / ".cursor" / "rules")
    candidates.append((home / ".codeium" / "windsurf" / "memories" / "global_rules.md", "windsurf", "global_rules", False))
    add("windsurf", workspace / ".windsurf" / "rules", ("*.md",))
    for parent in project:
        cline = parent / ".clinerules"
        if cline.is_dir():
            add("cline", cline, ("*.md",))
            break
        if cline.is_file():
            candidates.append((cline, "cline", "clinerules", False))
            break
    copilot_dirs = [Path(item.strip()).expanduser() for item in os.getenv("COPILOT_CUSTOM_INSTRUCTIONS_DIRS", "").split(",")
                    if item.strip()]
    for parent in (workspace, *copilot_dirs):
        candidates.extend((path, "github", path.name.removesuffix(".instructions.md"), False)
                          for path in rule_files(parent / ".github" / "instructions", ("*.instructions.md",), recursive=True))
    add("builtin-defaults", BUILTINS, ("*.md",))
    found = {}
    for candidate in candidates:
        if rule := load_rule(*candidate):
            found.setdefault(rule.name, rule)
    return list(found.values())


def read_rule(uri, rules):
    name = unquote(uri.removeprefix("rule://"))
    if not name:
        return "\n".join(f"rule://{rule.name} — {rule.description}" for rule in rules.values()) or "No rules."
    if name not in rules:
        raise ValueError(f"Unknown rule {name!r}.")
    return rules[name].content
