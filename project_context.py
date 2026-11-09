"""Project settings and instruction files."""

import json
from pathlib import Path

import yaml


def settings_layers(workspace):
    for name, parse in (("settings.json", json.loads), ("config.yml", yaml.safe_load)):
        path = workspace / ".omp" / name
        try:
            data = parse(path.read_text(encoding="utf-8-sig"))
        except FileNotFoundError:
            continue
        except (ValueError, yaml.YAMLError) as error:
            raise ValueError(f"Invalid project settings in {path}: {error}") from None
        if data is not None and not isinstance(data, dict):
            raise ValueError(f"Project settings in {path} must be an object.")
        yield data or {}


def project_directories(workspace):
    home = Path.home().resolve()
    lineage = [workspace, *workspace.parents]
    root = next((item for item in lineage if (item / ".git").exists()), None)
    if workspace.is_relative_to(home):
        return lineage[:lineage.index(home) + (0 if root and root != home and root.is_relative_to(home) else 1)]
    return lineage[:lineage.index(root) + 1] if root else [workspace]


def load(workspace, *, include_rules=True):
    directories = project_directories(workspace)
    files = []
    seen = set()

    def read(path):
        try:
            target = path.resolve()
            if target in seen or not any(target.is_relative_to(parent) for parent in directories):
                return ""
            content = target.read_text(encoding="utf-8-sig").strip()
        except (OSError, UnicodeError):
            return ""
        seen.add(target)
        return content

    native = None
    for directory in directories:
        candidate = directory / ".omp"
        try:
            if not any(candidate.iterdir()):
                continue
        except OSError:
            continue
        native = candidate
        break

    for directory in reversed(directories):
        paths = []
        if native and native.parent == directory:
            paths.append(native / "AGENTS.md")
        if directory == workspace:
            paths.append(directory / ".claude" / "CLAUDE.md")
        paths.extend(directory / folder / "AGENTS.md" for folder in (".agent", ".agents"))
        if directory == workspace:
            paths.extend((
                directory / ".gemini" / "GEMINI.md",
                directory / ".github" / "copilot-instructions.md",
            ))
        if not directory.name.startswith("."):
            paths.extend(directory / name for name in ("AGENTS.md", "CLAUDE.md"))
        for path in paths:
            if content := read(path):
                files.append(f"### {path}\n\n{content}")
                break

    rules = read(native / "RULES.md") if native and include_rules else ""
    sections = []
    if files:
        sections.append("## Context\n<instructions>\n" + "\n\n".join(files) + "\n</instructions>")
    if rules:
        sections.append("<sticky-rules>\n" + rules + "\n</sticky-rules>")
    return "\n\n".join(sections)
