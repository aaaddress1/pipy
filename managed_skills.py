"""An isolated store for skills explicitly captured by the agent."""

import re

import yaml
from filelock import FileLock

from memory_files import atomic_text, neutral_line, read_utf8


NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,63}\Z")
MAX_BYTES = 64_000


def skill_metadata(file):
    try:
        raw = read_utf8(file)
        if not raw.startswith("---\n"):
            return None
        header, separator, _ = raw[4:].partition("\n---\n")
        if not separator:
            return None
        values = yaml.safe_load(header)
    except (OSError, UnicodeError, yaml.YAMLError):
        return None
    if not isinstance(values, dict):
        return None
    name, description = values.get("name"), values.get("description")
    if not isinstance(name, str) or not NAME.fullmatch(name) or not isinstance(description, str):
        return None
    return name, neutral_line(description)


class ManagedSkills:
    def __init__(self, agent_dir, workspace, redact):
        self.root = agent_dir / "managed-skills"
        self.authored = (workspace / ".omp" / "skills", workspace / ".agents" / "skills",
                         workspace / ".claude" / "skills", agent_dir / "skills")
        self.redact = redact
        self.catalog = None

    def entries(self, roots):
        for root in roots:
            for file in sorted(root.glob("*/SKILL.md")):
                try:
                    contained = file.resolve().is_relative_to(root.resolve())
                except (OSError, RuntimeError):
                    continue
                if not contained:
                    continue
                if metadata := skill_metadata(file):
                    name, description = metadata
                    yield name, description, file

    def discovered(self):
        if self.catalog is not None:
            return self.catalog
        found = {}
        for name, description, file in self.entries((*self.authored, self.root)):
            found.setdefault(name, (self.redact(description), file))
        self.catalog = found
        return found

    def instructions(self):
        found = self.discovered()
        if not found:
            return ""
        return "Available skills (read the corresponding SKILL.md before using one):\n" + "\n".join(
            f"- skill://{name}: {description}" for name, (description, _) in found.items()
        )

    def read(self, uri):
        name = uri.removeprefix("skill://").removesuffix("/SKILL.md")
        if not NAME.fullmatch(name):
            raise ValueError("Use skill://<name> to read a skill.")
        if not (found := self.discovered().get(name)):
            raise ValueError(f"Unknown skill {name!r}.")
        return self.redact(read_utf8(found[1]))

    def mutate(self, args):
        if not isinstance(args, dict) or set(args) - {"action", "name", "description", "body"}:
            raise ValueError("Skill mutation expects action, name, and optional description and body.")
        action, raw = args.get("action"), args.get("name")
        if not isinstance(raw, str) or not NAME.fullmatch(name := raw.strip().lower()):
            raise ValueError("Use a skill name of 1–64 letters, digits, or hyphens.")
        if action not in ("create", "update", "delete"):
            raise ValueError("Skill action must be create, update, or delete.")
        content = None
        if action != "delete":
            description, body = args.get("description"), args.get("body")
            if not isinstance(description, str) or not isinstance(body, str):
                raise ValueError(f"{action} requires both description and body.")
            description, body = self.redact(neutral_line(description)), self.redact(body.strip())
            if not description or not body:
                raise ValueError("A managed skill requires a description and body.")
            metadata = yaml.safe_dump({"name": name, "description": description}, sort_keys=False, allow_unicode=True)
            content = f"---\n{metadata}---\n\n{body}\n"
            if len(content.encode("utf-8")) > MAX_BYTES:
                raise ValueError("Managed skill exceeds the 64,000-byte limit.")
        if action == "create" and any(authored == name for authored, _, _ in self.entries(self.authored)):
            raise ValueError(f"An authored skill named {name!r} already exists.")
        self.root.mkdir(parents=True, exist_ok=True)
        root = self.root.resolve()
        directory, file = root / name, root / name / "SKILL.md"
        with FileLock(root / (".mini-pi-" + name + ".lock")):
            if directory.resolve() != directory or file.resolve() != file:
                raise ValueError("A managed skill cannot be changed through a symlink.")
            exists = file.is_file()
            if action == "create":
                if file.exists():
                    raise ValueError(f"Managed skill {name!r} already exists. Use update.")
                directory.mkdir(exist_ok=True)
            elif not exists:
                raise ValueError(f"Managed skill {name!r} does not exist.")
            if action == "delete":
                file.unlink()
                try:
                    directory.rmdir()
                except OSError:
                    pass
            else:
                if exists and file.stat().st_nlink > 1:
                    raise ValueError("Cannot overwrite a hard-linked managed skill.")
                atomic_text(file, content)
        self.catalog = None
        verb = {"create": "Created", "update": "Updated", "delete": "Deleted"}[action]
        return f"{verb} managed skill {name!r}.", name
