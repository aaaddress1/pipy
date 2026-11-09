"""Inspect and exercise stream rules without making model requests."""

import argparse
import json
import os
import sys
from pathlib import Path

from pathspec import GitIgnoreSpec

from ttsr import Rules, applies_to_agent, compile_rule, matching_patterns, tool_name
from ttsr_rules import load_rule, settings_for


SOURCE_EXTENSIONS = frozenset(
    ("ts tsx js jsx mjs cjs rs py go java kt swift c cc cpp h hpp rb php lua css scss html json yaml yml "
     "toml md mdc cs razor cshtml fs fsx vb sh bash sql zig dart scala ex exs proto tf").split()
)
MAX_SCAN_BYTES = 5 * 1024 * 1024


def parse(argv):
    parser = argparse.ArgumentParser(prog="mini_pi.py ttsr", description=__doc__)
    parser.add_argument("action", nargs="?", default="list", choices=("list", "test", "scan"))
    parser.add_argument("snippet", nargs="?", help="inline test snippet, snippet file, or scan directory")
    parser.add_argument("--cwd", type=Path, default=Path.cwd(), help="project directory")
    parser.add_argument("--state-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--file", help="test file or - for stdin")
    parser.add_argument("-r", "--rule", type=Path, help="test or scan a rule file in isolation")
    parser.add_argument("--source", choices=("text", "thinking", "tool"), help="test source")
    parser.add_argument("--tool", help="tool name for --source tool")
    parser.add_argument("-p", "--path", help="candidate path for test globs and AST language")
    parser.add_argument("--agent", default="main", help="agent name for test scope")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--json", action="store_true", help="output JSON")
    parser.add_argument("--no-gitignore", action="store_true")
    parser.add_argument("--max-bytes", type=int, default=MAX_SCAN_BYTES)
    return parser.parse_intermixed_args(argv)


def rule_inventory(rule):
    data = {"name": rule.name, "path": str(rule.path), "provider": rule.source,
            "condition": list(rule.conditions), "astCondition": list(rule.ast), "scope": list(rule.scopes),
            "globs": list(rule.paths), "agents": list(rule.agents)}
    if rule.question:
        data["question"] = rule.question
    if rule.description:
        data["description"] = rule.description
    return data


def load(args, cwd, user, defaults):
    agent = args.agent.strip().lower() or "main"
    if args.rule and args.action in ("test", "scan"):
        path = (cwd / args.rule).resolve()
        rule = load_rule(path, "ttsr-cli", explicit=True)
        if rule is None:
            raise ValueError(f"Rule file could not be read: {path}")
        if args.action == "test" and not applies_to_agent(rule, agent):
            raise ValueError(f'Rule "{rule.name}" is scoped to agents [{", ".join(rule.agents)}] and does not apply to agent "{agent}".')
        item = compile_rule(rule)
        if not item:
            raise ValueError(f'Rule "{rule.name}" has no usable TTSR condition.')
        return (item,)
    settings, plugins = settings_for(cwd, defaults)
    return Rules(cwd, user, settings, agent if args.action == "test" else None, plugins=plugins).triggers


def read_snippet(args, cwd):
    source = args.file
    if source is None and args.snippet:
        try:
            if (cwd / args.snippet).is_file():
                source = args.snippet
        except OSError:
            pass
    if source == "-" or source is None and args.snippet is None and not sys.stdin.isatty():
        content = sys.stdin.read() if sys.stdin.isatty() else sys.stdin.buffer.read().decode("utf-8-sig", errors="replace")
        return content, None
    if source is not None:
        path = (cwd / source).resolve()
        if not path.is_file():
            raise ValueError(f"Snippet file not found: {path}")
        return path.read_bytes().decode("utf-8-sig", errors="replace"), str(path)
    if args.snippet is not None:
        return args.snippet, None
    raise ValueError("No snippet provided. Pass inline text, --file <path>, or pipe via --file -.")


def test_report(args, cwd, items):
    snippet, inferred_path = read_snippet(args, cwd)
    path = args.path if args.path is not None else inferred_path
    ext = Path(path.replace("\\", "/")).suffix.lower() if path else ""
    source = args.source or ("tool" if ext.lstrip(".") in SOURCE_EXTENSIONS else "text")
    tool = args.tool or ("edit" if source == "tool" else None)
    paths = (path,) if path else ()
    ast = source == "tool" and bool(ext)
    preview = " ".join(snippet.split())
    report = {"source": source, "snippetPreview": preview if len(preview) <= 80 else preview[:77] + "…",
              "snippetBytes": len(snippet.encode("utf-16-le")) // 2, "evaluated": len(items),
              "triggered": [], "notTriggered": [], "agent": args.agent.strip().lower() or "main"}
    if tool is not None:
        report["tool"] = tool
    if path is not None:
        report["filePath"] = path
        if args.source is None and source == "text":
            report["inferenceNote"] = (f"inferred --source text from {ext or path!r} (not in the source-file extension set)"
                                       " — pass --source tool --tool edit to evaluate tool-scoped rules")
    for item in items:
        rule = item.rule
        detail = {"name": rule.name, "path": str(rule.path), "sourceProvider": rule.source,
                  "matched": matching_patterns(item, snippet, paths, ast=ast),
                  "defined": {"regex": list(rule.conditions), "ast": list(rule.ast)}}
        if rule.question:
            detail["question"] = rule.question
        if rule.agents:
            detail["agents"] = list(rule.agents)
        if rule.ast and not ast:
            detail["skippedAst"] = "astCondition requires --source tool and a --path with a file extension"
        hit = (not rule.question and item.accepts(source, tool_name(tool or ""), paths)
               and any(detail["matched"].values()))
        report["triggered" if hit else "notTriggered"].append(detail)
    return report


def ignored(path, layers, directory=False):
    result = False
    for base, spec in layers:
        relative = path.relative_to(base).as_posix()
        decision = spec.check_file(relative + ("/" if directory else "")).include
        if decision is not None:
            result = decision
    return result


def ignore_layer(base, file):
    try:
        source = file.read_text(encoding="utf-8-sig", errors="replace")
        return [(base, GitIgnoreSpec.from_lines(source.splitlines()))]
    except OSError:
        return []


def scan_files(root, directory, gitignore):
    initial = []
    if gitignore:
        initial += ignore_layer(root, root / ".git" / "info" / "exclude")
        relative = directory.relative_to(root)
        for depth in range(len(relative.parts)):
            base = root.joinpath(*relative.parts[:depth])
            initial += ignore_layer(base, base / ".gitignore")
            child = root.joinpath(*relative.parts[:depth + 1])
            if ignored(child, initial, directory=True):
                return
    pending = [(directory, initial)]
    while pending:
        base, inherited = pending.pop()
        layers = inherited + ignore_layer(base, base / ".gitignore") if gitignore else inherited
        try:
            children = sorted(base.iterdir())
        except OSError:
            continue
        for path in children:
            if path.name == ".git" or path.is_symlink():
                continue
            if path.is_dir():
                if not ignored(path, layers, directory=True):
                    pending.append((path, layers))
            elif path.is_file() and not ignored(path, layers):
                yield path


def project_path(cwd, path):
    try:
        return Path(os.path.relpath(path, cwd)).as_posix()
    except ValueError:
        return path.as_posix()


def read_scan_file(path, maximum):
    try:
        with path.open("rb") as file:
            if maximum and os.fstat(file.fileno()).st_size > maximum:
                return None, "large"
            prefix = file.read(8192)
            if b"\0" in prefix:
                return None, "binary"
            remainder = file.read(max(1, maximum + 1 - len(prefix))) if maximum else file.read()
            if maximum and len(prefix) + len(remainder) > maximum:
                return None, "large"
            return (prefix + remainder).decode("utf-8", errors="replace"), None
    except OSError:
        return None, "unreadable"


def scan_directory(args, cwd):
    directory = (cwd / args.snippet).resolve() if args.snippet else cwd
    if not directory.is_dir():
        raise ValueError(f"Directory not found: {directory}")
    return directory


def scan_report(args, cwd, directory, items):
    root = cwd if directory.is_relative_to(cwd) else directory
    root = next((base for base in (root, *root.parents) if (base / ".git").exists()), root)
    files = sorted(scan_files(root, directory, not args.no_gitignore))
    limit = max(0, args.max_bytes)
    skipped = dict.fromkeys(("binary", "large", "unreadable", "noRelevantRules"), 0)
    matched, scanned, total = [], 0, 0
    for path in files:
        relative = project_path(cwd, path)
        paths = (path.as_posix(), relative, path.name)
        rules = [item for item in items if item.accepts("tool", "edit", paths) or item.accepts("tool", "write", paths)]
        if not rules:
            skipped["noRelevantRules"] += 1
            continue
        content, reason = read_scan_file(path, limit)
        if reason:
            skipped[reason] += 1
            continue
        scanned += 1
        hits = []
        for item in rules:
            patterns = matching_patterns(item, content, paths, ast=True)
            if any(patterns.values()):
                hits.append({"name": item.rule.name, "path": str(item.rule.path), "matched": patterns})
        if hits:
            total += len(hits)
            matched.append({"filePath": relative, "matches": hits})
    return {"files": matched, "summary": {
        "totalFiles": len(files), "scannedFiles": scanned, "matchedFiles": len(matched), "totalMatches": total,
        "evaluatedRules": len(items), "skippedFiles": sum(skipped.values()), "skipped": skipped,
        "gitignore": not args.no_gitignore, "maxBytes": limit,
    }}


def describe(data, keys):
    return "  ".join(f"{key}: {', '.join(value) if isinstance(value, list) else value}"
                     for key in keys if (value := data.get(key)))


def describe_match(hit, matched=True):
    patterns = hit["matched" if matched else "defined"]
    data = {"condition": [f"/{pattern}/" for pattern in patterns["regex"]], "astCondition": patterns["ast"],
            "agents": hit.get("agents"), "skippedAst": hit.get("skippedAst"), "question": hit.get("question")}
    description = describe(data, data) or "no active conditions"
    provider = f" [{hit['sourceProvider']}]" if hit.get("sourceProvider") else ""
    return f"  {'✓' if matched else '✗'} {hit['name']}  {description}{provider}"


def display(args, result, directory=None):
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    elif args.action == "list":
        if not result:
            print("No TTSR rules registered for this project.")
        else:
            print(f"TTSR rules ({len(result)})")
            for rule in result:
                fields = describe(rule, ("condition", "astCondition", "question", "scope", "globs", "agents"))
                print(f"  {rule['name']} [{rule['provider']}] {fields}")
                if description := rule.get("description"):
                    print(f"    {description}")
    elif args.action == "test":
        context = f"tool:{result.get('tool', '?')}" if result["source"] == "tool" else result["source"]
        path = f" path={result['filePath']}" if "filePath" in result else ""
        print(f"TTSR test — source={context}{path} agent={result['agent']} snippet={result['snippetBytes']}b")
        print(f'  "{result["snippetPreview"]}"\n')
        if note := result.get("inferenceNote"):
            print(f"note: {note}\n")
        if hits := result["triggered"]:
            print(f"Triggered ({len(hits)})")
            print("\n".join(describe_match(hit) for hit in hits))
        else:
            print(f"No rules triggered. (evaluated {result['evaluated']})")
        if args.verbose and (misses := result["notTriggered"]):
            print(f"\nNot triggered ({len(misses)})")
            print("\n".join(describe_match(hit, False) for hit in misses))
    else:
        summary = result["summary"]
        if not summary["totalFiles"]:
            print(f"No files found to scan in {directory}")
            return
        gitignore = "on" if summary["gitignore"] else "off"
        print(f"TTSR scan — directory={directory} files={summary['totalFiles']} scanned={summary['scannedFiles']}"
              f" rules={summary['evaluatedRules']} gitignore={gitignore} max-bytes={summary['maxBytes'] or 'off'}")
        if summary["skippedFiles"]:
            print("  skipped: " + " ".join(f"{key}={count}" for key, count in summary["skipped"].items()))
        print(f"{summary['totalMatches']} matches across {summary['matchedFiles']} files")
        if args.verbose:
            for file in result["files"]:
                print(file["filePath"])
                for hit in file["matches"]:
                    print(describe_match(hit))
        elif summary["matchedFiles"]:
            print("  rerun with --verbose to list matched files and conditions")


def run(argv, defaults, user):
    args = parse(argv)
    cwd = args.cwd.resolve()
    if not sys.stdout.isatty() and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        if not cwd.is_dir():
            raise ValueError(f"Workspace directory not found: {cwd}")
        directory = scan_directory(args, cwd) if args.action == "scan" else None
        items = load(args, cwd, user, defaults)
        if args.action == "scan":
            items = tuple(item for item in items if not item.rule.question)
            if not items:
                raise ValueError("No TTSR rules registered for this project.")
            result = scan_report(args, cwd, directory, items)
        elif args.action == "test":
            if not items:
                raise ValueError(f'No TTSR rules registered for this project as agent "{args.agent}".')
            result = test_report(args, cwd, items)
        else:
            result = [rule_inventory(item.rule) for item in items]
        display(args, result, directory)
        if args.action == "test" and args.rule and not args.json and not result["triggered"]:
            return 1
    except (OSError, ValueError) as error:
        if args.json:
            print(json.dumps({"error": str(error)}, ensure_ascii=False))
        else:
            print(f"mini-pi ttsr: {error}", file=sys.stderr)
        return 1
    return 0
