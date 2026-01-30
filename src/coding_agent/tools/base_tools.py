"""File operation tools backed by a FileService."""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path

from coding_agent.services.file_service import FileService
from coding_agent.tools import Tool

MAX_GREP_MATCHES = 200


def create_base_tools(service: FileService) -> list[Tool]:
    def view_file(args: dict) -> str:
        path = args["path"]
        content = service.read_file(path)
        lines = content.splitlines()
        offset = args.get("offset", 0)
        limit = args.get("limit")
        if offset or limit:
            end = offset + limit if limit else len(lines)
            lines = lines[offset:end]
        numbered = [f"{i + offset + 1:>4} | {line}" for i, line in enumerate(lines)]
        return f"File: {path}\n" + "\n".join(numbered)

    def edit_file(args: dict) -> str:
        path = args["path"]
        old_text = args["old_text"]
        new_text = args["new_text"]
        replace_all = args.get("replace_all", False)
        if replace_all:
            content = service.read_file(path)
            if old_text not in content:
                return f"old_text not found in {path}"
            updated = content.replace(old_text, new_text)
            service.write_file(path, updated)
            count = content.count(old_text)
            return f"Replaced {count} occurrence(s) in {path}"
        service.edit_file(path, old_text, new_text)
        return f"Successfully edited {path}"

    def create_file(args: dict) -> str:
        path = args["path"]
        content = args["content"]
        service.write_file(path, content)
        return f"Created {path}"

    def list_directory(args: dict) -> str:
        path = args.get("path", ".")
        entries = service.list_directory(path)
        return "\n".join(entries) if entries else "(empty directory)"

    def grep(args: dict) -> str:
        pattern = args["pattern"]
        search_path = args.get("path", ".")
        include = args.get("include")
        try:
            regex = re.compile(pattern)
        except re.error:
            regex = re.compile(re.escape(pattern))

        root = _resolve_path(service, search_path)
        matches: list[str] = []
        for fpath in _walk_files(root, include):
            try:
                text = fpath.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            rel = str(fpath.relative_to(_work_dir(service)))
            for i, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    matches.append(f"{rel}:{i}: {line.rstrip()}")
                    if len(matches) >= MAX_GREP_MATCHES:
                        matches.append(f"... (truncated at {MAX_GREP_MATCHES} matches)")
                        return "\n".join(matches)
        if not matches:
            return f"No matches for '{pattern}'"
        return "\n".join(matches)

    def find_files(args: dict) -> str:
        pattern = args["pattern"]
        search_path = args.get("path", ".")
        root = _resolve_path(service, search_path)
        work = _work_dir(service)
        results: list[str] = []
        for fpath in sorted(root.rglob("*")):
            if fpath.is_file() and fnmatch.fnmatch(fpath.name, pattern):
                results.append(str(fpath.relative_to(work)))
            if len(results) >= 500:
                results.append("... (truncated at 500 results)")
                break
        if not results:
            return f"No files matching '{pattern}'"
        return "\n".join(results)

    return [
        Tool(
            name="view_file",
            description="Read the contents of a file. Use offset/limit for large files.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"},
                    "offset": {"type": "integer", "description": "Starting line (0-based)"},
                    "limit": {"type": "integer", "description": "Max lines to return"},
                },
                "required": ["path"],
            },
            execute=view_file,
        ),
        Tool(
            name="edit_file",
            description=(
                "Replace an exact text snippet in a file with new text. "
                "Set replace_all=true to replace every occurrence."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to edit"},
                    "old_text": {"type": "string", "description": "Exact text to find and replace"},
                    "new_text": {"type": "string", "description": "Replacement text"},
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace all occurrences instead of just the first (default: false)",
                    },
                },
                "required": ["path", "old_text", "new_text"],
            },
            execute=edit_file,
        ),
        Tool(
            name="create_file",
            description="Create a new file with the given content.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to create"},
                    "content": {"type": "string", "description": "File content"},
                },
                "required": ["path", "content"],
            },
            execute=create_file,
        ),
        Tool(
            name="list_directory",
            description="List files and directories in the given path.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path (default: '.')"},
                },
                "required": [],
            },
            execute=list_directory,
        ),
        Tool(
            name="grep",
            description=(
                "Search file contents for a pattern (regex or literal). "
                "Returns matching lines as file:line:text. "
                "Use include to filter by glob (e.g. '*.py')."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex or literal string to search for",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory or file to search in (default: '.')",
                    },
                    "include": {
                        "type": "string",
                        "description": "Glob to filter files, e.g. '*.py', '*.yaml'",
                    },
                },
                "required": ["pattern"],
            },
            execute=grep,
        ),
        Tool(
            name="find_files",
            description=(
                "Find files by name pattern (glob). "
                "Searches recursively from the given path. "
                "Example: pattern='*.test.py', path='src'."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern to match file names, e.g. '*.py', 'test_*'",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search in (default: '.')",
                    },
                },
                "required": ["pattern"],
            },
            execute=find_files,
        ),
    ]


def _work_dir(service: FileService) -> Path:
    """Get the working directory from the service."""
    return getattr(service, "work_dir", Path.cwd())


def _resolve_path(service: FileService, path: str) -> Path:
    """Resolve a relative path against the service work_dir."""
    p = Path(path)
    if p.is_absolute():
        return p
    return _work_dir(service) / p


def _walk_files(root: Path, include: str | None = None) -> list[Path]:
    """Recursively collect files, optionally filtered by glob."""
    if root.is_file():
        return [root]
    files: list[Path] = []
    for fpath in sorted(root.rglob("*")):
        if not fpath.is_file():
            continue
        if include and not fnmatch.fnmatch(fpath.name, include):
            continue
        files.append(fpath)
    return files
