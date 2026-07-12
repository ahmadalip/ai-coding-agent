"""
tools.py
--------
File-system tools with animated spinners for every operation.

Each tool shows:
  ⟳  <verb> <filename>...   (spinner while working)
  ✓  <verb> <filename>      (green on success)
  ✗  <verb> <filename>      (red on failure)
"""

import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.prompt import Confirm
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.text import Text

console = Console()

# ---------------------------------------------------------------------------
# Animated operation context manager
# ---------------------------------------------------------------------------

TOOL_ICONS = {
    "create":  ("📝", "Creating",   "Created"),
    "edit":    ("✏️ ", "Editing",    "Edited"),
    "delete":  ("🗑️ ", "Deleting",   "Deleted"),
    "read":    ("📖", "Reading",    "Read"),
    "list":    ("📁", "Scanning",   "Scanned"),
}


@contextmanager
def _animated(verb: str, target: str):
    """
    Context manager that shows a spinner while the body runs, then
    prints a clean ✓/✗ completion line.

    Usage:
        with _animated("create", "app.py") as status:
            ... do work ...
            # on error: raise an exception — ✗ will be shown automatically
    """
    icon, present, past = TOOL_ICONS.get(verb, ("⚙️ ", verb.capitalize() + "ing", verb.capitalize() + "ed"))
    spinner_text = Text.assemble(
        (f"  {icon}  ", ""),
        (present + " ", "bold cyan"),
        (target, "cyan"),
        ("…", "dim"),
    )
    spinner = Spinner("dots2", text=spinner_text)

    success = False
    try:
        with Live(spinner, console=console, refresh_per_second=15, transient=True):
            yield
        success = True
    finally:
        if success:
            console.print(Text.assemble(
                ("  ✓  ", "bold green"),
                (past + " ", "green"),
                (target, "bold cyan"),
            ))
        else:
            console.print(Text.assemble(
                ("  ✗  ", "bold red"),
                ("Failed — ", "red"),
                (target, "bold cyan"),
            ))


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def create_file(file_path: str, content: str) -> str:
    path = Path(file_path)
    name = path.name

    with _animated("create", name):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        lines = len(content.splitlines())

    console.print(
        f"    [dim]{file_path}[/dim]  "
        f"[dim]{lines} lines · {len(content)} chars[/dim]"
    )
    return f"File '{file_path}' created successfully ({lines} lines, {len(content)} chars)."


def edit_file(file_path: str, old_snippet: str, new_snippet: str) -> str:
    path = Path(file_path)
    name = path.name

    if not path.exists():
        msg = f"ERROR: '{file_path}' does not exist. Use create_file first."
        console.print(f"  [bold red]✗[/] {msg}")
        return msg

    try:
        with _animated("edit", name):
            original = path.read_text(encoding="utf-8")
            if old_snippet not in original:
                raise ValueError("snippet not found")
            updated = original.replace(old_snippet, new_snippet, 1)
            path.write_text(updated, encoding="utf-8")

        added   = len(new_snippet.splitlines())
        removed = len(old_snippet.splitlines())
        console.print(
            f"    [dim]{file_path}[/dim]  "
            f"[green]+{added}[/green] [red]-{removed}[/red] [dim]lines[/dim]"
        )
        return f"File '{file_path}' edited successfully (+{added}/-{removed} lines)."

    except ValueError:
        msg = (
            f"ERROR: Snippet not found in '{file_path}'. "
            "Read the file first to get exact content."
        )
        console.print(f"  [bold red]✗[/] {msg}")
        return msg
    except Exception as exc:
        return f"ERROR editing '{file_path}': {exc}"


def delete_file(file_path: str) -> str:
    path = Path(file_path)
    name = path.name

    if not path.exists():
        msg = f"ERROR: '{file_path}' does not exist."
        console.print(f"  [bold red]✗[/] {msg}")
        return msg

    console.print(
        f"\n  [bold yellow]⚠  Delete request:[/] [cyan]{file_path}[/]"
    )
    confirmed = Confirm.ask("  [bold red]Confirm deletion?[/]", default=False)

    if not confirmed:
        console.print("  [dim]Deletion cancelled.[/dim]")
        return f"Deletion of '{file_path}' was cancelled by the user."

    with _animated("delete", name):
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

    return f"'{file_path}' deleted successfully."


def read_file(file_path: str) -> str:
    path = Path(file_path)
    name = path.name

    if not path.exists():
        msg = f"ERROR: '{file_path}' does not exist."
        console.print(f"  [bold red]✗[/] {msg}")
        return msg

    with _animated("read", name):
        content = path.read_text(encoding="utf-8")

    # Show a compact syntax-highlighted preview (first 40 lines)
    suffix = path.suffix.lstrip(".") or "text"
    lexer_map = {
        "py": "python", "js": "javascript", "ts": "typescript",
        "json": "json", "yaml": "yaml", "yml": "yaml",
        "md": "markdown", "html": "html", "css": "css",
        "sh": "bash", "bash": "bash", "txt": "text",
        "toml": "toml", "rs": "rust", "go": "go",
    }
    lexer = lexer_map.get(suffix, "text")
    all_lines = content.splitlines()
    preview = "\n".join(all_lines[:40])
    if len(all_lines) > 40:
        preview += f"\n  … {len(all_lines) - 40} more lines"

    console.print(
        Syntax(preview, lexer, theme="monokai", line_numbers=True,
               background_color="default")
    )
    console.print(
        f"    [dim]{file_path}[/dim]  [dim]{len(all_lines)} lines[/dim]"
    )
    return content


def list_files(directory: str = ".") -> str:
    root = Path(directory).resolve()

    if not root.exists():
        msg = f"ERROR: Directory '{directory}' does not exist."
        console.print(f"  [bold red]✗[/] {msg}")
        return msg

    SKIP = {
        "__pycache__", ".git", ".venv", "venv", "node_modules",
        ".mypy_cache", ".pytest_cache", "dist", "build",
    }

    lines: list[str] = []

    def _should_skip(name: str) -> bool:
        return name in SKIP or name.endswith(".egg-info")

    def _walk(path: Path, prefix: str = "") -> None:
        try:
            entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))
        except PermissionError:
            return
        entries = [e for e in entries if not _should_skip(e.name)]
        for i, entry in enumerate(entries):
            connector = "└── " if i == len(entries) - 1 else "├── "
            lines.append(f"{prefix}{connector}{entry.name}")
            if entry.is_dir():
                ext = "    " if i == len(entries) - 1 else "│   "
                _walk(entry, prefix + ext)

    with _animated("list", str(root.name)):
        lines.append(str(root))
        _walk(root)

    tree_str = "\n".join(lines)
    console.print(f"[dim]{tree_str}[/dim]")
    return tree_str


# ---------------------------------------------------------------------------
# Registry & dispatcher
# ---------------------------------------------------------------------------

TOOL_FUNCTIONS: dict[str, Any] = {
    "create_file": create_file,
    "edit_file":   edit_file,
    "delete_file": delete_file,
    "read_file":   read_file,
    "list_files":  list_files,
}


def execute_tool(tool_name: str, tool_args: dict[str, Any]) -> str:
    fn = TOOL_FUNCTIONS.get(tool_name)
    if fn is None:
        return f"ERROR: Unknown tool '{tool_name}'."
    try:
        return fn(**tool_args)
    except TypeError as exc:
        return f"ERROR: Bad arguments for '{tool_name}': {exc}"


# ---------------------------------------------------------------------------
# OpenAI function-calling schema
# ---------------------------------------------------------------------------

TOOLS_SCHEMA: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": "Create a new file with content. Parent dirs auto-created.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path for the new file."},
                    "content":   {"type": "string", "description": "Full file content."},
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace old_snippet with new_snippet in a file. Call read_file first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path":   {"type": "string", "description": "Path to the file."},
                    "old_snippet": {"type": "string", "description": "Exact text to replace."},
                    "new_snippet": {"type": "string", "description": "Replacement text."},
                },
                "required": ["file_path", "old_snippet", "new_snippet"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_file",
            "description": "Delete a file or directory. User confirmation required.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to delete."},
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read and return the full content of a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to read."},
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files as a tree. Use '.' for current directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {"type": "string", "description": "Directory to list."},
                },
                "required": [],
            },
        },
    },
]
