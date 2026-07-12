"""
tools.py
--------
File-system tools exposed to the AI via OpenAI function calling.

Each tool is defined in two places:
  1. A plain Python function that does the actual work.
  2. A JSON schema entry in TOOLS_SCHEMA that OpenAI uses to decide when/how
     to call the function.

The dispatch helper ``execute_tool`` routes a tool-call from the model to
the correct Python function and returns a string result that is fed back into
the conversation as a "tool" role message.
"""

import os
import shutil
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.prompt import Confirm
from rich.syntax import Syntax

console = Console()

# ---------------------------------------------------------------------------
# Individual tool implementations
# ---------------------------------------------------------------------------

def create_file(file_path: str, content: str) -> str:
    """
    Create a new file at *file_path* and write *content* into it.
    Parent directories are created automatically.
    Returns a human-readable status string.
    """
    path = Path(file_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        console.print(f"  [bold green]✓ Created[/] [cyan]{file_path}[/]")
        return f"File '{file_path}' created successfully ({len(content)} chars)."
    except Exception as exc:
        console.print(f"  [bold red]✗ create_file failed:[/] {exc}")
        return f"ERROR creating '{file_path}': {exc}"


def edit_file(file_path: str, old_snippet: str, new_snippet: str) -> str:
    """
    Replace the first occurrence of *old_snippet* with *new_snippet* in the
    file at *file_path*.

    The AI should call read_file first so it has the exact current content
    before constructing a diff.
    """
    path = Path(file_path)
    if not path.exists():
        msg = f"ERROR: '{file_path}' does not exist. Use create_file first."
        console.print(f"  [bold red]✗[/] {msg}")
        return msg

    try:
        original = path.read_text(encoding="utf-8")
        if old_snippet not in original:
            msg = (
                f"ERROR: The specified snippet was not found in '{file_path}'. "
                "Make sure you read the file first and copy the snippet exactly."
            )
            console.print(f"  [bold red]✗[/] {msg}")
            return msg

        updated = original.replace(old_snippet, new_snippet, 1)
        path.write_text(updated, encoding="utf-8")
        console.print(f"  [bold green]✓ Edited[/] [cyan]{file_path}[/]")
        return f"File '{file_path}' edited successfully."
    except Exception as exc:
        console.print(f"  [bold red]✗ edit_file failed:[/] {exc}")
        return f"ERROR editing '{file_path}': {exc}"


def delete_file(file_path: str) -> str:
    """
    Delete the file (or empty directory) at *file_path*.
    Always asks the user for y/n confirmation before proceeding.
    """
    path = Path(file_path)
    if not path.exists():
        msg = f"ERROR: '{file_path}' does not exist."
        console.print(f"  [bold red]✗[/] {msg}")
        return msg

    console.print(f"\n  [bold yellow]⚠  The AI wants to delete:[/] [cyan]{file_path}[/]")
    confirmed = Confirm.ask("  [bold red]Confirm deletion?[/]", default=False)

    if not confirmed:
        console.print("  [dim]Deletion cancelled by user.[/]")
        return f"Deletion of '{file_path}' was cancelled by the user."

    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        console.print(f"  [bold green]✓ Deleted[/] [cyan]{file_path}[/]")
        return f"'{file_path}' deleted successfully."
    except Exception as exc:
        console.print(f"  [bold red]✗ delete_file failed:[/] {exc}")
        return f"ERROR deleting '{file_path}': {exc}"


def read_file(file_path: str) -> str:
    """
    Read and return the full content of *file_path*.
    Also pretty-prints a syntax-highlighted preview in the terminal.
    """
    path = Path(file_path)
    if not path.exists():
        msg = f"ERROR: '{file_path}' does not exist."
        console.print(f"  [bold red]✗[/] {msg}")
        return msg

    try:
        content = path.read_text(encoding="utf-8")
        # Determine lexer from extension for syntax highlighting
        suffix = path.suffix.lstrip(".") or "text"
        lexer_map = {
            "py": "python", "js": "javascript", "ts": "typescript",
            "json": "json", "yaml": "yaml", "yml": "yaml",
            "md": "markdown", "html": "html", "css": "css",
            "sh": "bash", "bash": "bash", "txt": "text",
            "toml": "toml", "rs": "rust", "go": "go",
        }
        lexer = lexer_map.get(suffix, "text")

        console.print(f"  [bold blue]📄 Reading[/] [cyan]{file_path}[/]")
        # Show a condensed preview (first 60 lines) — full content goes to the model
        preview_lines = content.splitlines()[:60]
        preview = "\n".join(preview_lines)
        if len(content.splitlines()) > 60:
            preview += f"\n… ({len(content.splitlines()) - 60} more lines)"
        console.print(Syntax(preview, lexer, theme="monokai", line_numbers=True))

        return content
    except Exception as exc:
        console.print(f"  [bold red]✗ read_file failed:[/] {exc}")
        return f"ERROR reading '{file_path}': {exc}"


def list_files(directory: str = ".") -> str:
    """
    Recursively list files and directories under *directory*, rendered as an
    indented tree.  Returns the tree as a plain string for the model.
    """
    root = Path(directory).resolve()
    if not root.exists():
        msg = f"ERROR: Directory '{directory}' does not exist."
        console.print(f"  [bold red]✗[/] {msg}")
        return msg

    lines: list[str] = []

    # Folders/files to skip (keep output clean)
    SKIP = {
        "__pycache__", ".git", ".venv", "venv", "node_modules",
        ".mypy_cache", ".pytest_cache", "dist", "build", "*.egg-info",
    }

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
                extension = "    " if i == len(entries) - 1 else "│   "
                _walk(entry, prefix + extension)

    lines.append(str(root))
    _walk(root)
    tree_str = "\n".join(lines)

    console.print(f"  [bold blue]📁 Listing[/] [cyan]{directory}[/]")
    console.print(f"[dim]{tree_str}[/dim]")
    return tree_str


# ---------------------------------------------------------------------------
# Tool registry — maps name → callable
# ---------------------------------------------------------------------------

TOOL_FUNCTIONS: dict[str, Any] = {
    "create_file": create_file,
    "edit_file": edit_file,
    "delete_file": delete_file,
    "read_file": read_file,
    "list_files": list_files,
}


def execute_tool(tool_name: str, tool_args: dict[str, Any]) -> str:
    """
    Dispatch a tool call from the model to the correct Python function.

    Parameters
    ----------
    tool_name : str
        The name returned by the model in the tool-call object.
    tool_args : dict
        The parsed JSON arguments from the model.

    Returns
    -------
    str
        A plain-text result that will be sent back as a ``tool`` role message.
    """
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
            "description": (
                "Create a new file at the given path and write content into it. "
                "Parent directories are created automatically."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Relative or absolute path for the new file.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full text content to write into the file.",
                    },
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Edit an existing file by replacing old_snippet with new_snippet. "
                "Always call read_file first to get the exact current content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to edit.",
                    },
                    "old_snippet": {
                        "type": "string",
                        "description": "The exact text to be replaced (must match verbatim).",
                    },
                    "new_snippet": {
                        "type": "string",
                        "description": "The new text to substitute in place of old_snippet.",
                    },
                },
                "required": ["file_path", "old_snippet", "new_snippet"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_file",
            "description": (
                "Delete a file or directory. The user will be asked to confirm "
                "before deletion actually happens."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file or directory to delete.",
                    },
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
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to read.",
                    },
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": (
                "List files and directories in a given directory as a tree. "
                "Use '.' for the current working directory."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Directory path to list. Defaults to '.'.",
                    },
                },
                "required": [],
            },
        },
    },
]
