"""
cli.py
------
Main entry point for the AI Coding Agent CLI.

Commands
--------
  aicoder              — start an interactive chat session (default)
  aicoder config       — view / update configuration
  aicoder version      — print version info

Slash commands (inside the chat session)
-----------------------------------------
  /help                — show available slash commands
  /model <name>        — switch the active model mid-session
  /clear               — clear the conversation history
  /config              — show current config
  /exit  | /quit       — leave the chat
"""

from __future__ import annotations

import sys
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from ai_agent import __version__
from ai_agent.config import load_config, update_config, show_config
from ai_agent.llm import run_agent

# ---------------------------------------------------------------------------
# Typer app
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="aicoder",
    help="AI Coding Agent — your terminal-native AI pair programmer.",
    add_completion=False,
    rich_markup_mode="rich",
)

console = Console()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

WELCOME_BANNER = """
[bold cyan]
  █████╗ ██╗ ██████╗ ██████╗ ██████╗ ███████╗██████╗
 ██╔══██╗██║██╔════╝██╔═══██╗██╔══██╗██╔════╝██╔══██╗
 ███████║██║██║     ██║   ██║██║  ██║█████╗  ██████╔╝
 ██╔══██║██║██║     ██║   ██║██║  ██║██╔══╝  ██╔══██╗
 ██║  ██║██║╚██████╗╚██████╔╝██████╔╝███████╗██║  ██║
 ╚═╝  ╚═╝╚═╝ ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝
[/bold cyan]
[dim]AI Coding Agent  •  Type [bold]/help[/bold] for commands  •  [bold]/exit[/bold] to quit[/dim]
"""

SLASH_HELP = """\
[bold cyan]Slash Commands[/bold cyan]

  [bold]/help[/bold]              Show this help message
  [bold]/model[/bold] [dim]<name>[/dim]      Switch model  (e.g. /model gpt-4o-mini)
  [bold]/clear[/bold]             Clear conversation history
  [bold]/config[/bold]            Show current configuration
  [bold]/exit[/bold]  [dim]or[/dim]  [bold]/quit[/bold]   Exit the chat session
"""


def _print_welcome(model: str) -> None:
    console.print(WELCOME_BANNER)
    console.print(
        Panel(
            f"[bold]Model:[/] [green]{model}[/]   "
            "[dim]Your files are safe — deletes require confirmation.[/dim]",
            border_style="cyan",
            padding=(0, 2),
        )
    )
    console.print()


def _render_reply(reply: str) -> None:
    """Render the assistant reply as Markdown inside a styled panel."""
    console.print()
    console.print(
        Panel(
            Markdown(reply),
            title="[bold cyan]Assistant[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )
    )
    console.print()


def _handle_slash(
    command: str,
    history: list[dict],
    cfg,
) -> tuple[bool, list[dict], object]:
    """
    Handle slash commands typed inside the chat session.

    Returns
    -------
    (should_continue, updated_history, updated_cfg)
    """
    parts = command.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("/exit", "/quit"):
        console.print("\n[bold cyan]Goodbye! Happy coding.[/bold cyan]\n")
        raise typer.Exit()

    elif cmd == "/help":
        console.print(Panel(SLASH_HELP, border_style="dim", padding=(0, 2)))

    elif cmd == "/clear":
        history.clear()
        console.print("[dim]Conversation history cleared.[/dim]")

    elif cmd == "/config":
        show_config()

    elif cmd == "/model":
        if not arg:
            console.print("[bold red]Usage:[/] /model <model-name>")
        else:
            cfg = update_config(model=arg)
            console.print(f"[bold green]✓[/] Model switched to [cyan]{cfg.model}[/]")

    else:
        console.print(f"[bold red]Unknown command:[/] {cmd}  (type /help for options)")

    return True, history, cfg


# ---------------------------------------------------------------------------
# Default command — interactive chat
# ---------------------------------------------------------------------------

@app.callback(invoke_without_command=True)
def chat(ctx: typer.Context) -> None:
    """
    Start an interactive AI coding session in the terminal.
    If a sub-command is provided this callback is skipped.
    """
    if ctx.invoked_subcommand is not None:
        return

    # Load (or create) configuration
    try:
        cfg = load_config()
    except Exception as exc:
        console.print(f"[bold red]Failed to load config:[/] {exc}")
        raise typer.Exit(1)

    _print_welcome(cfg.model)

    # Conversation history shared across turns
    history: list[dict] = []

    # -----------------------------------------------------------------------
    # Main REPL loop
    # -----------------------------------------------------------------------
    while True:
        try:
            # Prompt  (empty line = skip)
            console.print(Rule(style="dim"))
            user_input = console.input("[bold green]You ❯[/bold green] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold cyan]Goodbye! Happy coding.[/bold cyan]\n")
            break

        if not user_input:
            continue

        # --- Slash commands ---
        if user_input.startswith("/"):
            try:
                _, history, cfg = _handle_slash(user_input, history, cfg)
            except typer.Exit:
                break
            continue

        # --- Send to the agent ---
        try:
            reply, history = run_agent(cfg, user_input, history)
            _render_reply(reply)
        except (ValueError, ConnectionError, RuntimeError) as exc:
            console.print(f"\n[bold red]Error:[/] {exc}\n")
        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted. Type /exit to quit.[/dim]")


# ---------------------------------------------------------------------------
# `aicoder config` sub-command
# ---------------------------------------------------------------------------

@app.command("config")
def config_cmd(
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Set the OpenAI model (e.g. gpt-4o)."
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", "-k", help="Set your OpenAI API key."
    ),
    base_url: Optional[str] = typer.Option(
        None, "--base-url", help="Override the API base URL."
    ),
    max_tokens: Optional[int] = typer.Option(
        None, "--max-tokens", help="Maximum tokens per response."
    ),
    temperature: Optional[float] = typer.Option(
        None, "--temperature", help="Sampling temperature (0.0 – 2.0)."
    ),
    show: bool = typer.Option(
        False, "--show", "-s", help="Print current configuration and exit."
    ),
) -> None:
    """
    View or update the agent configuration.

    \b
    Examples:
      aicoder config --show
      aicoder config --model gpt-4-turbo
      aicoder config --api-key sk-... --model gpt-4o
    """
    if show or all(
        v is None for v in [model, api_key, base_url, max_tokens, temperature]
    ):
        show_config()
        return

    try:
        cfg = update_config(
            model=model,
            api_key=api_key,
            base_url=base_url,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        console.print("[bold green]✓ Configuration updated.[/]")
        show_config()
    except Exception as exc:
        console.print(f"[bold red]Failed to update config:[/] {exc}")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# `aicoder version` sub-command
# ---------------------------------------------------------------------------

@app.command("version")
def version_cmd() -> None:
    """Print the current version and exit."""
    console.print(
        f"[bold cyan]AI Coding Agent[/bold cyan] version [bold]{__version__}[/bold]"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Setuptools entry-point wrapper."""
    app()


if __name__ == "__main__":
    main()
