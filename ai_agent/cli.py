"""
cli.py
------
Main entry point for the AI Coding Agent CLI.

Commands
--------
  aicoder              — start an interactive chat session
  aicoder config       — view / update configuration
  aicoder models       — list all available models
  aicoder version      — print version info

Slash commands (inside chat)
-----------------------------
  /help                — show available slash commands
  /model               — interactive model picker (numbered list)
  /clear               — clear conversation history
  /config              — show current config
  /keys                — add / update API keys mid-session
  /exit | /quit        — leave the chat
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from ai_agent import __version__
from ai_agent.config import (
    MODELS,
    AgentConfig,
    ask_api_key,
    get_model_info,
    load_config,
    print_models_table,
    save_config,
    show_config,
    update_config,
)
from ai_agent.llm import run_agent

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="aicoder",
    help="AI Coding Agent — terminal-native AI pair programmer.",
    add_completion=False,
    rich_markup_mode="rich",
)

console = Console()

PROVIDER_STYLE = {"OpenAI": "green", "DeepSeek": "magenta"}

# ---------------------------------------------------------------------------
# Big pixel-art banner (rendered with Rich markup block characters)
# Looks like Gemini / Command Code style large logo
# ---------------------------------------------------------------------------

BANNER = """\
[bold bright_cyan] █████╗ ██╗ ██████╗ ██████╗ ██████╗ ███████╗██████╗ [/]
[bold bright_cyan]██╔══██╗██║██╔════╝██╔═══██╗██╔══██╗██╔════╝██╔══██╗[/]
[bold cyan]███████║██║██║     ██║   ██║██║  ██║█████╗  ██████╔╝[/]
[bold cyan]██╔══██║██║██║     ██║   ██║██║  ██║██╔══╝  ██╔══██╗[/]
[bold blue]██║  ██║██║╚██████╗╚██████╔╝██████╔╝███████╗██║  ██║[/]
[bold blue]╚═╝  ╚═╝╚═╝ ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝[/]"""

TIPS = [
    "Ask me to create files, fix bugs, or explain code.",
    "Be specific for best results — paste errors directly.",
    "Use [bold cyan]/model[/] to switch between OpenAI and DeepSeek.",
    "Type [bold cyan]/help[/] for all available slash commands.",
    "I always read files before editing — your code is safe.",
]

# ---------------------------------------------------------------------------
# Welcome screen — two-panel layout like Claude Code
# ---------------------------------------------------------------------------

def _print_welcome(cfg: AgentConfig) -> None:
    info = get_model_info(cfg.model)
    provider = info["provider"] if info else "Custom"
    provider_style = PROVIDER_STYLE.get(provider, "cyan")
    model_note = info["note"] if info else ""
    cwd = Path.cwd()

    # ── Big banner ──────────────────────────────────────────────────────────
    console.print()
    console.print(Align.center(BANNER))
    console.print()
    console.print(
        Align.center(
            Text.from_markup(
                f"[dim]AI Coding Agent[/]  [bold white]v{__version__}[/]"
            )
        )
    )
    console.print()

    # ── Two-column panel (left: model info  |  right: tips) ─────────────────
    left = Table.grid(padding=(0, 1))
    left.add_column(style="dim", width=14)
    left.add_column()
    left.add_row("Model",    f"[{provider_style}]{cfg.model}[/]")
    left.add_row("Provider", f"[{provider_style}]{provider}[/]")
    left.add_row("Note",     f"[dim]{model_note}[/dim]")
    left.add_row("Dir",      f"[dim]{cwd}[/dim]")

    tips_text = Text()
    tips_text.append("Tips for getting started:\n", style="bold cyan")
    for i, tip in enumerate(TIPS, 1):
        tips_text.append(f"{i}. ", style="bold dim")
        tips_text.append_text(Text.from_markup(f"{tip}\n"))

    console.print(
        Panel(
            Columns(
                [
                    Panel(left,       border_style="cyan",  padding=(1, 2)),
                    Panel(tips_text,  border_style="dim",   padding=(1, 2)),
                ],
                equal=True,
                expand=True,
            ),
            border_style="bright_cyan",
            padding=(0, 1),
        )
    )
    console.print()

# ---------------------------------------------------------------------------
# Bottom status bar — like Command Code's footer
# ---------------------------------------------------------------------------

def _status_bar(cfg: AgentConfig, turn: int) -> Text:
    info = get_model_info(cfg.model)
    provider = info["provider"] if info else "?"
    provider_style = PROVIDER_STYLE.get(provider, "white")
    cwd = Path.cwd()

    bar = Text()
    bar.append(f" {cwd} ", style="dim on grey11")
    bar.append("  ")
    bar.append(f" {provider} ", style=f"bold white on {'dark_green' if provider == 'OpenAI' else 'dark_magenta'}")
    bar.append(" ")
    bar.append(f" {cfg.model} ", style=f"bold {provider_style}")
    bar.append("  ")
    bar.append(f"turn {turn}", style="dim")
    bar.append("  ")
    bar.append("? for /help", style="dim")
    return bar


def _print_prompt_line(cfg: AgentConfig, turn: int) -> None:
    """Print the status bar + input arrow."""
    console.print(_status_bar(cfg, turn))


# ---------------------------------------------------------------------------
# /model interactive picker
# ---------------------------------------------------------------------------

def _model_picker(cfg: AgentConfig) -> AgentConfig:
    console.print()
    print_models_table(console)
    console.print()

    current_idx = next(
        (str(i + 1) for i, m in enumerate(MODELS) if m["id"] == cfg.model), "1"
    )

    while True:
        choice = Prompt.ask(
            f"[bold yellow]Pick a model number[/] [dim](current: {current_idx})[/]",
            default=current_idx,
        )
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(MODELS):
                break
            console.print(f"[bold red]Enter a number between 1 and {len(MODELS)}[/]")
        except ValueError:
            console.print("[bold red]Please enter a number.[/]")

    chosen = MODELS[idx]
    provider = chosen["provider"]
    provider_style = PROVIDER_STYLE.get(provider, "white")

    # Always ask for the API key when switching model
    if provider == "DeepSeek":
        current_key = cfg.deepseek_api_key or ""
        masked = (current_key[:4] + "••••" + current_key[-4:]) if len(current_key) > 8 else ""
        hint = f"current: {masked}" if masked else "not set"
        console.print(f"\n[bold cyan]DeepSeek API key[/] [dim]({hint})[/dim]")
        console.print("[dim]Press Enter to keep current, or paste a new key.[/dim]")
        key = ask_api_key("DeepSeek API key:", console)
        if key:
            cfg = update_config(deepseek_api_key=key)
        elif not current_key:
            console.print("[bold red]✗ No DeepSeek API key. Switch cancelled.[/]")
            return cfg
    else:
        current_key = cfg.api_key or ""
        masked = (current_key[:4] + "••••" + current_key[-4:]) if len(current_key) > 8 else ""
        hint = f"current: {masked}" if masked else "not set"
        console.print(f"\n[bold cyan]OpenAI API key[/] [dim]({hint})[/dim]")
        console.print("[dim]Press Enter to keep current, or paste a new key.[/dim]")
        key = ask_api_key("OpenAI API key:", console)
        if key:
            cfg = update_config(api_key=key)
        elif not current_key:
            console.print("[bold red]✗ No OpenAI API key. Switch cancelled.[/]")
            return cfg

    cfg = update_config(model=chosen["id"], base_url=chosen["base_url"])
    console.print(
        f"\n[bold green]✓ Switched →[/] [{provider_style}]{chosen['provider']}[/] "
        f"[cyan]{chosen['id']}[/] [dim]{chosen['note']}[/dim]\n"
    )
    return cfg

# ---------------------------------------------------------------------------
# Slash command handler
# ---------------------------------------------------------------------------

SLASH_HELP_TEXT = """\
[bold cyan]Slash Commands[/]

  [bold cyan]/help[/]         Show this message
  [bold cyan]/model[/]        Interactive model picker  (OpenAI + DeepSeek)
  [bold cyan]/clear[/]        Wipe conversation history
  [bold cyan]/config[/]       Show current configuration
  [bold cyan]/keys[/]         Update API keys
  [bold cyan]/exit[/]         Quit  (also [bold cyan]/quit[/])
"""


def _update_keys_wizard(cfg: AgentConfig) -> AgentConfig:
    console.print("\n[bold cyan]Update API Keys[/] [dim](Enter = keep current)[/]\n")
    openai_key   = ask_api_key("OpenAI API key   :", console)
    deepseek_key = ask_api_key("DeepSeek API key :", console)
    cfg = update_config(
        api_key=openai_key or cfg.api_key,
        deepseek_api_key=deepseek_key or cfg.deepseek_api_key,
    )
    console.print("[bold green]✓ Keys updated.[/]\n")
    return cfg


def _handle_slash(command: str, history: list[dict], cfg: AgentConfig):
    parts = command.strip().split(maxsplit=1)
    cmd   = parts[0].lower()

    if cmd in ("/exit", "/quit"):
        console.print("\n[bold cyan]Goodbye! Happy coding. 👋[/]\n")
        raise typer.Exit()

    elif cmd == "/help":
        console.print(
            Panel(SLASH_HELP_TEXT, border_style="cyan", padding=(0, 2), title="[cyan]Help[/]")
        )

    elif cmd == "/clear":
        history.clear()
        console.print("[dim]  ✓ Conversation history cleared.[/dim]")

    elif cmd == "/config":
        show_config()

    elif cmd == "/model":
        cfg = _model_picker(cfg)

    elif cmd == "/keys":
        cfg = _update_keys_wizard(cfg)

    else:
        console.print(
            f"[bold red]Unknown command:[/] {cmd}  — type [bold]/help[/] for options"
        )

    return history, cfg

# ---------------------------------------------------------------------------
# Message rendering
# ---------------------------------------------------------------------------

def _render_user_msg(text: str) -> None:
    ts = datetime.now().strftime("%H:%M")
    console.print(
        Panel(
            Text(text, style="white"),
            title=f"[bold green]You[/]  [dim]{ts}[/]",
            title_align="right",
            border_style="green",
            padding=(0, 2),
        )
    )


def _render_assistant_msg(reply: str, model_id: str) -> None:
    info   = get_model_info(model_id)
    provider = info["provider"] if info else "AI"
    style  = PROVIDER_STYLE.get(provider, "cyan")
    ts     = datetime.now().strftime("%H:%M")
    console.print()
    console.print(
        Panel(
            Markdown(reply),
            title=f"[bold {style}]✦ {provider} · {model_id}[/]  [dim]{ts}[/]",
            title_align="left",
            border_style=style,
            padding=(1, 2),
        )
    )
    console.print()


def _render_error(msg: str) -> None:
    console.print(
        Panel(
            Text(msg, style="bold red"),
            title="[bold red]Error[/]",
            border_style="red",
            padding=(0, 2),
        )
    )


# ---------------------------------------------------------------------------
# Main chat REPL
# ---------------------------------------------------------------------------

@app.callback(invoke_without_command=True)
def chat(ctx: typer.Context) -> None:
    """Start an interactive AI coding session."""
    if ctx.invoked_subcommand is not None:
        return

    try:
        cfg = load_config()
    except Exception as exc:
        console.print(f"[bold red]Failed to load config:[/] {exc}")
        raise typer.Exit(1)

    _print_welcome(cfg)

    history: list[dict] = []
    turn = 0

    while True:
        # Status bar + prompt
        _print_prompt_line(cfg, turn)
        try:
            user_input = console.input("[bold bright_cyan]❯[/] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold cyan]Goodbye! Happy coding. 👋[/]\n")
            break

        if not user_input:
            continue

        # Slash commands
        if user_input.startswith("/"):
            try:
                history, cfg = _handle_slash(user_input, history, cfg)
            except typer.Exit:
                break
            continue

        # Send to agent
        turn += 1
        _render_user_msg(user_input)
        try:
            reply, history = run_agent(cfg, user_input, history)
            _render_assistant_msg(reply, cfg.model)
        except (ValueError, ConnectionError, RuntimeError) as exc:
            _render_error(str(exc))
        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted — type /exit to quit.[/dim]")

# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------

@app.command("config")
def config_cmd(
    model:       Optional[str]   = typer.Option(None, "--model",       "-m", help="Set active model."),
    api_key:     Optional[str]   = typer.Option(None, "--api-key",     "-k", help="Set OpenAI API key."),
    deepseek_key:Optional[str]   = typer.Option(None, "--deepseek-key","-d", help="Set DeepSeek API key."),
    base_url:    Optional[str]   = typer.Option(None, "--base-url",          help="Override API base URL."),
    max_tokens:  Optional[int]   = typer.Option(None, "--max-tokens",        help="Max tokens per response."),
    temperature: Optional[float] = typer.Option(None, "--temperature",       help="Sampling temperature."),
    show:        bool            = typer.Option(False,"--show", "-s",         help="Print config and exit."),
) -> None:
    """
    View or update configuration.

    \b
    Examples:
      aicoder config --show
      aicoder config --model deepseek-chat --deepseek-key sk-...
      aicoder config --model gpt-4o --api-key sk-...
    """
    any_set = any(
        v is not None
        for v in [model, api_key, deepseek_key, base_url, max_tokens, temperature]
    )
    if show or not any_set:
        show_config()
        return

    if model and base_url is None:
        from ai_agent.config import get_base_url
        base_url = get_base_url(model)

    try:
        update_config(
            model=model, api_key=api_key, deepseek_api_key=deepseek_key,
            base_url=base_url, max_tokens=max_tokens, temperature=temperature,
        )
        console.print("[bold green]✓ Configuration updated.[/]")
        show_config()
    except Exception as exc:
        console.print(f"[bold red]Failed:[/] {exc}")
        raise typer.Exit(1)


@app.command("models")
def models_cmd() -> None:
    """List all available models (OpenAI + DeepSeek)."""
    print_models_table(console)
    console.print(
        "[dim]Switch inside chat with [bold]/model[/], "
        "or via [bold]aicoder config --model <id>[/][/dim]\n"
    )


@app.command("version")
def version_cmd() -> None:
    """Print version and exit."""
    console.print(
        f"[bold bright_cyan]AI Coding Agent[/] [bold white]v{__version__}[/]"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    app()


if __name__ == "__main__":
    main()
