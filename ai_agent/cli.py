"""
cli.py — AI Coding Agent entry point.

Chat display rules:
  • AI replies stream live, token by token.
  • Code blocks are stripped — all code lives in files, not in the chat.
  • Tool operations show animated spinners with file names.
  • A status bar shows model / provider / turn count at every prompt.
"""

from __future__ import annotations

import re
import time
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

import typer
from rich.align import Align
from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
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
    show_config,
    update_config,
)
from ai_agent.llm import run_agent

# ---------------------------------------------------------------------------
# Typer app
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
# Banner
# ---------------------------------------------------------------------------

BANNER = """\
[bold bright_cyan] █████╗ ██╗ ██████╗ ██████╗ ██████╗ ███████╗██████╗ [/]
[bold bright_cyan]██╔══██╗██║██╔════╝██╔═══██╗██╔══██╗██╔════╝██╔══██╗[/]
[bold cyan]███████║██║██║     ██║   ██║██║  ██║█████╗  ██████╔╝[/]
[bold cyan]██╔══██║██║██║     ██║   ██║██║  ██║██╔══╝  ██╔══██╗[/]
[bold blue]██║  ██║██║╚██████╗╚██████╔╝██████╔╝███████╗██║  ██║[/]
[bold blue]╚═╝  ╚═╝╚═╝ ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝[/]"""

TIPS = [
    "Ask me to create, edit, or explain any file.",
    "Paste error messages — I'll fix them.",
    "Use [bold cyan]/model[/] to switch between OpenAI & DeepSeek.",
    "[bold cyan]/help[/] shows all slash commands.",
    "All code goes into files — chat stays clean.",
]

# ---------------------------------------------------------------------------
# Welcome screen
# ---------------------------------------------------------------------------

def _print_welcome(cfg: AgentConfig) -> None:
    info           = get_model_info(cfg.model)
    provider       = info["provider"] if info else "Custom"
    provider_style = PROVIDER_STYLE.get(provider, "cyan")
    note           = info["note"] if info else ""

    console.print()
    console.print(Align.center(BANNER))
    console.print()
    console.print(Align.center(
        Text.from_markup(f"[dim]AI Coding Agent[/]  [bold white]v{__version__}[/]")
    ))
    console.print()

    left = Table.grid(padding=(0, 1))
    left.add_column(style="dim", width=12)
    left.add_column()
    left.add_row("Model",    f"[{provider_style}]{cfg.model}[/]")
    left.add_row("Provider", f"[{provider_style}]{provider}[/]")
    left.add_row("Note",     f"[dim]{note}[/dim]")
    left.add_row("Dir",      f"[dim]{Path.cwd()}[/dim]")

    tips_text = Text()
    tips_text.append("Tips for getting started:\n", style="bold cyan")
    for i, tip in enumerate(TIPS, 1):
        tips_text.append(f"{i}. ", style="bold dim")
        tips_text.append_text(Text.from_markup(f"{tip}\n"))

    console.print(Panel(
        Columns(
            [
                Panel(left,       border_style="cyan", padding=(1, 2)),
                Panel(tips_text,  border_style="dim",  padding=(1, 2)),
            ],
            equal=True, expand=True,
        ),
        border_style="bright_cyan",
        padding=(0, 1),
    ))
    console.print()

# ---------------------------------------------------------------------------
# Status bar
# ---------------------------------------------------------------------------

def _status_bar(cfg: AgentConfig, turn: int) -> Text:
    info           = get_model_info(cfg.model)
    provider       = info["provider"] if info else "?"
    provider_style = PROVIDER_STYLE.get(provider, "white")
    bg             = "dark_green" if provider == "OpenAI" else "dark_magenta"

    bar = Text()
    bar.append(f" {Path.cwd()} ", style="dim on grey11")
    bar.append("  ")
    bar.append(f" {provider} ", style=f"bold white on {bg}")
    bar.append(" ")
    bar.append(f" {cfg.model} ", style=f"bold {provider_style}")
    bar.append("  ")
    bar.append(f"turn {turn}", style="dim")
    bar.append("   ")
    bar.append("/help for commands", style="dim")
    return bar

# ---------------------------------------------------------------------------
# Code-stripping for chat display
# ---------------------------------------------------------------------------

# Matches fenced code blocks: ```lang\n...\n``` or ~~~...~~~
_CODE_BLOCK_RE = re.compile(r"```[\w]*\n.*?```|~~~[\w]*\n.*?~~~", re.DOTALL)
# Matches inline code: `something`
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")


def _strip_code(text: str) -> str:
    """
    Remove code blocks and inline code from the reply.
    Replace fenced blocks with a dim note referencing the file.
    """
    # Replace fenced blocks with a placeholder
    def _block_sub(m: re.Match) -> str:
        block = m.group(0)
        # Try to extract a filename hint from the text just before the block
        return "[dim]  ↳ code written to file[/dim]"

    stripped = _CODE_BLOCK_RE.sub(_block_sub, text)
    # Remove inline backticks — just show the bare word
    stripped = _INLINE_CODE_RE.sub(lambda m: m.group(0)[1:-1], stripped)
    return stripped.strip()

# ---------------------------------------------------------------------------
# Live streaming renderer
# ---------------------------------------------------------------------------

def _stream_reply(token_iter: Iterator[str], cfg: AgentConfig) -> str:
    """
    Consume token_iter and display each token as it arrives inside a Live
    panel.  Code blocks are stripped in real-time so the chat stays clean.
    Returns the full accumulated plain text.
    """
    info           = get_model_info(cfg.model)
    provider       = info["provider"] if info else "AI"
    provider_style = PROVIDER_STYLE.get(provider, "cyan")
    ts             = datetime.now().strftime("%H:%M")
    title          = f"[bold {provider_style}]✦ {provider} · {cfg.model}[/]  [dim]{ts}[/]"

    accumulated    = ""
    display_text   = Text()

    console.print()

    with Live(
        Panel(display_text, title=title, border_style=provider_style, padding=(1, 2)),
        console=console,
        refresh_per_second=20,
        vertical_overflow="visible",
    ) as live:
        for token in token_iter:
            accumulated += token

            # Strip code from what we display — update live panel
            clean = _strip_code(accumulated)
            display_text = Text.from_markup(clean) if clean else Text()

            live.update(
                Panel(display_text, title=title, border_style=provider_style, padding=(1, 2))
            )

    console.print()
    return accumulated

# ---------------------------------------------------------------------------
# /model picker
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
            console.print(f"[bold red]Enter 1–{len(MODELS)}[/]")
        except ValueError:
            console.print("[bold red]Enter a number.[/]")

    chosen         = MODELS[idx]
    provider       = chosen["provider"]
    provider_style = PROVIDER_STYLE.get(provider, "white")

    # Always ask for the key when switching
    if provider == "DeepSeek":
        current_key = cfg.deepseek_api_key or ""
        masked = (current_key[:4] + "••••" + current_key[-4:]) if len(current_key) > 8 else ""
        hint   = f"current: {masked}" if masked else "not set"
        console.print(f"\n[bold cyan]DeepSeek API key[/] [dim]({hint})[/dim]")
        console.print("[dim]Press Enter to keep, or paste a new key.[/dim]")
        key = ask_api_key("DeepSeek API key:", console)
        if key:
            cfg = update_config(deepseek_api_key=key)
        elif not current_key:
            console.print("[bold red]✗ No key — switch cancelled.[/]")
            return cfg
    else:
        current_key = cfg.api_key or ""
        masked = (current_key[:4] + "••••" + current_key[-4:]) if len(current_key) > 8 else ""
        hint   = f"current: {masked}" if masked else "not set"
        console.print(f"\n[bold cyan]OpenAI API key[/] [dim]({hint})[/dim]")
        console.print("[dim]Press Enter to keep, or paste a new key.[/dim]")
        key = ask_api_key("OpenAI API key:", console)
        if key:
            cfg = update_config(api_key=key)
        elif not current_key:
            console.print("[bold red]✗ No key — switch cancelled.[/]")
            return cfg

    cfg = update_config(model=chosen["id"], base_url=chosen["base_url"])
    console.print(
        f"\n[bold green]✓ Switched →[/] [{provider_style}]{chosen['provider']}[/] "
        f"[cyan]{chosen['id']}[/] [dim]{chosen['note']}[/dim]\n"
    )
    return cfg

# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------

SLASH_HELP = """\
[bold cyan]Slash Commands[/]

  [bold cyan]/help[/]       This message
  [bold cyan]/model[/]      Pick model  (OpenAI + DeepSeek)
  [bold cyan]/clear[/]      Clear conversation history
  [bold cyan]/config[/]     Show configuration
  [bold cyan]/keys[/]       Update API keys
  [bold cyan]/exit[/]       Quit  (also [bold cyan]/quit[/])
"""


def _keys_wizard(cfg: AgentConfig) -> AgentConfig:
    console.print("\n[bold cyan]Update API Keys[/] [dim](Enter = keep current)[/]\n")
    openai_key   = ask_api_key("OpenAI API key   :", console)
    deepseek_key = ask_api_key("DeepSeek API key :", console)
    cfg = update_config(
        api_key=openai_key           or cfg.api_key,
        deepseek_api_key=deepseek_key or cfg.deepseek_api_key,
    )
    console.print("[bold green]✓ Keys updated.[/]\n")
    return cfg


def _slash(cmd_line: str, history: list[dict], cfg: AgentConfig):
    parts = cmd_line.strip().split(maxsplit=1)
    cmd   = parts[0].lower()

    if cmd in ("/exit", "/quit"):
        console.print("\n[bold cyan]Goodbye! 👋[/]\n")
        raise typer.Exit()
    elif cmd == "/help":
        console.print(Panel(SLASH_HELP, border_style="cyan", padding=(0, 2), title="[cyan]Help[/]"))
    elif cmd == "/clear":
        history.clear()
        console.print("[dim]  ✓ History cleared.[/dim]")
    elif cmd == "/config":
        show_config()
    elif cmd == "/model":
        cfg = _model_picker(cfg)
    elif cmd == "/keys":
        cfg = _keys_wizard(cfg)
    else:
        console.print(f"[bold red]Unknown:[/] {cmd} — type [bold]/help[/]")

    return history, cfg

# ---------------------------------------------------------------------------
# Main REPL
# ---------------------------------------------------------------------------

@app.callback(invoke_without_command=True)
def chat(ctx: typer.Context) -> None:
    """Start an interactive AI coding session."""
    if ctx.invoked_subcommand is not None:
        return

    try:
        cfg = load_config()
    except Exception as exc:
        console.print(f"[bold red]Config error:[/] {exc}")
        raise typer.Exit(1)

    _print_welcome(cfg)
    history: list[dict] = []
    turn = 0

    while True:
        console.print(_status_bar(cfg, turn))
        try:
            user_input = console.input("[bold bright_cyan]❯[/] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold cyan]Goodbye! 👋[/]\n")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            try:
                history, cfg = _slash(user_input, history, cfg)
            except typer.Exit:
                break
            continue

        # Show user message
        ts = datetime.now().strftime("%H:%M")
        console.print(Panel(
            Text(user_input, style="white"),
            title=f"[bold green]You[/]  [dim]{ts}[/]",
            title_align="right",
            border_style="green",
            padding=(0, 2),
        ))

        turn += 1

        try:
            token_stream, history = run_agent(cfg, user_input, history)
            _stream_reply(token_stream, cfg)
        except (ValueError, ConnectionError, RuntimeError) as exc:
            console.print(Panel(
                Text(str(exc), style="bold red"),
                title="[bold red]Error[/]",
                border_style="red",
                padding=(0, 2),
            ))
        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted — /exit to quit.[/dim]")

# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------

@app.command("config")
def config_cmd(
    model:        Optional[str]   = typer.Option(None, "--model",        "-m"),
    api_key:      Optional[str]   = typer.Option(None, "--api-key",      "-k"),
    deepseek_key: Optional[str]   = typer.Option(None, "--deepseek-key", "-d"),
    base_url:     Optional[str]   = typer.Option(None, "--base-url"),
    max_tokens:   Optional[int]   = typer.Option(None, "--max-tokens"),
    temperature:  Optional[float] = typer.Option(None, "--temperature"),
    show:         bool            = typer.Option(False, "--show", "-s"),
) -> None:
    """View or update configuration."""
    any_set = any(v is not None for v in [model, api_key, deepseek_key, base_url, max_tokens, temperature])
    if show or not any_set:
        show_config()
        return
    if model and base_url is None:
        from ai_agent.config import get_base_url
        base_url = get_base_url(model)
    try:
        update_config(model=model, api_key=api_key, deepseek_api_key=deepseek_key,
                      base_url=base_url, max_tokens=max_tokens, temperature=temperature)
        console.print("[bold green]✓ Updated.[/]")
        show_config()
    except Exception as exc:
        console.print(f"[bold red]Failed:[/] {exc}")
        raise typer.Exit(1)


@app.command("models")
def models_cmd() -> None:
    """List all available models."""
    print_models_table(console)


@app.command("version")
def version_cmd() -> None:
    """Print version."""
    console.print(f"[bold bright_cyan]AI Coding Agent[/] [bold white]v{__version__}[/]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
