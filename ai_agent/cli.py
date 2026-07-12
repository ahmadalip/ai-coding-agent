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

from datetime import datetime
from typing import Optional

import typer
from rich import box
from rich.columns import Columns
from rich.console import Console
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
    get_model_info,
    load_config,
    print_models_table,
    save_config,
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

# ---------------------------------------------------------------------------
# Provider colour helper
# ---------------------------------------------------------------------------

PROVIDER_STYLE = {"OpenAI": "green", "DeepSeek": "magenta"}


def _provider_badge(model_id: str) -> str:
    info = get_model_info(model_id)
    if not info:
        return f"[dim]{model_id}[/dim]"
    style = PROVIDER_STYLE.get(info["provider"], "white")
    return f"[{style}]{info['provider']}[/] [cyan]{model_id}[/]"


# ---------------------------------------------------------------------------
# Welcome banner
# ---------------------------------------------------------------------------

BANNER = r"""[bold cyan]
   ___  ____   ___           __
  / _ |/  _/  / __/__  ___  / /__ ____
 / __ |/ /   / _// _ \/ _ \/ / _ `/ _ \
/_/ |_/___/ /___/\___/\_,_/_/\_,_/_//_/
[/bold cyan]"""


def _print_welcome(cfg: AgentConfig) -> None:
    info = get_model_info(cfg.model)
    provider = info["provider"] if info else "Custom"
    provider_style = PROVIDER_STYLE.get(provider, "white")
    model_note = info["note"] if info else ""

    console.print(BANNER)
    console.print(
        Panel(
            Columns(
                [
                    Text.from_markup(
                        f"[bold]Model   :[/] [{provider_style}]{cfg.model}[/]  "
                        f"[dim]({model_note})[/dim]"
                    ),
                    Text.from_markup(
                        f"[bold]Provider:[/] [{provider_style}]{provider}[/]"
                    ),
                    Text.from_markup(
                        "[dim]Type [bold]/help[/bold] for commands[/dim]"
                    ),
                ],
                equal=False,
                expand=False,
            ),
            border_style="cyan",
            padding=(0, 2),
        )
    )
    console.print()


# ---------------------------------------------------------------------------
# Interactive /model picker
# ---------------------------------------------------------------------------

def _model_picker(cfg: AgentConfig) -> AgentConfig:
    """
    Show a numbered table of all models and let the user pick by number.
    Returns the updated config.
    """
    console.print()
    print_models_table(console)
    console.print()

    # Find the current model's index for the default prompt
    current_idx = next(
        (str(i + 1) for i, m in enumerate(MODELS) if m["id"] == cfg.model),
        "1",
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

    # Check the right key exists for this provider
    if provider == "DeepSeek" and not cfg.deepseek_api_key:
        console.print(
            f"\n[bold yellow]DeepSeek key not set.[/] "
            "Enter it now or press Enter to cancel."
        )
        key = Prompt.ask("[bold yellow]DeepSeek API key[/]", password=True, default="")
        if not key:
            console.print("[dim]Model switch cancelled.[/dim]")
            return cfg
        cfg = update_config(deepseek_api_key=key)

    elif provider == "OpenAI" and not cfg.api_key:
        console.print(
            "\n[bold yellow]OpenAI key not set.[/] "
            "Enter it now or press Enter to cancel."
        )
        key = Prompt.ask("[bold yellow]OpenAI API key[/]", password=True, default="")
        if not key:
            console.print("[dim]Model switch cancelled.[/dim]")
            return cfg
        cfg = update_config(api_key=key)

    # Switch the model
    cfg = update_config(model=chosen["id"], base_url=chosen["base_url"])

    console.print(
        f"\n[bold green]✓ Switched to[/] [{provider_style}]{chosen['provider']}[/] "
        f"[cyan]{chosen['id']}[/] [dim]— {chosen['note']}[/dim]\n"
    )
    return cfg


# ---------------------------------------------------------------------------
# Slash command handler
# ---------------------------------------------------------------------------

SLASH_HELP_TEXT = """\
[bold cyan]Slash Commands[/]

  [bold]/help[/]         Show this message
  [bold]/model[/]        Pick a model from a numbered list (OpenAI + DeepSeek)
  [bold]/clear[/]        Wipe conversation history
  [bold]/config[/]       Show current configuration
  [bold]/keys[/]         Add or update API keys
  [bold]/exit[/]         Leave the chat  (also [bold]/quit[/])
"""


def _handle_slash(
    command: str,
    history: list[dict],
    cfg: AgentConfig,
) -> tuple[list[dict], AgentConfig]:
    parts = command.strip().split(maxsplit=1)
    cmd = parts[0].lower()

    if cmd in ("/exit", "/quit"):
        console.print("\n[bold cyan]Goodbye! Happy coding. 👋[/]\n")
        raise typer.Exit()

    elif cmd == "/help":
        console.print(
            Panel(SLASH_HELP_TEXT, border_style="dim", padding=(0, 2), title="Help")
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


def _update_keys_wizard(cfg: AgentConfig) -> AgentConfig:
    """Prompt the user to update one or both API keys."""
    console.print("\n[bold cyan]Update API Keys[/] [dim](press Enter to keep current)[/]\n")
    openai_key = Prompt.ask(
        "[bold yellow]OpenAI API key[/]", password=True, default=cfg.api_key or ""
    )
    deepseek_key = Prompt.ask(
        "[bold yellow]DeepSeek API key[/]", password=True, default=cfg.deepseek_api_key or ""
    )
    cfg = update_config(api_key=openai_key or cfg.api_key,
                        deepseek_api_key=deepseek_key or cfg.deepseek_api_key)
    console.print("[bold green]✓ Keys updated.[/]\n")
    return cfg


# ---------------------------------------------------------------------------
# Chat message rendering
# ---------------------------------------------------------------------------

def _render_user_msg(text: str) -> None:
    """Render the user's message in a right-aligned style."""
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
    """Render the assistant's Markdown reply in a styled panel."""
    info = get_model_info(model_id)
    provider = info["provider"] if info else "AI"
    provider_style = PROVIDER_STYLE.get(provider, "cyan")
    ts = datetime.now().strftime("%H:%M")

    console.print()
    console.print(
        Panel(
            Markdown(reply),
            title=f"[bold {provider_style}]✦ {provider} / {model_id}[/]  [dim]{ts}[/]",
            title_align="left",
            border_style=provider_style,
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
# Status bar (shown above the input prompt)
# ---------------------------------------------------------------------------

def _print_status_bar(cfg: AgentConfig, turn: int) -> None:
    info = get_model_info(cfg.model)
    provider = info["provider"] if info else "?"
    provider_style = PROVIDER_STYLE.get(provider, "white")

    bar = (
        f"[{provider_style}]●[/] [cyan]{cfg.model}[/]  "
        f"[dim]│[/]  [dim]turn {turn}[/]  "
        f"[dim]│[/]  [dim]temp {cfg.temperature}[/]  "
        f"[dim]│[/]  [dim]/help for commands[/]"
    )
    console.print(Rule(Text.from_markup(bar), style="dim"))


# ---------------------------------------------------------------------------
# Default command — interactive chat
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
        _print_status_bar(cfg, turn)

        try:
            user_input = console.input("[bold green]You ❯[/bold green] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold cyan]Goodbye! Happy coding. 👋[/bold cyan]\n")
            break

        if not user_input:
            continue

        # ── Slash command ──────────────────────────────────────────────────
        if user_input.startswith("/"):
            try:
                history, cfg = _handle_slash(user_input, history, cfg)
            except typer.Exit:
                break
            continue

        # ── Send to agent ──────────────────────────────────────────────────
        turn += 1
        _render_user_msg(user_input)

        try:
            reply, history = run_agent(cfg, user_input, history)
            _render_assistant_msg(reply, cfg.model)
        except (ValueError, ConnectionError, RuntimeError) as exc:
            _render_error(str(exc))
        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted. Type /exit to quit.[/dim]")


# ---------------------------------------------------------------------------
# `aicoder config` sub-command
# ---------------------------------------------------------------------------

@app.command("config")
def config_cmd(
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Set active model (e.g. gpt-4o, deepseek-chat)."
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", "-k", help="Set your OpenAI API key."
    ),
    deepseek_key: Optional[str] = typer.Option(
        None, "--deepseek-key", "-d", help="Set your DeepSeek API key."
    ),
    base_url: Optional[str] = typer.Option(
        None, "--base-url", help="Override API base URL."
    ),
    max_tokens: Optional[int] = typer.Option(
        None, "--max-tokens", help="Maximum tokens per response."
    ),
    temperature: Optional[float] = typer.Option(
        None, "--temperature", help="Sampling temperature (0.0–2.0)."
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
      aicoder config --model deepseek-chat --deepseek-key sk-...
      aicoder config --model gpt-4o --api-key sk-...
      aicoder config --temperature 0.5 --max-tokens 8192
    """
    any_set = any(
        v is not None
        for v in [model, api_key, deepseek_key, base_url, max_tokens, temperature]
    )

    if show or not any_set:
        show_config()
        return

    # If switching to a known model, auto-set base_url
    if model and base_url is None:
        from ai_agent.config import get_base_url
        base_url = get_base_url(model)

    try:
        update_config(
            model=model,
            api_key=api_key,
            deepseek_api_key=deepseek_key,
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
# `aicoder models` sub-command
# ---------------------------------------------------------------------------

@app.command("models")
def models_cmd() -> None:
    """List all available models (OpenAI + DeepSeek)."""
    print_models_table(console)
    console.print(
        "[dim]Switch model inside chat with [bold]/model[/bold], "
        "or via [bold]aicoder config --model <id>[/bold][/dim]\n"
    )


# ---------------------------------------------------------------------------
# `aicoder version` sub-command
# ---------------------------------------------------------------------------

@app.command("version")
def version_cmd() -> None:
    """Print the version and exit."""
    console.print(
        f"[bold cyan]AI Coding Agent[/bold cyan] [bold]{__version__}[/bold]"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    app()


if __name__ == "__main__":
    main()
