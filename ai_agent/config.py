"""
config.py
---------
Handles all persistent configuration for the AI Coding Agent.

Config is stored at ~/.ai_coding_agent/config.json and managed
via Pydantic for validation. On first run the user is prompted
to supply an API key and preferred model.
"""

import json
from pathlib import Path

from pydantic import BaseModel, Field
from rich.console import Console
from rich.prompt import Prompt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CONFIG_DIR = Path.home() / ".ai_coding_agent"
CONFIG_FILE = CONFIG_DIR / "config.json"

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class AgentConfig(BaseModel):
    """All user-configurable settings, validated by Pydantic."""

    api_key: str = Field(..., description="OpenAI API key")
    model: str = Field(default="gpt-4o", description="OpenAI model name")
    base_url: str = Field(
        default="https://api.openai.com/v1",
        description="API base URL (useful for proxies / local models)",
    )
    max_tokens: int = Field(default=4096, description="Max tokens per response")
    temperature: float = Field(default=0.2, description="Sampling temperature")

# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _ensure_config_dir() -> None:
    """Create the config directory if it doesn't exist."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> AgentConfig:
    """
    Load config from disk.  If the file doesn't exist, run first-time setup
    so the user can enter their credentials interactively.
    """
    if CONFIG_FILE.exists():
        try:
            raw = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            return AgentConfig(**raw)
        except Exception as exc:
            console = Console()
            console.print(
                f"[bold red]Config file is corrupted:[/] {exc}\n"
                "Starting first-time setup…"
            )

    return _first_time_setup()


def save_config(cfg: AgentConfig) -> None:
    """Persist an AgentConfig instance to disk."""
    _ensure_config_dir()
    CONFIG_FILE.write_text(
        cfg.model_dump_json(indent=2), encoding="utf-8"
    )


def _first_time_setup() -> AgentConfig:
    """
    Interactive wizard run the very first time (or when config is missing).
    Prompts the user for the minimum required settings.
    """
    console = Console()
    console.print(
        "\n[bold cyan]Welcome to AI Coding Agent![/]\n"
        "Let's get you set up. Your settings will be saved to "
        f"[dim]{CONFIG_FILE}[/]\n"
    )

    api_key = Prompt.ask("[bold yellow]Enter your OpenAI API key[/]", password=True)
    model = Prompt.ask(
        "[bold yellow]Preferred model[/]",
        default="gpt-4o",
    )

    cfg = AgentConfig(api_key=api_key, model=model)
    save_config(cfg)
    console.print("[bold green]✓ Config saved![/]\n")
    return cfg


def update_config(**kwargs) -> AgentConfig:
    """
    Load existing config, apply keyword-argument overrides, and save.

    Example
    -------
    update_config(model="gpt-4-turbo", api_key="sk-...")
    """
    cfg = load_config()
    updated = cfg.model_copy(update={k: v for k, v in kwargs.items() if v is not None})
    save_config(updated)
    return updated


def show_config() -> AgentConfig:
    """Print the current config (masking the API key) and return it."""
    cfg = load_config()
    console = Console()
    console.print("\n[bold cyan]Current configuration[/]")
    console.print(f"  [dim]Config file :[/] {CONFIG_FILE}")
    console.print(f"  [dim]Model       :[/] {cfg.model}")
    console.print(f"  [dim]Base URL    :[/] {cfg.base_url}")
    console.print(f"  [dim]Max tokens  :[/] {cfg.max_tokens}")
    console.print(f"  [dim]Temperature :[/] {cfg.temperature}")
    # Show only first/last 4 chars of the key for safety
    masked = cfg.api_key[:4] + "****" + cfg.api_key[-4:] if len(cfg.api_key) > 8 else "****"
    console.print(f"  [dim]API key     :[/] {masked}\n")
    return cfg
