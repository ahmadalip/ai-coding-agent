"""
config.py
---------
Handles all persistent configuration for the AI Coding Agent.

Config is stored at ~/.ai_coding_agent/config.json and managed
via Pydantic for validation. On first run the user is prompted
to supply an API key and preferred model.

Supported providers: OpenAI, DeepSeek
"""

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table
from rich import box

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CONFIG_DIR = Path.home() / ".ai_coding_agent"
CONFIG_FILE = CONFIG_DIR / "config.json"

# ---------------------------------------------------------------------------
# Models Registry
# All supported models across providers.
# Each entry: (display_name, model_id, provider, base_url, notes)
# ---------------------------------------------------------------------------

MODELS: list[dict] = [
    # ── OpenAI ──────────────────────────────────────────────────────────────
    {
        "provider": "OpenAI",
        "name": "GPT-4o",
        "id": "gpt-4o",
        "base_url": "https://api.openai.com/v1",
        "note": "Best overall — fast + smart",
        "key_field": "api_key",
    },
    {
        "provider": "OpenAI",
        "name": "GPT-4o Mini",
        "id": "gpt-4o-mini",
        "base_url": "https://api.openai.com/v1",
        "note": "Cheapest OpenAI option",
        "key_field": "api_key",
    },
    {
        "provider": "OpenAI",
        "name": "GPT-4 Turbo",
        "id": "gpt-4-turbo",
        "base_url": "https://api.openai.com/v1",
        "note": "128k context window",
        "key_field": "api_key",
    },
    {
        "provider": "OpenAI",
        "name": "GPT-3.5 Turbo",
        "id": "gpt-3.5-turbo",
        "base_url": "https://api.openai.com/v1",
        "note": "Fast & very cheap",
        "key_field": "api_key",
    },
    {
        "provider": "OpenAI",
        "name": "o1 Preview",
        "id": "o1-preview",
        "base_url": "https://api.openai.com/v1",
        "note": "Advanced reasoning model",
        "key_field": "api_key",
    },
    {
        "provider": "OpenAI",
        "name": "o1 Mini",
        "id": "o1-mini",
        "base_url": "https://api.openai.com/v1",
        "note": "Fast reasoning, lower cost",
        "key_field": "api_key",
    },
    # ── DeepSeek ─────────────────────────────────────────────────────────────
    {
        "provider": "DeepSeek",
        "name": "DeepSeek Chat (V3)",
        "id": "deepseek-chat",
        "base_url": "https://api.deepseek.com/v1",
        "note": "Best DeepSeek model — very cheap",
        "key_field": "deepseek_api_key",
    },
    {
        "provider": "DeepSeek",
        "name": "DeepSeek Reasoner (R1)",
        "id": "deepseek-reasoner",
        "base_url": "https://api.deepseek.com/v1",
        "note": "Chain-of-thought reasoning",
        "key_field": "deepseek_api_key",
    },
    {
        "provider": "DeepSeek",
        "name": "DeepSeek Coder",
        "id": "deepseek-coder",
        "base_url": "https://api.deepseek.com/v1",
        "note": "Optimised for coding tasks",
        "key_field": "deepseek_api_key",
    },
]

# Lookup helpers
def get_model_info(model_id: str) -> Optional[dict]:
    """Return the registry entry for a model id, or None."""
    return next((m for m in MODELS if m["id"] == model_id), None)


def get_provider(model_id: str) -> str:
    """Return provider name for a model id ('OpenAI' or 'DeepSeek')."""
    info = get_model_info(model_id)
    return info["provider"] if info else "OpenAI"


def get_base_url(model_id: str) -> str:
    """Return the correct base URL for a given model id."""
    info = get_model_info(model_id)
    return info["base_url"] if info else "https://api.openai.com/v1"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class AgentConfig(BaseModel):
    """All user-configurable settings, validated by Pydantic."""

    api_key: str = Field(default="", description="OpenAI API key")
    deepseek_api_key: str = Field(default="", description="DeepSeek API key")
    model: str = Field(default="gpt-4o", description="Active model id")
    base_url: str = Field(
        default="https://api.openai.com/v1",
        description="API base URL — auto-set when switching models",
    )
    max_tokens: int = Field(default=4096, description="Max tokens per response")
    temperature: float = Field(default=0.2, description="Sampling temperature")

    def active_api_key(self) -> str:
        """Return the correct API key for the currently active model."""
        info = get_model_info(self.model)
        if info and info["key_field"] == "deepseek_api_key":
            return self.deepseek_api_key
        return self.api_key


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _ensure_config_dir() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> AgentConfig:
    """Load config from disk or run first-time setup."""
    if CONFIG_FILE.exists():
        try:
            raw = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            return AgentConfig(**raw)
        except Exception as exc:
            console = Console()
            console.print(f"[bold red]Config corrupted:[/] {exc}\nStarting setup…")

    return _first_time_setup()


def save_config(cfg: AgentConfig) -> None:
    """Persist config to disk."""
    _ensure_config_dir()
    CONFIG_FILE.write_text(cfg.model_dump_json(indent=2), encoding="utf-8")


def update_config(**kwargs) -> AgentConfig:
    """Load, patch with kwargs, save and return updated config."""
    cfg = load_config()
    updated = cfg.model_copy(update={k: v for k, v in kwargs.items() if v is not None})
    save_config(updated)
    return updated


# ---------------------------------------------------------------------------
# First-time setup wizard
# ---------------------------------------------------------------------------

def _first_time_setup() -> AgentConfig:
    console = Console()
    console.print(
        "\n[bold cyan]╔══════════════════════════════════════╗[/]"
        "\n[bold cyan]║   Welcome to AI Coding Agent! 🤖     ║[/]"
        "\n[bold cyan]╚══════════════════════════════════════╝[/]\n"
    )
    console.print(f"[dim]Settings will be saved to {CONFIG_FILE}[/]\n")

    # Show model table so user can decide
    print_models_table(console)

    openai_key = Prompt.ask(
        "\n[bold yellow]OpenAI API key[/] [dim](press Enter to skip)[/]",
        password=True,
        default="",
    )
    deepseek_key = Prompt.ask(
        "[bold yellow]DeepSeek API key[/] [dim](press Enter to skip)[/]",
        password=True,
        default="",
    )

    # Default model based on which key was provided
    if openai_key:
        default_model = "gpt-4o"
    elif deepseek_key:
        default_model = "deepseek-chat"
    else:
        default_model = "gpt-4o"

    model_id = Prompt.ask(
        "[bold yellow]Default model ID[/]",
        default=default_model,
    )

    info = get_model_info(model_id)
    base_url = info["base_url"] if info else "https://api.openai.com/v1"

    cfg = AgentConfig(
        api_key=openai_key,
        deepseek_api_key=deepseek_key,
        model=model_id,
        base_url=base_url,
    )
    save_config(cfg)
    console.print("\n[bold green]✓ Config saved! Let's code.[/]\n")
    return cfg


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_models_table(console: Optional[Console] = None) -> None:
    """Print a nicely formatted table of all available models."""
    if console is None:
        console = Console()

    table = Table(
        title="Available Models",
        box=box.ROUNDED,
        border_style="cyan",
        header_style="bold cyan",
        show_lines=True,
    )
    table.add_column("#", style="dim", width=3, justify="right")
    table.add_column("Provider", style="bold", width=10)
    table.add_column("Name", width=24)
    table.add_column("Model ID", style="cyan", width=22)
    table.add_column("Notes", style="dim")

    provider_styles = {"OpenAI": "green", "DeepSeek": "magenta"}

    for i, m in enumerate(MODELS, start=1):
        style = provider_styles.get(m["provider"], "white")
        table.add_row(
            str(i),
            f"[{style}]{m['provider']}[/]",
            m["name"],
            m["id"],
            m["note"],
        )

    console.print(table)


def show_config() -> AgentConfig:
    """Print current config (API keys masked) and return it."""
    cfg = load_config()
    console = Console()

    def mask(key: str) -> str:
        if not key:
            return "[dim]not set[/dim]"
        return key[:4] + "••••" + key[-4:] if len(key) > 8 else "••••"

    info = get_model_info(cfg.model)
    provider = info["provider"] if info else "Unknown"
    provider_color = "green" if provider == "OpenAI" else "magenta"

    table = Table(box=box.ROUNDED, border_style="cyan", show_header=False, padding=(0, 2))
    table.add_column("Key", style="dim", width=18)
    table.add_column("Value")

    table.add_row("Config file", str(CONFIG_FILE))
    table.add_row("Active model", f"[cyan]{cfg.model}[/]")
    table.add_row("Provider", f"[{provider_color}]{provider}[/]")
    table.add_row("Base URL", cfg.base_url)
    table.add_row("Max tokens", str(cfg.max_tokens))
    table.add_row("Temperature", str(cfg.temperature))
    table.add_row("OpenAI key", mask(cfg.api_key))
    table.add_row("DeepSeek key", mask(cfg.deepseek_api_key))

    console.print("\n[bold cyan]Current Configuration[/]")
    console.print(table)
    console.print()
    return cfg
