"""
llm.py
------
OpenAI-compatible client wrapper and the core agentic loop.

Works with both OpenAI and DeepSeek (DeepSeek uses the same
OpenAI SDK — just a different base_url and api_key).

Agentic loop:
  1. Send full conversation history to the model.
  2. If the model returns tool calls → execute them, append results, loop.
  3. If the model returns plain text → task complete, return to caller.
"""

import json
from typing import Generator

from openai import OpenAI, AuthenticationError, APIConnectionError, RateLimitError
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

from ai_agent.config import AgentConfig, get_model_info, get_base_url
from ai_agent.tools import TOOLS_SCHEMA, execute_tool

console = Console()

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert AI coding assistant running inside a developer's terminal.

Your capabilities:
- Read, create, edit, and delete files on the local file system.
- List directory trees to understand project structure.
- Write clean, well-commented, production-quality code.
- Explain code, debug issues, suggest improvements.

Behaviour rules:
1. At the start of every new task, call list_files('.') to understand the
   project layout before making any changes.
2. Always call read_file before editing a file — never guess the content.
3. Be concise in text replies; put the detail in code and comments.
4. Work through multi-step tasks tool-by-tool, then give a short summary.
5. If a model does not support tool/function calling (e.g. o1-preview,
   deepseek-reasoner) just answer in plain text — do not attempt tool calls.
"""

# ---------------------------------------------------------------------------
# Models that do NOT support function/tool calling
# ---------------------------------------------------------------------------

NO_TOOL_MODELS = {
    "o1-preview",
    "o1-mini",
    "deepseek-reasoner",
}


# ---------------------------------------------------------------------------
# Client factory — auto-selects base_url and api_key by model provider
# ---------------------------------------------------------------------------

def build_client(cfg: AgentConfig) -> OpenAI:
    """
    Create an OpenAI-SDK client pointed at the right provider.

    - OpenAI models  → api.openai.com,   uses cfg.api_key
    - DeepSeek models → api.deepseek.com, uses cfg.deepseek_api_key
    """
    api_key = cfg.active_api_key()
    base_url = get_base_url(cfg.model)

    if not api_key:
        info = get_model_info(cfg.model)
        provider = info["provider"] if info else "OpenAI"
        key_field = "OpenAI API key" if provider == "OpenAI" else "DeepSeek API key"
        raise ValueError(
            f"No {key_field} set for model '{cfg.model}'. "
            f"Run `aicoder config` to add it."
        )

    return OpenAI(api_key=api_key, base_url=base_url)


# ---------------------------------------------------------------------------
# Single model call (with Rich spinner)
# ---------------------------------------------------------------------------

def _chat_completion(client: OpenAI, cfg: AgentConfig, messages: list[dict]):
    """
    Call the Chat Completions endpoint and return the first choice message.
    Shows a spinner while waiting so the terminal never looks frozen.
    """
    use_tools = cfg.model not in NO_TOOL_MODELS

    # DeepSeek Reasoner doesn't accept a temperature parameter
    no_temp_models = {"o1-preview", "o1-mini", "deepseek-reasoner"}
    kwargs: dict = {
        "model": cfg.model,
        "messages": messages,
        "max_tokens": cfg.max_tokens,
    }
    if cfg.model not in no_temp_models:
        kwargs["temperature"] = cfg.temperature
    if use_tools:
        kwargs["tools"] = TOOLS_SCHEMA
        kwargs["tool_choice"] = "auto"

    spinner = Spinner("dots2", text=Text("  Thinking…", style="bold cyan"))
    with Live(spinner, console=console, refresh_per_second=15, transient=True):
        try:
            response = client.chat.completions.create(**kwargs)
        except AuthenticationError:
            raise ValueError(
                "Invalid API key. Run `aicoder config` to update your credentials."
            )
        except APIConnectionError as exc:
            raise ConnectionError(
                f"Could not reach the API ({cfg.base_url}): {exc}"
            ) from exc
        except RateLimitError:
            raise RuntimeError(
                "Rate limit hit. Wait a moment or switch to a cheaper model (/model)."
            )

    return response.choices[0].message


# ---------------------------------------------------------------------------
# Agentic loop
# ---------------------------------------------------------------------------

def run_agent(
    cfg: AgentConfig,
    user_message: str,
    history: list[dict],
) -> tuple[str, list[dict]]:
    """
    Run the full agentic loop for one user turn.

    Parameters
    ----------
    cfg          : AgentConfig — current settings
    user_message : str         — what the user typed
    history      : list[dict]  — full conversation history (mutated in-place)

    Returns
    -------
    (final_reply, updated_history)
    """
    client = build_client(cfg)

    # Ensure system message is at position 0
    if not history or history[0].get("role") != "system":
        history.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

    history.append({"role": "user", "content": user_message})

    max_iterations = 20
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        message = _chat_completion(client, cfg, history)

        # ── Case 1: model wants to call tools ──────────────────────────────
        if getattr(message, "tool_calls", None):
            history.append(_message_to_dict(message))

            for tool_call in message.tool_calls:
                name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                console.print(
                    f"\n[bold magenta]⚙  Tool call:[/] [cyan]{name}[/] "
                    f"[dim]{_summarise_args(args)}[/dim]"
                )

                result = execute_tool(name, args)

                history.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": name,
                        "content": result,
                    }
                )

        # ── Case 2: plain text reply — done ───────────────────────────────
        else:
            final_text = message.content or ""
            history.append({"role": "assistant", "content": final_text})
            return final_text, history

    fallback = (
        "I hit the maximum tool-call limit (20). "
        "Please break your request into smaller steps."
    )
    history.append({"role": "assistant", "content": fallback})
    return fallback, history


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _message_to_dict(message) -> dict:
    """Convert an OpenAI ChatCompletionMessage object to a serialisable dict."""
    d: dict = {"role": message.role}
    if message.content:
        d["content"] = message.content
    if getattr(message, "tool_calls", None):
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in message.tool_calls
        ]
    return d


def _summarise_args(args: dict) -> str:
    """Short human-readable summary of tool call arguments."""
    parts = []
    for k, v in args.items():
        if isinstance(v, str) and len(v) > 60:
            v = v[:57] + "…"
        parts.append(f"{k}={repr(v)}")
    return ", ".join(parts)
