"""
llm.py
------
OpenAI-compatible client + agentic loop with streaming final reply.

Flow:
  1. Send history to model.
  2. Tool calls → execute with animated spinners → loop.
  3. Final text reply → stream token-by-token via a generator.
"""

import json
from typing import Generator, Iterator

from openai import (
    OpenAI,
    AuthenticationError,
    APIConnectionError,
    RateLimitError,
    BadRequestError,
)
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

Capabilities:
- Read, create, edit, and delete files on the local file system.
- List directory trees to understand project structure.
- Write clean, well-commented, production-quality code.
- Explain code, debug issues, suggest improvements.

Rules:
1. At the start of every new task call list_files('.') first.
2. Always call read_file before editing — never guess file content.
3. In your FINAL text reply: never include raw code blocks.
   Instead describe what you did and which files were changed.
   All code belongs in files, not in the chat.
4. Work tool-by-tool for multi-step tasks, then give a short plain-text summary.
5. Models without tool support (o1-preview, deepseek-reasoner) answer in plain text.
"""

# ---------------------------------------------------------------------------
# Models with no tool / no temperature support
# ---------------------------------------------------------------------------

NO_TOOL_MODELS = {"o1-preview", "o1-mini", "deepseek-reasoner"}
NO_TEMP_MODELS = {"o1-preview", "o1-mini", "deepseek-reasoner"}

# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

def build_client(cfg: AgentConfig) -> OpenAI:
    api_key  = cfg.active_api_key()
    base_url = get_base_url(cfg.model)

    if not api_key:
        info     = get_model_info(cfg.model)
        provider = info["provider"] if info else "OpenAI"
        field    = "OpenAI API key" if provider == "OpenAI" else "DeepSeek API key"
        raise ValueError(f"No {field} for '{cfg.model}'. Run `aicoder config`.")

    return OpenAI(api_key=api_key, base_url=base_url)

# ---------------------------------------------------------------------------
# Single non-streaming call (used for tool-call turns only)
# ---------------------------------------------------------------------------

def _call(client: OpenAI, cfg: AgentConfig, messages: list[dict], with_tools: bool = True):
    """
    One blocking call — only used while the model is still issuing tool calls.
    Shows a 'Thinking…' spinner.
    """
    use_tools = with_tools and cfg.model not in NO_TOOL_MODELS

    def _kw(tools: bool) -> dict:
        kw: dict = {"model": cfg.model, "messages": messages}
        # Try max_completion_tokens first (newer SDK), fall back to max_tokens
        kw["max_completion_tokens"] = cfg.max_tokens
        if cfg.model not in NO_TEMP_MODELS:
            kw["temperature"] = cfg.temperature
        if tools:
            kw["tools"]       = TOOLS_SCHEMA
            kw["tool_choice"] = "auto"
        return kw

    spinner = Spinner("dots2", text=Text("  Thinking…", style="bold cyan"))
    with Live(spinner, console=console, refresh_per_second=15, transient=True):
        try:
            return client.chat.completions.create(**_kw(use_tools)).choices[0].message

        except BadRequestError as exc:
            err = str(exc).lower()
            if any(k in err for k in ("tool", "function", "max_completion_tokens")):
                kw = _kw(False)
                kw.pop("max_completion_tokens", None)
                kw["max_tokens"] = cfg.max_tokens
                try:
                    return client.chat.completions.create(**kw).choices[0].message
                except BadRequestError as exc2:
                    raise RuntimeError(f"Model rejected request: {exc2}") from exc2
            raise RuntimeError(f"Bad request: {exc}") from exc

        except AuthenticationError:
            raise ValueError("Invalid API key. Run `aicoder config`.")
        except APIConnectionError as exc:
            raise ConnectionError(f"Cannot reach API: {exc}") from exc
        except RateLimitError:
            raise RuntimeError("Rate limit hit. Try /model to switch.")

# ---------------------------------------------------------------------------
# Streaming call — yields text tokens for the final reply
# ---------------------------------------------------------------------------

def _stream(client: OpenAI, cfg: AgentConfig, messages: list[dict]) -> Iterator[str]:
    """
    Stream the final assistant reply token-by-token.
    Only called when the model returns no tool calls.
    """
    kw: dict = {
        "model":    cfg.model,
        "messages": messages,
        "stream":   True,
    }
    # Use whichever token param the SDK accepts
    try:
        kw["max_completion_tokens"] = cfg.max_tokens
        if cfg.model not in NO_TEMP_MODELS:
            kw["temperature"] = cfg.temperature

        for chunk in client.chat.completions.create(**kw):
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    except BadRequestError:
        # Retry with legacy max_tokens
        kw.pop("max_completion_tokens", None)
        kw["max_tokens"] = cfg.max_tokens
        for chunk in client.chat.completions.create(**kw):
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    except AuthenticationError:
        raise ValueError("Invalid API key. Run `aicoder config`.")
    except APIConnectionError as exc:
        raise ConnectionError(f"Cannot reach API: {exc}") from exc
    except RateLimitError:
        raise RuntimeError("Rate limit hit. Try /model to switch.")

# ---------------------------------------------------------------------------
# Agentic loop
# ---------------------------------------------------------------------------

def run_agent(
    cfg: AgentConfig,
    user_message: str,
    history: list[dict],
) -> tuple[Iterator[str], list[dict]]:
    """
    Run the agentic loop.

    Returns
    -------
    (token_stream, updated_history)
        token_stream  : iterator that yields text tokens of the final reply
        updated_history : history with all turns appended
    """
    client = build_client(cfg)

    if not history or history[0].get("role") != "system":
        history.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

    history.append({"role": "user", "content": user_message})

    max_iterations = 20

    for iteration in range(max_iterations):
        message = _call(client, cfg, history)

        # ── Tool calls ──────────────────────────────────────────────────────
        if getattr(message, "tool_calls", None):
            history.append(_to_dict(message))

            for tc in message.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                # Tool header line
                file_target = (
                    args.get("file_path")
                    or args.get("directory")
                    or ""
                )
                console.print(
                    Text.assemble(
                        ("\n  ⚙  ", "bold magenta"),
                        (name.replace("_", " ").title(), "bold cyan"),
                        (" → ", "dim"),
                        (file_target, "cyan"),
                    )
                )

                result = execute_tool(name, args)

                history.append({
                    "role":        "tool",
                    "tool_call_id": tc.id,
                    "name":        name,
                    "content":     result,
                })

        # ── Final text reply — stream it ────────────────────────────────────
        else:
            # We need to collect the full text too so we can append to history
            def _collecting_stream() -> Iterator[str]:
                collected: list[str] = []
                for token in _stream(client, cfg, history):
                    collected.append(token)
                    yield token
                history.append({
                    "role":    "assistant",
                    "content": "".join(collected),
                })

            return _collecting_stream(), history

    # Safety cap
    fallback = "Reached max tool iterations. Please break the task into smaller steps."
    history.append({"role": "assistant", "content": fallback})

    def _fallback_stream() -> Iterator[str]:
        yield fallback

    return _fallback_stream(), history

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_dict(message) -> dict:
    d: dict = {"role": message.role}
    if message.content:
        d["content"] = message.content
    if getattr(message, "tool_calls", None):
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name":      tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in message.tool_calls
        ]
    return d
