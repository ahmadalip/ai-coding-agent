"""
llm.py
------
OpenAI client wrapper and the core agentic loop.

The agentic loop works as follows:
  1. Send the conversation history (system + messages) to the model.
  2. If the model returns tool calls, execute each one, append the results,
     and loop back to step 1.
  3. If the model returns a plain text message (no tool calls), the task is
     complete — return that message to the caller.

A Rich spinner is shown while waiting for each model response so the
terminal doesn't look frozen.
"""

import json
from typing import Generator

from openai import OpenAI, AuthenticationError, APIConnectionError, RateLimitError
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

from ai_agent.config import AgentConfig
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
2. Always call read_file before editing a file so you have the exact current
   content.
3. Be concise in your text replies — save detail for code and comments.
4. When a task involves multiple steps, work through them one by one using
   the available tools, then give a short summary of what you did.
5. Never guess file contents — read first, then act.
"""

# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

def build_client(cfg: AgentConfig) -> OpenAI:
    """Create an OpenAI client from the current AgentConfig."""
    return OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)


# ---------------------------------------------------------------------------
# Single model call (with spinner)
# ---------------------------------------------------------------------------

def _chat_completion(client: OpenAI, cfg: AgentConfig, messages: list[dict]) -> dict:
    """
    Call the OpenAI Chat Completions API and return the first choice message
    as a plain dict.  Shows a spinner while waiting.
    """
    spinner = Spinner("dots", text=Text(" Thinking…", style="bold cyan"))

    with Live(spinner, console=console, refresh_per_second=12, transient=True):
        try:
            response = client.chat.completions.create(
                model=cfg.model,
                messages=messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
            )
        except AuthenticationError:
            raise ValueError(
                "Invalid API key. Run `aicoder config` to update your credentials."
            )
        except APIConnectionError as exc:
            raise ConnectionError(f"Could not reach the OpenAI API: {exc}") from exc
        except RateLimitError:
            raise RuntimeError(
                "Rate limit hit. Wait a moment or switch to a different model."
            )

    # Return as a plain dict so we can append it to messages easily
    choice = response.choices[0].message
    return choice


# ---------------------------------------------------------------------------
# Agentic loop
# ---------------------------------------------------------------------------

def run_agent(
    cfg: AgentConfig,
    user_message: str,
    history: list[dict],
) -> tuple[str, list[dict]]:
    """
    Run the full agentic loop for a single user turn.

    Parameters
    ----------
    cfg          : AgentConfig  — current settings
    user_message : str          — what the user just typed
    history      : list[dict]   — conversation history (mutated in-place)

    Returns
    -------
    (final_reply, updated_history)
        final_reply   : the model's last plain-text message
        updated_history : the full updated history list
    """
    client = build_client(cfg)

    # Ensure system message is present
    if not history or history[0].get("role") != "system":
        history.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

    # Append the new user message
    history.append({"role": "user", "content": user_message})

    # -----------------------------------------------------------------------
    # Agentic loop — keep going until the model stops calling tools
    # -----------------------------------------------------------------------
    max_iterations = 20  # safety cap to prevent runaway loops
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        message = _chat_completion(client, cfg, history)

        # -- Case 1: model wants to call tools --------------------------------
        if message.tool_calls:
            # Append the assistant message (with tool_calls) to history
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

                # Append tool result as a "tool" role message
                history.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": name,
                        "content": result,
                    }
                )

        # -- Case 2: model returned a plain text reply — we're done ----------
        else:
            final_text = message.content or ""
            history.append({"role": "assistant", "content": final_text})
            return final_text, history

    # Safety: if we hit the iteration cap return whatever the last message was
    fallback = "I reached the maximum number of tool-call iterations. Please refine your request."
    history.append({"role": "assistant", "content": fallback})
    return fallback, history


# ---------------------------------------------------------------------------
# Streaming variant (for long responses)
# ---------------------------------------------------------------------------

def stream_agent_reply(
    cfg: AgentConfig,
    messages: list[dict],
) -> Generator[str, None, None]:
    """
    Yield text tokens as they stream in from the model.
    Used for pure text responses (no tool calls) to give a live typing feel.
    """
    client = build_client(cfg)
    with client.chat.completions.stream(
        model=cfg.model,
        messages=messages,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
    ) as stream:
        for chunk in stream.text_stream:
            yield chunk


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _message_to_dict(message) -> dict:
    """Convert an OpenAI ChatCompletionMessage object to a plain dict."""
    d: dict = {"role": message.role}
    if message.content:
        d["content"] = message.content
    if message.tool_calls:
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
    """Return a short human-readable summary of tool arguments."""
    parts = []
    for k, v in args.items():
        if isinstance(v, str) and len(v) > 60:
            v = v[:57] + "…"
        parts.append(f"{k}={repr(v)}")
    return ", ".join(parts)
