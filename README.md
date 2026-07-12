# AI Coding Agent 🤖

A CLI-based AI pair programmer that runs entirely in your terminal.  
Powered by **OpenAI**, built with **Typer** + **Rich** — it can read, create, edit, and delete files in your project while you chat with it naturally.

---

## Features

- **Interactive chat** — conversational REPL right in your terminal
- **Agentic loop** — the AI keeps calling tools until your task is fully done, then gives you a clean summary
- **File system tools** — create, read, edit, delete files and list directory trees
- **Context awareness** — automatically scans your project structure at the start of each task
- **Safe deletes** — always asks for `y/n` confirmation before deleting anything
- **Persistent config** — API key and model saved to `~/.ai_coding_agent/config.json`
- **Beautiful output** — syntax-highlighted file previews, spinners, Markdown rendering

---

## Project Structure

```
ai-coding-agent/
├── ai_agent/
│   ├── __init__.py      # package + version
│   ├── cli.py           # Typer CLI entry point & chat REPL
│   ├── llm.py           # OpenAI client + agentic loop
│   ├── tools.py         # File-system tools + OpenAI schema
│   └── config.py        # Pydantic settings + config persistence
├── pyproject.toml       # Build config & entry-point declaration
├── requirements.txt     # Pinned dependencies
└── README.md
```

---

## Requirements

- Python **3.10** or newer
- An **OpenAI API key** — get one at <https://platform.openai.com/api-keys>

---

## Installation

### Option A — Install from GitHub (recommended for end users)

```bash
pip install git+https://github.com/YOUR_USERNAME/ai-coding-agent.git
```

### Option B — Clone and install locally (for development)

```bash
git clone https://github.com/YOUR_USERNAME/ai-coding-agent.git
cd ai-coding-agent
pip install .
```

### Option C — Editable install (for contributors)

```bash
git clone https://github.com/YOUR_USERNAME/ai-coding-agent.git
cd ai-coding-agent
pip install -e .
```

After any of the above, the `aicoder` command will be available in your shell.

---

## Quick Start

### 1. First run — enter your credentials

The first time you run `aicoder`, a setup wizard will ask for your API key and preferred model:

```
$ aicoder

Welcome to AI Coding Agent!
Enter your OpenAI API key: sk-...
Preferred model [gpt-4o]: gpt-4o
✓ Config saved!
```

Your settings are stored at `~/.ai_coding_agent/config.json`.

### 2. Start a chat session

```bash
aicoder
```

You'll see the welcome banner and a `You ❯` prompt. Start typing:

```
You ❯ Create a FastAPI app with a /health endpoint
```

The agent will:
1. Call `list_files('.')` to understand your project layout
2. Call `create_file(...)` to write the code
3. Return a Markdown summary of what it did

---

## Chat Slash Commands

Inside the chat session these slash commands are available:

| Command | Description |
|---|---|
| `/help` | Show all slash commands |
| `/model <name>` | Switch model mid-session (e.g. `/model gpt-4o-mini`) |
| `/clear` | Wipe the conversation history and start fresh |
| `/config` | Display current configuration |
| `/exit` or `/quit` | Leave the chat session |

---

## CLI Sub-commands

### `aicoder config` — view or update settings

```bash
# Show current config
aicoder config --show

# Change model
aicoder config --model gpt-4-turbo

# Update API key and model together
aicoder config --api-key sk-... --model gpt-4o

# Override API base URL (e.g. for a proxy or local model)
aicoder config --base-url http://localhost:11434/v1

# Tune generation parameters
aicoder config --max-tokens 8192 --temperature 0.1
```

### `aicoder version` — print version

```bash
aicoder version
# AI Coding Agent version 0.1.0
```

---

## Available AI Tools

The AI can use these tools autonomously during a task:

| Tool | What it does |
|---|---|
| `list_files(directory)` | Print a directory tree |
| `read_file(file_path)` | Read full file content |
| `create_file(file_path, content)` | Create a new file (parents auto-created) |
| `edit_file(file_path, old_snippet, new_snippet)` | Replace a snippet in an existing file |
| `delete_file(file_path)` | Delete a file — **requires your confirmation** |

---

## Example Session

```
$ aicoder

  ██████╗ ██████╗ ██████╗ ...
  AI Coding Agent  •  Type /help for commands  •  /exit to quit

─────────────────────────────────────────────────────
You ❯ Add a Dockerfile to this project

⚙  Tool call: list_files  directory='.'
  📁 Listing .
  ...

⚙  Tool call: create_file  file_path='Dockerfile'
  ✓ Created Dockerfile

╭─ Assistant ──────────────────────────────────────╮
│                                                  │
│  Done! I created a `Dockerfile` that:            │
│  - Uses `python:3.12-slim` as the base image     │
│  - Installs dependencies from `requirements.txt` │
│  - Exposes port 8000                             │
│  - Runs `uvicorn app.main:app`                   │
│                                                  │
╰──────────────────────────────────────────────────╯

You ❯ /exit
Goodbye! Happy coding.
```

---

## Configuration File

Settings are stored at `~/.ai_coding_agent/config.json`:

```json
{
  "api_key": "sk-...",
  "model": "gpt-4o",
  "base_url": "https://api.openai.com/v1",
  "max_tokens": 4096,
  "temperature": 0.2
}
```

You can edit this file directly or use `aicoder config` flags.

---

## Using a Local / Alternative Model

Point the agent at any OpenAI-compatible endpoint (Ollama, LM Studio, etc.):

```bash
aicoder config --base-url http://localhost:11434/v1 --model llama3
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Invalid API key` | Run `aicoder config --api-key sk-...` |
| `Could not reach the OpenAI API` | Check your internet connection or `--base-url` |
| `Rate limit hit` | Wait a moment or switch to `gpt-4o-mini` |
| `The snippet was not found` | The AI will re-read the file automatically |
| Config file corrupted | Delete `~/.ai_coding_agent/config.json` and re-run `aicoder` |

---

## License

MIT — see [LICENSE](LICENSE) for details.
