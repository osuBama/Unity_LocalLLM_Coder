# Local AI Orchestrator

Run two local LLMs on two GPUs as one system: a **primary model** that does the work, and a **memory model**
that maintains persistent project memory in the background, so long sessions stay accurate inside a small
context window. Drop-in compatible with the Ollama API.

```
client ──► orchestrator :8000 ──► Ollama A (GPU 1)  primary model
(OpenClaw,       │
 any Ollama      └─ memory, summaries, digests ──► Ollama B (GPU 2)  memory + embedding model
 client)
```

## Features

- **Dual-GPU by design**: each model pinned to its own card; background work never slows your answers.
- **Persistent memory**: human-readable Markdown, written only by validated code, never directly by the LLM.
- **Fits any context window**: compresses old tool output, then trims history only when it wouldn't fit, with
  session summaries standing in for what was cut.
- **Prompt-cache aware**: stable memory sits in the cached system prompt; everything else is placed so
  Ollama's cache keeps hitting.
- **Semantic search and history recall**: hybrid keyword + vector retrieval, plus excerpts from past
  conversations when they're relevant.
- **Reference libraries**: index documentation and code (e.g. Unity docs, packages) for the model to look
  up on demand through MCP tools, or as small automatic excerpts, without bloating memory or the context.
- **Adaptive thinking**: skips the thinking phase on simple turns.
- **Self-maintaining**: idle-time consolidation of duplicates, with backups and one-command restore.
- **Web console**: everything the CLI does, at `http://127.0.0.1:8000/ui`: both GPUs' status, chat, memory,
  reference libraries, settings, and evaluation runs with side-by-side comparison.
- **Evaluation harness**: replays your own sessions to compare settings on cost and accuracy, explains why
  each answer passed or failed, and builds its test set from your corrections and from facts in your
  recorded sessions.

## Requirements

- Two GPUs supported by [Ollama](https://ollama.com) (one works, without isolation)
- Python 3.11+
- Windows + NVIDIA for the setup script; Linux and AMD work with manual setup

## Installation

**Windows (recommended):**

```powershell
git clone <this repo> D:\AI
cd D:\AI
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

The script detects your GPUs, picks models for your VRAM, starts two pinned Ollama instances, measures
the largest context that fits each card, writes the config, starts the orchestrator, and (if present)
configures OpenClaw in WSL. It asks before changing anything outside the project folder and backs it up.
Re-run any time.

**Linux / AMD / manual:** see [Installation](docs/DOCUMENTATION.md#3-install) and
[pinned Ollama instances](docs/DOCUMENTATION.md#4-start-two-pinned-ollama-instances).

## Usage

Point any Ollama-native client at `http://127.0.0.1:8000` instead of `:11434`. Memory is added
automatically.

Open the console at **http://127.0.0.1:8000/ui** (or run `.\ai ui`) to see both GPUs, chat, browse and edit
memory, change settings and compare evaluation runs. From the terminal:

```powershell
.\ai status                  # instances, GPU residency, queue, health
.\ai chat -v                 # chat from the terminal; -v shows the memory used
.\ai memory show             # what the system remembers
.\ai memory search "query"   # search memory
.\ai docs add <folder> --name unity-6 --version 6000.0   # index documentation or code
.\ai eval run                # compare settings on your own sessions
```

On Linux, use `python -m app.cli` instead of `.\ai`.

Clients using the OpenAI-compatible `/v1` API are passed through **without** memory; choose "Ollama"
as the provider type.

## Configuration

Everything lives in `config/config.yaml`; the defaults work without changes. The console's Settings tab
edits the common ones safely (with a backup, keeping your comments). Settings you might tune:

| Setting | What it controls |
|---|---|
| `ollama.*.model`, `num_ctx` | models and context size per GPU (set by setup) |
| `memory.max_context_tokens` | how much memory is injected per prompt |
| `proxy.trim_target_ratio` | how deep history is cut when the window fills |
| `thinking.mode` | `auto` / `client` / `on` / `off` |
| `history_recall.min_similarity` | how readily past conversations are recalled |

## Documentation

Full documentation, covering setup by hand, how each feature works, tuning, the evaluation harness, API
reference and troubleshooting, is in [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md).
