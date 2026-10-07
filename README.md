# Unity LocalLLM Coder

A local, two-GPU AI coding assistant for **building games**: Unity first, but nothing in it is engine-specific.
It sits between your agent (e.g. [OpenClaw](https://docs.openclaw.ai)) and Ollama, and gives a mid-size local
model what long agentic sessions need: **persistent project memory**, **the engine's documentation on
demand**, and **a context window that never silently overflows**.

```
OpenClaw / any       orchestrator :8000             Ollama A (GPU 1)  primary model: writes the code
Ollama client  ───►  memory, docs, context  ───►
                     management, console            Ollama B (GPU 2)  memory model + embeddings:
                                                                      remembers, summarises, indexes
```

The primary model only does the work. Everything else (deciding what to remember, summarising long
sessions, compressing old tool output, searching docs) happens on the second GPU, in the background.

## Why

Agentic game development means long sessions: dozens of tool calls, big files and logs, and an engine API
that changes between versions. A 14B model on a 12 GB card runs out of context fast and forgets yesterday
entirely. This project keeps it working:

- **Remembers your project across sessions:** Unity version, render pipeline, conventions, decisions,
  fixes that worked. Stored as readable Markdown, written only through validated code.
- **Knows your engine:** index Unity's documentation, packages and your shared code. The agent looks up
  exact APIs (`Rigidbody.MovePosition`) through MCP tools when it's unsure, instead of guessing from
  another version.
- **Stays inside the window:** old tool output is compressed and old turns are summarised, *only* when a
  session would no longer fit, in steps that keep Ollama's prompt cache hitting.
- **Spends tokens where they matter:** stable memory is cached, simple turns skip the thinking phase,
  and token counts are calibrated to your model.

## Features

- **Dual-GPU by design:** each model pinned to its own card; one-GPU mode also supported.
- **Persistent memory:** extraction, consolidation and history recall, hybrid keyword + semantic search.
- **Reference libraries:** documentation and code indexed per library and version (C# split per member),
  available as MCP tools (`docs_search`, `docs_lookup`) or as small automatic excerpts.
- **Tool access control:** every tool your agent offers (built-ins, each MCP server, e.g. a Unity Editor
  server) is listed by source in the console; set each to *off* (hidden and blocked), *automatic* or
  *on demand* (the model loads it only when needed), and see what each choice costs per request.
- **Long-session context management:** size-based trimming, session summaries, tool-output digests.
- **Web console:** everything the CLI does, at `http://127.0.0.1:8000/ui`: GPU status, chat, memory,
  libraries, settings, and evaluation runs side by side.
- **Evaluation harness:** replays *your* sessions to compare settings on cost and accuracy, with
  leak-free memory, paired statistics, a diagnosis of why each answer passed or failed, and test
  questions generated from your own work and corrections.
- **One-command setup** on Windows, including measuring how much context fits each GPU.

## Requirements

- Two GPUs supported by [Ollama](https://ollama.com) (one works, without isolation)
- Python 3.11+
- Windows + NVIDIA for `setup.ps1`; Linux and AMD with manual setup
- An agent that speaks Ollama's API, e.g. OpenClaw (WSL or native)

## Installation

```powershell
git clone <this repo>
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

`setup.ps1` detects your GPUs, picks models for your VRAM, starts two pinned Ollama instances, measures the
largest context that fits each card, writes the config, starts the orchestrator, and configures OpenClaw in
WSL (model provider and docs tools). It asks before changing anything outside the project folder, backs up
what it changes, and can be re-run any time.

Linux, AMD or manual setup: see [Installation](docs/DOCUMENTATION.md#3-install).

## Getting started with a Unity project

1. **Tell it the fixed facts.** In the console's *Memory* tab, add entries for your Unity version, render
   pipeline (URP/HDRP), Input System, folder layout and coding conventions. They go into the cached
   memory base: nearly free per turn, and they stop wrong-version answers.
2. **Index the docs.** Download Unity's offline documentation and add it in the *Library* tab, or:
   ```powershell
   .\ai docs add D:\Docs\Unity6 --name unity-6 --version 6000.0
   .\ai docs add D:\Packages\com.unity.inputsystem --name input-system --version 1.11 --auto
   ```
   Mark as *automatic* only the one or two libraries you use constantly; the rest stay on demand.
3. **Point your agent at the orchestrator** (`http://127.0.0.1:8000`, Ollama API) and work as usual.
   OpenClaw also gets the docs tools (`setup.ps1` registers them; `.\ai docs mcp` shows how).
4. **If you add a Unity Editor MCP server** (Unity CLI `unity mcp`, or the community MCP for Unity), open the
   *Tools* tab after the first request: set the Unity source to *on demand*, keep the compile/console/test
   tools *automatic*, and turn arbitrary-code tools *off*.
5. **After a week or two, measure.** Generate test questions from your sessions and compare settings:
   ```powershell
   .\ai eval generate --mode new-session --last 20
   .\ai eval run --variant baseline,full
   ```

## Usage

```powershell
.\ai ui                      # open the console
.\ai status                  # both GPUs, residency, queue, health
.\ai chat -v                 # chat from the terminal; -v shows the memory used
.\ai memory search "query"   # search project memory
.\ai docs lookup Rigidbody.MovePosition
.\ai eval run                # compare settings on your own sessions
```

On Linux, use `python -m app.cli` instead of `.\ai`. Clients using the OpenAI-compatible `/v1` API are
passed through **without** memory; choose "Ollama" as the provider type.

## Configuration

Everything lives in `config/config.yaml`; the defaults work without changes, and the console's *Settings*
tab edits the common ones safely. Most often tuned:

| Setting | What it controls |
|---|---|
| `ollama.*.model`, `num_ctx` | models and context size per GPU (set by setup) |
| `memory.max_context_tokens` | memory injected per prompt |
| `references.auto_max_tokens` | automatic documentation excerpts per turn |
| `proxy.trim_target_ratio` | how deep history is cut when the window fills |
| `thinking.mode` | `auto` / `client` / `on` / `off` |

## Documentation

- [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md): full manual covering manual setup, how each feature works,
  tuning, the console, the evaluation harness, API reference and troubleshooting.
- [AGENTS.md](AGENTS.md): project history, architecture and conventions, for AI agents and contributors
  continuing the work.

## Status

Built and tested against simulated Ollama instances (244 automated tests, plus browser checks of the
console). Behaviour on real GPUs (pinning, VRAM fit, model quality) is what the setup script measures and
the evaluation harness is for. Run both on your machine before relying on specific numbers.
