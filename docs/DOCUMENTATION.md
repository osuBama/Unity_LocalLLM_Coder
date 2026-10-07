# Local AI Orchestrator: full documentation

*Continuing development (architecture, history, invariants, workflow): see [AGENTS.md](../AGENTS.md).*

Run two local LLMs on two GPUs as one system:

- **GPU A, the primary model**, does the actual work: chat, coding, agent tool calls.
- **GPU B, the memory model**, works in the background. It maintains a persistent project
  memory, rolling session summaries and digests of large tool output, so the primary gets the
  context it needs in far fewer tokens.

```
your client ──► 127.0.0.1:8000  orchestrator ──► Ollama A (GPU A)  primary model
(OpenClaw, ai chat,     ├ injects relevant memory + session summary
 any Ollama client)     ├ trims old history / compresses old tool output (cache-friendly)
                        ├ strips the primary's <memory_flag> hints
                        └ queues finished turns ──► worker ──► Ollama B (GPU B)  memory model
                                                      └► validator ─► Markdown memory + SQLite
```

To your client, the orchestrator looks like a normal Ollama server. Everything is local; nothing
leaves the machine.

**Contents:** 1 Requirements · 2 Plan your setup · 3 Install · 4 Start two pinned Ollama instances ·
5 Configure · 6 Start and verify · 7 Connect a client · 8 Tune for your VRAM · 9 How it works ·
10 Web console · 11 CLI and API · 12 Evaluate · 13 Troubleshooting · 14 Limitations

---

## 1. Requirements

- **Two GPUs**, any combination that Ollama supports. The included launcher script is for
  **Windows + NVIDIA**. Linux and AMD work too, with the manual commands in §4.
- **Ollama**, recent enough for both of your cards. New GPU generations need recent builds.
- **Python 3.11+**.
- Enough **PSU** headroom for both cards at full load, with separate PCIe power cables per card.
  Check both cards' rated board power plus your CPU.
- The second card can sit in an x4-electrical slot. Inference barely uses PCIe once a model is loaded.

One GPU also works (see §4.4), just without the isolation benefits.

## 2. Plan your setup

### 2.1 Which GPU does what

| Role | Give it | Why |
|---|---|---|
| Primary | The GPU with **more VRAM** (if equal, the faster one) | Answer quality and speed come from here |
| Memory | The other GPU | Its work happens in the background, so it can be older or slower |

The memory GPU needs to be good at following instructions and producing JSON, not at deep reasoning.

### 2.2 Sizing models to VRAM

Every model must fit **entirely** on its GPU. With partial CPU offload, performance collapses.
Rough rules for Q4_K_M quantization:

- **Weights** ≈ 0.6 GB per billion parameters (8B ≈ 5 GB, 14B ≈ 9 GB, 32B ≈ 20 GB).
- **KV cache** (the context) with `q8_0` cache type ≈ 0.04–0.1 GB per 1,000 tokens for 7–14B models.
  It is roughly double that with the default f16 cache.
- **Overhead** ≈ 0.5–1 GB. If the GPU also drives your monitors, subtract what the desktop uses
  (check `nvidia-smi` at idle).

Starting points (verify with §6; the measurement wins over this table):

| GPU VRAM | Primary model (num_ctx) | Memory model (num_ctx) |
|---|---|---|
| 4–6 GB | not recommended | `qwen3:4b` (4k–8k) or `qwen3:1.7b` |
| 8 GB | `qwen3:8b` (8k) | `qwen3:8b` (8k) |
| 12 GB | `qwen3:14b` (16k) | `qwen3:8b` (16k) |
| 16 GB | `qwen3:14b` (32k) | `qwen3:14b` (8k) |
| 24 GB | `qwen3:32b` (8k–16k) or `qwen3:14b` (64k) | overkill; consider giving it more work (§13) |

Any Ollama model works; the Qwen3 family is just a consistent example. For the **memory** model,
prefer models that handle structured output well. Smaller memory models produce more rejected
changes; you can see that in `ai memory changes` (§12).

## 3. Install

### 3.0 Quick setup (Windows + NVIDIA)

`setup.ps1` does everything in sections 3–7 for you, and is safe to re-run:

```powershell
cd D:\AI
powershell -ExecutionPolicy Bypass -File .\setup.ps1          # interactive (recommended)
powershell -ExecutionPolicy Bypass -File .\setup.ps1 -Yes     # accept every default
```

What it does, in order:

| Step | What happens | Touches outside this folder? |
|---|---|---|
| Preflight | Checks Python 3.11+, creates `.venv`, installs dependencies | no |
| GPUs | Detects cards with `nvidia-smi`; larger VRAM = primary (`-PrimaryGpu/-MemoryGpu <UUID>` to override; one card = single-GPU mode) | no |
| Ollama | Checks the install; offers to disable the tray app's autostart (shortcut and registry value are backed up) and to stop running Ollama | yes, asks first |
| Models | Suggests models and context from VRAM (subtracting what your desktop already uses); you can override. Adds the embedding model (`-EmbeddingModel`, default `nomic-embed-text`; `none` to disable) | no |
| Instances | Starts the two pinned instances and pulls the models into one shared folder | no |
| Context fit | Loads each model (the embedding model first, on the memory GPU) and steps `num_ctx` down until everything is **100% on its GPU**, measured, not guessed | no |
| config.yaml | Writes models, ports, fitted contexts and memory budgets; keeps comments; validates; backs up; rolls back if invalid | no |
| Orchestrator | Starts it (restarting an old one), checks `/health`, offers to index existing history for recall | no |
| WSL | Adds `networkingMode=mirrored` (and a RAM cap if none) to `.wslconfig`, keeping your other settings; offers `wsl --shutdown` | yes, asks first |
| OpenClaw | Finds the distro with OpenClaw, backs up its config, points the Ollama provider at the orchestrator (`openclaw config patch`, falling back to `config set`, or leaves a file for a manual merge), sets the default model, restarts the gateway, and tests the connection from inside WSL | yes, asks first |
| Autostart | Optionally registers two logon tasks (instances, then the orchestrator 45 s later) | yes, asks first (default no) |

Useful options: `-PrimaryModel/-MemoryModel` and `-PrimaryContext/-MemoryContext` (starting points for the
fit), `-ModelsDir`, `-WslDistro`, `-SingleGpu`, `-SkipPull`, `-SkipFit`, `-SkipWsl`, `-SkipOpenClaw`,
`-NoStart`, `-Autostart`/`-NoAutostart`, and the ports. A full transcript goes to `logs\setup-<time>.log`.
Backups go to `backups\setup-<time>\`, next to `config.yaml` (`config.yaml.bak-*`) and next to the
OpenClaw config (`openclaw.json.bak-*`). To undo autostart, delete the two "AI Orchestrator" tasks in
Task Scheduler.

The rest of this section and sections 4–7 describe the same steps by hand, for Linux, AMD, or anyone
who prefers to see each piece. `python -m app.setup_tools` (GPU detection, model plan, config editing,
context fitting, OpenClaw patch) works on Linux too.

### 3.1 Manual install

Pick an install folder (examples use `D:\AI`; any path works). Copy this project so the folder
contains `app\`, `config\`, `prompts\`, `scripts\`, `memory\`.

**Windows:**
```powershell
cd D:\AI
powershell -ExecutionPolicy Bypass -File .\scripts\start-orchestrator.ps1 -Test
```
This creates `.venv`, installs dependencies and runs the test suite.

**Linux:**
```bash
cd ~/ai
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt && python -m pytest -q
```

Then edit `config\config.yaml` and set `paths.root` to your install folder. Alternatively, set the
`AI_ROOT` environment variable; `AI_CONFIG` points at an alternative config file.

**Seed memory:** `memory\*.md` ships with example entries describing one specific setup. Edit them
to describe yours, or delete the `.md` files and they'll be recreated empty on first start.

## 4. Start two pinned Ollama instances

The idea: two `ollama serve` processes, each limited to one GPU and listening on its own port, and
both reading the same models folder (weights are stored once).

### 4.1 Windows + NVIDIA (script)

First stop the Ollama tray app and disable its autostart (Task Manager → Startup apps). It runs its
own unpinned server on port 11434, which would grab the port and both GPUs.

```powershell
.\scripts\start-ollama-instances.ps1 -ListGpus           # names, UUIDs, VRAM
.\scripts\start-ollama-instances.ps1 -ModelsDir D:\AI\models
```

How roles are chosen: with exactly two GPUs of different VRAM, the **larger is primary** automatically.
Otherwise pick them by name or UUID:

```powershell
.\scripts\start-ollama-instances.ps1 -PrimaryMatch "4070" -MemoryMatch "3060"
.\scripts\start-ollama-instances.ps1 -PrimaryGpu GPU-1a2b... -MemoryGpu GPU-9f8e...
```

Other options: `-PrimaryPort/-MemoryPort` (11434/11435), `-PrimaryContext/-MemoryContext` (default
context per instance), `-KvCacheType` (`q8_0`; use `f16` if a model misbehaves), `-LogDir`,
`-Force` (stop running Ollama processes first), `-Status`, `-Stop`.

Each instance gets: `CUDA_VISIBLE_DEVICES=<GPU UUID>` (UUIDs, because CUDA indices don't reliably
follow slot order), flash attention on, `q8_0` KV cache, `OLLAMA_NUM_PARALLEL=1` (each parallel slot
costs a full KV cache) and `OLLAMA_MAX_LOADED_MODELS=1`.

### 4.2 Linux + NVIDIA (manual)

```bash
nvidia-smi -L                                   # get the UUIDs
common="OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_MODELS=/srv/ai/models"
env $common CUDA_VISIBLE_DEVICES=GPU-aaaa OLLAMA_HOST=127.0.0.1:11434 OLLAMA_CONTEXT_LENGTH=16384 ollama serve &
env $common CUDA_VISIBLE_DEVICES=GPU-bbbb OLLAMA_HOST=127.0.0.1:11435 OLLAMA_CONTEXT_LENGTH=8192  ollama serve &
```

To run them permanently, create two systemd services (or override the stock `ollama.service` for
instance A and copy it as `ollama-memory.service` for B), each with those values as `Environment=`
lines. Stop the stock service first if it isn't one of the two, since it binds 11434 unpinned.

### 4.3 AMD

Same idea, but Ollama selects AMD GPUs with `ROCR_VISIBLE_DEVICES` / `HIP_VISIBLE_DEVICES` instead of
`CUDA_VISIBLE_DEVICES`; see Ollama's GPU documentation for your platform. The Windows script is
NVIDIA-only, so start the instances manually as in §4.2. Mixing an NVIDIA and an AMD card also works
in principle, since each instance only needs its own GPU, but it is untested here.

### 4.4 One GPU only

Point both `ollama.primary.base_url` and `ollama.memory.base_url` at the same instance and set
`OLLAMA_MAX_LOADED_MODELS=2` on it. Both models must fit in VRAM together, or Ollama will swap them
in and out on every background task, which is very slow. A small memory model (`qwen3:1.7b`/`4b`)
helps. `/health` will warn that there is no isolation.

### 4.5 Pull models and verify pinning

Pull once; both instances share the models folder:
```powershell
$env:OLLAMA_HOST="127.0.0.1:11434"; ollama pull qwen3:14b; ollama pull qwen3:8b; Remove-Item Env:OLLAMA_HOST
```
(Linux: `OLLAMA_HOST=127.0.0.1:11434 ollama pull ...`)

Load each model on its own instance, then check:
```powershell
$env:OLLAMA_HOST="127.0.0.1:11434"; ollama run qwen3:14b "hi"
$env:OLLAMA_HOST="127.0.0.1:11435"; ollama run qwen3:8b "hi"; Remove-Item Env:OLLAMA_HOST
.\scripts\start-ollama-instances.ps1 -Status        # Linux: OLLAMA_HOST=127.0.0.1:1143x ollama ps ; nvidia-smi
```
**Both models must show 100% GPU, and each on the GPU you intended.** If not, lower that instance's
context or pick a smaller model or quant (§2.2).

## 5. Configure

`config\config.yaml` controls everything. The settings you must check:

| Setting | Set it to |
|---|---|
| `paths.root` | your install folder |
| `ollama.primary.model` / `num_ctx` | your primary model and the context that fits (§2.2) |
| `ollama.memory.model` / `num_ctx` | your memory model and its context |
| `ollama.*.base_url` | your two instances' ports |
| `ollama.memory.think` | `false` for models with a thinking mode (faster, cleaner JSON); remove for others |
| `memory.max_context_tokens` | ~15% of the primary's `num_ctx` (2,500 at 16k) |

Everything else has working defaults; §8 explains when to change them.

## 6. Start and verify

```powershell
.\scripts\start-orchestrator.ps1        # Linux: python -m app.main
.\ai status                             # Linux: python -m app.cli status
.\ai chat -v
```

`ai status` shows both instances, whether each loaded model is 100% on GPU, the background queue and
memory-file health. `ai chat -v` shows which memory entries were injected for each message.

The orchestrator listens on `127.0.0.1:8000` and has no authentication. Don't expose it to your LAN.

## 7. Connect a client

### 7.1 Any Ollama-native client

Point the client's Ollama URL at `http://127.0.0.1:8000` instead of `:11434`. It will list the primary
instance's models and chat as usual, with memory added automatically.

Memory only works on Ollama's **native** `/api/chat` endpoint. Clients that use the OpenAI-compatible
`/v1` endpoints are passed through **without** memory. If a client offers both, choose "Ollama".

Conversation ids: native requests don't carry one, so the orchestrator derives one from the model and
the first user message. Clients that can send an `X-Conversation-Id` header get exact session tracking.

### 7.2 OpenClaw

Merge `examples\openclaw-provider.json5` into `~/.openclaw/openclaw.json`, set your model id and
`contextWindow` (equal to `ollama.primary.num_ctx`), then run `openclaw gateway restart`. Keep
`api: "ollama"`. Check the snippet against your OpenClaw version's provider docs.

### 7.3 Clients inside WSL

To reach `127.0.0.1:8000` on Windows from WSL, copy `examples\wslconfig.example` to
`%UserProfile%\.wslconfig` (mirrored networking, plus a RAM cap so WSL doesn't take half your memory)
and run `wsl --shutdown`.

### 7.4 Your own code

```
POST http://127.0.0.1:8000/chat   {"conversation_id": "optional", "message": "..."}
```
This returns the answer, the conversation id and which memory entries were used. It keeps the last
`conversation.recent_turns` exchanges verbatim and lets memory and the session summary carry the rest.

## 8. Tune for your VRAM

All context-saving features are on by default and scale with your settings.

| Feature | Settings | Rule of thumb |
|---|---|---|
| Memory injection | `memory.max_context_tokens` | ~15% of primary `num_ctx`. Clients with large system prompts or many tools (agents) need more free room. |
| Keeping history in the window | `proxy.trim_mode` (`size`), `trim_target_ratio`, `reply_reserve_tokens`, `trim_keep_user_turns` | `size` mode needs no tuning: nothing happens until the prompt would not fit `num_ctx`. Lower `trim_target_ratio` (e.g. 0.4) is cheaper on very long sessions; verify accuracy with golden questions (§12). |
| Tool-output compression | `compression.min_result_tokens`, `keep_recent_user_turns`, `digest_max_tokens` | Defaults suit most setups. `keep_recent` must be smaller than `trim_keep`. |
| Session summaries | `session.summary_max_tokens` | 400; raise it if long sessions lose details. |
| Memory flags | `flags.enabled` | On; costs the primary ~20–50 tokens only on turns that flag something. |
| Primary thinking | `thinking.mode` (`auto`), `thinking.simple_max_words` | `auto` turns thinking off only for clearly simple turns (§9.7). Use `client` to never touch it. |
| Memory-model thinking | `memory.think_extraction`, `memory.think_consolidation` | On: background jobs nobody waits on get better judgement. Summaries and digests never think. |
| Semantic search | `embeddings.*`, `memory.enable_semantic_retrieval` | On with `nomic-embed-text` on the memory GPU (~0.3 GB). Falls back to keywords if unavailable (§9.9). |
| History recall | `history_recall.*` | 600 tokens max, top 2, similarity 0.6 (0.45 when you say "last time", "earlier"...). Raise `min_similarity` if irrelevant excerpts appear. |
| Token calibration | `proxy.calibrate_tokens` | On: size-based trimming learns your model's real chars/token (§9.8). |
| Memory base (cached) | `stable_memory.max_tokens`, `categories` | ~40% of `memory.max_context_tokens`. It's part of that budget, not extra. |

If `ai status` ever shows less than 100% GPU, **lower `num_ctx` first**. Keep the client's own
context-window setting (e.g. OpenClaw `contextWindow`) equal to `ollama.primary.num_ctx`.

## 9. How it works

### 9.1 Per request (`/api/chat`)

1. The turn plan is fixed once per user turn and reused for every tool-call round trip: the trim
   point, which old tool results become digests, and the memory block.
2. Old history is trimmed in steps, and old large tool results are swapped for digests (§9.4).
3. Rarely-changing memory (constraints, objectives, environment) goes into the **system prompt** as a
   frozen `<PROJECT_MEMORY_BASE>`, where the prompt cache reaches it (§9.5).
4. Everything turn-specific (current state, relevant lessons, decisions and discoveries, the session
   summary, and any base changes since it was frozen) is **appended** as `<PROJECT_MEMORY>…</PROJECT_MEMORY>`
   to the **last message** of the request: the user's message, or the newest tool result mid tool-loop.
   On the next request that message comes back without the block, so only the block itself drops out
   of the cache. The evaluation harness showed that putting it in front of the user's message instead
   made every agent turn re-read the previous turn's tool output.
5. `prompts\primary_system.txt` (and the flag instruction) is appended to the client's system prompt,
   and `num_ctx` is set if the client didn't.
6. The reply streams back with tool calls intact and `<memory_flag>` tags removed.
7. When the reply has no pending tool calls, the turn is over. It is written to raw JSONL, and the
   background queue gets, in priority order: the session-summary update, digests for large tool
   results, and memory extraction (if the turn was flagged or matched the trigger heuristics).

### 9.2 Persistent memory

`memory\*.md` holds one file per category: STATE, OBJECTIVES, CONSTRAINTS, DECISIONS, LESSONS,
DISCOVERIES, ENVIRONMENT. Each entry has a stable id:

```markdown
### L-003 — HTTP 404 debugging
<!-- meta: status=active created=... updated=... source=... -->
A successful HTTP connection does not prove the requested endpoint exists.
```

- You can edit entry text by hand; keep the `###` heading and the meta line. The store refuses to
  rewrite a file with sections it doesn't recognise rather than lose your text. `ai memory validate`
  tells you what is wrong.
- Every write is: backup to `memory\history\` → re-parse and verify → atomic replace.
- The memory model only **proposes** changes, as JSON constrained by Ollama's structured output. A
  validator treats that output as untrusted. It rejects bad schemas, path-like titles, prompt-injection
  phrasing, dangerous commands and anything that looks like a secret, and it dedupes against existing
  memory. Every proposal, including rejections and the reason, is logged in SQLite.
- Raw conversation history (`conversations\*.jsonl`) is never deleted, so memory can always be rebuilt
  from it (`ai memory rebuild --replay`).

### 9.3 Memory flags

The primary is told to end a reply with up to three lines like
`<memory_flag category="lesson">MCP route is /mcp, not /sse</memory_flag>` when something durable
happened. They are stripped from the stream (even when split across chunks), always trigger
extraction, and reach the memory model as hints to verify. Keyword heuristics (English + Portuguese)
remain as a fallback for turns the primary forgets to flag.

### 9.4 Session summaries, trimming and compression

- **Session summary:** after every turn the memory model updates a rolling summary of that
  conversation. Summaries only move forward (a late retry can't overwrite a newer one).
- **Keeping history inside the window (`trim_mode: size`, default):** each turn the orchestrator
  estimates the full prompt: history, the client's tool schemas, our system additions, plus reserves for
  the memory block and the reply. **While it fits `num_ctx`, nothing is trimmed or compressed**, however
  long the session. When it would not fit:
  1. old tool results are replaced by digests first (they're usually the bulk);
  2. if the prompt is still above `trim_target_ratio` (50%) of the window, the oldest turns are cut until
     it isn't, and the session summary is injected as `SESSION SO FAR`.

  Cutting well below the limit (hysteresis) means the layout then stays fixed for many turns, so the
  prompt prefix is identical and Ollama's cache hits until the window fills again. The layout only ever
  moves forward, at least `trim_keep_user_turns` recent turns are always kept, and turns the summary
  doesn't cover yet are never cut. If the prompt can't be made to fit without breaking those rules, the
  request is flagged `context_over_budget` in `/metrics` instead of silently losing content.
- **Turn-count trimming (`trim_mode: turns`):** the older schedule. Past `trim_trigger_user_turns`
  turns, history is cut back to about `trim_keep_user_turns` in fixed steps (with 10/4: turns 11, 16, 22, …),
  and the compression boundary moves on the same turns. It's simpler, but it trims even when everything
  would have fit, which costs cache misses on sessions that never needed it.
- **Tool-output compression:** large tool results from older turns are replaced by digests labelled
  `[compressed tool output: original ~N tokens; re-run the tool if exact output is needed]`. The
  current turn and the last `keep_recent_user_turns` are never touched. A digest is only used if it
  exists, respected its length limit and saves at least half. The compression boundary only moves when
  the layout changes (in size mode: when the window fills; in turn mode: with the trim point), and the
  set of compressed results is frozen in between.
- Cuts only happen at user-message boundaries, so tool-call chains are never split. The client keeps its
  full history on its side; only what is sent to the model shrinks.

### 9.5 The cached memory base

Without it, all injected memory sits in the latest user message, after the cached prefix, so Ollama
re-processes the whole block (up to `memory.max_context_tokens`) on every turn. Most of it is the
same every time: constraints, objectives, environment. Those now go at the end of the system prompt,
which is part of the cached prefix:

- **Frozen per epoch.** The base is rebuilt only when the trim point or compression boundary moves,
  i.e. on turns where the prefix changes and the cache misses anyway. With both features off, it
  refreshes every `stable_memory.refresh_turns` turns. Entries are ordered by id, so unchanged memory
  rebuilds to byte-identical text and causes no miss at all.
- **Never stale.** If a base entry is added, edited or deactivated mid-epoch, the next turn's
  per-turn block carries an `UPDATED SINCE THE MEMORY BASE` section with the new version (or
  "no longer applies"), which the model is told takes precedence. At the next refresh the change moves
  into the base and the update line disappears.
- **No duplication, no extra context.** Entries in the base are skipped by per-turn retrieval. Entries
  that didn't fit the base's `max_tokens` fall back to normal relevance retrieval. The base comes out of
  `memory.max_context_tokens`, so total memory in the context window doesn't grow.
- `current state` is deliberately not in the base by default: it changes most often. Add or remove
  categories with `stable_memory.categories`.

`ai memory context "prompt"` prints both parts. In `/metrics`, `memory_base_tokens` is the cached
part and `memory_tokens` is what is re-processed every turn.

### 9.6 Idle-time consolidation

Memory grows by accretion, so near-duplicates and overgrown entries pile up and both inflate the
injected context and weaken retrieval. When the system is idle, the memory GPU tidies up:

- **When it runs:** no client request for `consolidation.idle_minutes` (10), the task queue is empty,
  at least `min_interval_hours` (12) since the last run, and at least `min_changes_since_last` (5)
  memory changes since then. A run stops as soon as a request or queued task arrives, and resumes
  later. Run it on demand with `ai memory consolidate` or `POST /memory/consolidate`; add
  `--dry-run` / `?dry_run=true` to see proposals without applying them.
- **What it does:** per category, a deterministic similarity pre-filter forms small groups of related
  entries, and only those groups are shown to the memory model. It may propose a **merge** (keep the
  oldest id, deactivate the others) or a **rewrite** of a long entry. It can never add entries, and it
  never deactivates anything except as part of a merge.
- **Checked in code before anything is written:** every id must exist, be active and belong to the
  group shown; the same safety rules as extraction apply; the new text must keep at least
  `identifier_coverage` (90%) of the names, numbers, ports, paths and versions of the originals;
  rewrites must shrink to at most `rewrite_max_ratio` (80%); and at most `max_changes_per_run` entries
  change per run.
- **Reversible:** a backup snapshot is taken before the first change of a run. Merged entries stay in
  the file's inactive section, and every applied or rejected proposal is in `ai memory changes`.
  `ai memory restore <backup-dir>` puts a snapshot's memory files back (after snapshotting the
  present) and rebuilds the SQLite index.
- **Unused entries are listed, never removed.** Every time an entry is injected, its use is recorded.
  `ai memory review` (`GET /memory/review`) lists active entries older than `stale_after_days` (30)
  that haven't been injected in that time, for you to decide on.

`GET /memory/consolidation` shows whether a run is due, why or why not, and the last run's report.

### 9.7 Thinking per turn

Models with a thinking mode (Qwen3 and others) can spend hundreds to thousands of tokens reasoning
before they answer. That's worth it for debugging and design, and pure latency for "run the tests". With
`thinking.mode: auto` the orchestrator decides once per user turn, and keeps that decision for every
tool-call step of the turn:

- Thinking is turned **off** only when the turn is short (`simple_max_words`, 25) and shows none of: code,
  error or diagnostic output, or words like why / how does / debug / design / implement / fix / explain /
  compare (English and Portuguese).
- It is **never turned on** if the client didn't ask, and a client that disabled it stays disabled.
- Saying "think", "step by step" or "carefully" keeps thinking on regardless.
- `/metrics` shows `thinking` (on / off / client) and the reason per request.

Check the trade-off on your work with `ai eval run --think default --variant thinking-always,thinking-auto`
(definitions in `examples\variants.example.yaml`): compare golden pass rates and request times.

The memory model thinks on **extraction and consolidation**, which run in the background. If a thinking
run doesn't produce valid JSON (some model and Ollama combinations don't mix thinking with structured
output well), it is retried once without thinking before counting as a failure. Session summaries and tool
digests never think, because the next prompt may be waiting for them.

### 9.8 Token calibration

Size-based trimming (§9.4) has to estimate tokens before Ollama sees the prompt. The safe default
(~3.5 characters per token) overestimates for most models and so trims earlier than needed. With
`proxy.calibrate_tokens`, the orchestrator learns each model's ratio from Ollama's own counts:

- Ollama reports the tokens it processed, and cache hits only lower that number, so only requests that
  processed close to the whole prompt are recorded.
- The **lowest** recorded ratio is used, plus a 5% margin. Template tokens and tool schemas only push it
  down, which is the safe direction: an overestimated ratio would mean silent context overflow.
- It needs 5 observations; until then the default applies. Values persist across restarts. Current values
  are in `/metrics` under `token_calibration`.

### 9.9 Semantic search and history recall

An embedding model (default `nomic-embed-text`, ~0.3 GB) runs on the memory GPU next to the memory model.
The memory instance is started with two loaded models allowed, so embedding calls never evict it. Vectors
are stored in the existing SQLite database; at this scale an exact search over all of them takes
milliseconds, so there is no separate vector database to run.

- **Hybrid memory retrieval.** Keyword search (strong on ports, error codes, model names, paths) and
  vector search (paraphrases, other languages) are merged with reciprocal rank fusion. An entry found
  *only* by the vector side must reach `embeddings.min_similarity`, so weak semantic matches can't pad the
  prompt.
- **History recall.** Every finished turn becomes a searchable excerpt. When a prompt clearly relates to an
  older exchange that is **not already in the prompt** (another session, or turns trimmed away), up to
  `history_recall.top_k` excerpts are added under `RELATED PAST EXCHANGES`. They're labelled with date and
  conversation, and marked as possibly outdated. The similarity bar is lower when you say "last time",
  "earlier", "remember", "da última vez", etc. Excerpts never exceed `history_recall.max_tokens`.
- **Cheap in the request path.** Only your message is embedded, once per user turn (tool-loop steps reuse
  it), with a `timeout_seconds` limit. If the embedder is slow or down, retrieval silently uses keywords
  and `/health` shows a warning. Memory entries and history excerpts are embedded by the background worker.
- **Existing history:** `ai memory reindex` makes everything already in `conversations\*.jsonl` searchable
  (setup.ps1 offers to run it). New turns are indexed automatically.
- Changing `embeddings.model` makes old vectors unusable: run `ai memory reindex` afterwards.

### 9.10 Reference libraries (documentation and code)

Documentation and libraries the model should *consult* (the Unity scripting reference, a package's
source, your studio's shared code) are too big for memory and would swamp the context window. They get
their own index instead:

```
ai docs add D:\Docs\Unity6 --name unity-6 --version 6000.0 --description "Unity 6 manual + scripting API"
ai docs add D:\Packages\com.unity.inputsystem --name input-system --version 1.11 --auto
ai docs list | update unity-6 | remove input-system | enable/disable NAME | auto NAME on|off
ai docs search "move a kinematic rigidbody" | lookup Rigidbody.MovePosition | mcp
```

The console's **Library** tab does all of the same, with a folder browser and live indexing progress.

- **What gets indexed:** Markdown, HTML (navigation and scripts stripped), text, C#, Python, JS/TS and
  shader files. `Library`, `Temp`, `obj`, `bin`, `.git` and `node_modules` are skipped. **C#** is split per
  type and per member, with its `///` comments and its full name (`Game.Physics.Mover.MoveKinematic`), so
  an exact lookup returns exactly that member. Re-indexing only processes changed files, and removed files
  drop out of the index.
- **Search** is hybrid: SQLite's full-text index (BM25) for exact names, plus embeddings for meaning,
  merged by rank. Vectors are stored per library, at half precision by default (`references.half_precision`).
- **Two ways into the model:**
  - **On demand (recommended for agent work):** the orchestrator serves an MCP server at `/mcp` with
    `docs_search`, `docs_lookup` and `docs_libraries`. setup.ps1 registers it with OpenClaw as `local-docs`;
    `ai docs mcp` prints the snippet for doing it by hand. The model looks things up only when it needs to,
    and results are capped at `references.tool_max_tokens`. Like any tool output, they're later compressed
    into digests.
  - **Automatic:** for libraries marked *automatic*, up to `references.auto_max_tokens` (800) of excerpts are
    added in a `<REFERENCE>` block, but only on a keyword hit or a similarity above
    `references.auto_min_similarity`. That budget is reserved in the context-window estimate *only while an
    automatic library exists*, so it never pushes a prompt past `num_ctx`.
- **Cost:** no VRAM beyond the embedding model you already run. System RAM for vectors is roughly
  (sections × 768 dimensions × 2 bytes) with nomic-embed-text at half precision, about 150 MB per
  100,000 sections. Indexing the namespaces and packages you actually use keeps it small.
- **Safety:** documentation text is data. It is sanitised so it can't close the `<REFERENCE>` or memory
  blocks, the primary is told not to follow instructions inside it, and `/mcp` is read-only and refuses
  requests from non-local browser origins.

**A Unity setup that works well:**
1. Put the project's fixed facts in memory (Memory tab, *Add entry*, category *constraint* or
   *environment*): Unity version, render pipeline (URP/HDRP), Input System, scripting conventions, folder
   layout. They go into the cached memory base, which is close to free per turn and prevents the classic
   wrong-version answers.
2. Index Unity's offline documentation (downloadable from Unity's documentation site) as one library with
   its version, and the packages you depend on as separate libraries. Mark *automatic* only the one or two
   you use constantly.
3. Let the agent read your own scripts through its file tools. Tool-output compression keeps old reads from
   filling the window.
4. Add golden questions about APIs, e.g. `expect_all: ["MovePosition"]`. The diagnosis shows whether a miss
   came from the library, retrieval, or the model (location `reference`).

### 9.11 Tool access control

Every tool your client offers the model passes through the orchestrator in the request's `tools` field:
OpenClaw's built-ins and every tool of every MCP server it connects to (a Unity MCP server alone can
offer dozens). The orchestrator registers them automatically, groups them by **source**, and decides per
tool what the model actually gets. The model never sees the settings, only the resulting tool list.

| Mode | The model gets | Per request |
|---|---|---|
| **Off** | Nothing. If it calls the tool anyway (e.g. remembering the name), the call is **blocked**: removed from the response and replaced by a short note, so your client never executes it. | 0 tokens |
| **Automatic** | The full definition, every request (the behaviour without this feature). | full schema |
| **On demand** | One line in the catalog of a single `load_tools` tool. When the model calls `load_tools`, the orchestrator answers it itself, adds the full definitions, and re-asks the model within the same response (your client sees one normal answer). Loaded tools stay loaded for the rest of that conversation. | one catalog line until loaded |

- **Sources:** by name prefix (`unity__read_console` → `unity`, `mcp__github__…` → `github`), the
  orchestrator's own docs tools as `local-docs`, everything else as `client`. Grouping rules (`unity_*` → Unity,
  first match wins) and per-tool moves override that.
- **Modes** are set per source, with optional per-tool overrides; tools nobody configured follow
  `tools.default_mode` (`automatic` by default, so nothing changes until you decide). Set it to `on_demand` to
  make new MCP servers cheap by default.
- **Cache:** tool definitions sit in the cached start of the prompt. A conversation's loaded set only grows,
  so the cache breaks only when something is loaded or a setting changes.
- **Size estimates** use the filtered list, so turning tools off or on demand also leaves more room for history.
- Limits: `tools.max_load_rounds` (2) internal rounds per request; `tools.block_disabled_calls`. `/metrics`
  shows `tools_sent`, `tools_hidden`, `tools_in_catalog`, `tools_blocked` and `tool_load_rounds` per request.

Manage it in the console's **Tools** tab (with the per-request token cost of your choices) or with `ai tools`:

```
ai tools list [--source unity]
ai tools set --source unity on_demand
ai tools set unity__execute_code off
ai tools set unity__read_console automatic      # or: inherit
ai tools default on_demand
ai tools rule add "unity_*" unity | rule list | rule remove 3
ai tools source read_file workspace | source read_file --clear
ai tools forget --older-than 30
```

**A reasonable start with a Unity MCP server:** the whole Unity source *on demand*; the compile-fix loop
(`read_console`, refresh/recompile, `run_tests`) *automatic*; arbitrary-code tools (`execute_code` and similar)
*off*.

## 10. Web console

`http://127.0.0.1:8000/ui` (or `ai ui`) is served by the orchestrator itself: nothing extra to install,
and it works offline. The header always shows both GPUs side by side: each instance's model, how much of
every loaded model sits on the GPU, and the memory GPU's task queue. Tabs:

- **Overview:** request and background-work totals, token calibration, and a table of recent requests
  (processed tokens, memory tokens, cached base, thinking decision, timing, turns cut, tool tokens saved).
- **Chat:** a conversation through `/chat`. Under each answer you see which memory entries were used, any
  recalled past exchanges, whether the model thought and why, and the time taken.
- **Memory:** browse by category with usage counts, search (hybrid when embeddings are on), edit, add or
  deactivate entries (edits are backed up and logged as `ui` changes), preview exactly what a prompt
  would receive (cached base and per-turn block), and maintenance: back up, preview or run
  consolidation, unused entries, reindex, recent changes.
- **Settings:** a curated, documented subset of `config.yaml`. Saving writes the file with a backup and
  your comments intact, and validates it first (nothing is written if it wouldn't load). Settings marked
  *applies now* take effect immediately; *after restart* ones (models, context sizes, memory budget,
  flags) need the orchestrator restarted.
- **Tools:** every tool the client offers, grouped by source, with Off / Automatic / On demand per source and
  per tool, usage counts, the per-request token cost of your choices, grouping rules and forgetting (§9.11).
- **Library:** reference libraries (§9.10): add with a folder browser, re-index, edit version and description,
  enable or disable, automatic excerpts on or off, remove, search and exact-symbol lookup, live indexing
  progress with cancel, and the MCP connection details.
- **Evals:** start an evaluation (it runs as a separate process; the log streams into the page), select
  reports and compare them with bars for processed tokens and golden pass rate (with 95% intervals), a
  details table and a per-question matrix, and review captured corrections. Suggested checks are
  prefilled: values in your correction become *must contain*, values you negated ("not 11434") become
  *must not contain*.

**Everything the CLI does, the console does:**

| CLI | Console |
|---|---|
| `ai status`, `ai metrics` | header lanes, Overview |
| `ai chat` | Chat |
| `ai memory show / search / context / changes / review` | Memory: list, search, preview, *Recent changes*, *Show unused entries* |
| `ai memory tasks / sessions / validate` | Memory: *Background tasks*, *Session summaries*, *Check memory files* |
| `ai memory backup / consolidate [--dry-run] / reindex` | Memory: *Back up now*, *Preview consolidation* / *Consolidate now*, *Reindex for search* |
| `ai memory rebuild [--replay [--reset]] / restore` | Memory: *Rebuild…*, *Restore from a backup…* |
| `ai docs …` (all subcommands) | Library |
| `ai tools …` (all subcommands) | Tools |
| `config.yaml` edits | Settings (curated, validated, comments kept) |
| `ai eval sessions / run (all options)` | Evals: *More options* (sessions, memory mode, max turns, seed, answer length, extraction, client system prompt) |
| `ai eval candidates / accept / dismiss / generate` | Evals: *Questions waiting for review*, *Generate questions* |

`ai serve` (start the server) and `ai ui` (open the console) are the only CLI-only commands, for obvious reasons.

**Security.** The console and the admin API only listen on `127.0.0.1`. Every state-changing request
outside `/api/*`, `/v1/*` and `/chat` must carry an `X-AI-Client` header. Browsers can't add that
cross-site, so a web page you have open can't trigger backups, rebuilds or settings changes. When calling
these endpoints yourself, add the header: `curl -X POST -H "X-AI-Client: 1" http://127.0.0.1:8000/memory/backup`.

## 11. CLI and API

`ai` is `ai.cmd` on Windows and `python -m app.cli` on Linux.

```
ai serve | ui | chat [-v] | status | metrics
ai memory show [category] | search "text" | context "prompt" | changes | tasks | sessions
ai memory validate | backup | rebuild [--replay [--reset]] | restore <backup-dir> | reindex
ai memory consolidate [--dry-run] | review
ai eval sessions | ai eval run [--variant a,b,…] [--sessions …] [--golden …] [--max-turns N]
ai eval candidates [--all] | ai eval accept <id> [--expect X] [--forbid Y] | ai eval dismiss <id>
ai eval generate [--mode later|new-session|both] [--last N] [--gap 6] [--no-model] [--accept-all]
ai docs list | add <folder> --name N [--version V] [--description D] [--auto] | update N [--path P] | remove N
ai docs enable|disable N | auto N on|off | set N [--version] [--description] | search "q" | lookup Symbol | mcp
ai tools list | set TOOL|--source SRC off|automatic|on_demand|inherit | default MODE | source | rule | forget
```

`ai memory context "prompt"` shows exactly what would be injected for a prompt. `rebuild` alone
rebuilds SQLite from the Markdown; `--replay` re-queues all raw history through the memory model;
`--reset` starts from empty memory first. A snapshot to `backups\` is always taken beforehand. Stop the
server (or use the API) before CLI commands that write memory: rebuild, restore, consolidate.

| Method | Path | |
|---|---|---|
| POST | `/chat` | spec API (§7.4) |
| GET | `/health` | both instances, GPU %, worker, queue, memory-file health, warnings |
| GET | `/metrics` | per-request tokens, prefill/generation time, TTFT, tok/s, trimming/compression savings |
| GET | `/memory/state`, `/memory/search?q=`, `/memory/context?q=` | inspect memory |
| GET | `/memory/changes`, `/memory/tasks`, `/memory/sessions` | audit trail, queue, summaries |
| POST | `/memory/rebuild`, `/memory/backup` | maintenance |
| POST | `/memory/consolidate[?dry_run=true]` | run consolidation now (§9.6) |
| GET | `/memory/consolidation`, `/memory/review` | consolidation status and last report; unused entries |
| POST | `/mcp` | MCP server (Streamable HTTP): `docs_search`, `docs_lookup`, `docs_libraries` |
| * | `/api/*`, `/v1/*`, `/` | passthrough to the primary instance (memory only on `/api/chat`) |

Logs: `logs\orchestrator.log`, `primary.log`, `memory.log` (JSON lines), plus each Ollama
instance's log when started by the script.

## 12. Evaluate

Every feature trades some risk against tokens. The evaluation harness measures both on **your**
sessions, **your** models and **your** GPUs, so you can choose settings by numbers instead of by feel.

```
ai eval sessions                                   # recorded sessions available for replay
ai eval run                                        # baseline vs full on your last 3 sessions + golden.yaml
ai eval run --variant baseline,full,no-compression --sessions oc-1234,oc-5678 --max-turns 40
```

### 12.1 What it measures

- **Cost (context replay).** Each recorded turn is re-sent through the real orchestrator code with the
  *recorded* history and just 1 generated token. Every variant therefore sees the identical conversation
  and differs only in context strategy. Session summaries and tool digests are produced live by your
  memory model from the recorded answers. Per variant you get:
  - **processed prompt tokens:** what Ollama actually had to evaluate, cache hits excluded
  - **prefill time**
  - **estimated cache hit rate**
  - **peak context size**
  - **turns over `num_ctx`**
  - memory tokens per turn, tool tokens saved, turns trimmed
- **Accuracy (golden questions).** After replaying a session, a question is asked with real generation
  (temperature 0, fixed seed, thinking off by default) and graded deterministically: `expect_all`,
  `expect_any`, `forbid`, with `/regex/` support. There is no LLM judge, so there is no judge bias.
  Copy `examples\golden.example.yaml` to `evals\golden.yaml` and write cases about your own work. The
  most informative ones ask about details from **early** in a long session (tests trimming and
  summaries), from **old tool output** (tests compression), or facts that only live in long-term memory.

### 12.1b Golden questions from your own corrections

When you correct the model, e.g. "no, it's 11435" or "that's wrong, it should be qwen3:14b", the
orchestrator records a **candidate**: the question that got the wrong answer, the wrong answer, your
correction, and a suggested check (identifiers in your correction that weren't in the wrong answer).
Nothing changes until you decide:

```
ai eval candidates                         # review what was captured
ai eval accept 7                           # use the suggested check
ai eval accept 8 --expect dogs --forbid cats --name pet-topic
ai eval dismiss 9
```

Accepting appends a case to `evals\golden.yaml` (your comments are kept; the file is validated and left
unchanged if the result wouldn't load). The case replays that recorded session up to just before the
question and asks it again. Over time your golden set becomes a record of what actually went wrong,
which is the most useful thing it can test. Turn it off with `conversation.capture_corrections: false`.

### 12.1c Generated golden questions

Writing questions by hand is the bottleneck, so the harness can propose them from your recorded sessions:

```
ai eval generate                          # last 5 sessions, both kinds, worded by the memory model
ai eval generate --mode new-session --last 20 --per-session 3
ai eval generate --no-model               # fill-in-the-blank only, no model needed
```

The console has the same controls in the Evals tab. **How it picks facts:** specific values (ports, error
codes, host names, paths, versions, model tags, IPs) that are rare in the session. Nothing is generated from
values mentioned more than twice or from trivial numbers. Two kinds of question:

- **Later in the same session:** the value is not mentioned again for at least `--gap` turns (6), and the
  question is asked after that gap. Depending on session length and settings, the answer is still in the
  prompt (tests the model's attention), trimmed away (tests summaries and memory), or in compressed tool
  output (tests digests). The diagnosis (§12.3c) shows which.
- **In a new conversation:** asked at the start of a fresh conversation an hour after the source session
  ended, with memory and history as of then. Only memory extraction or history recall can answer, so this is
  the real cross-session test.

**Wording:** the memory model writes the question the way you'd ask it. Code checks that it doesn't contain
the answer and is a real question, and the model can mark an excerpt as too vague. Anything that fails
becomes a fill-in-the-blank question built from the excerpt, which is always valid. Every question becomes a
**candidate** next to your captured corrections. Review them in the console (with *Accept all shown* for
speed) or with `ai eval candidates` / `accept` / `dismiss`. `--accept-all` skips review. The same value from
the same session is never proposed twice.

Golden cases can now also have `as_of: <timestamp>` instead of `session` or `turns`: a fresh conversation at
that moment. That is how *new conversation* questions are stored, and you can write such cases by hand too.

### 12.2 Variants

A variant is a set of `config.yaml` overrides with dotted keys. `baseline` (plain Ollama behaviour:
full history, nothing injected) and `full` (your config as-is) always exist. Define more in
`evals\variants.yaml`; see `examples\variants.example.yaml`. The first variant in `--variant` is the
one the others are compared against.

### 12.2b Repeats and regressions

A golden set of a dozen questions can't tell a 5-point difference from noise. Two tools help:

- `--repeats N` asks each golden question N times with different seeds (at temperature 0.7, so the
  answers actually vary), replaying the session only once. Cells show e.g. `4/5`, and each variant's pass
  rate gets a **95% interval**. If two variants' intervals overlap, the difference may be noise: add cases
  or repeats before deciding.
- Every report is compared with the **previous report that ran the same variants**. A pass rate that fell
  below the previous run's interval, processed tokens up more than 10% on the same sessions, or more turns
  over `num_ctx` are flagged at the end of the report, and `ai eval run` exits with code 3. That makes it
  usable as a check after changing models or settings.

### 12.3 Isolation and fairness

- **No knowledge from the future (`--memory asof`, the default).** Replaying an old session against
  today's memory would let memory variants answer from facts learned *later*. Instead, each recorded
  session and golden question starts from memory **as it was at that moment**. That's reconstructed from
  the snapshots taken before every memory write: the state at time T is the first snapshot after T.
  Entries created after T are removed in any case. If snapshot retention has already pruned the relevant
  history, the case is listed as *approximate* in the report. History recall is held to the same rule: only
  excerpts from before the question, never from the question's own session. Scripted golden questions
  start from empty memory, since their facts are in their own turns. `--memory current` (today's memory)
  and `--memory empty` exist for comparison; the report says which was used.
- **Interleaved, shuffled order.** All variants are set up first, then each session and question is run on
  every variant in a seeded random order (`--seed`, recorded in the report). No variant is systematically
  first (cold caches) or last (warm).
- Each variant runs in its own workspace (`evals\runs\<timestamp>\<variant>\`) with a fresh database. Your
  real memory, database and logs are never written. Extraction is off unless you pass `--extract`.
- A unique marker at the start of each variant's system prompt stops one variant from reusing another's
  cached prefix.
- **Stop the orchestrator server** while evaluating (the harness warns if it's running), or at least avoid
  using it: live traffic on the same Ollama instances distorts cache and timing numbers.
- Your client's real system prompt and tool schemas aren't in the logs. Pass them with
  `--client-system file.txt` for realistic absolute numbers; relative comparisons are valid either way.

### 12.3b Paired comparison

Variants are judged on **the same questions and the same turns**, so the report compares them pair by
pair against the first variant, not just through two separate pass rates:

- **Accuracy:** for each question (and repeat), both pass, only one passes, or neither. Only the
  disagreements carry information, and McNemar's exact test turns them into a p-value. p < 0.05 means the
  difference is unlikely to be chance. This detects real differences with far fewer questions than
  comparing confidence intervals. Repeats of one question aren't fully independent, so p-values from
  `--repeats` runs are somewhat optimistic; more distinct questions beat more repeats.
- **Cost:** the median ratio of processed tokens per turn, and on how many turns the variant was cheaper.

The console's Evals tab shows the same paired table when you compare reports.

### 12.3c Why an answer passed or failed

For every golden answer, the harness records the exact prompt the primary received and traces the
expected answer (whatever the case checks with `expect_all` / `expect_any`) through the system:

| Diagnosis | Meaning | Look at |
|---|---|---|
| Answered from the prompt | passed; the answer was in front of the model | — |
| Answered without evidence | passed, but the answer wasn't in the prompt (known or guessed) | the question may be too easy |
| Model missed it | failed although the answer **was in the prompt** | model, thinking, how much else is in the prompt |
| Not retrieved | the answer was **in memory or an earlier conversation**, but didn't reach the prompt | `embeddings.min_similarity`, `history_recall.*`, memory budget |
| Lost from the session | said **earlier in this session**, then trimmed or compressed away, and the summary didn't keep it | `trim_target_ratio`, `session.summary_max_tokens`, digests |
| Never available | nothing the system had contained it | extraction, flags, or the question can't be answered |

"Available" follows the point-in-time rules (§12.3): memory as of the question, and past conversations from
before it. For passes, the report also lists *where* the answer came from (conversation, cached memory base,
per-turn memory, session summary, past exchange). **Memory reached the prompt** is a retrieval score: of the
answers that existed in memory or past conversations, how many made it into the prompt. In the console the
same table appears when comparing reports, and hovering a cell in the per-question grid shows its diagnosis.

### 12.4 Reading the report

Reports go to `evals\reports\<timestamp>.md` (plus `.json` with every turn). Rules of thumb:

- **Turns over `num_ctx` is the first column to read.** Those prompts didn't fit, so Ollama silently
  dropped the oldest content. A variant can look cheap on those turns precisely because it lost
  information. Keeping long sessions inside the window is what trimming and compression are for.
- Among variants with no overflow, prefer the one with fewer processed tokens **only if** its golden
  pass rate is no worse.
- When a variant loses questions, the diagnosis table (§12.3c) tells you which setting to look at first.
- What to expect, from synthetic sessions with a 16k window (run it on your own; that's the point):

  | Session | baseline | `full` (size mode) |
  |---|---|---|
  | 24 short chatty turns, fits the window | — | +10% processed tokens (mostly the one-time, then-cached system prompt) |
  | 24 tool-heavy turns, ~45k tokens | over `num_ctx` on 16 turns | never over; +27–34% processed tokens |
  | 60 tool-heavy turns, ~59k tokens | over `num_ctx` on 43 turns | never over; +33% (+17% with `trim_target_ratio: 0.4`) |

  The extra processing on long sessions is the price of staying inside the window: occasional full
  re-reads when the layout moves, plus the summary. Baseline looks cheaper there only because Ollama is
  silently throwing content away.
- Estimated cache hit is calibrated on each run's first (cold) request and is approximate. Processed
  tokens and prefill time are exact.

For day-to-day monitoring without a full eval, `/metrics` shows the same per-request fields from live
traffic. `ai memory changes` (rejection rate), `ai memory sessions` (summary quality) and
`ai memory review` (unused entries) cover the memory side.

## 13. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Port 11434 already in use | The Ollama tray app or the stock service is running; stop it, or use `-Force` |
| A model is under 100% GPU | Context too large for the VRAM: lower `num_ctx`, use `q8_0` KV cache or a smaller model |
| Both models on the same GPU | Instances not pinned. Check `-Status` / `nvidia-smi`, and pin by UUID |
| Client works but no memory appears | Client uses `/v1` (OpenAI mode); switch it to the Ollama provider |
| Client in WSL can't reach :8000 | Mirrored networking not enabled (§7.3) |
| Memory changes are mostly rejected | Memory model too small or thinking mode on; set `think: false` or try a larger model |
| `memory file errors` in status | A hand edit broke the format; run `ai memory validate` and fix the reported line |
| Consolidation never runs | Check `GET /memory/consolidation`: it states the reason (not idle, queue busy, ran recently, too few changes) |
| History never gets trimmed | In size mode that's normal while the prompt fits `num_ctx`. If `context_over_budget` shows up, the summary isn't keeping up (check `ai memory tasks`) |
| Client compacts or refuses at small context | Its own context precheck; raise `num_ctx` and the client's context window together if VRAM allows |

## 14. Limitations and ideas

- **Not built yet:** multi-project namespaces (the schema already has
  `project_id`).
- A large second GPU could take on more: embeddings, a bigger memory model, or splitting one large
  primary model across both cards instead. That trades the memory system for raw model size.
- Tested with fake Ollama instances (244 tests: validator, atomic writes, streaming, flag stripping,
  trimming, compression and memory-base cache stability, consolidation guards, evaluation harness, setup helpers, thinking decisions, token calibration, correction capture, hybrid retrieval, history recall, repeat runs, the console's backend, point-in-time memory, paired statistics, answer diagnosis, question generation, reference libraries, MCP, tool access control, retries…). Real-GPU behaviour (pinning, VRAM fit, a given
  model's JSON quality) can only be verified on your machine (§4.5); the evaluation harness (§12) is how you do that.
