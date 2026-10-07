# AGENTS.md

Guide for AI agents (and humans) continuing work on this repository. Read this first, then
`docs/DOCUMENTATION.md` for user-facing behaviour. Everything here reflects **version 0.17.0**.

---

## 1. What this project is

A local orchestrator that turns two Ollama instances on two GPUs into one coding assistant for long
**agentic game-development sessions** (Unity first; nothing in the code is engine-specific):

- **GPU 1, primary model** (reference: qwen3:14b on an RTX 5070, 12 GB): does the actual work.
- **GPU 2, memory model** (reference: qwen3:8b + nomic-embed-text on an RTX 2070 SUPER, 8 GB): works in the
  background. It extracts and consolidates project memory, writes session summaries and tool-output digests,
  embeds memory, history and documentation, and words generated test questions.

The orchestrator is a FastAPI server on `127.0.0.1:8000` that looks like an Ollama server to clients (OpenClaw
in WSL is the reference client). It injects memory and documentation into requests, keeps prompts inside
`num_ctx`, and queues background work. It also serves an MCP server (`/mcp`) with documentation tools and a web
console (`/ui`).

**Reference environment of the owner:** Windows, 16 GB DDR4, OpenClaw gateway in a WSL distro. The second GPU
may or may not be installed yet; single-GPU mode exists for that case. The seed files in `memory/*.md`
describe this setup. They are example data, and other users replace them.

**Direction:** agentic game building in Unity: long OpenClaw coding sessions, engine docs and packages indexed
as reference libraries, project conventions held in memory.

---

## 2. History: how it got here and why

Each step below names the problem it solved. **Bold** items were found by tests or by the evaluation harness,
not by design. They are worth knowing because they show where the system is fragile.

| Version | What was added | Why / what we learned |
|---|---|---|
| 0.1–0.2 | Phases 1–2 of the original spec: Markdown memory (7 categories, stable ids), strict validator, SQLite (mirror, audit, task queue), JSONL raw history, background worker with retries, CLI, backups, rebuild. Ollama-compatible proxy so OpenClaw uses the orchestrator as its Ollama provider. | Spec assumed a 16 GB card; adapted to 12 GB (16k ctx, q8_0 KV). OpenClaw talks native `/api/chat` with streaming and tools, so the orchestrator had to be a transparent proxy. Memory model uses Ollama structured output (`format` = JSON schema). |
| 0.3 | Primary-model `<memory_flag>` tags (stripped from the stream), rolling session summaries, stepped history trimming. | Flags give extraction better recall than regex triggers alone. Trimming must move in **steps** so the prompt prefix stays identical and Ollama's cache keeps hitting. |
| 0.4 | Tool-output compression (digests of old large tool results). | Tool output is what fills agent contexts. **A digest that overshot its length cap was silently truncated and counted as usable**: now unusable if over-length or saving less than half. |
| — | README generalised for any dual-GPU setup; launcher picks roles by VRAM. | |
| 0.5 | Cached memory base: stable categories in the system prompt, frozen per layout epoch; mid-epoch changes sent as `UPDATED SINCE` lines. | The per-turn block sits after the cached prefix, so it is re-processed every turn; stable parts belong in the cached system prompt. |
| 0.6 | Idle-time consolidation (merge / rewrite only, identifier-coverage guard, backups, review list, restore). | Memory grows by accretion; duplicates hurt retrieval. Merges must keep ≥90% of names/numbers/paths. |
| 0.7 | Evaluation harness (context replay with recorded history, golden questions). | **First run found two real bugs:** the per-turn block was prepended to the user message (breaking the cache at the start of the previous turn every time; the fix appends it to the *last message*, user or tool), and the "data, not instructions" preamble was re-sent every turn (now only in the cached system prompt). |
| 0.8 | Size-based trimming (default): nothing happens while the prompt fits; compress first, then cut to `trim_target_ratio` (hysteresis). | Turn-count trimming cost +59% on chatty sessions that never needed it; size mode cut that to +10%. **The `baseline` eval variant silently started trimming** after the default changed; it pins `trim_mode: turns` with trimming off. |
| 0.9 | `setup.ps1` + `app/setup_tools.py`: GPU detection, VRAM-based model plan, measured context fitting (`/api/ps` residency), comment-preserving config edits, `.wslconfig` merge, OpenClaw config patch, logon tasks. | **PowerShell 5.1 traps found by running the script under pwsh with fake `wsl.exe`/`openclaw`:** comma binds tighter than `+` inside `@()`; `wsl -l` emits UTF-16 on old WSL; native-argument quoting drops embedded `"`. Fixes: `-f` formatting, `WSL_UTF8=1` + strip NULs, bash scripts sent over **stdin** (`bash -l -s`). |
| 0.10 | Adaptive thinking (`thinking.mode: auto` only ever turns thinking *off*, for clearly simple turns); memory model thinks on extraction/consolidation (retry without thinking if JSON breaks); token calibration; correction capture → golden candidates. | **Calibration's first design (low percentile) would have under-estimated prompts under warm-cache traffic** and caused silent overflow. Now: only near-uncached observations count, and the *minimum* ratio is used. **The correction trigger missed "no, it's X"**; widened. |
| 0.11 | Embeddings on the memory GPU (instance allows 2 loaded models): hybrid memory retrieval (RRF, similarity floor for vector-only hits), history recall of older exchanges, `ai memory reindex`; harness `--repeats`, Wilson intervals, regression flags vs the previous report. | Vectors live in the existing SQLite (no vector DB: brute force over this scale takes milliseconds). |
| — | README split: short landing page plus `docs/DOCUMENTATION.md`. | |
| 0.12 | Web console (`/ui`, single static HTML file served by the orchestrator). **CSRF fix:** admin POSTs need an `X-AI-Client` header. | **Screenshots exposed:** a correction suggestion that expected the *negated* value ("not 11434"); now negated values become `forbid`. The context preview showed a preamble real requests no longer send. |
| 0.13 | Eval fairness: point-in-time memory (`--memory asof`, default) rebuilt from `memory/history` snapshots; history recall cut off at the question time; McNemar paired test; seeded interleaved variant order. | Replaying old sessions against today's memory leaked future knowledge and flattered memory variants. |
| 0.14 | Answer diagnosis: answered from prompt / without evidence / model missed / not retrieved / lost from session / never available; memory retrieval rate. | **The prompt capture silently never worked**: a `str.replace` anchor also matched inside `touch()`, which resets on every request. |
| 0.15 | Generated golden questions (needle mining + model wording + fill-in-the-blank fallback); `as_of` golden cases (fresh conversation at a past moment). | Same-session questions only test memory once a session no longer fits the window; `new-session` questions are the real cross-session test. |
| 0.16 | Reference libraries (`ai docs`, Library tab): chunking (C# per member), FTS5 + per-library float16 vectors, hybrid search, exact symbol lookup; MCP server at `/mcp` (`docs_search`, `docs_lookup`, `docs_libraries`), registered in OpenClaw by setup; automatic `<REFERENCE>` excerpts; full CLI↔console parity. | **C# members weren't chunked** (scope closed before the next-line `{`). **The excerpt budget was reserved even with no automatic library.** **A route collision** (`/ui/api/evals/{name}` swallowing `/sessions`); the list moved to `/ui/api/recorded-sessions`. |

| 0.17 | Tool access control (`app/tool_acl.py`, Tools tab, `ai tools`): tools registered from request traffic, grouped by source (prefix heuristics + rules + overrides), per-source/per-tool modes off / automatic / on_demand; `load_tools` meta-tool answered by the proxy in an internal round (streaming and non-streaming); calls to off tools blocked in responses. | The proxy already sees every tool definition the model gets, so it can rewrite the list; a Unity MCP server's dozens of schemas cost a large share of a 16k window. Blocking in the response makes "off" real access control, not just hiding. |

---

## 3. Architecture

### 3.1 Request path (`POST /api/chat` from OpenClaw)

1. `proxy.api_chat` → `prepare()`: conversation id from `X-Conversation-Id` or a hash of model + first user message.
2. **Turn plan** (`plan_for`, cached per conversation + turn + user text, reused for every tool-call step of a turn):
   - layout: `size_layout` (default) or `stepped_drop` → `drop` (user turns cut) and compression `boundary`;
     cuts never exceed what the session summary covers;
   - frozen digest set for that layout (`frozen_digests`);
   - memory base snapshot for the layout epoch (`orch.memory_base`);
   - query embedding (once per turn, timeout → keywords only);
   - per-turn memory block (`ContextBuilder.build`: updates since the base, session summary, history recall,
     relevant entries) and automatic reference excerpts (`orch.reference_block`);
   - thinking decision (`thinking.decide`).
3. Tools: `ToolACL.observe` registers the client's tools; `ToolACL.effective` drops *off* tools, keeps
   *automatic* and already-loaded ones, and replaces the remaining *on-demand* tools with one `load_tools`
   catalog tool. The size estimate uses this filtered list.
4. Messages: trimmed → compressed → system prompt + primary_system (+ flags instruction) + memory base appended
   to the client's system message → memory/reference block **appended to the last message** → `num_ctx`
   default → `think` if decided.
5. Response relayed (streaming or not): `<memory_flag>` tags stripped (`FlagStripper`); tool calls classified
   (`ToolACL.classify_calls`): calls to *off* tools removed with a note, a `load_tools` call answered by the
   proxy and the request continued with a follow-up round (the client sees one response).
6. `finalize`: metrics, token calibration, correction capture, and on a final answer (no tool calls):
   `record_turn` (JSONL) + `queue_memory` (summary 10 → embed 8 → digests 5 → extraction 0, by priority).

`POST /chat` (`Orchestrator.chat`) is the simpler non-proxy path with the same memory machinery.

### 3.2 Module map (`app/`)

| Module | Responsibility |
|---|---|
| `api.py` | App factory, lifespan, core endpoints, CSRF middleware, router wiring |
| `proxy.py` | Ollama-compatible proxy, turn plans, layouts, size estimation, flag stripping, finalize |
| `orchestrator.py` | Service container; `/chat` flow; memory base cache; queueing; correction capture; reference block |
| `config.py` | Pydantic config (`extra="forbid"`), cross-section validators, path resolution |
| `markdown_store.py` | Canonical memory files: parse/render, atomic write with history snapshot |
| `memory_manager.py` | Applies validated changes (Markdown → SQLite mirror → audit), backups, restore, validation |
| `memory_validator.py` | Untrusted-output validation and `safety_issue` rules (paths, injection, commands, secrets) |
| `memory_worker.py` | Persistent task queue: extraction, summaries, digests, embeddings; retries; idle hook |
| `context_builder.py` | Memory base snapshots and per-turn block under a hard token budget |
| `memory_retriever.py` | Keyword retriever + `HybridRetriever` (RRF) |
| `embeddings.py` | Embedder client, `VectorIndex`, `Indexer` (memory/history sync, recall, backfill) |
| `references.py` | Reference libraries: chunkers, ingest jobs, FTS5 + vector search, lookup, excerpts |
| `mcp.py` | MCP Streamable HTTP server for the docs tools |
| `tool_acl.py` | Tool registry, sources/rules, modes, `load_tools` catalog, call classification |
| `compression.py` | Tool-result digest boundary and application |
| `flags.py` | Streaming-safe `<memory_flag>` extraction |
| `thinking.py` | Per-turn thinking classifier/decision |
| `calibration.py` | Chars-per-token learning (conservative) |
| `consolidation.py` | Idle-time merges/rewrites with guards; review list; `identifiers()` |
| `triggers.py` | Heuristics for when to extract memory (EN + PT) |
| `evaluation.py` | Harness: variants, replay, golden runs, stats (Wilson, McNemar), reports, candidate accept |
| `eval_asof.py` | Point-in-time memory reconstruction |
| `eval_diagnose.py` | Why an answer passed/failed |
| `eval_generate.py` | Generated golden questions |
| `ui.py` + `ui/index.html` | Console backend + single-file frontend (vanilla JS, no build, works offline) |
| `setup_tools.py` | Testable logic behind `setup.ps1` |
| `cli.py` | `ai` command (`ai.cmd` on Windows, `python -m app.cli` elsewhere) |
| `database.py`, `conversation_logger.py`, `metrics.py`, `ollama_client.py`, `logging_setup.py`, `rebuild.py`, `schemas.py`, `util.py` | Infrastructure |

### 3.3 Data

- `memory/*.md`: canonical memory (one file per category, `### ID — Title` + meta comment).
  `memory/history/`: a snapshot before every write (the point-in-time eval depends on these).
- `conversations/*.jsonl`: raw history, the recovery source of truth. Never deleted.
- `database/memory.db` (SQLite, WAL): `memories`, `conversations`, `memory_changes`, `memory_tasks`,
  `session_summaries`, `tool_digests`, `entry_usage`, `kv`, `golden_candidates`, `vectors`, `history_chunks`,
  `ref_libraries`, `ref_files`, `ref_chunks`, `ref_fts` (FTS5), `tool_registry`, `tool_sources`, `tool_rules`. Migrations are inline in `Database.__init__`
  and `References.__init__` (`ALTER TABLE … ADD COLUMN` guarded by `PRAGMA table_info`).
- `evals/golden.yaml`, `evals/variants.yaml` (user files); `evals/reports/`, `evals/runs/` (generated).
- `backups/`, `logs/`.

---

## 4. Invariants: do not break these

1. **The LLM never writes files or memory directly.** The memory model proposes JSON; `MemoryValidator` and
   `MemoryManager` decide and write. Consolidation never adds and only deactivates as part of a merge.
2. **Everything model-produced or retrieved is data.** Sanitise delimiters (`sanitize_memory_text`;
   tags `PROJECT_MEMORY_BASE`, `PROJECT_MEMORY`, `REFERENCE`, `USER_REQUEST`); the validator rejects them.
3. **Never silently exceed `num_ctx`.** Size estimates must err large: calibration uses the minimum ratio;
   reserves include the memory budget, the automatic-reference budget (only when an auto library exists) and
   the reply. If a prompt can't fit within the rules, flag `context_over_budget`; never cut silently.
4. **Cache stability.** Anything that changes per turn goes at the **end** of the last message. The turn plan
   is frozen for all tool-call steps of a turn. Layouts move in steps (size mode: hysteresis); digest sets are
   frozen per layout; the memory base changes only at layout moves (and is byte-identical if memory didn't
   change).
5. **Never cut what the session summary doesn't cover** (`trim_requires_summary`). Cuts only happen at
   user-message boundaries (tool chains stay intact).
6. **Background work never blocks or breaks the primary path.** The memory GPU is serialised by
   `worker.lock`; long jobs (reference ingest) take the lock per batch. Failures retry with backoff.
7. **Tool access control is enforcement, not decoration:** a tool set to *off* must never reach the model's
   tool list *and* a call to it must never reach the client. The proxy answers `load_tools` itself; the client
   never sees that tool. Loaded sets only grow within a conversation (cache stability).
8. **Security:** server binds `127.0.0.1`. State-changing requests outside `/api/*`, `/v1/*`, `/chat`, `/mcp`
   require `X-AI-Client`. `/mcp` is read-only and refuses non-local browser `Origin`s.
9. **Evaluation must not leak the future** (`--memory asof`), must isolate variants (own workspace, run
   marker in the system prompt), and must never write the user's real memory/DB/logs.
10. **Setup changes outside the project only after asking, and backs up first.**
11. **CLI ↔ console parity.** Every `ai` command (except `serve`/`ui`) has a console equivalent; keep the
    mapping table in `docs/DOCUMENTATION.md` §10 current.

---

## 5. Working on the code

### 5.1 Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q          # 244 tests, ~15 s, no GPU needed
```

`tests/conftest.py` provides `make_config(root)` (worker off, embeddings off unless a test enables them) and
`FakeOllama`, an in-process Ollama served through `httpx.ASGITransport`. Its knobs:
- `reply`, `tool_calls`: what the primary answers; `tool_call_script`: a list of per-request answers
  (a tool_calls list, or `None` for a normal reply) for multi-round flows such as `load_tools`;
- `memory_json`: structured-output answers;
- `fail_status`: return an HTTP error;
- `simulate_cache`: `prompt_eval_count` counts only characters after the prefix shared with the previous
  request (/4), so cache effects are testable;
- `ps_models`, `vram_fits_at`: `/api/ps` residency for context fitting;
- `/api/embed` with `fake_embedding()`: deterministic bag-of-words plus a small synonym table;
  `embed_fail` makes it error.

Fakes prove mechanics, not model quality. Similarity thresholds that pass with the fake embedder don't
transfer to real models.

Patterns used throughout:
- **End-to-end through the real app:** `create_app(cfg, primary_transport=..., memory_transport=...)` +
  `TestClient`. Drive the worker with `client.portal.call(orch.worker.process_next)`.
- **Harness tests** call `evaluation.run_eval(...)` with fake transports.
- **Every bug fix gets a test that would have caught it.**

### 5.2 Other checks before shipping

- **PowerShell:** parse every `.ps1` with the PowerShell AST parser (pwsh works on Linux). Behaviour of
  `setup.ps1` sections can be exercised under pwsh with fake `wsl.exe` / `openclaw` scripts on `PATH`
  (extract a section between its step comments and run it with prepared variables).
- **Console JavaScript:** extract the `<script>` block and run `node --check`.
- **Console visuals:** run the server against fake Ollama servers (uvicorn) and screenshot each tab with
  Playwright/Chromium. Check for page errors, then *look* at the screenshots. Several real bugs were only
  visible there.
- Bump `app/__init__.py` `__version__`; update `docs/DOCUMENTATION.md` (feature section, CLI/API tables, the
  test count line in Limitations), `README.md` if user-visible, and this file's history table.
- Package without `conversations/ database/ logs/ backups/ evals/ __pycache__/ config/*.bak-*`.

### 5.3 Conventions

- Python 3.11+, type hints, `from __future__ import annotations`. Pydantic models use `extra="forbid"`.
- New config goes in `config.py` *and* `config/config.yaml` (with a comment), plus `ui.py` `SETTINGS` if it
  is something a user tunes. Mark `live: True` only if the value is read at request time.
- Console: plain HTML/CSS/JS in one file. Colour encodes the lane: cobalt (`--primary`) for primary-side
  data, teal (`--memory`) for memory-side; ochre/brick only for warnings/errors. Sentence case, no
  icon-plus-heading card grids. Fonts: Bahnschrift (numbers/headings) and Segoe UI Variable, with fallbacks.
  All admin calls go through `api()`/`post()`, which add `X-AI-Client`.
- User-facing docs: plain language, no hype, explain *why* a setting exists and when to change it.

### 5.4 Lessons that cost time

- **Editing by string replacement:** check that the anchor is unique. A repeated anchor silently patched
  `touch()` as well (0.14).
- **PowerShell:** `,` binds tighter than `+` inside `@()`; build strings with `-f`. Never pass `"` through
  native arguments in Windows PowerShell 5.1; send scripts over stdin. Avoid PS7-only syntax (`??`, ternary).
- **FastAPI:** a path-parameter route (`/x/{name}`) declared earlier swallows a literal sibling (`/x/sessions`).
- **C# parsing:** the class body `{` is usually on the next line; scopes must open on entry, not declaration.
- **Statistics:** percentiles over mostly-warm cache traffic are not conservative; use the minimum over
  near-uncached samples. Repeats of one question are not independent, so p-values from repeats are optimistic.
- **Design notes can be wrong:** "same-session questions test memory" was false while sessions still fit.
  The diagnosis feature exists to catch exactly this.

---

## 6. Verification status

**Tested here (with fakes):** every feature above, 244 tests; `setup.ps1` sections under pwsh with fake
WSL/OpenClaw; console in headless Chromium.

**Not yet verified on real hardware, so check first on the owner's machine:**
- GPU pinning by UUID, two instances, residency at the fitted `num_ctx` (`setup.ps1` measures; `ai status` shows).
- Ollama `think` combined with `format` (JSON schema) for the memory model; the fallback exists, but the share
  of fallbacks is unknown.
- OpenClaw accepting the MCP server's plain-JSON Streamable HTTP responses: `openclaw mcp probe local-docs`.
- OpenClaw `config patch` / `config set` paths; Task Scheduler registration; tray-app autostart removal.
- Real embedding similarity thresholds (`embeddings.min_similarity`, `history_recall.*`,
  `references.auto_min_similarity`) are defaults, not tuned values.
- How reliably a 14B model uses `load_tools` (asks for the right tools, then calls them). If it under-uses
  it, keep the essential tools *automatic* and put only the long tail *on demand*.
- Tool naming of OpenClaw's MCP tools (the source heuristic assumes `server__tool`-style prefixes); fix with
  grouping rules if they arrive differently.
- Whether `thinking.mode: auto` costs accuracy on real work: run
  `ai eval run --think default --variant thinking-always,thinking-auto`.

---

## 7. Roadmap (open items, roughly by value)

1. **First real-hardware evaluation:** generated `new-session` questions plus `baseline` vs `full`. This turns
   the estimates in the docs into measurements and should precede most tuning.
2. **Relevance-gated memory budget:** shrink the per-turn block on simple prompts. It is the main remaining
   overhead on chatty sessions.
3. **Flag calibration:** compare `<memory_flag>` output against accepted/rejected changes; tune `primary_flags.txt`.
4. **Harness:** latency distributions (p50/p95), memory-GPU time per variant, dry-run time estimate,
   resume an interrupted run, LongMemEval/LoCoMo adapter, optional LLM judge for open-ended questions.
5. **Unity-specific:** connect a Unity Editor MCP server (official Unity CLI `unity mcp` on Unity 6 LTS+, or
   the community MCP for Unity) for the compile → console → test loop, managed through the Tools tab; add
   system-prompt guidance for retrying after domain reloads; golden-question templates for API usage and
   compile-fix tasks; a stronger C# chunker (e.g. Roslyn-based) if the regex one proves too coarse.
   Note: those servers are stdio processes on Windows, while OpenClaw runs in WSL; verify launching through
   WSL interop, or use an HTTP mode where available.
6. **Tools:** optional relevance-based auto-loading (embed tool descriptions; preload likely tools for a
   turn) if the model under-uses `load_tools`; per-conversation unload after long disuse.
7. **Speculative decoding** for the primary: needs llama.cpp `llama-server` instead of Ollama and an adapter
   in `ollama_client`/`proxy`.
8. **Multi-project namespaces:** the schema has `project_id`; stores and paths do not yet.

---

## 8. Continuing in a new session: checklist

1. Read this file, then skim `docs/DOCUMENTATION.md` §9 (how it works) and the relevant module.
2. Run the test suite; it must be green before and after your change.
3. Ask the owner what was observed on real hardware since 0.17.0 (setup output, `ai status`, any eval report).
   Real numbers outrank anything estimated here.
4. Make the change with tests, keep the invariants (§4), keep CLI↔console parity, update docs, bump the version,
   and add a row to the history table (§2) with *why*, not just *what*.
