"""Configuration loading.

Everything (paths, models, endpoints, limits) comes from config.yaml.
AI_ROOT overrides paths.root; AI_CONFIG selects a different config file.
"""
from __future__ import annotations

import os
from pathlib import Path, PureWindowsPath
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

APP_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = APP_DIR.parent / "config" / "config.yaml"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PathsConfig(_Strict):
    root: str
    memory: str = "memory"
    conversations: str = "conversations"
    database: str = "database/memory.db"
    logs: str = "logs"
    backups: str = "backups"
    prompts: str = "prompts"


class OllamaEndpoint(_Strict):
    base_url: str
    model: str
    timeout_seconds: float = 600
    num_ctx: int | None = None
    keep_alive: str | None = None
    think: bool | None = None


class OllamaConfig(_Strict):
    primary: OllamaEndpoint
    memory: OllamaEndpoint


class MemoryConfig(_Strict):
    project_id: str = "default"
    max_context_tokens: int = Field(2500, gt=0)
    max_memory_file_tokens: int = 4000
    max_entry_tokens: int = 600
    enable_semantic_retrieval: bool = True   # hybrid keyword + vector search (needs embeddings)
    asynchronous_updates: bool = True
    create_backups: bool = True
    history_versions_per_file: int = 50
    max_entry_chars: int = 1500
    min_confidence: float = 0.6
    trigger_mode: Literal["heuristic", "always"] = "heuristic"
    extractor_memory_tokens: int = 1500
    extractor_interaction_tokens: int = 3000
    worker_enabled: bool = True
    worker_poll_seconds: float = 2.0
    max_attempts: int = 5
    retry_base_seconds: float = 30.0
    # Background jobs nobody waits on may let the memory model think (better judgement).
    # Summaries and digests stay fast: the next prompt may need them.
    think_extraction: bool = True
    think_consolidation: bool = True


class ConversationConfig(_Strict):
    retain_raw_history: bool = True
    format: Literal["jsonl"] = "jsonl"
    recent_turns: int = Field(2, ge=0)   # short-term window for POST /chat (memory covers the rest)
    capture_corrections: bool = True     # your corrections become golden-question candidates


class ProxyConfig(_Strict):
    enabled: bool = True
    inject_memory: bool = True
    append_system_prompt: bool = True
    queue_memory_updates: bool = True
    # How history is kept inside the context window:
    #  "size"  - only when the estimated prompt would not fit num_ctx: first compress old tool
    #            output, then cut old turns back to trim_target_ratio of the window (default)
    #  "turns" - stepped by turn count (trim_trigger_user_turns / trim_keep_user_turns)
    trim_mode: Literal["size", "turns"] = "size"
    trim_target_ratio: float = Field(0.5, gt=0.1, lt=1.0)
    reply_reserve_tokens: int = Field(1024, ge=0)
    calibrate_tokens: bool = True     # learn chars/token per model from Ollama's own counts
    # Stepped trimming: once history exceeds trim_trigger_user_turns, cut it back to
    # about trim_keep_user_turns. The cut point then stays fixed for several turns,
    # so Ollama's prompt cache keeps hitting. 0 disables trimming.
    trim_trigger_user_turns: int = Field(10, ge=0)
    trim_keep_user_turns: int = Field(4, ge=1)
    # Never cut turns the session summary does not cover yet.
    trim_requires_summary: bool = True

    @model_validator(mode="after")
    def _check_trim(self):
        if self.trim_trigger_user_turns and self.trim_trigger_user_turns <= self.trim_keep_user_turns:
            raise ValueError("proxy.trim_trigger_user_turns must be greater than trim_keep_user_turns (or 0)")
        return self


class FlagsConfig(_Strict):
    enabled: bool = True
    max_per_turn: int = Field(3, ge=1, le=10)
    max_chars: int = Field(300, ge=20, le=1000)


class CompressionConfig(_Strict):
    """Replace old, large tool results with helper-written digests (stepped, cache-friendly)."""
    enabled: bool = True
    keep_recent_user_turns: int = Field(2, ge=1)   # tool results this recent always stay verbatim
    min_result_tokens: int = Field(400, ge=50)     # smaller results are not worth a digest
    digest_max_tokens: int = Field(200, ge=40, le=1000)
    step_turns: int = Field(6, ge=1)               # schedule used before/without trimming
    never_compress_tools: list[str] = Field(default_factory=list)


class StableMemoryConfig(_Strict):
    """Rarely-changing memory goes into the system prompt, frozen per epoch, so the
    prompt cache reaches it. Changes in between arrive as small per-turn updates."""
    enabled: bool = True
    categories: list[Literal["constraint", "objective", "environment", "state",
                             "decision", "lesson", "discovery"]] = \
        Field(default_factory=lambda: ["constraint", "objective", "environment"])
    max_tokens: int = Field(1000, ge=100)        # comes out of memory.max_context_tokens
    refresh_turns: int = Field(6, ge=1)          # used only when trimming and compression are off


class ConsolidationConfig(_Strict):
    """Idle-time memory hygiene: merge near-duplicates, tighten overgrown entries,
    list never-used entries for human review (never auto-deleted)."""
    enabled: bool = True
    idle_minutes: float = Field(10, gt=0)          # no requests for this long = idle
    min_interval_hours: float = Field(12, ge=0)     # at most one automatic run per interval
    min_changes_since_last: int = Field(5, ge=0)    # and only if memory changed this much since
    similarity_threshold: float = Field(0.5, gt=0, le=1)  # pre-filter for merge candidates
    rewrite_min_tokens: int = Field(120, ge=30)     # entries longer than this may be tightened
    rewrite_max_ratio: float = Field(0.8, gt=0, le=1)     # a rewrite must be at most this long
    identifier_coverage: float = Field(0.9, gt=0, le=1)   # share of names/numbers/paths to keep
    max_changes_per_run: int = Field(10, ge=1)
    max_cluster_size: int = Field(6, ge=2)
    stale_after_days: int = Field(30, ge=1)         # review list: active but never injected


class ThinkingConfig(_Strict):
    """Per-turn thinking for the primary. auto = turn it off only for clearly simple turns."""
    mode: Literal["auto", "client", "on", "off"] = "auto"
    simple_max_words: int = Field(25, ge=1)


class EmbeddingsConfig(_Strict):
    """Embedding model (on the memory GPU by default) for semantic memory and history recall."""
    enabled: bool = True
    model: str = "nomic-embed-text"
    base_url: str | None = None          # default: ollama.memory.base_url
    query_prefix: str | None = None      # None = automatic for known models (nomic: "search_query: ")
    document_prefix: str | None = None
    timeout_seconds: float = Field(3.0, gt=0)   # query embedding in the request path; falls back to keywords
    min_similarity: float = Field(0.35, ge=0, le=1)  # vector-only memory hits below this are ignored


class HistoryRecallConfig(_Strict):
    """Inject short verbatim excerpts of older exchanges that are no longer in the prompt."""
    enabled: bool = True
    max_tokens: int = Field(600, ge=0)
    top_k: int = Field(2, ge=1, le=10)
    min_similarity: float = Field(0.6, ge=0, le=1)
    cue_min_similarity: float = Field(0.45, ge=0, le=1)  # when the prompt says "last time", "earlier", ...
    chunk_chars: int = Field(1500, ge=200)


class ReferencesConfig(_Strict):
    """Documentation and code libraries the model can consult (README: reference libraries)."""
    enabled: bool = True
    chunk_tokens: int = Field(350, ge=80, le=2000)
    half_precision: bool = True            # store vectors as float16: half the RAM, same ranking in practice
    min_similarity: float = Field(0.35, ge=0, le=1)     # vector-only hits for searches the model asks for
    auto_max_tokens: int = Field(800, ge=0)             # automatic excerpts per turn (0 = never automatic)
    auto_top_k: int = Field(3, ge=1, le=10)
    auto_min_similarity: float = Field(0.55, ge=0, le=1)  # stricter: nobody asked for these
    mcp_enabled: bool = True               # docs_search / docs_lookup tools at /mcp
    tool_max_tokens: int = Field(1500, ge=200, le=8000)  # cap on one tool result


class SessionConfig(_Strict):
    summaries_enabled: bool = True
    summary_max_tokens: int = Field(400, ge=50, le=2000)


class ApplicationConfig(_Strict):
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"


class Config(_Strict):
    paths: PathsConfig
    ollama: OllamaConfig
    memory: MemoryConfig = MemoryConfig()
    conversation: ConversationConfig = ConversationConfig()
    proxy: ProxyConfig = ProxyConfig()
    flags: FlagsConfig = FlagsConfig()
    session: SessionConfig = SessionConfig()
    compression: CompressionConfig = CompressionConfig()
    stable_memory: StableMemoryConfig = StableMemoryConfig()
    consolidation: ConsolidationConfig = ConsolidationConfig()
    thinking: ThinkingConfig = ThinkingConfig()
    embeddings: EmbeddingsConfig = EmbeddingsConfig()
    history_recall: HistoryRecallConfig = HistoryRecallConfig()
    references: ReferencesConfig = ReferencesConfig()

    @model_validator(mode="after")
    def _check_compression(self):
        if (self.compression.enabled and self.proxy.trim_trigger_user_turns
                and self.compression.keep_recent_user_turns >= self.proxy.trim_keep_user_turns):
            raise ValueError("compression.keep_recent_user_turns must be smaller than "
                             "proxy.trim_keep_user_turns, or compression would never apply")
        if self.stable_memory.enabled and self.stable_memory.max_tokens >= self.memory.max_context_tokens:
            raise ValueError("stable_memory.max_tokens must be smaller than memory.max_context_tokens "
                             "(the base is part of that budget)")
        return self
    application: ApplicationConfig = ApplicationConfig()

    # Populated by load_config; not part of the YAML.
    source_path: str | None = None

    # ---- resolved paths -------------------------------------------------
    def _resolve(self, value: str) -> Path:
        p = _native_path(value)
        if p.is_absolute():
            return p
        return self.root_dir / p

    @property
    def root_dir(self) -> Path:
        return _native_path(self.paths.root)

    @property
    def memory_dir(self) -> Path:
        return self._resolve(self.paths.memory)

    @property
    def history_dir(self) -> Path:
        return self.memory_dir / "history"

    @property
    def conversations_dir(self) -> Path:
        return self._resolve(self.paths.conversations)

    @property
    def database_path(self) -> Path:
        return self._resolve(self.paths.database)

    @property
    def logs_dir(self) -> Path:
        return self._resolve(self.paths.logs)

    @property
    def backups_dir(self) -> Path:
        return self._resolve(self.paths.backups)

    @property
    def prompts_dir(self) -> Path:
        p = self._resolve(self.paths.prompts)
        if p.exists():
            return p
        # Fall back to the prompts shipped next to the application code.
        return APP_DIR.parent / "prompts"

    def ensure_dirs(self) -> None:
        for d in (self.memory_dir, self.history_dir, self.conversations_dir,
                  self.database_path.parent, self.logs_dir, self.backups_dir):
            d.mkdir(parents=True, exist_ok=True)

    def prompt(self, name: str) -> str:
        return (self.prompts_dir / name).read_text(encoding="utf-8").strip()


def _native_path(value: str) -> Path:
    """Accept Windows-style paths in YAML even when running elsewhere (tests, WSL)."""
    if os.name != "nt" and ("\\" in value or (len(value) > 1 and value[1] == ":")):
        win = PureWindowsPath(value)
        if win.drive:
            # G:\AI -> /mnt/g/AI when running under WSL/Linux.
            drive = win.drive.rstrip(":").lower()
            return Path("/mnt", drive, *win.parts[1:])
        return Path(*win.parts)
    return Path(value)


def load_config(path: str | os.PathLike | None = None) -> Config:
    cfg_path = Path(path or os.environ.get("AI_CONFIG") or DEFAULT_CONFIG_PATH)
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if os.environ.get("AI_ROOT"):
        data.setdefault("paths", {})["root"] = os.environ["AI_ROOT"]
    cfg = Config(**data)
    cfg.source_path = str(cfg_path)
    return cfg
