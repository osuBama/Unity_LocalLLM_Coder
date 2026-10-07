"""Service container and the primary request flow (spec §15).

Used by both entry points:
  * POST /chat          - the spec's own API
  * POST /api/chat      - Ollama-compatible proxy for OpenClaw (see proxy.py)
"""
from __future__ import annotations

import logging
import time
import uuid
from collections import OrderedDict, deque

import httpx

from .config import Config
from .context_builder import ContextBuilder, StableSnapshot, wrap_user_request
from .flags import strip_flags
from .conversation_logger import ConversationLogger
from .database import Database
from .memory_manager import MemoryManager
from .memory_retriever import HybridRetriever, KeywordRetriever
from .memory_worker import MemoryWorker
from .metrics import Metrics
from .ollama_client import OllamaClient, OllamaError
from .schemas import Category, InteractionTask
from .util import estimate_tokens, now_iso

log = logging.getLogger("orchestrator")
plog = logging.getLogger("primary")


class Orchestrator:
    def __init__(self, config: Config, *, primary_transport: httpx.AsyncBaseTransport | None = None,
                 memory_transport: httpx.AsyncBaseTransport | None = None):
        config.ensure_dirs()
        self.config = config
        self.project_id = config.memory.project_id
        self.db = Database(config.database_path)
        self.conv_log = ConversationLogger(config.conversations_dir, config.conversation.retain_raw_history)
        self.manager = MemoryManager(config, self.db)
        self.manager.ensure_files()
        self.embedder = None
        self.indexer = None
        if config.embeddings.enabled:
            from .embeddings import Embedder, Indexer
            emb_url = config.embeddings.base_url or config.ollama.memory.base_url
            emb_transport = memory_transport if emb_url == config.ollama.memory.base_url else None
            self.embedder = Embedder(emb_url, config.embeddings.model,
                                     query_prefix=config.embeddings.query_prefix,
                                     document_prefix=config.embeddings.document_prefix,
                                     timeout=config.embeddings.timeout_seconds, transport=emb_transport)
        if self.embedder is not None and config.memory.enable_semantic_retrieval:
            from .embeddings import VectorIndex
            self.retriever = HybridRetriever(self.manager.stores, VectorIndex(self.db, self.embedder.model),
                                             config.embeddings.min_similarity)
        else:
            self.retriever = KeywordRetriever(self.manager.stores)
        self.context_builder = ContextBuilder(
            self.manager.stores, self.retriever, config.prompt("context_builder.txt"),
            config.memory.max_context_tokens, config.memory.max_entry_tokens)
        self.primary = OllamaClient(config.ollama.primary, transport=primary_transport)
        self.memory_client = OllamaClient(config.ollama.memory, transport=memory_transport)
        self.metrics = Metrics()
        from .calibration import TokenCalibrator
        self.calibrator = TokenCalibrator(self.db)
        self.worker = MemoryWorker(config, self.db, self.manager, self.memory_client,
                                   self.context_builder, self.metrics)
        if self.embedder is not None:
            from .embeddings import Indexer
            self.indexer = Indexer(self, self.embedder)
            if isinstance(self.retriever, HybridRetriever):
                self.indexer.index = self.retriever.index     # one shared, cache-invalidated index
            self.worker.indexer = self.indexer
            self.queue_embed()                                 # bring vectors up to date at startup
        self.primary_system = config.prompt("primary_system.txt")
        if config.flags.enabled:
            self.primary_system += "\n\n" + config.prompt("primary_flags.txt")
        self._recent: dict[str, deque] = {}
        self._turns: dict[str, int] = {}
        self._bases: OrderedDict[tuple, StableSnapshot] = OrderedDict()
        from .tool_acl import ToolACL
        self.tools_acl = ToolACL(self)
        self.references = None
        if config.references.enabled:
            from .references import References
            self.references = References(self)
        self.last_request_at = time.time()
        # Evaluation only: when set to a dict, the proxy records the exact messages sent to the
        # primary per conversation (for tracing where a golden answer's evidence was).
        self.capture: dict | None = None
        from .consolidation import Consolidator
        self.consolidator = Consolidator(self)
        self.worker.idle_hook = self.idle_work

    async def aclose(self) -> None:
        await self.worker.stop()
        await self.primary.aclose()
        await self.memory_client.aclose()
        if self.embedder is not None:
            await self.embedder.aclose()

    def queue_embed(self) -> None:
        """Queue a background vector sync (memory entries + new history chunks), once."""
        if self.indexer is not None and not self.db.has_pending_task("embed"):
            self.db.enqueue_task({"conversation_id": None}, kind="embed", priority=8)
            self.worker._wake.set()

    async def query_vector(self, text: str):
        if self.embedder is None or not text:
            return None
        return await self.embedder.query(text)

    def reference_block(self, query: str, qvec) -> tuple[str, list[dict]]:
        """Automatic documentation excerpts for this turn ('' if none qualify)."""
        if self.references is None:
            return "", []
        try:
            from .references import render_reference_block
            hits = self.references.auto(query, qvec)
            return render_reference_block(hits), hits
        except Exception:
            log.exception("reference search failed")
            return "", []

    def recall_history(self, query: str, qvec, conversation_id: str, in_prompt_after_turn: int) -> list[dict]:
        if self.indexer is None:
            return []
        try:
            return self.indexer.recall(query, qvec, conversation_id, in_prompt_after_turn)
        except Exception:
            log.exception("history recall failed")
            return []

    # ------------------------------------------------------------ memory
    async def queue_memory(self, task: InteractionTask) -> bool:
        """Queue the session-summary update and (if triggered) memory extraction."""
        self.worker.enqueue_summary(task)
        self.worker.enqueue_digests(task)
        if self.indexer is not None:
            self.indexer.add_history(task.conversation_id, task.turn_number, task.user_message,
                                     task.assistant_response, task.tool_events)
            self.queue_embed()
        _, queued, _ = self.worker.enqueue(task)
        if not self.config.memory.asynchronous_updates:
            # Synchronous mode (debugging): process inline, still isolated from errors.
            try:
                while await self.worker.process_next():
                    pass
            except Exception:
                log.exception("inline memory processing failed")
        return queued

    def memory_base(self, conversation_id: str, epoch: tuple) -> StableSnapshot | None:
        """Frozen memory base for this conversation and epoch (None when disabled).

        Rebuilt only when the epoch changes; if memory hasn't changed, the rebuilt
        text is byte-identical, so the cache still hits.
        """
        sm = self.config.stable_memory
        if not sm.enabled:
            return None
        key = (conversation_id, epoch)
        if key in self._bases:
            self._bases.move_to_end(key)
            return self._bases[key]
        snap = self.context_builder.build_stable([Category(c) for c in sm.categories], sm.max_tokens)
        self._bases[key] = snap
        while len(self._bases) > 256:
            self._bases.popitem(last=False)
        return snap

    def touch(self) -> None:
        """A client request arrived: idle-time work must yield."""
        self.last_request_at = time.time()

    def record_usage(self, entry_ids) -> None:
        try:
            self.db.record_usage([i for i in entry_ids if i not in ("SESSION", "UPDATES")
                                  and not str(i).startswith("HISTORY:")], self.project_id)
        except Exception:
            log.exception("usage recording failed")

    async def idle_work(self) -> bool:
        """Called by the worker when the queue is empty. Returns True if it did something."""
        due, why = self.consolidator.due()
        if not due:
            return False
        log.info("idle consolidation starting", extra={"detail": why})
        await self.consolidator.run(trigger="idle")
        return True

    def capture_correction(self, conversation_id: str, turn: int, user_message: str,
                           prev_question: str | None, prev_answer: str | None) -> int | None:
        """If this turn corrects the previous answer, save a golden-question candidate."""
        if not (self.config.conversation.capture_corrections and prev_question and prev_answer and turn >= 2):
            return None
        from .consolidation import identifiers
        from .triggers import _RULES
        if not _RULES["correction"].search(user_message or ""):
            return None
        import re as _re
        # "no, it's 11435, not 11434": what the user negates is what the answer must NOT say.
        negated = set()
        for m in _re.finditer(r"\b(?:not|isn['’]?t|wasn['’]?t|instead of|rather than|não|nao|em vez de)\s+"
                              r"((?:the\s+|a\s+|o\s+|a\s+)?\S+)", user_message, _re.I):
            negated |= identifiers(m.group(1))
        new = identifiers(user_message) - identifiers(prev_answer) - identifiers(prev_question)
        suggested = sorted(new - negated)
        forbid = sorted(negated & identifiers(prev_answer)) or sorted(negated)
        try:
            cid = self.db.add_candidate(conversation_id, turn, prev_question[:4000], prev_answer[:4000],
                                        user_message[:2000], suggested[:5], [f"/\\b{_re.escape(x)}\\b/" for x in forbid[:5]])
            log.info("golden candidate captured", extra={"conversation_id": conversation_id,
                                                         "detail": f"#{cid} turn {turn}"})
            return cid
        except Exception:
            log.exception("could not store golden candidate")
            return None

    def system_prompt(self, base: StableSnapshot | None) -> str:
        return self.primary_system + (f"\n\n{base.text}" if base and base.text else "")

    def session_summary(self, conversation_id: str) -> dict | None:
        if not self.config.session.summaries_enabled:
            return None
        return self.db.get_summary(conversation_id)

    def record_turn(self, *, conversation_id: str, user_message: str, assistant_response: str,
                    tool_events: list[dict], source: str, log_user: bool = True,
                    flags: list[dict] | None = None, turn_number: int = 0) -> InteractionTask:
        """Persist a finished turn to raw history and build the memory task."""
        self.db.touch_conversation(conversation_id, self.project_id)
        if log_user:
            self.conv_log.message(conversation_id, "user", user_message, project_id=self.project_id)
        for ev in tool_events:
            if ev.get("type") == "tool_call":
                self.conv_log.tool_call(conversation_id, ev.get("tool", "?"), ev.get("arguments"),
                                        project_id=self.project_id)
            elif ev.get("type") == "tool_result":
                self.conv_log.tool_result(conversation_id, ev.get("tool", "?"), ev.get("result"),
                                          project_id=self.project_id)
        extra = {"memory_flags": flags} if flags else {}
        self.conv_log.message(conversation_id, "assistant", assistant_response,
                              project_id=self.project_id, **extra)
        return InteractionTask(conversation_id=conversation_id, timestamp=now_iso(),
                               user_message=user_message, assistant_response=assistant_response,
                               tool_events=tool_events, project_id=self.project_id, source=source,
                               flags=list(flags or []), turn_number=turn_number)

    # -------------------------------------------------------------- /chat
    async def chat(self, message: str, conversation_id: str | None = None) -> dict:
        request_id = uuid.uuid4().hex[:12]
        conversation_id = conversation_id or uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        self.touch()
        self.db.touch_conversation(conversation_id, self.project_id)
        self.conv_log.message(conversation_id, "user", message, project_id=self.project_id,
                              request_id=request_id)

        if conversation_id not in self._turns:
            prev = self.db.get_summary(conversation_id)
            self._turns[conversation_id] = prev["covered_turns"] if prev else 0
        self._turns[conversation_id] += 1
        turn_number = self._turns[conversation_id]
        summary = self.session_summary(conversation_id)
        base = self.memory_base(conversation_id, ("chat", turn_number // self.config.stable_memory.refresh_turns))
        budget = self.config.memory.max_context_tokens - (base.tokens if base else 0)
        qvec = await self.query_vector(message)
        history = self.recall_history(message, qvec, conversation_id,
                                      max(0, turn_number - 1 - self.config.conversation.recent_turns))
        # /chat never resends full history, so the summary is always useful once it exists.
        ctx = self.context_builder.build(message, max_tokens=budget, base=base, preamble=False,
                                         session_summary=summary["summary"] if summary else None,
                                         query_vec=qvec, history=history)
        ref_text, ref_hits = self.reference_block(message, qvec)
        messages = [{"role": "system", "content": self.system_prompt(base)}]
        self.record_usage(list(ctx.included) + list(base.fingerprints if base else ()))
        recent = self._recent.setdefault(conversation_id, deque(maxlen=max(1, self.config.conversation.recent_turns) * 2))
        if self.config.conversation.recent_turns:
            messages.extend(recent)
        block = "\n\n".join(x for x in (ctx.text, ref_text) if x)
        messages.append({"role": "user", "content": wrap_user_request(block, message)})

        rec = {"request_id": request_id, "conversation_id": conversation_id, "mode": "chat",
               "primary_model": self.config.ollama.primary.model,
               "memory_model": self.config.ollama.memory.model,
               "memory_retrieval_count": len(ctx.included), "memory_tokens": ctx.token_estimate,
               "memory_base_tokens": base.tokens if base else 0,
               "reference_tokens": sum(h["tokens"] for h in ref_hits),
               "total_context_tokens": sum(estimate_tokens(m["content"]) for m in messages)}
        from . import thinking
        think, think_reason = thinking.decide(self.config.thinking.mode, None, message,
                                              self.config.thinking.simple_max_words)
        rec["thinking"] = "default" if think is thinking.KEEP else ("on" if think else "off")
        rec["thinking_reason"] = think_reason
        try:
            resp = await self.primary.chat(messages, think=None if think is thinking.KEEP else think)
        except OllamaError as e:
            rec.update(error=str(e), total_request_time=round(time.perf_counter() - t0, 3))
            self.metrics.record_request(rec)
            self.conv_log.system_event(conversation_id, "primary_failed", project_id=self.project_id,
                                       request_id=request_id, error=str(e))
            plog.error("primary request failed", extra=rec)
            raise
        answer = resp["message"].get("content", "")
        flags: list[dict] = []
        if self.config.flags.enabled:
            answer, found = strip_flags(answer, self.config.flags.max_per_turn, self.config.flags.max_chars)
            flags = [f.to_dict() for f in found]
        rec["memory_flags"] = len(flags)
        rec.update(Metrics.from_ollama(resp))
        rec["total_request_time"] = round(time.perf_counter() - t0, 3)
        self.metrics.record_request(rec)
        plog.info("primary request", extra=rec)

        prev = list(recent)
        if len(prev) >= 2 and prev[-2]["role"] == "user" and prev[-1]["role"] == "assistant":
            self.capture_correction(conversation_id, turn_number, message, prev[-2]["content"], prev[-1]["content"])
        recent.append({"role": "user", "content": message})
        recent.append({"role": "assistant", "content": answer})
        task = self.record_turn(conversation_id=conversation_id, user_message=message,
                                assistant_response=answer, tool_events=[], source="chat",
                                log_user=False, flags=flags, turn_number=turn_number)
        queued = await self.queue_memory(task)
        return {"conversation_id": conversation_id, "response": answer, "memory_update_queued": queued,
                "request_id": request_id,
                "memory_entries_used": sorted(set(ctx.included) | set(base.fingerprints if base else ())),
                "references_used": [f"{h['library']}: {h['title']}" for h in ref_hits],
                "memory_flags": flags, "turn": turn_number,
                "thinking": rec.get("thinking"), "thinking_reason": rec.get("thinking_reason"),
                "seconds": rec.get("total_request_time")}
