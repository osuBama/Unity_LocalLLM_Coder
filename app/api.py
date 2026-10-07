"""Local HTTP API (spec §36) + the Ollama-compatible proxy for OpenClaw.

Binds to 127.0.0.1 by default. Do not expose to the LAN.
"""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from . import __version__
from .config import Config, load_config
from .logging_setup import setup_logging
from .ollama_client import OllamaError
from .orchestrator import Orchestrator
from .proxy import build_router
from .rebuild import rebuild_memory
from .schemas import Category

log = logging.getLogger("orchestrator")


class ChatRequest(BaseModel):
    conversation_id: str | None = Field(None, max_length=64)
    message: str = Field(min_length=1)


class RebuildRequest(BaseModel):
    replay: bool = False   # re-run raw history through the memory model
    reset: bool = False    # with replay: start from empty memory files (snapshot taken first)


def create_app(config: Config | None = None, *,
               primary_transport: httpx.AsyncBaseTransport | None = None,
               memory_transport: httpx.AsyncBaseTransport | None = None,
               console_logs: bool = True) -> FastAPI:
    config = config or load_config()
    config.ensure_dirs()
    setup_logging(config.logs_dir, config.application.log_level, console=console_logs)
    orch = Orchestrator(config, primary_transport=primary_transport, memory_transport=memory_transport)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if config.memory.worker_enabled:
            orch.worker.start()
        log.info("orchestrator started", extra={
            "primary": config.ollama.primary.base_url, "memory": config.ollama.memory.base_url,
            "detail": f"http://{config.application.host}:{config.application.port}"})
        yield
        await orch.aclose()

    app = FastAPI(title="Local AI Orchestrator", version=__version__, lifespan=lifespan)
    app.state.orch = orch

    from fastapi.responses import JSONResponse as _JSON
    from .ui import build_ui_router, csrf_guard_paths

    @app.middleware("http")
    async def csrf_guard(request, call_next):
        # Admin endpoints must not be triggerable by a web page open in your browser.
        if request.method in ("POST", "PUT", "PATCH", "DELETE") and csrf_guard_paths(request.url.path) \
                and not request.headers.get("x-ai-client"):
            return _JSON({"error": "missing X-AI-Client header (required for admin requests; "
                                   "e.g. curl -H 'X-AI-Client: 1')"}, status_code=403)
        return await call_next(request)

    # ------------------------------------------------------------ spec API
    @app.post("/chat")
    async def chat(req: ChatRequest):
        try:
            return await orch.chat(req.message, req.conversation_id)
        except OllamaError as e:
            raise HTTPException(502, f"primary model failed: {e}") from e

    @app.get("/health")
    async def health():
        primary, memory = await asyncio.gather(orch.primary.health(), orch.memory_client.health())
        same = primary.get("base_url") == memory.get("base_url")
        warnings = primary.get("warnings", []) + memory.get("warnings", [])
        if same:
            warnings.append("primary and memory use the same Ollama endpoint: no GPU isolation")
        files = orch.manager.validate_files()
        bad = [f for f in files if not f["ok"]]
        emb = None
        if orch.indexer is not None:
            emb = {"model": orch.embedder.model,
                   "memory_vectors": orch.indexer.index.count("memory"),
                   "history_vectors": orch.indexer.index.count("history"),
                   "query_ok": (await orch.query_vector("health check")) is not None}
            if not emb["query_ok"]:
                warnings.append(f"embedding model {orch.embedder.model} unavailable: keyword search only")
        status = "ok" if primary.get("reachable") and not bad else "degraded"
        return {"status": status, "version": __version__, "primary": primary, "memory": memory,
                "worker_running": orch.worker._task is not None and not orch.worker._task.done(),
                "memory_tasks": orch.db.task_counts(),
                "embeddings": emb,
                "references": None if orch.references is None else {
                    "libraries": len(orch.references.libraries()),
                    "mcp": f"http://{config.application.host}:{config.application.port}/mcp"
                    if config.references.mcp_enabled else None},
                "memory_files_ok": not bad, "memory_file_errors": bad, "warnings": warnings}

    @app.get("/memory/state")
    async def memory_state():
        store = orch.manager.stores[Category.state]
        return {"file": str(store.path),
                "entries": [vars(e) | {"category": e.category.value} for e in store.entries()],
                "markdown": store.path.read_text(encoding="utf-8") if store.path.exists() else ""}

    @app.get("/memory/search")
    async def memory_search(q: str = Query(..., min_length=1), limit: int = Query(10, ge=1, le=50),
                            include_inactive: bool = False):
        hits = orch.retriever.search(q, limit, include_inactive=include_inactive,
                                     query_vec=await orch.query_vector(q))
        return {"query": q, "results": [{"score": h.score, "entry_id": h.entry.entry_id,
                                         "category": h.entry.category.value, "title": h.entry.title,
                                         "content": h.entry.content, "active": h.entry.active}
                                        for h in hits]}

    @app.get("/memory/context")
    async def memory_context(q: str = Query(..., min_length=1)):
        """Debug: exactly what would be injected for this prompt."""
        base = orch.memory_base("__debug__", ("debug", q))
        budget = config.memory.max_context_tokens - (base.tokens if base else 0)
        qvec = await orch.query_vector(q)
        # Same shape as a real request: no preamble when the (cached) system prompt carries it.
        ctx = orch.context_builder.build(q, max_tokens=budget, base=base, query_vec=qvec,
                                         history=orch.recall_history(q, qvec, "__debug__", 0),
                                         preamble=not config.proxy.append_system_prompt)
        return {"budget": config.memory.max_context_tokens,
                "base": {"token_estimate": base.tokens if base else 0,
                         "included": list(base.fingerprints) if base else [],
                         "dropped": base.dropped if base else [], "text": base.text if base else ""},
                "turn": {"token_estimate": ctx.token_estimate, "included": ctx.included,
                         "dropped": ctx.dropped, "text": ctx.text}}

    @app.get("/memory/changes")
    async def memory_changes(limit: int = Query(50, ge=1, le=500)):
        return {"changes": orch.db.list_changes(limit)}

    @app.get("/memory/tasks")
    async def memory_tasks(limit: int = Query(20, ge=1, le=500), status: str | None = None):
        return {"counts": orch.db.task_counts(), "tasks": orch.db.list_tasks(limit, status)}

    @app.get("/memory/sessions")
    async def memory_sessions(limit: int = Query(20, ge=1, le=200)):
        return {"sessions": orch.db.list_summaries(limit)}

    @app.post("/memory/rebuild")
    async def memory_rebuild(req: RebuildRequest = RebuildRequest()):
        return rebuild_memory(orch, replay=req.replay, reset=req.reset)

    @app.post("/memory/backup")
    async def memory_backup():
        return {"backup": str(orch.manager.snapshot("manual"))}

    @app.post("/memory/consolidate")
    async def memory_consolidate(dry_run: bool = False):
        """Run consolidation now (waits for the memory GPU to be free)."""
        async with orch.worker.lock:
            report = await orch.consolidator.run(trigger="api", dry_run=dry_run, force=True)
        return report.to_dict()

    @app.get("/memory/consolidation")
    async def memory_consolidation_status():
        due, why = orch.consolidator.due()
        last = orch.db.kv_get("consolidation.last_report")
        return {"due": due, "reason": why, "last_report": json.loads(last) if last else None}

    @app.get("/memory/review")
    async def memory_review():
        return {"stale_after_days": config.consolidation.stale_after_days,
                "entries": orch.consolidator.review()}

    @app.get("/eval/candidates")
    async def eval_candidates(all: bool = False):
        return {"candidates": orch.db.candidates(status=None if all else "pending")}

    @app.get("/metrics")
    async def metrics():
        snap = orch.metrics.snapshot()
        snap["memory_tasks"] = orch.db.task_counts()
        snap["memory_store_tokens"] = orch.manager.memory_tokens()
        snap["tool_digests"] = orch.db.digest_stats()
        snap["token_calibration"] = orch.calibrator.snapshot()
        return snap

    app.include_router(build_ui_router(orch))
    from .mcp import build_mcp_router
    app.include_router(build_mcp_router(orch))

    # ---------------------------------------------- OpenClaw / Ollama proxy
    app.include_router(build_router(orch))
    return app
