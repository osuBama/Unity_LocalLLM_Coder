"""Ollama-compatible front door for OpenClaw (native API).

OpenClaw's Ollama provider talks to /api/chat (streaming + tool calls) and
discovers models via /api/tags and /api/show. Point its baseUrl here instead
of at Ollama and this module:

  1. injects the <PROJECT_MEMORY> block into the latest user message
     (not the front of the conversation, so Ollama's prompt cache for the
     history prefix survives between turns);
  2. appends primary_system.txt to the client's system prompt;
  3. trims old history in steps (proxy.trim_*), replacing what was cut with
     the session summary kept by the memory model;
  4. relays the response, streaming or not, tool calls intact, with the
     primary's <memory_flag> tags stripped out;
  5. when a turn ends (final answer, no pending tool calls) logs it to raw
     JSONL and queues the summary update and memory extraction.

Every other /api/* and /v1/* request is passed straight through to the
primary instance. Only /api/chat gets memory.
"""
from __future__ import annotations

import asyncio
import copy
from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import json
import math
import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from . import compression
from .context_builder import BuiltContext, StableSnapshot, wrap_user_request
from .flags import FlagStripper
from . import thinking
from .metrics import Metrics
from .ollama_client import OllamaError
from .orchestrator import Orchestrator
from .util import estimate_tokens

log = logging.getLogger("orchestrator")
plog = logging.getLogger("primary")

_HOP_HEADERS = {"host", "content-length", "connection", "keep-alive", "transfer-encoding",
                "accept-encoding", "te", "trailer", "upgrade", "proxy-authorization"}
_RESP_DROP = {"content-length", "transfer-encoding", "connection", "content-encoding"}


def _fwd_headers(request: Request) -> dict[str, str]:
    return {k: v for k, v in request.headers.items() if k.lower() not in _HOP_HEADERS}


def _resp_headers(headers) -> dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in _RESP_DROP}


# --------------------------------------------------------------- helpers
def last_user_index(messages: list[dict]) -> int | None:
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            return i
    return None


def conversation_id_for(request: Request, body: dict) -> str:
    for h in ("x-conversation-id", "x-session-id"):
        if request.headers.get(h):
            return request.headers[h][:64]
    first_user = next((m.get("content", "") for m in body.get("messages", [])
                       if m.get("role") == "user" and isinstance(m.get("content"), str)), "")
    digest = hashlib.sha1(f"{body.get('model', '')}\n{first_user[:4000]}".encode("utf-8")).hexdigest()
    return "oc-" + digest[:14]


def stepped_drop(user_turns: int, trigger: int, keep: int) -> int:
    """How many leading user turns to drop.

    Nothing until history exceeds `trigger`; then cut back to ~`keep`. The cut
    point moves in steps of (trigger - keep), so for that many turns in a row
    the prompt prefix is identical and Ollama's cache keeps hitting.
    e.g. trigger=10 keep=4: turns 1-10 -> drop 0; 11-15 -> drop 6; 16-21 -> drop 12; 22-27 -> 18.
    """
    if trigger <= 0 or user_turns <= trigger:
        return 0
    step = trigger - keep
    return ((user_turns - keep) // step) * step


def drop_user_turns(messages: list[dict], drop: int) -> list[dict]:
    """Remove the first `drop` user turns (and everything belonging to them).

    Cutting only at user-message boundaries never splits a tool-call chain.
    Leading system messages are always kept.
    """
    if drop <= 0:
        return messages
    user_idx = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    if drop >= len(user_idx):
        return messages
    head = []
    for m in messages:
        if m.get("role") == "system":
            head.append(m)
        else:
            break
    return head + messages[user_idx[drop]:]


def tool_events_since(messages: list[dict], start: int) -> list[dict]:
    events: list[dict] = []
    call_names: list[str] = []
    for m in messages[start + 1:]:
        role = m.get("role")
        if role == "assistant":
            for tc in m.get("tool_calls") or []:
                fn = (tc or {}).get("function", {}) or {}
                name = fn.get("name", "?")
                call_names.append(name)
                events.append({"type": "tool_call", "tool": name, "arguments": fn.get("arguments")})
        elif role == "tool":
            name = m.get("tool_name") or m.get("name") or (call_names.pop(0) if call_names else "?")
            events.append({"type": "tool_result", "tool": name, "result": m.get("content", "")})
    return events


class _StreamAccumulator:
    """Parses Ollama NDJSON as it streams; optionally strips memory flags.

    Without a stripper, bytes are relayed exactly as received. With one, lines
    carrying message.content are re-serialised with the flag text removed.
    """

    def __init__(self, stripper: FlagStripper | None = None):
        self.stripper = stripper
        self.buf = b""
        self.content: list[str] = []
        self.tool_calls: list[Any] = []
        self.final: dict = {}
        self.first_token_at: float | None = None
        self.error: str | None = None

    def feed(self, chunk: bytes, now: float) -> bytes:
        self.buf += chunk
        out: list[bytes] = []
        while b"\n" in self.buf:
            line, self.buf = self.buf.split(b"\n", 1)
            out.append(self._line(line, now))
        if self.stripper is None:
            return chunk
        return b"".join(out)

    def close(self, now: float) -> bytes:
        rest, self.buf = self.buf, b""
        emitted = self._line(rest, now) if rest.strip() else b""
        return b"" if self.stripper is None else emitted

    def _line(self, line: bytes, now: float) -> bytes:
        raw = line + b"\n"
        if not line.strip():
            return raw
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return raw
        if not isinstance(obj, dict):
            return raw
        if "error" in obj:
            self.error = str(obj["error"])
        msg = obj.get("message") or {}
        content = msg.get("content") or ""
        done = bool(obj.get("done"))
        if self.stripper is not None:
            visible = self.stripper.feed(content)
            if done:
                visible += self.stripper.finish()
        else:
            visible = content
        if visible:
            self.content.append(visible)
            if self.first_token_at is None:
                self.first_token_at = now
        if msg.get("tool_calls"):
            self.tool_calls.extend(msg["tool_calls"])
            if self.first_token_at is None:
                self.first_token_at = now
        if done:
            self.final = obj
        if self.stripper is None or visible == content:
            return raw
        if not visible and not done and not msg.get("tool_calls") and not msg.get("thinking"):
            return b""  # chunk was entirely flag text (or held back)
        obj["message"] = {**msg, "content": visible}
        return (json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


@dataclass
class TurnPlan:
    """Decided once at the first request of a turn, reused for every tool-call step."""
    drop: int
    ctx: BuiltContext
    summary_turns: int = 0
    flags: list[dict] = field(default_factory=list)
    boundary: int = 0                                   # tool results of turns <= this may be digests
    frozen: dict[str, str] = field(default_factory=dict)  # result hash -> digest
    base: StableSnapshot | None = None                  # frozen memory base for the system prompt
    est_size: int | None = None                         # size mode: estimated prompt tokens
    think: object = None                                # thinking.KEEP or a bool to send
    ref_text: str = ""                                  # automatic <REFERENCE> excerpts
    ref_tokens: int = 0
    think_reason: str = ""


# ---------------------------------------------------------------- router
def build_router(orch: Orchestrator) -> APIRouter:
    router = APIRouter()
    cfg = orch.config
    plans: OrderedDict[tuple, TurnPlan] = OrderedDict()
    layouts: OrderedDict[tuple, dict[str, str]] = OrderedDict()
    size_state: OrderedDict[str, tuple[int, int]] = OrderedDict()   # conversation -> (cut, boundary)
    never_tools = set(cfg.compression.never_compress_tools)

    def new_stripper() -> FlagStripper | None:
        if not cfg.flags.enabled:
            return None
        return FlagStripper(cfg.flags.max_per_turn, cfg.flags.max_chars)

    def frozen_digests(conversation_id: str, drop: int, boundary: int,
                       kept: list[dict]) -> dict[str, str]:
        """Which results get digests is decided once per boundary position, then frozen."""
        if boundary <= 0:
            return {}
        key = (conversation_id, drop, boundary)
        if key in layouts:
            layouts.move_to_end(key)
            return layouts[key]
        hashes = [h for _, h in compression.candidates(kept, drop + 1, boundary,
                                                       cfg.compression.min_result_tokens, never_tools)]
        found = orch.db.get_digests(hashes)
        frozen = {h: d["digest"] for h, d in found.items() if d["usable"]}
        layouts[key] = frozen
        while len(layouts) > 256:
            layouts.popitem(last=False)
        return frozen

    def size_mode() -> bool:
        return cfg.proxy.trim_mode == "size" and bool(cfg.ollama.primary.num_ctx)

    def estimate_prompt(conversation_id: str, original: list[dict], tools_chars: int,
                        cut: int, boundary: int, model: str = "") -> int:
        """Rough prompt size if history were cut at `cut` and compressed up to `boundary`.

        Includes the system prompt we add, the client's tool schemas, a per-message
        template allowance, and reserves for the memory block and the reply. Characters
        are converted with the per-model calibrated ratio (proxy.calibrate_tokens) or the
        conservative default.
        """
        kept = drop_user_turns(original, cut)
        if boundary and cfg.compression.enabled:
            frozen = frozen_digests(conversation_id, cut, boundary, kept)
            kept, _, _ = compression.apply(kept, cut + 1, boundary, frozen,
                                           cfg.compression.min_result_tokens, never_tools)
        chars = sum(len(m["content"]) for m in kept if isinstance(m.get("content"), str))
        chars += tools_chars + len(orch.primary_system)
        if cfg.proxy.calibrate_tokens:
            body = orch.calibrator.tokens(model, chars)
        else:
            body = math.ceil(chars / 3.5)
        refs = cfg.references.auto_max_tokens if (orch.references is not None and orch.references.has_auto()) else 0
        return (body + 4 * len(kept) + cfg.memory.max_context_tokens + refs + cfg.proxy.reply_reserve_tokens)

    def size_layout(conversation_id: str, turn: int, original: list[dict], tools_chars: int,
                    covered: int, model: str = "") -> tuple[int, int, int]:
        """(cut, boundary, estimated tokens) for size-triggered trimming.

        Nothing changes while the prompt fits num_ctx. When it would not fit:
        compress old tool output first; if still above the target, cut the oldest
        turns until the estimate is at trim_target_ratio of the window. The layout
        only ever moves forward and then stays put until the window fills again,
        so the prompt prefix (and Ollama's cache) is stable for many turns.
        """
        limit = cfg.ollama.primary.num_ctx
        target = int(limit * cfg.proxy.trim_target_ratio)
        cut, boundary = size_state.get(conversation_id, (0, 0))
        cut = min(cut, max(0, turn - 1))
        size = estimate_prompt(conversation_id, original, tools_chars, cut, boundary, model)
        if size > limit:
            if cfg.compression.enabled:
                boundary = max(boundary, turn - cfg.compression.keep_recent_user_turns)
                size = estimate_prompt(conversation_id, original, tools_chars, cut, boundary, model)
            if size > target:
                max_cut = max(cut, turn - cfg.proxy.trim_keep_user_turns)
                if cfg.proxy.trim_requires_summary and cfg.session.summaries_enabled:
                    max_cut = max(cut, min(max_cut, covered))
                for d in range(cut + 1, max_cut + 1):
                    cut = d
                    size = estimate_prompt(conversation_id, original, tools_chars, cut, boundary, model)
                    if size <= target:
                        break
        size_state[conversation_id] = (cut, boundary)
        size_state.move_to_end(conversation_id)
        while len(size_state) > 512:
            size_state.popitem(last=False)
        return cut, boundary, size

    def plan_key(conversation_id: str, turn: int, user_text: str) -> tuple:
        return (conversation_id, turn, hash(user_text))

    def plan_for(conversation_id: str, turn: int, user_text: str, original: list[dict],
                 tools_chars: int = 0, client_think=None, model: str = "", qvec=None) -> TurnPlan:
        key = plan_key(conversation_id, turn, user_text)
        if key in plans:
            plans.move_to_end(key)
            return plans[key]
        est_size = None
        if size_mode():
            summary = orch.session_summary(conversation_id)
            covered = summary["covered_turns"] if summary else 0
            drop, boundary, est_size = size_layout(conversation_id, turn, original, tools_chars, covered, model)
        else:
            drop = stepped_drop(turn, cfg.proxy.trim_trigger_user_turns, cfg.proxy.trim_keep_user_turns)
            summary = orch.session_summary(conversation_id) if drop else None
            covered = summary["covered_turns"] if summary else 0
            if drop and cfg.proxy.trim_requires_summary and cfg.session.summaries_enabled:
                drop = min(drop, covered)  # never cut what the summary doesn't cover yet
            boundary = 0
            if cfg.compression.enabled:
                boundary = compression.compress_boundary(
                    turn, drop, trim_keep=cfg.proxy.trim_keep_user_turns,
                    keep_recent=cfg.compression.keep_recent_user_turns,
                    step=cfg.compression.step_turns, trimming=bool(cfg.proxy.trim_trigger_user_turns))

        # The memory base refreshes only when the prefix changes anyway (trim point or
        # compression boundary moved); with both off, on its own schedule.
        base = None
        if cfg.proxy.inject_memory and cfg.proxy.append_system_prompt:
            if size_mode() or cfg.proxy.trim_trigger_user_turns or cfg.compression.enabled:
                epoch = ("layout", drop, boundary)
            else:
                epoch = ("turns", turn // cfg.stable_memory.refresh_turns)
            base = orch.memory_base(conversation_id, epoch)

        if cfg.proxy.inject_memory:
            budget = cfg.memory.max_context_tokens - (base.tokens if base else 0)
            # Older exchanges, excluding turns of this conversation still verbatim in the prompt.
            history = orch.recall_history(user_text, qvec, conversation_id, drop)
            ctx = orch.context_builder.build(
                user_text, max_tokens=budget, base=base,
                session_summary=summary["summary"] if (drop and summary) else None,
                preamble=not cfg.proxy.append_system_prompt, query_vec=qvec, history=history)
        else:
            ctx = BuiltContext("", 0)
        plan = TurnPlan(drop, ctx, covered, boundary=boundary, base=base, est_size=est_size)
        if cfg.proxy.inject_memory:
            plan.ref_text, hits = orch.reference_block(user_text, qvec)
            plan.ref_tokens = sum(h["tokens"] for h in hits)
        # Decided once per turn so the model doesn't switch modes between tool-call steps.
        plan.think, plan.think_reason = thinking.decide(cfg.thinking.mode, client_think, user_text,
                                                        cfg.thinking.simple_max_words)
        orch.record_usage(list(ctx.included) + list(base.fingerprints if base else ()))  # once per turn
        if cfg.compression.enabled:
            plan.frozen = frozen_digests(conversation_id, drop, boundary, drop_user_turns(original, drop))
        plans[key] = plan
        while len(plans) > 256:
            plans.popitem(last=False)
        return plan

    async def prepare(body: dict, conversation_id: str) -> tuple[dict, dict]:
        """Return (outgoing body, info about the turn)."""
        out = copy.deepcopy(body)
        original: list[dict] = list(body.get("messages") or [])
        info: dict[str, Any] = {"memory_tokens": 0, "memory_ids": [], "user_text": None,
                                "user_index_original": None, "trimmed": 0, "turn": 0, "plan": None}
        uidx = last_user_index(original)
        user_text = original[uidx].get("content") if uidx is not None else None
        messages = original
        if isinstance(user_text, str):
            turn = sum(1 for m in original if m.get("role") == "user")
            tools_chars = len(json.dumps(body["tools"], ensure_ascii=False)) if body.get("tools") else 0
            qvec = None
            if plan_key(conversation_id, turn, user_text) not in plans and cfg.proxy.inject_memory:
                qvec = await orch.query_vector(user_text)     # once per turn; None -> keywords only
            plan = plan_for(conversation_id, turn, user_text, original, tools_chars, body.get("think"),
                            str(body.get("model") or ""), qvec)
            info.update(user_text=user_text, user_index_original=uidx, turn=turn, plan=plan)
            messages = drop_user_turns(original, plan.drop)
            info["trimmed"] = len(original) - len(messages)
            messages, n, saved = compression.apply(messages, plan.drop + 1, plan.boundary, plan.frozen,
                                                   cfg.compression.min_result_tokens, never_tools)
            info["tool_results_compressed"], info["tool_tokens_saved"] = n, saved
        else:
            messages = list(original)

        if cfg.proxy.append_system_prompt:
            system_add = orch.system_prompt(info["plan"].base if info["plan"] else None)
            if messages and messages[0].get("role") == "system" and isinstance(messages[0].get("content"), str):
                messages[0] = {**messages[0],
                               "content": messages[0]["content"].rstrip() + "\n\n" + system_add}
            else:
                messages.insert(0, {"role": "system", "content": system_add})

        plan = info["plan"]
        if plan is not None and (plan.ctx.text or plan.ref_text):
            # Attach the block to the LAST message (the user's message, or the newest tool
            # result mid tool-loop). Next request that message reappears without the block,
            # so only the block itself drops out of the cache, never the tool output after it.
            idx = len(messages) - 1
            if messages[idx].get("role") not in ("user", "tool") or not isinstance(messages[idx].get("content"), str):
                idx = last_user_index(messages)
            block = "\n\n".join(x for x in (plan.ctx.text, plan.ref_text) if x)
            messages[idx] = {**messages[idx], "content": wrap_user_request(block, messages[idx]["content"])}
            info["memory_tokens"] = plan.ctx.token_estimate
            info["memory_ids"] = plan.ctx.included
        out["messages"] = messages
        if orch.capture is not None:
            orch.capture[conversation_id] = copy.deepcopy(messages)
        if plan is not None and plan.think is not thinking.KEEP:
            out["think"] = plan.think

        if orch.config.ollama.primary.num_ctx:
            opts = dict(out.get("options") or {})
            opts.setdefault("num_ctx", orch.config.ollama.primary.num_ctx)
            out["options"] = opts
        info["total_context_tokens"] = sum(
            estimate_tokens(m.get("content", "")) for m in messages if isinstance(m.get("content"), str))
        info["prompt_chars"] = sum(len(m["content"]) for m in messages if isinstance(m.get("content"), str)) \
            + (len(json.dumps(out["tools"], ensure_ascii=False)) if out.get("tools") else 0)
        return out, info

    async def finalize(body: dict, info: dict, conversation_id: str, request_id: str,
                       content: str, tool_calls: list, final: dict, t0: float,
                       first_token_at: float | None, error: str | None,
                       stripper: FlagStripper | None) -> None:
        plan: TurnPlan | None = info.get("plan")
        if plan is not None and stripper is not None:
            plan.flags.extend(f.to_dict() for f in stripper.flags)
            plan.flags[:] = plan.flags[: cfg.flags.max_per_turn]
        rec = {"request_id": request_id, "conversation_id": conversation_id, "mode": "proxy",
               "primary_model": body.get("model"), "memory_model": cfg.ollama.memory.model,
               "turn": info.get("turn"),
               "memory_retrieval_count": len(info["memory_ids"]),
               "memory_tokens": info["memory_tokens"],
               "memory_base_tokens": plan.base.tokens if plan and plan.base else 0,
               "reference_tokens": plan.ref_tokens if plan else 0,
               "total_context_tokens": info["total_context_tokens"],
               "history_messages_trimmed": info["trimmed"],
               "user_turns_dropped": plan.drop if plan else 0,
               "est_prompt_tokens": plan.est_size if plan else None,
               "context_over_budget": bool(plan and plan.est_size and cfg.ollama.primary.num_ctx
                                           and plan.est_size > cfg.ollama.primary.num_ctx),
               "tool_results_compressed": info.get("tool_results_compressed", 0),
               "tool_tokens_saved": info.get("tool_tokens_saved", 0),
               "memory_flags": len(stripper.flags) if stripper else 0,
               "tool_calls_in_response": len(tool_calls),
               "thinking": ("client" if not plan or plan.think is thinking.KEEP else
                            ("on" if plan.think else "off")),
               "thinking_reason": plan.think_reason if plan else "",
               "total_request_time": round(time.perf_counter() - t0, 3)}
        if first_token_at is not None:
            rec["time_to_first_token"] = round(first_token_at - t0, 3)
        rec.update(Metrics.from_ollama(final))
        if error:
            rec["error"] = error
        elif final.get("prompt_eval_count"):
            orch.calibrator.observe(str(body.get("model") or ""), info.get("prompt_chars", 0),
                                    final.get("prompt_eval_count"))
        orch.metrics.record_request(rec)
        (plog.error if error else plog.info)("proxied chat", extra=rec)

        if error:
            orch.conv_log.system_event(conversation_id, "primary_failed", project_id=orch.project_id,
                                       request_id=request_id, error=error,
                                       user_message=info.get("user_text"))
            return
        if tool_calls or info.get("user_text") is None:
            return  # mid-turn step: the model asked for tools; the turn is not over yet
        original = body.get("messages") or []
        uidx = info.get("user_index_original")
        events = tool_events_since(original, uidx) if uidx is not None else []
        if uidx is not None:
            users = [i for i, m in enumerate(original[:uidx]) if m.get("role") == "user"]
            if users:
                prev_q = original[users[-1]].get("content")
                prev_a = next((m.get("content") for m in reversed(original[users[-1] + 1:uidx])
                               if m.get("role") == "assistant" and m.get("content")), None)
                if isinstance(prev_q, str) and isinstance(prev_a, str):
                    orch.capture_correction(conversation_id, info.get("turn", 0), info["user_text"], prev_q, prev_a)
        try:
            task = orch.record_turn(conversation_id=conversation_id, user_message=info["user_text"],
                                    assistant_response=content, tool_events=events, source="openclaw",
                                    flags=plan.flags if plan else [], turn_number=info.get("turn", 0))
            if cfg.proxy.queue_memory_updates:
                await orch.queue_memory(task)
        except Exception:
            log.exception("failed to record proxied turn", extra={"conversation_id": conversation_id})

    @router.post("/api/chat")
    async def api_chat(request: Request):
        raw = await request.body()
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        if not cfg.proxy.enabled or not isinstance(body, dict):
            return await passthrough(request, "api/chat", raw)

        request_id = uuid.uuid4().hex[:12]
        orch.touch()
        conversation_id = conversation_id_for(request, body)
        out, info = await prepare(body, conversation_id)
        stream = out.get("stream", True) is not False
        t0 = time.perf_counter()
        headers = {"content-type": "application/json"}
        stripper = new_stripper()

        if not stream:
            try:
                resp = await orch.primary.http.post("/api/chat", json=out, headers=headers)
            except Exception as e:
                await finalize(body, info, conversation_id, request_id, "", [], {}, t0, None, str(e), None)
                return JSONResponse({"error": f"orchestrator: primary Ollama unreachable: {e}"},
                                    status_code=502)
            data: dict = {}
            error = None
            if resp.status_code >= 400:
                error = f"HTTP {resp.status_code}: {resp.text[:300]}"
            else:
                try:
                    data = resp.json()
                except ValueError:
                    error = "malformed JSON from primary"
            msg = data.get("message") or {}
            content = msg.get("content", "") or ""
            changed = False
            if stripper is not None and content and not error:
                visible = stripper.feed(content) + stripper.finish()
                if stripper.flags:
                    visible = visible.rstrip()
                if visible != content:
                    content, changed = visible, True
                    data["message"] = {**msg, "content": content}
            await finalize(body, info, conversation_id, request_id, content,
                           msg.get("tool_calls") or [], data, t0,
                           time.perf_counter() if not error else None, error, stripper)
            if changed:
                return JSONResponse(data, status_code=resp.status_code)
            return Response(resp.content, status_code=resp.status_code,
                            headers=_resp_headers(resp.headers))

        try:
            upstream, body_iter = await orch.primary.stream_raw("POST", "/api/chat", json_body=out,
                                                                headers=headers)
        except OllamaError as e:
            await finalize(body, info, conversation_id, request_id, "", [], {}, t0, None, str(e), None)
            return JSONResponse({"error": f"orchestrator: primary Ollama unreachable: {e}"},
                                status_code=502)

        if upstream.status_code >= 400:
            content = await upstream.aread()
            await upstream.aclose()
            await finalize(body, info, conversation_id, request_id, "", [], {}, t0, None,
                           f"HTTP {upstream.status_code}: {content[:300]!r}", None)
            return Response(content, status_code=upstream.status_code,
                            headers=_resp_headers(upstream.headers))

        acc = _StreamAccumulator(stripper)

        async def relay():
            completed = False
            try:
                async for chunk in body_iter:
                    emitted = acc.feed(chunk, time.perf_counter())
                    if emitted:
                        yield emitted
                tail = acc.close(time.perf_counter())
                if tail:
                    yield tail
                completed = True
            finally:
                await upstream.aclose()
                error = acc.error or (None if completed and acc.final else "stream ended early")
                text = "".join(acc.content)
                if stripper is not None and stripper.flags:
                    text = text.rstrip()
                # Shielded so a client disconnect cannot cancel logging/queuing.
                await asyncio.shield(finalize(
                    body, info, conversation_id, request_id, text, acc.tool_calls,
                    acc.final, t0, acc.first_token_at, error, stripper))

        return StreamingResponse(relay(), status_code=upstream.status_code,
                                 media_type=upstream.headers.get("content-type", "application/x-ndjson"),
                                 headers=_resp_headers({k: v for k, v in upstream.headers.items()
                                                        if k.lower() != "content-type"}))

    async def passthrough(request: Request, path: str, raw: bytes | None = None):
        if raw is None:
            raw = await request.body()
        try:
            upstream, body_iter = await orch.primary.stream_raw(
                request.method, "/" + path, content=raw or None, headers=_fwd_headers(request),
                params=list(request.query_params.multi_items()))
        except OllamaError as e:
            return JSONResponse({"error": f"orchestrator: primary Ollama unreachable: {e}"},
                                status_code=502)

        async def relay():
            try:
                async for chunk in body_iter:
                    yield chunk
            finally:
                await upstream.aclose()

        if request.method == "HEAD":
            await upstream.aclose()
            return Response(status_code=upstream.status_code, headers=_resp_headers(upstream.headers))
        return StreamingResponse(relay(), status_code=upstream.status_code,
                                 headers=_resp_headers(upstream.headers))

    @router.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "HEAD"])
    async def api_passthrough(path: str, request: Request):
        return await passthrough(request, "api/" + path)

    @router.api_route("/v1/{path:path}", methods=["GET", "POST", "HEAD"])
    async def v1_passthrough(path: str, request: Request):
        # OpenAI-compatible surface is passed through WITHOUT memory. Use api: "ollama" in OpenClaw.
        return await passthrough(request, "v1/" + path)

    @router.api_route("/", methods=["GET", "HEAD"])
    async def root(request: Request):
        return await passthrough(request, "")

    return router
