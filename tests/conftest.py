import json
import sys
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Config  # noqa: E402


def make_config(root: Path, **memory_overrides) -> Config:
    mem = {"worker_enabled": False, "retry_base_seconds": 0, "max_attempts": 3,
           "max_context_tokens": 2500}
    mem.update(memory_overrides)
    return Config(
        paths={"root": str(root)},
        ollama={
            "primary": {"base_url": "http://primary.test", "model": "qwen3:14b", "num_ctx": 16384,
                        "timeout_seconds": 5},
            "memory": {"base_url": "http://memory.test", "model": "qwen3:8b", "num_ctx": 8192,
                       "timeout_seconds": 5, "think": False},
        },
        memory=mem,
        embeddings={"enabled": False},   # tests that need vectors turn this on explicitly
        application={"log_level": "WARNING"},
    )


@pytest.fixture
def cfg(tmp_path):
    return make_config(tmp_path)


_SYNONYMS = {"refused": "connect", "connection": "connect", "connecting": "connect", "reach": "connect",
             "unreachable": "connect", "porta": "port", "ports": "port", "gpu": "card", "graphics": "card",
             "vram": "card", "memória": "memory"}
_PREFIXES = ("search_query: ", "search_document: ")


def fake_embedding(text: str, dim: int = 128) -> list[float]:
    """Deterministic bag-of-words embedding with a tiny synonym table (paraphrase-aware enough for tests)."""
    import hashlib
    import re
    for p in _PREFIXES:
        if text.startswith(p):
            text = text[len(p):]
    v = [0.0] * dim
    for w in re.findall(r"[a-zà-ú0-9]+", text.lower()):
        if len(w) < 3:
            continue
        w = _SYNONYMS.get(w, w)
        h = int(hashlib.md5(w.encode()).hexdigest(), 16)
        v[h % dim] += 1.0
    n = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / n for x in v]


class FakeOllama:
    """A tiny stand-in for one Ollama instance, served in-process via ASGITransport."""

    def __init__(self, name: str):
        self.name = name
        self.requests: list[dict] = []
        self.reply = "fake answer"
        self.tool_calls: list | None = None
        self.memory_json: dict | str = {"changes": []}
        self.fail_status: int | None = None
        self.simulate_cache = False     # prompt_eval_count = chars after the shared prefix / 4
        self.tool_call_script: list = []  # per request: a tool_calls list, or None for a normal reply
        self._last_prompt = ""
        self.app = FastAPI()
        app = self.app

        @app.post("/api/chat")
        async def chat(request: Request):
            body = await request.json()
            self.requests.append(body)
            if self.fail_status:
                return JSONResponse({"error": "boom"}, status_code=self.fail_status)
            if self.tool_call_script and body.get("format") is None:
                self.tool_calls = self.tool_call_script.pop(0)
            if body.get("format") is not None:  # memory model call
                content = self.memory_json if isinstance(self.memory_json, str) else json.dumps(self.memory_json)
                return {"model": body["model"], "message": {"role": "assistant", "content": content},
                        "done": True, "total_duration": 1_000_000}
            evaluated = 100
            if self.simulate_cache:
                prompt = "".join(f"{m.get('role')}:{m.get('content', '')}\n" for m in body["messages"])
                lcp = 0
                for a, b in zip(prompt, self._last_prompt):
                    if a != b:
                        break
                    lcp += 1
                self._last_prompt = prompt
                evaluated = max(1, (len(prompt) - lcp) // 4)
            final = {"model": body["model"], "done": True, "done_reason": "stop",
                     "message": {"role": "assistant", "content": ""},
                     "prompt_eval_count": evaluated, "prompt_eval_duration": evaluated * 100_000,
                     "eval_count": 10, "eval_duration": 200_000_000, "total_duration": 300_000_000}
            if body.get("stream", True) is False:
                msg = {"role": "assistant", "content": self.reply}
                if self.tool_calls:
                    msg = {"role": "assistant", "content": "", "tool_calls": self.tool_calls}
                return {**final, "message": msg}

            def gen():
                if self.tool_calls:
                    yield json.dumps({"model": body["model"], "done": False, "message": {
                        "role": "assistant", "content": "", "tool_calls": self.tool_calls}}) + "\n"
                else:
                    for word in self.reply.split(" "):
                        yield json.dumps({"model": body["model"], "done": False, "message": {
                            "role": "assistant", "content": word + " "}}) + "\n"
                yield json.dumps(final) + "\n"
            return StreamingResponse(gen(), media_type="application/x-ndjson")

        self.ps_models: dict[str, dict] = {}         # name -> /api/ps entry
        self.vram_fits_at: dict[str, int] = {}       # model -> largest num_ctx that is 100% on GPU

        @app.post("/api/generate")
        async def generate(request: Request):
            body = await request.json()
            self.requests.append(body)
            name = body["model"]
            if body.get("keep_alive") == 0:
                self.ps_models.pop(name, None)
                return {"model": name, "done": True}
            ctx = (body.get("options") or {}).get("num_ctx", 2048)
            limit = self.vram_fits_at.get(name, 10**9)
            size = 1000 + ctx // 10
            vram = size if ctx <= limit else int(size * 0.8)
            self.ps_models[name] = {"name": name, "size": size, "size_vram": vram, "context_length": ctx}
            return {"model": name, "done": True, "response": ""}

        self.embed_calls = 0
        self.embed_fail = False

        @app.post("/api/embed")
        async def embed(request: Request):
            body = await request.json()
            self.embed_calls += 1
            if self.embed_fail:
                return JSONResponse({"error": "embed model not loaded"}, status_code=500)
            texts = body["input"] if isinstance(body["input"], list) else [body["input"]]
            if body.get("keep_alive") and self.ps_models is not None and body["model"] not in self.ps_models \
                    and self.vram_fits_at:
                self.ps_models[body["model"]] = {"name": body["model"], "size": 300, "size_vram": 300}
            return {"model": body["model"], "embeddings": [fake_embedding(t) for t in texts]}

        @app.get("/api/tags")
        async def tags():
            return {"models": [{"name": "qwen3:14b"}]}

        @app.get("/api/version")
        async def version():
            return {"version": "0.99.0"}

        @app.get("/api/ps")
        async def ps():
            if self.ps_models:
                return {"models": list(self.ps_models.values())}
            return {"models": [{"name": "qwen3:14b", "size": 100, "size_vram": 100, "context_length": 16384}]}

        @app.get("/")
        async def root():
            return "Ollama is running"

    @property
    def transport(self):
        return httpx.ASGITransport(app=self.app)


@pytest.fixture
def fakes():
    return FakeOllama("primary"), FakeOllama("memory")
